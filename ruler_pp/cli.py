"""Command-line driver: fit each task to a token budget and write jsonl.

Examples
--------
    python -m ruler_pp --task all --max_seq_length 4096 --num_samples 20
    python -m ruler_pp --task count_predicate --max_seq_length 16384 \
        --tokenizer hf:Qwen/Qwen2.5-0.5B --out_dir data
    python -m ruler_pp --list
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
from datetime import datetime

from .lengths import fit_size, get_tokenizer
from .registry import TASKS, get_task
from .output import finalize_row, training_length, write_jsonl


def resolve_workers(n):
    """0 / negative -> all cores; else the requested count (capped at cpu_count)."""
    cpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    cpu = min(cpu, int(os.environ.get("SLURM_CPUS_PER_TASK", cpu)))
    return cpu if n is None or n <= 0 else min(n, cpu)


def _prompt(fewshot_block: str, sample) -> str:
    """What the model is fed at inference (input + the answer cue)."""
    return fewshot_block + sample.input + "\n" + sample.answer_prefix


def shape_row(row, fmt, prompt, answer_prefix, answer_text):
    """Add the trainable representation for the requested output format.

    pt  -> a single flat `text` field (prompt + cue + answer), continuous text.
    sft -> an OpenAI `messages` array (user = task, assistant = full answer).

    `answer_prefix` may be empty (e.g. corpus tasks that have no cue); the join
    is stripped so no stray leading space leaks in.
    """
    answer = f"{answer_prefix} {answer_text}".strip()
    if fmt == "pt":
        row["text"] = f"{prompt}\n{answer}"
    elif fmt == "sft":
        row["messages"] = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
    else:
        raise ValueError(f"unknown format {fmt!r}")
    return row


def build_task_row(i, *, task, count, fewshot_block, budget, size, seed,
                   max_seq_length, num_fewshot, fmt):
    """Build one row for `task`. Deterministic in `i` alone (seed-per-sample),
    so it is safe to run in any order / across processes."""
    rng = random.Random(seed * 1_000_003 + i)
    cur = size
    while True:
        s = task.build(cur, rng)
        length = count(_prompt(fewshot_block, s))
        candidate = shape_row({"format": fmt}, fmt, fewshot_block + s.input, s.answer_prefix, s.answer_text())
        if training_length(candidate, count)[0] <= max_seq_length:
            break
        if cur <= task.min_size:
            raise ValueError(f"{task.name}: minimum sample does not fit {max_seq_length} tokens")
        cur = max(task.min_size, int(cur * 0.9))
    prompt = fewshot_block + s.input
    row = {
        "id": f"{task.name}-{i:06d}",
        "task": task.name,
        "format": fmt,
        "input": prompt,
        "instruction": s.instruction,
        "context": s.context,
        "question": s.question,
        "answer_prefix": s.answer_prefix,
        "answers": s.answers,
        "gold": s.gold,
        "answer_type": s.answer_type,
        "length": length,
        "max_seq_length": max_seq_length,
        "num_fewshot": num_fewshot,
        "seed": seed,
        "meta": s.meta,
    }
    shape_row(row, fmt, prompt, s.answer_prefix, s.answer_text())
    return finalize_row(row, count, max_seq_length)


# Per-worker state. The HF tokenizer isn't picklable and tasks are cheap to
# rebuild, so each worker reconstructs both from the spec / task name.
_W: dict = {}


def _worker_init(tokenizer_spec, task_name, fewshot_block, budget, size, seed,
                 max_seq_length, num_fewshot, fmt):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    tok = get_tokenizer(tokenizer_spec)
    _W.update(count=tok.count, task=get_task(task_name), fewshot_block=fewshot_block,
              budget=budget, size=size, seed=seed, max_seq_length=max_seq_length,
              num_fewshot=num_fewshot, fmt=fmt)


def _worker_build(i):
    return build_task_row(
        i, task=_W["task"], count=_W["count"], fewshot_block=_W["fewshot_block"],
        budget=_W["budget"], size=_W["size"], seed=_W["seed"],
        max_seq_length=_W["max_seq_length"], num_fewshot=_W["num_fewshot"], fmt=_W["fmt"])


def generate_task(task, tokenizer, n, max_seq_length, seed, num_fewshot, fmt,
                  *, tokenizer_spec=None, workers=1, stream=False):
    fewshot_block = task.fewshot(random.Random(seed * 7 + 1), num_fewshot)
    if n < 1 or max_seq_length < 1 or num_fewshot < 0:
        raise ValueError("num_samples and max_seq_length must be positive; num_fewshot nonnegative")
    budget = max_seq_length

    def measure(size: int) -> int:
        s = task.build(size, random.Random(seed))
        row = shape_row({"format": fmt}, fmt, fewshot_block + s.input, s.answer_prefix, s.answer_text())
        return training_length(row, tokenizer.count)[0]

    size = fit_size(measure, budget, task.min_size)

    workers = resolve_workers(workers)
    if workers <= 1 or n <= 1 or not tokenizer_spec:
        rows = (build_task_row(i, task=task, count=tokenizer.count,
                               fewshot_block=fewshot_block, budget=budget, size=size,
                               seed=seed, max_seq_length=max_seq_length,
                               num_fewshot=num_fewshot, fmt=fmt) for i in range(n))
    else:
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context("spawn")
        chunk = max(1, n // (workers * 8))
        initargs = (tokenizer_spec, task.name, fewshot_block, budget, size, seed,
                    max_seq_length, num_fewshot, fmt)
        def parallel_rows():
            with ctx.Pool(workers, initializer=_worker_init, initargs=initargs) as ppool:
                yield from ppool.imap(_worker_build, range(n), chunksize=min(chunk, 8))
        rows = parallel_rows()
    return (rows if stream else list(rows)), size


def main(argv=None):
    p = argparse.ArgumentParser("ruler_pp", description="ruler++ synthetic long-context datagen")
    p.add_argument("--task", default="all", help="task name or 'all'")
    p.add_argument("--max_seq_length", type=int, default=4096, help="total token budget (prompt + answer)")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_fewshot", type=int, default=0)
    p.add_argument("--format", choices=["pt", "sft"], default="pt",
                   help="pt: flat 'text' field; sft: OpenAI 'messages' array")
    p.add_argument("--tokenizer", default="hf:Qwen/Qwen2.5-0.5B",
                   help="hf:<name> | tiktoken[:enc] | whitespace")
    p.add_argument("--workers", type=int, default=0,
                   help="parallel sample-builder processes per task (0 = all CPU cores). "
                        "Output is identical regardless of worker count.")
    p.add_argument("--output_dir", "--out_dir", dest="output_dir", default=None,
                   help="dir for the per-task jsonl files; defaults to a timestamp like 2026-06-22_14-30-05")
    p.add_argument("--list", action="store_true", help="list tasks and exit")
    args = p.parse_args(argv)

    if args.list:
        print("\n".join(TASKS))
        return

    names = list(TASKS) if args.task == "all" else [args.task]
    tokenizer = get_tokenizer(args.tokenizer)
    output_dir = args.output_dir or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"writing to {os.path.abspath(output_dir)}/ (one file per task)")

    for name in names:
        task = get_task(name)
        rows, size = generate_task(
            task, tokenizer, args.num_samples, args.max_seq_length,
            args.seed, args.num_fewshot, args.format,
            tokenizer_spec=args.tokenizer, workers=args.workers, stream=True,
        )
        out = os.path.join(output_dir, f"{name}.jsonl")
        totals = [0, 0]
        def measured_rows():
            for row in rows:
                totals[0] += row["length"]
                totals[1] += 1
                yield row
        write_jsonl(out, measured_rows())
        avg = totals[0] / totals[1]
        print(f"{name:22s} size={size:<7d} avg_len={avg:7.0f}/{args.max_seq_length}  -> {out}")


if __name__ == "__main__":
    main()
