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


def resolve_workers(n):
    """0 / negative -> all cores; else the requested count (capped at cpu_count)."""
    cpu = os.cpu_count() or 1
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
        if length <= budget or cur <= task.min_size:
            break
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
    return shape_row(row, fmt, prompt, s.answer_prefix, s.answer_text())


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
                  *, tokenizer_spec=None, workers=1):
    fewshot_block = task.fewshot(random.Random(seed * 7 + 1), num_fewshot)
    budget = max_seq_length - task.reserve_tokens

    def measure(size: int) -> int:
        s = task.build(size, random.Random(seed))
        return tokenizer.count(_prompt(fewshot_block, s))

    size = fit_size(measure, budget, task.min_size)

    workers = resolve_workers(workers)
    if workers <= 1 or n <= 1 or not tokenizer_spec:
        rows = [build_task_row(i, task=task, count=tokenizer.count,
                               fewshot_block=fewshot_block, budget=budget, size=size,
                               seed=seed, max_seq_length=max_seq_length,
                               num_fewshot=num_fewshot, fmt=fmt) for i in range(n)]
    else:
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context("spawn")
        chunk = max(1, n // (workers * 8))
        initargs = (tokenizer_spec, task.name, fewshot_block, budget, size, seed,
                    max_seq_length, num_fewshot, fmt)
        with ctx.Pool(workers, initializer=_worker_init, initargs=initargs) as ppool:
            rows = ppool.map(_worker_build, range(n), chunksize=chunk)
    return rows, size


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


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
            tokenizer_spec=args.tokenizer, workers=args.workers,
        )
        out = os.path.join(output_dir, f"{name}.jsonl")
        write_jsonl(out, rows)
        avg = sum(r["length"] for r in rows) / len(rows)
        print(f"{name:22s} size={size:<7d} avg_len={avg:7.0f}/{args.max_seq_length}  -> {out}")


if __name__ == "__main__":
    main()
