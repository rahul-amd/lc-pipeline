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
import os
import random
from datetime import datetime

from .lengths import fit_size, get_tokenizer
from .registry import TASKS, get_task


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


def generate_task(task, tokenizer, n, max_seq_length, seed, num_fewshot, fmt):
    fewshot_block = task.fewshot(random.Random(seed * 7 + 1), num_fewshot)
    budget = max_seq_length - task.reserve_tokens

    def measure(size: int) -> int:
        s = task.build(size, random.Random(seed))
        return tokenizer.count(_prompt(fewshot_block, s))

    size = fit_size(measure, budget, task.min_size)

    rows = []
    for i in range(n):
        rng = random.Random(seed * 1_000_003 + i)
        cur = size
        while True:
            s = task.build(cur, rng)
            length = tokenizer.count(_prompt(fewshot_block, s))
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
        rows.append(shape_row(row, fmt, prompt, s.answer_prefix, s.answer_text()))
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
        )
        out = os.path.join(output_dir, f"{name}.jsonl")
        write_jsonl(out, rows)
        avg = sum(r["length"] for r in rows) / len(rows)
        print(f"{name:22s} size={size:<7d} avg_len={avg:7.0f}/{args.max_seq_length}  -> {out}")


if __name__ == "__main__":
    main()
