"""IDK ("I don't know") abstention MCQ over real FinePDFs passages (no LLM).

Inspired by Michelangelo's IDK task (arXiv:2409.12640): a long context, a
multiple-choice question, and one option always being "I don't know". The model
must abstain when the answer is not present. We make it deterministic and
leakage-proof by layering a synthetic, queryable structure over real FinePDFs
text: a subset of passages get a **binding sentence** appended stating a random
code for a named item, e.g.

    Paragraph 12:
    ...real FinePDFs passage...
    The access code for the lemon is 7F3A-21.

The question asks for "the {attribute} code for the {marker}" and offers four
options (shuffled, so "I don't know" can be any letter):

  * answerable (~30%)      : the (attribute, marker) pair IS bound -> gold = its
                             code; distractors are other real codes from context.
  * unanswerable - absent  : the marker appears in NO binding -> gold = IDK.
  * unanswerable - subtle  : the marker IS bound, but under a DIFFERENT attribute
                             than the one asked -> gold = IDK. Its real code is
                             planted among the distractors as bait, so a model
                             that ignores the attribute gets caught.

Codes are random (not guessable from pretraining), so gold is deterministic and
the score is plain accuracy on the chosen letter. `size` = number of paragraphs,
binary-searched by the shared ruler++ fitter to fill the budget.

    python -m idk --input_dir finpdf_sample --output_dir idk_out
    python -m idk --max_seq_length 131072 --num_samples 200 --format sft
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re

from mrcr.mrcr import build_pool, load_markers
from ruler_pp.base import Sample
from ruler_pp.cli import shape_row
from ruler_pp.lengths import fit_size, get_tokenizer

IDK = "I don't know"
ATTRIBUTES = ["access", "security", "license", "registration", "vault"]
BINDING_TMPL = "The {attr} code for the {marker} is {code}."
BINDING_RE = re.compile(r"The (\w+) code for the (\w+) is ([0-9A-Z][0-9A-Z-]*)\.")
QUESTION_RE = re.compile(r"What is the (\w+) code for the (\w+)\?")
OPTION_RE = re.compile(r"^\(([A-Z])\)\s+(.*)$", re.MULTILINE)

INSTRUCTION = (
    "You are given a long list of numbered paragraphs. Some paragraphs state a "
    "code (access / security / license / registration / vault) for a named item. "
    "Answer the multiple-choice question using only the information in the "
    "paragraphs. If the paragraphs do not contain the answer, choose "
    f'"{IDK}". Respond with the single letter of the correct option and nothing else.'
)


def make_code(rng, seen):
    """A random, distinct, non-guessable code like '7F3A-21'."""
    while True:
        code = f"{rng.randint(0, 0xFFFF):04X}-{rng.randint(0, 99):02d}"
        if code not in seen:
            seen.add(code)
            return code


class IDKTask:
    """Builds one abstention MCQ Sample. `size` = number of paragraphs."""

    name = "idk"

    def __init__(self, pool, markers, *, attributes=ATTRIBUTES, n_choices=4,
                 binding_frac=0.3, unanswerable_frac=0.7, subtle_frac=0.15,
                 min_size=8, reserve_tokens=16):
        if not 0.0 <= unanswerable_frac <= 1.0:
            raise ValueError("unanswerable_frac must be in [0, 1]")
        if subtle_frac > unanswerable_frac:
            raise ValueError("subtle_frac cannot exceed unanswerable_frac")
        self.pool = pool
        self.markers = markers
        self.attributes = attributes
        self.n_choices = n_choices
        self.binding_frac = binding_frac
        self.unanswerable_frac = unanswerable_frac
        self.subtle_frac = subtle_frac
        self.min_size = min_size
        self.reserve_tokens = reserve_tokens

    def _n_bindings(self, n):
        lower = max(self.n_choices, 4)
        upper = min(n, len(self.markers) - 1)
        return min(max(round(n * self.binding_frac), lower), upper)

    def build(self, size, rng):
        n = max(self.min_size, size)
        if n <= len(self.pool):
            passages = rng.sample(self.pool, n)
        else:
            passages = [rng.choice(self.pool) for _ in range(n)]

        # assign distinct (marker, attribute, code) bindings to random passages
        n_bind = self._n_bindings(n)
        bind_idx = rng.sample(range(n), n_bind)
        bind_markers = rng.sample(self.markers, n_bind)
        seen_codes: set[str] = set()
        bindings = []  # {idx, marker, attr, code}
        for idx, marker in zip(bind_idx, bind_markers):
            bindings.append({
                "idx": idx,
                "marker": marker,
                "attr": rng.choice(self.attributes),
                "code": make_code(rng, seen_codes),
            })
        by_idx = {b["idx"]: b for b in bindings}
        all_codes = [b["code"] for b in bindings]

        # choose the question kind
        answerable_frac = 1.0 - self.unanswerable_frac
        roll = rng.random()
        can_subtle = len(self.attributes) >= 2
        if roll < answerable_frac:
            kind = "answerable"
        elif roll < answerable_frac + self.subtle_frac and can_subtle:
            kind = "subtle"
        else:
            kind = "absent"

        nd = self.n_choices - 1  # number of non-IDK options
        if kind == "answerable":
            b = rng.choice(bindings)
            q_attr, q_marker = b["attr"], b["marker"]
            correct = b["code"]
            others = [c for c in all_codes if c != correct]
            distract = rng.sample(others, nd - 1)
            opts, correct_opt = [correct] + distract, correct
        elif kind == "subtle":
            b = rng.choice(bindings)
            q_marker = b["marker"]
            q_attr = rng.choice([a for a in self.attributes if a != b["attr"]])
            others = [c for c in all_codes if c != b["code"]]
            distract = [b["code"]] + rng.sample(others, nd - 1)  # plant the bait
            opts, correct_opt = distract, IDK
        else:  # absent
            bound = {b["marker"] for b in bindings}
            unused = [m for m in self.markers if m not in bound]
            q_marker = rng.choice(unused)
            q_attr = rng.choice(self.attributes)
            opts, correct_opt = rng.sample(all_codes, nd), IDK

        options = opts + [IDK]
        rng.shuffle(options)
        gold_i = options.index(correct_opt)
        gold_letter = chr(ord("A") + gold_i)

        # render the context (numbered passages; bound ones get a binding line)
        blocks = []
        for i in range(n):
            block = f"Paragraph {i + 1}:\n{passages[i]}"
            if i in by_idx:
                b = by_idx[i]
                block += "\n" + BINDING_TMPL.format(attr=b["attr"], marker=b["marker"], code=b["code"])
            blocks.append(block)
        context = "\n\n".join(blocks)

        options_block = "\n".join(f"({chr(ord('A') + j)}) {opt}" for j, opt in enumerate(options))
        question = f"Question: What is the {q_attr} code for the {q_marker}?\n\n{options_block}"

        return Sample(
            instruction=INSTRUCTION,
            context=context,
            question=question,
            answer_prefix="",
            answers=[gold_letter],
            gold=gold_letter,
            answer_type="string",
            meta={
                "kind": kind,
                "answerable": kind == "answerable",
                "asked_marker": q_marker,
                "asked_attribute": q_attr,
                "correct_code": correct_opt if kind == "answerable" else None,
                "gold_letter": gold_letter,
                "options": {chr(ord("A") + j): opt for j, opt in enumerate(options)},
                "n_passages": n,
                "n_bindings": n_bind,
            },
        )


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser("idk", description="IDK abstention MCQ over FinePDFs")
    p.add_argument("--input_dir", default="finpdf_sample")
    p.add_argument("--output_dir", default="idk_out")
    p.add_argument("--max_seq_length", type=int, default=8192, help="token budget (prompt + answer)")
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--format", choices=["pt", "sft"], default="sft",
                   help="pt: flat 'text' field; sft: OpenAI 'messages' array")
    p.add_argument("--tokenizer", default="whitespace", help="hf:<name> | tiktoken[:enc] | whitespace")
    # task shape
    p.add_argument("--n_choices", type=int, default=4, help="options incl. 'I don't know'")
    p.add_argument("--binding_frac", type=float, default=0.3,
                   help="fraction of paragraphs that carry a code binding")
    p.add_argument("--unanswerable_frac", type=float, default=0.7,
                   help="fraction of questions whose answer is absent (gold = I don't know)")
    p.add_argument("--subtle_frac", type=float, default=0.15,
                   help="fraction of ALL questions that are subtle unanswerables (marker "
                        "present but the asked attribute is unbound); must be <= unanswerable_frac")
    p.add_argument("--min_size", type=int, default=8, help="minimum paragraphs per sample")
    # passage pool
    p.add_argument("--pool_size", type=int, default=20000, help="passages to load into the pool")
    p.add_argument("--target_passage_tokens", type=int, default=256)
    p.add_argument("--min_passage_tokens", type=int, default=64)
    p.add_argument("--max_passage_tokens", type=int, default=512)
    p.add_argument("--min_doc_tokens", type=int, default=400, help="skip docs shorter than this")
    p.add_argument("--min_edu", type=float, default=0.75)
    p.add_argument("--keep_dups", action="store_true")
    p.add_argument("--max_docs", type=int, default=None, help="cap docs scanned while filling the pool")
    args = p.parse_args(argv)

    tokenizer = get_tokenizer(args.tokenizer)
    count = tokenizer.count

    print(f"loading passage pool from {args.input_dir} (target {args.pool_size}) ...")
    pool = build_pool(
        args.input_dir, count, pool_size=args.pool_size,
        target_tokens=args.target_passage_tokens, min_tokens=args.min_passage_tokens,
        max_tokens=args.max_passage_tokens, min_doc_tokens=args.min_doc_tokens,
        min_edu=args.min_edu, drop_dups=not args.keep_dups, max_docs=args.max_docs)
    if len(pool) < args.min_size:
        raise SystemExit(f"pool too small ({len(pool)} passages); lower --min_doc_tokens "
                         "or point --input_dir at more shards")
    print(f"pool: {len(pool)} passages")

    markers = load_markers()
    if len(markers) < args.n_choices + 4:
        raise SystemExit(f"need more markers in res/markers.txt (have {len(markers)})")

    task = IDKTask(pool, markers, n_choices=args.n_choices, binding_frac=args.binding_frac,
                   unanswerable_frac=args.unanswerable_frac, subtle_frac=args.subtle_frac,
                   min_size=args.min_size)
    budget = args.max_seq_length - task.reserve_tokens

    def measure(sz):
        return count(task.build(sz, random.Random(args.seed)).input)

    size = fit_size(measure, budget, task.min_size)

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "idk.jsonl")
    rows, by_kind = [], {}
    for i in range(args.num_samples):
        rng = random.Random(args.seed * 1_000_003 + i)
        cur = size
        while True:
            s = task.build(cur, rng)
            length = count(s.input)
            if length <= budget or cur <= task.min_size:
                break
            cur = max(task.min_size, int(cur * 0.9))
        prompt = s.input
        by_kind[s.meta["kind"]] = by_kind.get(s.meta["kind"], 0) + 1
        row = {
            "id": f"idk-{i:06d}",
            "task": "idk",
            "format": args.format,
            "input": prompt,
            "instruction": s.instruction,
            "context": s.context,
            "question": s.question,
            "answer_prefix": s.answer_prefix,
            "answers": s.answers,
            "gold": s.gold,
            "answer_type": s.answer_type,
            "length": length,
            "max_seq_length": args.max_seq_length,
            "seed": args.seed,
            "meta": s.meta,
        }
        shape_row(row, args.format, prompt, s.answer_prefix, s.answer_text())
        rows.append(row)

    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    avg = sum(r["length"] for r in rows) / len(rows)
    summary = {
        "output_dir": args.output_dir,
        "file": out,
        "samples": len(rows),
        "by_kind": by_kind,
        "fitted_size": size,
        "avg_len": round(avg),
        "max_seq_length": args.max_seq_length,
        "pool_passages": len(pool),
        "tokenizer": args.tokenizer,
        "format": args.format,
        "seed": args.seed,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
