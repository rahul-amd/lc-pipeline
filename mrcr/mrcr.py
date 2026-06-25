"""MRCR-style marker retrieval over real FinePDFs passages (no LLM, deterministic).

Inspired by Michelangelo's MRCR (arXiv:2409.12640): a single long prompt holds
many candidate passages; the model must locate the ONE that matches an
adversarial query and reproduce it verbatim. We replace the paper's
LLM-generated writing samples with real FinePDFs passages, so the gold is the
exact source text (fully deterministic and regeneratable).

The chat is collapsed to a single turn. The context is a numbered list of
paragraphs; each paragraph's text sits between ``<<<`` / ``>>>`` fences and is
followed by one or more plain-English sentences saying what it is *marked as*
(and, sometimes, what it is *not*):

    Paragraph 7:
    <<<
    ...real FinePDFs passage text...
    >>>
    This paragraph is a lemon. It is also a car.

Each paragraph carries a set of **markers** (concrete nouns from
``res/markers.txt``). Exactly one paragraph — the target — is positively marked
with *both* query markers; confounders share one of them (some with a negation
near-miss, e.g. "a car, but not a lemon") and the rest are noise. The query asks
for "the paragraph that is both a {a} and a {b}", and the answer reproduces that
passage verbatim, prefixed with a per-sample random string (Michelangelo's
guard against degenerate output). Grading = ``SequenceMatcher`` ratio in [0,1].

`size` is the number of paragraphs; the shared ruler++ budget fitter searches it
to fill ``--max_seq_length`` (so the same task scales across 32k/128k/1M).

    python -m mrcr --input_dir finpdf_sample --output_dir mrcr_out
    python -m mrcr --max_seq_length 131072 --num_samples 200 --format sft
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re

from corpus import loader
from ruler_pp.base import Sample
from ruler_pp.cli import shape_row
from ruler_pp.lengths import fit_size, get_tokenizer

PARA_SPLIT = re.compile(r"\n\s*\n+")
OPEN_FENCE, CLOSE_FENCE = "<<<", ">>>"

# --------------------------------------------------------------------------- #
# markers + sentence templates
# --------------------------------------------------------------------------- #
_MARKERS: list[str] | None = None


def _markers_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "res", "markers.txt")


def load_markers(path: str | None = None) -> list[str]:
    """Load and cache the concrete-noun marker vocabulary."""
    global _MARKERS
    if _MARKERS is None:
        out, seen = [], set()
        with open(path or _markers_path(), encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if not w or w.startswith("#"):
                    continue
                if w.isascii() and w.isalpha() and w.lower() not in seen:
                    seen.add(w.lower())
                    out.append(w.lower())
        _MARKERS = out
    return _MARKERS


def article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _a(word: str) -> str:
    return f"{article(word)} {word}"


# Plain-English templates. The first positive marker opens; extra positives and
# negatives are appended as separate sentences (keeps the verifier's parse
# unambiguous). "{x}" is filled with "a lemon" / "an apple".
POS_OPEN = [
    "This paragraph is {x}.",
    "Mark this paragraph as {x}.",
    "Note that this passage is {x}.",
    "Remember, this one is {x}.",
    "This paragraph has been tagged {x}.",
]
POS_ADD = [
    "It is also {x}.",
    "It additionally counts as {x}.",
    "This one is also marked as {x}.",
]
NEG = [
    "But remember that it is not {x}.",
    "Note that it is not {x}.",
    "It is not {x}, however.",
]


def marker_sentences(positives, negatives, rng) -> str:
    """Render the marker description for one paragraph as natural sentences."""
    sents = [rng.choice(POS_OPEN).format(x=_a(positives[0]))]
    for m in positives[1:]:
        sents.append(rng.choice(POS_ADD).format(x=_a(m)))
    for m in negatives:
        sents.append(rng.choice(NEG).format(x=_a(m)))
    return " ".join(sents)


# --------------------------------------------------------------------------- #
# parsing back out (used by the independent verifier)
# --------------------------------------------------------------------------- #
_NOT_RE = re.compile(r"\bnot\s+an?\s+([a-z]+)")
_POS_RE = re.compile(r"\ban?\s+([a-z]+)")


def parse_markers(desc: str, vocab: set[str]):
    """(positives, negatives) marker sets parsed from a marker-sentence string."""
    negs = {m for m in _NOT_RE.findall(desc) if m in vocab}
    pos = {m for m in _POS_RE.findall(desc) if m in vocab} - negs
    return pos, negs


# --------------------------------------------------------------------------- #
# passage pool from the corpus
# --------------------------------------------------------------------------- #
def iter_passages(input_dir, count, *, target_tokens, min_tokens, max_tokens,
                  min_doc_tokens, min_edu, drop_dups, max_docs):
    """Yield readable passages (`target_tokens`-ish, <= `max_tokens`) from docs.

    Splits each quality-filtered doc on blank lines, greedily packs paragraphs up
    to `target_tokens`, and hard-splits any oversized paragraph into word windows.
    Passages containing the fence/header markup are skipped so the rendered prompt
    stays unambiguous to parse.
    """
    def ok(text):
        return (OPEN_FENCE not in text and CLOSE_FENCE not in text
                and "Paragraph " not in text)

    for _did, _src, txt, _ntok in loader.iter_docs(
            input_dir, min_tokens=min_doc_tokens, min_edu=min_edu,
            drop_dups=drop_dups, max_docs=max_docs):
        paras = [p.strip() for p in PARA_SPLIT.split(txt) if p.strip()]
        buf, buftok = [], 0
        for p in paras:
            pt = count(p)
            if pt > max_tokens:
                if buf:
                    cand = "\n\n".join(buf)
                    if buftok >= min_tokens and ok(cand):
                        yield cand
                    buf, buftok = [], 0
                words = p.split()
                win = max(1, round(target_tokens * 0.9))
                for i in range(0, len(words), win):
                    chunk = " ".join(words[i:i + win])
                    if count(chunk) >= min_tokens and ok(chunk):
                        yield chunk
                continue
            if buf and buftok + pt > target_tokens:
                cand = "\n\n".join(buf)
                if buftok >= min_tokens and ok(cand):
                    yield cand
                buf, buftok = [], 0
            buf.append(p)
            buftok += pt
        if buf and buftok >= min_tokens:
            cand = "\n\n".join(buf)
            if ok(cand):
                yield cand


def build_pool(input_dir, count, *, pool_size, **kw):
    pool = []
    for passage in iter_passages(input_dir, count, **kw):
        pool.append(passage)
        if len(pool) >= pool_size:
            break
    return pool


# --------------------------------------------------------------------------- #
# the task
# --------------------------------------------------------------------------- #
INSTRUCTION = (
    "You are given a long list of numbered paragraphs. Each paragraph's text is "
    "shown between <<< and >>> fences and is followed by one or more sentences "
    "stating what that paragraph is marked as (and, sometimes, what it is not "
    "marked as)."
)


class MRCRTask:
    """Builds one marker-retrieval Sample. `size` = number of paragraphs."""

    name = "mrcr"

    def __init__(self, pool, markers, *, k=2, confounder_frac=0.5,
                 neg_frac=0.35, min_size=6, reserve_tokens=560):
        self.pool = pool
        self.markers = markers
        self.k = k
        self.confounder_frac = confounder_frac
        self.neg_frac = neg_frac
        self.min_size = min_size
        self.reserve_tokens = reserve_tokens

    # -- marker assignment ------------------------------------------------- #
    def _roles(self, n, a, b, rng):
        """Marker (positives, negatives) for each of the n paragraphs.

        Index 0 is always the target (both query markers, no negation). The rest
        share exactly one query marker — some as negation near-misses ("a car,
        but not a lemon") — or are pure noise. Only the target carries both query
        markers positively, which guarantees a unique gold.
        """
        others = n - 1
        n_ref = round(others * self.confounder_frac)
        n_neg = round(n_ref * self.neg_frac)
        n_pos = n_ref - n_neg
        noise_vocab = [m for m in self.markers if m not in (a, b)]

        def extra(exclude):
            """Maybe tack on a non-query marker so confounders vary in length."""
            if rng.random() < 0.5:
                pool = [m for m in noise_vocab if m not in exclude]
                if pool:
                    return [rng.choice(pool)]
            return []

        roles = [([a, b], [])]  # target
        # negation near-misses: present one query word, explicitly deny the other
        for i in range(n_neg):
            if i % 2 == 0:
                roles.append(([a] + extra({a, b}), [b]))
            else:
                roles.append(([b] + extra({a, b}), [a]))
        # plain single-marker confounders
        for i in range(n_pos):
            q = a if i % 2 == 0 else b
            roles.append(([q] + extra({a, b}), []))
        # noise: neither query marker (may carry an unrelated negation)
        for _ in range(others - n_ref):
            pos = rng.sample(noise_vocab, min(len(noise_vocab), rng.randint(1, 2)))
            negs = []
            if rng.random() < 0.3:
                cand = [m for m in noise_vocab if m not in pos]
                if cand:
                    negs = [rng.choice(cand)]
            roles.append((pos, negs))
        return roles

    # -- build ------------------------------------------------------------- #
    def build(self, size, rng):
        n = max(self.min_size, size)
        a, b = rng.sample(self.markers, self.k)

        # distinct passages where possible; target drawn first so it is unique.
        if n <= len(self.pool):
            passages = rng.sample(self.pool, n)
        else:
            passages = [rng.choice(self.pool) for _ in range(n)]
        target_passage = passages[0]

        roles = self._roles(n, a, b, rng)
        blocks = list(zip(passages, roles))  # [(passage, (pos, neg))], block 0 = target
        order = list(range(n))
        rng.shuffle(order)

        rendered, target_pos = [], None
        for slot, idx in enumerate(order):
            passage, (pos, negs) = blocks[idx]
            pos = pos[:]            # copy before shuffling display order
            rng.shuffle(pos)
            desc = marker_sentences(pos, negs, rng)
            rendered.append(
                f"Paragraph {slot + 1}:\n{OPEN_FENCE}\n{passage}\n{CLOSE_FENCE}\n{desc}"
            )
            if idx == 0:
                target_pos = slot + 1

        context = "\n\n".join(rendered)
        prefix = f"{rng.getrandbits(32):08x}"
        question = (
            f"Find the single paragraph that is marked as both {_a(a)} and {_a(b)}. "
            f"Begin your answer with the exact string {prefix} on its own line, then "
            "reproduce that paragraph's text (everything between its <<< and >>> "
            "fences) verbatim. Output nothing else."
        )
        answer = f"{prefix}\n{target_passage}"

        return Sample(
            instruction=INSTRUCTION,
            context=context,
            question=question,
            answer_prefix="",
            answers=[answer],
            gold=target_passage,
            answer_type="string",
            meta={
                "query_markers": [a, b],
                "k": self.k,
                "random_prefix": prefix,
                "n_paragraphs": n,
                "target_paragraph": target_pos,
                "n_confounders": round((n - 1) * self.confounder_frac),
            },
        )


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser("mrcr", description="MRCR-style marker retrieval over FinePDFs")
    p.add_argument("--input_dir", default="finpdf_sample")
    p.add_argument("--output_dir", default="mrcr_out")
    p.add_argument("--max_seq_length", type=int, default=8192, help="token budget (prompt + answer)")
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--format", choices=["pt", "sft"], default="sft",
                   help="pt: flat 'text' field; sft: OpenAI 'messages' array")
    p.add_argument("--tokenizer", default="whitespace", help="hf:<name> | tiktoken[:enc] | whitespace")
    # task shape
    p.add_argument("--k", type=int, default=2, help="markers conjoined by the query")
    p.add_argument("--confounder_frac", type=float, default=0.5,
                   help="fraction of non-target paragraphs that share a query marker")
    p.add_argument("--neg_frac", type=float, default=0.35,
                   help="fraction of confounders that are negation near-misses")
    p.add_argument("--min_size", type=int, default=6, help="minimum paragraphs per sample")
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
    if len(markers) < args.k + 4:
        raise SystemExit(f"need more markers in res/markers.txt (have {len(markers)})")
    reserve = args.max_passage_tokens + 48
    task = MRCRTask(pool, markers, k=args.k, confounder_frac=args.confounder_frac,
                    neg_frac=args.neg_frac, min_size=args.min_size, reserve_tokens=reserve)

    budget = args.max_seq_length - reserve

    def measure(sz):
        s = task.build(sz, random.Random(args.seed))
        return count(s.input)

    size = fit_size(measure, budget, task.min_size)

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "mrcr.jsonl")
    rows = []
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
        row = {
            "id": f"mrcr-{i:06d}",
            "task": "mrcr",
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
        "fitted_size": size,
        "avg_len": round(avg),
        "max_seq_length": args.max_seq_length,
        "pool_passages": len(pool),
        "k": args.k,
        "tokenizer": args.tokenizer,
        "format": args.format,
        "seed": args.seed,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
