"""Untie the Knots (UtK) — paragraph-unshuffle long-context SFT data (Phase 2b).

Adapted from "Untie the Knots" (arXiv:2409.04774). The paper chunks a document
into a few random pieces, shuffles them, and trains the model to recover the
order. Here we adapt it to *paragraph*-level units over real FinePDFs text and
produce two task flavours ("flows"), both as `Sample`s shaped by the shared
ruler++ output machinery (so the JSONL schema and pt/sft formats match Phase 1):

  flow=reconstruct : input  = shuffled, labelled paragraphs (~n/2 tokens)
                     answer = the document text in its ORIGINAL order (~n/2)
  flow=permutation : input  = shuffled, labelled paragraphs (~n tokens)
                     answer = the label order that reconstructs it (a short list)

where n = --context_len (default 131072 = 128k).

This is a `CorpusTask`: there is no synthetic `size` knob — length is set by how
much source text we pack under the budget. Source text comes from a mix of:
  * clusters : concat each cluster's member docs (rank order) into one long text
               -> the long, knotted, cross-document samples
  * single   : individual FinePDFs docs that are long on their own

Corpus loading/filtering is shared with the clustering stage via `corpus.loader`.
Tokens are measured with the shared pluggable tokenizer (default `whitespace`,
dependency-free; pass --tokenizer hf:<name> for exact budgets).

  python -m untie --output_dir untie_out
  python -m untie --sources single --max_single_docs 2000 --format pt
  python -m untie --target_para_tokens 2000   # smaller chunks, harder reorder
  python -m untie --num_samples 5000          # per-flow target; reuse sources to hit it
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re

from corpus import loader
from ruler_pp.base import CorpusTask, Sample
from ruler_pp.cli import shape_row
from ruler_pp.lengths import get_tokenizer

PARA_SPLIT = re.compile(r"\n\s*\n+")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

INSTR = {
    "reconstruct": (
        "You are given the paragraphs of a long document in shuffled order. Each "
        "paragraph is prefixed with a numeric label like [k] giving its position "
        "in the shuffled list. Reconstruct the original document by writing out "
        "its paragraphs in the correct original order. Output only the "
        "reconstructed document text, paragraphs separated by a blank line, no labels."
    ),
    "permutation": (
        "You are given the paragraphs of a long document in shuffled order. Each "
        "paragraph is prefixed with a numeric label like [k]. Work out the original "
        "ordering of the paragraphs. Output only the sequence of labels that "
        "reconstructs the original document, in order, separated by commas "
        "(e.g. '12, 5, 8, 1'). Output nothing else."
    ),
}
QUESTION = {
    "reconstruct": "Reconstruct the original document.",
    "permutation": "Give the label order that reconstructs the original document.",
}
CONTEXT_PREFIX = "Shuffled paragraphs:\n\n"


# --------------------------------------------------------------------------- #
# text -> paragraph units
# --------------------------------------------------------------------------- #
def make_units(text, count, target_para_tokens, max_para_tokens, rng, jitter=0.0):
    """Split text into paragraph units of roughly target_para_tokens.

    `count(s)` returns a token count. Steps: split on blank lines (fall back to
    single newlines then sentences for badly-formatted PDF text); hard-split any
    paragraph above max_para_tokens into word windows; greedily pack small
    adjacent paragraphs up to a per-unit target. Each unit's target is jittered
    by +/- `jitter` (fraction) around target_para_tokens so chunk sizes vary
    rather than all landing at the same length. Returns [(unit_text, n_tokens)].
    """
    paras = [p.strip() for p in PARA_SPLIT.split(text) if p.strip()]
    if len(paras) < 2:
        paras = [p.strip() for p in text.split("\n") if p.strip()]
    if len(paras) < 2:
        paras = [p.strip() for p in SENT_SPLIT.split(text) if p.strip()]

    win_words = max(1, round(target_para_tokens * 0.75))
    split_paras = []
    for p in paras:
        if count(p) <= max_para_tokens:
            split_paras.append(p)
            continue
        words = p.split()
        for i in range(0, len(words), win_words):
            split_paras.append(" ".join(words[i:i + win_words]))

    def next_target():
        if jitter <= 0:
            return target_para_tokens
        return max(1, target_para_tokens * (1 + rng.uniform(-jitter, jitter)))

    units = []
    buf, buf_tok, cur_target = [], 0, next_target()
    for p in split_paras:
        pt = count(p)
        if buf and buf_tok + pt > cur_target:
            units.append(("\n\n".join(buf), buf_tok))
            buf, buf_tok, cur_target = [], 0, next_target()
        buf.append(p)
        buf_tok += pt
    if buf:
        units.append(("\n\n".join(buf), buf_tok))
    return units


# --------------------------------------------------------------------------- #
# the task
# --------------------------------------------------------------------------- #
class UntieKnots(CorpusTask):
    """One instance per flow. `build` turns paragraph units into a Sample."""

    name = "untie_knots"
    label_overhead = 3  # heuristic tokens for a "[123] " marker

    def __init__(self, flow, context_len=131072, min_units=4):
        if flow not in INSTR:
            raise ValueError(f"unknown flow {flow!r}")
        self.flow = flow
        self.context_len = context_len
        self.min_units = min_units

    def _budget(self):
        # reconstruct: answer mirrors the input, so cap input at half the window.
        # permutation: answer is a short label list; reserve a little headroom.
        if self.flow == "reconstruct":
            return self.context_len // 2
        return self.context_len - 1500

    def _take_span(self, units, budget, rng):
        """Pick a contiguous original-order span of units that fits `budget`."""
        total = sum(t + self.label_overhead for _, t in units)
        if total <= budget:
            start = 0
        else:
            running, max_start = 0, 0
            for i in range(len(units) - 1, -1, -1):
                running += units[i][1] + self.label_overhead
                if running > budget:
                    max_start = i
                    break
            start = rng.randint(0, max_start) if max_start > 0 else 0
        chosen, tot = [], 0
        for u, t in units[start:]:
            c = t + self.label_overhead
            if chosen and tot + c > budget:
                break
            chosen.append((u, t))
            tot += c
        return chosen

    def build(self, source, rng):
        """`source` is the list of (unit_text, n_tokens) for one document."""
        chosen = self._take_span(source, self._budget(), rng)
        k = len(chosen)
        if k < self.min_units:
            return None

        orig = [u for u, _ in chosen]
        perm = list(range(k))
        rng.shuffle(perm)
        display = [orig[perm[d]] for d in range(k)]  # what the model sees, in order
        listing = "\n\n".join(f"[{d + 1}] {display[d]}" for d in range(k))

        # pos[i] = the display label (1-based) holding original unit i; reading
        # display paragraphs in pos-order reproduces the original document.
        pos = [0] * k
        for d, i in enumerate(perm):
            pos[i] = d + 1

        if self.flow == "reconstruct":
            answers = ["\n\n".join(orig)]
            gold = pos                 # the ordering, for verification
            answer_type = "string"
        else:
            answers = [str(p) for p in pos]
            gold = pos
            answer_type = "list"
            assert [display[pos[i] - 1] for i in range(k)] == orig  # round-trip

        return Sample(
            instruction=INSTR[self.flow],
            context=CONTEXT_PREFIX + listing,
            question=QUESTION[self.flow],
            answer_prefix="",
            answers=answers,
            gold=gold,
            answer_type=answer_type,
            meta={"flow": self.flow, "n_units": k},
        )


# --------------------------------------------------------------------------- #
# corpus sources  (loading/filtering shared via corpus.loader)
# --------------------------------------------------------------------------- #
def iter_single_docs(input_dir, min_doc_tokens, min_tokens, min_edu, drop_dups, max_docs):
    """Yield (id, text) for individually-long, quality-filtered docs."""
    for did, _src, txt, _ntok in loader.iter_docs(
            input_dir, min_tokens=max(min_tokens, min_doc_tokens),
            min_edu=min_edu, drop_dups=drop_dups, max_docs=max_docs):
        yield did, txt


def iter_cluster_docs(input_dir, clusters_path, max_clusters, rng):
    """Yield assembled long texts, one per multi-doc cluster (rank order)."""
    import pyarrow.parquet as pq

    if not os.path.exists(clusters_path):
        raise SystemExit(f"clusters file not found: {clusters_path!r} "
                         "(run `python -m corpus.cluster` first, or use --sources single)")
    tb = pq.read_table(clusters_path, columns=["doc_id", "cluster_id", "rank"]).to_pydict()
    groups = {}
    for did, cid, rk in zip(tb["doc_id"], tb["cluster_id"], tb["rank"]):
        groups.setdefault(int(cid), []).append((int(rk), str(did)))
    multi = [(cid, sorted(v)) for cid, v in groups.items() if len(v) >= 2]
    rng.shuffle(multi)
    multi = multi[:max_clusters]
    needed = {did for _, members in multi for _, did in members}
    print(f"clusters: {len(multi)} multi-doc clusters selected, {len(needed)} member docs to fetch")

    texts = loader.fetch_texts_by_id(input_dir, needed)
    print(f"clusters: fetched text for {len(texts)}/{len(needed)} member docs")

    for cid, members in multi:
        parts = [texts[did] for _, did in members if did in texts]
        if len(parts) >= 2:
            yield (f"cluster:{cid}", "\n\n".join(parts))


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser("untie", description="paragraph-unshuffle long-context datagen")
    p.add_argument("--input_dir", default="finpdf_sample")
    p.add_argument("--clusters", default="cluster_out/clusters.parquet")
    p.add_argument("--output_dir", default="untie_out", help="dir for one jsonl per flow")
    p.add_argument("--context_len", type=int, default=131072, help="n: full context budget")
    p.add_argument("--flows", default="reconstruct,permutation", help="comma list: reconstruct, permutation")
    p.add_argument("--sources", default="cluster,single", help="comma list: cluster, single")
    p.add_argument("--format", choices=["pt", "sft"], default="sft",
                   help="pt: flat 'text' field; sft: OpenAI 'messages' array")
    p.add_argument("--tokenizer", default="whitespace", help="hf:<name> | tiktoken[:enc] | whitespace")
    p.add_argument("--target_para_tokens", type=int, default=10000,
                   help="approx tokens per chunk (jittered by --target_jitter)")
    p.add_argument("--target_jitter", type=float, default=0.25,
                   help="per-chunk size varies by +/- this fraction around the target")
    p.add_argument("--max_para_tokens", type=int, default=None,
                   help="paragraphs above this are hard-split into word windows (default: target)")
    p.add_argument("--min_units", type=int, default=2, help="skip samples with fewer chunks")
    p.add_argument("--samples_per_doc", type=int, default=1, help="random spans drawn per source doc per pass")
    p.add_argument("--num_samples", type=int, default=None,
                   help="optional target number of samples PER FLOW. Once the source "
                        "pool is exhausted, reuse it with fresh spans/shuffles until "
                        "the target is hit. Unset = a single pass over the sources.")
    p.add_argument("--max_reuse", type=int, default=5,
                   help="with --num_samples: cap on extra passes over the source pool "
                        "when it is exhausted before the target (stops early if a full "
                        "pass adds nothing). The reuse pool is bounded by --max_clusters "
                        "+ --max_single_docs.")
    p.add_argument("--max_clusters", type=int, default=1000)
    p.add_argument("--max_single_docs", type=int, default=1000)
    p.add_argument("--min_doc_tokens", type=int, default=8000, help="single-doc source: min token_count")
    p.add_argument("--min_tokens", type=int, default=200)
    p.add_argument("--min_edu", type=float, default=0.75)
    p.add_argument("--keep_dups", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    flows = [f.strip() for f in args.flows.split(",") if f.strip()]
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    max_para_tokens = args.max_para_tokens or args.target_para_tokens
    rng = random.Random(args.seed)
    tokenizer = get_tokenizer(args.tokenizer)
    tasks = {f: UntieKnots(f, args.context_len, args.min_units) for f in flows}

    os.makedirs(args.output_dir, exist_ok=True)
    writers = {f: open(os.path.join(args.output_dir, f"{f}.jsonl"), "w", encoding="utf-8")
               for f in flows}
    counts = {f: 0 for f in flows}
    counts["skipped"] = 0
    n_src = 0
    target = args.num_samples

    def all_done():
        return target is not None and all(counts[f] >= target for f in flows)

    def emit(src_id, text):
        nonlocal n_src
        sidx = n_src
        n_src += 1
        # one deterministic rng per emit call so chunking (incl. jitter) is the
        # same across both flows and stable across runs. sidx increments on every
        # call (including reuse passes), so a reused source re-chunks and re-spans
        # under a fresh seed -> a distinct sample rather than a byte-for-byte copy.
        unit_rng = random.Random((args.seed * 2_000_003 + sidx) & 0x7FFFFFFF)
        units = make_units(text, tokenizer.count, args.target_para_tokens,
                           max_para_tokens, unit_rng, args.target_jitter)
        if len(units) < args.min_units:
            counts["skipped"] += 1
            return
        for j in range(args.samples_per_doc):
            for fi, flow in enumerate(flows):
                if target is not None and counts[flow] >= target:
                    continue  # this flow has hit its per-flow target
                # deterministic per (seed, source index, span, flow); int-only so
                # it is stable across runs (unlike salted string hashing).
                seed_i = (((args.seed * 1_000_003 + sidx) * 131 + j) * 7 + fi) & 0x7FFFFFFF
                s = tasks[flow].build(units, random.Random(seed_i))
                if s is None:
                    counts["skipped"] += 1
                    continue
                prompt = s.input
                length = tokenizer.count(prompt)
                row = {
                    "id": f"untie_knots-{flow[:4]}-{counts[flow]:06d}",
                    "task": "untie_knots",
                    "flow": flow,
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
                    "context_len": args.context_len,
                    "source": src_id,
                    "seed": args.seed,
                    "meta": s.meta,
                }
                shape_row(row, args.format, prompt, s.answer_prefix, s.answer_text())
                writers[flow].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[flow] += 1

    def source_stream():
        if "cluster" in sources:
            yield from iter_cluster_docs(args.input_dir, args.clusters, args.max_clusters, rng)
        if "single" in sources:
            yield from iter_single_docs(
                args.input_dir, args.min_doc_tokens, args.min_tokens,
                args.min_edu, not args.keep_dups, args.max_single_docs)

    reuse_passes = 0
    distinct_sources = 0
    try:
        # Pass 1: stream every source once. When a per-flow target is set, also
        # buffer the sources (pool bounded by the --max_* caps) so we can reuse
        # them, and stop as soon as every flow has hit the target.
        buffer = []
        for src_id, text in source_stream():
            emit(src_id, text)
            if target is not None:
                buffer.append((src_id, text))
                if all_done():
                    break
        distinct_sources = n_src  # emit calls so far == distinct sources streamed

        # Reuse: cycle the buffered pool with fresh spans/shuffles until the
        # target is hit, capped at --max_reuse passes; bail if a full pass adds
        # nothing (e.g. every source is too short to ever reach min_units).
        if target is not None and buffer and args.max_reuse > 0:
            for _r in range(args.max_reuse):
                if all_done():
                    break
                before = sum(counts[f] for f in flows)
                for src_id, text in buffer:
                    emit(src_id, text)
                    if all_done():
                        break
                reuse_passes += 1
                if sum(counts[f] for f in flows) == before:
                    break  # a whole pass produced no new samples -> give up
    finally:
        for w in writers.values():
            w.close()

    total = sum(counts[f] for f in flows)
    summary = {
        "output_dir": args.output_dir,
        "source_documents": distinct_sources,
        "source_instances": n_src,  # includes reuse passes
        "samples": total,
        "by_flow": {f: counts[f] for f in flows},
        "skipped": counts["skipped"],
        "num_samples_per_flow": target,
        "target_met": all_done() if target is not None else None,
        "reuse_passes": reuse_passes,
        "context_len": args.context_len,
        "target_para_tokens": args.target_para_tokens,
        "tokenizer": args.tokenizer,
        "format": args.format,
        "sources": sources,
        "seed": args.seed,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
