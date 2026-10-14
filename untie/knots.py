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
import itertools
import json
import multiprocessing as mp
import os
import random
import re

from corpus import loader
from corpus.chunks import split_text
from ruler_pp.base import CorpusTask, Sample
from ruler_pp.cli import resolve_workers, shape_row
from ruler_pp.lengths import get_tokenizer
from ruler_pp.output import finalize_row, training_length

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

    split_paras = []
    for p in paras:
        if count(p) <= max_para_tokens:
            split_paras.append(p)
            continue
        split_paras.extend(split_text(p, count, max_para_tokens))

    def next_target():
        if jitter <= 0:
            return target_para_tokens
        return max(1, target_para_tokens * (1 + rng.uniform(-jitter, jitter)))

    units = []
    buf, buf_tok, cur_target = [], 0, next_target()
    for p in split_paras:
        pt = count(p)
        if buf and buf_tok + pt > cur_target:
            units.append(("\n\n".join(buf), count("\n\n".join(buf))))
            buf, buf_tok, cur_target = [], 0, next_target()
        buf.append(p)
        buf_tok += pt
    if buf:
        units.append(("\n\n".join(buf), count("\n\n".join(buf))))
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
# pure per-source computation (parallelizable) + sequential commit
# --------------------------------------------------------------------------- #
# The heavy work (chunking + build + tokenizing the full prompt) is a pure
# function of (seed, sidx, source text), so it is safe to run across processes.
# The stateful accounting — per-flow counters, target gating, row ids, the reuse
# loop and its early-breaks — stays in the single-process `commit` so output is
# byte-identical regardless of worker count.
def compute_source(sidx, src_id, text, *, flows, samples_per_doc, tasks, count, seed,
                   target_para_tokens, max_para_tokens, target_jitter, min_units,
                   fmt, context_len):
    """Return {"units_ok", "src_id", "items"} for one source instance.

    `items` is a list of (flow, row-or-None) in the exact (j, flow) order the
    serial `emit` produced them; `row` omits the `id` (assigned at commit)."""
    unit_rng = random.Random((seed * 2_000_003 + sidx) & 0x7FFFFFFF)
    units = make_units(text, count, target_para_tokens, max_para_tokens, unit_rng, target_jitter)
    if len(units) < min_units:
        return {"units_ok": False, "src_id": src_id, "items": []}
    items = []
    for j in range(samples_per_doc):
        for fi, flow in enumerate(flows):
            seed_i = (((seed * 1_000_003 + sidx) * 131 + j) * 7 + fi) & 0x7FFFFFFF
            s = tasks[flow].build(units, random.Random(seed_i))
            if s is None:
                items.append((flow, None))
                continue
            # Fit the actual serialized answer too; reconstruct duplicates the text.
            trial = shape_row({"format": fmt}, fmt, s.input, s.answer_prefix, s.answer_text())
            trial_units = units
            while training_length(trial, count)[0] > context_len and len(trial_units) > min_units:
                trial_units = trial_units[:max(min_units, int(len(trial_units) * 0.9))]
                s = tasks[flow].build(trial_units, random.Random(seed_i))
                if s is None:
                    break
                trial = shape_row({"format": fmt}, fmt, s.input, s.answer_prefix, s.answer_text())
            if s is None or training_length(trial, count)[0] > context_len:
                items.append((flow, None))
                continue
            prompt = s.input
            length = count(prompt)
            row = {
                "task": "untie_knots",
                "flow": flow,
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
                "context_len": context_len,
                "source": src_id,
                "seed": seed,
                "meta": s.meta,
            }
            shape_row(row, fmt, prompt, s.answer_prefix, s.answer_text())
            finalize_row(row, count, context_len)
            items.append((flow, row))
    return {"units_ok": True, "src_id": src_id, "items": items}


# Per-worker state (populated once by the pool initializer). The HF tokenizer is
# not picklable, so each worker rebuilds it (and the cheap task objects) here.
_W: dict = {}


def _worker_init(tokenizer_spec, kw):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    tok = get_tokenizer(tokenizer_spec)
    tasks = {f: UntieKnots(f, kw["context_len"], kw["min_units"]) for f in kw["flows"]}
    _W.update(count=tok.count, tasks=tasks, kw=kw)


def _worker_compute(payload):
    sidx, src_id, text = payload
    kw = _W["kw"]
    return compute_source(
        sidx, src_id, text, flows=kw["flows"], samples_per_doc=kw["samples_per_doc"],
        tasks=_W["tasks"], count=_W["count"], seed=kw["seed"],
        target_para_tokens=kw["target_para_tokens"], max_para_tokens=kw["max_para_tokens"],
        target_jitter=kw["target_jitter"], min_units=kw["min_units"],
        fmt=kw["fmt"], context_len=kw["context_len"])


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
    p.add_argument("--workers", type=int, default=0,
                   help="parallel source-compute processes (0 = all CPU cores). "
                        "Output is identical regardless of worker count.")
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
    if not flows or set(flows) - INSTR.keys() or len(flows) != len(set(flows)):
        p.error("flows must be unique values from reconstruct,permutation")
    if not sources or set(sources) - {"single", "cluster"}:
        p.error("sources must be cluster and/or single")
    if min(args.context_len, args.min_units, args.target_para_tokens, args.samples_per_doc) < 1:
        p.error("length, units and samples must be positive")
    if args.num_samples is not None and args.num_samples < 1:
        p.error("num_samples must be positive")
    max_para_tokens = args.max_para_tokens or args.target_para_tokens
    rng = random.Random(args.seed)
    tokenizer = get_tokenizer(args.tokenizer)
    tasks = {f: UntieKnots(f, args.context_len, args.min_units) for f in flows}

    os.makedirs(args.output_dir, exist_ok=True)
    writers = {f: open(os.path.join(args.output_dir, f".{f}.jsonl.tmp"), "w", encoding="utf-8")
               for f in flows}
    counts = {f: 0 for f in flows}
    counts["skipped"] = 0
    n_src = 0
    target = args.num_samples

    def all_done():
        return target is not None and all(counts[f] >= target for f in flows)

    # commit is the single-process accounting: it assigns row ids from the live
    # per-flow counters, applies target gating, and writes. Identical to the old
    # `emit`'s bookkeeping, just fed pre-computed rows. `sidx` (baked into the
    # sample seeds by compute_source) must equal the value n_src holds *before*
    # this source is committed — the driver assigns it accordingly.
    def commit(res):
        nonlocal n_src
        n_src += 1
        if not res["units_ok"]:
            counts["skipped"] += 1
            return
        for flow, row in res["items"]:
            if target is not None and counts[flow] >= target:
                continue  # this flow has hit its per-flow target
            if row is None:
                counts["skipped"] += 1
                continue
            out_row = {"id": f"untie_knots-{flow[:4]}-{counts[flow]:06d}", **row}
            writers[flow].write(json.dumps(out_row, ensure_ascii=False) + "\n")
            counts[flow] += 1

    def source_stream():
        if "cluster" in sources:
            yield from iter_cluster_docs(args.input_dir, args.clusters, args.max_clusters, rng)
        if "single" in sources:
            yield from iter_single_docs(
                args.input_dir, args.min_doc_tokens, args.min_tokens,
                args.min_edu, not args.keep_dups, args.max_single_docs)

    workers = resolve_workers(args.workers)
    kw = dict(flows=flows, samples_per_doc=args.samples_per_doc, context_len=args.context_len,
              min_units=args.min_units, seed=args.seed, target_para_tokens=args.target_para_tokens,
              max_para_tokens=max_para_tokens, target_jitter=args.target_jitter, fmt=args.format)

    def _compute_local(payload):
        sidx, src_id, text = payload
        return compute_source(sidx, src_id, text, tasks=tasks, count=tokenizer.count, **kw)

    pool = None
    if workers > 1:
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context("spawn")
        pool = ctx.Pool(workers, initializer=_worker_init, initargs=(args.tokenizer, kw))

    def mapper(batch):
        """Compute a batch of source instances, results in input order."""
        if pool is None:
            return [_compute_local(p) for p in batch]
        return pool.map(_worker_compute, batch, chunksize=max(1, len(batch) // (workers * 4)))

    # over-computation on an early break is capped at one batch (results are just
    # discarded and their sources are not buffered), so output stays exact.
    batch_size = max(64, workers * 16)

    reuse_passes = 0
    distinct_sources = 0
    try:
        # Pass 1: stream every source once, in fixed batches. sidx == stream
        # position (== n_src at commit, since we commit contiguously). When a
        # per-flow target is set, buffer committed sources for reuse and stop as
        # soon as every flow has hit the target.
        buffer = []
        stream = source_stream()
        pos = 0
        done = False
        while not done:
            chunk = list(itertools.islice(stream, batch_size))
            if not chunk:
                break
            batch = [(pos + off, src_id, text) for off, (src_id, text) in enumerate(chunk)]
            pos += len(chunk)
            for (sidx, src_id, text), res in zip(batch, mapper(batch)):
                commit(res)
                if target is not None:
                    buffer.append((src_id, text))
                    if all_done():
                        done = True
                        break
        distinct_sources = n_src  # sources committed so far == distinct sources streamed

        # Reuse: cycle the buffered pool with fresh spans/shuffles until the
        # target is hit, capped at --max_reuse passes; bail if a full pass adds
        # nothing (e.g. every source is too short to ever reach min_units).
        if target is not None and buffer and args.max_reuse > 0:
            for _r in range(args.max_reuse):
                if all_done():
                    break
                before = sum(counts[f] for f in flows)
                base = n_src  # sidx continues globally across passes
                broke = False
                for start in range(0, len(buffer), batch_size):
                    chunk = buffer[start:start + batch_size]
                    batch = [(base + start + off, src_id, text)
                             for off, (src_id, text) in enumerate(chunk)]
                    for (sidx, src_id, text), res in zip(batch, mapper(batch)):
                        commit(res)
                        if all_done():
                            broke = True
                            break
                    if broke:
                        break
                reuse_passes += 1
                if sum(counts[f] for f in flows) == before:
                    break  # a whole pass produced no new samples -> give up
    finally:
        if pool is not None:
            pool.close()
            pool.join()
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
        "workers": workers,
        "tokenizer": args.tokenizer,
        "format": args.format,
        "sources": sources,
        "seed": args.seed,
    }
    print(json.dumps(summary, indent=2))
    if not total or (target is not None and not all_done()):
        raise SystemExit("generation did not meet the requested sample target")
    for flow in flows:
        os.replace(os.path.join(args.output_dir, f".{flow}.jsonl.tmp"),
                   os.path.join(args.output_dir, f"{flow}.jsonl"))


if __name__ == "__main__":
    main()
