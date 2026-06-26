"""Multi-hop QA generation over FinePDFs documents (LLM-in-the-loop, two phases).

Unlike the other generators, this one needs an LLM to *write* the questions, so it
is split into two offline phases with the inference run in between (we never call
the model ourselves):

  prepare   chunk each document into ~512-token chunks, sample a few chunks per
            request, and emit a `messages` array per request to `requests.jsonl`
            (plus `docs.jsonl` mapping each doc id back to its full text). You feed
            `requests.jsonl` to any inference engine.

  assemble  given the engine's responses, parse the `Question:/Answer:` pairs
            (dropping any "cannot generate"), group them back per document, and
            write the final `multihop.jsonl` in `pt` or `sft` format.

The model is asked to write ONE multi-hop QA pair per request whose answer needs a
fact from *every* chunk shown; if the chunks share no usable common ground it
replies with the exact phrase "cannot generate". We over-generate requests per doc
so that, after the inevitable "cannot generate" misses, each document still yields
roughly the target number of pairs.

    python -m multihop prepare  --input_dir finpdf_sample --output_dir multihop_out
    # ... run requests.jsonl through your inference engine -> responses.jsonl ...
    python -m multihop assemble --requests_dir multihop_out \
        --responses responses.jsonl --format sft
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re

from corpus import loader
from ruler_pp.lengths import get_tokenizer

# --------------------------------------------------------------------------- #
# the LLM prompt
# --------------------------------------------------------------------------- #
CANNOT = "cannot generate"

SYSTEM_PROMPT = (
    "You are a meticulous dataset author. You will be shown several text chunks "
    "taken from the same source document. Your job is to write ONE multi-hop "
    "question-answer pair.\n\n"
    "Requirements:\n"
    "- The question must require combining information from ALL of the chunks "
    "shown. Each chunk must contribute at least one fact that is necessary to "
    "answer the question.\n"
    "- The question must NOT be answerable from any single chunk on its own.\n"
    "- The question and answer must be fully grounded in the chunks. Do not rely "
    "on outside knowledge and do not invent facts.\n"
    "- Keep the answer concise and correct.\n\n"
    "If the chunks do not share enough common ground to support such a question — "
    "for example they are about unrelated topics, or the text is too low quality — "
    "do NOT force one. In that case reply with exactly this phrase and nothing "
    f"else:\n{CANNOT}\n\n"
    "Otherwise reply in EXACTLY this format and nothing else:\n"
    "Question: <your question>\n"
    "Answer: <your answer>"
)


def build_user_message(chunks):
    """Render the sampled chunks as the user turn."""
    parts = ["Here are the chunks from the document:\n"]
    for i, c in enumerate(chunks, 1):
        parts.append(f"Chunk {i}:\n{c}")
    return "\n\n".join(parts)


def build_messages(chunks):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(chunks)},
    ]


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #
def chunk_doc(text, count, *, chunk_tokens, min_chunk_tokens):
    """Greedily pack a document's words into ~`chunk_tokens`-token chunks.

    Measured with the chosen tokenizer so the target holds for sub-word encoders
    too (for the whitespace tokenizer it is exactly `chunk_tokens` words). A
    trailing chunk shorter than `min_chunk_tokens` is dropped.
    """
    words = text.split()
    n = len(words)
    chunks = []
    i = 0
    step = max(1, chunk_tokens // 4)
    while i < n:
        hi = min(n, i + chunk_tokens)  # whitespace: already on target
        while hi < n and count(" ".join(words[i:hi])) < chunk_tokens:
            hi = min(n, hi + step)
        chunk = " ".join(words[i:hi])
        if count(chunk) >= min_chunk_tokens:
            chunks.append(chunk)
        i = hi
    return chunks


# --------------------------------------------------------------------------- #
# phase A: prepare requests
# --------------------------------------------------------------------------- #
def prepare(args):
    tokenizer = get_tokenizer(args.tokenizer)
    count = tokenizer.count

    os.makedirs(args.output_dir, exist_ok=True)
    req_path = os.path.join(args.output_dir, "requests.jsonl")
    doc_path = os.path.join(args.output_dir, "docs.jsonl")

    n_docs = n_requests = 0
    skipped_short = 0
    with open(req_path, "w", encoding="utf-8") as rf, \
            open(doc_path, "w", encoding="utf-8") as df:
        for did, src, txt, _ntok in loader.iter_docs(
                args.input_dir, min_tokens=args.min_doc_tokens, min_edu=args.min_edu,
                drop_dups=not args.keep_dups, max_docs=args.max_docs):
            chunks = chunk_doc(txt, count, chunk_tokens=args.chunk_tokens,
                               min_chunk_tokens=args.min_chunk_tokens)
            if len(chunks) < args.min_doc_chunks:
                skipped_short += 1
                continue

            doc_local = f"mh-{n_docs:06d}"
            rng = random.Random(args.seed * 1_000_003 + n_docs)
            df.write(json.dumps(
                {"doc": doc_local, "doc_id": did, "source": src,
                 "n_chunks": len(chunks), "text": txt},
                ensure_ascii=False) + "\n")

            for j in range(args.requests_per_doc):
                k = min(len(chunks), rng.randint(args.min_chunks, args.max_chunks))
                idx = sorted(rng.sample(range(len(chunks)), k))
                rf.write(json.dumps(
                    {"id": f"{doc_local}-q{j:02d}", "doc": doc_local, "doc_id": did,
                     "chunk_ids": idx, "messages": build_messages([chunks[c] for c in idx])},
                    ensure_ascii=False) + "\n")
                n_requests += 1

            n_docs += 1
            if args.num_docs and n_docs >= args.num_docs:
                break

    print(json.dumps({
        "phase": "prepare",
        "output_dir": args.output_dir,
        "requests_file": req_path,
        "docs_file": doc_path,
        "docs": n_docs,
        "requests": n_requests,
        "requests_per_doc": args.requests_per_doc,
        "skipped_too_few_chunks": skipped_short,
        "chunk_tokens": args.chunk_tokens,
        "tokenizer": args.tokenizer,
        "seed": args.seed,
    }, indent=2))


# --------------------------------------------------------------------------- #
# phase B: assemble final dataset from responses
# --------------------------------------------------------------------------- #
QA_RE = re.compile(r"Question:\s*(.*?)\s*Answer:\s*(.*)", re.DOTALL | re.IGNORECASE)


def extract_text(row, text_field=None):
    """Best-effort pull of the generated text from one response row.

    Handles our simple schema and common engine outputs (OpenAI chat/batch, vLLM,
    raw text fields). `text_field` forces a top-level key when given.
    """
    if text_field:
        return row.get(text_field, "")
    for key in ("response", "completion", "output", "generated_text", "content", "text"):
        v = row.get(key)
        if isinstance(v, str) and v:
            return v
    # OpenAI-style choices
    choices = row.get("choices")
    if isinstance(choices, list) and choices:
        ch = choices[0]
        if isinstance(ch, dict):
            msg = ch.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
            if isinstance(ch.get("text"), str):
                return ch["text"]
    # batch API wrapper: {"response": {"body": {"choices": [...]}}}
    body = (row.get("response") or {})
    if isinstance(body, dict):
        body = body.get("body", body)
        chs = body.get("choices") if isinstance(body, dict) else None
        if isinstance(chs, list) and chs and isinstance(chs[0], dict):
            msg = chs[0].get("message", {})
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
    # trailing assistant turn
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs and isinstance(msgs[-1], dict):
        return msgs[-1].get("content", "") or ""
    return ""


def parse_pair(text):
    """Return (question, answer) or None for a 'cannot generate' / unparseable reply."""
    t = (text or "").strip()
    if not t or t.lower().startswith(CANNOT):
        return None
    m = QA_RE.search(t)
    if not m:
        return None
    q, a = m.group(1).strip(), m.group(2).strip()
    if not q or not a or a.lower().startswith(CANNOT):
        return None
    return q, a


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def assemble(args):
    docs = {r["doc"]: r for r in load_jsonl(os.path.join(args.requests_dir, "docs.jsonl"))}
    req_to_doc = {r["id"]: r["doc"]
                  for r in load_jsonl(os.path.join(args.requests_dir, "requests.jsonl"))}

    resp_files = sorted(glob.glob(args.responses)) if any(c in args.responses for c in "*?[") \
        else [args.responses]
    if not resp_files:
        raise SystemExit(f"no response files matched {args.responses!r}")

    pairs_by_doc = {}        # doc_local -> [(q, a), ...] in request order
    n_resp = n_pairs = n_cannot = n_unmatched = 0
    for path in resp_files:
        for row in load_jsonl(path):
            n_resp += 1
            rid = row.get(args.id_field) or row.get("id") or row.get("custom_id")
            doc = req_to_doc.get(rid)
            if doc is None:
                n_unmatched += 1
                continue
            pair = parse_pair(extract_text(row, args.text_field))
            if pair is None:
                n_cannot += 1
                continue
            order = int(rid.rsplit("-q", 1)[1]) if "-q" in rid else len(pairs_by_doc.get(doc, []))
            pairs_by_doc.setdefault(doc, []).append((order, pair))
            n_pairs += 1

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "multihop.jsonl")
    kept_docs = dropped_docs = kept_pairs = 0
    with open(out, "w", encoding="utf-8") as f:
        for doc in sorted(pairs_by_doc):
            qas = [p for _o, p in sorted(pairs_by_doc[doc], key=lambda t: t[0])]
            qas = qas[:args.max_pairs]
            if len(qas) < args.min_pairs:
                dropped_docs += 1
                continue
            meta_doc = docs.get(doc, {})
            text = meta_doc.get("text", "")
            row = {
                "id": doc,
                "task": "multihop",
                "format": args.format,
                "doc_id": meta_doc.get("doc_id"),
                "source": meta_doc.get("source"),
                "n_pairs": len(qas),
                "qa_pairs": [{"question": q, "answer": a} for q, a in qas],
            }
            shape_multihop(row, args.format, text, qas)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept_docs += 1
            kept_pairs += len(qas)

    print(json.dumps({
        "phase": "assemble",
        "output_dir": args.output_dir,
        "file": out,
        "responses_read": n_resp,
        "pairs_parsed": n_pairs,
        "cannot_generate_or_unparsed": n_cannot,
        "unmatched_ids": n_unmatched,
        "docs_written": kept_docs,
        "docs_dropped_below_min": dropped_docs,
        "pairs_written": kept_pairs,
        "avg_pairs_per_doc": round(kept_pairs / kept_docs, 2) if kept_docs else 0,
        "format": args.format,
    }, indent=2))


def shape_multihop(row, fmt, doc, qas):
    """Attach the final training field(s) for one document's QA set."""
    if fmt == "pt":
        body = "\n".join(f"Question: {q}\nAnswer: {a}" for q, a in qas)
        row["text"] = f"{doc}\n\n{body}"
    else:  # sft
        user = f"{doc}\n\n" + "\n".join(f"{i}. {q}" for i, (q, _a) in enumerate(qas, 1))
        assistant = "\n".join(f"{i}. {a}" for i, (_q, a) in enumerate(qas, 1))
        row["messages"] = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser("multihop", description="Multi-hop QA datagen (LLM-in-the-loop)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="chunk docs and emit LLM requests")
    pp.add_argument("--input_dir", default="finpdf_sample")
    pp.add_argument("--output_dir", default="multihop_out")
    pp.add_argument("--num_docs", type=int, default=None, help="cap documents processed")
    pp.add_argument("--requests_per_doc", type=int, default=13,
                    help="QA requests emitted per doc (over-generated vs the pair target)")
    pp.add_argument("--min_chunks", type=int, default=3, help="min chunks sampled per request")
    pp.add_argument("--max_chunks", type=int, default=4, help="max chunks sampled per request")
    pp.add_argument("--chunk_tokens", type=int, default=512, help="target tokens per chunk")
    pp.add_argument("--min_chunk_tokens", type=int, default=128,
                    help="drop a trailing chunk shorter than this")
    pp.add_argument("--min_doc_chunks", type=int, default=4,
                    help="skip docs that yield fewer than this many chunks")
    pp.add_argument("--tokenizer", default="whitespace",
                    help="hf:<name> | tiktoken[:enc] | whitespace")
    pp.add_argument("--min_doc_tokens", type=int, default=2000, help="skip docs shorter than this")
    pp.add_argument("--min_edu", type=float, default=0.75)
    pp.add_argument("--keep_dups", action="store_true")
    pp.add_argument("--max_docs", type=int, default=None, help="cap docs scanned in the corpus")
    pp.add_argument("--seed", type=int, default=42)
    pp.set_defaults(func=prepare)

    pa = sub.add_parser("assemble", help="stitch LLM responses into the final dataset")
    pa.add_argument("--requests_dir", default="multihop_out",
                    help="dir holding requests.jsonl + docs.jsonl from prepare")
    pa.add_argument("--responses", required=True,
                    help="LLM responses jsonl (glob allowed)")
    pa.add_argument("--output_dir", default=None,
                    help="where to write multihop.jsonl (default: --requests_dir)")
    pa.add_argument("--format", choices=["pt", "sft"], default="sft")
    pa.add_argument("--max_pairs", type=int, default=10, help="cap QA pairs kept per doc")
    pa.add_argument("--min_pairs", type=int, default=1, help="drop docs with fewer valid pairs")
    pa.add_argument("--id_field", default=None, help="response key holding the request id")
    pa.add_argument("--text_field", default=None, help="response key holding the generated text")
    pa.set_defaults(func=assemble)

    args = p.parse_args(argv)
    if getattr(args, "cmd", None) == "assemble" and not args.output_dir:
        args.output_dir = args.requests_dir
    args.func(args)


if __name__ == "__main__":
    main()
