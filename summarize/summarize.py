"""Long-context summarization datagen over FinePDFs documents (LLM-in-the-loop).

Same two-phase shape as `multihop` — the repo never calls the model; you run the
inference in between — but simpler: one request per document (summarization always
succeeds, so there is no chunking, no over-generation, and no "cannot generate").

  prepare   for each quality-filtered doc, emit ONE request whose user turn is the
            full document followed by the summarization instruction (instruction
            *at the end*, after the long context — the long-context-friendly layout
            also used by mrcr/idk). Writes `requests.jsonl` + `docs.jsonl`.

  assemble  given the engine's responses, take each summary, join it back to its
            document, and write the final `summarize.jsonl` in `pt` or `sft`.

Final shapes (the summarization instruction sits after the document):
  pt   "{doc}\n\n{instruction}\n{summary}"
  sft  [{"role":"user","content":"{doc}\n\n{instruction}"},
        {"role":"assistant","content":"{summary}"}]

    python -m summarize prepare  --input_dir finpdf_sample --output_dir summarize_out
    # ... run requests.jsonl through your inference engine -> responses.jsonl ...
    python -m summarize assemble --requests_dir summarize_out \
        --responses responses.jsonl --format sft
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

from corpus import loader
from multihop.multihop import extract_text, load_jsonl
from ruler_pp.lengths import get_tokenizer

# --------------------------------------------------------------------------- #
# the LLM prompt
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are an expert summarizer. The user will give you a document followed by an "
    "instruction. Write a single, faithful summary of the entire document.\n\n"
    "Requirements:\n"
    "- Cover the document's main points, key facts, and overall arc; do not fixate "
    "on just the opening sections.\n"
    "- Stay strictly faithful to the source: do not add information, opinions, or "
    "facts that are not in the document.\n"
    "- Write clear, well-organized prose. Output only the summary itself, with no "
    "preamble such as \"Here is the summary\" and no surrounding quotes or headings."
)


def final_instruction(summary_words=None):
    """The instruction shown after the document (in both the request and the sample)."""
    if summary_words:
        return ("Write a comprehensive, faithful summary of the document above in "
                f"approximately {summary_words} words.")
    return "Write a comprehensive, faithful summary of the document above."


def build_messages(doc, instruction):
    """system + user turns; the instruction trails the document in the user turn."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{doc}\n\n{instruction}"},
    ]


# --------------------------------------------------------------------------- #
# phase A: prepare requests
# --------------------------------------------------------------------------- #
def prepare(args):
    tokenizer = get_tokenizer(args.tokenizer)
    count = tokenizer.count
    instruction = final_instruction(args.summary_words)

    # Cap the final prompt: it is the doc verbatim plus the instruction and (for
    # pt) the summary, so skip docs longer than max_seq_length minus a reserve.
    max_doc_tokens = (args.max_seq_length - args.reserve_tokens
                      if args.max_seq_length else None)
    if max_doc_tokens is not None and max_doc_tokens <= 0:
        raise SystemExit(f"--max_seq_length {args.max_seq_length} too small for "
                         f"--reserve_tokens {args.reserve_tokens}")

    os.makedirs(args.output_dir, exist_ok=True)
    req_path = os.path.join(args.output_dir, "requests.jsonl")
    doc_path = os.path.join(args.output_dir, "docs.jsonl")

    n_docs = skipped_long = 0
    doc_tok_sum = 0
    with open(req_path, "w", encoding="utf-8") as rf, \
            open(doc_path, "w", encoding="utf-8") as df:
        for did, src, txt, _ntok in loader.iter_docs(
                args.input_dir, min_tokens=args.min_doc_tokens, min_edu=args.min_edu,
                drop_dups=not args.keep_dups, max_docs=args.max_docs):
            doc_tokens = count(txt)
            if max_doc_tokens is not None and doc_tokens > max_doc_tokens:
                skipped_long += 1
                continue

            doc_local = f"sum-{n_docs:06d}"
            doc_tok_sum += doc_tokens
            df.write(json.dumps(
                {"doc": doc_local, "doc_id": did, "source": src,
                 "doc_tokens": doc_tokens, "instruction": instruction, "text": txt},
                ensure_ascii=False) + "\n")
            rf.write(json.dumps(
                {"id": doc_local, "doc": doc_local, "doc_id": did,
                 "messages": build_messages(txt, instruction)},
                ensure_ascii=False) + "\n")
            n_docs += 1
            if args.num_docs and n_docs >= args.num_docs:
                break

    print(json.dumps({
        "phase": "prepare",
        "output_dir": args.output_dir,
        "requests_file": req_path,
        "docs_file": doc_path,
        "docs": n_docs,
        "requests": n_docs,
        "skipped_too_long": skipped_long,
        "max_seq_length": args.max_seq_length,
        "max_doc_tokens": max_doc_tokens,
        "avg_doc_tokens": round(doc_tok_sum / n_docs) if n_docs else 0,
        "summary_words": args.summary_words,
        "tokenizer": args.tokenizer,
        "seed": args.seed,
    }, indent=2))


# --------------------------------------------------------------------------- #
# phase B: assemble final dataset from responses
# --------------------------------------------------------------------------- #
_PREAMBLE_RE = re.compile(r"^\s*(?:here\s+is\s+(?:a\s+|the\s+)?summary[:.]?|summary[:.]?)\s*",
                          re.IGNORECASE)


def clean_summary(text):
    """Strip whitespace, a stray leading 'Summary:' preamble, and wrapping quotes."""
    s = (text or "").strip()
    s = _PREAMBLE_RE.sub("", s, count=1).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    return s


def assemble(args):
    docs = {r["doc"]: r for r in load_jsonl(os.path.join(args.requests_dir, "docs.jsonl"))}
    req_to_doc = {r["id"]: r["doc"]
                  for r in load_jsonl(os.path.join(args.requests_dir, "requests.jsonl"))}

    resp_files = sorted(glob.glob(args.responses)) if any(c in args.responses for c in "*?[") \
        else [args.responses]
    if not resp_files:
        raise SystemExit(f"no response files matched {args.responses!r}")

    summaries = {}        # doc_local -> summary text
    n_resp = n_empty = n_unmatched = 0
    for path in resp_files:
        for row in load_jsonl(path):
            n_resp += 1
            rid = row.get(args.id_field) or row.get("id") or row.get("custom_id")
            doc = req_to_doc.get(rid)
            if doc is None:
                n_unmatched += 1
                continue
            summary = clean_summary(extract_text(row, args.text_field))
            if not summary:
                n_empty += 1
                continue
            summaries[doc] = summary

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "summarize.jsonl")
    written = 0
    with open(out, "w", encoding="utf-8") as f:
        for doc in sorted(summaries):
            meta_doc = docs.get(doc, {})
            text = meta_doc.get("text", "")
            instruction = meta_doc.get("instruction", final_instruction())
            summary = summaries[doc]
            row = {
                "id": doc,
                "task": "summarize",
                "format": args.format,
                "doc_id": meta_doc.get("doc_id"),
                "source": meta_doc.get("source"),
                "doc_tokens": meta_doc.get("doc_tokens"),
                "instruction": instruction,
                "summary": summary,
            }
            shape_summarize(row, args.format, text, instruction, summary)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(json.dumps({
        "phase": "assemble",
        "output_dir": args.output_dir,
        "file": out,
        "responses_read": n_resp,
        "summaries_written": written,
        "empty_or_unparsed": n_empty,
        "unmatched_ids": n_unmatched,
        "format": args.format,
    }, indent=2))


def shape_summarize(row, fmt, doc, instruction, summary):
    """Attach the final training field(s) — instruction sits after the document."""
    if fmt == "pt":
        row["text"] = f"{doc}\n\n{instruction}\n{summary}"
    else:  # sft
        row["messages"] = [
            {"role": "user", "content": f"{doc}\n\n{instruction}"},
            {"role": "assistant", "content": summary},
        ]


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser("summarize", description="Long-context summarization datagen (LLM-in-the-loop)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="emit one summarization request per doc")
    pp.add_argument("--input_dir", default="finpdf_sample")
    pp.add_argument("--output_dir", default="summarize_out")
    pp.add_argument("--num_docs", type=int, default=None, help="cap documents processed")
    pp.add_argument("--summary_words", type=int, default=None,
                    help="target summary length in words (added to the instruction); unset = open-ended")
    pp.add_argument("--tokenizer", default="whitespace",
                    help="hf:<name> | tiktoken[:enc] | whitespace")
    pp.add_argument("--max_seq_length", type=int, default=None,
                    help="cap on the final prompt; skip docs longer than "
                         "max_seq_length - reserve_tokens (measured with --tokenizer). "
                         "Unset = no upper bound")
    pp.add_argument("--reserve_tokens", type=int, default=1024,
                    help="tokens held back from --max_seq_length for the instruction + summary")
    pp.add_argument("--min_doc_tokens", type=int, default=4000, help="skip docs shorter than this")
    pp.add_argument("--min_edu", type=float, default=0.75)
    pp.add_argument("--keep_dups", action="store_true")
    pp.add_argument("--max_docs", type=int, default=None, help="cap docs scanned in the corpus")
    pp.add_argument("--seed", type=int, default=42)
    pp.set_defaults(func=prepare)

    pa = sub.add_parser("assemble", help="stitch LLM summaries into the final dataset")
    pa.add_argument("--requests_dir", default="summarize_out",
                    help="dir holding requests.jsonl + docs.jsonl from prepare")
    pa.add_argument("--responses", required=True, help="LLM responses jsonl (glob allowed)")
    pa.add_argument("--output_dir", default=None,
                    help="where to write summarize.jsonl (default: --requests_dir)")
    pa.add_argument("--format", choices=["pt", "sft"], default="sft")
    pa.add_argument("--id_field", default=None, help="response key holding the request id")
    pa.add_argument("--text_field", default=None, help="response key holding the generated text")
    pa.set_defaults(func=assemble)

    args = p.parse_args(argv)
    if getattr(args, "cmd", None) == "assemble" and not args.output_dir:
        args.output_dir = args.requests_dir
    args.func(args)


if __name__ == "__main__":
    main()
