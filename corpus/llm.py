"""Integrity checks shared by offline and API-driven generation."""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

from ruler_pp.lengths import get_tokenizer
from ruler_pp.output import finalize_row, write_jsonl


def load_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def document_key(prefix, did, text, args):
    settings = {k: v for k, v in vars(args).items()
                if k not in {"func", "output_dir", "input_dir"}}
    value = json.dumps([did, text, settings], sort_keys=True, ensure_ascii=False)
    return prefix + "-" + hashlib.sha256(value.encode()).hexdigest()[:20]


def save_manifest(args, n_docs, n_requests):
    root = Path(args.output_dir)
    if not n_docs or not n_requests:
        raise ValueError("prepare produced no requests; adjust source filters or token budget")
    value = {"version": 1, "tokenizer": args.tokenizer,
             "max_seq_length": args.max_seq_length, "docs": n_docs, "requests": n_requests,
             "sha256": {name: fingerprint(root / name)
                        for name in ("docs.jsonl", "requests.jsonl")}}
    temporary = root / ".manifest.json.tmp"
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(root / "manifest.json")


def validate_response(row):
    """Reject transport failures and explicitly incomplete generations."""
    if row.get("error"):
        raise ValueError("response contains an inference error")
    response = row.get("response", row)
    if not isinstance(response, dict):
        response = row
    status = response.get("status_code", 200)
    if not isinstance(status, int) or not 200 <= status < 300:
        raise ValueError(f"response HTTP status {status}")
    body = response.get("body", response)
    if not isinstance(body, dict) or body.get("error"):
        raise ValueError("invalid response body")
    if body.get("finish_reason") not in (None, "stop", "eos"):
        raise ValueError("incomplete response")
    choices = body.get("choices")
    if choices is not None:
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("expected exactly one completion")
        if choices[0].get("finish_reason") not in (None, "stop", "eos"):
            raise ValueError("incomplete or filtered completion")


def load_bundle(args):
    root = Path(args.requests_dir)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, expected in manifest["sha256"].items():
            if fingerprint(root / name) != expected:
                raise ValueError(f"prepared bundle changed: {name}")
        tokenizer, limit = manifest["tokenizer"], manifest["max_seq_length"]
    else:
        # Legacy dispatcher bundles remain usable with explicit length settings.
        tokenizer, limit = args.tokenizer, args.max_seq_length
        if tokenizer is None or limit is None:
            raise ValueError("legacy bundle needs explicit --tokenizer and --max_seq_length")
    if args.tokenizer is not None:
        tokenizer = args.tokenizer
    if args.max_seq_length is not None:
        limit = args.max_seq_length
    docs, requests = {}, {}
    for row in load_jsonl(root / "docs.jsonl"):
        if row["doc"] in docs or not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("duplicate or empty source document")
        docs[row["doc"]] = row
    for row in load_jsonl(root / "requests.jsonl"):
        if row["id"] in requests or row["doc"] not in docs:
            raise ValueError("duplicate request or missing source document")
        requests[row["id"]] = row
    if not requests:
        raise ValueError("empty request bundle")
    responses = {}
    paths = sorted(glob.glob(args.responses))
    if not paths:
        raise ValueError(f"no responses matched {args.responses!r}")
    for path in paths:
        for row in load_jsonl(path):
            rid = row.get(args.id_field) or row.get("custom_id") or row.get("id")
            if rid not in requests or rid in responses:
                raise ValueError(f"unknown or duplicate response ID: {rid!r}")
            validate_response(row)
            responses[rid] = row
    missing = requests.keys() - responses.keys()
    if missing and not args.allow_partial:
        raise ValueError(f"missing {len(missing)}/{len(requests)} responses; use --allow_partial explicitly")
    return docs, requests, responses, get_tokenizer(tokenizer), limit, len(missing)


def publish(args, task, rows, tokenizer, limit, report):
    if not rows:
        raise ValueError(f"{task}: no usable training rows; no dataset published")
    for row in rows:
        row["max_seq_length"] = limit
        finalize_row(row, tokenizer.count, limit)
    root = Path(args.output_dir)
    write_jsonl(root / f"{task}.jsonl", rows)
    report.update(task=task, rows=len(rows), max_seq_length=limit)
    (root / f"{task}.report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def assembly_arguments(parser):
    parser.add_argument("--allow_partial", action="store_true",
                        help="explicitly allow missing responses; errors/duplicates still fail")
    parser.add_argument("--tokenizer", default=None, help="override prepared tokenizer; required for legacy bundles")
    parser.add_argument("--max_seq_length", type=int, default=None,
                        help="override final training token cap; required for legacy bundles")
