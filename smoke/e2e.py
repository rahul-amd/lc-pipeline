"""Unattended integration run. Invoke on a compute node with a real API server.

python smoke/e2e.py --source finpdf_sample --api-base http://HOST:PORT/v1
Every artifact, sampled document and subprocess log is kept under --output.
The source corpus is read-only. No model quality score is implied by a pass.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

from corpus.loader import iter_docs
from corpus.llm import load_jsonl
from ruler_pp.lengths import get_tokenizer
from ruler_pp.output import training_length, write_jsonl
from scripts.verify_golds import check
from scripts.verify_idk import check_row as check_idk
from scripts.verify_mrcr import check_row as check_mrcr
from mrcr.mrcr import load_markers


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--config", default="smoke/config.json")
    parser.add_argument("--output", default="smoke/runs/e2e")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    root = Path(args.output).resolve()
    if root.exists():
        raise ValueError("use a fresh output directory for an unattended run")
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps(cfg, indent=2))
    tokenizer = get_tokenizer(cfg["tokenizer"])
    count = tokenizer.count
    started = time.time()
    commands = []

    def run(module, *arguments):
        command = [sys.executable, "-m", module, *map(str, arguments)]
        index = len(commands)
        log = root / f"{index:03d}-{module}.log"
        print(f"[{index}] {' '.join(command)}", flush=True)
        with log.open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        commands.append({"argv": command, "returncode": result.returncode, "log": str(log)})
        (root / "commands.json").write_text(json.dumps(commands, indent=2))
        if result.returncode:
            raise RuntimeError(f"stage failed: {log}\n{log.read_text()[-4000:]}")

    # Bounded, reproducible real-corpus sample. ASCII fraction is only a smoke
    # sampling convenience; it is not a production language/quality filter.
    docs, provenance = [], []
    selected = Counter()
    for did, src, text, metadata_tokens in iter_docs(args.source, batch_size=16, max_docs=1000):
        if sum(c.isascii() for c in text) / len(text) < .95:
            continue
        if len(text) > 200000:
            continue
        tokens = count(text)
        bucket = "short" if 2500 <= tokens <= 5500 else "long" if 10000 <= tokens <= 22000 else None
        if bucket is None or selected[bucket] >= 3:
            continue
        docs.append({"id": did, "text": text, "token_count": tokens})
        provenance.append({"doc_id": did, "original_source": src, "bucket": bucket,
                           "tokens": tokens, "original_token_count": metadata_tokens,
                           "sha256": hashlib.sha256(text.encode()).hexdigest()})
        selected[bucket] += 1
        if selected["short"] == selected["long"] == 3:
            break
    if selected["short"] != 3 or selected["long"] != 3:
        raise ValueError(f"insufficient bounded source sample: {selected}")
    corpus = root / "corpus"
    corpus.mkdir()
    # Original quality filters already ran; do not invent replacement scores.
    # Copied text is verbatim and hashed in provenance.json.
    pq.write_table(pa.Table.from_pylist(docs), corpus / "sample.parquet")
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2))

    from huggingface_hub import snapshot_download
    embedding = snapshot_download(cfg["embedding_model"], cache_dir=root / "hf_cache")
    cluster = root / "clusters"
    run("corpus.cluster", "--input_dir", corpus, "--output_dir", cluster,
        "--model", embedding, "--embed_tokens", 256, "--batch_size", 2,
        "--budget", 131072, "--floor", 0.1, "--device", "cuda")
    labels = pq.read_table(cluster / "clusters.parquet").to_pydict()
    if max(Counter(labels["cluster_id"]).values()) < 2:
        raise ValueError("no multi-document cluster to exercise cluster assembly")
    run("corpus.cluster", "--input_dir", corpus, "--output_dir", cluster,
        "--model", embedding, "--budget", 131072, "--floor", 0.1, "--device", "cuda", "--reuse_cache")

    final_files = []
    for length in cfg["lengths"]:
        for fmt in ["pt", "sft"]:
            output = root / f"{length}-{fmt}"
            common = ["--tokenizer", cfg["tokenizer"], "--format", fmt, "--seed", cfg["seed"],
                      "--workers", cfg["workers"]]
            run("ruler_pp", "--task", "all", "--max_seq_length", length,
                "--num_samples", cfg["synthetic_samples_per_task"], "--num_fewshot", 1,
                "--output_dir", output / "ruler", *common)
            final_files.extend(sorted((output / "ruler").glob("*.jsonl")))
            for source in ["cluster", "single"]:
                dest = output / f"untie-{source}"
                run("untie", "--input_dir", corpus, "--clusters", cluster / "clusters.parquet",
                    "--sources", source, "--context_len", length, "--target_para_tokens", length // 16,
                    "--num_samples", 2, "--max_reuse", 4, "--min_doc_tokens", 2000,
                    "--output_dir", dest, *common)
                final_files.extend(sorted(dest.glob("*.jsonl")))
            for family in ["mrcr", "idk"]:
                dest = output / family
                run(family, "--input_dir", corpus, "--max_seq_length", length,
                    "--num_samples", cfg["corpus_samples"], "--pool_size", 200,
                    "--output_dir", dest, *common)
                final_files.append(dest / f"{family}.jsonl")

        for family in ["multihop", "summarize"]:
            prepared = root / f"{length}-llm" / family
            extra = ["--requests_per_doc", cfg["requests_per_doc"]] if family == "multihop" else ["--summary_words", 250]
            run(family, "prepare", "--input_dir", corpus, "--output_dir", prepared,
                "--min_doc_tokens", 2000 if length == 8192 else 10000,
                "--max_seq_length", length, "--num_docs", 2,
                "--tokenizer", cfg["tokenizer"], "--seed", cfg["seed"], *extra)
            run("corpus.infer", "--requests", prepared / "requests.jsonl",
                "--output", prepared / "responses.jsonl", "--base_url", args.api_base,
                "--model", cfg["model"], "--max_tokens", cfg["inference_max_tokens"],
                "--chat_template_kwargs", '{"enable_thinking":false}')
            for fmt in ["pt", "sft"]:
                dest = prepared / fmt
                run(family, "assemble", "--requests_dir", prepared,
                    "--responses", prepared / "responses.jsonl", "--output_dir", dest, "--format", fmt)
                final_files.append(dest / f"{family}.jsonl")

    # Exercise real 128K serialization and large variable-length answer fitting.
    for task in ["graph_descendants", "set_ops", "variable_tracking"]:
        dest = root / "128k"
        run("ruler_pp", "--task", task, "--max_seq_length", 131072, "--num_samples", 2,
            "--output_dir", dest, "--tokenizer", cfg["tokenizer"], "--format", "sft", "--workers", 1)
        final_files.append(dest / f"{task}.jsonl")

    vocab = set(load_markers())
    counts, kinds, flows, modes = Counter(), Counter(), Counter(), Counter()
    seen, lengths = set(), []
    for path in final_files:
        rows = list(load_jsonl(path))
        assert rows, f"empty final artifact: {path}"
        for row in rows:
            assert row["id"] not in seen, f"duplicate ID: {path} {row['id']}"
            seen.add(row["id"])
            actual, mode = training_length(row, count)
            limit = row.get("max_seq_length", row.get("context_len"))
            assert actual == row["training_length"] <= limit, path
            lengths.append(actual)
            modes[mode] += 1
            task = row["task"]
            if task == "mrcr":
                assert check_mrcr(row, vocab)[0], path
            elif task == "idk":
                assert check_idk(row, vocab)[0], path
                kinds[row["meta"]["kind"]] += 1
            elif task == "untie_knots":
                pieces = re.findall(r"(?:^|\n\n)\[(\d+)\] (.*?)(?=\n\n\[\d+\] |\Z)",
                                    row["context"].removeprefix("Shuffled paragraphs:\n\n"), re.S)
                mapping = {int(label): text for label, text in pieces}
                assert sorted(row["gold"]) == sorted(mapping), path
                if row["flow"] == "reconstruct":
                    assert "\n\n".join(mapping[i] for i in row["gold"]) == row["answers"][0]
                flows[row["flow"]] += 1
            elif task not in {"multihop", "summarize"}:
                assert check(row), (path, row["id"])
            counts[task] += 1
    assert set(kinds) == {"answerable", "absent", "subtle"}, kinds
    report = {"status": "passed", "rows": len(seen), "by_task": counts, "idk_kinds": kinds,
              "untie_flows": flows, "length_modes": modes, "max_training_tokens": max(lengths),
              "files": len(final_files), "commands": len(commands),
              "seconds": round(time.time() - started, 1),
              "quality_limit": "Generated QA grounding/multi-hop necessity and summary faithfulness are not scored."}
    (root / "RESULT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
