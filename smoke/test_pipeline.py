"""Regression tests for data integrity; run with --basetemp under smoke/runs."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from corpus import loader
from corpus.chunks import split_text
from corpus.cluster import greedy_chain, topk_neighbours
from corpus.llm import load_bundle, validate_response
from idk.idk import IDKTask, build_row as idk_row
from mrcr.mrcr import MRCRTask, build_row as mrcr_row, load_markers
from multihop.multihop import chunk_doc, extract_text, main as mh_main
from ruler_pp.cli import generate_task
from ruler_pp.lengths import fit_size, get_tokenizer
from ruler_pp.output import training_length, write_jsonl
from ruler_pp.registry import TASKS, get_task
from scripts.verify_golds import check
from scripts.verify_idk import check_row as check_idk
from scripts.verify_mrcr import check_row as check_mrcr
from summarize.summarize import clean_summary, main as sum_main
from untie.knots import UntieKnots, compute_source

TOK = get_tokenizer("whitespace")
POOL = [f"Source passage {i}. " + "The equipment was tested at the laboratory. " * 10 for i in range(30)]


@pytest.mark.parametrize("task_name", list(TASKS))
@pytest.mark.parametrize("fmt", ["pt", "sft"])
def test_synthetic_gold_and_full_length(task_name, fmt):
    rows, _ = generate_task(get_task(task_name), TOK, 3, 8192, 17, 1, fmt)
    for row in rows:
        assert check(row), row["id"]
        assert training_length(row, TOK.count)[0] == row["training_length"] <= 8192
    assert len({r["id"] for r in rows}) == len(rows)


@pytest.mark.parametrize("task_name", list(TASKS))
def test_minimum_gold(task_name):
    task = get_task(task_name)
    for seed in range(30):
        sample = task.build(task.min_size, random.Random(seed))
        assert check(dict(task=task_name, context=sample.context, question=sample.question,
                          gold=sample.gold, meta=sample.meta)), (task_name, seed)


def test_impossible_budget_and_unknown_tokenizer():
    with pytest.raises(ValueError):
        fit_size(lambda n: n + 100, 10, 1)
    with pytest.raises(ValueError):
        generate_task(get_task("common_words"), TOK, 1, 20, 0, 0, "pt")
    with pytest.raises(ValueError):
        get_tokenizer("typo")


def test_cross_process_hash_determinism():
    code = """import json, random
from ruler_pp.registry import get_task
for name in ['freq_words', 'variable_tracking', 'variable_arithmetic']:
 print(json.dumps(get_task(name).build(100, random.Random(7)).__dict__, sort_keys=True))
"""
    values = [subprocess.check_output([sys.executable, "-c", code],
              env={**os.environ, "PYTHONHASHSEED": str(seed)}) for seed in [1, 2]]
    assert values[0] == values[1]


def test_parallel_reproducibility():
    task = get_task("variable_tracking")
    args = (task, TOK, 4, 8192, 42, 0, "sft")
    assert generate_task(*args, tokenizer_spec="whitespace", workers=1)[0] == generate_task(
        *args, tokenizer_spec="whitespace", workers=2)[0]


def test_unique_ids_across_buckets_and_seeds():
    rows = [generate_task(get_task("count_distinct"), TOK, 1, length, seed, 0, "sft")[0][0]
            for length in [4096, 8192] for seed in [1, 2]]
    assert len({r["id"] for r in rows}) == 4


def test_source_ids_batch_shard_and_schema(tmp_path):
    for directory, col in [("a", "text"), ("b", "content")]:
        (tmp_path / directory).mkdir()
        pq.write_table(pa.table({col: ["a", "", "c", "d", "e"]}), tmp_path / directory / "same.parquet")
    small = list(loader.iter_docs(str(tmp_path), batch_size=2))
    large = list(loader.iter_docs(str(tmp_path), batch_size=7))
    assert small == large
    assert len({r[0] for r in small}) == 8
    assert small[1][1] == "a/same.parquet:2"
    assert len(loader.fetch_texts_by_id(str(tmp_path), {r[0] for r in small})) == 8
    with pytest.raises(ValueError, match="missing"):
        loader.fetch_texts_by_id(str(tmp_path), {"absent"})


@pytest.mark.parametrize("score", [None, [], [None], [float("nan")], float("inf")])
def test_bad_education_scores(score):
    assert loader.doc_mean(score) == 0


def test_exact_chunks_preserve_newlines():
    text = "def example():\n    return '中文 symbols ++'\n\n" * 30
    chunks = list(split_text(text, TOK.count, 32))
    assert "".join(chunks) == text
    assert all(TOK.count(c) <= 32 for c in chunks)
    assert all(TOK.count(c) <= 32 for c in chunk_doc(text, TOK.count, chunk_tokens=32, min_chunk_tokens=1))


@pytest.mark.parametrize("kind", ["answerable", "subtle", "absent", "mrcr"])
def test_corpus_gold_and_length(kind):
    markers = load_markers()
    if kind == "mrcr":
        task, build, verify = MRCRTask(POOL, markers), mrcr_row, check_mrcr
    else:
        task = IDKTask(POOL, markers, unanswerable_frac=0 if kind == "answerable" else 1,
                       subtle_frac=1 if kind == "subtle" else 0)
        build, verify = idk_row, check_idk
    for i in range(8):
        row = build(i, count=TOK.count, task=task, budget=4096, size=100, seed=5,
                    fmt="sft", max_seq_length=4096)
        assert verify(row, set(markers))[0]
        assert row["training_length"] <= 4096
        if kind != "mrcr":
            assert row["meta"]["kind"] == kind
    with pytest.raises(ValueError):
        build(0, count=TOK.count, task=task, budget=1, size=10, seed=5, fmt="sft", max_seq_length=1)


def test_conflicting_source_bindings_and_invalid_mrcr_k():
    poison = "The access code for the lemon is ABCD-12."
    task = IDKTask([poison] + POOL, load_markers())
    assert poison not in task.pool
    with pytest.raises(ValueError, match="two"):
        MRCRTask(POOL, load_markers(), k=3)


@pytest.mark.parametrize("flow", ["reconstruct", "permutation"])
def test_untie_budget(flow):
    result = compute_source(0, "source", "\n\n".join(POOL * 3), flows=[flow], samples_per_doc=2,
                            tasks={flow: UntieKnots(flow, 4096, 2)}, count=TOK.count, seed=3,
                            target_para_tokens=200, max_para_tokens=300, target_jitter=0.2,
                            min_units=2, fmt="sft", context_len=4096)
    for _, row in result["items"]:
        assert row is not None
        assert row["training_length"] <= 4096


@pytest.mark.parametrize("raw,expected", [
    ("Question: Q?\nAnswer: A.", "Question: Q?\nAnswer: A."),
    ("<think>private</think>answer", "answer"),
    ("seeded private</think>answer", "answer"),
    ("<think>unfinished", ""),
    ("reason</think><think>unfinished", ""),
])
def test_reasoning_extraction(raw, expected):
    assert extract_text({"choices": [{"message": {"content": raw}}]}) == expected


@pytest.mark.parametrize("row", [
    {"error": "failure"}, {"response": {"status_code": 500}},
    {"choices": [{"finish_reason": "length"}]},
    {"response": {"body": {"choices": [{"finish_reason": "content_filter"}]}}},
    {"choices": []},
])
def test_failed_responses_rejected(row):
    with pytest.raises(ValueError):
        validate_response(row)


def test_summary_preamble_not_content():
    assert clean_summary("Summary statistics describe the trial.") == "Summary statistics describe the trial."
    assert clean_summary("Summary: A trial.") == "A trial."


@pytest.fixture
def bundle(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    text = "\n\n".join(POOL)
    pq.write_table(pa.table({"id": ["doc"], "text": [text], "token_count": [TOK.count(text)]}), source / "docs.parquet")
    output = tmp_path / "prepared"
    sum_main(["prepare", "--input_dir", str(source), "--output_dir", str(output),
              "--min_doc_tokens", "1", "--tokenizer", "whitespace", "--max_seq_length", "8192"])
    request = json.loads((output / "requests.jsonl").read_text())
    response = {"custom_id": request["id"], "choices": [{"message": {"content": "A factual summary."}, "finish_reason": "stop"}]}
    responses = tmp_path / "responses.jsonl"
    write_jsonl(responses, [response])
    return output, responses, response


def test_prepare_api_schema_assemble_and_overflow(bundle):
    root, responses, _ = bundle
    args = ["assemble", "--requests_dir", str(root), "--responses", str(responses)]
    sum_main(args)
    row = json.loads((root / "summarize.jsonl").read_text())
    assert row["summary"] == "A factual summary."
    before = (root / "summarize.jsonl").read_bytes()
    with pytest.raises(ValueError, match="exceeds"):
        sum_main(args + ["--max_seq_length", "10"])
    assert (root / "summarize.jsonl").read_bytes() == before


@pytest.mark.parametrize("issue", ["missing", "duplicate", "unknown", "tamper"])
def test_strict_bundle_join(bundle, issue):
    root, responses, response = bundle
    if issue == "missing":
        write_jsonl(responses, [])
    elif issue == "duplicate":
        write_jsonl(responses, [response, response])
    elif issue == "unknown":
        write_jsonl(responses, [{**response, "custom_id": "wrong"}])
    else:
        with (root / "docs.jsonl").open("a") as stream:
            stream.write("\n")
    with pytest.raises(ValueError):
        sum_main(["assemble", "--requests_dir", str(root), "--responses", str(responses)])
    assert not (root / "summarize.jsonl").exists()


def test_atomic_publication(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"old": True}])
    before = path.read_bytes()
    def broken():
        yield {"new": True}
        raise RuntimeError("failure")
    with pytest.raises(RuntimeError):
        write_jsonl(path, broken())
    assert path.read_bytes() == before


def test_clustering_singleton_and_antipodal_vectors():
    idx, sim = topk_neighbours(np.array([[1., 0.]], dtype=np.float32), 3, "cpu")
    assert idx.shape == (1, 0)
    idx, sim = topk_neighbours(np.array([[1., 0.], [-1., 0.]], dtype=np.float32), 1, "cpu")
    assert idx.tolist() == [[1], [0]]
    with pytest.raises(ValueError):
        greedy_chain(idx, sim, np.array([100, 100]), 50, 0)


def test_hf_chat_template_budget():
    tokenizer = get_tokenizer("hf:Qwen/Qwen3-8B")
    rows, _ = generate_task(get_task("graph_descendants"), tokenizer, 3, 4096, 7, 0, "sft")
    for row in rows:
        actual = len(tokenizer.tok.apply_chat_template(row["messages"], tokenize=True,
                                                      add_generation_prompt=False,
                                                      return_dict=False))
        assert row["length_mode"] == "chat_template"
        assert actual == row["training_length"] <= 4096


def test_tokenizer_loading_failure_is_not_silent(monkeypatch):
    import ruler_pp.lengths as lengths
    def fail(name):
        raise RuntimeError("unavailable tokenizer")
    monkeypatch.setattr(lengths, "_HF", fail)
    with pytest.raises(RuntimeError, match="unavailable"):
        lengths.get_tokenizer("hf:missing")


def test_large_vocabulary_stays_distinct(monkeypatch):
    from ruler_pp import wordbank
    monkeypatch.setattr(wordbank, "_WORDS", ["one", "two"])
    words = wordbank.sample_words(100, random.Random(8))
    assert len(set(words)) == 100


def test_api_runner_payload_retry_and_completion(monkeypatch):
    import corpus.infer as infer
    from urllib.error import HTTPError
    from io import BytesIO
    calls = []
    def respond(request, timeout):
        calls.append(json.loads(request.data))
        if len(calls) == 1:
            raise HTTPError(request.full_url, 429, "rate limited", {}, None)
        return BytesIO(json.dumps({"choices": [{"message": {"content": "answer"},
                                               "finish_reason": "stop"}]}).encode())
    monkeypatch.setattr(infer, "urlopen", respond)
    monkeypatch.setattr(infer.time, "sleep", lambda _: None)
    result = infer.run_request({"id": "req", "messages": [{"role": "user", "content": "prompt"}]},
                              base_url="http://test/v1", model="model", api_key=None,
                              max_tokens=100, timeout=3, retries=1)
    assert result["custom_id"] == "req"
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 100


def test_multihop_plain_import_abstention_and_duplicate_pair(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    text = "\n\n".join(POOL)
    pq.write_table(pa.table({"id": ["source"], "text": [text], "token_count": [TOK.count(text)]}), corpus / "a.parquet")
    prepared = tmp_path / "prepared"
    mh_main(["prepare", "--input_dir", str(corpus), "--output_dir", str(prepared),
             "--min_doc_tokens", "1", "--tokenizer", "whitespace", "--requests_per_doc", "3",
             "--max_seq_length", "8192"])
    requests = [json.loads(line) for line in (prepared / "requests.jsonl").read_text().splitlines()]
    responses = tmp_path / "responses.jsonl"
    write_jsonl(responses, [{"id": r["id"], "text": (
        "Question: Where?\nAnswer: The laboratory." if i == 0 else
        "Question: WHERE?\nAnswer: In the laboratory building." if i == 1 else "cannot generate")}
                           for i, r in enumerate(requests)])
    mh_main(["assemble", "--requests_dir", str(prepared), "--responses", str(responses)])
    row = json.loads((prepared / "multihop.jsonl").read_text())
    assert row["n_pairs"] == 1
    report = json.loads((prepared / "multihop.report.json").read_text())
    assert report["abstentions"] == report["duplicate_pairs"] == 1


def test_empty_prepare_fails(tmp_path):
    pq.write_table(pa.table({"text": ["too short"], "token_count": [2]}), tmp_path / "input.parquet")
    with pytest.raises(ValueError, match="no requests"):
        sum_main(["prepare", "--input_dir", str(tmp_path), "--output_dir", str(tmp_path / "out")])
