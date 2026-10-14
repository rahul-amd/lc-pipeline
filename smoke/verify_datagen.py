"""Independently validate the small shell orchestration run."""
import argparse
from collections import Counter
import json
from pathlib import Path

from corpus.llm import load_jsonl, validate_response
from mrcr.mrcr import load_markers
from ruler_pp.lengths import get_tokenizer
from ruler_pp.output import training_length
from scripts.verify_golds import check
from scripts.verify_idk import check_row as check_idk
from scripts.verify_mrcr import check_row as check_mrcr


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root")
    args = parser.parse_args()
    root = Path(args.root)
    tokenizer = get_tokenizer("hf:Qwen/Qwen3-8B")
    counts, seen = Counter(), set()
    vocab = set(load_markers())
    for path in sorted(root.rglob("*.jsonl")):
        if path.name in {"requests.jsonl", "responses.jsonl", "docs.jsonl"}:
            continue
        rows = list(load_jsonl(path))
        assert rows, path
        for row in rows:
            assert row["id"] not in seen, path
            seen.add(row["id"])
            length, mode = training_length(row, tokenizer.count)
            assert length == row["training_length"] <= 8192, path
            assert mode == "chat_template", path
            task = row["task"]
            if task == "idk":
                assert check_idk(row, vocab)[0], path
            elif task == "mrcr":
                assert check_mrcr(row, vocab)[0], path
            elif task == "multihop":
                questions = [" ".join(pair["question"].casefold().split()) for pair in row["qa_pairs"]]
                assert len(questions) == len(set(questions)), path
            elif task not in {"untie_knots", "summarize"}:
                assert check(row), path
            counts[task] += 1
    assert len(seen) == 36, counts
    assert len(list(root.rglob(".complete"))) == 6
    assert len(list(root.rglob(".assembled"))) == 2
    responses = 0
    for path in root.rglob("responses.jsonl"):
        for row in load_jsonl(path):
            validate_response(row)
            responses += 1
    assert responses == 14
    result = {"status": "passed", "rows": len(seen), "by_task": counts,
              "api_responses": responses, "complete_markers": 6, "assembled_markers": 2}
    (root / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
