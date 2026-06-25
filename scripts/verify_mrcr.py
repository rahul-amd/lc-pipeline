"""Independent re-checker for mrcr golds.

Does not trust the generator's bookkeeping: it re-parses each rendered `context`,
extracts every paragraph's fenced text plus its marker sentences, recomputes the
unique paragraph satisfying the query conjunction (positives superset of the
query, none of the query markers negated), and confirms it equals the recorded
gold. Also checks the required random prefix is present in the trained answer.

    python scripts/verify_mrcr.py mrcr_out
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

# import the shared marker vocab + parser from the mrcr package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mrcr.mrcr import load_markers, parse_markers  # noqa: E402

BLOCK_RE = re.compile(
    r"Paragraph\s+(\d+):\n<<<\n(.*?)\n>>>\n(.*?)(?=\n\nParagraph\s+\d+:\n<<<|\Z)",
    re.DOTALL,
)


def check_row(row, vocab):
    meta = row["meta"]
    a, b = meta["query_markers"]
    query = {a, b}
    gold = row["gold"]

    matches = []
    for m in BLOCK_RE.finditer(row["context"]):
        text = m.group(2)
        desc = m.group(3).strip()
        pos, neg = parse_markers(desc, vocab)
        if query <= pos and not (query & neg):
            matches.append(text)

    if len(matches) != 1:
        return False, f"{len(matches)} paragraphs match query {sorted(query)} (want 1)"
    if matches[0] != gold:
        return False, "matched paragraph text != gold"

    # the trained answer must echo the random prefix then the gold passage
    prefix = meta["random_prefix"]
    ans = row["answers"][0]
    if not ans.startswith(prefix):
        return False, "answer does not start with random prefix"
    if gold not in ans:
        return False, "answer does not contain gold passage"
    return True, ""


def main(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.jsonl"), recursive=True))
    if not files:
        raise SystemExit(f"no jsonl files under {data_dir!r}")
    vocab = set(load_markers())
    grand_ok = grand_n = 0
    for path in files:
        ok = n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("task") != "mrcr":
                    continue
                n += 1
                good, why = check_row(row, vocab)
                if good:
                    ok += 1
                elif ok + 1 == n or n <= 5 or True:
                    print(f"  FAIL {row.get('id')}: {why}")
        if n:
            print(f"{os.path.basename(path):24s} {ok}/{n} verified")
            grand_ok += ok
            grand_n += n
    print(f"\nTOTAL: {grand_ok}/{grand_n} samples verified")
    return 0 if grand_ok == grand_n else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "mrcr_out"))
