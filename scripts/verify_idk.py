"""Independent re-checker for idk golds.

Does not trust the generator's bookkeeping: it re-parses the rendered context for
code bindings, re-parses the question's asked (attribute, marker) and its A/B/C/D
options, recomputes the correct letter purely from the rendered text, and confirms
it matches the recorded gold.

Rule recomputed: the question is answerable iff the asked (attribute, marker) pair
is bound in the context. If answerable, the correct option is the one whose text is
that pair's code; otherwise the correct option is "I don't know". This naturally
handles the subtle case (marker bound under a different attribute -> not the asked
pair -> abstain).

    python scripts/verify_idk.py idk_out
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from idk.idk import BINDING_RE, IDK, OPTION_RE, QUESTION_RE  # noqa: E402
from mrcr.mrcr import load_markers  # noqa: E402


def parse_bindings(context, vocab):
    """{(attribute, marker): code} for every binding sentence in the context."""
    out = {}
    for attr, marker, code in BINDING_RE.findall(context):
        if marker in vocab:
            out[(attr, marker)] = code
    return out


def check_row(row, vocab):
    bindings = parse_bindings(row["context"], vocab)
    qm = QUESTION_RE.search(row["question"])
    if not qm:
        return False, "could not parse the question"
    q_attr, q_marker = qm.group(1), qm.group(2)
    options = {letter: text.strip() for letter, text in OPTION_RE.findall(row["question"])}
    if IDK not in options.values():
        return False, "no 'I don't know' option present"

    key = (q_attr, q_marker)
    if key in bindings:
        target = bindings[key]
        matches = [L for L, t in options.items() if t == target]
        if len(matches) != 1:
            return False, f"{len(matches)} options match the bound code (want 1)"
        want = matches[0]
    else:
        want = [L for L, t in options.items() if t == IDK][0]

    if want != row["gold"]:
        return False, f"recomputed gold {want!r} != recorded {row['gold']!r}"

    # cross-check the recorded 'answerable' flag against the parsed truth
    if row["meta"].get("answerable") != (key in bindings):
        return False, "meta.answerable disagrees with parsed bindings"
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
                if row.get("task") != "idk":
                    continue
                n += 1
                good, why = check_row(row, vocab)
                if good:
                    ok += 1
                else:
                    print(f"  FAIL {row.get('id')}: {why}")
        if n:
            print(f"{os.path.basename(path):24s} {ok}/{n} verified")
            grand_ok += ok
            grand_n += n
    print(f"\nTOTAL: {grand_ok}/{grand_n} samples verified")
    return 0 if grand_ok == grand_n else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "idk_out"))
