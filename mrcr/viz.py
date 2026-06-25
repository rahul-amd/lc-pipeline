"""Gradio viewer for MRCR marker-retrieval samples.

Reads an `mrcr` output dir (`mrcr.jsonl`), gives a slider over N random samples,
and renders the mrcr-specific structure the generic viewer hides:

  * the query: the two markers whose conjunction selects the target paragraph;
  * one table row per paragraph — its positive markers, its negated markers, a
    "query match" verdict (the unique TARGET vs. near-misses vs. noise), and a
    text preview — so you can eyeball that exactly one paragraph qualifies;
  * the random prefix and the assistant's answer (prefix + verbatim passage).

    python -m mrcr.viz --data_dir mrcr_out
    python -m mrcr.viz --data_dir mrcr_out --num 20 --share
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re

import gradio as gr

from .mrcr import load_markers, parse_markers

BLOCK_RE = re.compile(
    r"Paragraph\s+(\d+):\n<<<\n(.*?)\n>>>\n(.*?)(?=\n\nParagraph\s+\d+:\n<<<|\Z)",
    re.DOTALL,
)
_BULK = {"input", "text", "messages", "context", "instruction", "question", "answers"}


def load_samples(data_dir, num, seed):
    """Up to `num` randomly chosen mrcr rows from every jsonl in `data_dir`."""
    rows = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("task") == "mrcr":
                    rows.append(r)
    if not rows:
        return []
    return random.Random(seed).sample(rows, min(num, len(rows)))


def _answer(row):
    if "messages" in row:
        return row["messages"][-1]["content"]
    ans = row.get("answers") or [""]
    return ans[0]


def _verdict(pos, neg, query):
    """Human-readable query-match label for one paragraph."""
    a, b = query
    if {a, b} <= pos and not ({a, b} & neg):
        return "✓ TARGET (both)"
    bits = []
    for m in (a, b):
        if m in neg:
            bits.append(f"{m}✗")
        elif m in pos:
            bits.append(m)
    return ", ".join(bits) if bits else "—"


def _info_md(row):
    m = row.get("meta", {})
    q = m.get("query_markers", ["?", "?"])
    bits = [
        f"**id** `{row.get('id', '?')}`",
        f"**query** `{q[0]} AND {q[1]}`",
        f"**target** paragraph #{m.get('target_paragraph', '?')}",
        f"**paragraphs** {m.get('n_paragraphs', '?')}",
        f"**length** {row.get('length', '?')} tok",
        f"**format** `{row.get('format', '?')}`",
    ]
    return " &nbsp;|&nbsp; ".join(bits)


def _render(row, preview, vocab):
    m = row.get("meta", {})
    query = tuple(m.get("query_markers", ["", ""]))
    table = []
    for blk in BLOCK_RE.finditer(row.get("context", "")):
        num = int(blk.group(1))
        text = blk.group(2)
        desc = blk.group(3).strip()
        pos, neg = parse_markers(desc, vocab)
        table.append([
            num,
            ", ".join(sorted(pos)) or "—",
            ", ".join(sorted(neg)) or "—",
            _verdict(pos, neg, query),
            text[:preview].replace("\n", " "),
        ])

    q_md = (f"**Query:** find the paragraph marked as **both a {query[0]} and a "
            f"{query[1]}**. &nbsp; **Random prefix:** `{m.get('random_prefix', '?')}`")
    rest = {k: v for k, v in row.items() if k not in _BULK}
    return _info_md(row), q_md, table, gr.update(value=_answer(row)), rest


def build_app(data_dir, num, seed, preview):
    rows = load_samples(data_dir, num, seed)
    if not rows:
        raise SystemExit(f"no mrcr samples found in {data_dir!r}")
    vocab = set(load_markers())

    def render(i):
        return _render(rows[min(max(i, 1), len(rows)) - 1], preview, vocab)

    with gr.Blocks(title="mrcr viewer", fill_height=True) as demo:
        gr.Markdown(f"# mrcr marker-retrieval viewer\n`{os.path.abspath(data_dir)}` — "
                    f"{len(rows)} samples.")
        idx = gr.Slider(1, len(rows), step=1, value=1, label="Example")
        info = gr.Markdown()
        query = gr.Markdown()
        gr.Markdown("### Paragraphs (markers parsed back from the prompt) — "
                    "exactly one row should read **✓ TARGET**")
        paras = gr.Dataframe(headers=["#", "marked as", "not", "query match", "preview"],
                             datatype=["number", "str", "str", "str", "str"], wrap=True,
                             column_widths=["5%", "20%", "12%", "16%", "47%"])
        answer = gr.Textbox(label="Answer (random prefix + verbatim target passage)",
                            lines=16, max_lines=16)
        with gr.Accordion("Metadata", open=False):
            meta = gr.JSON()

        outs = [info, query, paras, answer, meta]
        idx.change(render, idx, outs)
        demo.load(lambda: render(1), None, outs)

    return demo


def main(argv=None):
    p = argparse.ArgumentParser("mrcr.viz", description="Gradio viewer for mrcr samples")
    p.add_argument("--data_dir", default="mrcr_out", help="dir containing mrcr.jsonl")
    p.add_argument("--num", type=int, default=10, help="random samples to load")
    p.add_argument("--seed", type=int, default=0, help="seed for sample selection")
    p.add_argument("--preview", type=int, default=300, help="chars of each paragraph shown")
    p.add_argument("--share", action="store_true", help="create a public shareable link")
    p.add_argument("--port", type=int, default=7862)
    args = p.parse_args(argv)

    app = build_app(args.data_dir, args.num, args.seed, args.preview)
    app.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
