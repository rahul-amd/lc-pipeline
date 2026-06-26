"""Gradio viewer for IDK abstention-MCQ samples.

Reads an `idk` output dir (`idk.jsonl`), gives a slider over N random samples, and
renders the idk-specific structure:

  * the question kind (answerable / subtle / absent) and the gold letter;
  * one table row per paragraph — whether it carries a code binding, the binding's
    (attribute, marker, code), and a text preview;
  * the question, its A/B/C/D options (the gold option flagged), and the answer.

    python -m idk.viz --data_dir idk_out
    python -m idk.viz --data_dir idk_out --num 20 --share
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re

import gradio as gr

from idk.idk import BINDING_RE

BLOCK_RE = re.compile(r"Paragraph\s+(\d+):\n(.*?)(?=\n\nParagraph\s+\d+:\n|\Z)", re.DOTALL)
_BULK = {"input", "text", "messages", "context", "instruction", "question", "answers"}


def load_samples(data_dir, num, seed):
    rows = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("task") == "idk":
                    rows.append(r)
    if not rows:
        return []
    return random.Random(seed).sample(rows, min(num, len(rows)))


def _answer(row):
    if "messages" in row:
        return row["messages"][-1]["content"]
    ans = row.get("answers") or [""]
    return ans[0]


def _info_md(row):
    m = row.get("meta", {})
    bits = [
        f"**id** `{row.get('id', '?')}`",
        f"**kind** `{m.get('kind', '?')}`",
        f"**answerable** `{m.get('answerable', '?')}`",
        f"**asked** `{m.get('asked_attribute', '?')} code for the {m.get('asked_marker', '?')}`",
        f"**gold** `{row.get('gold', '?')}`",
        f"**bindings** {m.get('n_bindings', '?')}/{m.get('n_paragraphs', '?')}",
        f"**length** {row.get('length', '?')} tok",
        f"**format** `{row.get('format', '?')}`",
    ]
    return " &nbsp;|&nbsp; ".join(bits)


def _render(row, preview):
    m = row.get("meta", {})
    asked_marker = m.get("asked_marker")
    table = []
    for blk in BLOCK_RE.finditer(row.get("context", "")):
        num = int(blk.group(1))
        body = blk.group(2)
        bm = BINDING_RE.search(body)
        if bm:
            attr, marker, code = bm.group(1), bm.group(2), bm.group(3)
            flag = "  <-- asked marker" if marker == asked_marker else ""
            binding = f"{attr} / {marker} = {code}{flag}"
            text = body[:bm.start()].strip()
        else:
            binding = "—"
            text = body.strip()
        table.append([num, binding, text[:preview].replace("\n", " ")])

    gold = row.get("gold")
    opts = m.get("options", {})
    opt_md = "\n".join(
        f"- **({L}) {t}**  ← gold" if L == gold else f"- ({L}) {t}"
        for L, t in sorted(opts.items())
    )
    q = row.get("question", "").split("\n\n")[0]  # the "Question: ..." line
    q_md = f"**{q}**\n\n{opt_md}"
    rest = {k: v for k, v in row.items() if k not in _BULK}
    return _info_md(row), q_md, table, gr.update(value=_answer(row)), rest


def build_app(data_dir, num, seed, preview):
    rows = load_samples(data_dir, num, seed)
    if not rows:
        raise SystemExit(f"no idk samples found in {data_dir!r}")

    def render(i):
        return _render(rows[min(max(i, 1), len(rows)) - 1], preview)

    with gr.Blocks(title="idk viewer", fill_height=True) as demo:
        gr.Markdown(f"# idk abstention-MCQ viewer\n`{os.path.abspath(data_dir)}` — "
                    f"{len(rows)} samples.")
        idx = gr.Slider(1, len(rows), step=1, value=1, label="Example")
        info = gr.Markdown()
        question = gr.Markdown()
        gr.Markdown("### Paragraphs (code bindings parsed back from the prompt)")
        paras = gr.Dataframe(headers=["#", "binding (attribute / marker = code)", "preview"],
                             datatype=["number", "str", "str"], wrap=True,
                             column_widths=["5%", "35%", "60%"])
        answer = gr.Textbox(label="Answer (option letter)", lines=1, max_lines=1)
        with gr.Accordion("Metadata", open=False):
            meta = gr.JSON()

        outs = [info, question, paras, answer, meta]
        idx.change(render, idx, outs)
        demo.load(lambda: render(1), None, outs)

    return demo


def main(argv=None):
    p = argparse.ArgumentParser("idk.viz", description="Gradio viewer for idk samples")
    p.add_argument("--data_dir", default="idk_out", help="dir containing idk.jsonl")
    p.add_argument("--num", type=int, default=10, help="random samples to load")
    p.add_argument("--seed", type=int, default=0, help="seed for sample selection")
    p.add_argument("--preview", type=int, default=300, help="chars of each paragraph shown")
    p.add_argument("--share", action="store_true", help="create a public shareable link")
    p.add_argument("--port", type=int, default=7863)
    args = p.parse_args(argv)

    app = build_app(args.data_dir, args.num, args.seed, args.preview)
    app.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
