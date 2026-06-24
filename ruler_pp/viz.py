"""Gradio viewer for generated samples — eyeball one example at a time.

Reads a directory of ``<task>.jsonl`` files, builds a task dropdown, and shows N
random samples per task. SFT rows render as a proper user/assistant chat; PT
rows render as flat text. Launch with ``--share`` for a public link.

    python -m ruler_pp.viz --data_dir data
    python -m ruler_pp.viz --data_dir samples --num 10 --share
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re

import gradio as gr

# Big fields shown in their own panels, not in the metadata blob.
_BULK_FIELDS = {"input", "text", "messages", "context", "instruction", "question"}

_EDGE_RE = re.compile(r"([0-9a-f]+) -> ([0-9a-f]+)")


def _graph_figure(row):
    """For `graph_*` tasks: draw the edge list, highlighting the operation's focus
    node(s) (red) and the gold-answer nodes (green). Returns a matplotlib Figure,
    or None if this isn't a graph row / the optional draw deps are missing."""
    if not row.get("task", "").startswith("graph_"):
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
    except Exception:
        return None  # networkx/matplotlib not installed -> text view still works

    edges = _EDGE_RE.findall(row.get("context", ""))
    if not edges:
        return None
    g = nx.DiGraph()
    g.add_edges_from(edges)

    meta = row.get("meta", {})
    focus = {meta[k] for k in ("start", "node", "src", "dst") if isinstance(meta.get(k), str)}
    gold = row.get("gold")
    gold_nodes = set(gold) if row.get("answer_type") == "list" and isinstance(gold, list) else set()

    colors, sizes = [], []
    for n in g.nodes():
        if n in focus:
            colors.append("#d62728"); sizes.append(640)        # red: operation node(s)
        elif n in gold_nodes:
            colors.append("#2ca02c"); sizes.append(460)        # green: gold answer
        else:
            colors.append("#c7d0d9"); sizes.append(240)

    pos = nx.spring_layout(g, seed=0)
    fig, ax = plt.subplots(figsize=(7, 6))
    nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.35, edge_color="#888",
                           arrowsize=8, width=0.7, node_size=sizes)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colors, node_size=sizes,
                           linewidths=0.4, edgecolors="#333")
    nx.draw_networkx_labels(g, pos, labels={n: n[:4] for n in g.nodes()},
                            ax=ax, font_size=6)
    ax.set_title(f"{row.get('task')}  —  red: operation node(s),  green: gold answer "
                 f"({g.number_of_nodes()} nodes, {g.number_of_edges()} edges)", fontsize=9)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def load_samples(data_dir: str, num: int, seed: int) -> dict[str, list[dict]]:
    """{task -> up to `num` randomly chosen rows} from every jsonl in `data_dir`."""
    out: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.jsonl"))):
        task = os.path.splitext(os.path.basename(path))[0]
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        if rows:
            rng = random.Random(seed)
            out[task] = rng.sample(rows, min(num, len(rows)))
    return out


def _fmt_gold(gold) -> str:
    return json.dumps(gold, ensure_ascii=False)


def _info_md(row: dict) -> str:
    fmt = row.get("format") or ("sft" if "messages" in row else "pt")
    bits = [
        f"**id** `{row.get('id', '?')}`",
        f"**format** `{fmt}`",
        f"**length** {row.get('length', '?')} tok",
        f"**answer_type** `{row.get('answer_type', '?')}`",
    ]
    return " &nbsp;|&nbsp; ".join(bits) + f"\n\n**Expected answer:** `{_fmt_gold(row.get('gold'))}`"


def _render(row: dict):
    """Return updates for (info, chat, text, graph, meta)."""
    fmt = row.get("format") or ("sft" if "messages" in row else "pt")
    meta = {k: v for k, v in row.items() if k not in _BULK_FIELDS}

    if fmt == "sft" and "messages" in row:
        chat = gr.update(value=row["messages"], visible=True)
        text = gr.update(visible=False)
    else:
        body = row.get("text") or (row.get("input", "") + "\n" + row.get("answer_prefix", ""))
        chat = gr.update(visible=False)
        text = gr.update(value=body, visible=True)

    fig = _graph_figure(row)
    graph = gr.update(value=fig, visible=True) if fig is not None else gr.update(visible=False)
    return _info_md(row), chat, text, graph, meta


def build_app(data_dir: str, num: int, seed: int) -> gr.Blocks:
    cache = load_samples(data_dir, num, seed)
    tasks = list(cache)
    if not tasks:
        raise SystemExit(f"no .jsonl files found in {data_dir!r}")

    def render(task: str, i: int):
        rows = cache.get(task, [])
        if not rows:
            return ("no samples", gr.update(visible=False),
                    gr.update(value="", visible=True), gr.update(visible=False), {})
        return _render(rows[min(max(i, 1), len(rows)) - 1])

    def on_task(task: str):
        n = len(cache.get(task, []))
        return (gr.update(minimum=1, maximum=max(n, 1), value=1), *render(task, 1))

    with gr.Blocks(title="ruler++ sample viewer", fill_height=True) as demo:
        gr.Markdown(f"# ruler++ sample viewer\n`{os.path.abspath(data_dir)}` — {len(tasks)} tasks, up to {num} samples each.")
        with gr.Row():
            task_dd = gr.Dropdown(choices=tasks, value=tasks[0], label="Task", scale=2)
            idx = gr.Slider(1, max(len(cache[tasks[0]]), 1), step=1, value=1, label="Example", scale=3)
        info = gr.Markdown()
        chat = gr.Chatbot(label="Conversation (SFT)", height=460, visible=False)
        text = gr.Textbox(label="Sample (PT)", lines=22, max_lines=22, visible=False)
        graph = gr.Plot(label="Graph (graph_* tasks)", visible=False)
        with gr.Accordion("Metadata", open=False):
            meta = gr.JSON()

        outs = [info, chat, text, graph, meta]
        task_dd.change(on_task, task_dd, [idx, *outs])
        idx.change(render, [task_dd, idx], outs)
        demo.load(on_task, task_dd, [idx, *outs])

    return demo


def main(argv=None):
    p = argparse.ArgumentParser("ruler_pp.viz", description="Gradio viewer for generated samples")
    p.add_argument("--data_dir", default="data", help="directory of <task>.jsonl files")
    p.add_argument("--num", type=int, default=10, help="random samples per task")
    p.add_argument("--seed", type=int, default=0, help="seed for sample selection")
    p.add_argument("--share", action="store_true", help="create a public shareable link")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args(argv)

    app = build_app(args.data_dir, args.num, args.seed)
    app.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
