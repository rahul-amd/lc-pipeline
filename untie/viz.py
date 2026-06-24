"""Gradio viewer for Untie-the-Knots samples — eyeball the shuffle and the answer.

Reads an `untie` output dir (`reconstruct.jsonl` / `permutation.jsonl`), gives a
flow dropdown and a slider over N random samples, and renders the untie-specific
structure that the generic ruler++ viewer hides:

  * the shuffled chunks in the order the model sees them, with their numeric
    labels, word counts, and a text preview (one table row per chunk);
  * the gold reading order (the label sequence that reconstructs the document);
  * the assistant's answer (the label list, or the reconstructed text).

    python -m untie.viz --data_dir untie_out
    python -m untie.viz --data_dir untie_out --num 20 --share
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re

import gradio as gr

CONTEXT_PREFIX = "Shuffled paragraphs:\n\n"
_BULK = {"input", "text", "messages", "context", "instruction", "question", "answers"}


def load_samples(data_dir, num, seed):
    """{flow -> up to `num` randomly chosen rows} from every jsonl in `data_dir`."""
    out = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.jsonl"))):
        flow = os.path.splitext(os.path.basename(path))[0]
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        if rows:
            out[flow] = random.Random(seed).sample(rows, min(num, len(rows)))
    return out


def _walk_chunks(body):
    """Sequential fallback: walk `[1] [2] ...` taking the first match each step.

    Cheap and correct unless a chunk body itself contains the next label preceded
    by a "\\n\\n" join (e.g. an in-document numbered reference list), which splits
    early. `parse_chunks` prefers the n_units-aware solver when meta is available.
    """
    chunks, expected, pos = [], 1, len("[1] ")
    while True:
        marker = f"\n\n[{expected + 1}] "
        i = body.find(marker, pos)
        if i == -1:
            chunks.append((expected, body[pos:].strip()))
            break
        chunks.append((expected, body[pos:i].strip()))
        expected += 1
        pos = i + len(marker)
    return chunks


def parse_chunks(context, n_units=None):
    """context -> [(label:int, text:str)] in the (shuffled) display order.

    The listing is `"\\n\\n".join(f"[{j}] {chunk}")`, so the real `[1] [2] ...`
    markers run strictly in order. The catch: a chunk body can itself contain a
    sequential numbered list (bibliographies, footnotes) whose `[n]` entries are
    also preceded by "\\n\\n" and so masquerade as chunk markers — a naive walk
    latches onto those and over-splits.

    When `n_units` (= k, the true chunk count from meta) is known, we pick the
    boundary chain that *maximises the minimum chunk size*. Real chunks are large
    and roughly equal; the spurious in-body markers are tiny and clustered, so the
    max-min objective rejects them. Falls back to the sequential walk otherwise.
    """
    body = context[len(CONTEXT_PREFIX):] if context.startswith(CONTEXT_PREFIX) else context
    if not body.startswith("[1] "):
        return []
    if not n_units or n_units < 1:
        return _walk_chunks(body)

    k = n_units
    # All marker occurrences: `[label] ` at the very start or after a "\n\n" join.
    cand = {j: [] for j in range(1, k + 1)}  # label -> [(boundary_pos, content_pos)]
    for m in re.finditer(r"(?:\A|\n\n)\[(\d+)\] ", body):
        lab = int(m.group(1))
        if 1 <= lab <= k:
            cand[lab].append((m.start(), m.end()))
    if any(not cand[j] for j in range(1, k + 1)):
        return _walk_chunks(body)

    # DP over labels 1..k: choose one occurrence per label with strictly
    # increasing boundary, maximising the smallest chunk length along the chain.
    L = len(body)
    INF = float("inf")
    # state per label j: list of (min_chunk_so_far, boundary, content, prev_idx)
    prev_states = [(INF, b, c, -1) for (b, c) in cand[1]]
    back = [list(prev_states)]
    for j in range(2, k + 1):
        states = []
        for (b, c) in cand[j]:
            best = None  # (min_so_far, prev_idx)
            for pi, (pmin, pb, pc, _) in enumerate(prev_states):
                if pb < b:
                    chunk_len = b - pc  # length of chunk j-1 (prev content -> this boundary)
                    cand_min = min(pmin, chunk_len)
                    if best is None or cand_min > best[0]:
                        best = (cand_min, pi)
            if best is not None:
                states.append((best[0], b, c, best[1]))
        if not states:
            return _walk_chunks(body)
        back.append(states)
        prev_states = states

    # Close the final chunk (label k content -> end) and pick the best chain end.
    best_i, best_val = -1, -INF
    for i, (smin, b, c, _) in enumerate(prev_states):
        val = min(smin, L - c)
        if val > best_val:
            best_val, best_i = val, i

    # Backtrack to recover the chosen boundary/content positions per label.
    picks = [None] * k
    i = best_i
    for j in range(k, 0, -1):
        smin, b, c, pi = back[j - 1][i]
        picks[j - 1] = (b, c)
        i = pi

    chunks = []
    for j in range(k):
        c = picks[j][1]
        end = picks[j + 1][0] if j + 1 < k else L
        chunks.append((j + 1, body[c:end].strip()))
    return chunks


def _answer(row):
    if "messages" in row:
        return row["messages"][-1]["content"]
    if row.get("answer_type") == "list":
        return ", ".join(row.get("answers", []))
    ans = row.get("answers") or [""]
    return ans[0]


def _info_md(row):
    m = row.get("meta", {})
    bits = [
        f"**id** `{row.get('id', '?')}`",
        f"**flow** `{row.get('flow', m.get('flow', '?'))}`",
        f"**chunks** {m.get('n_units', '?')}",
        f"**length** {row.get('length', '?')} tok",
        f"**format** `{row.get('format', '?')}`",
        f"**source** `{row.get('source', '?')}`",
    ]
    return " &nbsp;|&nbsp; ".join(bits)


def _render(row, preview):
    chunks = parse_chunks(row.get("context", ""), row.get("meta", {}).get("n_units"))
    table = [[lbl, len(txt.split()), txt[:preview].replace("\n", " ")] for lbl, txt in chunks]

    gold = row.get("gold")
    if isinstance(gold, list):
        order_md = ("**Gold reading order** (read the shuffled chunks in this label "
                    "sequence to recover the original):\n\n`"
                    + " → ".join(str(g) for g in gold) + "`")
    else:
        order_md = ""

    flow = row.get("flow", row.get("meta", {}).get("flow", ""))
    ans = _answer(row)
    ans_label = ("Answer — label order" if flow == "permutation"
                 else "Answer — reconstructed document (original order)")
    return _info_md(row), table, order_md, gr.update(value=ans, label=ans_label), \
        {k: v for k, v in row.items() if k not in _BULK}


def build_app(data_dir, num, seed, preview):
    cache = load_samples(data_dir, num, seed)
    flows = list(cache)
    if not flows:
        raise SystemExit(f"no .jsonl files found in {data_dir!r}")

    def render(flow, i):
        rows = cache.get(flow, [])
        if not rows:
            return "no samples", [], "", gr.update(value=""), {}
        return _render(rows[min(max(i, 1), len(rows)) - 1], preview)

    def on_flow(flow):
        n = len(cache.get(flow, []))
        return (gr.update(minimum=1, maximum=max(n, 1), value=1), *render(flow, 1))

    with gr.Blocks(title="untie-the-knots viewer", fill_height=True) as demo:
        gr.Markdown(f"# untie-the-knots viewer\n`{os.path.abspath(data_dir)}` — "
                    f"flows: {', '.join(flows)}; up to {num} samples each.")
        with gr.Row():
            flow_dd = gr.Dropdown(choices=flows, value=flows[0], label="Flow", scale=2)
            idx = gr.Slider(1, max(len(cache[flows[0]]), 1), step=1, value=1, label="Example", scale=3)
        info = gr.Markdown()
        gr.Markdown("### Shuffled chunks (in the order the model sees them)")
        chunks = gr.Dataframe(headers=["label", "words", "preview"],
                              datatype=["number", "number", "str"],
                              wrap=True, column_widths=["8%", "10%", "82%"])
        order = gr.Markdown()
        answer = gr.Textbox(label="Answer", lines=16, max_lines=16)
        with gr.Accordion("Metadata", open=False):
            meta = gr.JSON()

        outs = [info, chunks, order, answer, meta]
        flow_dd.change(on_flow, flow_dd, [idx, *outs])
        idx.change(render, [flow_dd, idx], outs)
        demo.load(on_flow, flow_dd, [idx, *outs])

    return demo


def main(argv=None):
    p = argparse.ArgumentParser("untie.viz", description="Gradio viewer for untie-the-knots samples")
    p.add_argument("--data_dir", default="untie_out", help="dir of <flow>.jsonl files")
    p.add_argument("--num", type=int, default=10, help="random samples per flow")
    p.add_argument("--seed", type=int, default=0, help="seed for sample selection")
    p.add_argument("--preview", type=int, default=300, help="chars of each chunk shown in the table")
    p.add_argument("--share", action="store_true", help="create a public shareable link")
    p.add_argument("--port", type=int, default=7861)
    args = p.parse_args(argv)

    app = build_app(args.data_dir, args.num, args.seed, args.preview)
    app.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
