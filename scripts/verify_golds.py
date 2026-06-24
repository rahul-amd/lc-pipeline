"""Independent verifier: re-parse each sample's context and recompute the gold.

This does NOT trust the generator's bookkeeping — it parses the rendered text
the model would see and checks the stored gold matches. Run after generation.
"""

import glob
import json
import re
import statistics
import sys
from collections import Counter, defaultdict

WORD_RE = re.compile(r"\d+\.\s+([a-z]+)")           # "12. word"
NUM_RE = re.compile(r"\d+\.\s+(\d+)")               # "12. 345"
REC_RE = re.compile(r"\d+\.\s+([a-z]+) \| category: (\w+) \| value: (\d+)")
VAR_SET = re.compile(r"VAR ([A-Z]+) = (\d+)")
VAR_COPY = re.compile(r"VAR ([A-Z]+) = VAR ([A-Z]+)\s*$", re.M)
VAR_ARITH = re.compile(r"VAR ([A-Z]+) = VAR ([A-Z]+) ([+\-*]) (\d+)")
EDGE_RE = re.compile(r"([0-9a-f]+) -> ([0-9a-f]+)")


def words(ctx):
    return WORD_RE.findall(ctx)


def records(ctx):
    return [(n, c, int(v)) for n, c, v in REC_RE.findall(ctx)]


def parse_graph(ctx):
    """context edge list -> (nodes_in_order, adj, radj)."""
    adj, radj, nodes, seen = {}, {}, [], set()
    def touch(x):
        if x not in seen:
            seen.add(x); nodes.append(x); adj[x] = []; radj[x] = []
    for u, v in EDGE_RE.findall(ctx):
        touch(u); touch(v)
        adj[u].append(v); radj[v].append(u)
    return nodes, adj, radj


def g_layers(adj, start):
    layers, seen, frontier = [[start]], {start}, [start]
    while frontier:
        nxt = []
        for u in frontier:
            for w in adj.get(u, []):
                if w not in seen:
                    seen.add(w); nxt.append(w)
        if nxt:
            layers.append(nxt)
        frontier = nxt
    return layers


def g_reach(adj, start):
    seen, stack = {start}, [start]
    while stack:
        u = stack.pop()
        for w in adj.get(u, []):
            if w not in seen:
                seen.add(w); stack.append(w)
    seen.discard(start)
    return seen


def g_dist(adj, src, dst):
    if src == dst:
        return 0
    seen, frontier, d = {src}, [src], 0
    while frontier:
        d += 1; nxt = []
        for u in frontier:
            for w in adj.get(u, []):
                if w == dst:
                    return d
                if w not in seen:
                    seen.add(w); nxt.append(w)
        frontier = nxt
    return None


def g_has_cycle(nodes, adj):
    indeg = {v: 0 for v in nodes}
    for u in nodes:
        for w in adj[u]:
            indeg[w] += 1
    queue = [v for v in nodes if indeg[v] == 0]
    visited = 0
    while queue:
        u = queue.pop(); visited += 1
        for w in adj[u]:
            indeg[w] -= 1
            if indeg[w] == 0:
                queue.append(w)
    return visited != len(nodes)


def g_weak_component(adj, radj, start):
    seen, stack = {start}, [start]
    while stack:
        u = stack.pop()
        for w in adj.get(u, []) + radj.get(u, []):
            if w not in seen:
                seen.add(w); stack.append(w)
    return seen


def check(row):
    t, ctx, gold, meta = row["task"], row["context"], row["gold"], row["meta"]
    q = row["question"]

    if t == "common_words":
        cnt = Counter(words(ctx))
        top = {w for w, _ in cnt.most_common(meta["num_cw"])}
        return set(gold) == top
    if t == "freq_words":
        toks = [w for w in ctx.split() if w != "..."]
        top3 = [w for w, _ in Counter(toks).most_common(3)]
        return top3 == gold
    if t == "variable_tracking":
        val = meta["value"]
        assigned = {v for v, x in VAR_SET.findall(ctx) if x == val}
        # follow copies to closure
        copies = VAR_COPY.findall(ctx)
        changed = True
        while changed:
            changed = False
            for dst, src in copies:
                if src in assigned and dst not in assigned:
                    assigned.add(dst); changed = True
        return set(gold) == assigned
    if t == "count_occurrences":
        return words(ctx).count(meta["target"]) == gold
    if t == "count_distinct":
        return len(set(words(ctx))) == gold
    if t == "count_predicate":
        recs = records(ctx)
        p = meta["predicate"]
        if p == "value >":
            g = sum(v > meta["threshold"] for _, _, v in recs)
        elif p == "value <":
            g = sum(v < meta["threshold"] for _, _, v in recs)
        elif p == "category ==":
            g = sum(c == meta["category"] for _, c, _ in recs)
        else:
            g = sum(meta["low"] <= v <= meta["high"] for _, _, v in recs)
        return g == gold
    if t == "tally_by_category":
        c = Counter(c for _, c, _ in records(ctx))
        return {k: c[k] for k in c} == {k: int(v) for k, v in gold.items()}
    if t == "numeric_agg":
        nums = [int(x) for x in NUM_RE.findall(ctx)]
        agg = meta["agg"]
        g = {"sum": sum, "min": min, "max": max}.get(agg)
        if g:
            return g(nums) == gold
        if agg == "mean":
            return round(sum(nums) / len(nums), 2) == gold
        return statistics.median(nums) == gold
    if t == "group_by":
        tot = defaultdict(int)
        for _, c, v in records(ctx):
            tot[c] += v
        return {k: tot[k] for k in tot} == {k: int(v) for k, v in gold.items()}
    if t == "top_k":
        cnt = Counter(words(ctx))
        return [w for w, _ in cnt.most_common(meta["k"])] == gold
    if t == "set_ops":
        ma = re.search(r"Set A: (.+?)\n", ctx)
        mb = re.search(r"Set B: (.+)$", ctx, re.S)
        a = set(ma.group(1).replace(",", " ").split())
        b = set(mb.group(1).replace(",", " ").split())
        op = meta["op"]
        if op == "intersection":
            return sorted(a & b) == gold
        if op == "union_size":
            return len(a | b) == gold
        return sorted(a - b) == gold
    if t == "variable_arithmetic":
        vals = {}
        for v, x in VAR_SET.findall(ctx):
            vals[v] = int(x)
        for dst, src, op, c in VAR_ARITH.findall(ctx):
            c = int(c)
            vals[dst] = vals[src] + c if op == "+" else vals[src] - c if op == "-" else vals[src] * c
        return vals[meta["target"]] == gold
    if t.startswith("graph_"):
        nodes, adj, radj = parse_graph(ctx)
        op = meta["op"]
        if op == "bfs":
            layers = g_layers(adj, meta["start"])
            d = meta["depth"]
            exp = layers[d] if d < len(layers) else []
            return sorted(exp) == gold
        if op == "parents":
            return sorted(radj[meta["node"]]) == gold
        if op == "children":
            return sorted(adj[meta["node"]]) == gold
        if op == "descendants":
            return sorted(g_reach(adj, meta["node"])) == gold
        if op == "ancestors":
            return sorted(g_reach(radj, meta["node"])) == gold
        if op == "shortest_path":
            return g_dist(adj, meta["src"], meta["dst"]) == gold
        if op == "reachable":
            hit = meta["dst"] in g_reach(adj, meta["src"]) or meta["dst"] == meta["src"]
            return ("Yes" if hit else "No") == gold
        if op == "cycle":
            return ("Yes" if g_has_cycle(nodes, adj) else "No") == gold
        if op == "component_size":
            return len(g_weak_component(adj, radj, meta["node"])) == gold
        if op == "max_degree":
            deg = {v: len(adj[v]) + len(radj[v]) for v in nodes}
            top = max(deg.values())
            return sorted(v for v in nodes if deg[v] == top) == gold
        raise SystemExit(f"no verifier for graph op {op}")
    raise SystemExit(f"no verifier for task {t}")


def main(data_dir="data"):
    total = ok = 0
    by_task = defaultdict(lambda: [0, 0])
    for path in sorted(glob.glob(f"{data_dir}/*.jsonl")):
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            passed = check(row)
            total += 1; ok += passed
            by_task[row["task"]][0] += passed
            by_task[row["task"]][1] += 1
    for t in sorted(by_task):
        p, n = by_task[t]
        mark = "OK " if p == n else "FAIL"
        print(f"  [{mark}] {t:22s} {p}/{n}")
    print(f"\nTOTAL: {ok}/{total} samples verified")
    sys.exit(0 if ok == total else 1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
