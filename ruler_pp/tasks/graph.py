"""Graph-reasoning tasks over a shuffled directed-edge list (GraphWalks-style).

A single synthetic graph backs a whole family of operations. The context is the
graph's edges in the GraphWalks surface form (``A -> B``, one per line, shuffled),
and each task asks a different question over it. Golds are computed deterministically
from the graph, so this slots straight into the `Task` family: `size` scales the
number of nodes (hence edges, hence tokens), the driver binary-searches it, and
`pt`/`sft` shaping is inherited unchanged.

Answer surface forms (GraphWalks-style node lists, but no "Final Answer:" cue):
  * node sets   -> ``["abc", "def"]``  (JSON list, ``[]`` if empty)
  * counts      -> ``5``
  * yes/no      -> ``Yes`` / ``No``

Determinism note: node IDs and every choice are drawn from the task `rng` via
lists + a seen-set — never by iterating a Python `set` of strings (whose order is
hash-randomised across runs).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..base import Sample, Task

GRAPH_PREAMBLE = (
    "You are given a directed graph as a list of edges, one per line in the form "
    '"A -> B", meaning there is a directed edge from node A to node B. Node names '
    "are random hexadecimal strings, the edges are listed in arbitrary order, and "
    "every node has degree at least 1.\n\n"
    "Definitions: the children (direct successors) of a node X are the nodes Y with "
    "an edge X -> Y; the parents (direct predecessors) of X are the nodes Y with an "
    "edge Y -> X; a BFS from X reaches a node at depth d when its shortest directed "
    "path from X uses exactly d edges.\n\n"
    "Perform the requested operation and respond with only the answer:\n"
    '  ["node1", "node2", ...]   for a set of nodes (use [] if empty)\n'
    "  <integer>                 for a numeric answer\n"
    "  Yes   (or No)             for a yes/no answer"
)


# --------------------------------------------------------------------------- #
# graph construction + algorithms
# --------------------------------------------------------------------------- #
@dataclass
class GraphData:
    nodes: list[str]                 # node IDs in creation order
    adj: dict[str, list[str]]        # out-neighbours (children)
    radj: dict[str, list[str]]       # in-neighbours (parents)
    edges: list[tuple[str, str]]     # directed edges (u -> v)


def _node_ids(n, rng):
    ids, seen = [], set()
    while len(ids) < n:
        x = format(rng.getrandbits(40), "010x")
        if x not in seen:
            seen.add(x)
            ids.append(x)
    return ids


def gen_graph(n, rng, avg_deg=2.5, acyclic=False):
    """A random directed graph on `n` nodes, every node of degree >= 1.

    If `acyclic`, every edge is oriented from the lower- to the higher-indexed
    endpoint, which makes the graph a DAG by construction.
    """
    nodes = _node_ids(n, rng)
    idx = {v: i for i, v in enumerate(nodes)}
    edge_set, edges = set(), []

    def add(u, v):
        if u == v:
            return
        if acyclic and idx[u] > idx[v]:
            u, v = v, u
        if (u, v) in edge_set:
            return
        edge_set.add((u, v))
        edges.append((u, v))

    # guarantee degree >= 1: wire every node to one distinct random partner.
    for i, u in enumerate(nodes):
        j = rng.randrange(n - 1)
        if j >= i:
            j += 1
        add(u, nodes[j])

    # top up to the target edge count with random edges.
    target = int(n * avg_deg)
    attempts = 0
    while len(edges) < target and attempts < target * 5 + 10:
        add(nodes[rng.randrange(n)], nodes[rng.randrange(n)])
        attempts += 1

    adj = {v: [] for v in nodes}
    radj = {v: [] for v in nodes}
    for u, v in edges:
        adj[u].append(v)
        radj[v].append(u)
    return GraphData(nodes, adj, radj, edges)


def render_edges(edges, rng):
    lines = [f"{u} -> {v}" for u, v in edges]
    rng.shuffle(lines)
    return "\n".join(lines)


def bfs_layers(adj, start):
    """`layers[d]` = nodes whose shortest path from `start` has exactly d edges."""
    layers = [[start]]
    seen = {start}
    frontier = [start]
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
        if nxt:
            layers.append(nxt)
        frontier = nxt
    return layers


def reachable_set(adj, start):
    """All nodes reachable from `start` (excluding `start` itself)."""
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    seen.discard(start)
    return seen


def shortest_path_len(adj, src, dst):
    if src == dst:
        return 0
    seen = {src}
    frontier = [src]
    d = 0
    while frontier:
        d += 1
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v == dst:
                    return d
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
        frontier = nxt
    return None


def has_cycle(nodes, adj):
    """Kahn's topological peel: leftover nodes => a cycle exists."""
    indeg = {v: 0 for v in nodes}
    for u in nodes:
        for v in adj[u]:
            indeg[v] += 1
    queue = [v for v in nodes if indeg[v] == 0]
    visited = 0
    while queue:
        u = queue.pop()
        visited += 1
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    return visited != len(nodes)


def weak_component(adj, radj, start):
    """Nodes in `start`'s weakly-connected component (edges as undirected)."""
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in adj[u] + radj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def _list_answer(ids):
    """(answers, gold, answer_type) for a node-set answer rendered as JSON."""
    s = sorted(ids)
    return [json.dumps(s)], s, "list"


# --------------------------------------------------------------------------- #
# task base + the family
# --------------------------------------------------------------------------- #
class GraphTask(Task):
    reserve_tokens = 512
    min_size = 6
    acyclic = False
    avg_deg = 2.5

    def _acyclic(self, rng):
        return self.acyclic

    def query(self, g, rng):
        """-> (question, answers, gold, answer_type, meta). Implemented per task."""
        raise NotImplementedError

    def build(self, size, rng):
        n = max(self.min_size, size)
        g = gen_graph(n, rng, avg_deg=self.avg_deg, acyclic=self._acyclic(rng))
        question, answers, gold, answer_type, meta = self.query(g, rng)
        context = "The graph has the following edges:\n" + render_edges(g.edges, rng)
        return Sample(
            GRAPH_PREAMBLE, context, question, "",
            answers=answers, gold=gold, answer_type=answer_type, meta=meta,
        )


class GraphBFS(GraphTask):
    name = "graph_bfs"

    def query(self, g, rng):
        starts = list(g.nodes)
        rng.shuffle(starts)
        for start in starts:
            layers = bfs_layers(g.adj, start)
            depths = [d for d in range(1, len(layers)) if layers[d]]
            if depths:
                depth = rng.choice(depths)
                answers, gold, at = _list_answer(layers[depth])
                q = (f"Operation:\nPerform a BFS from node {start} and list every node "
                     f"that is exactly {depth} step(s) away.")
                return q, answers, gold, at, {"op": "bfs", "start": start, "depth": depth}
        start = g.nodes[0]
        answers, gold, at = _list_answer(g.adj[start])
        q = (f"Operation:\nPerform a BFS from node {start} and list every node that is "
             f"exactly 1 step(s) away.")
        return q, answers, gold, at, {"op": "bfs", "start": start, "depth": 1}


class GraphParents(GraphTask):
    name = "graph_parents"

    def query(self, g, rng):
        nodes = list(g.nodes)
        rng.shuffle(nodes)
        node = next((x for x in nodes if g.radj[x]), g.nodes[0])
        answers, gold, at = _list_answer(g.radj[node])
        q = f"Operation:\nList all parents (direct predecessors) of node {node}."
        return q, answers, gold, at, {"op": "parents", "node": node}


class GraphChildren(GraphTask):
    name = "graph_children"

    def query(self, g, rng):
        nodes = list(g.nodes)
        rng.shuffle(nodes)
        node = next((x for x in nodes if g.adj[x]), g.nodes[0])
        answers, gold, at = _list_answer(g.adj[node])
        q = f"Operation:\nList all children (direct successors) of node {node}."
        return q, answers, gold, at, {"op": "children", "node": node}


class GraphDescendants(GraphTask):
    name = "graph_descendants"

    def query(self, g, rng):
        nodes = list(g.nodes)
        rng.shuffle(nodes)
        node = next((x for x in nodes if g.adj[x]), g.nodes[0])
        answers, gold, at = _list_answer(reachable_set(g.adj, node))
        q = (f"Operation:\nList all descendants of node {node} (every node reachable by "
             f"following directed edges from it).")
        return q, answers, gold, at, {"op": "descendants", "node": node}


class GraphAncestors(GraphTask):
    name = "graph_ancestors"

    def query(self, g, rng):
        nodes = list(g.nodes)
        rng.shuffle(nodes)
        node = next((x for x in nodes if g.radj[x]), g.nodes[0])
        answers, gold, at = _list_answer(reachable_set(g.radj, node))
        q = (f"Operation:\nList all ancestors of node {node} (every node that can reach "
             f"it by following directed edges).")
        return q, answers, gold, at, {"op": "ancestors", "node": node}


class GraphShortestPath(GraphTask):
    name = "graph_shortest_path"

    def query(self, g, rng):
        nodes = list(g.nodes)
        rng.shuffle(nodes)
        for src in nodes:
            reach = reachable_set(g.adj, src)
            if reach:
                dst = rng.choice(sorted(reach))
                gold = shortest_path_len(g.adj, src, dst)
                q = (f"Operation:\nFind the length (number of edges) of the shortest "
                     f"directed path from node {src} to node {dst}.")
                return q, [str(gold)], gold, "int", {"op": "shortest_path", "src": src, "dst": dst}
        src, dst = g.nodes[0], g.nodes[1]
        q = (f"Operation:\nFind the length (number of edges) of the shortest directed "
             f"path from node {src} to node {dst}.")
        return q, ["0"], 0, "int", {"op": "shortest_path", "src": src, "dst": src}


class GraphReachable(GraphTask):
    name = "graph_reachable"

    def query(self, g, rng):
        nodes = list(g.nodes)
        rng.shuffle(nodes)
        src = nodes[0]
        reach = reachable_set(g.adj, src)
        not_reach = [x for x in g.nodes if x != src and x not in reach]
        want_yes = rng.random() < 0.5
        if reach and (want_yes or not not_reach):
            dst, ans = rng.choice(sorted(reach)), "Yes"
        elif not_reach:
            dst, ans = rng.choice(not_reach), "No"
        else:  # degenerate: nothing reachable and nothing else -> n==1 never happens
            dst, ans = src, "Yes"
        q = (f"Operation:\nIs node {dst} reachable from node {src} by following directed "
             f"edges? Answer Yes or No.")
        return q, [ans], ans, "string", {"op": "reachable", "src": src, "dst": dst}


class GraphCycle(GraphTask):
    name = "graph_cycle"

    def _acyclic(self, rng):
        # ~half the samples are DAGs so the yes/no answer is balanced.
        return rng.random() < 0.5

    def query(self, g, rng):
        ans = "Yes" if has_cycle(g.nodes, g.adj) else "No"
        q = "Operation:\nDoes this directed graph contain a cycle? Answer Yes or No."
        return q, [ans], ans, "string", {"op": "cycle"}


class GraphComponentSize(GraphTask):
    name = "graph_component_size"
    avg_deg = 1.3  # sparser -> more than one weak component

    def query(self, g, rng):
        node = rng.choice(g.nodes)
        gold = len(weak_component(g.adj, g.radj, node))
        q = (f"Operation:\nHow many nodes are in the connected component containing node "
             f"{node} (treating every edge as undirected)?")
        return q, [str(gold)], gold, "int", {"op": "component_size", "node": node}


class GraphMaxDegree(GraphTask):
    name = "graph_max_degree"

    def query(self, g, rng):
        deg = {v: len(g.adj[v]) + len(g.radj[v]) for v in g.nodes}
        top = max(deg.values())
        winners = [v for v in g.nodes if deg[v] == top]
        answers, gold, at = _list_answer(winners)
        q = (f"Operation:\nList the node(s) with the highest total degree (in-degree "
             f"plus out-degree).")
        return q, answers, gold, at, {"op": "max_degree", "degree": top}
