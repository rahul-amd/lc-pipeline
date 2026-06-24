"""Cluster long documents by embedding similarity, for long-context assembly.

Phase 2a. Pipeline (each stage cached so re-clustering is cheap):
  1. load   : stream parquet docs + quality filters (shared `corpus.loader`)
  2. embed  : Qwen3-Embedding-0.6B on the first N tokens, fp16 GPU, L2-normalized
  3. neighbours : exact top-k cosine via batched GPU matmul (faiss not needed at this scale)
  4. cluster : greedy single-linkage NN chaining, bounded by a token budget + sim floor

Token budget uses the dataset's own `token_count` column (no re-tokenizing 24 GB).

Output: per-doc cluster labels (doc_id -> cluster_id, rank, n_tokens) plus a
cluster summary. A "cluster" is a set of mutually-near docs whose combined length
fits the budget, ready to be concatenated later into one long-context sample.

    python -m corpus.cluster --input_dir finpdf_sample --output_dir cluster_out
    python -m corpus.cluster --input_dir finpdf_sample --reuse_cache   # re-cluster, skip embedding
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import time
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

from corpus import loader

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001
    def tqdm(x, **k):
        return x


# --------------------------------------------------------------------------- #
# 1. load  (quality filtering lives in corpus.loader; we just collect arrays)
# --------------------------------------------------------------------------- #
def load_docs(input_dir, max_docs=None, min_tokens=200, min_edu=0.75, drop_dups=True,
              char_cap=None):
    files, _cols, text_col, id_col = loader.schema(input_dir)
    print(f"text column: {text_col!r} | id column: {id_col!r}")

    texts, ids, srcs, toks = [], [], [], []
    stats: Counter = Counter()
    for did, src, txt, ntok in loader.iter_docs(
            input_dir, min_tokens=min_tokens, min_edu=min_edu, drop_dups=drop_dups,
            char_cap=char_cap, batch_size=8192, max_docs=max_docs, stats=stats):
        texts.append(txt)
        ids.append(did)
        srcs.append(src)
        toks.append(ntok)

    print(f"scanned {stats['seen']} rows | dropped: empty {stats['empty']}, "
          f"short {stats['short']}, dup {stats['dup']}, low-edu {stats['lowedu']}")
    print(f"kept {len(texts)} docs from {len(files)} files")
    return texts, ids, srcs, np.array(toks, dtype=np.int64)


# --------------------------------------------------------------------------- #
# 2. embed
# --------------------------------------------------------------------------- #
def embed(texts, model_name, embed_tokens, batch_size, device):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device,
                                model_kwargs={"torch_dtype": "float16",
                                              "attn_implementation": "sdpa"})
    model.max_seq_length = embed_tokens
    # Char pre-truncation avoids tokenizing megabytes; ST then truncates to embed_tokens.
    char_cap = embed_tokens * 12
    clipped = [t[:char_cap] for t in texts]
    emb = model.encode(
        clipped, batch_size=batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True,
    )
    return emb.astype(np.float32)


# --------------------------------------------------------------------------- #
# 3. exact top-k neighbours (batched GPU matmul)
# --------------------------------------------------------------------------- #
def topk_neighbours(emb, k, device, chunk=2048):
    import torch

    n = emb.shape[0]
    k = min(k, n - 1)
    # fp16 engages tensor cores: the matmul is O(n^2 * d) and dominates at 500k docs.
    # Loss is negligible for top-k neighbour selection on L2-normalized vectors.
    x = torch.from_numpy(emb).to(device=device, dtype=torch.float16)
    idx_out = np.empty((n, k), dtype=np.int64)
    sim_out = np.empty((n, k), dtype=np.float32)
    for s in tqdm(range(0, n, chunk), desc="neighbours"):
        q = x[s:s + chunk]
        sims = q @ x.T                      # (b, n) cosine since rows are normalized
        sims[torch.arange(q.shape[0]), torch.arange(s, s + q.shape[0])] = -1.0  # mask self
        vals, ind = torch.topk(sims, k, dim=1)
        idx_out[s:s + q.shape[0]] = ind.cpu().numpy()
        sim_out[s:s + q.shape[0]] = vals.float().cpu().numpy()
    del x
    if device == "cuda":
        torch.cuda.empty_cache()
    return idx_out, sim_out


# --------------------------------------------------------------------------- #
# 4. greedy single-linkage NN chaining, bounded by token budget + sim floor
# --------------------------------------------------------------------------- #
def greedy_chain(nbr_idx, nbr_sim, tokens, budget, floor):
    n = len(tokens)
    used = np.zeros(n, dtype=bool)
    cluster_id = np.full(n, -1, dtype=np.int64)
    rank = np.full(n, -1, dtype=np.int64)

    def push_neighbours(heap, i):
        for j, s in zip(nbr_idx[i], nbr_sim[i]):
            if s < floor:
                break  # neighbours are sorted desc; rest are worse
            if not used[j]:
                heapq.heappush(heap, (-float(s), int(j)))

    cid = 0
    for seed in range(n):
        if used[seed]:
            continue
        used[seed] = True
        cluster_id[seed] = cid
        rank[seed] = 0
        total = int(tokens[seed])
        r = 1
        heap: list = []
        push_neighbours(heap, seed)
        while heap and total < budget:
            neg_s, cand = heapq.heappop(heap)
            if used[cand]:
                continue
            if -neg_s < floor:
                break
            if total + int(tokens[cand]) > budget:
                continue  # too big for the remaining room; may seed its own cluster later
            used[cand] = True
            cluster_id[cand] = cid
            rank[cand] = r
            r += 1
            total += int(tokens[cand])
            push_neighbours(heap, cand)
        cid += 1
    return cluster_id, rank, cid


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser("corpus.cluster")
    p.add_argument("--input_dir", default="finpdf_sample")
    p.add_argument("--output_dir", default="cluster_out")
    p.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--embed_tokens", type=int, default=256, help="prefix tokens embedded per doc")
    p.add_argument("--budget", type=int, default=131072, help="max tokens per cluster")
    p.add_argument("--floor", type=float, default=0.35, help="min cosine similarity to join a cluster")
    p.add_argument("--min_tokens", type=int, default=200, help="drop docs shorter than this (token_count)")
    p.add_argument("--min_edu", type=float, default=0.75, help="drop docs with mean fw_edu_scores below this")
    p.add_argument("--keep_dups", action="store_true", help="keep near-duplicates (minhash_cluster_size > 1)")
    p.add_argument("--topk", type=int, default=64, help="neighbours considered per doc")
    p.add_argument("--batch_size", type=int, default=32, help="embedding batch size")
    p.add_argument("--max_docs", type=int, default=500000, help="cap docs kept after filtering")
    p.add_argument("--device", default=None, help="cuda | cpu (auto if unset)")
    p.add_argument("--reuse_cache", action="store_true", help="reuse cached embeddings/metadata")
    args = p.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    emb_path = os.path.join(args.output_dir, "embeddings.npy")
    meta_path = os.path.join(args.output_dir, "docs.parquet")

    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {args.device}")

    t0 = time.time()
    if args.reuse_cache and os.path.exists(emb_path) and os.path.exists(meta_path):
        print("reusing cached embeddings + metadata")
        meta = pq.read_table(meta_path).to_pydict()
        ids, srcs = meta["doc_id"], meta["source"]
        tokens = np.array(meta["n_tokens"], dtype=np.int64)
        emb = np.load(emb_path)
    else:
        texts, ids, srcs, tokens = load_docs(
            args.input_dir, max_docs=args.max_docs, min_tokens=args.min_tokens,
            min_edu=args.min_edu, drop_dups=not args.keep_dups,
            char_cap=args.embed_tokens * 12,
        )
        emb = embed(texts, args.model, args.embed_tokens, args.batch_size, args.device)
        np.save(emb_path, emb)
        import pyarrow as pa
        pq.write_table(pa.table({
            "doc_id": [str(x) for x in ids],
            "source": srcs,
            "n_tokens": tokens.tolist(),
        }), meta_path)
        del texts
    print(f"embeddings: {emb.shape} | tokens: total {int(tokens.sum()):,}  median {int(np.median(tokens))}")

    nbr_idx, nbr_sim = topk_neighbours(emb, args.topk, args.device)
    cluster_id, rank, n_clusters = greedy_chain(nbr_idx, nbr_sim, tokens, args.budget, args.floor)

    # --- write labels + summary ---
    import pyarrow as pa
    labels_path = os.path.join(args.output_dir, "clusters.parquet")
    pq.write_table(pa.table({
        "doc_id": [str(x) for x in ids],
        "source": srcs,
        "n_tokens": tokens.tolist(),
        "cluster_id": cluster_id.tolist(),
        "rank": rank.tolist(),
    }), labels_path)

    sizes = np.bincount(cluster_id, minlength=n_clusters)
    ctoks = np.bincount(cluster_id, weights=tokens, minlength=n_clusters).astype(np.int64)
    multi = int((sizes > 1).sum())
    summary = {
        "n_docs": int(len(ids)),
        "n_clusters": int(n_clusters),
        "multi_doc_clusters": multi,
        "singletons": int((sizes == 1).sum()),
        "budget": args.budget,
        "floor": args.floor,
        "embed_tokens": args.embed_tokens,
        "model": args.model,
        "filters": {
            "min_tokens": args.min_tokens,
            "min_edu": args.min_edu,
            "drop_dups": not args.keep_dups,
        },
        "cluster_tokens_pct": {
            "p50": int(np.percentile(ctoks, 50)),
            "p90": int(np.percentile(ctoks, 90)),
            "max": int(ctoks.max()),
        },
        "biggest_clusters_docs": sorted(sizes.tolist(), reverse=True)[:10],
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {labels_path}")


if __name__ == "__main__":
    main()
