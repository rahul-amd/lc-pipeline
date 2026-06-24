# lc-pipeline — long-context datagen

A phased pipeline for building long-context training data. Every generator emits
the **same `Sample` contract** and the **same JSONL schema / `pt`+`sft` formats**,
so data from different phases drops into one training mix unchanged.

## Phases

**Phase 1 — `ruler_pp/` (synthetic, no LLM).** RULER-style algorithmic tasks
whose gold answers are computed deterministically. Each task implements
`build(size, rng) -> Sample`; a shared driver binary-searches `size` to fill a
token budget. See [`ruler_pp/README.md`](ruler_pp/README.md). 22 tasks; sample
output in `data/`. The set includes a **GraphWalks-style graph-reasoning family**
(`ruler_pp/tasks/graph.py`): one synthetic directed graph is rendered as a
shuffled `A -> B` edge list, and ten ops query it — `graph_bfs`,
`graph_parents`, `graph_children`, `graph_descendants`, `graph_ancestors`,
`graph_shortest_path`, `graph_reachable`, `graph_cycle`, `graph_component_size`,
`graph_max_degree`. Answers use GraphWalks' `Final Answer:` cue (a JSON node list,
an integer, or Yes/No), so they shape into `pt`/`sft` like every other task.

**Phase 2 — real long documents.** Two stages over a FinePDFs sample, both
reading the corpus through one shared loader (`corpus/loader.py` — parquet
streaming + quality filters, reused by both stages):

1. **`corpus.cluster`** (Phase 2a) — embed docs (Qwen3-Embedding-0.6B), find
   exact top-k cosine neighbours, then greedily chain near-docs into ~128k-token
   clusters so related documents can be concatenated into one long-context
   window. Output in `cluster_out/` (`clusters.parquet` = doc→cluster labels;
   embeddings cached for cheap re-clustering via `--reuse_cache`).
2. **`untie`** (Phase 2b) — "Untie the Knots" (arXiv:2409.04774) adapted to
   paragraph level. Splits a long document into paragraph units, shuffles them,
   and asks the model to recover the order. Two flows:
   - `reconstruct`: input = shuffled paragraphs (~n/2 tokens); answer = the text
     in original order (~n/2).
   - `permutation`: input = shuffled paragraphs (~n tokens); answer = the label
     order that reconstructs it (a short list of indices).

   Source text is a mix of **clusters** (concatenated cross-document, the long
   knots) and **single** long docs. It is a `CorpusTask` — no synthetic `size`
   knob; length is set by how much source text packs under `--context_len`.

## How Phase 2 reuses the Phase-1 framework

`ruler_pp/base.py` defines `Sample` plus two task bases: `Task` (synthetic) and
`CorpusTask` (corpus-driven). `untie/knots.py` subclasses `CorpusTask`, builds
`Sample`s, and reuses `ruler_pp`'s pluggable tokenizer (`get_tokenizer`) and
output shaping (`shape_row`). So the untie JSONL carries the same fields
(`instruction/context/question/answers/gold/answer_type/length/meta`) and
honours the same `--format pt|sft` as the synthetic tasks. The FinePDFs reading
itself is shared the other way — both Phase-2 stages stream and quality-filter
the corpus through `corpus/loader.py`, so the load logic lives in one place.

## Layout

```
lc-pipeline/
  ruler_pp/            Phase 1 engine (synthetic tasks) + shared Sample/CorpusTask/tokenizer/shaping
  corpus/              Phase 2 corpus tooling
    loader.py            shared FinePDFs parquet streaming + quality filters (used by both stages)
    cluster.py           Phase 2a: cluster FinePDFs by embedding similarity into ~128k bundles
  untie/               Phase 2b: paragraph-unshuffle task
    knots.py             the UntieKnots CorpusTask + generator (python -m untie)
    viz.py               Gradio viewer for the generated samples (python -m untie.viz)
  scripts/             utilities, checkers & one-off benchmarks
    download_finepdfs.py   pull the FinePDFs sample into finpdf_sample/
    verify_golds.py        independent re-checker for Phase-1 golds
    st_bench.py            SentenceTransformers throughput bench (cu128 toolchain)
    vllm_bench.py          vLLM embedding bench (ruled out on this GPU; kept for the record)
  requirements.txt     dependencies, grouped by stage (see comments inside)
  data/                Phase-1 sample output (one jsonl per task)
  res/                 word lists etc. for synthetic tasks
  finpdf_sample/       FinePDFs parquet shards (input corpus)
  cluster_out/         Phase-2a output (cluster labels, cached embeddings, summary)
```

## Usage

```bash
# Install deps (see requirements.txt for per-stage groups; note the cu128 torch
# caveat in the hardware note below before installing torch into the WSL venv).
pip install -r requirements.txt

# Phase 1: synthetic tasks
python -m ruler_pp --task all --max_seq_length 4096 --num_samples 20

# Phase 2a: cluster the corpus (run once; re-cluster cheaply with --reuse_cache)
python -m corpus.cluster --input_dir finpdf_sample --output_dir cluster_out

# Phase 2b: generate unshuffle data (both flows, clusters + single docs)
python -m untie --output_dir untie_out --max_clusters 1000 --max_single_docs 1000

# Eyeball the generated samples (shuffled chunks, gold order, answer) in a browser
python -m untie.viz --data_dir untie_out

# Phase 2b at full scale: use the ENTIRE corpus — every cluster AND every single
# doc. Caps are set absurdly high so nothing is dropped; --min_doc_tokens 20000
# (= 2x the chunk target) keeps single docs that can still split into >=2 chunks.
python -m untie --output_dir untie_out_full \
    --sources cluster,single \
    --max_clusters 100000000 --max_single_docs 100000000 \
    --min_doc_tokens 20000
```

> All `python -m ...` commands run from the repo root (so `corpus`, `untie`, and
> `ruler_pp` resolve as packages).

> Memory: the cluster source loads the text of every selected cluster's member
> docs into RAM in one pass. Using all clusters pulls most of the corpus into
> memory at once — run it on a box with enough RAM, or lower `--max_clusters`
> and run in batches if it gets tight.

### `python -m untie` flags

`--context_len` (n, default 131072), `--flows` (`reconstruct,permutation`),
`--sources` (`cluster,single`), `--format` (`pt`|`sft`), `--tokenizer`
(`whitespace` default; `hf:<name>` / `tiktoken[:enc]` for exact budgets),
`--target_para_tokens` (unit size; raise toward the paper's few-large-chunks
regime), `--max_clusters` / `--max_single_docs` / `--samples_per_doc`,
quality filters `--min_doc_tokens` / `--min_tokens` / `--min_edu` / `--keep_dups`.

`--num_samples` sets an optional **per-flow** target (each `<flow>.jsonl` aims for
this many rows). The sources are streamed once; if the pool is exhausted before
the target, it is reused — each reuse re-chunks/re-spans/re-shuffles under a fresh
seed, so reused docs yield distinct samples (variety is greatest for docs larger
than the window; docs that fit only vary by shuffle). `--max_reuse` (default 5)
caps the extra passes and stops early if a full pass adds nothing. The reuse pool
is bounded by `--max_clusters` + `--max_single_docs`, so raise those for a larger,
more varied pool. Unset `--num_samples` = the original single pass.

> Hardware note: embedding in Phase 2a runs on an RTX 5060 Ti (Blackwell, sm_120,
> 16 GB) via a CUDA-12.8 ("cu128") PyTorch toolchain in WSL — the newer cu130
> build deadlocks on this card. Run the embedding/generation steps with that
> interpreter (e.g. `~/emb-cu128/bin/python -m untie ...`).
