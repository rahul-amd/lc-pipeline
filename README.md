# lc-pipeline — long-context datagen

A pipeline for building long-context training data. Every generator emits the
**same `Sample` contract** and the **same JSONL schema / `pt`+`sft` formats**, so
data from different generators drops into one training mix unchanged.

## Generators

**`ruler_pp/` — synthetic, no LLM.** RULER-style algorithmic tasks whose gold
answers are computed deterministically. Each task implements
`build(size, rng) -> Sample`; a shared driver binary-searches `size` to fill a
token budget. See [`ruler_pp/README.md`](ruler_pp/README.md). 22 tasks; sample
output in `data/`. The set includes a **GraphWalks-style graph-reasoning family**
(`ruler_pp/tasks/graph.py`): one synthetic directed graph is rendered as a
shuffled `A -> B` edge list, and ten ops query it — `graph_bfs`,
`graph_parents`, `graph_children`, `graph_descendants`, `graph_ancestors`,
`graph_shortest_path`, `graph_reachable`, `graph_cycle`, `graph_component_size`,
`graph_max_degree`. Answers are a JSON node list, an integer, or Yes/No, so they
shape into `pt`/`sft` like every other task.

The remaining generators run over a **FinePDFs sample**, all reading the corpus
through one shared loader (`corpus/loader.py` — parquet streaming + quality
filters):

**`corpus.cluster`** — embed docs (Qwen3-Embedding-0.6B), find exact top-k cosine
neighbours, then greedily chain near-docs into ~128k-token clusters so related
documents can be concatenated into one long-context window. Output in
`cluster_out/` (`clusters.parquet` = doc→cluster labels; embeddings cached for
cheap re-clustering via `--reuse_cache`). Run this before `untie`'s cluster source.

**`untie/`** — "Untie the Knots" (arXiv:2409.04774) adapted to paragraph level.
Splits a long document into paragraph units, shuffles them, and asks the model to
recover the order. Two flows:
- `reconstruct`: input = shuffled paragraphs (~n/2 tokens); answer = the text in
  original order (~n/2).
- `permutation`: input = shuffled paragraphs (~n tokens); answer = the label
  order that reconstructs it (a short list of indices).

Source text is a mix of **clusters** (concatenated cross-document, the long
knots) and **single** long docs. It is a `CorpusTask` — no synthetic `size` knob;
length is set by how much source text packs under `--context_len`.

**`mrcr/` — MRCR-style marker retrieval.** Michelangelo's MRCR
(arXiv:2409.12640) adapted to single-turn, deterministic, no-LLM data. The
context is a long numbered list of real FinePDFs passages; each passage's text
sits between `<<<`/`>>>` fences and is followed by plain-English sentences saying
what it is **marked as** — concrete nouns from `res/markers.txt`
(`"This paragraph is a lemon. It is also a car."`) — and sometimes what it is
**not** (`"but remember that it is not a dune."`). Exactly one paragraph (the
target) is positively marked with *both* query markers; confounders share one
marker, some as negation near-misses (`"a pepper... not a dune"`), the rest are
noise. The query asks for "the paragraph that is both a {a} and a {b}", and the
answer reproduces that passage **verbatim**, prefixed with a per-sample random
string (a guard against degenerate output). Gold = the exact source passage; eval
metric = `difflib.SequenceMatcher` ratio ∈ [0,1]. `size` = number of paragraphs,
binary-searched by the shared fitter to fill the budget, so the same task scales
across 32k/128k/1M. Difficulty knobs (in `meta`): `--k` (query arity),
`--confounder_frac`, `--neg_frac` — all decoupled from length.

## Shared framework

`ruler_pp/base.py` defines `Sample` plus two task bases: `Task` (synthetic) and
`CorpusTask` (corpus-driven). `untie` and `mrcr` reuse `ruler_pp`'s pluggable
tokenizer (`get_tokenizer`) and output shaping (`shape_row`), so their JSONL
carries the same fields (`instruction/context/question/answers/gold/answer_type/
length/meta`) and honours the same `--format pt|sft` as the synthetic tasks. The
FinePDFs reading is shared the other way — every corpus generator streams and
quality-filters through `corpus/loader.py`, so the load logic lives in one place.

## Layout

```
lc-pipeline/
  ruler_pp/            synthetic-task engine + shared Sample/CorpusTask/tokenizer/shaping
  corpus/              corpus tooling
    loader.py            shared FinePDFs parquet streaming + quality filters
    cluster.py           cluster FinePDFs by embedding similarity into ~128k bundles
  untie/               paragraph-unshuffle task
    knots.py             the UntieKnots CorpusTask + generator (python -m untie)
    viz.py               Gradio viewer for the generated samples (python -m untie.viz)
  mrcr/                MRCR-style marker retrieval
    mrcr.py              MRCRTask + passage-pool loader + generator (python -m mrcr)
    viz.py               Gradio viewer for the generated samples (python -m mrcr.viz)
  scripts/             utilities, checkers & one-off benchmarks
    download_finepdfs.py   pull the FinePDFs sample into finpdf_sample/
    verify_golds.py        independent re-checker for ruler++ golds
    verify_mrcr.py         independent re-checker for mrcr golds
    st_bench.py            SentenceTransformers throughput bench (cu128 toolchain)
    vllm_bench.py          vLLM embedding bench (ruled out on this GPU; kept for the record)
  requirements.txt     dependencies, grouped by stage (see comments inside)
  data/                ruler++ sample output (one jsonl per task)
  res/                 word lists, marker nouns etc. for synthetic/mrcr tasks
  finpdf_sample/       FinePDFs parquet shards (input corpus)
  cluster_out/         clustering output (cluster labels, cached embeddings, summary)
  mrcr_out/            mrcr sample output (mrcr.jsonl)
```

## Usage

All `python -m ...` commands run from the repo root (so `corpus`, `untie`,
`mrcr`, and `ruler_pp` resolve as packages).

```bash
# Install deps (see requirements.txt; note the cu128 torch caveat in the
# hardware note below before installing torch into the WSL venv).
pip install -r requirements.txt

# ruler++ synthetic tasks
python -m ruler_pp --task all --max_seq_length 4096 --num_samples 20

# Cluster the corpus (run once; re-cluster cheaply with --reuse_cache).
# Required before untie's cluster source.
python -m corpus.cluster --input_dir finpdf_sample --output_dir cluster_out

# untie: paragraph-unshuffle data (both flows, clusters + single docs)
python -m untie --output_dir untie_out --max_clusters 1000 --max_single_docs 1000
python -m untie.viz --data_dir untie_out          # eyeball shuffles, gold order, answer

# mrcr: marker-retrieval data (single-turn, deterministic gold)
python -m mrcr --input_dir finpdf_sample --output_dir mrcr_out \
    --max_seq_length 8192 --num_samples 200 --format sft
python scripts/verify_mrcr.py mrcr_out            # -> TOTAL: N/N samples verified
python -m mrcr.viz --data_dir mrcr_out            # eyeball markers, the TARGET, answer
```

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

To run at full scale — every cluster AND every single doc — set the caps absurdly
high so nothing is dropped (`--min_doc_tokens 20000`, = 2x the chunk target, keeps
single docs that can still split into >=2 chunks):

```bash
python -m untie --output_dir untie_out_full \
    --sources cluster,single \
    --max_clusters 100000000 --max_single_docs 100000000 \
    --min_doc_tokens 20000
```

> Hardware note: embedding in the clustering stage runs on an RTX 5060 Ti
> (Blackwell, sm_120, 16 GB) via a CUDA-12.8 ("cu128") PyTorch toolchain in WSL —
> the newer cu130 build deadlocks on this card. Run the embedding/generation
> steps with that interpreter (e.g. `~/emb-cu128/bin/python -m untie ...`).
