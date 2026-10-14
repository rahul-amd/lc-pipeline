# Long-context pipeline validation

All new tests, sampled input, logs and generated datasets live here. `runs/` is
ignored: it contains source documents, downloaded model weights and large data.
Do not publish those artifacts without checking their source licensing. The
tracked scripts/configuration are the reproducible validation entry points.

Run from the repository root in an environment with the repository dependencies,
`pytest`, and a PyTorch build for your hardware:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest smoke/test_pipeline.py -q \
  -p no:cacheprovider --basetemp=smoke/runs/pytest-tmp \
  --junitxml=smoke/runs/regression.xml
```

The real-tokenizer regression uses `Qwen/Qwen3-8B`. Integration uses a GPU for
Qwen3-Embedding-0.6B and a running OpenAI-compatible Qwen3-8B server. For example,
in an environment with vLLM installed:

```bash
vllm serve Qwen/Qwen3-8B --max-model-len 32768 --port 8000
PYTHONDONTWRITEBYTECODE=1 python smoke/e2e.py \
  --source finpdf_sample --api-base http://localhost:8000/v1 \
  --output smoke/runs/e2e
```

`e2e.py` selects six bounded real documents, preserves their source IDs and hashes,
embeds/clusters them on GPU, tests cluster cache reuse, and runs every generator:

| Path | Coverage |
|---|---|
| 22 synthetic tasks | 8K/32K budgets, PT/SFT, few-shot, serial/parallel support |
| Untie | reconstruction/permutation × cluster/single × PT/SFT |
| MRCR/IDK | gold verification, all three IDK kinds, PT/SFT |
| Multi-hop/summarization | prepare → real API → strict assembly, PT/SFT |
| Large output cases | three synthetic tasks at 128K, including long list answers |

The input snapshot and embedding download stay inside the run directory. The
ASCII fraction used for smoke sampling is not a language-identification model.
The embedding floor is deliberately relaxed to exercise multi-document assembly;
this does not establish cluster coherence. Final validation checks nonempty files,
unique IDs, actual training token counts, deterministic golds and Untie round-trip
reconstruction. Model-generated QA/summary correctness requires separate review.
`RESULT.json` is written only after all final validations pass. A successful shell
exit alone is insufficient evidence on some ROCm container builds.

Test the main shell orchestrator separately against that snapshot:

```bash
INPUT_DIR=smoke/runs/e2e/corpus CLUSTER_DIR=smoke/runs/e2e/clusters \
  API_BASE_URL=http://localhost:8000/v1 bash smoke/run_datagen.sh
```

`gpu_exec.sh` is the local Slurm/ROCm container adapter; its default container/cache
paths are specific to this cluster. Supply `LC_CONTAINER` and `HF_HOME` elsewhere,
or run the commands directly in your own environment. Allocate an approved Slurm
partition first, and use `srun --ntasks=1 ... bash smoke/gpu_exec.sh ...`.

See [AUDIT.md](AUDIT.md) for findings and remaining quality/release work, and
[RESULTS.md](RESULTS.md) for the recorded validation outcome.
