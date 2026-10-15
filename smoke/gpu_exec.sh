#!/usr/bin/env bash
# Invoke inside a Slurm step; all validation caches are kept below smoke/.
set -euo pipefail
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"
export APPTAINERENV_PYTHONPATH="$root_dir"
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_HF_HOME="${HF_HOME:-/shared_silo/scratch/hf_cache}"
export APPTAINERENV_OMP_NUM_THREADS=2
export APPTAINERENV_OPENBLAS_NUM_THREADS=2
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
export APPTAINERENV_ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
export APPTAINERENV_HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export APPTAINERENV_VLLM_WORKER_MULTIPROC_METHOD=spawn
export APPTAINERENV_VLLM_CACHE_ROOT="$root_dir/smoke/runs/cache/vllm"
export APPTAINERENV_TRITON_CACHE_DIR="$root_dir/smoke/runs/cache/triton"
export APPTAINERENV_TMPDIR=/tmp
exec singularity exec --rocm --cleanenv --bind /shared_silo/scratch \
  "${LC_CONTAINER:-/shared_silo/scratch/containers/vllm-openai-rocm-gemma4.sif}" "$@"
