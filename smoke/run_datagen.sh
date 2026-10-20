#!/usr/bin/env bash
# Small real orchestration test; set corpus/cluster paths from smoke/e2e.py.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${INPUT_DIR:?set INPUT_DIR to the smoke corpus snapshot}"
: "${CLUSTER_DIR:?set CLUSTER_DIR to its tested clusters}"
: "${API_BASE_URL:?set API_BASE_URL including /v1}"
export OUT="${OUT:-smoke/runs/orchestrator}"
export PYTHON="${PYTHON:-python}"
export TOKENIZER=hf:Qwen/Qwen3-8B
export API_MODEL=Qwen/Qwen3-8B
export API_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
export CONTEXT_LENGTHS=8192
export RULER_TOTAL=22 UNTIE_TOTAL=4 MRCR_TOTAL=4 IDK_TOTAL=4
export MULTIHOP_DOCS=1 SUMMARIZE_DOCS=1 WORKERS=1 RUN_LLM_INFER=1
bash scripts/datagen.sh
