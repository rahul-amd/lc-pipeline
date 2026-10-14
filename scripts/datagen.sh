#!/usr/bin/env bash
#
# datagen.sh — generate ~1,000,000 long-context SFT samples across the offline
# generators, plus prepare the LLM-in-the-loop request files.
#
# Balanced allocation (offline, deterministic — sums to 1,000,000):
#     ruler++ : 400,000   (across 22 synthetic tasks)
#     untie   : 300,000   (across 2 flows: reconstruct + permutation)
#     mrcr    : 150,000
#     idk     : 150,000
#
# Each generator's quota is split evenly across four context-length buckets
# (8k / 32k / 128k / 160k) so the model sees a range of context sizes.
#
# multihop + summarize are LLM-in-the-loop: this script only runs their `prepare`
# step (emits requests.jsonl + docs.jsonl). Run those requests through your
# inference engine, then `python -m <gen> assemble ...` to produce final samples.
# Those are NOT counted in the 1M above.
#
# Everything is tunable via env vars (see the config block). Individual stages
# can be toggled with RUN_RULER / RUN_UNTIE / RUN_MRCR / RUN_IDK / RUN_LLM_PREP.
#
# Usage:
#     bash scripts/datagen.sh
#     OUT=/data/lc FORMAT=sft TOKENIZER=hf:Qwen/Qwen2.5-0.5B bash scripts/datagen.sh
#     RUN_UNTIE=0 RUN_LLM_PREP=0 bash scripts/datagen.sh      # skip stages
#
set -euo pipefail

# Run from the repo root so `python -m corpus/untie/mrcr/idk/...` resolve.
cd "$(dirname "$0")/.."

# Generators are deterministic across hash seeds; pin third-party behavior too.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

# --------------------------------------------------------------------------- #
# config (override any of these via the environment)
# --------------------------------------------------------------------------- #
PYTHON="${PYTHON:-python}"                     # interpreter for datagen
EMB_PYTHON="${EMB_PYTHON:-$PYTHON}"            # clustering embeds on GPU; on the
                                               # RTX 5060 Ti use the cu128 venv,
                                               # e.g. EMB_PYTHON=~/emb-cu128/bin/python
INPUT_DIR="${INPUT_DIR:-finpdf_sample}"        # FinePDFs parquet shards
OUT="${OUT:-datagen_out}"                      # root output dir
FORMAT="${FORMAT:-sft}"                        # pt | sft
TOKENIZER="${TOKENIZER:-hf:Qwen/Qwen2.5-0.5B}" # exact budgets across generators
SEED="${SEED:-42}"
WORKERS="${WORKERS:-0}"                         # parallel builder procs per stage
                                               # (0 = all CPU cores). Tokenizing
                                               # long prompts is the bottleneck;
                                               # this is the main speed knob.

# Context-length buckets (tokens). Each generator's quota is split evenly here.
read -r -a LENGTHS <<< "${CONTEXT_LENGTHS:-8192 32768 131072 163840}"
NB=${#LENGTHS[@]}

# Per-generator TOTAL quotas (sum = 1,000,000).
RULER_TOTAL="${RULER_TOTAL:-400000}"
UNTIE_TOTAL="${UNTIE_TOTAL:-300000}"
MRCR_TOTAL="${MRCR_TOTAL:-150000}"
IDK_TOTAL="${IDK_TOTAL:-150000}"
N_RULER_TASKS=$("$PYTHON" -m ruler_pp --list | wc -l)
CLUSTER_DIR="${CLUSTER_DIR:-$OUT/clusters}"
CLUSTER_MODEL="${CLUSTER_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
RUN_LLM_INFER="${RUN_LLM_INFER:-0}"

# untie pool caps (raise for more variety; the reuse mechanism tops up to the
# per-flow target by re-chunking/re-shuffling under fresh seeds). NOTE: the
# cluster source loads member texts into RAM — see the memory note in README.
UNTIE_MAX_CLUSTERS="${UNTIE_MAX_CLUSTERS:-20000}"
UNTIE_MAX_SINGLE="${UNTIE_MAX_SINGLE:-20000}"
UNTIE_MAX_REUSE="${UNTIE_MAX_REUSE:-100}"

# LLM-in-the-loop prepare (NOT part of the 1M): docs to prepare per length bucket.
MULTIHOP_DOCS="${MULTIHOP_DOCS:-5000}"
SUMMARIZE_DOCS="${SUMMARIZE_DOCS:-5000}"
SUMMARIZE_WORDS="${SUMMARIZE_WORDS:-300}"

# Stage toggles.
RUN_RULER="${RUN_RULER:-1}"
RUN_UNTIE="${RUN_UNTIE:-1}"
RUN_MRCR="${RUN_MRCR:-1}"
RUN_IDK="${RUN_IDK:-1}"
RUN_LLM_PREP="${RUN_LLM_PREP:-1}"

# Per-bucket sample counts (integer division; ~rounding is negligible at 1M).
RULER_PER=$(( RULER_TOTAL / NB / N_RULER_TASKS ))   # per task, per bucket
UNTIE_PER=$(( UNTIE_TOTAL / NB / 2 ))               # per flow, per bucket
MRCR_PER=$((  MRCR_TOTAL  / NB ))                   # per bucket
IDK_PER=$((   IDK_TOTAL   / NB ))                   # per bucket

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# Only successful stages receive a completion marker. A partial directory is
# retried. Use a fresh OUT when changing settings; these are not cache keys.
needs_gen() {
  [ ! -f "$1/.complete" ]
}
for spec in "$RUN_RULER:$RULER_PER" "$RUN_UNTIE:$UNTIE_PER" "$RUN_MRCR:$MRCR_PER" "$RUN_IDK:$IDK_PER"; do
  if [[ "$spec" == 1:* && "${spec#*:}" -lt 1 ]]; then
    echo "Enabled generator quota rounds to zero; raise its total or reduce CONTEXT_LENGTHS" >&2
    exit 1
  fi
done

log "config"
cat <<EOF
  python        : $PYTHON   (emb: $EMB_PYTHON)
  input_dir     : $INPUT_DIR
  out           : $OUT
  format        : $FORMAT
  tokenizer     : $TOKENIZER
  workers       : $WORKERS (0 = all cores)   hashseed: $PYTHONHASHSEED
  lengths       : ${LENGTHS[*]}
  quotas        : ruler=$RULER_TOTAL untie=$UNTIE_TOTAL mrcr=$MRCR_TOTAL idk=$IDK_TOTAL
  per-bucket    : ruler/task=$RULER_PER  untie/flow=$UNTIE_PER  mrcr=$MRCR_PER  idk=$IDK_PER
EOF
mkdir -p "$OUT"

# --------------------------------------------------------------------------- #
# 1) ruler++ synthetic — 22 tasks x 4 buckets
# --------------------------------------------------------------------------- #
if [ "$RUN_RULER" = "1" ]; then
  for i in "${!LENGTHS[@]}"; do
    L="${LENGTHS[$i]}"
    if ! needs_gen "$OUT/ruler_pp/$L"; then
      log "ruler++  len=$L  -> completed, skipping"
      continue
    fi
    log "ruler++  len=$L  num_samples/task=$RULER_PER"
    "$PYTHON" -m ruler_pp \
      --task all \
      --max_seq_length "$L" \
      --num_samples "$RULER_PER" \
      --format "$FORMAT" \
      --tokenizer "$TOKENIZER" \
      --workers "$WORKERS" \
      --seed "$(( SEED + i ))" \
      --output_dir "$OUT/ruler_pp/$L"
    touch "$OUT/ruler_pp/$L/.complete"
  done
fi

# --------------------------------------------------------------------------- #
# 2) untie — needs clusters first; then 2 flows x 4 buckets
#    The paragraph-unit size is scaled to the bucket (~L/8) so even the small
#    buckets yield >= 2 units per sample.
# --------------------------------------------------------------------------- #
if [ "$RUN_UNTIE" = "1" ]; then
  if [ ! -f "$CLUSTER_DIR/clusters.parquet" ]; then
    log "clustering corpus (prerequisite for untie's cluster source)"
    "$EMB_PYTHON" -m corpus.cluster --input_dir "$INPUT_DIR" --output_dir "$CLUSTER_DIR" --model "$CLUSTER_MODEL"
  else
    log "clusters already present ($CLUSTER_DIR/clusters.parquet) — skipping"
  fi

  for i in "${!LENGTHS[@]}"; do
    L="${LENGTHS[$i]}"
    if ! needs_gen "$OUT/untie/$L"; then
      log "untie  context_len=$L  -> completed, skipping"
      continue
    fi
    TPT=$(( L / 8 )); [ "$TPT" -lt 512 ] && TPT=512   # paragraph-unit tokens
    log "untie  context_len=$L  num_samples/flow=$UNTIE_PER  target_para_tokens=$TPT"
    "$PYTHON" -m untie \
      --input_dir "$INPUT_DIR" \
      --clusters "$CLUSTER_DIR/clusters.parquet" \
      --output_dir "$OUT/untie/$L" \
      --context_len "$L" \
      --flows reconstruct,permutation \
      --sources cluster,single \
      --format "$FORMAT" \
      --tokenizer "$TOKENIZER" \
      --target_para_tokens "$TPT" \
      --num_samples "$UNTIE_PER" \
      --max_clusters "$UNTIE_MAX_CLUSTERS" \
      --max_single_docs "$UNTIE_MAX_SINGLE" \
      --max_reuse "$UNTIE_MAX_REUSE" \
      --workers "$WORKERS" \
      --seed "$(( SEED + i ))"
    touch "$OUT/untie/$L/.complete"
  done
fi

# --------------------------------------------------------------------------- #
# 3) mrcr — marker retrieval, 4 buckets
# --------------------------------------------------------------------------- #
if [ "$RUN_MRCR" = "1" ]; then
  for i in "${!LENGTHS[@]}"; do
    L="${LENGTHS[$i]}"
    if ! needs_gen "$OUT/mrcr/$L"; then
      log "mrcr  max_seq_length=$L  -> completed, skipping"
      continue
    fi
    log "mrcr  max_seq_length=$L  num_samples=$MRCR_PER"
    "$PYTHON" -m mrcr \
      --input_dir "$INPUT_DIR" \
      --output_dir "$OUT/mrcr/$L" \
      --max_seq_length "$L" \
      --num_samples "$MRCR_PER" \
      --format "$FORMAT" \
      --tokenizer "$TOKENIZER" \
      --workers "$WORKERS" \
      --seed "$(( SEED + i ))"
    touch "$OUT/mrcr/$L/.complete"
  done
fi

# --------------------------------------------------------------------------- #
# 4) idk — abstention MCQ, 4 buckets
# --------------------------------------------------------------------------- #
if [ "$RUN_IDK" = "1" ]; then
  for i in "${!LENGTHS[@]}"; do
    L="${LENGTHS[$i]}"
    if ! needs_gen "$OUT/idk/$L"; then
      log "idk  max_seq_length=$L  -> completed, skipping"
      continue
    fi
    log "idk  max_seq_length=$L  num_samples=$IDK_PER"
    "$PYTHON" -m idk \
      --input_dir "$INPUT_DIR" \
      --output_dir "$OUT/idk/$L" \
      --max_seq_length "$L" \
      --num_samples "$IDK_PER" \
      --format "$FORMAT" \
      --tokenizer "$TOKENIZER" \
      --workers "$WORKERS" \
      --seed "$(( SEED + i ))"
    touch "$OUT/idk/$L/.complete"
  done
fi

# --------------------------------------------------------------------------- #
# 5) LLM-in-the-loop PREPARE (multihop + summarize) — emits requests only.
#    Docs are partitioned into the length buckets by doc length so a doc lands
#    in the bucket matching its size (min_doc_tokens = previous bucket's cap).
#    After inference:  python -m multihop  assemble --requests_dir <dir> --responses <file> --format sft
#                      python -m summarize assemble --requests_dir <dir> --responses <file> --format sft
# --------------------------------------------------------------------------- #
if [ "$RUN_LLM_PREP" = "1" ]; then
  for i in "${!LENGTHS[@]}"; do
    L="${LENGTHS[$i]}"
    if [ "$i" = "0" ]; then MIN_MH=2000; MIN_SUM=2000; else
      MIN_MH=$(( LENGTHS[i-1] - 2048 + 1 ))
      MIN_SUM=$(( LENGTHS[i-1] - 1024 + 1 ))
    fi

    if needs_gen "$OUT/multihop/$L"; then
      log "multihop prepare  bucket=$L  min_doc_tokens=$MIN_MH  docs=$MULTIHOP_DOCS"
      "$PYTHON" -m multihop prepare \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUT/multihop/$L" \
        --num_docs "$MULTIHOP_DOCS" \
        --min_doc_tokens "$MIN_MH" \
        --max_seq_length "$L" \
        --tokenizer "$TOKENIZER" \
        --seed "$(( SEED + i ))"
      touch "$OUT/multihop/$L/.complete"
    else
      log "multihop prepare  bucket=$L  -> completed, skipping"
    fi

    if needs_gen "$OUT/summarize/$L"; then
      log "summarize prepare  bucket=$L  min_doc_tokens=$MIN_SUM  docs=$SUMMARIZE_DOCS"
      "$PYTHON" -m summarize prepare \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUT/summarize/$L" \
        --num_docs "$SUMMARIZE_DOCS" \
        --min_doc_tokens "$MIN_SUM" \
        --max_seq_length "$L" \
        --summary_words "$SUMMARIZE_WORDS" \
        --tokenizer "$TOKENIZER" \
        --seed "$(( SEED + i ))"
      touch "$OUT/summarize/$L/.complete"
    else
      log "summarize prepare  bucket=$L  -> completed, skipping"
    fi
  done
fi

# Optional small-run inference. Large runs may continue to use a dispatcher.
if [ "$RUN_LLM_INFER" = "1" ]; then
  : "${API_BASE_URL:?set API_BASE_URL including /v1}"
  : "${API_MODEL:?set API_MODEL}"
  for gen in multihop summarize; do
    for L in "${LENGTHS[@]}"; do
      bundle="$OUT/$gen/$L"
      if [ ! -f "$bundle/.assembled" ]; then
        "$PYTHON" -m corpus.infer --requests "$bundle/requests.jsonl" \
          --output "$bundle/responses.jsonl" --base_url "$API_BASE_URL" --model "$API_MODEL" \
          --max_tokens "${API_MAX_TOKENS:-1024}" \
          --chat_template_kwargs "${API_CHAT_TEMPLATE_KWARGS:-null}"
        "$PYTHON" -m "$gen" assemble --requests_dir "$bundle" \
          --responses "$bundle/responses.jsonl" --format "$FORMAT"
        touch "$bundle/.assembled"
      fi
    done
  done
fi

# --------------------------------------------------------------------------- #
# tally — count offline samples produced (excludes multihop/summarize requests)
# --------------------------------------------------------------------------- #
log "tally (offline samples)"
total=0
for gen in ruler_pp untie mrcr idk; do
  [ -d "$OUT/$gen" ] || continue
  # untie/multihop/summarize also drop non-sample files; count only *.jsonl that
  # are final samples. ruler/untie/mrcr/idk write only sample jsonl here.
  n=$(find "$OUT/$gen" -name '*.jsonl' -exec cat {} + 2>/dev/null | wc -l)
  printf '  %-10s %12d\n' "$gen" "$n"
  total=$(( total + n ))
done
printf '  %-10s %12d\n' "TOTAL" "$total"

if [ "$RUN_LLM_PREP" = "1" ]; then
  reqs=$(find "$OUT/multihop" "$OUT/summarize" -name 'requests.jsonl' -exec cat {} + 2>/dev/null | wc -l)
  printf '\n  LLM requests prepared (multihop+summarize): %d\n' "$reqs"
  printf '  -> run these through your engine, then `assemble` per dir.\n'
fi

log "done -> $OUT"
