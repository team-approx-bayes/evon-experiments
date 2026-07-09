#!/bin/bash
set -euo pipefail

# Resolve project root regardless of launch directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Activate virtual environment
source "${PROJECT_ROOT}/.venv/bin/activate"

# Script assumes it is launched from VON-SOAP root
cd "${PROJECT_ROOT}/modded-nanogpt"

# Required: checkpoint path
: "${CHECKPOINT:?Set CHECKPOINT to a checkpoint .pt path}"

resolve_from_root() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    printf "%s" "$p"
  else
    printf "%s" "${PROJECT_ROOT}/${p}"
  fi
}

CHECKPOINT="$(resolve_from_root "${CHECKPOINT}")"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

# Optional knobs
MC_SAMPLES="${MC_SAMPLES:-10}"
MC_SAMPLES_LIST="${MC_SAMPLES_LIST:-}"
VAL_TOKENS="${VAL_TOKENS:-10485760}"
BATCH_SIZE_PRE_GPU="${BATCH_SIZE_PRE_GPU:-16}"
INPUT_VAL_BIN="${INPUT_VAL_BIN:-}"
DEVICE="${DEVICE:-cuda}"
CSV_OUT="${CSV_OUT:-}"
APPEND_CSV="${APPEND_CSV:-false}"
RUN_LABEL="${RUN_LABEL:-}"
PROGRESS="${PROGRESS:-true}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

if [[ -n "${INPUT_VAL_BIN}" ]]; then
  INPUT_VAL_BIN="$(resolve_from_root "${INPUT_VAL_BIN}")"
fi

if [[ -n "${CSV_OUT}" ]]; then
  CSV_OUT="$(resolve_from_root "${CSV_OUT}")"
fi

args=(
  --checkpoint "${CHECKPOINT}"
  --mc_samples "${MC_SAMPLES}"
  --val_tokens "${VAL_TOKENS}"
  --batch_size_pre_gpu "${BATCH_SIZE_PRE_GPU}"
  --device "${DEVICE}"
  --append_csv "${APPEND_CSV}"
  --progress "${PROGRESS}"
  --progress_every "${PROGRESS_EVERY}"
)

if [[ -n "${MC_SAMPLES_LIST}" ]]; then
  args+=(--mc_samples_list "${MC_SAMPLES_LIST}")
fi

if [[ -n "${INPUT_VAL_BIN}" ]]; then
  args+=(--input_val_bin "${INPUT_VAL_BIN}")
fi

if [[ -n "${CSV_OUT}" ]]; then
  args+=(--csv_out "${CSV_OUT}")
fi

if [[ -n "${RUN_LABEL}" ]]; then
  args+=(--run_label "${RUN_LABEL}")
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  python -m torch.distributed.run --standalone --nproc_per_node "${NPROC_PER_NODE}" eval_checkpoint_mc_loss.py "${args[@]}"
else
  python eval_checkpoint_mc_loss.py "${args[@]}"
fi
