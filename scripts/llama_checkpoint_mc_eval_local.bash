#!/bin/bash
set -euo pipefail

# Resolve project root regardless of launch directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Activate virtual environment
source "${PROJECT_ROOT}/.venv/bin/activate"

# Script assumes it is launched from VON-SOAP root
cd "${PROJECT_ROOT}/Minimalist_LLM_Pretraining"

# Required: checkpoint directory
: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a saved model directory}"

resolve_from_root() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    printf "%s" "$p"
  else
    printf "%s" "${PROJECT_ROOT}/${p}"
  fi
}

CHECKPOINT_DIR="$(resolve_from_root "${CHECKPOINT_DIR}")"
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Checkpoint directory not found: ${CHECKPOINT_DIR}" >&2
  exit 1
fi

OPTIMIZER_PATH="${OPTIMIZER_PATH:-}"
if [[ -z "${OPTIMIZER_PATH}" ]]; then
  CANDIDATE_OPT="${CHECKPOINT_DIR}/optimizer.pt"
  if [[ -f "${CANDIDATE_OPT}" ]]; then
    OPTIMIZER_PATH="${CANDIDATE_OPT}"
  fi
fi
if [[ -n "${OPTIMIZER_PATH}" ]]; then
  OPTIMIZER_PATH="$(resolve_from_root "${OPTIMIZER_PATH}")"
fi

# Optional knobs
MC_SAMPLES="${MC_SAMPLES:-10}"
MC_SAMPLES_LIST="${MC_SAMPLES_LIST:-}"
VAL_TOKENS="${VAL_TOKENS:-10000000}"
BATCH_SIZE="${BATCH_SIZE:-}"
MAX_LENGTH="${MAX_LENGTH:-}"
DATASET_PATH="${DATASET_PATH:-}"
HF_DATASET="${HF_DATASET:-false}"
VAL_FILES="${VAL_FILES:-}"
DEVICE="${DEVICE:-cuda}"
CSV_OUT="${CSV_OUT:-}"
APPEND_CSV="${APPEND_CSV:-false}"
RUN_LABEL="${RUN_LABEL:-}"
PROGRESS="${PROGRESS:-true}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"
QKV_MODE="${QKV_MODE:-}"
ATTN_RATIO="${ATTN_RATIO:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

if [[ -n "${DATASET_PATH}" ]]; then
  DATASET_PATH="$(resolve_from_root "${DATASET_PATH}")"
fi

if [[ -n "${CSV_OUT}" ]]; then
  CSV_OUT="$(resolve_from_root "${CSV_OUT}")"
fi

args=(
  --checkpoint_dir "${CHECKPOINT_DIR}"
  --mc_samples "${MC_SAMPLES}"
  --val_tokens "${VAL_TOKENS}"
  --device "${DEVICE}"
  --append_csv "${APPEND_CSV}"
  --progress "${PROGRESS}"
  --progress_every "${PROGRESS_EVERY}"
)

if [[ -n "${OPTIMIZER_PATH}" ]]; then
  args+=(--optimizer_path "${OPTIMIZER_PATH}")
fi

if [[ -n "${MC_SAMPLES_LIST}" ]]; then
  args+=(--mc_samples_list "${MC_SAMPLES_LIST}")
fi

if [[ -n "${BATCH_SIZE}" ]]; then
  args+=(--batch_size "${BATCH_SIZE}")
fi

if [[ -n "${MAX_LENGTH}" ]]; then
  args+=(--max_length "${MAX_LENGTH}")
fi

if [[ -n "${DATASET_PATH}" ]]; then
  args+=(--dataset_path "${DATASET_PATH}")
fi

if [[ "${HF_DATASET}" == "true" || "${HF_DATASET}" == "1" ]]; then
  args+=(--hf_dataset)
fi

if [[ -n "${VAL_FILES}" ]]; then
  args+=(--val_files "${VAL_FILES}")
fi

if [[ -n "${CSV_OUT}" ]]; then
  args+=(--csv_out "${CSV_OUT}")
fi

if [[ -n "${RUN_LABEL}" ]]; then
  args+=(--run_label "${RUN_LABEL}")
fi

if [[ -n "${QKV_MODE}" ]]; then
  args+=(--qkv_mode "${QKV_MODE}")
fi

if [[ -n "${ATTN_RATIO}" ]]; then
  args+=(--attn_ratio "${ATTN_RATIO}")
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  python -m torch.distributed.run --standalone --nproc_per_node "${NPROC_PER_NODE}" \
    eval_llama_checkpoint_mc_loss.py "${args[@]}"
else
  python eval_llama_checkpoint_mc_loss.py "${args[@]}"
fi
