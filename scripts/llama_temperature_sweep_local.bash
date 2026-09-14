#!/bin/bash
set -euo pipefail

# Two-phase posterior temperature selection for LLaMA MC-BMA eval:
#   Phase 1: sweep TEMPERATURE_LIST on SWEEP_TOKENS tokens of a TRAIN shard.
#   Phase 2: pick the T with the lowest mc_bma_nll at the BEST_MC budget.
#   Phase 3: re-run the winner on the default validation shards.
# Wraps scripts/llama_checkpoint_mc_eval_local.bash, so all of that script's
# env knobs (CHECKPOINT_DIR, OPTIMIZER_PATH, DATASET_PATH, BATCH_SIZE,
# MAX_LENGTH, DEVICE, NPROC_PER_NODE, ...) pass through unchanged.

# Resolve project root regardless of launch directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a saved model directory}"

# Phase-1 (train sweep) knobs
TEMPERATURE_LIST="${TEMPERATURE_LIST:-1,2,4,8}"
MC_SAMPLES_LIST="${MC_SAMPLES_LIST:-1,4,16,32}"
SWEEP_TOKENS="${SWEEP_TOKENS:-2000000}"
TRAIN_VAL_FILES="${TRAIN_VAL_FILES:-c4-train.00000-of-01024.json.gz}"
BEST_MC="${BEST_MC:-}"   # defaults to the largest MC in MC_SAMPLES_LIST

# Phase-3 (validation) knobs
VAL_TOKENS="${VAL_TOKENS:-10000000}"
VAL_MC_SAMPLES_LIST="${VAL_MC_SAMPLES_LIST:-${MC_SAMPLES_LIST}}"

# Shared knobs
RUN_LABEL="${RUN_LABEL:-temperature_sweep}"
OUT_DIR="${OUT_DIR:-results/temperature_sweep}"
if [[ "${OUT_DIR}" != /* ]]; then
  OUT_DIR="${PROJECT_ROOT}/${OUT_DIR}"
fi
mkdir -p "${OUT_DIR}"

PY="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  PY="$(command -v python3)"
fi

# BEST_MC must be one of the swept MC budgets; default: largest in the list.
if [[ -z "${BEST_MC}" ]]; then
  BEST_MC="$(printf '%s' "${MC_SAMPLES_LIST}" | tr ',' '\n' | sort -n | tail -1)"
fi
if ! printf '%s' "${MC_SAMPLES_LIST}" | tr ',' '\n' | grep -qx "${BEST_MC}"; then
  echo "BEST_MC=${BEST_MC} is not in MC_SAMPLES_LIST=${MC_SAMPLES_LIST}" >&2
  exit 1
fi

SWEEP_CSV="${OUT_DIR}/${RUN_LABEL}_train_sweep.csv"
VAL_CSV="${OUT_DIR}/${RUN_LABEL}_validation.csv"

echo "=== Phase 1: temperature sweep on train shard ==="
echo "    T in {${TEMPERATURE_LIST}}, MC in {${MC_SAMPLES_LIST}}, ${SWEEP_TOKENS} tokens from ${TRAIN_VAL_FILES}"
echo "    csv: ${SWEEP_CSV}"
VAL_FILES="${TRAIN_VAL_FILES}" \
VAL_TOKENS="${SWEEP_TOKENS}" \
TEMPERATURE_LIST="${TEMPERATURE_LIST}" \
MC_SAMPLES_LIST="${MC_SAMPLES_LIST}" \
CSV_OUT="${SWEEP_CSV}" \
APPEND_CSV="${APPEND_CSV:-false}" \
RUN_LABEL="${RUN_LABEL}_sweep" \
CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
  bash "${SCRIPT_DIR}/llama_checkpoint_mc_eval_local.bash"

echo "=== Phase 2: pick best temperature at MC=${BEST_MC} ==="
BEST_T="$("${PY}" "${SCRIPT_DIR}/pick_best_temperature.py" \
  "${SWEEP_CSV}" "${RUN_LABEL}_sweep" "${BEST_MC}")"
echo "    best temperature: ${BEST_T}"

echo "=== Phase 3: validation at T=${BEST_T} ==="
echo "    MC in {${VAL_MC_SAMPLES_LIST}}, ${VAL_TOKENS} tokens on default validation shards"
echo "    csv: ${VAL_CSV}"
VAL_FILES="" \
VAL_TOKENS="${VAL_TOKENS}" \
TEMPERATURE="${BEST_T}" \
MC_SAMPLES_LIST="${VAL_MC_SAMPLES_LIST}" \
CSV_OUT="${VAL_CSV}" \
APPEND_CSV="${APPEND_CSV:-false}" \
RUN_LABEL="${RUN_LABEL}_val_T${BEST_T}" \
CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
  bash "${SCRIPT_DIR}/llama_checkpoint_mc_eval_local.bash"

echo "=== Done ==="
echo "    best T=${BEST_T} (train, MC=${BEST_MC}); validation rows in ${VAL_CSV}"
