#!/usr/bin/env bash

# Sweep Laplace Approximation evaluation across datasets found in checkpoints.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="${MODEL_NAME:-vit_base_patch16_224.openai_clip}"
SEED="${1:-${SEED:-42}}"
OPTIM="${OPTIM:-soap}"
TRAINABLE_SCOPE="${TRAINABLE_SCOPE:-linear}"
HESSIAN_STRUCTURE="${HESSIAN_STRUCTURE:-kfac}"
EVAL_METHOD="${EVAL_METHOD:-probit}"
WANDB_PROJECT="${WANDB_PROJECT:-laplace-clip}"

CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints/${MODEL_NAME}/${SEED}/${OPTIM}"

# Find datasets dynamically from checkpoint folder if available
if [ -d "$CHECKPOINT_DIR" ]; then
    DATASETS=($(find "$CHECKPOINT_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort))
else
    DATASETS=("cars" "dtd" "eurosat" "gtsrb" "mnist" "resisc45" "sun397" "svhn")
fi

echo "======================================================="
echo " Starting Laplace Approximation Sweep"
echo " Datasets (${#DATASETS[@]}): ${DATASETS[*]}"
echo " Model: ${MODEL_NAME}"
echo " Optimizer: ${OPTIM} | Seed: ${SEED} | Scope: ${TRAINABLE_SCOPE}"
echo " Hessian: ${HESSIAN_STRUCTURE} | Eval Method: ${EVAL_METHOD} | WandB Project: ${WANDB_PROJECT}"
echo "======================================================="

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo ">>> Running Laplace for dataset: ${DATASET} <<<"
    "${SCRIPT_DIR}/run_laplace.sh" \
        --model-name "$MODEL_NAME" \
        --dataset-name "$DATASET" \
        --seed "$SEED" \
        --optim "$OPTIM" \
        --trainable-scope "$TRAINABLE_SCOPE" \
        --hessian-structure "$HESSIAN_STRUCTURE" \
        --eval-method "$EVAL_METHOD" \
        --wandb-project "$WANDB_PROJECT"
done

echo ""
echo "======================================================="
echo " Laplace Sweep Complete!"
echo "======================================================="
