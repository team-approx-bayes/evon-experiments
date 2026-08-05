#!/usr/bin/env bash

# Sweep all datasets for fine-tuning with a specific configuration of seed, GPU, and optimizer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration (can be overridden via environment variables)
OPTIM="${OPTIM:-soap}"
SEED="${SEED:-43}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU:-0}}"
export CUDA_VISIBLE_DEVICES

# Find datasets dynamically from best_hparams directory if available
if [ -d "${SCRIPT_DIR}/best_hparams" ]; then
    DATASETS=($(find "${SCRIPT_DIR}/best_hparams" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort))
else
    DATASETS=("cars" "dtd" "eurosat" "gtsrb" "mnist" "resisc45" "sun397" "svhn")
fi

echo "======================================================="
echo " Starting Fine-Tuning Datasets Sweep"
echo " Datasets (${#DATASETS[@]}): ${DATASETS[*]}"
echo " Optimizer: ${OPTIM} | Seed: ${SEED} | GPU: ${CUDA_VISIBLE_DEVICES}"
echo "======================================================="

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo ">>> Running fine-tuning for dataset: ${DATASET} <<<"
    DATASET="$DATASET" OPTIM="$OPTIM" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" SEED="$SEED" bash "${SCRIPT_DIR}/finetune_best_hparams.sh"
done

echo ""
echo "======================================================="
echo " Fine-Tuning Sweep Complete!"
echo "======================================================="
