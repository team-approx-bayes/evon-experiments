#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3

# ── Configuration
MODEL_NAME="${MODEL_NAME:-vit_base_patch16_224.openai_clip}"
DATASET="${DATASET:-mnist}"
SEED="${SEED:-42}"
OPTIM="${OPTIM:-evon}"
BATCH_SIZE="${BATCH_SIZE:-128}"
TRAINABLE_SCOPE="${TRAINABLE_SCOPE:-linear}"
WANDB_PROJECT="${WANDB_PROJECT:-clip-finetune-best-hparams}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WHITEN_PREC_GRAD="${WHITEN_PREC_GRAD:-}"
ENABLE_PHASING="${ENABLE_PHASING:-}"
PRICE_CLIP_RATIO="${PRICE_CLIP_RATIO:-}"
DISABLE_CHECKPOINTING="${DISABLE_CHECKPOINTING:-}"

# Change to the directory where this script is located
cd "$(dirname "$0")" || exit 1

LOG_FILE="logs/finetune_best_hparams.log"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

# Activate virtual environment (if one exists under parent directory or globally setup)
# Usually we use the virtual environment activated before submitting, or a global/project env.
if [ -d "../.venv" ]; then
    unset PYTHONPATH
    VIRTUAL_ENV="$(cd ../.venv && pwd)"
    export VIRTUAL_ENV
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    source "$VIRTUAL_ENV/bin/activate"
elif [ -d "../../.venv" ]; then
    unset PYTHONPATH
    VIRTUAL_ENV="$(cd ../../.venv && pwd)"
    export VIRTUAL_ENV
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    source "$VIRTUAL_ENV/bin/activate"
fi

CMD=(
    bash run_finetune.sh
    --model-name "$MODEL_NAME"
    --dataset-name "$DATASET"
    --seed "$SEED"
    --optim "$OPTIM"
    --batch-size "$BATCH_SIZE"
    --trainable-scope "$TRAINABLE_SCOPE"
)

if [[ -n "${WANDB_PROJECT:-}" ]]; then
    CMD+=(--wandb-project "$WANDB_PROJECT")
fi

if [[ -n "${WANDB_ENTITY:-}" ]]; then
    CMD+=(--wandb-entity "$WANDB_ENTITY")
fi

if [[ -n "${WHITEN_PREC_GRAD:-}" ]]; then
    CMD+=(--optim-cfg-override "whiten_grad=$WHITEN_PREC_GRAD")
fi

if [[ -n "${ENABLE_PHASING:-}" ]]; then
    CMD+=(--optim-cfg-override "enable_alternating_grads=$ENABLE_PHASING")
fi

if [[ -n "${PRICE_CLIP_RATIO:-}" ]]; then
    CMD+=(--optim-cfg-override "price_clip_ratio=$PRICE_CLIP_RATIO")
fi

if [[ -n "${DISABLE_CHECKPOINTING:-}" ]]; then
    CMD+=(--disable-checkpointing)
fi

"${CMD[@]}"

echo "Job completed."
