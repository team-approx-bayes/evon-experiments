#!/usr/bin/env bash

# Helper script to run CLIP finetuning using best downloaded hyperparameters.

set -euo pipefail

usage() {
    echo "Usage: $0 --model-name <str> (--dataset-name <name> ...) --seed <int> --optim {adamw,sgd,evon,ivon,soap} --batch-size <int> --trainable-scope {all,linear} [--best-hparams-dir <dir>] [--wandb-project <proj>] [--wandb-entity <ent>] [--optim-cfg-override key=value]..."
    exit 1
}

MODEL_NAME=""
DATASET_ARGS=()
SEED=""
OPTIM=""
BATCH_SIZE=""
TRAINABLE_SCOPE=""
BEST_HPARAMS_DIR="best_hparams"
WANDB_PROJECT=""
WANDB_ENTITY=""
OPTIM_CFG_OVERRIDES=()
DISABLE_CHECKPOINTING=""

if [ -d "../.venv" ]; then
    unset PYTHONPATH
    source ../.venv/bin/activate
elif [ -d "../../.venv" ]; then
    unset PYTHONPATH
    source ../../.venv/bin/activate
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-name)    MODEL_NAME="$2";    shift 2 ;;
        --dataset-name)  DATASET_ARGS+=("$2"); shift 2 ;;
        --dataset-names)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                DATASET_ARGS+=("$1")
                shift
            done
            ;;
        --seed)          SEED="$2";          shift 2 ;;
        --optim)         OPTIM="$2";         shift 2 ;;
        --batch-size)    BATCH_SIZE="$2";    shift 2 ;;
        --trainable-scope) TRAINABLE_SCOPE="$2"; shift 2 ;;
        --best-hparams-dir) BEST_HPARAMS_DIR="$2"; shift 2 ;;
        --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
        --wandb-entity)  WANDB_ENTITY="$2";  shift 2 ;;
        --disable-checkpointing)
            DISABLE_CHECKPOINTING=true
            shift
            ;;
        --optim-cfg-override)
            OPTIM_CFG_OVERRIDES+=("$2")
            shift 2
            ;;
        -h|--help)       usage ;;
        *)               echo "Unknown arg: $1"; usage ;;
    esac
done

if [[ -z "$MODEL_NAME" || "${#DATASET_ARGS[@]}" -eq 0 || -z "$SEED" || -z "$OPTIM" || -z "$BATCH_SIZE" || -z "$TRAINABLE_SCOPE" ]]; then
    echo "Error: all arguments are required."
    usage
fi

# Normalize dataset names
DATASETS=()
for DATASET_ARG in "${DATASET_ARGS[@]}"; do
    IFS=',' read -ra SPLIT_DATASETS <<< "$DATASET_ARG"
    for DATASET_NAME in "${SPLIT_DATASETS[@]}"; do
        if [[ -n "$DATASET_NAME" ]]; then
            DATASETS+=("$DATASET_NAME")
        fi
    done
done

for DATASET_NAME in "${DATASETS[@]}"; do
    CONFIG_FILE="${BEST_HPARAMS_DIR}/${DATASET_NAME}/${OPTIM}/config.yaml"

    echo "=========================================="
    echo "  CLIP finetune with best hparams"
    echo "  model=$MODEL_NAME dataset=$DATASET_NAME seed=$SEED optim=$OPTIM"
    echo "  config=$CONFIG_FILE"
    echo "=========================================="

    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "Error: Config file $CONFIG_FILE not found."
        exit 1
    fi

    CMD=(
        python finetune.py
        --model-name "$MODEL_NAME"
        --dataset-name "$DATASET_NAME"
        --seed "$SEED"
        --optim "$OPTIM"
        --batch-size "$BATCH_SIZE"
        --trainable-scope "$TRAINABLE_SCOPE"
        --config-file "$CONFIG_FILE"
    )

    if [[ -n "$WANDB_PROJECT" ]]; then
        CMD+=(--wandb-project "$WANDB_PROJECT")
    fi
    if [[ -n "$WANDB_ENTITY" ]]; then
        CMD+=(--wandb-entity "$WANDB_ENTITY")
    fi
    if [[ "${DISABLE_CHECKPOINTING:-}" == "true" ]]; then
        CMD+=(--disable-checkpointing)
    fi

    for OVERRIDE in "${OPTIM_CFG_OVERRIDES[@]}"; do
        CMD+=(--optim-cfg-override "$OVERRIDE")
    done

    "${CMD[@]}"

done

echo "=========================================="
echo "  Done."
echo "=========================================="
