#!/usr/bin/env bash

# Helper script to run Laplace Approximation evaluation on fine-tuned CLIP checkpoints.

set -euo pipefail

usage() {
    echo "Usage: $0 --model-name <str> (--dataset-name <name> ...) --seed <int> --optim {adamw,sgd,evon,ivon,soap} --batch-size <int> --trainable-scope {all,linear} [--checkpoint-path <path>] [--epoch <int>] [--prior-precision <float>] [--hessian-structure {diag,kfac}] [--wandb-project <proj>] [--wandb-entity <ent>]"
    exit 1
}

MODEL_NAME=""
DATASET_ARGS=()
SEED=""
OPTIM=""
BATCH_SIZE="32"
TRAINABLE_SCOPE=""
CHECKPOINT_PATH=""
EPOCH=""
PRIOR_PRECISION="400.0"
VAR_SCALE="1.0"
EVAL_METHOD="both"
HESSIAN_STRUCTURE="kfac"
HESSIAN_BATCH_SIZE="5"
OPTIMIZE_PRIOR_PRECISION="true"
WANDB_PROJECT=""
WANDB_ENTITY=""

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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-name)                   MODEL_NAME="$2";                shift 2 ;;
        --dataset-name)                 DATASET_ARGS+=("$2");           shift 2 ;;
        --dataset-names)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                DATASET_ARGS+=("$1")
                shift
            done
            ;;
        --seed)                         SEED="$2";                      shift 2 ;;
        --optim)                        OPTIM="$2";                     shift 2 ;;
        --batch-size)                   BATCH_SIZE="$2";                shift 2 ;;
        --trainable-scope)              TRAINABLE_SCOPE="$2";           shift 2 ;;
        --checkpoint-path)              CHECKPOINT_PATH="$2";           shift 2 ;;
        --epoch)                        EPOCH="$2";                     shift 2 ;;
        --prior-precision)              PRIOR_PRECISION="$2";           shift 2 ;;
        --optimize-prior-precision)     OPTIMIZE_PRIOR_PRECISION="true"; shift ;;
        --no-optimize-prior-precision)  OPTIMIZE_PRIOR_PRECISION="false"; shift ;;
        --var-scale)                    VAR_SCALE="$2";                 shift 2 ;;
        --eval-method)                  EVAL_METHOD="$2";               shift 2 ;;
        --hessian-structure)            HESSIAN_STRUCTURE="$2";         shift 2 ;;
        --hessian-batch-size)           HESSIAN_BATCH_SIZE="$2";        shift 2 ;;
        --wandb-project)                WANDB_PROJECT="$2";             shift 2 ;;
        --wandb-entity)                 WANDB_ENTITY="$2";              shift 2 ;;
        -h|--help)                      usage ;;
        *)                              echo "Unknown arg: $1"; usage ;;
    esac
done

if [[ -z "$MODEL_NAME" || "${#DATASET_ARGS[@]}" -eq 0 || -z "$SEED" || -z "$OPTIM" || -z "$TRAINABLE_SCOPE" ]]; then
    echo "Error: missing required arguments."
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
    echo "=========================================="
    echo "  CLIP Laplace Approximation Evaluation"
    echo "  model=$MODEL_NAME dataset=$DATASET_NAME seed=$SEED optim=$OPTIM"
    echo "  hessian_structure=$HESSIAN_STRUCTURE hessian_batch_size=$HESSIAN_BATCH_SIZE prior_precision=$PRIOR_PRECISION optimize_prior=$OPTIMIZE_PRIOR_PRECISION var_scale=$VAR_SCALE eval_method=$EVAL_METHOD"
    echo "=========================================="

    CMD=(
        python laplace_approximation.py
        --model-name "$MODEL_NAME"
        --dataset-name "$DATASET_NAME"
        --seed "$SEED"
        --optim "$OPTIM"
        --batch-size "$BATCH_SIZE"
        --trainable-scope "$TRAINABLE_SCOPE"
        --prior-precision "$PRIOR_PRECISION"
        --var-scale "$VAR_SCALE"
        --eval-method "$EVAL_METHOD"
        --hessian-structure "$HESSIAN_STRUCTURE"
        --hessian-batch-size "$HESSIAN_BATCH_SIZE"
    )

    if [[ "$OPTIMIZE_PRIOR_PRECISION" == "true" ]]; then
        CMD+=(--optimize-prior-precision)
    else
        CMD+=(--no-optimize-prior-precision)
    fi

    if [[ -n "$CHECKPOINT_PATH" ]]; then
        CMD+=(--checkpoint-path "$CHECKPOINT_PATH")
    fi
    if [[ -n "$EPOCH" ]]; then
        CMD+=(--epoch "$EPOCH")
    fi
    if [[ -n "$WANDB_PROJECT" ]]; then
        CMD+=(--wandb-project "$WANDB_PROJECT")
    fi
    if [[ -n "$WANDB_ENTITY" ]]; then
        CMD+=(--wandb-entity "$WANDB_ENTITY")
    fi

    "${CMD[@]}"

done

echo "=========================================="
echo "  Done."
echo "=========================================="
