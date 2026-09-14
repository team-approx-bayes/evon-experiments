# Scripts

Helper launchers for training and checkpoint evaluation. They activate the
project virtualenv (`.venv`, created with `uv sync`) themselves. The eval and
finetune scripts can be launched from anywhere; the two `*_speedrun_local.bash`
scripts must be launched from the repo root.

| Script | Purpose |
|---|---|
| `gpt_speedrun_local.bash` | GPT-2 pretraining with EVON (modded-nanogpt) |
| `llama_speedrun_local.bash` | LLaMA-130M pretraining with EVON (C4) |
| `gpt_checkpoint_mc_eval_local.bash` | MC posterior (BMA) eval of a GPT-2 checkpoint |
| `llama_checkpoint_mc_eval_local.bash` | MC posterior (BMA) eval of an LLaMA checkpoint, with posterior temperature |
| `llama_temperature_sweep_local.bash` | Temperature selection workflow: sweep T on a train shard, validate the winner |
| `run_finetune.sh` | CLIP finetuning with tuned hyperparameters |
| `run_laplace.sh` | Laplace approximation eval on a finetuned CLIP checkpoint |

---

## `gpt_speedrun_local.bash`

GPT-2 pretraining with EVON (whitening enabled) using the tuned
hyperparameters from the paper, baked into the script. Edit
`CUDA_VISIBLE_DEVICES` and `--batch_size_pre_gpu` inside for your GPU (`32` as
shipped; `64` for 40GB, `128` for 80GB).

```bash
bash scripts/gpt_speedrun_local.bash
```

## `llama_speedrun_local.bash`

LLaMA-130M pretraining on C4 with EVON, tuned hyperparameters baked in. Adjust
`DATASET_PATH` and `CUDA_VISIBLE_DEVICES` inside.

```bash
bash scripts/llama_speedrun_local.bash
```

## `gpt_checkpoint_mc_eval_local.bash`

MC-BMA evaluation of a GPT-2 (modded-nanogpt) checkpoint: draws posterior
samples from the saved optimizer state and reports per-`MC` BMA NLL and the
posterior-mean NLL.

Required: `CHECKPOINT` (path to a `.pt` checkpoint). Optional knobs:

| Variable | Default | Description |
|---|---|---|
| `MC_SAMPLES` / `MC_SAMPLES_LIST` | `10` / unset | Sample budget; list form (e.g. `1,4,16`) writes one CSV row per value |
| `VAL_TOKENS` | `10485760` | Number of validation tokens |
| `BATCH_SIZE_PRE_GPU` | `16` | Eval batch size per GPU |
| `INPUT_VAL_BIN` | default in eval script | Validation data file |
| `DEVICE` | `cuda` | Device |
| `CSV_OUT` / `APPEND_CSV` / `RUN_LABEL` | unset / `false` / unset | Results CSV path, append mode, row label |
| `PROGRESS` / `PROGRESS_EVERY` | `true` / `25` | Progress reporting |
| `NPROC_PER_NODE` | `1` | `> 1` runs multi-GPU eval via `torch.distributed.run` |

```bash
CHECKPOINT=modded-nanogpt/checkpoints/nanoGPT/model_10000.pt \
MC_SAMPLES_LIST=1,4,16 CSV_OUT=results/mc_eval_gpt.csv APPEND_CSV=true \
  bash scripts/gpt_checkpoint_mc_eval_local.bash
```

## `llama_checkpoint_mc_eval_local.bash`

MC-BMA evaluation of an LLaMA checkpoint: same idea as the GPT-2 eval, plus
posterior temperature support. Requires the variational optimizer state
(`optimizer.pt`) for MC sampling.

Driven entirely by environment variables:

| Variable | Default | Description |
|---|---|---|
| `CHECKPOINT_DIR` | **(required)** | Saved model directory (relative to repo root or absolute) |
| `OPTIMIZER_PATH` | `$CHECKPOINT_DIR/optimizer.pt` | Variational optimizer state |
| `MC_SAMPLES` / `MC_SAMPLES_LIST` | `10` / unset | Sample budget; list form (e.g. `1,4,16,32`) writes one CSV row per value |
| `TEMPERATURE` / `TEMPERATURE_LIST` | `1.0` / unset | Posterior temperature T (EVON/IVON): noise std scaled by `sqrt(T)`; list form (e.g. `0.5,1,2,5`) runs one full eval per value, one CSV row per (T, MC) |
| `VAL_TOKENS` | `10000000` | Number of tokens to evaluate |
| `DATASET_PATH` | **(required)** unless `HF_DATASET=true` | Dataset directory containing the shard files |
| `VAL_FILES` | 8 C4 validation shards | Comma-separated shard files to evaluate (point at a train shard to evaluate on training data) |
| `HF_DATASET` | `false` | Stream HF `allenai/c4` validation instead of local shards |
| `BATCH_SIZE` / `MAX_LENGTH` | checkpoint config | Eval batch size / sequence length |
| `DEVICE` | `cuda` | Device |
| `CSV_OUT` / `APPEND_CSV` / `RUN_LABEL` | unset / `false` / unset | Results CSV path, append mode, row label |
| `PROGRESS` / `PROGRESS_EVERY` | `true` / `25` | Progress reporting |
| `QKV_MODE` / `ATTN_RATIO` | config default | Architecture overrides |
| `NPROC_PER_NODE` | `1` | `> 1` runs multi-GPU eval via `torch.distributed.run` |

Examples:

```bash
# Baseline eval, 10M validation tokens
CHECKPOINT_DIR=checkpoints/llama_350m/model_60001 \
DATASET_PATH=/data1/datasets/c4-t5/subset \
  bash scripts/llama_checkpoint_mc_eval_local.bash

# Temperature sweep in one run: T in {0.5,1,2,5}, 2M tokens, one CSV row per (T, MC)
CHECKPOINT_DIR=checkpoints/llama_350m/model_60001 \
DATASET_PATH=/data1/datasets/c4-t5/subset \
TEMPERATURE_LIST=0.5,1,2,5 MC_SAMPLES_LIST=1,4,16,32 VAL_TOKENS=2000000 \
CSV_OUT=results/posterior_temperature/val_sweep.csv APPEND_CSV=true RUN_LABEL=val_sweep \
  bash scripts/llama_checkpoint_mc_eval_local.bash
```

## `llama_temperature_sweep_local.bash`

Posterior temperature selection for LLaMA MC-BMA eval in two runs, wrapping
`llama_checkpoint_mc_eval_local.bash` (all of its env knobs pass through):

1. **Sweep**: evaluates `TEMPERATURE_LIST` on `SWEEP_TOKENS` tokens from a
   **training** shard, one CSV row per (T, MC).
2. **Validate**: picks the T with the lowest `mc_bma_nll` at the `BEST_MC`
   budget (via `scripts/pick_best_temperature.py`), prints the full ranking,
   and re-runs the winner on the default **validation** shards.

| Variable | Default | Description |
|---|---|---|
| `CHECKPOINT_DIR` | **(required)** | Saved model directory (used by both runs) |
| `TEMPERATURE_LIST` | `1,2,4,8` | Temperatures to sweep |
| `MC_SAMPLES_LIST` | `1,4,16,32` | Sweep MC budgets |
| `SWEEP_TOKENS` | `2000000` | Train-shard tokens per temperature |
| `TRAIN_VAL_FILES` | `c4-train.00000-of-01024.json.gz` | Train shard(s) used for the sweep |
| `BEST_MC` | largest in `MC_SAMPLES_LIST` | MC budget used to select the winner |
| `VAL_TOKENS` | `10000000` | Validation tokens for the final run |
| `VAL_MC_SAMPLES_LIST` | = `MC_SAMPLES_LIST` | MC budgets for the validation run |
| `RUN_LABEL` | `temperature_sweep` | Row-label prefix for the two CSVs |
| `OUT_DIR` | `results/temperature_sweep` | Output directory: `<label>_train_sweep.csv`, `<label>_validation.csv` |

```bash
CHECKPOINT_DIR=checkpoints/llama_350m/model_60001 \
DATASET_PATH=/data1/datasets/c4-t5/subset \
TEMPERATURE_LIST=1,2,4,8 MC_SAMPLES_LIST=1,4,16,32 \
  bash scripts/llama_temperature_sweep_local.bash
```

## `run_finetune.sh`

CLIP finetuning using the tuned hyperparameters in
`clip-finetuning/best_hparams/<dataset>/<optim>/config.yaml`. Runs once per
dataset given.

```
Usage: run_finetune.sh --model-name <str> (--dataset-name <name> ...)
  --seed <int> --optim {adamw,sgd,evon,ivon,soap} --batch-size <int>
  --trainable-scope {all,linear} [--best-hparams-dir <dir>]
  [--wandb-project <proj>] [--wandb-entity <ent>]
  [--disable-checkpointing] [--optim-cfg-override key=value]...
```

```bash
bash scripts/run_finetune.sh --model-name ViT-B-32 --dataset-name cifar100 \
  --seed 0 --optim evon --batch-size 128 --trainable-scope all
```

## `run_laplace.sh`

Laplace approximation evaluation on a finetuned CLIP checkpoint. Thin wrapper
that forwards all arguments to `clip-finetuning/run_laplace.sh`.

```
Usage: run_laplace.sh --model-name <str> (--dataset-name <name> ...)
  --seed <int> --optim {adamw,sgd,evon,ivon,soap} --batch-size <int>
  --trainable-scope {all,linear} [--checkpoint-path <path>] [--epoch <int>]
  [--prior-precision <float>] [--hessian-structure {diag,kfac}]
  [--wandb-project <proj>] [--wandb-entity <ent>]
```

```bash
bash scripts/run_laplace.sh --model-name ViT-B-32 --dataset-name cifar100 \
  --seed 0 --optim evon --batch-size 32 --trainable-scope all \
  --hessian-structure kfac
```
