# Scripts

Helper launchers for training and checkpoint evaluation. All scripts expect the
project virtualenv at the repo root (`.venv`, created with `uv sync`) and
activate it themselves.

| Script | Purpose |
|---|---|
| `gpt_speedrun_local.bash` | GPT-2 pretraining with EVON (modded-nanogpt) |
| `llama_speedrun_local.bash` | LLaMA-130M pretraining with EVON (C4) |
| `gpt_checkpoint_mc_eval_local.bash` | MC posterior (BMA) eval of a GPT-2 checkpoint |
| `llama_checkpoint_mc_eval_local.bash` | MC posterior (BMA) eval of a LLaMA checkpoint, with posterior temperature |
| `run_finetune.sh` | CLIP finetuning with tuned hyperparameters |
| `run_laplace.sh` | Laplace approximation eval on a finetuned CLIP checkpoint |

---

## `llama_checkpoint_mc_eval_local.bash`

Monte-Carlo BMA evaluation of a saved LLaMA checkpoint: draws `MC` posterior
samples from the saved EVON/IVON optimizer state and reports per-`MC` BMA NLL
plus the posterior-mean NLL. Requires the variational optimizer state
(`optimizer.pt`) for MC sampling.

Driven entirely by environment variables:

| Variable | Default | Description |
|---|---|---|
| `CHECKPOINT_DIR` | **(required)** | Saved model directory (relative to repo root or absolute) |
| `OPTIMIZER_PATH` | `$CHECKPOINT_DIR/optimizer.pt` | Variational optimizer state |
| `MC_SAMPLES` | `10` | Number of posterior samples per eval |
| `MC_SAMPLES_LIST` | unset | Comma-separated list (e.g. `1,4,16,32`); overrides `MC_SAMPLES`, one CSV row per value |
| `TEMPERATURE` | unset (1.0) | Posterior temperature T (EVON/IVON): noise std scaled by `sqrt(T)`, i.e. sample from `N(mean, T * Sigma_post)` |
| `TEMPERATURE_LIST` | unset | Comma-separated temperatures (e.g. `0.5,1,2,5`); overrides `TEMPERATURE`, one full val pass + one CSV row per (T, MC) |
| `VAL_TOKENS` | `10000000` | Number of validation tokens to evaluate |
| `BATCH_SIZE` / `MAX_LENGTH` | checkpoint config | Eval batch size / sequence length |
| `DATASET_PATH` / `VAL_FILES` / `HF_DATASET` | defaults in eval script | Validation data source |
| `DEVICE` | `cuda` | Device |
| `CSV_OUT` / `APPEND_CSV` / `RUN_LABEL` | unset / `false` / unset | Results CSV output, append mode, row label |
| `PROGRESS` / `PROGRESS_EVERY` | `true` / `25` | Progress reporting |
| `QKV_MODE` / `ATTN_RATIO` | unset | Architecture overrides |
| `NPROC_PER_NODE` | `1` | `> 1` runs multi-GPU eval via `torch.distributed.run` |

Examples:

```bash
# Baseline eval, 10M validation tokens
CHECKPOINT_DIR=checkpoints/llama_350m/model_60001 \
  bash scripts/llama_checkpoint_mc_eval_local.bash

# Posterior temperature sweep in one run: T in {0.5,1,2,5}, 2M tokens,
# several MC budgets -> one CSV row per (T, MC) pair
CHECKPOINT_DIR=checkpoints/llama_350m/model_60001 \
TEMPERATURE_LIST=0.5,1,2,5 MC_SAMPLES_LIST=1,4,16,32 VAL_TOKENS=2000000 \
CSV_OUT=results/posterior_temperature/val_sweep.csv APPEND_CSV=true RUN_LABEL=val_sweep \
  bash scripts/llama_checkpoint_mc_eval_local.bash
```

## `gpt_checkpoint_mc_eval_local.bash`

Same idea for GPT-2 (modded-nanogpt) checkpoints. Required: `CHECKPOINT` (path
to a `.pt` checkpoint). Knobs: `MC_SAMPLES` (`10`), `MC_SAMPLES_LIST`,
`VAL_TOKENS` (`10485760`), `BATCH_SIZE_PRE_GPU` (`16`), `INPUT_VAL_BIN`,
`DEVICE`, `CSV_OUT`, `APPEND_CSV`, `RUN_LABEL`, `PROGRESS`, `PROGRESS_EVERY`,
`NPROC_PER_NODE`.

```bash
CHECKPOINT=modded-nanogpt/checkpoints/nanoGPT/model_10000.pt \
MC_SAMPLES_LIST=1,4,16 CSV_OUT=results/mc_eval_gpt.csv APPEND_CSV=true \
  bash scripts/gpt_checkpoint_mc_eval_local.bash
```

## `gpt_speedrun_local.bash`

GPT-2 pretraining with EVON (whitening enabled) using the tuned
hyperparameters from the paper, baked into the script. Launch from the repo
root. Edit `CUDA_VISIBLE_DEVICES` / `--batch_size_pre_gpu` inside for your GPU
memory (128 for 80GB, 64 for 40GB, 32 default).

```bash
bash scripts/gpt_speedrun_local.bash
```

## `llama_speedrun_local.bash`

LLaMA-130M pretraining on C4 with EVON, tuned hyperparameters baked in.
Launch from the repo root; adjust `DATASET_PATH` and `CUDA_VISIBLE_DEVICES`
inside.

```bash
bash scripts/llama_speedrun_local.bash
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
