# CLIP Fine-Tuning with Custom Optimizers

> [!NOTE]
> Code adapted from [**crisostomi/model-merging**](https://github.com/crisostomi/model-merging).

This directory contains a minimal, self-contained implementation for fine-tuning CLIP visual encoders on image classification datasets. It supports custom optimizers (`EVON`, `IVON`, `SOAP`) from the included `vonsoap` package alongside other optimizers (`AdamW`, `SGD`).

Zero-shot text classification heads are dynamically generated locally using the CLIP text encoder (via `open_clip`) from the class names of each dataset, eliminating external dependencies.

## Structure

- `finetune.py`: Standalone Python script executing the fine-tuning training and evaluation loop.
- `run_finetune.sh`: Bash wrapper script that maps dataset name and optimizer to their pre-downloaded hyperparameter files in `best_hparams/`.
- `best_hparams/`: Local directory containing optimal hyperparameters for each dataset/optimizer pair.
- `finetune_best_hparams.slurm`: SLURM batch submission script to launch fine-tuning jobs on a cluster.

## Learning-rate convention

IVON and EVON can define the learning rate in two ways:

- **Rescaled**: the step size is `lr * (hess_init + weight_decay)`, so the size of the first step does not depend on `hess_init`. This is what `IVON` in `vonsoap` does (`rescale_lr=True`, the default), and the convention of the hyperparameter sweeps and of the learning rates $\alpha$ reported for IVON and EVON in the paper's CLIP tables.
- **Raw**: the step size is `lr` itself. This is what the frozen `EVON` in `vonsoap` does; it has no `rescale_lr` option.

The configs in `best_hparams/` are written for the code in this repository, so `optim_cfg_lr` is **rescaled for IVON** and **raw for EVON**. To convert an EVON learning rate between the two:

```
lr_raw = lr_rescaled * (hess_init + weight_decay)
lr_rescaled = lr_raw / (hess_init + weight_decay)
```

For example, `best_hparams/cars/evon/config.yaml` has `optim_cfg_lr: 7.875e-05` with `optim_cfg_hess_init: 0.01` and `optim_cfg_weight_decay: 2.32e-04`, which is a rescaled learning rate of `7.875e-05 / 0.010232 = 7.70e-03`. Since `hess_init + weight_decay` is usually about `hess_init`, passing a rescaled EVON learning rate to this code unchanged gives a step roughly `1 / hess_init` (here about 100x) too large.

## Usage

### Prerequisites
Make sure dependencies are installed. The package uses `clip_finetune` (configured in `pyproject.toml` packages search) which depends on:
- `timm`
- `open-clip-torch`
- `datasets` (Hugging Face)
- `torch` & `torchvision`
- `wandb`
- `pyyaml`
- `tqdm`

Install the local package through `uv` or with `pip` in editable mode:
```bash
pip install -e .
```

### Running Locally

To run the fine-tuning script directly, specify the model, dataset, optimizer, parameters, and the configuration file containing the best hyperparameters:
```bash
python finetune.py \
    --model-name vit_base_patch16_224.openai_clip \
    --dataset-name mnist \
    --seed 42 \
    --optim evon \
    --batch-size 128 \
    --trainable-scope linear \
    --config-file best_hparams/mnist/evon/config.yaml
```

Alternatively, use the `run_finetune.sh` wrapper which resolves the configuration file location automatically:
```bash
./run_finetune.sh \
    --model-name vit_base_patch16_224.openai_clip \
    --dataset-name mnist \
    --seed 42 \
    --optim evon \
    --batch-size 128 \
    --trainable-scope linear
```

### Running on a SLURM Cluster

Configure environment variables in `finetune_best_hparams.slurm` if needed, then submit the job:
```bash
sbatch finetune_best_hparams.slurm
```
You can override parameters at submission time:
```bash
sbatch --export=ALL,DATASET=eurosat,OPTIM=evon finetune_best_hparams.slurm
```
