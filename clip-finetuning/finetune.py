#!/usr/bin/env python3
"""
CLIP visual encoder fine-tuning script.
Dynamically builds classification heads locally and supports custom optimizers.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
import os
import urllib.parse
import yaml
from tqdm import tqdm

import timm
import timm.data
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from clip_finetune import (
    get_dataloaders,
    get_class_names,
    load_timm_model,
    freeze_for_finetune_fp,
    resolve_clip_text_model,
    build_clip_head,
)

# ---------------------------------------------------------------------------
# Default Optimizer Hyperparameters
# ---------------------------------------------------------------------------
LR_ADAMW: float = 1e-5
LR_SGD: float = 0.01
WD_ADAMW: float = 1e-1
WD_SGD: float = 0.0001
MOMENTUM_SGD: float = 0.9

CHECKPOINT_ROOT = Path(tempfile.mkdtemp(prefix="checkpoints_fp_"))

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _coerce_numeric_yaml_scalars(d: dict) -> dict:
    def _convert(value):
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_convert(v) for v in value]
        if isinstance(value, str):
            s = value.strip()
            try:
                if any(ch in s for ch in ".eE"):
                    return float(s)
                return int(s)
            except ValueError:
                return value
        return value
    return {k: _convert(v) for k, v in d.items()}

def _flatten_dict_for_wandb(d: dict, prefix: str) -> dict:
    flat: dict = {}
    for k, v in d.items():
        key = f"{prefix}_{k}"
        if isinstance(v, dict):
            flat.update(_flatten_dict_for_wandb(v, key))
        else:
            flat[key] = v
    return flat

def _parse_kv_overrides(pairs: list[str]) -> dict:
    overrides: dict = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Invalid override {item!r}; expected key=value")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override {item!r}; empty key")
        overrides[key] = yaml.safe_load(raw_value)
    return overrides

def _coerce_yaml_value(value: str | None):
    if value is None:
        return None
    return yaml.safe_load(value)

def _run_epoch(
    model: nn.Module,
    loader,
    classifier_weights: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    accum_steps: int,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    is_pbar = hasattr(loader, "set_postfix")
    step = 0
    if training:
        optimizer.zero_grad()

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            sampling_ctx = nullcontext()
            if training and optimizer is not None and hasattr(optimizer, "sampled_params"):
                sampling_ctx = optimizer.sampled_params(train=True)

            with sampling_ctx:
                image_features = model(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = image_features @ classifier_weights.T
                logits = logits * 100.0
                loss = F.cross_entropy(logits, labels)

                if training:
                    (loss / accum_steps).backward()
                    step += 1
                    wandb.log({"loss/train_step": loss.item()})

            if training and step % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                total_loss += loss.item() * images.size(0)
                total_correct += (logits.argmax(dim=-1) == labels).sum().item()
                total_samples += images.size(0)

            if is_pbar:
                loader.set_postfix(
                    loss=f"{total_loss / total_samples:.4f}",
                    acc=f"{100 * total_correct / total_samples:.2f}%",
                )

    if training and step % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / total_samples, total_correct / total_samples

def _evaluate_mc_sweep(
    model: nn.Module,
    loader,
    classifier_weights: torch.Tensor,
    device: torch.device,
    optimizer,
    mc_sample_counts: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
) -> list[tuple[int, float, float]]:
    if not hasattr(optimizer, "sampled_params"):
        return []

    model.eval()
    max_mc_samples = max(mc_sample_counts)
    targets = set(mc_sample_counts)
    totals_loss = {s: 0.0 for s in mc_sample_counts}
    totals_correct = {s: 0 for s in mc_sample_counts}
    totals_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            bsz = images.size(0)
            totals_samples += bsz

            probs_sum = None
            for s in range(1, max_mc_samples + 1):
                with optimizer.sampled_params(train=False):
                    image_features = model(images)
                    image_features = F.normalize(image_features, dim=-1)
                    logits = image_features @ classifier_weights.T
                    logits = logits * 100.0
                    probs = torch.softmax(logits, dim=-1)

                probs_sum = probs if probs_sum is None else (probs_sum + probs)
                avg_probs = probs_sum / s
                log_avg_probs = avg_probs.clamp_min(1e-12).log()
                loss = F.nll_loss(log_avg_probs, labels)
                pred = avg_probs.argmax(dim=-1)

                if s in targets:
                    totals_loss[s] += loss.item() * bsz
                    totals_correct[s] += (pred == labels).sum().item()

    return [
        (s, totals_loss[s] / totals_samples, totals_correct[s] / totals_samples)
        for s in mc_sample_counts
    ]

def _compute_calibration_metrics(
    model: nn.Module,
    loader,
    classifier_weights: torch.Tensor,
    device: torch.device,
    num_bins: int = 15,
    optimizer: torch.optim.Optimizer | None = None,
    mc_sample_counts: tuple[int, ...] | None = None,
) -> tuple[float, float] | list[tuple[int, float, float]]:
    model.eval()
    use_mc = optimizer is not None and hasattr(optimizer, "sampled_params") and mc_sample_counts is not None

    if not use_mc:
        total_samples = 0
        brier_sum = 0.0
        bin_totals = torch.zeros(num_bins, device=device)
        bin_conf_sums = torch.zeros(num_bins, device=device)
        bin_acc_sums = torch.zeros(num_bins, device=device)

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                image_features = model(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = image_features @ classifier_weights.T
                logits = logits * 100.0
                probs = torch.softmax(logits, dim=-1)

                conf, pred = probs.max(dim=-1)
                correct = (pred == labels).float()

                bin_ids = torch.clamp((conf * num_bins).long(), max=num_bins - 1)
                for b in range(num_bins):
                    mask = bin_ids == b
                    if mask.any():
                        count = mask.sum()
                        bin_totals[b] += count
                        bin_conf_sums[b] += conf[mask].sum()
                        bin_acc_sums[b] += correct[mask].sum()

                one_hot = torch.zeros_like(probs)
                one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
                brier_sum += ((probs - one_hot) ** 2).sum(dim=1).sum().item()
                total_samples += labels.numel()

        ece = 0.0
        for b in range(num_bins):
            if bin_totals[b] > 0:
                acc = (bin_acc_sums[b] / bin_totals[b]).item()
                avg_conf = (bin_conf_sums[b] / bin_totals[b]).item()
                ece += abs(acc - avg_conf) * (bin_totals[b].item() / total_samples)

        brier = brier_sum / total_samples if total_samples > 0 else 0.0
        return ece, brier

    max_mc = max(mc_sample_counts)
    targets = set(mc_sample_counts)

    totals_brier = {s: 0.0 for s in mc_sample_counts}
    totals_bin_totals = {s: torch.zeros(num_bins, device=device) for s in mc_sample_counts}
    totals_bin_conf = {s: torch.zeros(num_bins, device=device) for s in mc_sample_counts}
    totals_bin_acc = {s: torch.zeros(num_bins, device=device) for s in mc_sample_counts}
    totals_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            bsz = images.size(0)
            totals_samples += bsz

            probs_sum = None
            for s in range(1, max_mc + 1):
                with optimizer.sampled_params(train=False):
                    image_features = model(images)
                    image_features = F.normalize(image_features, dim=-1)
                    logits = image_features @ classifier_weights.T
                    logits = logits * 100.0
                    probs = torch.softmax(logits, dim=-1)

                probs_sum = probs if probs_sum is None else (probs_sum + probs)
                if s in targets:
                    avg_probs = probs_sum / s
                    one_hot = torch.zeros_like(avg_probs)
                    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
                    totals_brier[s] += ((avg_probs - one_hot) ** 2).sum(dim=1).sum().item()

                    conf, pred = avg_probs.max(dim=-1)
                    correct = (pred == labels).float()
                    bin_ids = torch.clamp((conf * num_bins).long(), max=num_bins - 1)
                    for b in range(num_bins):
                        mask = bin_ids == b
                        if mask.any():
                            count = mask.sum()
                            totals_bin_totals[s][b] += count
                            totals_bin_conf[s][b] += conf[mask].sum()
                            totals_bin_acc[s][b] += correct[mask].sum()

    results: list[tuple[int, float, float]] = []
    for s in mc_sample_counts:
        ece = 0.0
        if totals_samples > 0:
            for b in range(num_bins):
                if totals_bin_totals[s][b] > 0:
                    acc = (totals_bin_acc[s][b] / totals_bin_totals[s][b]).item()
                    avg_conf = (totals_bin_conf[s][b] / totals_bin_totals[s][b]).item()
                    ece += abs(acc - avg_conf) * (totals_bin_totals[s][b].item() / totals_samples)
        brier = totals_brier[s] / totals_samples if totals_samples > 0 else 0.0
        results.append((s, ece, brier))

    return results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a timm visual encoder in full precision.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-name",
        required=True,
        dest="model_name",
        help="timm model name, e.g. vit_base_patch16_224.openai_clip",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        dest="dataset_name",
        help="Dataset to fine-tune on.",
    )
    parser.add_argument("--seed", type=int, required=True, help="Global RNG seed.")
    parser.add_argument(
        "--optim",
        required=True,
        choices=["adamw", "sgd", "evon", "ivon", "soap"],
        help="Optimizer to use for training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        dest="batch_size",
        help="Physical (per-step) batch size.",
    )
    parser.add_argument(
        "--trainable-scope",
        type=str,
        default="all",
        dest="trainable_scope",
        choices=["all", "linear"],
        help="Which image-encoder parameters are trainable.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate for AdamW/SGD.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Override weight decay for AdamW/SGD.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=None,
        help="Override SGD momentum.",
    )
    parser.add_argument(
        "--optim-cfg-override",
        action="append",
        default=[],
        help="Override EVON/IVON/SOAP optimizer config key=value (repeatable).",
    )
    parser.add_argument(
        "--wandb-project",
        default="clip-finetune-best-hparams",
        dest="wandb_project",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        dest="wandb_entity",
        help="W&B entity (optional).",
    )
    parser.add_argument(
        "--disable-checkpointing",
        action="store_true",
        help="Disable checkpoint saving.",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Optional YAML config file to load hyperparameters from.",
    )
    args = parser.parse_args()

    model_name: str = args.model_name
    dataset_name: str = args.dataset_name
    seed: int = args.seed
    optim: str = args.optim
    batch_size: int = args.batch_size
    trainable_scope: str = args.trainable_scope
    lr_override: float | None = args.lr
    wd_override: float | None = args.weight_decay
    momentum_override: float | None = args.momentum
    optim_cfg_overrides: dict = _parse_kv_overrides(args.optim_cfg_override)

    # Load defaults from config file if available
    file_cfg = {}
    if args.config_file:
        with open(args.config_file, "r") as f:
            file_cfg = yaml.safe_load(f)
            
        if args.optim in ["adamw", "sgd"]:
            if lr_override is None and "lr" in file_cfg:
                lr_override = file_cfg["lr"]
            if wd_override is None and "weight_decay" in file_cfg:
                wd_override = file_cfg["weight_decay"]
            if momentum_override is None and "momentum" in file_cfg:
                momentum_override = file_cfg["momentum"]
        else:
            # Map parameters from config file to overrides dict
            for key in ["lr", "weight_decay", "ess", "hess_init", "beta1", "beta2", "betas", "shampoo_beta", "mc_samples", "whiten_prec_grad", "phasing", "price_clip_ratio"]:
                cfg_key = f"optim_cfg_{key}"
                if cfg_key in file_cfg:
                    optim_cfg_overrides[key] = file_cfg[cfg_key]
                elif key in file_cfg.get("optimizer_hparams", {}):
                    optim_cfg_overrides[key] = file_cfg["optimizer_hparams"][key]

    for key, value in optim_cfg_overrides.items():
        if value is not None:
            optim_cfg_overrides[key] = value

    # Reproducibility
    seed_everything(seed)
    print(f"[seed] Everything seeded with {seed}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using {device}")

    # Class names
    class_names = get_class_names(dataset_name)
    print(f"[dataset] Loaded {len(class_names)} classes for dataset {dataset_name!r}")

    # Build classification head locally using open_clip
    clip_arch, clip_pre = resolve_clip_text_model(model_name)
    print(f"[head] Generating classification head using CLIP text encoder {clip_arch!r} ...")
    classifier_weights = build_clip_head(
        clip_model_name=clip_arch,
        clip_pretrained=clip_pre,
        dataset_name=dataset_name,
        class_names=class_names,
        device=device,
        show_progress=True,
    )
    print(f"[head] Generated head tensor with shape: {tuple(classifier_weights.shape)}")

    # Load Model
    print(f"[model] Loading {model_name!r} ...")
    model, embed_dim, removed_text_attrs = load_timm_model(
        model_name,
        device,
        expected_embed_dim=int(classifier_weights.shape[1]),
    )
    if removed_text_attrs:
        print(f"[model] Removed text tower attrs: {removed_text_attrs}")
    freeze_for_finetune_fp(model, trainable_scope=trainable_scope)

    # Dataloaders
    data_cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)
    
    # Effective batch size is 128
    EFFECTIVE_BATCH_SIZE = 128
    if EFFECTIVE_BATCH_SIZE % batch_size != 0:
        raise ValueError(f"batch_size {batch_size} must evenly divide effective batch size {EFFECTIVE_BATCH_SIZE}")
    accum_steps = EFFECTIVE_BATCH_SIZE // batch_size
    
    print(f"[data] Loading loaders (physical batch={batch_size}, accumulation steps={accum_steps}) ...")
    train_loader, test_loader = get_dataloaders(
        dataset_name,
        batch_size=batch_size,
        transform=transform,
        seed=seed,
        num_workers=4,
    )

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_hparams: dict = {}
    
    if optim == "adamw":
        lr = lr_override if lr_override is not None else LR_ADAMW
        weight_decay = wd_override if wd_override is not None else WD_ADAMW
        optimizer_hparams = {"lr": lr, "weight_decay": weight_decay}
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optim == "sgd":
        lr = lr_override if lr_override is not None else LR_SGD
        weight_decay = wd_override if wd_override is not None else WD_SGD
        momentum = momentum_override if momentum_override is not None else MOMENTUM_SGD
        optimizer_hparams = {"lr": lr, "weight_decay": weight_decay, "momentum": momentum}
        optimizer = torch.optim.SGD(trainable_params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optim == "evon":
        # Load default EVON hparams
        evon_cfg = {
            "lr": 5e-2,
            "weight_decay": 1e-5,
            "ess": 5e8,
            "betas": (0.9, 0.99999),
            "shampoo_beta": 0.995,
            "hess_init": 1e-1,
            "mc_samples": 1,
            "phasing": False,
            "price_clip_ratio": 2.0,
            "sync": False,
            "precondition_1d": False,
            "whiten_prec_grad": True,
        }
        # Update with overrides from config or CLI
        for k, v in optim_cfg_overrides.items():
            # Map parameter names from config file format
            target_k = "whiten_prec_grad" if k == "whiten_grad" else k
            if target_k in evon_cfg or target_k == "betas":
                if target_k == "betas" and isinstance(v, list):
                    v = tuple(v)
                evon_cfg[target_k] = v
                
        optimizer_hparams = dict(evon_cfg)
        from vonsoap.optimizers import EVON
        optimizer = EVON(trainable_params, **evon_cfg)
    elif optim == "ivon":
        # Load default IVON hparams
        ivon_cfg = {
            "lr": 5e-2,
            "weight_decay": 1e-4,
            "ess": 1e9,
            "hess_init": 1e-2,
            "mc_samples": 1,
            "beta1": 0.9,
            "beta2": 0.99999,
        }
        # Update with overrides
        for k, v in optim_cfg_overrides.items():
            if k in ivon_cfg:
                ivon_cfg[k] = v
                
        optimizer_hparams = dict(ivon_cfg)
        from vonsoap.optimizers import IVON
        optimizer = IVON(trainable_params, **ivon_cfg)
    elif optim == "soap":
        # Load default SOAP hparams
        soap_cfg = {
            "lr": 1e-5,
            "weight_decay": 0.1,
            "betas": (0.9, 0.95),
            "shampoo_beta": 0.95,
        }
        # Update with overrides
        for k, v in optim_cfg_overrides.items():
            if k in soap_cfg or k == "betas":
                if k == "betas" and isinstance(v, list):
                    v = tuple(v)
                soap_cfg[k] = v
                
        optimizer_hparams = dict(soap_cfg)
        from vonsoap.optimizers import SOAP
        optimizer = SOAP(trainable_params, **soap_cfg)
    else:
        raise ValueError(f"Unknown optimizer: {optim!r}")

    # Initialize WandB
    run_name = f"{model_name}_{dataset_name}_seed{seed}_{optim}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            "model_name": model_name,
            "dataset_name": dataset_name,
            "seed": seed,
            "optim": optim,
            "lr": optimizer_hparams.get("lr"),
            "weight_decay": optimizer_hparams.get("weight_decay"),
            "momentum": optimizer_hparams.get("momentum"),
            "optimizer_hparams": optimizer_hparams,
            "batch_size": EFFECTIVE_BATCH_SIZE,
            "physical_batch_size": batch_size,
            "accum_steps": accum_steps,
            "trainable_scope": trainable_scope,
            "device": str(device),
            **_flatten_dict_for_wandb(optimizer_hparams, prefix="optim_cfg"),
        },
    )

    # Checkpoint settings
    from clip_finetune.data import DATASET_TO_FT_EPOCHS, DATASET_TO_CHECKPOINT_EPOCHS
    total_epochs = DATASET_TO_FT_EPOCHS.get(dataset_name, 5)
    checkpoint_epochs = set(DATASET_TO_CHECKPOINT_EPOCHS.get(dataset_name, [1, 2, 5]))
    if args.disable_checkpointing:
        checkpoint_epochs.clear()
        
    ckpt_dir = Path("checkpoints") / model_name / str(seed) / optim / dataset_name
    if not args.disable_checkpointing:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    patience = 3
    best_test_loss = float("inf")
    epochs_no_improve = 0

    print(f"\n[train] Starting: {total_epochs} epoch(s), checkpoints at {sorted(checkpoint_epochs)}")
    
    epoch_bar = tqdm(range(1, total_epochs + 1), desc="Epochs", unit="epoch")
    for epoch in epoch_bar:
        # Train
        train_loss, train_acc = _run_epoch(
            model, train_loader, classifier_weights, device, optimizer, accum_steps,
        )

        # Evaluate
        test_loss, test_acc = _run_epoch(
            model, test_loader, classifier_weights, device, optimizer=None, accum_steps=1,
        )

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            train_acc=f"{100 * train_acc:.2f}%",
            test_loss=f"{test_loss:.4f}",
            test_acc=f"{100 * test_acc:.2f}%",
        )

        wandb.log({
            "epoch": epoch,
            "loss/train": train_loss,
            "loss/test": test_loss,
            "acc/train": train_acc,
            "acc/test": test_acc,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
        })

        if test_loss + 1e-8 < best_test_loss:
            best_test_loss = test_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                tqdm.write(
                    f"[early-stop] No test loss improvement for {patience} epochs "
                    f"(best={best_test_loss:.4f}). Stopping at epoch {epoch}."
                )
                break

        # Save Checkpoint
        if epoch in checkpoint_epochs:
            ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "seed": seed,
                    "optim": optim,
                },
                ckpt_path,
            )
            tqdm.write(f"  [ckpt] Saved local checkpoint: {ckpt_path}")

    # Monte Carlo samples evaluation
    if hasattr(optimizer, "sampled_params"):
        mc_counts = (1, 2, 4, 8, 16, 32)
        tqdm.write(f"[mc] Evaluating test metrics vs Monte Carlo samples {mc_counts} ...")
        mc_results = _evaluate_mc_sweep(
            model=model,
            loader=test_loader,
            classifier_weights=classifier_weights,
            device=device,
            optimizer=optimizer,
            mc_sample_counts=mc_counts,
        )
        if mc_results:
            table = wandb.Table(columns=["mc_samples", "test_loss", "test_acc"])
            log_dict = {}
            for mc_samples, mc_loss, mc_acc in mc_results:
                table.add_data(mc_samples, mc_loss, mc_acc)
                log_dict[f"mc_eval/test_loss_{mc_samples}"] = mc_loss
                log_dict[f"mc_eval/test_acc_{mc_samples}"] = mc_acc

            log_dict["mc_eval/table"] = table
            last_n, last_loss, last_acc = mc_results[-1]
            log_dict[f"mc_eval/test_loss_{last_n}"] = last_loss
            log_dict[f"mc_eval/test_acc_{last_n}"] = last_acc
            wandb.log(log_dict)

    # Calibration evaluation
    ece_mean, brier_mean = _compute_calibration_metrics(
        model=model,
        loader=test_loader,
        classifier_weights=classifier_weights,
        device=device,
        optimizer=None,
    )

    if hasattr(optimizer, "sampled_params"):
        mc_counts = (1, 2, 4, 8, 16, 32)
        calib_results = _compute_calibration_metrics(
            model=model,
            loader=test_loader,
            classifier_weights=classifier_weights,
            device=device,
            optimizer=optimizer,
            mc_sample_counts=mc_counts,
        )

        log_dict = {
            "calibration_mean/ece": ece_mean,
            "calibration_mean/brier": brier_mean,
        }
        for n, ece_n, brier_n in calib_results:
            log_dict[f"calibration_mc{n}/ece"] = ece_n
            log_dict[f"calibration_mc{n}/brier"] = brier_n
        wandb.log(log_dict)
    else:
        wandb.log({
            "calibration_mean/ece": ece_mean,
            "calibration_mean/brier": brier_mean,
        })

    wandb.finish()
    tqdm.write("\n[done] Training complete.")


if __name__ == "__main__":
    main()
