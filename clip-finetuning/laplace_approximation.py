#!/usr/bin/env python3
"""
Laplace Approximation evaluation script for fine-tuned CLIP models.
Estimates Laplace posterior (Diagonal GGN or KFAC) over fine-tuned weights
from a saved checkpoint and evaluates MC ensemble metrics (NLL, Accuracy, ECE, Brier score).
Logs results to WandB matching finetune.py metrics.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Literal

import timm
import timm.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from clip_finetune import (
    get_dataloaders,
    get_class_names,
    load_timm_model,
    freeze_for_finetune_fp,
    resolve_clip_text_model,
    build_clip_head,
)


def seed_everything(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_checkpoint(
    ckpt_root: str | Path,
    model_name: str,
    seed: int,
    optim: str,
    dataset_name: str,
    epoch: int | None = None,
) -> Path:
    """Locate checkpoint path given experiment parameters."""
    dir_path = Path(ckpt_root) / model_name / str(seed) / optim / dataset_name
    if not dir_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {dir_path}")

    if epoch is not None:
        target = dir_path / f"epoch_{epoch}.pt"
        if not target.exists():
            raise FileNotFoundError(f"Specified epoch checkpoint not found: {target}")
        return target

    # Find highest epoch checkpoint
    ckpts = list(dir_path.glob("epoch_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint files found in {dir_path}")

    def _extract_epoch(p: Path) -> int:
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return -1

    ckpts.sort(key=_extract_epoch)
    return ckpts[-1]


def compute_diagonal_ggn(
    model: nn.Module,
    loader,
    classifier_weights: torch.Tensor,
    device: torch.device,
    max_samples: int | None = None,
    micro_batch_size: int | None = None,
) -> list[torch.Tensor]:
    """
    Compute diagonal Generalized Gauss-Newton (GGN) / Empirical Fisher
    over trainable parameters using training set samples.
    """
    model.eval()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    diag_ggn = [torch.zeros_like(p) for p in trainable_params]

    total_samples = 0
    dataset_size = len(loader.dataset) if hasattr(loader, "dataset") else None
    if dataset_size is not None and max_samples is not None and max_samples < dataset_size:
        print(f"[laplace] Warning: max_samples ({max_samples}) is smaller than dataset size ({dataset_size}). GGN estimation will cap at {max_samples} samples.")

    pbar = tqdm(loader, desc="[laplace] Computing Diagonal GGN", file=sys.stdout)

    for images, labels in pbar:
        if max_samples is not None and total_samples >= max_samples:
            break

        images = images.to(device, non_blocking=True)
        bsz = images.size(0)
        chunk_sz = micro_batch_size if micro_batch_size is not None and micro_batch_size > 0 else bsz

        for m in range(0, bsz, chunk_sz):
            if max_samples is not None and total_samples >= max_samples:
                break

            sub_images = images[m:m + chunk_sz]
            sub_bsz = sub_images.size(0)

            image_features = model(sub_images)
            image_features = F.normalize(image_features, dim=-1)
            logits = (image_features @ classifier_weights.T) * 100.0
            probs = torch.softmax(logits, dim=-1)
            sampled_y = torch.multinomial(probs, num_samples=1).squeeze(-1)

            for i in range(sub_bsz):
                if max_samples is not None and total_samples >= max_samples:
                    break

                loss_i = -torch.log_softmax(logits[i:i+1], dim=-1)[0, sampled_y[i]]
                grads = torch.autograd.grad(loss_i, trainable_params, retain_graph=True)

                for idx, g in enumerate(grads):
                    if g is not None:
                        diag_ggn[idx] += g.detach() ** 2

                total_samples += 1

    if dataset_size is not None and total_samples < dataset_size:
        print(f"[laplace] GGN computation completed on {total_samples}/{dataset_size} samples (capped by max_samples={max_samples}).")

    # Scale by sample count
    if total_samples > 0:
        for idx in range(len(diag_ggn)):
            diag_ggn[idx] = diag_ggn[idx] * (len(loader.dataset) / total_samples)

    return diag_ggn


def compute_kfac_ggn(
    model: nn.Module,
    loader,
    classifier_weights: torch.Tensor,
    device: torch.device,
    max_samples: int | None = None,
    micro_batch_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Kronecker-factored GGN for the final linear layer/projection.
    Returns factor matrices (A_inv_sqrt, B_inv_sqrt).
    """
    model.eval()
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear) and m.weight.requires_grad]
    if not linear_layers:
        raise RuntimeError("No trainable nn.Linear layer found for KFAC GGN.")

    target_layer = linear_layers[-1]
    inputs_list = []

    def hook_fn(module, input, output):
        inputs_list.append(input[0].detach())

    handle = target_layer.register_forward_hook(hook_fn)

    total_samples = 0
    A_accum = 0.0
    B_accum = 0.0

    dataset_size = len(loader.dataset) if hasattr(loader, "dataset") else None
    if dataset_size is not None and max_samples is not None and max_samples < dataset_size:
        print(f"[laplace] Warning: max_samples ({max_samples}) is smaller than dataset size ({dataset_size}). GGN estimation will cap at {max_samples} samples.")

    pbar = tqdm(loader, desc="[laplace] Computing KFAC GGN", file=sys.stdout)
    for images, labels in pbar:
        if max_samples is not None and total_samples >= max_samples:
            break

        images = images.to(device, non_blocking=True)
        bsz = images.size(0)
        chunk_sz = micro_batch_size if micro_batch_size is not None and micro_batch_size > 0 else bsz

        for m in range(0, bsz, chunk_sz):
            if max_samples is not None and total_samples >= max_samples:
                break

            sub_images = images[m:m + chunk_sz]
            sub_bsz = sub_images.size(0)

            inputs_list.clear()
            image_features = model(sub_images)
            activations = inputs_list[0] if inputs_list else image_features

            image_features_norm = F.normalize(image_features, dim=-1)
            logits = (image_features_norm @ classifier_weights.T) * 100.0
            probs = torch.softmax(logits, dim=-1)

            # Activation covariance matrix A
            A_batch = activations.T @ activations
            A_accum = A_accum + A_batch.detach()

            # Logit Hessian / curvature B
            p_diag = torch.diag_embed(probs)
            p_outer = torch.bmm(probs.unsqueeze(2), probs.unsqueeze(1))
            H_logits = (p_diag - p_outer) * (100.0 ** 2)
            H_logits_sum = H_logits.sum(dim=0)
            # Map curvature from logit space (C x C) to embedding space (d_out x d_out) via classifier head
            B_batch = classifier_weights.T @ H_logits_sum @ classifier_weights
            B_accum = B_accum + B_batch.detach()

            total_samples += sub_bsz

    handle.remove()

    if dataset_size is not None and total_samples < dataset_size:
        print(f"[laplace] GGN computation completed on {total_samples}/{dataset_size} samples (capped by max_samples={max_samples}).")

    scale_factor = math.sqrt(total_samples) if total_samples > 0 else 1.0
    A = A_accum / scale_factor
    B = B_accum / scale_factor

    return A, B, total_samples


def optimize_prior_precision_kfac(
    target_layer: nn.Module,
    A: torch.Tensor,
    B: torch.Tensor,
    lmbda_init: float = 1.0,
    n: float = 1.0,
    lr: float = 1e-2,
    num_steps: int = 300,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Optimize prior precision lambda for KFAC by maximizing Laplace Marginal Likelihood."""
    weight_norm_sq = (target_layer.weight ** 2).sum().detach()
    p_num = target_layer.weight.numel()

    A = A.to(device)
    B = B.to(device)

    log_lmbda = torch.nn.Parameter(
        torch.tensor(lmbda_init, device=device, requires_grad=True, dtype=torch.float32).log()
    )
    sqrt_n = torch.tensor(n, device=device, requires_grad=False, dtype=torch.float32).sqrt()

    optimizer = torch.optim.Adam([log_lmbda], lr=lr, maximize=True)

    pbar = tqdm(range(num_steps), desc="[laplace] Optimizing Prior Precision (KFAC MargLik)", file=sys.stdout)
    for _ in pbar:
        optimizer.zero_grad()

        lmbda = log_lmbda.exp()
        sqrt_lmbda = lmbda.sqrt()

        A_ = A * sqrt_n + sqrt_lmbda * torch.eye(A.shape[0], device=device, dtype=A.dtype)
        B_ = B * sqrt_n + sqrt_lmbda * torch.eye(B.shape[0], device=device, dtype=B.dtype)

        log_prior = -0.5 * lmbda * weight_norm_sq + 0.5 * p_num * torch.log(lmbda)

        logdet_A = torch.logdet(A_)
        logdet_B = torch.logdet(B_)
        p_dim, q_dim = A_.shape[0], B_.shape[0]
        log_det = 0.5 * (logdet_A * q_dim + logdet_B * p_dim)

        marglik = log_prior - log_det
        marglik.backward()
        optimizer.step()

        pbar.set_postfix(lambda_opt=f"{log_lmbda.exp().item():.4e}")

    opt_lmbda = log_lmbda.exp().item()
    print(f"  [MargLik] Optimized KFAC prior precision lambda = {opt_lmbda:.6e}")
    return opt_lmbda


def optimize_prior_precision_diag(
    model: nn.Module,
    diag_ggn: list[torch.Tensor],
    lmbda_init: float = 1.0,
    lr: float = 1e-2,
    num_steps: int = 300,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Optimize prior precision lambda for Diagonal GGN by maximizing Laplace Marginal Likelihood."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    weight_norm_sq = sum((p ** 2).sum() for p in trainable_params).detach()
    p_num = sum(p.numel() for p in trainable_params)

    diag_concat = torch.cat([g.flatten().to(device) for g in diag_ggn])

    log_lmbda = torch.nn.Parameter(
        torch.tensor(lmbda_init, device=device, requires_grad=True, dtype=torch.float32).log()
    )

    optimizer = torch.optim.Adam([log_lmbda], lr=lr, maximize=True)

    pbar = tqdm(range(num_steps), desc="[laplace] Optimizing Prior Precision (Diagonal MargLik)", file=sys.stdout)
    for _ in pbar:
        optimizer.zero_grad()

        lmbda = log_lmbda.exp()

        log_prior = -0.5 * lmbda * weight_norm_sq + 0.5 * p_num * torch.log(lmbda)
        log_det = 0.5 * torch.sum(torch.log(diag_concat + lmbda))

        marglik = log_prior - log_det
        marglik.backward()
        optimizer.step()

        pbar.set_postfix(lambda_opt=f"{log_lmbda.exp().item():.4e}")

    opt_lmbda = log_lmbda.exp().item()
    print(f"  [MargLik] Optimized Diagonal prior precision lambda = {opt_lmbda:.6e}")
    return opt_lmbda


def _evaluate_map(
    model: nn.Module,
    loader,
    classifier_weights: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate MAP model loss and accuracy."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            image_features = model(images)
            image_features = F.normalize(image_features, dim=-1)
            logits = (image_features @ classifier_weights.T) * 100.0
            loss = F.cross_entropy(logits, labels)

            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def _compute_calibration_metrics(
    model: nn.Module,
    loader,
    classifier_weights: torch.Tensor,
    device: torch.device,
    num_bins: int = 15,
) -> tuple[float, float]:
    """Compute ECE and Brier score for MAP model."""
    model.eval()
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
            logits = (image_features @ classifier_weights.T) * 100.0
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


def _evaluate_laplace_probit(
    model: nn.Module,
    loader,
    classifier_weights: torch.Tensor,
    device: torch.device,
    laplace_std: list[torch.Tensor] | None = None,
    kfac_cov: tuple[torch.Tensor, torch.Tensor] | None = None,
    prior_precision: float = 1.0,
    var_scale: float = 1.0,
    num_bins: int = 15,
) -> tuple[float, float, float, float]:
    """
    Perform evaluation using ProbCosine and Multiclass Probit approximation (closed-form, no sampling).
    Returns (test_loss, test_acc, ece, brier).
    """
    model.eval()

    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear) and m.weight.requires_grad]
    target_layer = linear_layers[-1] if linear_layers else None

    # Compute A_inv and B_inv_diag if KFAC
    A_inv, B_inv_diag = None, None
    if kfac_cov is not None:
        if len(kfac_cov) == 3:
            A, B, kfac_samples = kfac_cov
            n = kfac_samples
        else:
            A, B = kfac_cov
            n = len(loader.dataset)

        sqrt_n = math.sqrt(n)
        sqrt_lambda = math.sqrt(prior_precision)

        A_post = A * sqrt_n + sqrt_lambda * torch.eye(A.size(0), device=device, dtype=A.dtype)
        B_post = B * sqrt_n + sqrt_lambda * torch.eye(B.size(0), device=device, dtype=B.dtype)

        A_inv = torch.linalg.inv(A_post)
        B_inv_diag = torch.linalg.inv(B_post).diagonal()

    # If diagonal, retrieve target layer variance
    target_layer_var = None
    if laplace_std is not None and target_layer is not None:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        for idx, p in enumerate(trainable_params):
            if p is target_layer.weight:
                target_layer_var = (laplace_std[idx] ** 2) * var_scale
                break

    inputs_list = []
    def hook_fn(module, input, output):
        inputs_list.append(input[0].detach())

    handle = target_layer.register_forward_hook(hook_fn) if target_layer is not None else None

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    brier_sum = 0.0

    bin_totals = torch.zeros(num_bins, device=device)
    bin_conf_sums = torch.zeros(num_bins, device=device)
    bin_acc_sums = torch.zeros(num_bins, device=device)

    # Normalize classification head weights
    w_norm = F.normalize(classifier_weights, dim=-1)  # [C, d_out]
    expect_norm_target = (w_norm ** 2).sum(dim=-1, keepdim=True).T  # [1, C]

    pbar = tqdm(loader, desc="[laplace] Evaluating Probit Approximation", file=sys.stdout)
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            bsz = images.size(0)

            inputs_list.clear()
            source_embeds = model(images)  # MAP embeddings [B, d_out]
            activations = inputs_list[0] if inputs_list else source_embeds  # [B, d_in]

            # Compute variance propagation to embeddings: diag_cov_img [B, d_out]
            if kfac_cov is not None and A_inv is not None:
                quad_form = torch.einsum('bi,ij,bj->b', activations, A_inv, activations)  # [B]
                diag_cov_img = (quad_form[:, None] * B_inv_diag[None, :]) * var_scale  # [B, d_out]
            elif target_layer_var is not None:
                diag_cov_img = (activations ** 2) @ target_layer_var.T  # [B, d_out]
            else:
                diag_cov_img = torch.zeros_like(source_embeds)

            # ProbCosine Mean & Variance
            norm_source = (source_embeds ** 2) + diag_cov_img  # [B, d_out]
            expect_norm_source = norm_source.sum(dim=-1, keepdim=True).clamp_min(1e-8)  # [B, 1]

            expected_similarity = (source_embeds / torch.sqrt(expect_norm_source)) @ w_norm.T  # [B, C]
            mean_logits = expected_similarity * 100.0  # [B, C]

            var_similarity = (diag_cov_img @ (w_norm ** 2).T) / (expect_norm_source * expect_norm_target)  # [B, C]
            var_logits = var_similarity * (100.0 ** 2)  # [B, C]

            # Multiclass Probit Approximation
            kappa = 1.0 / torch.sqrt(1.0 + (math.pi / 8.0) * var_logits)  # [B, C]
            probit_probs = F.softmax(kappa * mean_logits, dim=-1)  # [B, C]

            # Metrics
            log_probit_probs = probit_probs.clamp_min(1e-12).log()
            loss = F.nll_loss(log_probit_probs, labels)
            pred = probit_probs.argmax(dim=-1)

            total_loss += loss.item() * bsz
            total_correct += (pred == labels).sum().item()
            total_samples += bsz

            conf, p_pred = probit_probs.max(dim=-1)
            correct = (p_pred == labels).float()
            bin_ids = torch.clamp((conf * num_bins).long(), max=num_bins - 1)
            for b in range(num_bins):
                mask = bin_ids == b
                if mask.any():
                    count = mask.sum()
                    bin_totals[b] += count
                    bin_conf_sums[b] += conf[mask].sum()
                    bin_acc_sums[b] += correct[mask].sum()

            one_hot = torch.zeros_like(probit_probs)
            one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
            brier_sum += ((probit_probs - one_hot) ** 2).sum(dim=1).sum().item()

    if handle is not None:
        handle.remove()

    test_loss = total_loss / total_samples
    test_acc = total_correct / total_samples
    brier = brier_sum / total_samples if total_samples > 0 else 0.0

    ece = 0.0
    if total_samples > 0:
        for b in range(num_bins):
            if bin_totals[b] > 0:
                acc = (bin_acc_sums[b] / bin_totals[b]).item()
                avg_conf = (bin_conf_sums[b] / bin_totals[b]).item()
                ece += abs(acc - avg_conf) * (bin_totals[b].item() / total_samples)

    return test_loss, test_acc, ece, brier


def _evaluate_laplace_mc(
    model: nn.Module,
    loader,
    classifier_weights: torch.Tensor,
    device: torch.device,
    laplace_std: list[torch.Tensor] | None = None,
    kfac_cov: tuple[torch.Tensor, torch.Tensor] | None = None,
    prior_precision: float = 1.0,
    var_scale: float = 1.0,
    mc_sample_counts: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    num_bins: int = 15,
) -> tuple[list[tuple[int, float, float]], list[tuple[int, float, float]]]:
    """
    Perform Monte Carlo sampling evaluation using Laplace posterior weights.
    Returns (mc_loss_acc_results, mc_calibration_results).
    """
    model.eval()
    max_mc_samples = max(mc_sample_counts)
    targets = set(mc_sample_counts)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Pre-calculate KFAC Cholesky/eigen factors if using KFAC
    A_inv_sqrt, B_inv_sqrt = None, None
    if kfac_cov is not None:
        if len(kfac_cov) == 3:
            A, B, kfac_samples = kfac_cov
            n = kfac_samples
        else:
            A, B = kfac_cov
            n = len(loader.dataset)

        sqrt_n = math.sqrt(n)
        sqrt_lambda = math.sqrt(prior_precision)

        A_post = A * sqrt_n + sqrt_lambda * torch.eye(A.size(0), device=device, dtype=A.dtype)
        B_post = B * sqrt_n + sqrt_lambda * torch.eye(B.size(0), device=device, dtype=B.dtype)

        A_inv = torch.linalg.inv(A_post)
        B_inv = torch.linalg.inv(B_post)

        evals_A, evecs_A = torch.linalg.eigh(A_inv)
        evals_B, evecs_B = torch.linalg.eigh(B_inv)

        # Apply variance scaling factor to square root matrices
        scale_factor = math.sqrt(var_scale)
        A_inv_sqrt = (evecs_A @ torch.diag(torch.clamp_min(evals_A, 1e-10).sqrt()) @ evecs_A.T) * math.sqrt(scale_factor)
        B_inv_sqrt = (evecs_B @ torch.diag(torch.clamp_min(evals_B, 1e-10).sqrt()) @ evecs_B.T) * math.sqrt(scale_factor)

    if laplace_std is not None and var_scale != 1.0:
        laplace_std = [s * var_scale for s in laplace_std]

    totals_loss = {s: 0.0 for s in mc_sample_counts}
    totals_correct = {s: 0 for s in mc_sample_counts}
    totals_brier = {s: 0.0 for s in mc_sample_counts}
    totals_bin_totals = {s: torch.zeros(num_bins, device=device) for s in mc_sample_counts}
    totals_bin_conf = {s: torch.zeros(num_bins, device=device) for s in mc_sample_counts}
    totals_bin_acc = {s: torch.zeros(num_bins, device=device) for s in mc_sample_counts}
    totals_samples = 0

    pbar = tqdm(loader, desc="[laplace] Evaluating MC Sweep", file=sys.stdout)
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            bsz = images.size(0)
            totals_samples += bsz

            probs_sum = None
            for s in range(1, max_mc_samples + 1):
                # Sample perturbation
                perturbations = []
                if kfac_cov is not None:
                    # KFAC sampling for final linear layer
                    target_layer = [m for m in model.modules() if isinstance(m, nn.Linear) and m.weight.requires_grad][-1]
                    Z = torch.randn(target_layer.weight.shape, device=device, dtype=target_layer.weight.dtype)
                    delta_W = B_inv_sqrt @ Z @ A_inv_sqrt
                    target_layer.weight.add_(delta_W)
                elif laplace_std is not None:
                    # Diagonal sampling across trainable parameters
                    for idx, p in enumerate(trainable_params):
                        delta = torch.randn_like(p) * laplace_std[idx]
                        perturbations.append(delta)
                        p.add_(delta)

                image_features = model(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = (image_features @ classifier_weights.T) * 100.0
                probs = torch.softmax(logits, dim=-1)

                # Restore model parameters
                if kfac_cov is not None:
                    target_layer.weight.sub_(delta_W)
                elif laplace_std is not None:
                    for idx, p in enumerate(trainable_params):
                        p.sub_(perturbations[idx])

                probs_sum = probs if probs_sum is None else (probs_sum + probs)

                if s in targets:
                    avg_probs = probs_sum / s
                    log_avg_probs = avg_probs.clamp_min(1e-12).log()
                    loss = F.nll_loss(log_avg_probs, labels)
                    pred = avg_probs.argmax(dim=-1)

                    totals_loss[s] += loss.item() * bsz
                    totals_correct[s] += (pred == labels).sum().item()

                    one_hot = torch.zeros_like(avg_probs)
                    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
                    totals_brier[s] += ((avg_probs - one_hot) ** 2).sum(dim=1).sum().item()

                    conf, p_pred = avg_probs.max(dim=-1)
                    correct = (p_pred == labels).float()
                    bin_ids = torch.clamp((conf * num_bins).long(), max=num_bins - 1)
                    for b in range(num_bins):
                        mask = bin_ids == b
                        if mask.any():
                            count = mask.sum()
                            totals_bin_totals[s][b] += count
                            totals_bin_conf[s][b] += conf[mask].sum()
                            totals_bin_acc[s][b] += correct[mask].sum()

    mc_loss_acc_results = [
        (s, totals_loss[s] / totals_samples, totals_correct[s] / totals_samples)
        for s in mc_sample_counts
    ]

    mc_calibration_results = []
    for s in mc_sample_counts:
        ece = 0.0
        if totals_samples > 0:
            for b in range(num_bins):
                if totals_bin_totals[s][b] > 0:
                    acc = (totals_bin_acc[s][b] / totals_bin_totals[s][b]).item()
                    avg_conf = (totals_bin_conf[s][b] / totals_bin_totals[s][b]).item()
                    ece += abs(acc - avg_conf) * (totals_bin_totals[s][b].item() / totals_samples)
        brier = totals_brier[s] / totals_samples if totals_samples > 0 else 0.0
        mc_calibration_results.append((s, ece, brier))

    return mc_loss_acc_results, mc_calibration_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Laplace Approximation evaluation for fine-tuned CLIP models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="timm model name, e.g. vit_base_patch16_224.openai_clip",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Dataset to evaluate on.",
    )
    parser.add_argument("--seed", type=int, required=True, help="Global RNG seed.")
    parser.add_argument(
        "--optim",
        required=True,
        choices=["adamw", "sgd", "evon", "ivon", "soap"],
        help="Optimizer used during fine-tuning (for checkpoint lookup/logging).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Physical batch size.",
    )
    parser.add_argument(
        "--trainable-scope",
        type=str,
        default="all",
        choices=["all", "linear"],
        help="Trainable scope used during fine-tuning.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Explicit path to checkpoint file (.pt).",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch checkpoint number (optional).",
    )
    parser.add_argument(
        "--prior-precision",
        type=float,
        default=400.0,
        help="Laplace prior precision lambda initial value (default 400.0 matching BayesVLM).",
    )
    parser.add_argument(
        "--optimize-prior-precision",
        action="store_true",
        default=True,
        help="Automatically tune prior precision lambda by maximizing Laplace Marginal Likelihood (default: True, matching BayesVLM).",
    )
    parser.add_argument(
        "--no-optimize-prior-precision",
        action="store_false",
        dest="optimize_prior_precision",
        help="Disable automatic prior precision tuning.",
    )
    parser.add_argument(
        "--prior-opt-steps",
        type=int,
        default=300,
        help="Number of optimization steps for tuning prior precision (default 300 matching BayesVLM).",
    )
    parser.add_argument(
        "--prior-opt-lr",
        type=float,
        default=1e-2,
        help="Learning rate for prior precision Adam optimizer (default 1e-2 matching BayesVLM).",
    )
    parser.add_argument(
        "--var-scale",
        type=float,
        default=1.0,
        help="Variance scaling factor (multiplier) for sampled weight perturbations / variance propagation.",
    )
    parser.add_argument(
        "--eval-method",
        type=str,
        default="both",
        choices=["sampling", "probit", "both"],
        help="Evaluation method: 'sampling' (MC ensemble sampling), 'probit' (closed-form ProbCosine + Probit approximation), or 'both'.",
    )
    parser.add_argument(
        "--hessian-structure",
        type=str,
        default="kfac",
        choices=["diag", "kfac"],
        help="Hessian structure for Laplace approximation (default kfac matching BayesVLM).",
    )
    parser.add_argument(
        "--hessian-batch-size",
        type=int,
        default=5,
        help="Micro-batch size for Hessian/GGN estimation (default 5 matching BayesVLM la_batch_size).",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Maximum training samples to use for GGN estimation (default: None, use full dataset).",
    )
    parser.add_argument(
        "--wandb-project",
        default="clip-finetune-best-hparams",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="W&B entity (optional).",
    )
    args = parser.parse_args()

    # Reproducibility
    seed_everything(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using {device}")

    # Determine checkpoint path
    if args.checkpoint_path:
        ckpt_path = Path(args.checkpoint_path)
    else:
        if not args.model_name:
            raise ValueError("Either --checkpoint-path or --model-name must be specified.")
        ckpt_path = find_checkpoint(
            ckpt_root="checkpoints",
            model_name=args.model_name,
            seed=args.seed,
            optim=args.optim,
            dataset_name=args.dataset_name,
            epoch=args.epoch,
        )

    print(f"[ckpt] Loading checkpoint from {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location=device)

    model_name = args.model_name or ckpt.get("model_name")
    if not model_name:
        raise RuntimeError("Model name could not be determined from arguments or checkpoint.")

    dataset_name = args.dataset_name or ckpt.get("dataset_name")

    # Class names and CLIP classification head
    class_names = get_class_names(dataset_name)
    print(f"[dataset] Loaded {len(class_names)} classes for dataset {dataset_name!r}")

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

    # Load Model architecture and weights
    print(f"[model] Loading {model_name!r} ...")
    model, embed_dim, removed_text_attrs = load_timm_model(
        model_name,
        device,
        expected_embed_dim=int(classifier_weights.shape[1]),
    )
    freeze_for_finetune_fp(model, trainable_scope=args.trainable_scope)

    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[ckpt] Successfully loaded state dict into model.")

    # Dataloaders
    data_cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)

    train_loader, test_loader = get_dataloaders(
        dataset_name,
        batch_size=args.batch_size,
        transform=transform,
        seed=args.seed,
        num_workers=4,
    )

    # Initialize WandB
    run_name = f"laplace_{model_name}_{dataset_name}_seed{args.seed}_{args.optim}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            "model_name": model_name,
            "dataset_name": dataset_name,
            "seed": args.seed,
            "optim": f"laplace_{args.hessian_structure}",
            "original_optim": args.optim,
            "prior_precision": args.prior_precision,
            "optimize_prior_precision": args.optimize_prior_precision,
            "var_scale": args.var_scale,
            "eval_method": args.eval_method,
            "hessian_structure": args.hessian_structure,
            "hessian_batch_size": args.hessian_batch_size,
            "trainable_scope": args.trainable_scope,
            "checkpoint_path": str(ckpt_path),
            "device": str(device),
        },
    )

    # MAP Evaluation
    print("\n[map] Evaluating MAP model ...")
    test_loss_map, test_acc_map = _evaluate_map(model, test_loader, classifier_weights, device)
    ece_map, brier_map = _compute_calibration_metrics(model, test_loader, classifier_weights, device)

    print(f"  [MAP] Test Loss: {test_loss_map:.4f} | Test Acc: {100 * test_acc_map:.2f}% | ECE: {ece_map:.4f} | Brier: {brier_map:.4f}")

    wandb.log({
        "loss/test": test_loss_map,
        "acc/test": test_acc_map,
        "test_loss": test_loss_map,
        "test_acc": test_acc_map,
        "calibration_mean/ece": ece_map,
        "calibration_mean/brier": brier_map,
    })

    # Estimate Laplace Posterior
    laplace_std = None
    kfac_cov = None

    if args.hessian_structure == "kfac":
        print("\n[laplace] Estimating KFAC GGN Laplace posterior ...")
        A, B, kfac_samples = compute_kfac_ggn(
            model=model,
            loader=train_loader,
            classifier_weights=classifier_weights,
            device=device,
            max_samples=args.max_train_samples,
            micro_batch_size=args.hessian_batch_size,
        )
        kfac_cov = (A, B, kfac_samples)

        if args.optimize_prior_precision:
            target_layer = [m for m in model.modules() if isinstance(m, nn.Linear) and m.weight.requires_grad][-1]
            args.prior_precision = optimize_prior_precision_kfac(
                target_layer=target_layer,
                A=A,
                B=B,
                lmbda_init=args.prior_precision,
                n=kfac_samples,
                lr=args.prior_opt_lr,
                num_steps=args.prior_opt_steps,
                device=device,
            )
            wandb.config.update({"optimized_prior_precision": args.prior_precision}, allow_val_change=True)
    else:
        print("\n[laplace] Estimating Diagonal GGN Laplace posterior ...")
        diag_ggn = compute_diagonal_ggn(
            model=model,
            loader=train_loader,
            classifier_weights=classifier_weights,
            device=device,
            micro_batch_size=args.hessian_batch_size,
        )
        if args.optimize_prior_precision:
            args.prior_precision = optimize_prior_precision_diag(
                model=model,
                diag_ggn=diag_ggn,
                lmbda_init=args.prior_precision,
                lr=args.prior_opt_lr,
                num_steps=args.prior_opt_steps,
                device=device,
            )
            wandb.config.update({"optimized_prior_precision": args.prior_precision}, allow_val_change=True)

        # Compute std dev: sigma = 1 / sqrt(diag_ggn + prior_precision)
        laplace_std = [
            1.0 / torch.sqrt(g + args.prior_precision)
            for g in diag_ggn
        ]
        all_stds = torch.cat([s.flatten() for s in laplace_std])
        print(f"[laplace] Parameter perturbation std dev stats (raw): min={all_stds.min().item():.6e}, mean={all_stds.mean().item():.6e}, max={all_stds.max().item():.6e}")
        if args.var_scale != 1.0:
            print(f"[laplace] Applying variance scaling factor {args.var_scale:.4f} (scaled std mean: {all_stds.mean().item() * args.var_scale:.6e})")

    log_dict = {}

    # Probit Approximation Evaluation
    if args.eval_method in ("probit", "both"):
        print("\n[probit] Evaluating ProbCosine + Multiclass Probit approximation ...")
        loss_p, acc_p, ece_p, brier_p = _evaluate_laplace_probit(
            model=model,
            loader=test_loader,
            classifier_weights=classifier_weights,
            device=device,
            laplace_std=laplace_std,
            kfac_cov=kfac_cov,
            prior_precision=args.prior_precision,
            var_scale=args.var_scale,
        )
        print(f"  [Probit] Test Loss: {loss_p:.4f} | Test Acc: {100 * acc_p:.2f}% | ECE: {ece_p:.4f} | Brier: {brier_p:.4f}")
        log_dict.update({
            "probit/test_loss": loss_p,
            "probit/test_acc": acc_p,
            "probit/ece": ece_p,
            "probit/brier": brier_p,
        })
        if args.eval_method == "probit":
            log_dict.update({
                "loss/test": loss_p,
                "acc/test": acc_p,
                "calibration_mean/ece": ece_p,
                "calibration_mean/brier": brier_p,
            })

    # Monte Carlo Sample Sweep & Calibration Evaluation
    if args.eval_method in ("sampling", "both"):
        mc_counts = (1, 2, 4, 8, 16, 32)
        print(f"\n[mc] Evaluating test metrics vs Monte Carlo samples {mc_counts} ...")
        mc_loss_acc, mc_calib = _evaluate_laplace_mc(
            model=model,
            loader=test_loader,
            classifier_weights=classifier_weights,
            device=device,
            laplace_std=laplace_std,
            kfac_cov=kfac_cov,
            prior_precision=args.prior_precision,
            var_scale=args.var_scale,
            mc_sample_counts=mc_counts,
        )

        table = wandb.Table(columns=["mc_samples", "test_loss", "test_acc"])
        for (mc_s, mc_loss, mc_acc), (_, ece_s, brier_s) in zip(mc_loss_acc, mc_calib):
            table.add_data(mc_s, mc_loss, mc_acc)
            log_dict[f"mc_eval/test_loss_{mc_s}"] = mc_loss
            log_dict[f"mc_eval/test_acc_{mc_s}"] = mc_acc
            log_dict[f"calibration_mc{mc_s}/ece"] = ece_s
            log_dict[f"calibration_mc{mc_s}/brier"] = brier_s

        log_dict["mc_eval/table"] = table
        last_n, last_loss, last_acc = mc_loss_acc[-1]
        log_dict[f"mc_eval/test_loss_{last_n}"] = last_loss
        log_dict[f"mc_eval/test_acc_{last_n}"] = last_acc

    if log_dict:
        wandb.log(log_dict)

    wandb.finish()
    print("\n[done] Laplace approximation evaluation complete.")


if __name__ == "__main__":
    main()
