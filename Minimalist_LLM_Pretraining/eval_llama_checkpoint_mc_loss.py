#!/usr/bin/env python3
import argparse
import csv
import math
import os
import sys
from contextlib import nullcontext
from datetime import datetime
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

import datasets
import datasets.distributed
from transformers import AutoConfig, AutoTokenizer

from safetensors.torch import load_file
from mem_eff_pt.eff_pretraining import training_utils
from mem_eff_pt.eff_pretraining.modeling_llama import LlamaForCausalLM
from mem_eff_pt.utils.train_utils import build_optimizer, build_model
from vonsoap.eval_utils import (
    SimpleProgress,
    cast_dtype_from_string,
    move_optimizer_state_to_device,
    parse_mc_samples_list,
    sampled_params_context,
    str2bool,
)


torch.set_float32_matmul_precision("high")


def _default_val_files():
    return [f"c4-validation.{str(i).zfill(5)}-of-00008.json.gz" for i in range(0, 8)]


def _parse_val_files(val_files):
    if not val_files:
        return _default_val_files()
    return [part.strip() for part in val_files.split(",") if part.strip()]


def _normalize_cast_dtype(value):
    try:
        return cast_dtype_from_string(value)
    except ValueError:
        if isinstance(value, torch.dtype):
            return value
        raise


def _namespace_from_config(config):
    defaults = {
        "optimizer": "adamw",
        "lr": 1e-4,
        "lr_cov": 1e-2,
        "momentum": 0.9,
        "damping": 1e-8,
        "weight_decay": 0.0,
        "ess": 1e9,
        "ivon_hess_init": 1e-3,
        "ivon_clip_radius": float("inf"),
        "price_clip_ratio": None,
        "collect_stats": False,
        "decoupled_wd": False,
        "debias_second_moment": False,
        "max_precond_dim": 10000,
        "shampoo_beta": None,
        "evon_phased_grads": False,
        "evon_noise_damping": 0.0,
        "von_sync": True,
        "whiten_evon_grad": False,
        "cast_dtype": "float32",
        "adam_lr": 2e-2,
        "adam_beta_1": 0.9,
        "adam_beta_2": 0.999,
        "adam_damping": 1e-8,
        "adam_weight_decay": 0.0,
        "freq": 10,
    }
    merged = dict(defaults)
    merged.update(config or {})
    merged["cast_dtype"] = _normalize_cast_dtype(merged["cast_dtype"])
    return SimpleNamespace(**merged)


def _load_optimizer_state(model, optimizer_path, config):
    if not optimizer_path:
        return None
    if not os.path.exists(optimizer_path):
        raise FileNotFoundError(f"optimizer.pt not found: {optimizer_path}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    args_ns = _namespace_from_config(config)
    optimizers = build_optimizer(model, trainable_params, args_ns)
    variational_optimizer = next(
        (opt for opt in optimizers if hasattr(opt, "sampled_params")),
        None,
    )

    optimizer_checkpoint = torch.load(optimizer_path, map_location="cpu")
    optimizer_state = optimizer_checkpoint.get("optimizer", None)
    if optimizer_state is None:
        return None

    if variational_optimizer is not None:
        try:
            variational_optimizer.load_state_dict(optimizer_state)
            return variational_optimizer
        except Exception:
            pass

    return None


def _looks_like_sampleable_optimizer_state(optimizer_state):
    if not isinstance(optimizer_state, dict):
        return False
    param_groups = optimizer_state.get("param_groups")
    state = optimizer_state.get("state")
    if not isinstance(param_groups, list) or not isinstance(state, dict):
        return False
    if not param_groups:
        return False
  
    sampleable_keys = {
        "ess",
        "hess",
        "momentum",
        "clip_radius",
        "hess_init",
        "shampoo_beta",
        "precondition_frequency",
    }
    first_group = param_groups[0]
    if not isinstance(first_group, dict):
        return False
    return any(key in first_group for key in sampleable_keys)


def _load_variational_optimizer_from_checkpoint(model, checkpoint_path, config):
    if not checkpoint_path:
        return None, None

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "variational_optimizer" in checkpoint:
        optimizer_state = checkpoint["variational_optimizer"]
    elif "optimizers" in checkpoint and isinstance(checkpoint["optimizers"], list):
        optimizer_state = None
        for state_dict in checkpoint["optimizers"]:
            if _looks_like_sampleable_optimizer_state(state_dict):
                optimizer_state = state_dict
                break
        if optimizer_state is None and checkpoint["optimizers"]:
            optimizer_state = checkpoint["optimizers"][0]
    else:
        optimizer_state = checkpoint.get("optimizer")

    if not _looks_like_sampleable_optimizer_state(optimizer_state):
        return None, checkpoint

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    args_ns = _namespace_from_config(config)
    optimizers = build_optimizer(model, trainable_params, args_ns)
    variational_optimizer = next(
        (opt for opt in optimizers if hasattr(opt, "sampled_params")),
        None,
    )
    if variational_optimizer is None:
        return None, checkpoint

    # DEBUG: Compare param structure before loading
    if optimizer_state and "param_groups" in optimizer_state:
        ckpt_pg = optimizer_state["param_groups"][0]
        fresh_pg = variational_optimizer.param_groups[0]
        
        ckpt_params = ckpt_pg.get("params", [])
        fresh_params = fresh_pg.get("params", [])
        
        print(f"[DEBUG] Parameter mismatch check:")
        print(f"  Checkpoint has {len(ckpt_params)} params in param_groups[0]")
        print(f"  Fresh optimizer has {len(fresh_params)} params in param_groups[0]")
        
        if len(ckpt_params) != len(fresh_params):
            print(f"  PARAMETER COUNT MISMATCH - This will cause NaN!")
            
            # Try to diagnose the cause
            ckpt_numel = sum(ckpt_pg.get(k, torch.tensor(0)).numel() 
                            for k in ["hess", "momentum"] 
                            if isinstance(ckpt_pg.get(k), torch.Tensor))
            fresh_numel = sum(fresh_pg.get(k, torch.tensor(0)).numel() 
                             for k in ["hess", "momentum"] 
                             if isinstance(fresh_pg.get(k), torch.Tensor))
            print(f"    Checkpoint state elements: ~{ckpt_numel}")
            print(f"    Fresh state elements: ~{fresh_numel}")

    # Attempt to load state dict
    # If parameters don't match, PyTorch will initialize missing ones with NaN
    print(f"[DEBUG] Loading optimizer state...")
    try:
        variational_optimizer.load_state_dict(optimizer_state)
        print(f"[DEBUG] load_state_dict completed (strict=False)")
    except Exception as e:
        print(f"[DEBUG] load_state_dict failed: {e}")
        raise
    
    # DEBUG: Check state after loading for NaN
    print(f"[DEBUG] After load_state_dict:")
    if variational_optimizer.param_groups[0]:
        pg = variational_optimizer.param_groups[0]
        for key in ["momentum", "hess", "var"]:
            if key in pg and isinstance(pg[key], torch.Tensor):
                has_nan = torch.isnan(pg[key]).any().item()
                has_inf = torch.isinf(pg[key]).any().item()
                all_zero = (pg[key] == 0).all().item() if pg[key].numel() > 0 else False
                status = "X NaN" if has_nan else ("X Inf" if has_inf else ("!!! All Zero !!!" if all_zero else "Good."))
                print(f"  {key}: {status}, shape={pg[key].shape}, dtype={pg[key].dtype}")
    
    return variational_optimizer, checkpoint


def _get_ddp_state():
    ddp_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = ddp_world_size > 1
    ddp_rank = 0
    ddp_local_rank = 0
    if use_ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed eval requires CUDA.")
        dist.init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(ddp_local_rank)
    return use_ddp, ddp_rank, ddp_local_rank, ddp_world_size


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MC-BMA loss for LLaMA checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--optimizer_path", type=str, default="")
    parser.add_argument("--variational_optimizer_path", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--hf_dataset", action="store_true", default=False)
    parser.add_argument("--val_files", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--val_tokens", type=int, default=10_000_000)
    parser.add_argument("--mc_samples", type=int, default=int(os.environ.get("MC_SAMPLES", "10")))
    parser.add_argument(
        "--mc_samples_list",
        type=str,
        default=os.environ.get("MC_SAMPLES_LIST", ""),
        help="Comma-separated MC sample counts, e.g. 1,5,10,20",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--qkv_mode", type=str, default="")
    parser.add_argument("--attn_ratio", type=float, default=None)
    parser.add_argument(
        "--csv_out",
        type=str,
        default="",
        help="Optional CSV output path. Default writes to ../logs.",
    )
    parser.add_argument(
        "--append_csv",
        type=str2bool,
        default=False,
        help="Append to --csv_out if it exists instead of overwriting.",
    )
    parser.add_argument(
        "--run_label",
        type=str,
        default="",
        help="Optional label to distinguish rows in aggregated CSVs.",
    )
    parser.add_argument(
        "--progress",
        type=str2bool,
        default=True,
        help="Show evaluation progress (tqdm if installed, text fallback otherwise).",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=25,
        help="Fallback text progress update frequency in eval steps.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mc_values = parse_mc_samples_list(args.mc_samples_list, args.mc_samples)

    use_ddp, ddp_rank, ddp_local_rank, ddp_world_size = _get_ddp_state()

    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if use_ddp and not args.device.startswith("cuda"):
        raise RuntimeError("Distributed eval supports CUDA devices only.")

    if use_ddp:
        device = torch.device(f"cuda:{ddp_local_rank}")
    else:
        device = torch.device(args.device)

    master_process = ddp_rank == 0

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    if not os.path.isdir(checkpoint_dir):
        raise RuntimeError(f"Checkpoint directory not found: {checkpoint_dir}")

    config = AutoConfig.from_pretrained(checkpoint_dir)

    qkv_mode = args.qkv_mode.strip() or getattr(config, "qkv_mode", "single")
    attn_ratio = args.attn_ratio
    if attn_ratio is None:
        attn_ratio = getattr(config, "attn_ratio", 1.0)

    optimizer_path = args.optimizer_path.strip() or os.path.join(checkpoint_dir, "optimizer.pt")
    variational_optimizer_path = args.variational_optimizer_path.strip()

    optimizer_config = {}
    optimizer_checkpoint = None
    if os.path.exists(optimizer_path):
        optimizer_checkpoint = torch.load(optimizer_path, map_location="cpu")
        optimizer_config = optimizer_checkpoint.get("config", {})

    model_config = AutoConfig.from_pretrained(checkpoint_dir)
    model = LlamaForCausalLM(model_config, qkv_mode=qkv_mode, attn_ratio=attn_ratio)

    state = load_file(os.path.join(checkpoint_dir, "model.safetensors"))
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()

    # model, info = LlamaForCausalLM.from_pretrained(
    #     checkpoint_dir,
    #     config=config,
    #     qkv_mode=qkv_mode,
    #     attn_ratio=attn_ratio,
    #     output_loading_info=True 
    # )
    # model = model.to(device)
    # missing = sorted(info["missing_keys"])
    # unexpected = sorted(info["unexpected_keys"])
    # mismatched = sorted(info.get("mismatched_keys", []))

    # print("missing:", missing[:20], len(missing))
    # print("unexpected:", unexpected[:20], len(unexpected))
    # print("mismatched:", mismatched[:20], len(mismatched))
    # exit()

#    model = build_model(model.to(device=device), args)
    model_dtype = optimizer_checkpoint.get("dtype", optimizer_config.get("dtype", "float32"))

    if model_dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    elif model_dtype in ["fp16", "float16"]:
        model = model.to(device=device, dtype=torch.float16)
    else:
        model = model.to(device=device, dtype=torch.float32)

    use_autocast = False

    if not variational_optimizer_path:
        for candidate in (
            os.path.join(checkpoint_dir, "variational_optimizer.pt"),
            os.path.join(checkpoint_dir, "optimizer_variational.pt"),
            os.path.join(checkpoint_dir, "optimizer_0.pt"),
        ):
            if os.path.exists(candidate):
                variational_optimizer_path = candidate
                break



    variational_optimizer = None
    if variational_optimizer_path and os.path.exists(variational_optimizer_path):
        variational_optimizer, variational_checkpoint = _load_variational_optimizer_from_checkpoint(
            model,
            variational_optimizer_path,
            optimizer_config,
        )
        if variational_checkpoint is not None and not optimizer_config:
            optimizer_config = variational_checkpoint.get("config", {})
    elif optimizer_checkpoint is not None:
        variational_optimizer, _ = _load_variational_optimizer_from_checkpoint(
            model,
            optimizer_path,
            optimizer_config,
        )

    if variational_optimizer is not None:
        move_optimizer_state_to_device(variational_optimizer, device)
        
        # DEBUG: Check state after device move
        print(f"[DEBUG] After move_optimizer_state_to_device to {device}:")
        if variational_optimizer.param_groups[0]:
            pg = variational_optimizer.param_groups[0]
            for key in ["momentum", "hess", "var"]:
                if key in pg and isinstance(pg[key], torch.Tensor):
                    has_nan = torch.isnan(pg[key]).any().item()
                    status = "X NaN" if has_nan else "Good!"
                    print(f"  {key}: {status}, device={pg[key].device}")

    if variational_optimizer is None and max(mc_values) > 1:
        optimizer_name = str(optimizer_config.get("optimizer", "unknown")).lower()
        raise RuntimeError(
            f"Checkpoint at {checkpoint_dir} does not contain a sampleable variational optimizer state. "
            f"optimizer.pt appears to store only {optimizer_name or 'the non-variational optimizer'} state. "
            f"MC samples > 1 require a checkpoint that saved the variational optimizer separately."
        )

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = int(optimizer_config.get("batch_size", 64)) if optimizer_config else 64

    max_length = args.max_length
    if max_length is None:
        max_length = int(optimizer_config.get("max_length", config.max_position_embeddings))

    dataset_path = args.dataset_path or optimizer_config.get("dataset_path", "")
    val_files = _parse_val_files(args.val_files)

    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=max_length, use_fast=False)

    def preprocess_batched(batch):
        return tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    if args.hf_dataset:
        val_data = datasets.load_dataset("allenai/c4", "en", split="validation", streaming=True)
    else:
        if not dataset_path:
            raise RuntimeError("--dataset_path is required when not using --hf_dataset")
        data_files_val = {"validation": val_files}
        val_data = datasets.load_dataset(
            path=dataset_path,
            data_files=data_files_val,
            split="validation",
            streaming=True,
            )

    val_data = val_data.shuffle(seed=42)
    if use_ddp:
        val_data = datasets.distributed.split_dataset_by_node(
            val_data, rank=ddp_rank, world_size=ddp_world_size
        )

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(
        val_data_mapped, batch_size
    )

    eval_dtype = torch.float32 #_normalize_cast_dtype(optimizer_config.get("cast_dtype", "float32"))
    use_autocast = False #device.type == "cuda" and eval_dtype in (torch.float16, torch.bfloat16)

    def get_amp_ctx():
        if use_autocast:
            return torch.amp.autocast(device_type="cuda", dtype=eval_dtype)
        return nullcontext()

    mc_values_sorted = sorted(mc_values)
    mc_values_set = set(mc_values_sorted)
    max_mc = mc_values_sorted[-1]
    eval_steps = math.ceil(args.val_tokens / (batch_size * max_length * ddp_world_size))

    total_nll_by_mc = {mc: 0.0 for mc in mc_values_sorted}
    total_nll_mean = 0.0
    total_count = 0

    progress_enabled = args.progress and master_process
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    with torch.no_grad():
        if progress_enabled and tqdm is not None:
            progress = tqdm(total=eval_steps, desc="eval", dynamic_ncols=True)
        else:
            progress = SimpleProgress(
                total=eval_steps,
                desc="eval",
                enabled=progress_enabled,
                every=args.progress_every,
            )

        for batch in val_data_mapped.batch(batch_size=batch_size):
            if total_count >= args.val_tokens:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            targets = labels[..., 1:].contiguous()
            count = int((targets != -100).sum().item())

            with get_amp_ctx():
                logits_mean = model(**batch).logits

            log_probs_mean = F.log_softmax(logits_mean[..., :-1, :].float(), dim=-1)
            nll_sum_mean = F.nll_loss(
                log_probs_mean.view(-1, log_probs_mean.size(-1)),
                targets.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_nll_mean += float(nll_sum_mean.item())

            if variational_optimizer is None:
                for mc in mc_values_sorted:
                    total_nll_by_mc[mc] += float(nll_sum_mean.item())
            else:
                sum_probs = None
                for mc_idx in range(1, max_mc + 1):
                    with sampled_params_context(variational_optimizer, train=False):
                        with get_amp_ctx():
                            logits = model(**batch).logits
                        probs = F.softmax(logits.float(), dim=-1)
                        sum_probs = probs if sum_probs is None else (sum_probs + probs)

                    if mc_idx in mc_values_set:
                        avg_probs = sum_probs / mc_idx
                        log_avg_probs = avg_probs[..., :-1, :].clamp_min(1e-12).log()
                        nll_sum = F.nll_loss(
                            log_avg_probs.view(-1, log_avg_probs.size(-1)),
                            targets.view(-1),
                            ignore_index=-100,
                            reduction="sum",
                        )
                        total_nll_by_mc[mc_idx] += float(nll_sum.item())

            total_count += count
            progress.update(1)
        progress.close()

    if use_ddp:
        total_nll_mean_t = torch.tensor(total_nll_mean, device=device, dtype=torch.float64)
        total_count_t = torch.tensor(total_count, device=device, dtype=torch.float64)
        dist.all_reduce(total_nll_mean_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_count_t, op=dist.ReduceOp.SUM)
        total_nll_mean = float(total_nll_mean_t.item())
        total_count = int(total_count_t.item())

        for mc in mc_values_sorted:
            total_nll_t = torch.tensor(total_nll_by_mc[mc], device=device, dtype=torch.float64)
            dist.all_reduce(total_nll_t, op=dist.ReduceOp.SUM)
            total_nll_by_mc[mc] = float(total_nll_t.item())

    if total_count == 0:
        raise RuntimeError("No valid targets encountered (all ignore_index).")

    mc_bma_nll_by_mc = {mc: total_nll_by_mc[mc] / total_count for mc in mc_values_sorted}
    mean_posterior_nll = total_nll_mean / total_count

    optimizer_name = str(optimizer_config.get("optimizer", "unknown")).lower()
    step = "unknown"
    training_state_path = os.path.join(checkpoint_dir, "training_state.json")
    if os.path.exists(training_state_path):
        try:
            import json

            with open(training_state_path, "r") as f:
                training_state = json.load(f)
            step = training_state.get("update_step", training_state.get("global_step", "unknown"))
        except Exception:
            pass

    checkpoint_basename = os.path.basename(checkpoint_dir.rstrip("/"))
    run_label = args.run_label.strip() if args.run_label else optimizer_name
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_logs_dir = os.path.abspath(os.path.join(script_dir, "..", "logs"))
    os.makedirs(default_logs_dir, exist_ok=True)

    if args.csv_out.strip():
        csv_out = args.csv_out
        csv_dir = os.path.dirname(os.path.abspath(csv_out))
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_out = os.path.join(default_logs_dir, f"mc_bma_llama_{checkpoint_basename}_{ts}.csv")

    fieldnames = [
        "checkpoint",
        "checkpoint_basename",
        "optimizer",
        "run_label",
        "step",
        "mc_samples",
        "val_tokens",
        "mc_bma_nll",
        "mean_posterior_nll",
    ]

    file_exists = os.path.exists(csv_out)
    write_header = not (args.append_csv and file_exists)
    mode = "a" if args.append_csv else "w"

    if master_process:
        with open(csv_out, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for mc in mc_values:
                writer.writerow(
                    {
                        "checkpoint": checkpoint_dir,
                        "checkpoint_basename": checkpoint_basename,
                        "optimizer": optimizer_name,
                        "run_label": run_label,
                        "step": step,
                        "mc_samples": mc,
                        "val_tokens": args.val_tokens,
                        "mc_bma_nll": f"{mc_bma_nll_by_mc[mc]:.8f}",
                        "mean_posterior_nll": f"{mean_posterior_nll:.8f}",
                    }
                )

        print("checkpoint:", checkpoint_dir)
        print("optimizer:", optimizer_name)
        print("step:", step)
        print("mc_samples_list:", ",".join(str(v) for v in mc_values))
        print("val_tokens:", args.val_tokens)
        print("world_size:", ddp_world_size)
        print("mean_posterior_nll:", f"{mean_posterior_nll:.8f}")
        for mc in mc_values:
            print(f"mc_bma_nll@{mc}:", f"{mc_bma_nll_by_mc[mc]:.8f}")
        print("csv_out:", csv_out)

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
