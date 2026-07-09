#!/usr/bin/env python3
import argparse
import csv
import glob
import math
import os
import sys
import time
from datetime import datetime
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn

from vonsoap.optimizers import IVON, EVON
from vonsoap.eval_utils import (
    SimpleProgress,
    cast_dtype_from_string,
    move_optimizer_state_to_device,
    parse_mc_samples_list,
    sampled_params_context,
    str2bool,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


torch.set_float32_matmul_precision("high")


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_ratio, is_merged):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.merged = is_merged

        self.hidden_dim = int(self.n_embd * attn_ratio)
        self.head_dim = self.hidden_dim // self.n_head
        assert self.hidden_dim % self.n_head == 0

        if self.merged:
            self.c_qkv = nn.Linear(self.n_embd, 3 * self.hidden_dim, bias=False)
        else:
            self.c_q = nn.Linear(self.n_embd, self.hidden_dim, bias=False)
            self.c_k = nn.Linear(self.n_embd, self.hidden_dim, bias=False)
            self.c_v = nn.Linear(self.n_embd, self.hidden_dim, bias=False)

        self.c_proj = nn.Linear(self.hidden_dim, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_()
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        bsz, seq_len, _ = x.size()

        if self.merged:
            qkv = self.c_qkv(x)
            q, k, v = qkv.split((self.hidden_dim, self.hidden_dim, self.hidden_dim), dim=-1)
            q = q.view(bsz, seq_len, self.n_head, self.head_dim)
            k = k.view(bsz, seq_len, self.n_head, self.head_dim)
            v = v.view(bsz, seq_len, self.n_head, self.head_dim)
        else:
            q = self.c_q(x).view(bsz, seq_len, self.n_head, self.head_dim)
            k = self.c_k(x).view(bsz, seq_len, self.n_head, self.head_dim)
            v = self.c_v(x).view(bsz, seq_len, self.n_head, self.head_dim)

        cos, sin = self.rotary(q)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, n_embd, mlp_ratio):
        super().__init__()
        hidden_dim = n_embd * mlp_ratio
        self.c_fc = nn.Linear(n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, n_embd, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_head, mlp_ratio, attn_ratio, is_merged):
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_head, attn_ratio, is_merged)
        self.mlp = MLP(n_embd, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, n_embd, mlp_ratio, attn_ratio, is_merged):
        super().__init__()
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, n_embd),
                h=nn.ModuleList(
                    [Block(n_embd, n_head, mlp_ratio, attn_ratio, is_merged) for _ in range(n_layer)]
                ),
            )
        )
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))

        logits = self.lm_head(x)
        logits = logits.float()

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss


def _peek_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        raise RuntimeError(f"Magic number mismatch in shard: {filename}")
    if header[1] != 1:
        raise RuntimeError(f"Unsupported shard version in {filename}: {header[1]}")
    return int(header[2])


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        if header[0] != 20240520:
            raise RuntimeError(f"Magic number mismatch in shard: {filename}")
        if header[1] != 1:
            raise RuntimeError(f"Unsupported shard version in {filename}: {header[1]}")
        ntok = int(header[2])
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    if len(tokens) != ntok:
        raise RuntimeError(
            f"Token count mismatch in {filename}: got {len(tokens)} expected {ntok}"
        )
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, batch_size, sequence_length, device, process_rank=0, num_processes=1):
        self.device = device
        self.B = batch_size
        self.T = sequence_length
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.files = sorted(glob.glob(filename_pattern))
        if not self.files:
            raise RuntimeError(f"No files matched: {filename_pattern}")
        min_ntok = self.num_processes * self.B * self.T + 1
        for fname in self.files:
            ntok = _peek_data_shard(fname)
            if ntok < min_ntok:
                raise RuntimeError(
                    f"Shard too small for world_size={self.num_processes}, batch={self.B}, seq={self.T}: {fname}"
                )
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def _advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        needed = self.B * self.T + 1
        if self.current_position + needed > len(self.tokens):
            self._advance()

        buf = self.tokens[self.current_position : self.current_position + needed]
        x = torch.tensor(buf[:-1].astype(np.int32), dtype=torch.long, device=self.device).view(self.B, self.T)
        y = torch.tensor(buf[1:].astype(np.int32), dtype=torch.long, device=self.device).view(self.B, self.T)

        self.current_position += self.B * self.T * self.num_processes
        if self.current_position + needed > len(self.tokens):
            self._advance()
        return x, y


def normalize_state_dict_keys(state_dict):
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("_orig_mod.") for k in keys):
        return {k[len("_orig_mod.") :]: v for k, v in state_dict.items()}
    return state_dict


def build_sampling_optimizer(model, ckpt_args, opt_name):
    if "ivon" in opt_name:
        return IVON(
            model.transformer.h.parameters(),
            lr=float(ckpt_args.get("lr", 2e-2)),
            beta1=float(ckpt_args.get("momentum", 0.9)),
            beta2=1.0 - float(ckpt_args.get("lr_cov", 2e-2)),
            ess=float(ckpt_args.get("ess", 1e9)),
            hess_init=float(ckpt_args.get("ivon_hess_init", 0.001)),
            clip_radius=float(ckpt_args.get("ivon_clip_radius", float("inf"))),
            weight_decay=float(ckpt_args.get("weight_decay", 1e-4)),
            sync=bool(ckpt_args.get("von_sync", True)),
        )

    if "evon" in opt_name:
        shampoo_beta = ckpt_args.get("shampoo_beta", None)
        cast_dtype = cast_dtype_from_string(ckpt_args.get("cast_dtype", "bfloat16"))
        return EVON(
            model.transformer.h.parameters(),
            lr=float(ckpt_args.get("lr", 2e-2)),
            betas=(
                float(ckpt_args.get("momentum", 0.9)),
                1.0 - float(ckpt_args.get("lr_cov", 2e-2)),
            ),
            hess_init=float(ckpt_args.get("ivon_hess_init", 0.001)),
            precondition_frequency=int(ckpt_args.get("T", 10)),
            prec_clip_radius=float(ckpt_args.get("ivon_clip_radius", float("inf"))),
            upd_grad_clip_radius=float(ckpt_args.get("ivon_clip_radius", float("inf"))),
            decoupled_wd=bool(ckpt_args.get("decoupled_wd", False)),
            debias_second_moment=bool(ckpt_args.get("debias_second_moment", False)),
            ess=float(ckpt_args.get("ess", 1e9)),
            correct_bias=True,
            eps=float(ckpt_args.get("damping", 1e-8)),
            max_precond_dim=int(ckpt_args.get("max_precond_dim", 10000)),
            weight_decay=float(ckpt_args.get("weight_decay", 1e-4)),
            precondition_1d=False,
            cast_dtype=cast_dtype,
            shampoo_beta=-1 if shampoo_beta is None else float(shampoo_beta),
            enable_alternating_grads=bool(ckpt_args.get("evon_phased_grads", False)),
            price_clip_ratio=ckpt_args.get("price_clip_ratio", None),
            collect_clip_stats=bool(ckpt_args.get("collect_stats", False)),
            noise_damping=float(ckpt_args.get("evon_noise_damping", 0.0)),
            sync=bool(ckpt_args.get("von_sync", True)),
        )

    raise ValueError(f"Only EVON and IVON are supported. Got opt={opt_name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MC-BMA loss from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_val_bin", type=str, default="")
    parser.add_argument("--batch_size_pre_gpu", type=int, default=None)
    parser.add_argument("--sequence_length", type=int, default=1024)
    parser.add_argument("--val_tokens", type=int, default=10_485_760)
    parser.add_argument("--mc_samples", type=int, default=int(os.environ.get("MC_SAMPLES", "10")))
    parser.add_argument(
        "--mc_samples_list",
        type=str,
        default=os.environ.get("MC_SAMPLES_LIST", ""),
        help="Comma-separated MC sample counts, e.g. 1,5,10,20",
    )
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
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--vocab_size", type=int, default=50304)
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

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "args" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain saved args. Re-save with updated train_gpt2.py checkpointing."
        )
    ckpt_args = checkpoint["args"]

    opt_name = str(ckpt_args.get("opt", "")).lower()
    if "evon" not in opt_name and "ivon" not in opt_name:
        raise ValueError(f"Checkpoint opt must be EVON or IVON. Found: {opt_name}")

    input_val_bin = args.input_val_bin or ckpt_args.get("input_val_bin", "data/fineweb10B/fineweb_val_*.bin")
    batch_size = args.batch_size_pre_gpu
    if batch_size is None:
        batch_size = int(ckpt_args.get("batch_size_pre_gpu", 64))

    n_head = int(ckpt_args.get("n_head", 6))
    n_embd = int(ckpt_args.get("n_embd", 768))
    mlp_ratio = int(ckpt_args.get("mlp_ratio", 4))
    attn_ratio = float(ckpt_args.get("attn_ratio", 1))
    is_merged = str2bool(ckpt_args.get("is_merged", False))

    model = GPT(
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=n_head,
        n_embd=n_embd,
        mlp_ratio=mlp_ratio,
        attn_ratio=attn_ratio,
        is_merged=is_merged,
    ).to(device)

    model_state = normalize_state_dict_keys(checkpoint["model"])
    model.load_state_dict(model_state, strict=True)
    model.eval()

    if "optimizers" not in checkpoint or not checkpoint["optimizers"]:
        raise RuntimeError("Checkpoint does not contain optimizer states.")

    optimizer2 = build_sampling_optimizer(model, ckpt_args, opt_name)
    optimizer_state = checkpoint["optimizers"][-1]
    optimizer2.load_state_dict(optimizer_state)
    move_optimizer_state_to_device(optimizer2, device)

    loader = DistributedDataLoader(
        filename_pattern=input_val_bin,
        batch_size=batch_size,
        sequence_length=args.sequence_length,
        device=device,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
    )
    loader.reset()

    mc_values_sorted = sorted(mc_values)
    mc_values_set = set(mc_values_sorted)
    max_mc = mc_values_sorted[-1]
    eval_steps = math.ceil(args.val_tokens / (batch_size * args.sequence_length * ddp_world_size))

    total_nll_by_mc = {mc: 0.0 for mc in mc_values_sorted}
    total_nll_mean = 0.0
    total_count = 0
    eval_dtype = cast_dtype_from_string(ckpt_args.get("cast_dtype", "float32"))
    use_autocast = device.type == "cuda" and eval_dtype in (torch.float16, torch.bfloat16)
    def get_amp_ctx():
        if use_autocast:
            return torch.amp.autocast(device_type="cuda", dtype=eval_dtype)
        return nullcontext()

    with torch.no_grad():
        progress_enabled = args.progress and master_process
        if progress_enabled and tqdm is not None:
            progress = tqdm(total=eval_steps, desc="eval", dynamic_ncols=True)
        else:
            progress = SimpleProgress(
                total=eval_steps,
                desc="eval",
                enabled=progress_enabled,
                every=args.progress_every,
            )
        for _ in range(eval_steps):
            x_val, y_val = loader.next_batch()
            sum_probs = None
            targets = y_val.view(-1)
            count = int((targets != -1).sum().item())

            with get_amp_ctx():
                logits_mean, _ = model(x_val, y_val)
            log_probs_mean = F.log_softmax(logits_mean, dim=-1)
            nll_sum_mean = F.nll_loss(
                log_probs_mean.view(-1, log_probs_mean.size(-1)),
                targets,
                ignore_index=-1,
                reduction="sum",
            )
            total_nll_mean += float(nll_sum_mean.item())

            for mc_idx in range(1, max_mc + 1):
                with sampled_params_context(optimizer2):
                    with get_amp_ctx():
                        logits, _ = model(x_val, y_val)
                    probs = F.softmax(logits, dim=-1)
                    sum_probs = probs if sum_probs is None else (sum_probs + probs)

                if mc_idx in mc_values_set:
                    avg_probs = sum_probs / mc_idx
                    log_avg_probs = avg_probs.clamp_min(1e-12).log()
                    nll_sum = F.nll_loss(
                        log_avg_probs.view(-1, log_avg_probs.size(-1)),
                        targets,
                        ignore_index=-1,
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

    mc_bma_nll_by_mc = {
        mc: total_nll_by_mc[mc] / total_count for mc in mc_values_sorted
    }
    mean_posterior_nll = total_nll_mean / total_count

    step = checkpoint.get("step", "unknown")
    checkpoint_basename = os.path.basename(args.checkpoint)
    run_label = args.run_label.strip() if args.run_label else opt_name
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_logs_dir = os.path.abspath(os.path.join(script_dir, "..", "logs"))
    os.makedirs(default_logs_dir, exist_ok=True)

    if args.csv_out.strip():
        csv_out = args.csv_out
        csv_dir = os.path.dirname(os.path.abspath(csv_out))
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
    else:
        ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_out = os.path.join(default_logs_dir, f"mc_bma_{ckpt_stem}_{ts}.csv")

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
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )
            if write_header:
                writer.writeheader()
            for mc in mc_values:
                writer.writerow(
                    {
                        "checkpoint": args.checkpoint,
                        "checkpoint_basename": checkpoint_basename,
                        "optimizer": opt_name,
                        "run_label": run_label,
                        "step": step,
                        "mc_samples": mc,
                        "val_tokens": args.val_tokens,
                        "mc_bma_nll": f"{mc_bma_nll_by_mc[mc]:.8f}",
                        "mean_posterior_nll": f"{mean_posterior_nll:.8f}",
                    }
                )

        print("checkpoint:", args.checkpoint)
        print("optimizer:", opt_name)
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
