import os
import sys
import math

with open(sys.argv[0]) as f:
    code = f.read()  # read the code of this file ASAP, for logging
import uuid
import glob
import time
from dataclasses import dataclass
from math import cos, pi
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP


from vonsoap.optimizers import SOAP, AdamWBF16, DistributedMuon, KLOpt, IVON
from vonsoap.optimizers import NewEVON, OldEVON

try:
    import wandb
except ImportError:
    wandb = None

import socket
import argparse

torch.set_float32_matmul_precision("high")


def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")



# PyTorch nn.Module definitions for the GPT-2 model
config_parser = parser = argparse.ArgumentParser(
    description="Training Config", add_help=False
)
parser.add_argument(
    "--experiment",
    default="nanoGPT",
    type=str,
    metavar="NAME",
    help="name of train experiment, name of sub-folder for output",
)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--opt", type=str)
parser.add_argument("--damping", default=1e-8, type=float)
parser.add_argument("--T", default=10, type=int)
parser.add_argument("--lr_cov", default=2e-2, type=float)
parser.add_argument("--init_factor", default=1.0, type=float)
parser.add_argument("--lr", default=2e-2, type=float)
parser.add_argument("--weight_decay", default=1e-4, type=float)
parser.add_argument("--batch_size_pre_gpu", default=128, type=int)
parser.add_argument("--total_batch_size", default=512, type=int)
parser.add_argument("--num_iterations", default=10000, type=int)
parser.add_argument("--n_embd", default=768, type=int)
parser.add_argument("--mlp_ratio", default=4, type=int)
parser.add_argument("--attn_ratio", default=1, type=float)
parser.add_argument("--n_head", default=6, type=int)
parser.add_argument("--is_merged", type=str2bool, default=False)
parser.add_argument("--no_lr_rescale", type=str2bool, default=False)
parser.add_argument("--warmup_iters", default=100, type=int)
parser.add_argument("--warmdown_iters", default=1450, type=int)
parser.add_argument("--val_loss_every", default=125, type=int)
parser.add_argument("--schd", default="linear", type=str)
parser.add_argument("--block_factor", default=4, type=int)
parser.add_argument("--block_frequency", default=4, type=int)
parser.add_argument("--ess", default=1e9, type=float)
parser.add_argument("--ess_min", default=None, type=float)
parser.add_argument("--ess_min_fac", default=None, type=float)
parser.add_argument("--ess_anneal_steps", default=8000, type=float)
parser.add_argument("--ivon_hess_init", default=0.001, type=float)
parser.add_argument("--ivon_clip_radius", default=float("inf"), type=float)
parser.add_argument("--price_clip_ratio", default=None, type=float)
parser.add_argument("--collect_stats", type=str2bool, default=False)
parser.add_argument("--max_precond_dim", default=10000, type=int)
parser.add_argument("--shampoo_beta", default=None, type=float)
parser.add_argument("--evon_phased_grads", type=str2bool, default=False)
parser.add_argument("--debias_second_moment", type=str2bool, default=False)
parser.add_argument(
    "--von_sync",
    type=str2bool,
    default=True,
    help="Enable distributed sync inside IVON/EVON and disable DDP grad all-reduce in backward.",
)
parser.add_argument("--cast_dtype", default="bfloat16", type=str)
parser.add_argument("--disable_early_stop", default=False, type=str2bool)
parser.add_argument("--whiten_evon_grad", default=False, type=str2bool)
parser.add_argument("--mc_samples", default=1, type=int)
parser.add_argument(
    "--save_every",
    default=0,
    type=int,
    help="Save a training checkpoint every N loop steps (0 = only final save).",
)
parser.add_argument(
    "--checkpoint_dir",
    default="",
    type=str,
    help="Directory for checkpoints. Defaults to checkpoints/<experiment>.",
)
parser.add_argument(
    "--resume_from",
    default="",
    type=str,
    help="Checkpoint path to resume from.",
)
parser.add_argument(
    "--diag_every",
    type=int,
    default=50,
    help="Log h_mom scalar stats (max/min/mean per param) every N training steps (0 = disable).",
)
parser.add_argument(
    "--hess_hist_freq",
    type=int,
    default=1000,
    help="Log full h_mom histograms per parameter to wandb every N training steps (0 = disable). "
         "Also always logged at end-of-run when --collect_stats is true.",
)

args0, _ = config_parser.parse_known_args()
print(args0)

if args0.cast_dtype == "bfloat16":
    args0.cast_dtype = torch.bfloat16
elif args0.cast_dtype == "float16":
    args0.cast_dtype = torch.float16
elif args0.cast_dtype == "float32":
    args0.cast_dtype = torch.float32
else:
    raise ValueError(f"Unsupported cast_dtype: {args0.cast_dtype}")


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
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.merged = config.is_merged

        self.hidden_dim = int(self.n_embd * config.attn_ratio)
        self.head_dim = self.hidden_dim // self.n_head
        assert self.hidden_dim % self.n_head == 0

        if self.merged:
            self.c_qkv = nn.Linear(self.n_embd, 3 * self.hidden_dim, bias=False)
        else:
            self.c_q = nn.Linear(self.n_embd, self.hidden_dim, bias=False)
            self.c_k = nn.Linear(self.n_embd, self.hidden_dim, bias=False)
            self.c_v = nn.Linear(self.n_embd, self.hidden_dim, bias=False)

        # output projection
        self.c_proj = nn.Linear(self.hidden_dim, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_()  # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        if self.merged:
            qkv = self.c_qkv(x)
            q, k, v = qkv.split(
                (self.hidden_dim, self.hidden_dim, self.hidden_dim), dim=-1
            )
            q = q.view(B, T, self.n_head, self.head_dim)
            k = k.view(B, T, self.n_head, self.head_dim)
            v = v.view(B, T, self.n_head, self.head_dim)
        else:
            q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
            k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
            v = self.c_v(x).view(B, T, self.n_head, self.head_dim)

        cos, sin = self.rotary(q)
        q, k = (
            F.rms_norm(q, (q.size(-1),)),
            F.rms_norm(k, (k.size(-1),)),
        )  # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        y = (
            y.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)
        )  # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.n_embd * config.mlp_ratio
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()  # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(
            x
        ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x


# -----------------------------------------------------------------------------
# The main GPT-2 model


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6  # head dim 128 suggested by @Grad62304977
    n_embd: int = 768
    mlp_ratio: int = 4
    attn_ratio: float = 1
    is_merged: bool = False


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

    def forward(self, idx, targets=None, return_logits=True):

        # forward the GPT model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float()  # use tf32/fp32 for logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            logits = logits.float()  # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss


# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader


def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print(
            "---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README"
        )
        print(
            "---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try"
        )
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]  # number of tokens (claimed)
    return ntok  # for now just return the number of tokens


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]  # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, (
            f"did not find any files that match the pattern {filename_pattern}"
        )

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()


# -----------------------------------------------------------------------------
# int main


@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin: str = "data/fineweb10B/fineweb_train_*.bin"  # input .bin to train on
    input_val_bin: str = (
        "data/fineweb10B/fineweb_val_*.bin"  # input .bin to eval validation loss on
    )
    # optimization hyperparams
    batch_size: int = (
        args0.total_batch_size
    )  # batch size, in sequences, across all devices
    device_batch_size: int = (
        args0.batch_size_pre_gpu
    )  # batch size, in sequences, per device
    sequence_length: int = 1024  # sequence length, in tokens
    num_iterations: int = args0.num_iterations  # number of iterations to run
    adamw_learning_rate: float = 0.0036
    adamw_weight_decay: float = 0
    warmup_iters: int = args0.warmup_iters  # 100
    warmdown_iters: int = args0.warmdown_iters  # 1450 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    # evaluation and logging hyperparams
    val_loss_every: int = (
        args0.val_loss_every
    )  # 125 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens: int = 10485760  # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every: int = args0.save_every
    checkpoint_dir: str = args0.checkpoint_dir
    resume_from: str = args0.resume_from

    opt_name: str = args0.opt
    experiment: str = args0.experiment
    damping: float = args0.damping
    T: int = args0.T
    lr_cov: float = args0.lr_cov
    lr: float = args0.lr
    momentum: float = args0.momentum
    weight_decay: float = args0.weight_decay
    ess: float = args0.ess
    ess_min: float = args0.ess_min
    ess_min_fac: float = args0.ess_min_fac
    ess_anneal_steps: int = args0.ess_anneal_steps
    ivon_hess_init: float = args0.ivon_hess_init
    ivon_clip_radius: float = args0.ivon_clip_radius
    price_clip_ratio: float = args0.price_clip_ratio
    debias_second_moment: bool = args0.debias_second_moment
    max_precond_dim: int = args0.max_precond_dim
    cast_dtype: torch.dtype = args0.cast_dtype
    shampoo_beta: float = args0.shampoo_beta
    disable_early_stop: bool = args0.disable_early_stop
    evon_phased_grads: bool = args0.evon_phased_grads
    collect_stats: bool = args0.collect_stats
    mc_samples: int = args0.mc_samples
    von_sync: bool = args0.von_sync
    no_lr_rescale: bool = args0.no_lr_rescale
    whiten_evon_grad: bool = args0.whiten_evon_grad


args = Hyperparameters()

if args.save_every < 0:
    raise ValueError("--save_every must be >= 0")

# set up DDP (distributed data parallel). torchrun sets this env variable
# assert torch.cuda.is_available()  
dist.init_process_group(backend="nccl")
ddp_rank = int(os.environ["RANK"])
ddp_local_rank = int(os.environ["LOCAL_RANK"])
ddp_world_size = int(os.environ["WORLD_SIZE"])
device = f"cuda:{ddp_local_rank}"
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.


if args.opt_name.find("muon") >= 0 or args.opt_name.find("adam") >= 0:
    args.T = 1
run_name = "%s-T%d-%s-%s" % (args.opt_name, args.T, args0.schd, socket.gethostname())
if master_process and wandb is not None:
    wandb.require("core")
    run = wandb.init(
        project=args.experiment, name=run_name, tags=["normal", "hess_hist"] if args.collect_stats else ["normal"], entity="adrianrob1-Sapienza Università di Roma", config=args
    )

checkpoint_dir = args.checkpoint_dir.strip() or os.path.join("checkpoints", args.experiment)

# convenience variables
B, T = args.device_batch_size, args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)
print("-----------------val steps is--------------------", val_steps)

# load tokens
train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
if master_process:
    print(
        f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files"
    )
    print(
        f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files"
    )
x, y = train_loader.next_batch()

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
num_vocab = 50304
model = GPT(
    GPTConfig(
        vocab_size=num_vocab,
        n_layer=12,
        n_head=args0.n_head,
        n_embd=args0.n_embd,
        mlp_ratio=args0.mlp_ratio,
        attn_ratio=args0.attn_ratio,
        is_merged=args0.is_merged,
    )
)
model = model.cuda()
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True  # suggested by @Chillee
model = torch.compile(model)
# here we wrap model into DDP container
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module  # always contains the "raw" unwrapped model

# _act_stats is populated by forward hooks when they are enabled.
# Hooks are currently disabled (OOM risk with torch.compile), so this
# dict stays empty; the logging loop in do_diag is a safe no-op.
_act_stats: dict = {}

ctx = torch.amp.autocast(
    device_type="cuda", dtype=torch.bfloat16
)  # keep model in bfloat16

optimizer2 = None

n_params = 0
for name, param in raw_model.named_parameters():
    if param.requires_grad:
        n_params += param.numel()
        print(name, param.shape)
        if len(param.shape) > 1:
            ratio = max(*param.shape) * 1.0 / min(*param.shape)
            print("ratio:", ratio)
print("model params", n_params)

if args.opt_name.find("adamw") >= 0:
    print("using org adamw")
    normalize_grads = False
    if args.opt_name.find("_norm") > 0:
        print("using norm")
        normalize_grads = True
    optimizer1 = AdamWBF16(
        raw_model.lm_head.parameters(),
        lr=args.lr,
        betas=(args.momentum, 1.0 - args.lr_cov),
        eps=args.damping,
        weight_decay=args.weight_decay,
        cast_dtype=torch.bfloat16,
        is_normalize=normalize_grads,
    )
    optimizer2 = AdamWBF16(
        raw_model.transformer.h.parameters(),
        lr=args.lr,
        betas=(args.momentum, 1.0 - args.lr_cov),
        eps=args.damping,
        weight_decay=args.weight_decay,
        cast_dtype=args.cast_dtype,
        is_normalize=normalize_grads,
    )
else:
    print(
        "the optimal adamw hyperparams for training 1d weight vectors and the embedding weight matrix"
    )
    if (
        args0.n_head == 6
        and args0.n_embd == 768
        and args0.mlp_ratio == 4
        and args0.attn_ratio == 1
    ):
        if args0.total_batch_size == 8:
            adamw_learning_rate = 0.0005501648906795802
            adamw_betas = (0.9345713572610516, 1.0 - 0.0004140037648329575)
            adamw_weight_decay = 4.263205973949167e-05
            adamw_damping = 3.2417242349947586e-11
        else:
            adamw_learning_rate = 0.0011744186743895023
            adamw_betas = (0.8909389870437245, 1.0 - 0.03063014733759629)
            adamw_weight_decay = 1.7075323105504728e-06
            adamw_damping = 8.602957455024406e-08

    elif (
        args0.n_head == 5
        and args0.n_embd == 640
        and args0.mlp_ratio == 5
        and args0.attn_ratio == 1.5
    ):
        adamw_learning_rate = 0.0018272023762808387
        adamw_betas = (0.9176078725520668, 1.0 - 0.024102453063468643)
        adamw_weight_decay = 4.591442768888015e-05
        adamw_damping = 2.5806022118890508e-11
    else:
        assert False

    param_groups = {}
    params = []
    merge_name = set()

    for name, param in raw_model.transformer.h.named_parameters():
        idx = name.find("c_")
        if idx > 0:
            if args.opt_name.find("_3dmerged_qk") > 0:
                if name[idx:].find("c_q.") == 0 or name[idx:].find("c_k.") == 0:
                    key = name[:idx]
                    merge_name.add(key)
                else:
                    key = name
            elif args.opt_name.find("_3dmerged") > 0:
                if (
                    name[idx:].find("c_q.") == 0
                    or name[idx:].find("c_k.") == 0
                    or name[idx:].find("c_v.") == 0
                ):
                    key = name[:idx]
                    merge_name.add(key)
                else:
                    key = name
            else:
                key = name
        else:
            key = name
        list_info = param_groups.setdefault(key, [])
        list_info.append(param)
    for name, info in param_groups.items():
        merged = False
        if name in merge_name:
            merged = True
        params.append({"params": info, "merged": merged})

    optimizer1 = AdamWBF16(
        raw_model.lm_head.parameters(),
        lr=adamw_learning_rate,
        betas=adamw_betas,
        weight_decay=adamw_weight_decay,
        eps=adamw_damping,
        cast_dtype=torch.bfloat16,
        is_normalize=False,
    )
    print("adamw opt", optimizer1)

my_lr = args.lr
if args.opt_name.find("dmuon") >= 0:
    if args.opt_name.find("_polar") > 0:
        backend = "polar"
    else:
        backend = "newtonschulz"
    optimizer2 = DistributedMuon(
        raw_model.transformer.h.parameters(),
        lr=my_lr,
        momentum=args.momentum,
        nesterov=True,
        ns_steps=5,
        weight_decay=args.weight_decay,
        backend=backend,
        adamw_betas=adamw_betas,
        adamw_eps=adamw_damping,
        muon_eps=args.damping,
    )

elif args.opt_name.find('klsoap')>=0:
    cast_dtype = torch.bfloat16
    if args.opt_name.find('_fp32')>=0:
        cast_dtype = torch.float32
    improve_orth = False
    if args.opt_name.find('_improve')>=0:
        improve_orth = True
    optimizer2 = KLOpt(raw_model.transformer.h.parameters(), lr=my_lr,
            betas=(args.momentum, 1.0-args.lr_cov),
            eps = args.damping,
            weight_decay=args.weight_decay,
            precondition_frequency=args.T,
            using_klsoap = True, #This method becomes KL-SOAP if using_klsoap = True
            normalize_grads = False,
            using_damping = False,
            using_clamping = True,
            improve_orth = improve_orth,
            cast_dtype = cast_dtype,
        )

elif args.opt_name.find('klshampoo')>=0:
    cast_dtype = torch.bfloat16
    if args.opt_name.find('_fp32')>=0:
        cast_dtype = torch.float32
    optimizer2 = KLOpt(raw_model.transformer.h.parameters(), lr=my_lr,
            betas=(args.momentum, 1.0-args.lr_cov),
            eps = args.damping,
            weight_decay=args.weight_decay,
            precondition_frequency=args.T,
            using_klsoap = False, #This method becomes KL-Shampoo if using_klsoap = False
            normalize_grads = False,
            using_damping = False,
            using_clamping = True,
            cast_dtype = cast_dtype,
        )

elif args.opt_name.find("soap") >= 0:
    normalize_grads = False
    if args.opt_name.find("_norm") > 0:
        print("using norm in soap")
        normalize_grads = True

    optimizer2 = SOAP(
        raw_model.transformer.h.parameters(),
        betas=(args.momentum, 1.0 - args.lr_cov),
        correct_bias=False,
        lr=my_lr,
        eps=args.damping,
        precondition_1d=False,
        normalize_grads=normalize_grads,
        weight_decay=args.weight_decay,
        precondition_frequency=args.T,
        cast_dtype=args.cast_dtype,
    )

elif args.opt_name.find("ivon") >= 0:
    optimizer2 = IVON(
        raw_model.transformer.h.parameters(),
        lr=my_lr,
        beta1=args.momentum,
        beta2=1.0 - args.lr_cov,
        ess=args.ess,
        hess_init=args.ivon_hess_init,
        clip_radius=args.ivon_clip_radius,
        weight_decay=args.weight_decay,
        sync=args.von_sync,
    )

elif args.opt_name.find("evon") >= 0:
    optimizer2 = NewEVON(
        raw_model.transformer.h.parameters(),
        lr=my_lr,
        betas=(args.momentum, 1.0 - args.lr_cov),
        hess_init=args.ivon_hess_init,  # reusing for evon h0
        precondition_frequency=args.T,
        debias_beta2=args.debias_second_moment,
        ess=args.ess,
        correct_bias=True,
        eps=args.damping,
        max_precond_dim=args.max_precond_dim,
        weight_decay=args.weight_decay,
        cast_dtype=args.cast_dtype,
        shampoo_beta=-1 if args.shampoo_beta is None else args.shampoo_beta,
        phasing=args.evon_phased_grads,
        price_clip_ratio=args.price_clip_ratio,
        sync=args.von_sync,
        whiten_grad=args.whiten_evon_grad
    )

print(optimizer2)

optimizers = [optimizer1, optimizer2]


# learning rate decay scheduler (linear warmup and warmdown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it + 1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        return decay_ratio


def get_cos_lr(it):
    alpha_f = 0.0001
    initial_lr = 1.0  # ratio
    assert it <= args.num_iterations
    max_steps = args.num_iterations
    eta_min = initial_lr * alpha_f
    if it < args.warmup_iters:
        return (it + 1) / args.warmup_iters
    else:
        if it < args.num_iterations - args.warmdown_iters:
            cur_it = it
        else:
            cur_it = args.num_iterations - args.warmdown_iters

        step = cur_it - args.warmup_iters
        max_steps = max_steps - args.warmup_iters
        ratio = eta_min + (initial_lr - eta_min) * (1 + cos(pi * step / max_steps)) / 2
        if it < args.num_iterations - args.warmdown_iters:
            return ratio
        else:
            decay_ratio = (args.num_iterations - it) / args.warmdown_iters
            return ratio * decay_ratio

    
def zero_optimizer_grads(opt):
    if opt is None:
        return
    for group in opt.param_groups:
        for p in group["params"]:
            if p is not None:
                p.grad = None

def get_ess(it, ess_start, ess_min, ess_min_fac, ess_anneal_steps):
    if ess_min is None:
        if ess_min_fac is None: 
            return ess_start
        else: 
            ess_min = ess_start * ess_min_fac
    elif ess_start == float("inf"):
        return float("inf")

    if ess_anneal_steps == 0:
        return ess_start
    elif it > ess_anneal_steps:
        return ess_min
    else:
        return ess_start * (1 - it / ess_anneal_steps) + ess_min * (
            it / ess_anneal_steps
        )

def get_ess_cosine(it, ess_start, ess_min_fac, ess_anneal_steps):
    if ess_anneal_steps == 0 or ess_min_fac is None:
        return ess_start

    if ess_start == float("inf"):
        return float("inf")

    ess_end = ess_start * ess_min_fac

    if it >= ess_anneal_steps:
        return ess_end

    t = it / ess_anneal_steps
    cosine_decay = 0.5 * (1 + math.cos(math.pi * t))

    return ess_end + (ess_start - ess_end) * cosine_decay

if args0.schd == "linear":
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]
elif args0.schd == "cosine":
    schedulers = [
        torch.optim.lr_scheduler.LambdaLR(opt, get_cos_lr) for opt in optimizers
    ]
else:
    assert False

if args.resume_from:
    if master_process:
        print(f"Resuming from checkpoint: {args.resume_from}")
    map_location = {"cuda:0": device}
    checkpoint = torch.load(args.resume_from, map_location=map_location)
    raw_model.load_state_dict(checkpoint["model"])
    for opt, state in zip(optimizers, checkpoint.get("optimizers", [])):
        opt.load_state_dict(state)
    for sched, state in zip(schedulers, checkpoint.get("schedulers", [])):
        sched.load_state_dict(state)
    if master_process:
        print(f"Resumed at loop step {checkpoint.get('step', 0)}")
    resume_start_step = int(checkpoint.get("step", 0))
else:
    resume_start_step = 0

if master_process:
    os.makedirs(checkpoint_dir, exist_ok=True)

# begin logging
if master_process:
    run_id = str(uuid.uuid4())
    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
    # create the log file
    with open(logfile, "w") as f:
        # begin the log by printing this file (the Python code)
        f.write('='*100 + '\n')
        f.write(code)
        f.write('='*100 + '\n')
        # log information about the hardware/software environment this is running on
        # and print the full `nvidia-smi` to file
        f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('='*100 + '\n')

training_time_ms = 0


def sampled_params_context(optimizer, train=False):
    if optimizer is None or not hasattr(optimizer, "sampled_params"):
        return nullcontext()
    try:
        return optimizer.sampled_params(train=train)
    except TypeError:
        return optimizer.sampled_params()


def sync_non_variational_grads(optimizers, variational_optimizer):
    # manually sync grads for AdamW parameters, since they are not updated with the variational optimizer's step and thus won't be automatically synced by it. This is important to do after the backward pass and before the step of the variational optimizer, to ensure that the variational optimizer's step sees the correct (synced) gradients for all parameters.
    if variational_optimizer is None:
        return
    if not dist.is_available() or not dist.is_initialized():
        return

    world_size = dist.get_world_size()
    if world_size <= 1:
        return

    variational_param_ids = {
        id(p)
        for group in variational_optimizer.param_groups
        for p in group["params"]
        if p is not None
    }
    reduced_param_ids = set()

    for optimizer in optimizers:
        if optimizer is variational_optimizer:
            continue
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p is None or p.grad is None:
                    continue
                pid = id(p)
                if pid in variational_param_ids or pid in reduced_param_ids:
                    continue
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(world_size)
                reduced_param_ids.add(pid)


# start the clock
torch.cuda.synchronize()
t0 = time.time()
# begin training

if hasattr(optimizer2, "sampled_params"):
    initial_ess = args.ess 
    if hasattr(optimizer2, "ess"):
        optimizer2.ess = initial_ess
    for group in optimizer2.param_groups:
        if "ess" in group:
            group["ess"] = initial_ess
    if master_process and initial_ess != args.ess:
        print(
            f"Adjusted ESS for distributed variational sync: base_ess={args.ess}, effective_ess={initial_ess}, world_size={ddp_world_size}"
        )

train_loader.reset()
is_early_stop = False
for step in range(resume_start_step, args.num_iterations + 1):
    if args.opt_name.find("evon") >= 0 or args.opt_name.find("ivon") >= 0:
        new_ess = get_ess_cosine(step, args.ess, args.ess_min_fac, args.ess_anneal_steps)

        if hasattr(optimizer2, "ess"):
            optimizer2.ess = new_ess 

        for group in optimizer2.param_groups:
            if "ess" in group:
                group["ess"] = new_ess
        
    last_step = step == args.num_iterations
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = (
        float("nan") if step <= 11 else (step - 10) + 1
    )  # <= 11 to avoid bug in val

    # once in a while evaluate the validation dataset
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        val_loss_mean = 0.0
        mc_samples = max(1, args.mc_samples) if hasattr(optimizer2, "sampled_params") else 1
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            if mc_samples > 1:
                avg_probs = None
                for _ in range(mc_samples):
                    with sampled_params_context(optimizer2, train=False):
                        with torch.no_grad():
                            with ctx:
                                logits, _ = model(x_val, y_val, return_logits=True)
                                probs = F.softmax(logits, dim=-1)
                                avg_probs = probs if avg_probs is None else (avg_probs + probs)
                                del logits, probs

                avg_probs /= mc_samples
                log_avg_probs = avg_probs.clamp_min(1e-12).log()
                loss_bma = F.nll_loss(
                    log_avg_probs.view(-1, log_avg_probs.size(-1)),
                    y_val.view(-1),
                    ignore_index=-1,
                )
                val_loss += loss_bma.detach()
                del avg_probs, log_avg_probs, loss_bma
                # Evaluate once at the mean parameters (no sampling) for a separate metric.
                with torch.no_grad():
                    with ctx:
                        _, loss_mean = model(x_val, y_val, return_logits=False)
                        val_loss_mean += loss_mean.detach()
                        del loss_mean
            else:
                # no_grad can trigger a separate compile for the inference path
                # on the first validation pass, which may look like a hang.
                with torch.no_grad():
                    with ctx:
                        _, loss = model(x_val, y_val, return_logits=False)
                        val_loss += loss.detach()
                        del loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        if mc_samples > 1:
            dist.all_reduce(val_loss_mean, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        if mc_samples > 1:
            val_loss_mean /= val_steps
        # log val loss to console and to logfile
        if master_process:
            print(
                f"step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / (timed_steps - 1):.2f}ms"
            )
            if wandb is not None:
                val_log = {"test_loss": val_loss.item()}
                if mc_samples > 1:
                    val_log["test_loss@mean"] = val_loss_mean.item()
                run.log(val_log, step=int(step + 1))
            # with open(logfile, "a") as f:
            #    f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')

        offset = 0.1
        if (step + 1) > 1200 and not args.disable_early_stop:
            if val_loss.item() < 3.8 + offset:
                pass
            else:
                is_early_stop = True

        if (step + 1) > 2000 and not args.disable_early_stop:
            if val_loss.item() < 3.6 + offset:
                pass
            else:
                is_early_stop = True

        if (step + 1) > 3000 and not args.disable_early_stop:
            if val_loss.item() < 3.5 + offset:
                pass
            else:
                is_early_stop = True

        if (step + 1) > 4000 and not args.disable_early_stop:
            if val_loss.item() < 3.45 + offset:
                pass
            else:
                is_early_stop = True

        if (step + 1) > 6000 and not args.disable_early_stop:
            if val_loss.item() < 3.4 + offset:
                pass
            else:
                is_early_stop = True

        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and (
        last_step or (args.save_every > 0 and step % args.save_every == 0)
    ):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # save the state of the training process
        log = dict(
            step=step,
            code=code,
            model=raw_model.state_dict(),
            optimizers=[opt.state_dict() for opt in optimizers],
            schedulers=[sched.state_dict() for sched in schedulers],
            args=vars(args0),
        )
        ckpt_path = os.path.join(checkpoint_dir, f"state_step{step:06d}.pt")
        latest_path = os.path.join(checkpoint_dir, "latest.pt")
        torch.save(log, ckpt_path)
        torch.save(log, latest_path)
        print(f"Saved checkpoint: {ckpt_path}")
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step or is_early_stop:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    is_von = args.opt_name.find("ivon") >= 0 or args.opt_name.find("evon") >= 0

    if is_von:
        assert getattr(optimizer2, "sync", False) or getattr(optimizer2, "_sync", False), (
            "This script assumes IVON/EVON handles distributed sync. "
            "Run with --von_sync true."
        )

    for i in range(1, train_accumulation_steps + 1):
        if is_von:
            # VON handles distributed sync internally; DDP should not sync these grads.
            with model.no_sync():
                with sampled_params_context(optimizer2, train=True):
                    # Clear only VON grads before this sampled microbatch.
                    zero_optimizer_grads(optimizer2)

                    with ctx:
                        _, loss = model(x, y, return_logits=False)
                        train_loss = loss.detach()

                    x, y = train_loader.next_batch()
                    loss.backward()

        else:
            sync_context = model.no_sync() if i < train_accumulation_steps else nullcontext()
            with sync_context:
                with ctx:
                    _, loss = model(x, y, return_logits=False)
                    train_loss = loss.detach()

                x, y = train_loader.next_batch()
                loss.backward()

    if is_von: 
        for g in optimizer1.param_groups:
            for p in g["params"]:
                p.grad /= train_accumulation_steps 
    else:
        for p in model.parameters():
            p.grad /= train_accumulation_steps 

    if is_von and (getattr(optimizer2, "sync", False)):
        # DDP all-reduce was skipped in backward for VON sync mode.
        # Explicitly sync gradients for non-variational parameter groups.
        sync_non_variational_grads(optimizers, optimizer2)

    # ---------------------------------------------------------------- #
    # Diagnostic: EVON h_mom scalar stats per parameter                #
    # (master_process only, every diag_every steps, collect_stats flag) #
    # ---------------------------------------------------------------- #
    do_diag = (
        master_process
        and args.collect_stats
        and args0.diag_every > 0
        and (step + 1) % args0.diag_every == 0
    )
    _diag_log: dict = {}

    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()

    if do_diag and isinstance(optimizer2, EVON):
        # Per-parameter h_mom scalars: max, min, mean (all .item() — no CPU copies)
        _param_id_to_name_diag = {id(p): n for n, p in raw_model.named_parameters()}
        for _gi, _group in enumerate(optimizer2.param_groups):
            for _pi, _p in enumerate(_group["params"]):
                _st = optimizer2.state.get(_p, {})
                if "h_mom" in _st:
                    _hm = _st["h_mom"]
                    _pname = _param_id_to_name_diag.get(id(_p), f"pg{_gi}/p{_pi}")
                    _diag_log[f"evon/h_mom/{_pname}/max"]  = _hm.max().item()
                    _diag_log[f"evon/h_mom/{_pname}/min"]  = _hm.min().item()
                    _diag_log[f"evon/h_mom/{_pname}/mean"] = _hm.mean().item()

    # null the gradients
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    # dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        org_lr = optimizer2.param_groups[0]["lr"]
        log_data = {
            "loss": train_loss.item(),
            "lr": org_lr,
            "train_time": approx_time,
        }

        if hasattr(optimizer2, "get_kl"):
            log_data["kl_div"] = optimizer2.get_kl(omit_constants=True)

        if torch.isnan(train_loss):
            assert False, "NaN loss detected"

        # Merge diagnostic metrics collected this step (h_mom scalars, etc.)
        if do_diag:
            log_data.update(_diag_log)

        # EVON clip statistics - logged every step (cheap: scalars only).
        # Keys: group index -> {"prec": [...], "upd": [...]}.
        # Each entry: {"clip_frac": float, "norm_ratio": float (post/pre L2)}.
        if hasattr(optimizer2, "clip_stats"):
            for _gi, _gstats in optimizer2.clip_stats.items():
                for _clip_name, _entries in _gstats.items():
                    if _entries:
                        _n = len(_entries)
                        _avg_frac  = sum(e["clip_frac"]  for e in _entries) / _n
                        _avg_ratio = sum(e["norm_ratio"] for e in _entries) / _n
                        log_data[f"evon/pg{_gi}/{_clip_name}/clip_frac"]  = _avg_frac
                        log_data[f"evon/pg{_gi}/{_clip_name}/norm_ratio"] = _avg_ratio

        # EVON h_mom histograms — logged every hess_hist_freq steps.
        do_hess_hist = (
            args.collect_stats
            and args.hess_hist_freq > 0
            and (step + 1) % args.hess_hist_freq == 0
            and isinstance(optimizer2, EVON)
            and wandb is not None
        )
        if do_hess_hist:
            _param_id_to_name_hist = {id(p): n for n, p in raw_model.named_parameters()}
            for _gi, _group in enumerate(optimizer2.param_groups):
                for _pi, _p in enumerate(_group["params"]):
                    _st = optimizer2.state.get(_p, {})
                    if "h_mom" in _st:
                        _pname = _param_id_to_name_hist.get(id(_p), f"pg{_gi}/p{_pi}")
                        _hm_cpu = _st["h_mom"].detach().float().cpu().flatten().numpy()
                        log_data[f"evon/h_mom_hist/{_pname}"] = wandb.Histogram(_hm_cpu)

        print(
            f"step:{step + 1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time / timed_steps:.2f}ms"
        )
        if wandb is not None:
            run.log(log_data, step=int(step + 1))
        # with open(logfile, "a") as f:
        #    f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n")


if master_process:
    if is_early_stop and wandb is not None:
        run.tags = [
            "early_stop",
        ]
    print(
        f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB"
    )
    print(optimizer2)

    # ---------------------------------------------------------------- #
    # End-of-run: log h_mom histograms per parameter to wandb.         #
    # Always fires when collect_stats=True, unless the final training  #
    # step already triggered the periodic histogram log (avoid dup).   #
    # ---------------------------------------------------------------- #
    _final_step_logged_hist = (
        args.hess_hist_freq > 0
        and args.num_iterations % args.hess_hist_freq == 0
    )
    if (
        args.collect_stats
        and not _final_step_logged_hist
        and isinstance(optimizer2, EVON)
        and wandb is not None
    ):
        _param_id_to_name_final = {id(p): n for n, p in raw_model.named_parameters()}
        _hist_log = {}
        for _gi, _group in enumerate(optimizer2.param_groups):
            for _pi, _p in enumerate(_group["params"]):
                _st = optimizer2.state.get(_p, {})
                if "h_mom" in _st:
                    _pname = _param_id_to_name_final.get(id(_p), f"pg{_gi}/p{_pi}")
                    _hm_cpu = _st["h_mom"].detach().float().cpu().flatten().numpy()
                    _hist_log[f"evon/h_mom_hist/{_pname}"] = wandb.Histogram(_hm_cpu)
        if _hist_log:
            run.log(_hist_log, step=int(args.num_iterations))

# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()
