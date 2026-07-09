import argparse
import torch

from mem_eff_pt.eff_pretraining import args_utils, training_utils


def parse_args(args):
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--debug", action='store_true')
    
    parser.add_argument("--dataset_path", type=str, default=None)
    
    parser.add_argument("--eval_at_begining", default=False, action="store_true")
    
    parser.add_argument("--start_tokenizing_idx", type=int, default=0)
    
    parser.add_argument("--no_slice", default=False, action="store_true")

    parser.add_argument("--keep_only_last_model", default=False, action="store_true")
    
    parser.add_argument("--adam_lr", type=float, default=2e-2) 
    parser.add_argument("--adam_beta_1", type=float, default=0.9) 
    parser.add_argument("--adam_beta_2", type=float, default=0.999) 
    parser.add_argument("--adam_damping", type=float, default=1e-8)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    
    # stream huggingface dataset instead of local
    parser.add_argument("--hf_dataset", default=False, action="store_true")
    
    parser.add_argument("--wandb_project_name", type=str)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--qkv_mode", type=str, default='single')
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_cov", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--damping", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--ess", type=float, default=1e9)
    parser.add_argument("--wd_target", type=float, default=1e-6)
    parser.add_argument("--wd_anneal_steps", type=int, default=0)
    parser.add_argument("--ess_min", type=float, default=None)
    parser.add_argument("--ess_min_fac", type=float, default=None)
    parser.add_argument("--ess_anneal_steps", type=float, default=None)
    parser.add_argument("--ivon_hess_init", type=float, default=1e-3)
    parser.add_argument("--ivon_clip_radius", type=float, default=float("inf"))
    parser.add_argument("--price_clip_ratio", type=float, default=None)
    parser.add_argument("--collect_stats", default=False, action="store_true")
    parser.add_argument("--decoupled_wd", default=False, action="store_true")
    parser.add_argument("--debias_second_moment", default=False, action="store_true")
    parser.add_argument("--max_precond_dim", type=int, default=10000)
    parser.add_argument("--shampoo_beta", type=float, default=None)
    parser.add_argument("--evon_phased_grads", default=False, action="store_true")
    parser.add_argument("--evon_noise_damping", type=float, default=0.0)
    parser.add_argument(
        "--von_sync",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable distributed sync inside IVON/EVON; when enabled, DDP grad all-reduce can be skipped in backward.",
    )
    parser.add_argument("--whiten_evon_grad", default=False, action="store_true")

    parser.add_argument("--cast_dtype", type=str, default="float32")
    parser.add_argument("--disable_early_stop", default=False, action="store_true")
    parser.add_argument("--attn_ratio", type=float, default=1.0)
    parser.add_argument("--num_attention_heads", type=int, default=-1)
    parser.add_argument("--freq", type=int, default=10)
    parser.add_argument("--mc_samples", type=int, default=10)
    parser.add_argument("--init_factor", type=float, default=1.0)
    parser.add_argument("--max_train_fileid", type=int, default=80)
    parser.add_argument("--block_factor", type=int, default=4)
    parser.add_argument("--block_freq", type=int, default=5)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "cosine_restarts","cosine_quick_recovery"],
    )
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument(
        "--num_training_steps",
        type=int,
        default=10_000,
        help="Number of **update steps** to train for. "
        "Notice that gradient accumulation is taken into account.",
    )
    parser.add_argument(
        "--max_train_tokens",
        type=training_utils.max_train_tokens_to_number,
        default=None,
        help="Number of tokens to train on. Overwrites num_training_steps. "
        "You can use M and B suffixes, e.g. 100M or 1B.",
    )
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16" if torch.cuda.is_bf16_supported() else "float32",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=0.0)


    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=False, action="store_true")

    parser.add_argument(
        "--distributed_type", type=str, default="ddp", choices=["fsdp", "ddp"]
    )

    args = parser.parse_args(args)

    if args.cast_dtype == "bfloat16":
        args.cast_dtype = torch.bfloat16
    elif args.cast_dtype == "float16":
        args.cast_dtype = torch.float16
    elif args.cast_dtype == "float32":
        args.cast_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported cast_dtype: {args.cast_dtype}")

    args = args_utils.check_args_torchrun_main(args)

    return args
