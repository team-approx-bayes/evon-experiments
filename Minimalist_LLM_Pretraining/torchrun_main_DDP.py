import os
import time
import json
import random
import math
import numpy as np
import socket
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.distributed as dist
from safetensors.torch import load_file

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM
import datasets
import datasets.distributed

import wandb
from tqdm import tqdm
from loguru import logger

from mem_eff_pt.eff_pretraining import training_utils
from mem_eff_pt.eff_pretraining.dataloader import PreprocessedIterableDataset, PreprocessedIterableDataset_noslice
from mem_eff_pt.eff_pretraining.dataloader_v2 import PreprocessedIterableDataset_v2

from mem_eff_pt.eff_pretraining.modeling_llama import LlamaForCausalLM

from mem_eff_pt.utils.train_utils import *
from mem_eff_pt.utils.args import parse_args


script_dir = os.path.dirname(os.path.abspath(__file__))
print("Script path:", script_dir)

transformers.logging.set_verbosity_error()
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


def get_wd(step, wd_target, wd_anneal_steps):
    if wd_anneal_steps == 0 or step >= wd_anneal_steps:
        return wd_target
    return wd_target * (step / wd_anneal_steps)


def get_ess(step, ess_start, ess_min, ess_anneal_steps):
    if ess_min is None or ess_start == float("inf"):
        return ess_start
    if ess_anneal_steps == 0:
        return ess_start
    if step > ess_anneal_steps:
        return ess_min
    return ess_start * (1 - step / ess_anneal_steps) + ess_min * (step / ess_anneal_steps)

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


def _wandb_safe_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_wandb_safe_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _wandb_safe_value(v) for k, v in value.items()}
    return str(value)


def sampled_params_context(optimizer, train=False):
    if optimizer is None or not hasattr(optimizer, "sampled_params"):
        return nullcontext()
    try:
        return optimizer.sampled_params(train=train)
    except TypeError:
        return optimizer.sampled_params()


def sync_non_variational_grads(opts, variational_optimizer):
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

    for optimizer in opts:
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


def zero_optimizer_grads(opt):
    if opt is None:
        return
    for group in opt.param_groups:
        for p in group["params"]:
            if p is not None:
                p.grad = None

@torch.no_grad()
def evaluate_model(
    model,
    preprocess_batched,
    pad_idx,
    global_rank,
    world_size,
    device,
    batch_size,
    variational_optimizer=None,
    mc_samples=1,
):
    _time = time.time()
    if not args.hf_dataset:
        logger.info(f"Using local dataset for validation")
        data_files_val= {"validation": [f"c4-validation.{str(i).zfill(5)}-of-00008.json.gz" for i in range(0,8)]}
        val_data = datasets.load_dataset(path=args.dataset_path,  data_files=data_files_val, split="validation", streaming=True)
    else:
        val_data = datasets.load_dataset(
            "allenai/c4", "en", split="validation", streaming=True
        )  # DGX
    val_data = val_data.shuffle(seed=42)
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    if not args.single_gpu:
        val_data = datasets.distributed.split_dataset_by_node(
            val_data, rank=global_rank, world_size=world_size
        )

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(
        val_data_mapped, batch_size
    )

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    was_training = model.training
    model.eval()

    try:
        for batch in val_data_mapped.batch(batch_size=batch_size):
            if evaluated_on_tokens > target_eval_tokens:
                break
            total_batches += 1

            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["input_ids"].clone()
            labels[labels == pad_idx] = -100
            if variational_optimizer is not None and mc_samples > 1:
                avg_probs = None
                for _ in range(mc_samples):
                    with sampled_params_context(variational_optimizer, train=False):
                        logits = model(**batch).logits
                        probs = F.softmax(logits.float(), dim=-1)
                        avg_probs = probs if avg_probs is None else (avg_probs + probs)
                        del logits, probs

                avg_probs = avg_probs / mc_samples
                shift_log_probs = avg_probs[..., :-1, :].clamp_min(1e-12).log().contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.nll_loss(
                    shift_log_probs.view(-1, shift_log_probs.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                ).detach()
                del avg_probs, shift_log_probs, shift_labels
            else:
                loss = model(**batch, labels=labels).loss.detach()
            total_loss += loss

            evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size
    finally:
        if was_training:
            model.train()

    total_loss = total_loss / total_batches

    # Gather losses across all GPUs
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, total_loss)
    total_loss = sum([t.item() for t in gathered_losses]) / world_size

    return total_loss, evaluated_on_tokens


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    logger.info(
        f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}"
    )

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert (
                args.total_batch_size % world_size == 0
            ), "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (
                args.batch_size * world_size
            )
            # logger.info(f"{args.gradient_accumulation}-{world_size}-{args.total_batch_size}-{args.batch_size}")
            assert (
                args.gradient_accumulation > 0
            ), "gradient_accumulation must be greater than 0"

    assert (
        args.gradient_accumulation * args.batch_size * world_size
        == args.total_batch_size
    ), "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # turn off logger
    if global_rank != 0:
        logger.remove()

    if global_rank == 0:
        #run_name = "%s-T%d-%s" % (args.opt_name, args.T, socket.gethostname())

        opt_name = args.optimizer.lower()
        if opt_name.find('muon')>=0 or opt_name.find('adam')>=0:
            args.freq = 1

        if args.qkv_mode == 'fused':
            run_name = '%s_fused'%opt_name
        elif args.qkv_mode == 'single':
            run_name = '%s_nofused'%opt_name
        else:
            run_name = '%s_qkv3d'%opt_name

        run_name = "%s-T%d-%s" % (run_name, args.freq, socket.gethostname())
        run = wandb.init(project=args.wandb_project_name, name=run_name,
                tags=["normal"], entity="adrianrob1-Sapienza Università di Roma"
                )
        run.define_metric("test/*", step_metric="test/step")
        for k, v in vars(args).items():
            run.summary[f"args/{k}"] = _wandb_safe_value(v)
        wandb_log_step = run.step

        logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
        logger.info("*" * 40)
        logger.info(f"Starting training with the arguments")
        for k, v in vars(args).items():
            logger.info(f"{k:30} {v}")
        logger.info("*" * 40)

    # data
    if not args.hf_dataset:
        logger.info(f"Using local dataset for training")
        data_files_train = {"train": [f"c4-train.{str(i).zfill(5)}-of-01024.json.gz" for i in range(0,args.max_train_fileid)]}
        logger.info(f"loading dataset")
        print(data_files_train)
        data = datasets.load_dataset(path=args.dataset_path,  data_files=data_files_train, split="train", streaming=True)
        logger.info(f"loaded dataset")
    else:
        data = datasets.load_dataset(
            "allenai/c4", "en", split="train", streaming=True
        )  # DGX

    seed_for_shuffle = 42

    logger.info(f"Shuffling data with seed {seed_for_shuffle}")
    # it doesn't matter which tokenizer we use, because we train from scratch
    # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice
    tokenizer = AutoTokenizer.from_pretrained(
        "t5-base", model_max_length=args.max_length, use_fast=False,
    )

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch


    # model
    model_config = AutoConfig.from_pretrained(os.path.join(script_dir, args.model_config))
    if args.num_attention_heads>0:
        model_config.num_attention_heads = args.num_attention_heads
    print(model_config)
    if args.use_hf_model:
        assert False
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config, qkv_mode=args.qkv_mode, attn_ratio=args.attn_ratio)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    global_step = 0
    update_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    # ====== starting config ======= #
    target_modules_list = ["attn", "mlp", "attention"]
    args.target_modules = target_modules_list

    # build model
    if args.dtype in ["bf16", "bfloat16"]:
        model = build_model(model.to(device=device, dtype=torch.bfloat16), args)
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = build_model(model.to(device=device), args)
        model = model.to(device=device)

    # Reseed after model init to keep identical weights but distinct per-rank noise.
    rank_seed = args.seed + global_rank
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)
    np.random.seed(rank_seed)
    random.seed(rank_seed)
    reseed_msg = f"Reseeded RNGs with rank_seed={rank_seed} (base seed={args.seed}, rank={global_rank})"
    if global_rank == 0:
        logger.info(reseed_msg)
    else:
        print(reseed_msg)
        

    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]


    # build optimizer
    #optimizer = build_optimizer(model, trainable_params, args)
    opts = build_optimizer(model, trainable_params, args)
    assert len(opts)<=2
    variational_optimizer = next(
        (optimizer for optimizer in opts if hasattr(optimizer, "sampled_params")),
        None,
    )

    eval_mc_samples = 1
    if variational_optimizer is not None:
        eval_mc_samples = max(1, int(getattr(args, "mc_samples", 10)))
        initial_ess = args.ess 
        if hasattr(variational_optimizer, "ess"):
            variational_optimizer.ess = initial_ess
        for group in variational_optimizer.param_groups:
            if "ess" in group:
                group["ess"] = initial_ess
        if global_rank == 0 and initial_ess != args.ess:
            logger.info(
                f"Adjusted ESS for distributed variational sync: base_ess={args.ess}, effective_ess={initial_ess}, world_size={world_size}"
            )
        logger.info(
            f"Using Monte Carlo evaluation with mc_samples={eval_mc_samples} for optimizer {type(variational_optimizer).__name__}"
        )
       

    schedulers = training_utils.get_scheculer(
            opts=opts,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )

    if args.continue_from is not None:
        assert False
        '''
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        checkpoint_path = os.path.join(args.continue_from, "pytorch_model.bin")
        
        if not os.path.exists(checkpoint_path): #safetensors -> bin  
            safetensors_file = os.path.join(args.continue_from, "model.safetensors")
            state_dict = load_file(safetensors_file)
            torch.save(state_dict, checkpoint_path)
 
            logger.info(f"safetensors {safetensors_file} converted to pytorch bin {checkpoint_path}")
        
        if args.peft_model.lower() in ["sltrain"]:
            model.wrapped_model.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu"), strict=True
            )
        else:
            model.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu"), strict=True
            )
        logger.info(f"Model successfully loaded (strict=True policy)")

        optimizer_checkpoint = torch.load(
            os.path.join(args.continue_from, "optimizer.pt"), map_location="cpu"
        )
        if "optimizers" in optimizer_checkpoint and len(optimizer_checkpoint["optimizers"]) == len(opts):
            for _optimizer, _optimizer_state in zip(opts, optimizer_checkpoint["optimizers"]):
                _optimizer.load_state_dict(_optimizer_state)
        else:
            optimizer.load_state_dict(optimizer_checkpoint["optimizer"])

        if "schedulers" in optimizer_checkpoint and len(optimizer_checkpoint["schedulers"]) == len(schedulers):
            for _scheduler, _scheduler_state in zip(schedulers, optimizer_checkpoint["schedulers"]):
                _scheduler.load_state_dict(_scheduler_state)
        else:
            scheduler.load_state_dict(optimizer_checkpoint["scheduler"])
        logger.info(f"Optimizer and scheduler restored from {args.continue_from}")

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(
                f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}"
            )
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(
                f"Will train for {args.num_training_steps - update_step} update steps"
            )
        else:
            logger.warning(
                f"Did not find training state in {args.continue_from}, global step will start from zero"
            )
        logger.info("*" * 40)'''

    scheduler_start_step = update_step


    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(
        f"All params: \n{[n for n,p in model.named_parameters() if p.requires_grad]}\n"
    )
    logger.info(
        f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M"
    )
    logger.info(
        f"Total non-low-rank and non-sparse parameters: "
        f"{sum(p.numel() for n,p in model.named_parameters() if 'lora_' not in n and 'sparse_' not in n) / 1_000_000:.2f}M"
    )
    logger.info(f"max_len:{args.max_length}")

    logger.info(
        f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M"
    )
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")

    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update(
        {
            "max_lr": run_config.pop(
                "lr"
            ),  # rename lr to max_lr to avoid conflicts with scheduler
            "total_params_M": n_total_params / 1_000_000,
            "dataset": "allenai/c4",
            "model": model_config.to_dict(),
            "world_size": world_size,
            "device": str(device),
        }
    )

    if global_rank == 0:
        #wandb.config.update(run_config, allow_val_change=True)
        #wandb.save(os.path.abspath(__file__), policy="now")  # save current script
        pbar = tqdm(
            total=args.num_training_steps - update_step, desc="Update steps", ncols=80
        )

    if not args.single_gpu:
        model: LlamaForCausalLM = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step

    # ##############################
    # TRAINING LOOP
    # ##############################

    grad_norm_prev = None
    opt_update_time = 0

    max_memory = torch.cuda.max_memory_allocated()
    if global_rank == 0:
        logger.info(f"Maximum memory allocated before training: {max_memory} bytes\n")
    torch.cuda.reset_peak_memory_stats()

    boo = False
    count = 0 #this should be saved in a checkpoint and reloaded from the checkpoint
    no_early_stop = True #this should be saved in a checkpoint and reloaded from the checkpoint

    while update_step <= args.num_training_steps and no_early_stop:
       data_shuffled = data.shuffle(seed=(seed_for_shuffle+count))
       count += 1
       if not args.single_gpu:
            data_shuffled = datasets.distributed.split_dataset_by_node(
                data_shuffled,
                rank=global_rank,
                world_size=world_size,
            )

       if args.continue_from is not None:
            dataset = PreprocessedIterableDataset_v2(
                data_shuffled, tokenizer, batch_size=args.batch_size, max_length=args.max_length, start_tokenizing_idx = args.start_tokenizing_idx
            )
       else:
            if args.no_slice:
                logger.info(f"Using PreprocessedIterableDataset_noslice !!")
                dataset = PreprocessedIterableDataset_noslice(
                    data_shuffled, tokenizer, batch_size=args.batch_size, max_length=args.max_length
                )
            else:
                 dataset = PreprocessedIterableDataset(
                    data_shuffled, tokenizer, batch_size=args.batch_size, max_length=args.max_length,
                )

       dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=args.workers)
       print('new epoch v2')
       for batch_idx, batch in enumerate(dataloader):
            if args.continue_from is not None and not boo:
                if batch_idx   <=   (update_step) * args.gradient_accumulation  - 1 :
                    if batch_idx % 1000 == 0:
                        print(batch_idx)
                    continue
                else:
                    print(f"\n start at {batch_idx} \n")
                    boo = True

            if update_step == 0 and args.eval_at_begining :
                logger.info(f"Performing evaluation at step {update_step}")
                total_loss, evaluated_on_tokens = evaluate_model(
                    model,
                    preprocess_batched,
                    pad_idx,
                    global_rank,
                    world_size,
                    device,
                    args.batch_size,
                    variational_optimizer=variational_optimizer,
                    mc_samples=eval_mc_samples,
                )
                if global_rank == 0:
                    log_info={
                            "test/step": update_step,
                            "test/final_eval_loss": total_loss,
                            "test/final_eval_perplexity": np.exp(total_loss),
                            "test/final_eval_tokens": evaluated_on_tokens
                            }
                    run.log(log_info,
                            step=wandb_log_step,
                            )
                logger.info(
                    f"Eval loss and perplexity at step {update_step}: {total_loss}, {np.exp(total_loss)}"
                )

            global_step += 1
            local_step += 1

            if update_step > args.num_training_steps:
                logger.info(
                    f"Reached max number of update steps (f{args.num_training_steps}). Stopping training."
                )
                print(f"Rank {global_rank} stopping training.")
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["input_ids"].clone()
            labels[labels == pad_idx] = -100
            tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

            if variational_optimizer is not None:
                new_wd = get_wd(update_step, args.wd_target, args.wd_anneal_steps)
                new_ess = get_ess_cosine(update_step, args.ess, args.ess_min_fac, args.ess_anneal_steps)
                effective_ess = new_ess 
                if hasattr(variational_optimizer, "weight_decay"):
                    variational_optimizer.weight_decay = new_wd
                if hasattr(variational_optimizer, "ess"):
                    variational_optimizer.ess = effective_ess
                for group in variational_optimizer.param_groups:
                    if "weight_decay" in group:
                        group["weight_decay"] = new_wd
                    if "ess" in group:
                        group["ess"] = effective_ess

            start_time = time.time()

            is_von_sync = (
                variational_optimizer is not None
                and getattr(variational_optimizer, "sync", False)
                and hasattr(model, "no_sync")
            )

            ddp_sync_context = model.no_sync() if is_von_sync else nullcontext()
            sample_context = sampled_params_context(variational_optimizer, train=True)

            with ddp_sync_context:
                with sample_context:
                    if variational_optimizer is not None:
                        zero_optimizer_grads(variational_optimizer)

                    loss = model(**batch, labels=labels).loss
                    loss.backward()

            # context = (
            #     variational_optimizer.sampled_params(train=True)
            #     if variational_optimizer is not None
            #     else nullcontext()
            # )
            # start_time = time.time()

            # with context:
            #     loss = model(**batch, labels=labels).loss
            #     scaled_loss = loss 

            #     if (
            #         variational_optimizer is not None
            #         and getattr(variational_optimizer, "sync", False)
            #         and hasattr(model, "no_sync")
            #     ):
            #         # Optimizer performs distributed synchronization of MC statistics.
            #         # Skip DDP gradient all-reduce during backward.
            #         with model.no_sync():
            #             scaled_loss.backward()
            #     else:
            #         scaled_loss.backward()

            if global_step % args.gradient_accumulation != 0:
                continue

            if variational_optimizer is None: 
                for p in model.parameters():
                    p.grad /= args.gradient_accumulation
            else:
                for g in opts[1].param_groups:
                    for p in g["params"]:
                        p.grad /= args.gradient_accumulation

            if (
                variational_optimizer is not None
                and getattr(variational_optimizer, "sync", False)
            ):
                # DDP all-reduce was skipped in backward for VON sync mode.
                # Explicitly sync gradients of non-variational parameter groups.
                sync_non_variational_grads(opts, variational_optimizer)

            if args.grad_clipping != 0.0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)

            grad_norm = sum(
                [
                    torch.norm(p.grad.clone().detach().cpu())
                    for p in model.parameters()
                    if p.grad is not None
                ]
            )

            if global_rank == 0:
                pbar.update(1)

            for optimizer in opts:
                optimizer.step()
                optimizer.zero_grad()

            for scheduler in schedulers:
                scheduler.step()

            update_step += 1

            if global_rank == 0:
                wandb_log_step += 1

            opt_update_time += time.time() - start_time 
            update_time = time.time() - update_time

            # save checkpoint by save_every

            if (
                local_step > args.gradient_accumulation
                and update_step % args.save_every == 0
                and global_rank == 0
            ):
                if args.keep_only_last_model:
                    current_model_directory = f"{args.save_dir}/model_last"
                else:
                    current_model_directory = f"{args.save_dir}/model_{update_step}"
                logger.info(
                    f"Saving model and optimizer to {current_model_directory}, update step {update_step}"
                )
                os.makedirs(args.save_dir, exist_ok=True)
                model.module.save_pretrained(
                    current_model_directory, max_shard_size="100GB"
                )

                optimizer_checkpoint = {
                    "optimizers": [opt.state_dict() for opt in opts],
                    "schedulers": [sch.state_dict() for sch in schedulers],
                    "update_step": update_step,
                    "global_step": global_step,
                    "config": run_config,
                    "wandb": wandb.run.dir,
                    "dtype": args.dtype,
                }
                torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

                training_state_checkpoint = {
                    "global_step": global_step,
                    "update_step": update_step,
                    "tokens_seen": tokens_seen,
                    "tokens_seen_before": tokens_seen_before,
                    "update_time": update_time,
                }
                with open(f"{current_model_directory}/training_state.json", "w") as f:
                    json.dump(training_state_checkpoint, f, indent=4)

                # save wandb related info
                wandb_info = {
                    "wandb_id": wandb.run.id,
                }
                with open(f"{args.save_dir}/wandb.json", "w") as f:
                    json.dump(wandb_info, f, indent=4)

            # evaluation
            if update_step % args.eval_every == 0:
                logger.info(f"Performing evaluation at step {update_step}")
                total_loss, evaluated_on_tokens = evaluate_model(
                    model,
                    preprocess_batched,
                    pad_idx,
                    global_rank,
                    world_size,
                    device,
                    args.batch_size,
                    variational_optimizer=variational_optimizer,
                    mc_samples=eval_mc_samples,
                )
                if global_rank == 0:
                    log_info={
                            "test/step": update_step,
                            "test/final_eval_loss": total_loss,
                            "test/final_eval_perplexity": np.exp(total_loss),
                            "test/final_eval_tokens": evaluated_on_tokens,
                            }
                    run.log(log_info,
                            step=wandb_log_step
                            )
                if not args.disable_early_stop:
                    if args.max_train_fileid == 20:
                        if total_loss> 4.8 and update_step>1000:
                            no_early_stop = False
                        if total_loss> 4.0 and update_step>2000:
                            no_early_stop = False
                        if total_loss> 3.8 and update_step>3000:
                            no_early_stop = False

                    if args.max_train_fileid == 100: #only for the 350m model
                        if total_loss> 3.6 and update_step>2000:
                            no_early_stop = False
                        elif total_loss> 3.3 and update_step>4000:
                            no_early_stop = False
                        elif total_loss> 3.2 and update_step>6000:
                            no_early_stop = False
                        elif total_loss> 3.1 and update_step>8000:
                            no_early_stop = False

                logger.info(
                    f"Eval loss and perplexity at step {update_step}: {total_loss}, {np.exp(total_loss)}"
                )

            lr = opts[0].param_groups[0]["lr"]
            tokens_in_update = tokens_seen - tokens_seen_before
            tokens_seen_before = tokens_seen
            batches_in_update = args.gradient_accumulation * world_size
            if len(opts)>1:
                lr2 = opts[1].param_groups[0]["lr"]
            else:
                lr2 = lr

            max_memory = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()

            early_stop = 1
            if no_early_stop:
                early_stop = 0

            if global_rank == 0:
                run.log(
                    {
                        "test/step": update_step,
                        "test/loss": loss.item(),
                        "test/lr": lr,
                        "test/lr2": lr2,
                        "test/update_step": update_step,
                        "test/tokens_seen": tokens_seen,
                        "test/throughput_tokens": tokens_in_update / update_time,
                        "test/throughput_examples": args.total_batch_size / update_time,
                        "test/throughput_batches": batches_in_update / update_time,
                        "test/gradnorm": grad_norm,
                        "test/max_memory": max_memory,
                        'test/early_stop': early_stop,
                        'test/opt_update_time': opt_update_time 
                    },
                    step=wandb_log_step,
                )
                if early_stop==1:
                    run.tags = ["early_stop",]

            opt_update_time = 0

            if early_stop==1:
                break

            update_time = time.time()

######################################################################
    # ##############################
    # END of training loop
    # ##############################
    
    if no_early_stop:
        logger.info("Training finished")
    else:
        logger.info("Training early stopped")

    if global_rank == 0:
        pbar.close()

    """
    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if global_rank == 0 and not os.path.exists(current_model_directory):
        logger.info(
            f"Saving model and optimizer to {current_model_directory}, update step {update_step}"
        )
        os.makedirs(args.save_dir, exist_ok=True)
        model.module.save_pretrained(current_model_directory)

        optimizer_checkpoint = {
            "optimizers": [opt.state_dict() for opt in opts],
            "schedulers": [sch.state_dict() for sch in schedulers],
            "update_step": update_step,
            "global_step": global_step,
            "config": run_config,
            "wandb": wandb.run.dir,
            "dtype": args.dtype,
        }
        torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "update_time": update_time,
        }
        with open(f"{current_model_directory}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)
    """

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, opts, schedulers
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    total_loss, evaluated_on_tokens = evaluate_model(
        model,
        preprocess_batched,
        pad_idx,
        global_rank,
        world_size,
        device,
        args.batch_size,
        variational_optimizer=variational_optimizer,
        mc_samples=eval_mc_samples,
    )

    if global_rank == 0:
        log_info={
                "test/step": update_step,
                "test/final_eval_loss": total_loss,
                "test/final_eval_perplexity": np.exp(total_loss),
                "test/final_eval_tokens": evaluated_on_tokens,
                'test/early_stop': early_stop
                }
        run.log(log_info,
                step=wandb_log_step
                )

        logger.info(
            f"Eval loss and perplexity at step {update_step}: {total_loss}, {np.exp(total_loss)}"
        )

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)
