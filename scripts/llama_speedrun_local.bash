#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate virtual environment
source .venv/bin/activate

# navigate to the directory since the script assumes it's run from there
cd Minimalist_LLM_Pretraining
DATASET_PATH=/data1/datasets/c4-t5/subset/

# EVON
uv run python -m torch.distributed.run --standalone --nproc_per_node=1 torchrun_main_DDP.py \
--wandb_project_name=llama --model_config=configs/llama_130m.json --total_batch_size=512 --warmup_steps=500 --dtype=bfloat16 --eval_every=500 --save_every=100 --seed=42 --scheduler=cosine --attn_ratio=1.0 --num_training_steps=10000 --max_train_fileid=20 --batch_size=128 --optimizer=evon --freq=10 --min_lr_ratio=1e-3 --mc_samples=0 --cast_dtype=float32 --adam_lr=0.0013838048243980062 --adam_beta_1=0.8338619785890484 --adam_beta_2=0.97863020202 --adam_weight_decay=1.8968983444163511e-06 --adam_damping=9.780454994293121e-13 --evon_noise_damping=0 --whiten_evon_grad --ess=7614799.545857169 --ivon_hess_init=0.9305205540052246 --lr=0.015185510466783668 --lr_cov=0.0003455352551726819 --momentum=0.88121574363312 --price_clip_ratio=1.2 --shampoo_beta=0.995 --weight_decay=8.83120236383412e-07 --evon_phased_grads --dataset_path=$DATASET_PATH #--continue_from=checkpoints/llama_130m-2026-07-24-16-16-08/model_100