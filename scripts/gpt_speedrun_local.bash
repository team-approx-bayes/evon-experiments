#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# navigate to the nanogpt directory since the script assumes it's run from there
cd modded-nanogpt

#Adam's Beta1 = momentum
#Adam's Beta2 = 1.0 - lr_cov
#Adam's eps = damping

#use 2 H100s (for 80GB GPU memory,  use batch_size_pre_gpu=128) or (for 40GB GPU memory,  use batch_size_pre_gpu=64)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# EVON with whitening
uv run python -m torch.distributed.run --standalone --nproc_per_node=1 train_gpt2.py --experiment=nanoGPT --batch_size_pre_gpu=32 --n_embd=768 --mlp_ratio=4 --n_head=6 --attn_ratio=1 --is_merged=false --opt=evon --momentum=0.9 --cast_dtype=float32 --damping=1e-10 --debias_second_moment=False --schd=linear --evon_noise_damping=0 --whiten_evon_grad=True --mc_samples=1 --decoupled_wd=False --ess=5266386.597665445 --evon_phased_grads=True --ivon_hess_init=0.581980713452398 --lr=0.0172353632 --lr_cov=0.0009120067230437592 --momentum=0.943040091284246 --price_clip_ratio=1.5 --shampoo_beta=0.995 --weight_decay=0.000001

echo "Job completed."
