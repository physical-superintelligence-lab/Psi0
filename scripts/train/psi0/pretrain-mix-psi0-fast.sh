#!/bin/bash

set -e

source "${PSI_VENV:-$([ -d /workspace/.venv-psi ] && echo /workspace/.venv-psi || echo .venv-psi)}/bin/activate"

export OMP_NUM_THREADS=8
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

NNODES=1
NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

args="pretrain_mix_qwen3vl_config \
model.action-tokenizer:fast \
--seed=7 \
--exp=pre \
--timestamp=$(date +"%y%m%d%H%M") \
--train.name=pretrain \
--train.data_parallel=deepspeed \
--train.deepspeed_config=scripts/deepspeed/zero3.json \
--train.mixed_precision=bf16 \
--train.train_batch_size=16 \
--train.resume_from_checkpoint=latest \
--train.max_checkpoints_to_keep=5 \
--train.gradient_accumulation_steps=1 \
--train.learning_rate=1e-4 \
--train.max_training_steps=1000000 \
--train.warmup_ratio=None \
--train.warmup_steps=0 \
--train.checkpointing_steps=100 \
--train.validation_steps=0 \
--train.max_grad_norm=1.0 \
--train.lr_scheduler_type=constant \
--train.lr_scheduler_kwargs.betas 0.9 0.999 \
--train.lr_scheduler_kwargs.weight_decay=0.0 \
--train.lr_scheduler_kwargs.eps=1e-8 \
--log.report_to=wandb \
--data.chunk_size=1 \
--data.upsample_rate=3 \
--data.use-delta-actions \
--data.he.root-dir=/hfm/data/HE_RAW \
--data.egodex.root-dir=/hfm/data/egodex \
--data.sampler=token_mixture \
--data.he.ratio=0.5 \
--data.egodex.ratio=0.5 \
--data.tokens-per-device=8640 \
--data.transform.repack.action-chunk-size=1 \
--data.transform.repack.use-delta-actions \
--data.transform.repack.robot-type=both \
--data.transform.field.action_norm_type=bounds_q99 \
--data.transform.field.stat-action-key=egodex \
--data.transform.field.stat-path=assets/stats/egodex_stat_all.json \
--data.transform.model.adaptive-resize \
--data.transform.model.img-sizes.egodex 270 480 \
--data.transform.model.img-sizes.he 240 320 \
--data.transform.model.no-img-aug \
--model.model_name_or_path=/hfm/cache/checkpoints/psi0/pre.fast.egodex.2512241941.ckpt200k \
--model.action_tokenizer.bins=2048 \
--model.action_tokenizer.pretrained_checkpoint=src/fast/egodex-rel-50w-1x48-v2048-s100 \
--model.tune-mm-llm \
--model.tune-mm-vision \
--model.tune-mm-mlp \
--model.mm_projector_lr=1e-5 \
--model.vision_tower_lr=1e-5
"

torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=29500 scripts/train.py \
    ${args}