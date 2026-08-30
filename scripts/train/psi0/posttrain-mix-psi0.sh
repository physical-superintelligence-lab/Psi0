#!/bin/bash

set -e

source "${PSI_VENV:-$([ -d /workspace/.venv-psi ] && echo /workspace/.venv-psi || echo .venv-psi)}/bin/activate"

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

NNODES=1
NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

args="posttrain_mix_psi0_config \
--seed=292285 \
--exp=posttrain \
--timestamp=$(date +"%y%m%d%H%M") \
--train.name=posttrain \
--train.data_parallel=ddp \
--train.mixed_precision=bf16 \
--train.train_batch_size=64 \
--train.resume_from_checkpoint=latest \
--train.max_checkpoints_to_keep=5 \
--train.gradient_accumulation_steps=1 \
--train.learning_rate=1e-4 \
--train.max_training_steps=1000000 \
--train.warmup_ratio=None \
--train.warmup_steps=500 \
--train.checkpointing_steps=100 \
--train.validation_steps=1000 \
--train.val_num_batches=20 \
--train.max_grad_norm=1.0 \
--train.lr_scheduler_type=constant \
--train.lr_scheduler_kwargs.betas 0.9 0.999 \
--train.lr_scheduler_kwargs.weight_decay=0.0 \
--train.lr_scheduler_kwargs.eps=1e-8 \
--log.report_to=wandb \
--log.log_freq=100 \
--data.upsample_rate=1 \
--data.chunk-size=16 \
--data.use-delta-actions \
--data.sampler=token_mixture \
--data.tokens_per_device=8640 \
--data.he.root-dir=/hfm/data/HE_RAW_no_static \
--data.he.ratio=0.5 \
--data.egodex.root-dir=/hfm/data/egodex \
--data.egodex.load-retarget \
--data.egodex.ratio=0.5 \
--data.use-delta-actions \
--data.transform.repack.action-chunk-size=16 \
--data.transform.repack.use-delta-actions \
--data.transform.repack.pad-action-dim=36 \
--data.transform.repack.pad-state-dim=36 \
--data.transform.repack.stage=postpre \
--data.transform.field.action_norm_type=bounds_q99 \
--data.transform.field.stat-path=assets/stats/he_raw_rel_stats_combined_no_static.json \
--data.transform.field.no-normalize-state \
--data.transform.field.pad-action-dim=36 \
--data.transform.field.pad-state-dim=36 \
--data.transform.field.action-norm-type=bounds_q99 \
--data.transform.model.adaptive-resize \
--data.transform.model.img-sizes.egodex 270 480 \
--data.transform.model.img-sizes.he 240 320 \
--data.transform.model.no-img-aug \
--model.model_name_or_path=/hfm/cache/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k \
--model.noise-scheduler=flow \
--model.n_conditions=0 \
--model.action-chunk-size=16 \
--model.action-dim=36 \
--model.action-exec-horizon=16 \
--model.observation-horizon=1 \
--model.odim=36 \
--model.view_feature_dim=2048 \
--model.no-tune-vlm \
--model.no-use_film \
--model.no-combined_temb
"

torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=29500 scripts/train.py \
    ${args}