#!/bin/bash

source .venv-psi/bin/activate

export CUDA_VISIBLE_DEVICES=0
echo "Training with $nprocs GPUs, which is/are $CUDA_VISIBLE_DEVICES"

python src/psi/deploy/psi_serve_rtc_token-sonic.py \
    --host 0.0.0.0 \
    --port 8014 \
    --action_exec_horizon 30 \
    --policy psi \
    --rtc \
    --run-dir=${CHECKPOINT_DIR} \
    --ckpt-step=${CHECKPOINT_STEP}
