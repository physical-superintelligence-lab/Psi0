#!/bin/bash
set -euo pipefail

source .venv-psi/bin/activate

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PSI_ATTN_IMPLEMENTATION="${PSI_ATTN_IMPLEMENTATION:-sdpa}"
: "${CHECKPOINT_DIR:?set CHECKPOINT_DIR to the local Psi0 SONIC run directory}"
: "${CHECKPOINT_STEP:?set CHECKPOINT_STEP to 35000 or 40000}"

echo "Serving $CHECKPOINT_DIR/checkpoints/ckpt_$CHECKPOINT_STEP on GPU $CUDA_VISIBLE_DEVICES"

python -u src/psi/deploy/psi_serve_rtc_token-sonic.py \
    --host 0.0.0.0 \
    --port 8014 \
    --action_exec_horizon 30 \
    --policy psi \
    --rtc \
    --run-dir=${CHECKPOINT_DIR} \
    --ckpt-step=${CHECKPOINT_STEP}
