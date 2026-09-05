#!/bin/bash

set -e

source .venv-psi/bin/activate

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "Serving on GPU $CUDA_VISIBLE_DEVICES"

# Accept RUN_DIR and CKPT_STEP as command line arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 RUN_DIR CKPT_STEP"
    exit 1
fi

RUN_DIR=$1
CKPT_STEP=$2

serve_psi0_simple \
    --host 0.0.0.0 \
    --port 22085 \
    --policy=psi0 \
    --run-dir=$RUN_DIR \
    --ckpt-step=$CKPT_STEP \
    --action-exec-horizon=24 \
    --rtc
