#!/bin/bash

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

source .venv-openpi/bin/activate

NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
ulimit -n 65535
echo "Training with $NPROC_PER_NODE GPUs"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <task> "
    echo "Example: $0 G1WholebodyXMoveBendPickTeleop-v0"
    exit 1
fi

export task="$1"
echo "Task: $task"

torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC_PER_NODE src/openpi/train_pytorch.py \
        $task \
        --exp_name=$task \
        --save_interval=20000 \
        --checkpoint_base_dir=.runs/openpi-05