#!/bin/bash

# Usage:
#   ./scripts/train/psi0/pretrain-egoverse-psi0-fast.sh [MASTER_ADDR] [MASTER_PORT] [NODE_RANK] [NNODES]
#
# Args (positional):
#   1) MASTER_ADDR  IP/hostname of rank-0 node (default: 172.19.0.101)
#   2) MASTER_PORT  torch.distributed master port (default: 29500)
#   3) NODE_RANK    rank of this node in [0, NNODES-1] (default: 1)
#   4) NNODES       total number of nodes (default: 2)
#
# Examples:
#   # Node 0 (2-node run)
#   ./scripts/train/psi0/pretrain-egoverse-psi0-fast.sh 172.19.0.101 29500 0 2
#
#   # Node 1 (2-node run)
#   ./scripts/train/psi0/pretrain-egoverse-psi0-fast.sh 172.19.0.101 29500 1 2

set -e

TORCHRUN_PID=
PYTHON_BIN=
CLEANUP_RUNNING=0

cleanup() {
    if [ "$CLEANUP_RUNNING" -eq 1 ]; then
        return
    fi
    CLEANUP_RUNNING=1

    echo "Interrupted - stopping torchrun and worker processes..."

    trap - INT TERM

    if [ -n "$TORCHRUN_PID" ] && kill -0 "$TORCHRUN_PID" 2>/dev/null; then
        kill -TERM "$TORCHRUN_PID" 2>/dev/null || true
    fi

    if [ -n "$PYTHON_BIN" ]; then
        pkill -TERM -f "$PYTHON_BIN" 2>/dev/null || true
    fi

    if [ -n "$TORCHRUN_PID" ]; then
        wait "$TORCHRUN_PID" 2>/dev/null || true
    fi

    if [ -n "$TORCHRUN_PID" ] && kill -0 "$TORCHRUN_PID" 2>/dev/null; then
        kill -KILL "$TORCHRUN_PID" 2>/dev/null || true
    fi

    if [ -n "$PYTHON_BIN" ]; then
        pkill -KILL -f "$PYTHON_BIN" 2>/dev/null || true
    fi
}
trap cleanup INT TERM

source "${PSI_VENV:-$([ -d /workspace/.venv-psi ] && echo /workspace/.venv-psi || echo .venv-psi)}/bin/activate"
PYTHON_BIN=$(readlink -f "$(command -v python3)")

export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} #

# IB Config for Multi-node Training
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_bond_0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_NET_GDR_LEVEL=2

# take everything offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# fix cuda  (specific to h1,h2 nodes)
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs 
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# multi-node training over IB connection
MASTER_ADDR=${1:-172.19.0.101}
MASTER_PORT=${2:-29500}
NODE_RANK=${3:-1}
NNODES=${4:-2}
NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# training args
args="pretrain_egoverse_qwen3vl_config \
model.action-tokenizer:fast \
--seed=7 \
--exp=pre \
--timestamp=202605051508 \
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
--train.checkpointing_steps=10000 \
--train.validation_steps=0 \
--train.max_grad_norm=1.0 \
--train.lr_scheduler_type=constant \
--train.lr_scheduler_kwargs.betas 0.9 0.999 \
--train.lr_scheduler_kwargs.weight_decay=0.0 \
--train.lr_scheduler_kwargs.eps=1e-8 \
--log.report_to=wandb \
--data.chunk_size=1 \
--data.upsample_rate=3 \
--data.transform.field.action_norm_type=bounds_q99 \
--data.transform.field.stat-action-key=egodex \
--data.transform.field.stat-path=assets/stats/egodex_stat_all.json \
--data.transform.model.resize.size 240 320 \
--data.root-dir=/mnt/beegfs/datasets/egoverse \
--data.use-delta-actions \
--data.transform.model.no-img-aug \
--model.model_name_or_path=Qwen/Qwen3-VL-2B-Instruct \
--model.action_tokenizer.bins=2048 \
--model.action_tokenizer.pretrained_checkpoint=src/fast/egodex-rel-50w-1x48-v2048-s100 \
--model.tune-mm-llm \
--model.tune-mm-vision \
--model.tune-mm-mlp \
--model.mm_projector_lr=1e-5 \
--model.vision_tower_lr=1e-5
"

echo "Running command:"
echo "torchrun --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT scripts/train.py ${args}"

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train.py \
    ${args} &
TORCHRUN_PID=$!
wait "$TORCHRUN_PID"