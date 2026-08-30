#!/bin/bash

TORCHRUN_PID=
PYTHON_BIN=
CLEANUP_RUNNING=0

cleanup() {
    if [ "$CLEANUP_RUNNING" -eq 1 ]; then return; fi
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

# When run directly on the host (not via sbatch), resolve to project root.
if [[ -z "$SLURM_JOB_ID" ]]; then
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi

source "${PSI_VENV:-$([ -d /workspace/.venv-psi ] && echo /workspace/.venv-psi || echo .venv-psi)}/bin/activate"
PYTHON_BIN=$(readlink -f "$(command -v python3)")

: "${OMP_NUM_THREADS:=32}"
export OMP_NUM_THREADS

NPROC_PER_NODE=$(echo "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" | tr ',' '\n' | wc -l)
ulimit -n 65535
echo "Training with $NPROC_PER_NODE GPUs"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <task> [exp]"
    echo "Example: $0 sonic g1neck"
    exit 1
fi

export task="$1"
#task_words=$(echo "$task" | tr '[:upper:]' '[:lower:]' | tr '_' ' ')
#default_exp=$(echo "$task_words" | awk '{if (NF>=2) print $1 "-" $2; else print $1}')
# export exp=${2:-$default_exp}
export exp=${2:-sonic}

echo "Task: $task"
echo "Experiment name: $exp"

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

args="
finetune_sonic_psi0_config \
--seed=292285 \
--exp=$exp-$task \
--train.name=finetune \
--train.data_parallel=ddp \
--train.mixed_precision=bf16 \
--train.train_batch_size=32 \
--train.max_checkpoints_to_keep=5 \
--train.gradient_accumulation_steps=1 \
--train.learning_rate=1e-4 \
--train.max_training_steps=40000 \
--train.warmup_ratio=None \
--train.warmup_steps=1000 \
--train.checkpointing_steps=5000 \
--train.validation_steps=1000 \
--train.val_num_batches=20 \
--train.max_grad_norm=1.0 \
--train.lr_scheduler_type=cosine \
--train.lr_scheduler_kwargs.weight_decay=1e-6 \
--train.lr_scheduler_kwargs.betas 0.95 0.999 \
--log.report_to=wandb \
--data.root_dir=data \
--data.train_repo_ids=cleanup_table_2026-07-01_train/g1 \
--data.val_repo_ids=cleanup_table_2026-07-01_val/g1 \
--data.transform.repack.action-keys action.body_token action[:14] \
--data.transform.repack.dataset-name=neckless \
--data.transform.repack.pad-action-dim=78 \
--data.transform.repack.pad-state-dim=43 \
--data.transform.repack.instruction-key=task_description \
--data.transform.field.stat-path=cleanup_table_2026-07-01_train/g1/meta/stats_psi0.json \
--data.transform.field.stat-action-keys action.body_token action[:14] \
--data.transform.field.action_norm_type=bounds \
--data.transform.field.normalize-state \
--data.transform.field.pad-action-dim=78 \
--data.transform.field.pad-state-dim=43 \
--data.transform.model.img-aug \
--data.transform.model.resize.size 240 320 \
--data.transform.model.center_crop.size 240 320 \
--model.model_name_or_path=cache/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k \
--model.pretrained-action-header-path=cache/checkpoints/psi0/postpre.1by1.pad36.2601131206.ckpt.he30k \
--model.noise-scheduler=flow \
--model.train-diffusion-steps=1000 \
--model.n_conditions=0 \
--model.action-chunk-size=30 \
--model.action-dim=78 \
--model.action-exec-horizon=30 \
--model.observation-horizon=1 \
--model.odim=43 \
--model.view_feature_dim=2048 \
--model.no-tune-vlm \
--model.no-use_film \
--model.no-combined_temb \
--model.no-rtc \
--model.max-delay=8
"

cat <<EOF
Running:
torchrun \\
  --nnodes=$NNODES \\
  --nproc_per_node=$NPROC_PER_NODE \\
  --node_rank=$NODE_RANK \\
  --master_addr=$MASTER_ADDR \\
  --master_port=$MASTER_PORT \\
  scripts/train.py \\
  ${args}
EOF

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
