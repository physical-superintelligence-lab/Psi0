#!/bin/bash
# psi0 single-task finetune on one SIMPLE MP (AMO) task, e.g. G1WholebodyBendPickMP-v0
# (32-D states, 36-D action, layout in meta/modality.json).
# Model/optimizer recipe follows finetune-sonic-simple-psi0.sh (6-block psi0 header, frozen
# VLM, no FiLM / combined-temb / RTC); data args follow finetune-simple-decoupled-wbc.sh
# (36-D action/state, min/max "bounds" norm over meta/stats_psi0.json).
#
# Split first (writes $TASK-{train,val} next to it; copies stats_psi0.json unchanged):
#   python scripts/data/split_lerobot_dataset.py /hfm/data/simple/$TASK --per-task 2 --seed 0
# Run:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/psi0/finetune-simple-psi0-mp.sh <TASK> [exp]
# Env: ROOT_DIR, MAX_STEPS, BATCH_SIZE (per GPU), MASTER_PORT, TS (resume run dir suffix), DRY_RUN.

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
    if [ -n "$TORCHRUN_PID" ]; then
        wait "$TORCHRUN_PID" 2>/dev/null || true
    fi
    if [ -n "$TORCHRUN_PID" ] && kill -0 "$TORCHRUN_PID" 2>/dev/null; then
        kill -KILL "$TORCHRUN_PID" 2>/dev/null || true
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
# .env sets CUDA_LAUNCH_BLOCKING=true for debugging; it serialises every kernel launch.
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
[ "$CUDA_LAUNCH_BLOCKING" = "true" ] && export CUDA_LAUNCH_BLOCKING=0

NPROC_PER_NODE=$(echo "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" | tr ',' '\n' | wc -l)
ulimit -n 65535
echo "Training with $NPROC_PER_NODE GPUs"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <task> [exp]"
    echo "Example: $0 G1WholebodyBendPickMP-v0"
    exit 1
fi

TASK="$1"
export exp="${2:-psi0-mp}"
ROOT_DIR="${ROOT_DIR:-/hfm/data/simple}"
MAX_STEPS="${MAX_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TS="${TS:-}"   # %y%m%d%H%M run-dir suffix; empty -> fresh run

for split in train val; do
    for f in meta/info.json meta/stats_psi0.json meta/episodes.jsonl; do
        [ -s "$ROOT_DIR/$TASK-$split/$f" ] || { echo "FATAL: missing $ROOT_DIR/$TASK-$split/$f" >&2; exit 1; }
    done
done

echo "Dataset: $ROOT_DIR/$TASK-{train,val}"
echo "Experiment name: $exp-$TASK"

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

args="
finetune_simple_psi0_config \
--seed=292285 \
--exp=$exp-$TASK \
${TS:+--timestamp=$TS --train.resume_from_checkpoint=latest} \
--train.name=finetune \
--train.data_parallel=ddp \
--train.mixed_precision=bf16 \
--train.train_batch_size=$BATCH_SIZE \
--train.max_checkpoints_to_keep=5 \
--train.gradient_accumulation_steps=1 \
--train.learning_rate=1e-4 \
--train.max_training_steps=$MAX_STEPS \
--train.warmup_ratio=None \
--train.warmup_steps=1000 \
--train.checkpointing_steps=5000 \
--train.validation_steps=1000 \
--train.val_num_batches=-1 \
--train.val_batch_size=$BATCH_SIZE \
--train.max_grad_norm=1.0 \
--train.lr_scheduler_type=cosine \
--train.lr_scheduler_kwargs.weight_decay=1e-6 \
--train.lr_scheduler_kwargs.betas 0.95 0.999 \
--log.report_to=wandb \
--data.root_dir=$ROOT_DIR \
--data.train-repo-ids=$TASK-train \
--data.val-repo-ids=$TASK-val \
--data.transform.repack.pad-action-dim=36 \
--data.transform.repack.pad-state-dim=36 \
--data.transform.field.stat-path=meta/stats_psi0.json \
--data.transform.field.stat-action-key=action \
--data.transform.field.stat-state-key=states \
--data.transform.field.action_norm_type=bounds \
--data.transform.field.no-use-norm-mask \
--data.transform.field.normalize-state \
--data.transform.field.pad-action-dim=36 \
--data.transform.field.pad-state-dim=36 \
--data.transform.model.img-aug \
--data.transform.model.resize.size 180 320 \
--data.transform.model.center_crop.size 180 320 \
--model.model_name_or_path=cache/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k \
--model.pretrained-action-header-path=cache/checkpoints/psi0/postpre.1by1.pad36.2601131206.ckpt.he30k \
--model.noise-scheduler=flow \
--model.train-diffusion-steps=1000 \
--model.n_conditions=0 \
--model.action-chunk-size=30 \
--model.action-dim=36 \
--model.action-exec-horizon=30 \
--model.observation-horizon=1 \
--model.odim=36 \
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
EOF
printf '  %s \\\n' ${args} | sed '$ s/ \\$//'
if [ -n "${DRY_RUN:-}" ]; then
    echo "DRY_RUN set - not launching torchrun"
    exit 0
fi

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
