#!/bin/bash
# psi0 finetune on one SIMPLE decoupled-WBC subset (50 fps; 32-D states, 36-D action):
#   teleop12: SIMPLE-decoupled-wbc-Teleop (12 tasks, 1128 episodes, 465k frames)
#   mp5:      SIMPLE-decoupled-wbc-MP     (5 tasks, 389 episodes, 141k frames)
# Recipe follows the mix24/mix33 psi0 runs (LN action head, rtc; but action+state norm uses
# "bounds" = stats min/max instead of their bounds_q99), with the
# model architecture and robustness augmentations of finetune-sonic-psi-dream-baseline.sh:
#   model: 12-block head with layer-wise VLM fusion (vlm-layer-indices), combined-temb adaLN
#     on a frozen CLIP pooled instruction embedding, proprio state as action token 0 with a
#     learned null token for dropped states, VLM tuned with per-component LRs.
#     NOTE the warm start is still the 6-block psi0 header: blocks 0-5 load, blocks 6-11, the
#     combined-temb path and the state/null tokens start fresh (see the header load warnings).
#   data: state temporal jitter (repack), Gaussian state noise (field), view-aug crop (model),
#     optional per-task sample weights.
#   Knobs (env): STATE_DROP_PROB STATE_JITTER STATE_JITTER_PROB STATE_NOISE_STD
#                VIEW_AUG_MIN_SCALE VIEW_AUG_PROB TASK_SAMPLE_WEIGHTS
#
# Download:
#   /hfm/data/simple/SIMPLE-decoupled-wbc-{Teleop,MP}
# Split (writes <name>-{train,val} next to each):
#   for d in Teleop MP; do
#     python scripts/data/split_lerobot_dataset.py /hfm/data/simple/SIMPLE-decoupled-wbc-$d --per-task 2 --seed 0
#   done
# Run:
#   bash scripts/train/psi0/finetune-simple-decoupled-wbc.sh {teleop12|mp5} [exp]
#   (exp defaults to dwbc-teleop12 / dwbc-mp5)
# Env: TASK (override the dataset name under ROOT_DIR; trains on $TASK-train, validates on $TASK-val),
#      ROOT_DIR, MAX_STEPS, BATCH_SIZE, TS (resume run dir suffix),
#      TUNE_VLM (1 = tune VLM with per-component LRs, 0 = freeze it), NUM_WORKERS (train
#      DataLoader workers per GPU), YAW_SINCOS (default 1 = encode target yaw, action[35], as
#      (sin, cos): 37-D model action with fixed [-1, 1] bounds on both; the server decodes back
#      to the 36-D action; 0 = raw 36-D action; see docs/plans/yaw-sincos-encoding.md),
#      STAT_PATH (action/state min/max stats; relative = under $TASK-train, or absolute, e.g. the
#      merged /hfm/data/simple/SIMPLE-decoupled-wbc/meta/stats_psi0.json for a single-task run).
# Val: $TASK-val holds 2 random episodes per task (seed 0; Teleop 24, MP 10), split off on
# disk; every 2000 steps eval runs over the whole val split (val_num_batches=-1).
# Default 40k steps x 256 global batch ~ 22 epochs (teleop12) / ~75 epochs (mp5).

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
# .env sets CUDA_LAUNCH_BLOCKING=true for debugging; it serialises every kernel launch.
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
[ "$CUDA_LAUNCH_BLOCKING" = "true" ] && export CUDA_LAUNCH_BLOCKING=0

NPROC_PER_NODE=$(echo "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" | tr ',' '\n' | wc -l)
ulimit -n 65535
echo "Training with $NPROC_PER_NODE GPUs"

SUBSET="${1:-}"
case "$SUBSET" in
    teleop12|telop12) DEFAULT_TASK=SIMPLE-decoupled-wbc-Teleop; SUBSET=teleop12 ;;
    mp5)              DEFAULT_TASK=SIMPLE-decoupled-wbc-MP ;;
    *) echo "usage: $0 {teleop12|mp5} [exp]" >&2; exit 1 ;;
esac

ROOT_DIR="${ROOT_DIR:-/hfm/data/simple}"
TASK="${TASK:-$DEFAULT_TASK}"
MAX_STEPS="${MAX_STEPS:-40000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TS="${TS:-}"   # %y%m%d%H%M run-dir suffix; empty -> fresh run
TUNE_VLM="${TUNE_VLM:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
vlm_args="--model.no-tune-vlm"
[ "$TUNE_VLM" = "1" ] && vlm_args="--model.tune-vlm"
STAT_PATH="${STAT_PATH:-meta/stats_psi0.json}"
YAW_SINCOS="${YAW_SINCOS:-1}"
ADIM=36
yaw_args=""
if [ "$YAW_SINCOS" = "1" ]; then
    ADIM=37
    yaw_args="--data.transform.repack.yaw-sincos --data.transform.field.yaw-sincos"
fi
export exp="${2:-dwbc-$SUBSET}"

for split in train val; do
    for f in meta/info.json meta/stats_psi0.json meta/episodes.jsonl; do
        [ -s "$ROOT_DIR/$TASK-$split/$f" ] || { echo "FATAL: missing $ROOT_DIR/$TASK-$split/$f" >&2; exit 1; }
    done
done

echo "Dataset: $ROOT_DIR/$TASK-{train,val}"
echo "Experiment name: $exp"
echo "Stats: $STAT_PATH"
echo "VLM: $vlm_args, train DataLoader workers per GPU: $NUM_WORKERS, yaw sin/cos: $YAW_SINCOS (action dim $ADIM)"

# --- robustness knobs (same defaults as finetune-sonic-psi-dream-baseline.sh) ---------
STATE_DROP_PROB="${STATE_DROP_PROB:-0.1}"     # per-sample state drop -> learned null token; 0 disables
STATE_JITTER="${STATE_JITTER:-10}"            # frames (50 fps here: +-0.2s); 0 disables
STATE_JITTER_PROB="${STATE_JITTER_PROB:-0.5}"
STATE_NOISE_STD="${STATE_NOISE_STD:-0.05}"    # in normalized [-1,1] units; 0 disables
VIEW_AUG_MIN_SCALE="${VIEW_AUG_MIN_SCALE:-0.85}"
VIEW_AUG_PROB="${VIEW_AUG_PROB:-1.0}"
# "<task substring>=<weight> ..." (space separated); empty -> uniform sampling
TASK_SAMPLE_WEIGHTS="${TASK_SAMPLE_WEIGHTS-}"
task_weight_args=""
[ -n "$TASK_SAMPLE_WEIGHTS" ] && task_weight_args="--data.task-sample-weights $TASK_SAMPLE_WEIGHTS"
echo "State aug: drop=${STATE_DROP_PROB} (learned null token) jitter=+-${STATE_JITTER}f p=${STATE_JITTER_PROB} noise=${STATE_NOISE_STD}; view aug: min_scale=${VIEW_AUG_MIN_SCALE} p=${VIEW_AUG_PROB}; task weights: ${TASK_SAMPLE_WEIGHTS:-none}"

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

args="
finetune_simple_psi0_config \
--seed=292285 \
--exp=$exp \
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
--train.warmup_steps=2000 \
--train.checkpointing_steps=10000 \
--train.validation_steps=2000 \
--train.skip-first-eval \
--train.val_num_batches=-1 \
--train.val_batch_size=$BATCH_SIZE \
--train.num_workers=$NUM_WORKERS \
--train.max_grad_norm=1.0 \
--train.lr_scheduler_type=cosine \
--train.lr_scheduler_kwargs.weight_decay=0.01 \
--train.lr_scheduler_kwargs.eps=1e-6 \
--train.lr_scheduler_kwargs.betas 0.95 0.999 \
--log.report_to=wandb \
--log.log_freq=1000 \
--data.root_dir=$ROOT_DIR \
--data.train-repo-ids=$TASK-train \
--data.val-repo-ids=$TASK-val \
--data.transform.repack.pad-action-dim=$ADIM \
--data.transform.repack.pad-state-dim=36 \
--data.transform.repack.state-temporal-jitter=$STATE_JITTER \
--data.transform.repack.state-temporal-jitter-prob=$STATE_JITTER_PROB \
--data.transform.field.stat-path=$STAT_PATH \
--data.transform.field.stat-action-key=action \
--data.transform.field.stat-state-key=states \
--data.transform.field.action_norm_type=bounds \
--data.transform.field.no-use-norm-mask \
--data.transform.field.normalize-state \
--data.transform.field.state-noise-std=$STATE_NOISE_STD \
--data.transform.field.pad-action-dim=$ADIM \
--data.transform.field.pad-state-dim=36 \
--data.transform.model.img-aug \
--data.transform.model.view-aug \
--data.transform.model.view-aug-min-scale=$VIEW_AUG_MIN_SCALE \
--data.transform.model.view-aug-prob=$VIEW_AUG_PROB \
--data.transform.model.resize.size 180 320 \
--data.transform.model.center_crop.size 180 320 \
$task_weight_args \
$yaw_args \
--model.model_name_or_path=cache/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k \
--model.pretrained-action-header-path=cache/checkpoints/psi0/postpre.1by1.pad36.2601131206.ckpt.he30k \
--model.noise-scheduler=flow \
--model.train-diffusion-steps=1000 \
--model.n_conditions=0 \
--model.action-chunk-size=30 \
--model.action-dim=$ADIM \
--model.action-exec-horizon=30 \
--model.observation-horizon=1 \
--model.odim=36 \
--model.dropout=0.0 \
--model.state-feature-dropout=0.0 \
--model.view_feature_dim=2048 \
$vlm_args \
--model.lang-backbone-lr=1e-6 \
--model.vision-tower-lr=1e-5 \
--model.mm-projector-lr=1e-4 \
--model.gradient-checkpointing \
--model.no-use_film \
--model.qk-norm=rms_norm \
--model.combined-temb \
--model.num-blocks=12 \
--model.vlm-layer-indices 3 5 8 10 12 14 17 19 21 23 26 28 \
--model.state-drop-prob=$STATE_DROP_PROB \
--model.state-as-action-token \
--model.state-null-token \
--model.pooled-text-encoder=clip \
--model.pooled-text-encoder-path=openai/clip-vit-large-patch14 \
--model.pooled-projection-dim=768 \
--model.pooled-cache-path=clip_pooled_cache_simple.pt \
--model.rtc \
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
