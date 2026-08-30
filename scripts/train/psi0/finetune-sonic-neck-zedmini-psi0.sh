#!/bin/bash
# psi0 (sonic, neck) finetune on .data/g1_sonic_lerobot_0810_merged_{train,val} @ native 672x384.
# Based on finetune-sonic-neck-realsense.sh.
#
# Delta vs ...-combined-dit-layerwise.sh: DEEPER action header, 6 -> 12 blocks
# (--model.num-blocks=12), with one VLM layer fused per block, so vlm-layer-indices
# now lists 12 layers (evenly spread across the 28-layer Qwen VLM, ending at 28).
#
# Data: 622 train / 9 val episodes, 50 tasks, 30 fps, single 'head' camera 384x672.
#
# Deltas vs the realsense script:
#   action      merged pack has ONE 80-D `action` column (no action.body_token /
#               action.neck columns), so keys are slices. Concat order is
#               body64 + hand14 + neck2 = 80; eval/deploy must slice the same way:
#                 action[16:80]  token  (was action.body_token)
#                 action[:14]    hand
#                 action[14:16]  neck   (was action.neck)
#   video       observation.images.egocentric -> observation.images.head
#   instruction task_description -> annotation.task
#   canvas      240x320 -> 384x672 (native; already on Qwen's factor-32 grid, so
#               smart_resize is a no-op). 252 vision tokens/img vs 80.
#   batch       32 -> 16 per GPU (3.15x vision sequence)
#   steps       40000 -> 100000
#   tune VLM    off -> on (whole VLM, one optimizer group per component)
#   state       unchanged: 45-D observation.state, odim/pad 45
#
# Train/val are the _train / _val packs from scripts/data/split_sonic_by_task_group.py
# (622 / 9 episodes; one held-out episode per task group). Both carry the SAME
# whole-dataset stats_psi0.json, so the two splits normalize identically.
#
# Every hyper-parameter is written literally in the `args` block below -- edit it
# there, not through the environment. VLM finetuning in particular:
#     --model.tune-vlm             all three components; swap for any subset of
#                                  --model.tune-mm-vision / -mm-mlp / -mm-llm
#     --model.lang-backbone-lr     1e-7   LLM
#     --model.vision-tower-lr      1e-6   vision tower
#     --model.mm-projector-lr      1e-5   patch merger
#   Each tuned component becomes its own optimizer group at its own LR
#   (FinetuneTrainer.vlm_trainable_components), and the three rates follow
#   QwenLM/Qwen3-VL's qwen-vl-finetune recipe -- note the LLM takes the SMALLEST.
#   Drop the tune flag entirely to freeze the VLM and train only the action head.
#
#   Tuning any component loads the VLM in fp32 (AdamW updates weights in place
#   and at these rates bf16 rounds every update away), which costs ~32GB/GPU for
#   the 2.13B backbone plus 7.4GB for the 0.5B action header. That is why
#   --model.gradient-checkpointing is on here: without it batch 16 needs ~71GB
#   and will not fit an 80GB card. Turn it off (--model.no-gradient-checkpointing)
#   only alongside a smaller batch.
#
#   The backend is ddp (--train.data_parallel=ddp). For ZeRO instead, set that to
#   deepspeed and add --train.deepspeed_config=scripts/deepspeed/zero3.json; pick
#   a json WITHOUT "optimizer"/"scheduler" sections (so not zero3_offload.json),
#   since this trainer builds both and accelerate then rejects a client optimizer.
#   ZeRO also keeps the VLM in bf16 with its own fp32 masters, clips gradients
#   engine-side (so train/grad_norm_* stops being logged) and shards the ~39GB of
#   params+grads+optimizer state across GPUs.
#
# Usage: bash scripts/train/psi0/finetune-sonic-neck-zedmini-as-a-baseline-270x480-10x-vlm-lr-combined-dit.sh [exp] [timestamp]
#
# The timestamp is the %y%m%d%H%M suffix of the run dir,
#   .runs/finetune/<exp>....b128.gpus8.<timestamp>
# and is minted at launch when omitted. Pass one to make the run resumable: the same
# <exp> <timestamp> pair reuses that run dir and picks up from its newest checkpoint
# (and the same W&B run), so re-running the identical command line continues training.
#   TS=$(date +%y%m%d%H%M)
#   bash scripts/train/psi0/finetune-sonic-neck-zedmini-as-a-baseline-270x480-10x-vlm-lr-combined-dit.sh ff "$TS"   # start
#   bash scripts/train/psi0/finetune-sonic-neck-zedmini-as-a-baseline-270x480-10x-vlm-lr-combined-dit.sh ff "$TS"   # resume

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

export exp="${1:-ff}"
TS="${2:-}"   # %y%m%d%H%M run-dir suffix; empty -> fresh run, config mints one

source "${PSI_VENV:-$([ -d /workspace/.venv-psi ] && echo /workspace/.venv-psi || echo .venv-psi)}/bin/activate"
PYTHON_BIN=$(readlink -f "$(command -v python3)")

: "${OMP_NUM_THREADS:=32}"
export OMP_NUM_THREADS

NPROC_PER_NODE=$(echo "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" | tr ',' '\n' | wc -l)
ulimit -n 65535
echo "Training with $NPROC_PER_NODE GPUs"


echo "Experiment name: $exp"

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-$((29600 + ${SLURM_JOB_ID:-0} % 2000))}

# --- data preflight ----------------------------------------------------------
ROOT_DIR=".data"
TRAIN_ID="g1_sonic_lerobot_0810_merged_train"
VAL_ID="g1_sonic_lerobot_0810_merged_val"
PACK="$ROOT_DIR/$TRAIN_ID"
STATS="$PACK/meta/stats_psi0.json"

for p in "$ROOT_DIR/$TRAIN_ID" "$ROOT_DIR/$VAL_ID"; do
    [ -d "$p" ]                 || { echo "FATAL: missing pack $p (run scripts/data/split_sonic_by_task_group.py)" >&2; exit 1; }
    [ -s "$p/meta/modality.json" ] || { echo "FATAL: missing $p/meta/modality.json" >&2; exit 1; }
done

# stats_psi0.json == stats.json by construction (simple_to_sonic_lerobot.py:219-220).
if [ ! -s "$STATS" ]; then
    [ -s "$PACK/meta/stats.json" ] || { echo "FATAL: neither stats_psi0.json nor stats.json in $PACK/meta" >&2; exit 1; }
    cp "$PACK/meta/stats.json" "$STATS"
    echo "Created $STATS (copy of stats.json)"
fi
[ -s "$STATS" ] || { echo "FATAL: stats missing: $STATS" >&2; exit 1; }

args="
finetune_sonic_psi0_config \
--seed=292285 \
--exp=$exp \
${TS:+--timestamp=$TS --train.resume_from_checkpoint=latest} \
--train.name=finetune \
--train.data_parallel=ddp \
--train.mixed_precision=bf16 \
--train.train_batch_size=16 \
--train.max_checkpoints_to_keep=5 \
--train.gradient_accumulation_steps=1 \
--train.learning_rate=1e-4 \
--train.max_training_steps=100000 \
--train.warmup_ratio=None \
--train.warmup_steps=1000 \
--train.checkpointing_steps=20000 \
--train.validation_steps=5000 \
--train.val_num_batches=20 \
--train.max_grad_norm=1.0 \
--train.lr_scheduler_type=cosine \
--train.lr_scheduler_kwargs.weight_decay=1e-6 \
--train.lr_scheduler_kwargs.betas 0.95 0.999 \
--log.report_to=wandb \
--data.root_dir=$ROOT_DIR \
--data.train_repo_ids=$TRAIN_ID \
--data.val_repo_ids=$VAL_ID \
--data.transform.repack.image-keys observation.images.head \
--data.transform.repack.action-keys action[16:80] action[:14] action[14:16] \
--data.transform.repack.dataset-name=g1sonic0810 \
--data.transform.repack.pad-action-dim=80 \
--data.transform.repack.pad-state-dim=45 \
--data.transform.repack.instruction-key=annotation.task \
--data.transform.field.stat-path=$STATS \
--data.transform.field.stat-action-keys action[16:80] action[:14] action[14:16] \
--data.transform.field.action_norm_type=bounds \
--data.transform.field.normalize-state \
--data.transform.field.pad-action-dim=80 \
--data.transform.field.pad-state-dim=45 \
--data.transform.model.img-aug \
--data.transform.model.resize.size 270 480 \
--data.transform.model.center_crop.size 270 480 \
--model.model_name_or_path=cache/checkpoints/psi0/pre.fast.2605160748.ckpt.ego390k \
--model.pretrained-action-header-path=cache/checkpoints/psi0/postpre.1by1.pad36.2601131206.ckpt.he30k \
--model.noise-scheduler=flow \
--model.train-diffusion-steps=1000 \
--model.n_conditions=0 \
--model.action-chunk-size=30 \
--model.action-dim=80 \
--model.action-exec-horizon=30 \
--model.observation-horizon=1 \
--model.odim=45 \
--model.view_feature_dim=2048 \
--model.tune-vlm \
--model.lang-backbone-lr=1e-6 \
--model.vision-tower-lr=1e-5 \
--model.mm-projector-lr=1e-4 \
--model.gradient-checkpointing \
--model.no-use_film \
--model.qk-norm=rms_norm \
--model.combined-temb \
--model.num-blocks=12 \
--model.vlm-layer-indices 3 5 8 10 12 14 17 19 21 23 26 28 \
--model.state-drop-prob=0.8 \
--model.pooled-text-encoder=clip \
--model.pooled-text-encoder-path=openai/clip-vit-large-patch14 \
--model.pooled-projection-dim=768 \
--model.pooled-cache-path=clip_pooled_cache.pt \
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
