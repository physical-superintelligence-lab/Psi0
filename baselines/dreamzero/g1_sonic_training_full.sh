#!/bin/bash
set -e

# Full fine-tuning for the g1_sonic embodiment (G1 + sonic body-token, NO neck).
#   action 78D = action.hand_joints(14) + action.body_token(64)
#   state  43D = state.joint_positions
#
# Data: data/real/cleanup_table_2026-07-01_{train,val}/g1
# Includes denoise-based validation: every VAL_STEPS steps it runs full 16-step
# inference on a few val batches and logs to wandb:
#   val/hand_l1, val/action_l1 (denoised pred vs GT), val/pred_video
# (see experiment/denoise_val_callback.py). Control cadence with VAL_STEPS.

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Prefer the node-local copy that slurm_job.sh stages and mounts at /workspace/.venv-dreamzero
# (scripts/train/stage_venv_local.sh); fall back to the beegfs venv when run directly.
source "${DZ_VENV:-$([ -d /workspace/.venv-dreamzero ] && echo /workspace/.venv-dreamzero || echo .venv-dreamzero)}/bin/activate"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export HYDRA_FULL_ERROR=1
if [ -n "${WANDB_ENTITY:-}" ]; then
    export WANDB_ENTITY
fi

# ============ USER CONFIGURATION ============
G1_SONIC_DATA_ROOT=${G1_SONIC_DATA_ROOT:-"$PSI_HOME/data/mobile_pick_place_0709_psix_train/g1"}
G1_SONIC_VAL_DATA_ROOT=${G1_SONIC_VAL_DATA_ROOT:-"$PSI_HOME/data/mobile_pick_place_0709_psix_val/g1"}
PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-"$PSI_HOME/cache/checkpoints/DreamZero-AgiBot"}

TASK_NAME=${1:-${TASK_NAME:-g1_sonic}}
OUTPUT_DIR=${OUTPUT_DIR:-".runs/dreamzero/$TASK_NAME"}

MAX_STEPS=${MAX_STEPS:-15000}
SAVE_STEPS=${SAVE_STEPS:-500}
LEARNING_RATE=${LEARNING_RATE:-1e-5}

PER_DEVICE_BS=${PER_DEVICE_BS:-1}
GRAD_ACCUM=${GRAD_ACCUM:-1}

# ---- Validation (denoise hand-L1 / action-L1 + pred video -> wandb) ----
VAL_STEPS=${VAL_STEPS:-1000}            # val at step 1 (baseline) then every VAL_STEPS
VAL_NUM_BATCHES=${VAL_NUM_BATCHES:-20}  # val samples denoised PER GPU each eval
                                        # (metric averages over all GPUs: 20 x 8 = 160)
VAL_NUM_VIDEOS=${VAL_NUM_VIDEOS:-1}     # how many pred videos to log (rank0)

if [ -z "${NUM_GPUS}" ]; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-8}

WAN_CKPT_DIR=${WAN_CKPT_DIR:-"$PSI_HOME/cache/checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"$PSI_HOME/cache/checkpoints/umt5-xxl"}

# zero3 + VAE leaf fix handled in base.py; full FT wants zero3.
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"src/dreamzero/groot/vla/configs/deepspeed/zero3.json"}

MAX_CHUNK_SIZE=${MAX_CHUNK_SIZE:-2}
NUM_FRAMES=${NUM_FRAMES:-17}
echo "Task: $TASK_NAME -> g1_sonic FULL FT, bs=$PER_DEVICE_BS x accum=$GRAD_ACCUM x gpus=$NUM_GPUS, lr=$LEARNING_RATE, val every $VAL_STEPS steps"

torchrun --nproc_per_node "$NUM_GPUS" --standalone \
    src/dreamzero/groot/vla/experiment/experiment.py \
    report_to=${REPORT_TO:-wandb} \
    data=dreamzero/g1_sonic \
    wandb_project=psi \
    wandb_group=dreamzero \
    train_architecture=full \
    num_frames=$NUM_FRAMES \
    action_horizon=32 \
    num_views=1 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=32 \
    num_state_per_block=1 \
    max_action_dim=78 \
    max_state_dim=43 \
    seed=42 \
    training_args.learning_rate=$LEARNING_RATE \
    training_args.deepspeed="$DEEPSPEED_CONFIG" \
    save_steps=$SAVE_STEPS \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=$PER_DEVICE_BS \
    gradient_accumulation_steps=$GRAD_ACCUM \
    max_steps=$MAX_STEPS \
    logging_steps=10 \
    weight_decay=1e-5 \
    save_total_limit=5 \
    keep_checkpoint_every=${KEEP_CHECKPOINT_EVERY:-1000} \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    per_device_eval_batch_size=1 \
    ++val_steps=$VAL_STEPS \
    ++val_num_batches=$VAL_NUM_BATCHES \
    ++val_num_videos=$VAL_NUM_VIDEOS \
    ++val_hand_dim=14 \
    dataloader_pin_memory=true \
    dataloader_num_workers=${DATALOADER_NUM_WORKERS:-4} \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=false \
    max_chunk_size=$MAX_CHUNK_SIZE \
    frame_seqlen=220 \
    save_strategy=steps \
    relative_action=false \
    g1_sonic_data_root=$G1_SONIC_DATA_ROOT \
    g1_sonic_val_data_root=$G1_SONIC_VAL_DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=$PRETRAINED_MODEL_PATH \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=false \
    "${@:2}"