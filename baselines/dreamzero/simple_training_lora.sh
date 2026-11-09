#!/bin/bash
set -e
# set -x

# ----------------------------------------------------------------------------
# DreamZero SIMPLE LoRA fine-tuning.
#
# Run directly on a node:
#   ./baselines/dreamzero/simple_training_lora.sh <TASK_NAME> [extra hydra overrides...]
#
# Run under Slurm (single- or multi-node) via the shared wrapper:
#   sbatch scripts/train/slurm_job.sh baselines/dreamzero/simple_training_lora.sh <TASK_NAME> [extra hydra overrides...]
#
# slurm_job.sh runs this inside the training container and exports
# NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT, which the torchrun launch below
# consumes. When run directly these default to a single-node rendezvous.
# ----------------------------------------------------------------------------

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

# When run directly on the host (not via sbatch), resolve to project root so the
# relative paths below (.env, .venv-dreamzero, src/...) work from anywhere.
if [[ -z "$SLURM_JOB_ID" ]]; then
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# -- for conda env dreamzero
# CONDA_ENV_NAME="${CONDA_ENV_NAME:-dreamzero}"
# if [ "${SKIP_VENV_ACTIVATE:-0}" != "1" ]; then
#     __conda_base="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
#     # shellcheck disable=SC1091
#     source "${__conda_base}/etc/profile.d/conda.sh"
#     conda activate "${CONDA_ENV_NAME}"
# fi
# export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

# -- for h100
# source /mnt/beegfs/shared/miniforge3/etc/profile.d/conda.sh
# conda activate /mnt/beegfs/scratch/songlinwei/conda-envs/dreamzero-official

# Prefer the node-local copy that slurm_job.sh stages and mounts at /workspace/.venv-dreamzero
# (scripts/train/stage_venv_local.sh); fall back to the beegfs venv when run directly.
source "${DZ_VENV:-$([ -d /workspace/.venv-dreamzero ] && echo /workspace/.venv-dreamzero || echo .venv-dreamzero)}/bin/activate"
PYTHON_BIN=$(readlink -f "$(command -v python3)")


# Ensure cuDNN sub-libraries load from the active env (conda or venv) rather than
# a mismatched system/container cuDNN. slurm_job.sh prepends /usr/lib to
# LD_LIBRARY_PATH, which otherwise pulls in mismatched sublibs and triggers
# CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED. See baselines/dreamzero/README.md.
_ENV_PREFIX="${CONDA_PREFIX:-${VIRTUAL_ENV:-}}"
_CUDNN_LIB="${_ENV_PREFIX}/lib/python3.11/site-packages/nvidia/cudnn/lib"
if [ -d "$_CUDNN_LIB" ]; then
    export LD_LIBRARY_PATH="${_CUDNN_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export HYDRA_FULL_ERROR=1
export WANDB_ENTITY=${WANDB_ENTITY:-ggsonglin}
ulimit -n 65535 2>/dev/null || true

# ============ USER CONFIGURATION ============
SIMPLE_DATA_ROOT=${SIMPLE_DATA_ROOT:-"$PSI_HOME/data/simple/dreamzero"}
PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-"$PSI_HOME/cache/checkpoints/DreamZero-AgiBot"}

TASK_NAME=${1:-${TASK_NAME:-$(date +"%Y%m%d%H%M")}}
OUTPUT_DIR=${OUTPUT_DIR:-".runs/dreamzero/$TASK_NAME-lora"}

echo "Output Directory: $OUTPUT_DIR"

MAX_STEPS=${MAX_STEPS:-40000}
SAVE_STEPS=${SAVE_STEPS:-500}

# Number of GPUs per node: prefer the count of CUDA_VISIBLE_DEVICES (set by
# slurm_job.sh), fall back to nvidia-smi, then to 8.
if [ -z "${NUM_GPUS}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
        NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
    else
        NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    fi
fi
NUM_GPUS=${NUM_GPUS:-8}

# ============ DISTRIBUTED RENDEZVOUS ============
# Provided by slurm_job.sh under sbatch; defaults give a single-node run.
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
echo "Distributed: nnodes=$NNODES node_rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT nproc_per_node=$NUM_GPUS"

WAN_CKPT_DIR=${WAN_CKPT_DIR:-"$PSI_HOME/cache/checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"$PSI_HOME/cache/checkpoints/umt5-xxl"}

# Precomputed prompt embedding cache (skips online text encoding at train time).
# Set PROMPT_CACHE_PATH="" to disable.
PROMPT_CACHE_PATH=${PROMPT_CACHE_PATH-"/hfm/cache/dreamzero/prompt_cache_g1_simple.pt"}

# ============ CHUNK / FRAME CONFIG ============
if [ -n "$MAX_CHUNK_SIZE" ] && [ -n "$NUM_FRAMES" ]; then
    :  # user override
elif [ "$TASK_NAME" = "mixture" ]; then
    MAX_CHUNK_SIZE=2; NUM_FRAMES=17
else
    case "$TASK_NAME" in
        G1WholebodyTabletopGraspMP-v0)
            MAX_CHUNK_SIZE=2; NUM_FRAMES=17 ;;
        G1WholebodyBendPickMP-v0|G1WholebodyXMovePickTeleop-v0|G1WholebodyXMoveBendPickTeleop-v0)
            MAX_CHUNK_SIZE=3; NUM_FRAMES=25 ;;
        G1WholebodyHandoverTeleop-v0|G1WholebodyLocomotionPickBetweenTablesTeleop-v0)
            MAX_CHUNK_SIZE=4; NUM_FRAMES=33 ;;
        *)
            MAX_CHUNK_SIZE=2; NUM_FRAMES=17 ;;
    esac
fi
echo "Task: $TASK_NAME → max_chunk_size=$MAX_CHUNK_SIZE, num_frames=$NUM_FRAMES"


torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$NUM_GPUS" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    src/dreamzero/groot/vla/experiment/experiment.py \
    report_to=${REPORT_TO:-wandb} \
    data=dreamzero/simple \
    wandb_project=psi \
    wandb_group=dreamzero \
    train_architecture=lora \
    num_frames=$NUM_FRAMES \
    action_horizon=48 \
    num_views=1 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=48 \
    simple_pad_freeze_action=true \
    num_state_per_block=1 \
    max_action_dim=36 \
    max_state_dim=44 \
    seed=42 \
    training_args.learning_rate=1e-4 \
    training_args.deepspeed="src/dreamzero/groot/vla/configs/deepspeed/zero2.json" \
    save_steps=$SAVE_STEPS \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
    max_steps=$MAX_STEPS \
    logging_steps=10 \
    weight_decay=1e-5 \
    save_total_limit=5 \
    keep_checkpoint_every=${KEEP_CHECKPOINT_EVERY:-1000} \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=true \
    dataloader_num_workers=${DATALOADER_NUM_WORKERS:-4} \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=$MAX_CHUNK_SIZE \
    frame_seqlen=220 \
    save_strategy=steps \
    relative_action=false \
    simple_data_root=$SIMPLE_DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=$PRETRAINED_MODEL_PATH \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true \
    "${MIXTURE_OVERRIDE[@]}" \
    "${@:2}" &

TORCHRUN_PID=$!
wait "$TORCHRUN_PID"
