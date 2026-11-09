#!/usr/bin/env bash
#SBATCH --job-name=psix
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=120
#SBATCH --time=23:30:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=.logs/slurm-train-%j.out
#SBATCH --error=.logs/slurm-train-%j.err

PROJECT_ROOT=$(pwd)
TRAIN_CMD="$*"

mkdir -p "$PROJECT_ROOT/.logs"

# Keep JIT/compiler scratch and cache off BeeGFS. Each node sees its own /tmp.
export H100_JOB_CACHE_ROOT="/tmp/h100-job-cache/${USER:-songlinwei}/${SLURM_JOB_ID}"
export TRITON_HOME="$H100_JOB_CACHE_ROOT/triton-home"
export TRITON_CACHE_DIR="$H100_JOB_CACHE_ROOT/triton-cache"
export TRITON_OVERRIDE_DIR="$H100_JOB_CACHE_ROOT/triton-override"
export TRITON_DUMP_DIR="$H100_JOB_CACHE_ROOT/triton-dump"
export TORCHINDUCTOR_CACHE_DIR="$H100_JOB_CACHE_ROOT/torchinductor"
export CUDA_CACHE_PATH="$H100_JOB_CACHE_ROOT/nv"
export TORCH_EXTENSIONS_DIR="$H100_JOB_CACHE_ROOT/torch_extensions"
export XDG_CACHE_HOME="$H100_JOB_CACHE_ROOT/xdg-cache"

# keep ONE persistent shared cache on beegfs scratch (per .env), pre-seeded.
export HF_DATASETS_CACHE="/var/lib/h100-job-cache/${USER:-songlinwei}/hf-datasets"
mkdir -p "$HF_DATASETS_CACHE" 2>/dev/null || true

# Node-local staged dataset.
# The stage under /var/lib/h100-job-cache/data is SHARED by every dataset and user on
# the node, so probing one hardcoded pack is wrong: a node holding somebody else's
# pack still flipped PSI_DATA_ROOT, and a job whose own pack was not staged died with
#   FATAL: missing pack /var/lib/h100-job-cache/data/<pack>        (job 1868)
# Use the stage only when EVERY pack this job needs is actually on it; otherwise fall
# back to beegfs, which is slower but always correct.
#
# The pack ids come from $PSI_STAGE_IDS when set, else from the TRAIN_ID / VAL_ID
# lines of the recipe being launched. Set PSI_STAGE_IDS explicitly if a recipe takes
# its ids from the environment instead of literals.
PSI_LOCAL_DATA=/var/lib/h100-job-cache/data
_recipe=$(printf '%s\n' $TRAIN_CMD | head -1)
if [ -z "${PSI_STAGE_IDS:-}" ] && [ -f "$_recipe" ]; then
    PSI_STAGE_IDS=$(grep -E '^(TRAIN_ID|VAL_ID)=' "$_recipe" \
        | sed -e 's/^[A-Za-z_]*=//' -e 's/"//g' -e 's/^\${[A-Za-z_]*:-//' -e 's/}$//' \
        | sort -u | tr '\n' ' ')
fi
_staged=1; _miss=
if [ -z "${PSI_STAGE_IDS:-}" ]; then
    _staged=0; _miss=" (could not determine packs from $_recipe)"
else
    for _id in $PSI_STAGE_IDS; do
        [ -d "$PSI_LOCAL_DATA/$_id/data" ] || { _staged=0; _miss="$_miss $_id"; }
    done
fi
if [ "$_staged" = 1 ]; then
    export PSI_DATA_ROOT="$PSI_LOCAL_DATA"
    echo "Data root:   $PSI_DATA_ROOT (node-local stage: $PSI_STAGE_IDS)"
else
    echo "Data root:   .data (BeeGFS - not staged:$_miss; expect metadata pressure)"
    echo "             stage it with: sudo bash scripts/data/stage_pack_local.sh $PSI_STAGE_IDS"
fi

# Node-local venv stage. The psi venv is baked into psix.sqsh at /workspace/.venv-psi and
# needs nothing; .venv-dreamzero lives on beegfs. transformers opens every file under
# models/ at import, and 32 ranks doing that at once makes the beegfs client return a
# transient ENOENT on an existing .py file: jobs 5470 (x2) and 5489 each lost a different
# node on a different file that way. Copy the venv to node-local disk once per node
# (incremental rsync after the first time) and mount it at /workspace/<venv>, which the
# dreamzero recipes prefer when present. PSI_STAGE_VENV=<name> overrides the recipe
# auto-detect; PSI_STAGE_VENV=0 disables it. Falls back to beegfs if any node fails.
PSI_LOCAL_VENV_ROOT="/var/lib/h100-job-cache/${USER:-songlinwei}/venv"
if [ -z "${PSI_STAGE_VENV+x}" ] && [ -f "$_recipe" ]; then
    PSI_STAGE_VENV=$(grep -oE '\.venv-[A-Za-z0-9_-]+' "$_recipe" | grep -vx '.venv-psi' | sort -u | head -1)
fi
_venv_mount=
if [ -n "${PSI_STAGE_VENV:-}" ] && [ "$PSI_STAGE_VENV" != 0 ] && [ -d "$PROJECT_ROOT/$PSI_STAGE_VENV" ]; then
    echo "Venv stage:  $PSI_STAGE_VENV -> $PSI_LOCAL_VENV_ROOT on $SLURM_NNODES node(s), started $(date '+%H:%M:%S')"
    if srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
          bash "$PROJECT_ROOT/scripts/train/stage_venv_local.sh" \
          "$PROJECT_ROOT/$PSI_STAGE_VENV" "$PSI_LOCAL_VENV_ROOT/$PSI_STAGE_VENV" "/workspace/$PSI_STAGE_VENV"; then
        _venv_mount=",$PSI_LOCAL_VENV_ROOT/$PSI_STAGE_VENV:/workspace/$PSI_STAGE_VENV"
        echo "Venv stage:  done $(date '+%H:%M:%S'); mounted at /workspace/$PSI_STAGE_VENV"
    else
        echo "Venv stage:  FAILED on at least one node; using the beegfs venv (expect import flakiness)"
    fi
else
    echo "Venv stage:  none (PSI_STAGE_VENV='${PSI_STAGE_VENV:-}')"
fi

# Enroot runtime/data/cache paths are managed by /etc/enroot/enroot.conf.d/10-h100-defaults.conf.
# Keep them node-local to avoid BeeGFS metadata pressure during Pyxis startup.
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(getent ahostsv4 "$MASTER_NODE" | awk '{print $1; exit}')
MASTER_ADDR=${MASTER_ADDR:-$MASTER_NODE}
MASTER_PORT=$((29500 + SLURM_JOB_ID % 10000))
NNODES=$SLURM_NNODES

echo "Job ID:      $SLURM_JOB_ID"
echo "Nodes:       $SLURM_JOB_NODELIST"
echo "Master:      $MASTER_ADDR:$MASTER_PORT"
echo "Num nodes:   $NNODES"
echo "Train cmd:   $TRAIN_CMD"


# NCCL config. NCCL_IB_HCA is set per node by scripts/train/nccl_rails.sh in the
# container step below -- mlx5_* numbering differs across nodes, and an export
# here would apply the first node's list to all of them.
# NCCL_SOCKET_IFNAME unset: bootstrap only; the default matches MASTER_ADDR.
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_RAS_ENABLE=0
# Not on by default; without it a NCCL hang runs out the walltime instead of
# failing into the --requeue path.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONFAULTHANDLER=1
# take everything offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# export WANDB_MODE=offline

# --- Auto-requeue on walltime -----------------------------------------------
# `#SBATCH --signal=B:USR1@120` makes Slurm send SIGUSR1 to THIS batch shell (the
# `B:` prefix targets the batch step, not srun/training) 120s before the walltime
# kill. Trap it and requeue the SAME job (`scontrol requeue $SLURM_JOB_ID`) so it
# restarts with the identical submission — same sbatch flags and same "$@" (e.g.
# `--resume <ts>`) — and auto-resumes from the latest checkpoint. `#SBATCH --requeue`
# permits it; `#SBATCH --open-mode=append` keeps the pre-requeue log.
_REQUEUED=0
_requeue_on_walltime() {
    echo ">>> $(date '+%H:%M:%S') SIGUSR1: nearing walltime -> scontrol requeue ${SLURM_JOB_ID}"
    if scontrol requeue "${SLURM_JOB_ID}"; then
        _REQUEUED=1
    else
        echo ">>> $(date '+%H:%M:%S') WARNING: scontrol requeue failed; job will end at walltime without restart." >&2
    fi
}
trap _requeue_on_walltime USR1

srun_exit=0
srun \
  --ntasks=$SLURM_NNODES \
  --ntasks-per-node=1 \
  --kill-on-bad-exit=1 \
  --container-image=/mnt/beegfs/containers/psix.sqsh \
  --container-mounts=/mnt/beegfs:/mnt/beegfs,"$PROJECT_ROOT":/workspace/psi0,/var/lib/h100-job-cache:/var/lib/h100-job-cache${_venv_mount} \
  --container-workdir=/workspace/psi0 \
  bash -lc "
    export TMPDIR=\"$H100_JOB_CACHE_ROOT/tmp\"
    export TMP=\"\$TMPDIR\"
    export TEMP=\"\$TMPDIR\"
    export TORCH_EXTENSIONS_DIR=\"$H100_JOB_CACHE_ROOT/torch_extensions\"
    export XDG_CACHE_HOME=\"$H100_JOB_CACHE_ROOT/xdg-cache\"
    mkdir -p \"\$TMPDIR\" \"$TRITON_HOME\" \"$TRITON_CACHE_DIR\" \"$TRITON_OVERRIDE_DIR\" \"$TRITON_DUMP_DIR\" \"$TORCHINDUCTOR_CACHE_DIR\" \"$CUDA_CACHE_PATH\" \"$TORCH_EXTENSIONS_DIR\" \"$XDG_CACHE_HOME\"
    export OMP_NUM_THREADS=16
    export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
    # per-node rail selection (see the NCCL note above)
    . /workspace/psi0/scripts/train/nccl_rails.sh
    export LIBRARY_PATH=/usr/local/cuda/lib64/stubs
    export NNODES=$NNODES
    export NODE_RANK=\$SLURM_PROCID
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    bash $TRAIN_CMD
" &
srun_pid=$!

# Run srun in the BACKGROUND and `wait` on it (not foreground): bash defers a trap
# until the current *foreground* command returns, so a foreground srun would only fire
# _requeue_on_walltime at the hard walltime kill — too late to requeue. `wait` is
# signal-interruptible, so SIGUSR1 runs the trap at once while the job is still RUNNING.
wait "$srun_pid"
srun_exit=$?

if [[ "$_REQUEUED" == "1" ]]; then
    # Requeue is scheduled; block until Slurm tears the step down (so we don't exit into a
    # completing/requeued race), then exit 0 -> job returns to the queue and restarts,
    # auto-resuming from the latest checkpoint via the unchanged "$@".
    wait "$srun_pid" 2>/dev/null || true
    echo ">>> $(date '+%H:%M:%S') walltime reached: job requeued (JobID $SLURM_JOB_ID); will auto-resume from checkpoint on restart."
    exit 0
fi

if [[ $srun_exit -ne 0 ]]; then
    echo "srun exited with code $srun_exit"
    exit $srun_exit
fi
