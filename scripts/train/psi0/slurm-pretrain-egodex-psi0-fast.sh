#!/usr/bin/env bash
#SBATCH --job-name=pretrain-egodex-psi0-fast
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=/mnt/beegfs/scratch/songlinwei/psi0/.logs/slurm-pretrain-%j.out
#SBATCH --error=/mnt/beegfs/scratch/songlinwei/psi0/.logs/slurm-pretrain-%j.err
#SBATCH --chdir=/mnt/beegfs/scratch/songlinwei/psi0

# Drop -e so a non-zero srun exit (due to SIGKILL at walltime) doesn't prevent resubmission
set -uo pipefail

mkdir -p /mnt/beegfs/scratch/songlinwei/psi0/.logs

# Auto-resubmit on walltime: SLURM sends SIGUSR1 120s before the limit
RESUBMITTED=0
resubmit_on_timeout() {
    echo "SIGUSR1 received: approaching walltime, resubmitting job..."
    sbatch "$(realpath "${BASH_SOURCE[0]}")"
    RESUBMITTED=1
}
trap resubmit_on_timeout USR1

# Keep JIT/compiler scratch and cache off BeeGFS. Each node sees its own /tmp.
export H100_JOB_CACHE_ROOT="/tmp/h100-job-cache/${USER:-songlinwei}/${SLURM_JOB_ID}"
export TRITON_HOME="$H100_JOB_CACHE_ROOT/triton-home"
export TRITON_CACHE_DIR="$H100_JOB_CACHE_ROOT/triton-cache"
export TRITON_OVERRIDE_DIR="$H100_JOB_CACHE_ROOT/triton-override"
export TRITON_DUMP_DIR="$H100_JOB_CACHE_ROOT/triton-dump"
export TORCHINDUCTOR_CACHE_DIR="$H100_JOB_CACHE_ROOT/torchinductor"
export CUDA_CACHE_PATH="$H100_JOB_CACHE_ROOT/nv"

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

# NCCL config — aligned with tested 48-GPU all-reduce reference
# bond0.1417 is the VLAN-tagged IB interface; bond0 alone misses it
export NCCL_SOCKET_IFNAME=ibp24s0,ibp25s0,ibp66s0,ibs5,ibs7,ibs8,ibs10,ibs11
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9,mlx5_10
export NCCL_DEBUG=INFO
export NCCL_RAS_ENABLE=0
export NCCL_NVLS_ENABLE=0
export NCCL_MNNVL_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

srun_exit=0
srun \
  --ntasks=$SLURM_NNODES \
  --ntasks-per-node=1 \
  --container-image=/mnt/beegfs/containers/nvidia-pytorch-25.06-py3-cuda12.9.sqsh \
  --container-mounts=/mnt/beegfs:/mnt/beegfs \
  --container-workdir=/mnt/beegfs/scratch/songlinwei/psi0 \
  bash -lc "
    export TMPDIR=\"$H100_JOB_CACHE_ROOT/tmp\"
    export TMP=\"\$TMPDIR\"
    export TEMP=\"\$TMPDIR\"
    mkdir -p \"\$TMPDIR\" \"$TRITON_HOME\" \"$TRITON_CACHE_DIR\" \"$TRITON_OVERRIDE_DIR\" \"$TRITON_DUMP_DIR\" \"$TORCHINDUCTOR_CACHE_DIR\" \"$CUDA_CACHE_PATH\" \"$H100_JOB_CACHE_ROOT/wandb-cache\"
    export WANDB_CACHE_DIR=\"$H100_JOB_CACHE_ROOT/wandb-cache\"
    export DEEPSPEED_CONFIG=\"$H100_JOB_CACHE_ROOT/zero3.json\"
    NPROC_PER_NODE=8
    python3 - <<PYDS
import json
from pathlib import Path
cfg = json.loads(Path(\"scripts/deepspeed/zero3.json\").read_text())
cfg[\"train_micro_batch_size_per_gpu\"] = 16
cfg[\"gradient_accumulation_steps\"] = 2
cfg[\"train_batch_size\"] = 16 * 2 * int(\"$NNODES\") * int(\"\$NPROC_PER_NODE\")
Path(\"\$DEEPSPEED_CONFIG\").write_text(json.dumps(cfg, indent=2) + \"\\n\")
PYDS
    export WANDB_DIR=\"/mnt/beegfs/scratch/songlinwei/psi0/.logs\"
    .venv-psi/bin/python -c \"import deepspeed; import deepspeed.ops.transformer.inference.triton.matmul_ext; print(deepspeed.__version__)\" || echo \"warning: deepspeed/triton prewarm failed; continuing\"
    bash scripts/train/psi0/pretrain-egodex-psi0-fast.sh \
        $MASTER_ADDR \
        $MASTER_PORT \
        \$SLURM_PROCID \
        $NNODES
" || srun_exit=$?

if [[ $RESUBMITTED -eq 1 ]]; then
    echo "Job timed out and has been resubmitted. Exiting."
    exit 0
fi

if [[ $srun_exit -ne 0 ]]; then
    echo "srun exited with code $srun_exit"
    exit $srun_exit
fi
