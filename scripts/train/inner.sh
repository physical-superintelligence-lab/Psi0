#!/usr/bin/env bash
# Runs INSIDE the enroot container as root, invoked by build-sqsh.sh.
# Keep in step with scripts/train/Dockerfile -- same pins, same steps.
#
# This starts from a PLAIN CUDA base (nvidia/cuda:*-cudnn-devel-ubuntu24.04) rather
# than the 23G NGC PyTorch image. That base ships no python and no torch,
# which is the entire point: the psi group installs torch 2.7 + the nvidia-* wheel set
# itself, so NGC's bundled stack was dead weight shadowed by .venv-psi anyway.
#
# `--root` here means UID 0 inside the container's user namespace only; no host
# privileges are involved.
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive

REPO=${PSI_REPO:-/workspace/psi0}
VENV=/workspace/.venv-psi
MIRROR=${PSI_MIRROR:-}
LEROBOT_URL=https://github.com/songlin/lerobot.git
# GitHub's release CDN is unreachable from the compute nodes 
FLASH_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
FLASH_LOCAL=$(ls /flash-wheels/flash_attn-*cu12torch2.7*cp311*.whl 2>/dev/null | head -1 || true)

retry() {
  local n=1 max=6
  until "$@"; do
    if [ $n -ge $max ]; then echo "[build] FATAL: '$*' failed after $max attempts"; return 1; fi
    echo "[build] attempt $n/$max failed, retrying in $((n*10))s: $*"
    sleep $((n*10)); n=$((n+1))
  done
}

# The CUDA base is minimal: no python, no git, no ffmpeg, no compiler. build-essential
# is needed because a few locked deps ship only sdists (accelerate,
# arm_pytorch_utilities) and uv builds them here.
retry apt-get update
retry apt-get install -y --no-install-recommends \
    ca-certificates curl git git-lfs ffmpeg build-essential
rm -rf /var/lib/apt/lists/*

# uv shells out to /usr/bin/git for the lerobot dependency; point that one URL at the
# mirror build-sqsh.sh pre-fetched, so the build never hits GitHub over git.
# uv.lock is untouched -- the pinned SHA still resolves, just locally.
if [ -n "$MIRROR" ] && [ -d "$MIRROR" ]; then
  git config --global url."$MIRROR".insteadOf "$LEROBOT_URL"
  echo "[build] using local lerobot mirror: $MIRROR"
else
  echo "[build] WARNING: no lerobot mirror, falling back to GitHub (flaky here)"
fi
git config --global http.lowSpeedLimit 1000
git config --global http.lowSpeedTime 60

retry curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh
sh /tmp/uv-install.sh
export PATH=/root/.local/bin:$PATH
export VIRTUAL_ENV=$VENV
export GIT_LFS_SKIP_SMUDGE=1
export UV_CACHE_DIR=${UV_CACHE_DIR:-/uv-cache}

# No system python on this base -- uv fetches a managed CPython 3.11 (pyproject pins
# ==3.11.*). Install it into the image, not the bind-mounted cache, or the interpreter
# vanishes when the mount goes away at runtime.
export UV_PYTHON_INSTALL_DIR=/opt/uv-python
retry uv python install 3.11
uv venv "$VENV" --python 3.11
export PATH="$VENV/bin:/root/.local/bin:$PATH"

cd "$REPO"
retry uv sync --group serve --group viz --group psi --index-strategy unsafe-best-match --active
if [ -n "$FLASH_LOCAL" ]; then
  echo "[build] using staged flash-attn wheel: $FLASH_LOCAL"
  uv pip install --python "$VENV/bin/python" "$FLASH_LOCAL"
else
  echo "[build] WARNING: no staged wheel at /flash-wheels, fetching from GitHub (hangs here)"
  retry uv pip install --python "$VENV/bin/python" "$FLASH_WHEEL"
fi

python -c "import torch, flash_attn, psi, lerobot; print('VERIFY', torch.__version__, flash_attn.__version__, psi.__file__)"
python -c "import torch; print('VERIFY cuda build:', torch.version.cuda)"

# Bake the env into the image: enroot exports /etc/environment as the image ENV.
sed -i 's|^PATH=|PATH=/workspace/.venv-psi/bin:/root/.local/bin:|' /etc/environment
if ! grep -q '^VIRTUAL_ENV=' /etc/environment; then
  {
    echo "VIRTUAL_ENV=/workspace/.venv-psi"
    echo "GIT_LFS_SKIP_SMUDGE=1"
    # Triton's bundled ptxas is older than the base CUDA toolkit; point at the real one
    # (same reason as the cosmos Dockerfile).
    echo "TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas"
  } >> /etc/environment
fi
grep -E '^(PATH|VIRTUAL_ENV|TRITON_PTXAS_PATH)=' /etc/environment
echo "[build] inner done"
