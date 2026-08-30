# Build from the repo root:
#   docker build -f scripts/train/Dockerfile -t psi:train .
#
# Python 3.11, torch 2.7 cu12, the `psi` group, on a PLAIN CUDA base rather than
# nvcr.io/nvidia/pytorch. NGC's bundled python+torch are completely shadowed by
# /workspace/.venv-psi, so they were ~15G of dead weight.
# Keep this in step with scripts/train/inner.sh.
#
# CUDA 12.9, not the cosmos Dockerfile's 13.0.2: pyproject pins torch==2.7.0 and the
# flash-attn wheel below is cu12torch2.7cxx11abiTRUE-cp311.
ARG CUDA_VERSION=12.9.1
ARG BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# The CUDA base ships no python, no git and no compiler. build-essential is required
# because a few locked deps (accelerate, arm_pytorch_utilities) are sdist-only.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        git-lfs \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace

# .venv-psi is the environment scripts/train/**/*.sh activate, so keep the name and
# location the training scripts expect. Setting VIRTUAL_ENV + PATH is the Docker
# equivalent of `source .venv-psi/bin/activate`: each RUN is its own shell.
ENV VIRTUAL_ENV=/workspace/.venv-psi \
    PATH="/workspace/.venv-psi/bin:/root/.local/bin:${PATH}" \
    UV_NO_CACHE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    GIT_LFS_SKIP_SMUDGE=1

# No system python here -- fetch a managed CPython 3.11 (pyproject pins ==3.11.*).
RUN uv python install 3.11 && uv venv "${VIRTUAL_ENV}" --python 3.11

# Dependency layer: resolve/install the groups without the project itself, so it stays
# cached when only source under src/ changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync \
        --group serve \
        --group viz \
        --group psi \
        --index-strategy unsafe-best-match \
        --active \
        --no-install-project

# flash-attn: PyPI carries only an sdist (a ~1h source build), but upstream publishes
# prebuilt wheels on GitHub Releases. This one matches the layer above -- cu12 /
# torch 2.7 / cxx11abi TRUE / cp311 -- and contains sm80 + sm90 SASS, so it runs on
# H100. Re-pin if the torch or python pin moves; the import check below catches an ABI
# mismatch.
ARG FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
RUN uv pip install --python "${VIRTUAL_ENV}/bin/python" "${FLASH_ATTN_WHEEL}"

# Project sources, then the editable `psi` install itself. --inexact because flash-attn
# is absent from uv.lock: a plain `uv sync` would treat it as extraneous and uninstall
# the wheel the layer above just placed.
COPY . .
RUN uv sync \
        --group serve \
        --group viz \
        --group psi \
        --index-strategy unsafe-best-match \
        --active \
        --inexact

# Triton's bundled ptxas is older than the base CUDA toolkit.
ENV TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"

RUN python -c "import torch, flash_attn, psi; print(torch.__version__, flash_attn.__version__, torch.version.cuda)"
