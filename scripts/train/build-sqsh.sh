#!/usr/bin/env bash
# Build the psix training image *on the cluster*:
#   plain CUDA base sqsh -> enroot container -> uv sync + flash-attn -> exported sqsh.
#
# Python 3.11, torch 2.7 cu12, the `psi` group, built on
# nvidia/cuda:*-cudnn-devel-ubuntu24.04 rather than the 23G NGC PyTorch image.
# NGC's bundled torch/python were dead weight -- .venv-psi shadows them completely --
# so dropping them is what shrinks the result. Modeled on cosmos3's
# scripts/build-cosmos-sqsh.sh, which already solved the registry and quota problems.
#
# CUDA 12.9, NOT cosmos's 13.0.2: pyproject pins torch==2.7.0 and the flash-attn wheel
# below is cu12torch2.7cxx11abiTRUE-cp311. A CUDA 13 base would leave the toolkit and
# the wheels a major version apart.
#
#   bash scripts/train/build-sqsh.sh
#
# Overridable:
#   PSIX_BASE_SQSH    base sqsh        (default: <containers>/nvidia-cuda--12.9.1-....sqsh)
#   PSIX_BASE_IMAGE   base image ref   (default: nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04)
#   PSIX_BASE_REGISTRIES  tried in order (default: "nvcr.io docker.io")
#   PSIX_SQSH_OUT     output path      (default: /mnt/beegfs/scratch/$USER/containers/psix.sqsh)
#   PSIX_MIRROR       lerobot mirror   (default: /mnt/beegfs/scratch/$USER/mirrors/lerobot.git)
#   PSIX_DRY_RUN=1    print resolved paths and exit without building
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)

NAME=${PSIX_CONTAINER_NAME:-psix-build}
OUT=${PSIX_SQSH_OUT:-/mnt/beegfs/scratch/$USER/containers/psix.sqsh}
MIRROR=${PSIX_MIRROR:-/mnt/beegfs/scratch/$USER/mirrors/lerobot.git}
LEROBOT_URL=https://github.com/songlin/lerobot.git

BASE_IMAGE=${PSIX_BASE_IMAGE:-nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04}
# Docker Hub (registry-1.docker.io) just times out from these compute nodes. nvcr.io
# serves the same nvidia/cuda images and allows anonymous pulls, so try it first and
# keep Docker Hub as a fallback for anyone on a less restricted network.
read -r -a BASE_REGISTRIES <<< "${PSIX_BASE_REGISTRIES:-nvcr.io docker.io}"
BASE=${PSIX_BASE_SQSH:-/mnt/beegfs/scratch/$USER/containers/$(echo "$BASE_IMAGE" | tr '/:' '--').sqsh}

CACHE=/tmp/uv-cache-$USER
# Pre-downloaded flash-attn wheel(s). GitHub release CDN times out from the compute
# nodes, so stage the wheel once from somewhere with working egress.
FLASH_WHEELS=${PSIX_FLASH_WHEELS:-/mnt/beegfs/scratch/$USER/flash-wheels}
MOUNTS=(-m "/mnt/beegfs:/mnt/beegfs:none:x-create=dir,bind,rw"
        -m "$CACHE:/uv-cache:none:x-create=dir,bind,rw"
        -m "$REPO:/workspace/psi0:none:x-create=dir,bind,rw")
[ -d "$FLASH_WHEELS" ] && MOUNTS+=(-m "$FLASH_WHEELS:/flash-wheels:none:x-create=dir,bind,ro")

LOCAL=/tmp/enroot-$USER
export ENROOT_DATA_PATH=$LOCAL/data
export ENROOT_TEMP_PATH=$LOCAL/tmp
export ENROOT_RUNTIME_PATH=$LOCAL/runtime
# Layer cache on BeeGFS: a handful of big write-once tarballs (no small-file penalty),
# and it keeps ~6G off /tmp, which is under a per-user quota on some nodes.
export ENROOT_CACHE_PATH=${PSIX_ENROOT_CACHE:-/mnt/beegfs/scratch/$USER/enroot-cache}
export ENROOT_SQUASH_OPTIONS=${ENROOT_SQUASH_OPTIONS:-'-comp zstd -Xcompression-level 10'}

echo "[build] repo:    $REPO -> /workspace/psi0"
echo "[build] baseimg: $BASE_IMAGE via ${BASE_REGISTRIES[*]}"
echo "[build] base:    $BASE"
echo "[build] out:     $OUT"
echo "[build] mirror:  $MIRROR"
echo "[build] cache:   $CACHE -> /uv-cache"
echo "[build] flash:   $FLASH_WHEELS"
[ "${PSIX_DRY_RUN:-0}" = "1" ] && { echo "[build] dry run, exiting"; exit 0; }

test -f "$REPO/pyproject.toml" || { echo "[build] FATAL: no psi0 checkout at $REPO"; exit 1; }
mkdir -p "$CACHE" "$ENROOT_DATA_PATH" "$ENROOT_TEMP_PATH" "$ENROOT_RUNTIME_PATH" \
         "$ENROOT_CACHE_PATH" "$(dirname "$OUT")" "$(dirname "$MIRROR")" "$(dirname "$BASE")"

# Drop a previous build container BEFORE the space check: a leftover rootfs from an
# aborted run is ~15G, enough to fail the check against a corpse we were about to
# delete anyway.
enroot remove -f "$NAME" 2>/dev/null || true

# Check the QUOTA, not df. /tmp is under a 50G per-user quota on some nodes, so df can
# report plenty free while the quota is exhausted -- the base extract then dies with a
# thoroughly misleading "parallel: Is the disk full?".
NEED=35
quota_headroom_gb() {
    local dev used limit
    dev=$(df --output=source /tmp 2>/dev/null | tail -1) || return 1
    [ -n "$dev" ] || return 1
    read -r used limit < <(quota 2>/dev/null | awk -v d="$dev" '$1==d {gsub(/\*/,"",$2); print $2, $4; exit}')
    [ -n "${limit:-}" ] && [ "${limit:-0}" -gt 0 ] || return 1
    echo $(( (limit - used) / 1024 / 1024 ))
}
if avail=$(quota_headroom_gb); then
    echo "[build] /tmp quota headroom: ${avail}G (need ~${NEED}G)"
    hint="free quota ('enroot list' then 'enroot remove -f <stale>'; /var/lib/h100-enroot counts too)"
else
    avail=$(df -BG --output=avail /tmp | tail -1 | tr -dc '0-9')
    echo "[build] /tmp avail: ${avail}G (no quota set; need ~${NEED}G)"
    hint="free node-local space"
fi
[ "$avail" -lt "$NEED" ] && { echo "[build] FATAL: not enough node-local space -- $hint"; exit 1; }

# GitHub over git resets intermittently from this region and uv does not retry the
# `git fetch` it runs for lerobot. Pre-fetch once here, where retrying is cheap.
if [ ! -d "$MIRROR" ]; then
  echo "[build] $(date +%T) cloning lerobot mirror (one-off, ~110MB)"
  for i in 1 2 3 4 5; do
    git clone --mirror "$LEROBOT_URL" "$MIRROR" && break
    echo "[build] mirror clone attempt $i failed, retrying"; rm -rf "$MIRROR"; sleep 10
  done
  git -C "$MIRROR" config uploadpack.allowAnySHA1InWant true
  git -C "$MIRROR" config uploadpack.allowReachableSHA1InWant true
fi
test -d "$MIRROR" || { echo "[build] FATAL: lerobot mirror missing"; exit 1; }

# One-off base import, cached as a sqsh so later rebuilds skip the pull.
if [ ! -f "$BASE" ]; then
  for reg in "${BASE_REGISTRIES[@]}"; do
    case "$reg" in
      docker.io|"") uri="docker://$BASE_IMAGE" ;;
      *)            uri="docker://$reg#$BASE_IMAGE" ;;
    esac
    echo "[build] $(date +%T) importing base $uri (one-off, several GB)"
    for i in 1 2 3; do
      enroot import -o "$BASE" "$uri" && break 2
      echo "[build] base import attempt $i from $reg failed"; rm -f "$BASE"; sleep 10
    done
    echo "[build] giving up on $reg"
  done
fi
test -f "$BASE" || { echo "[build] FATAL: could not obtain base image $BASE_IMAGE"; exit 1; }

# Nothing in this build touches a GPU, and enroot's 98-nvidia.sh hook hard-fails on
# hosts with a broken libnvidia-container ("nvml error: unknown error").
NO_GPU="-e NVIDIA_VISIBLE_DEVICES=void"

echo "[build] $(date +%T) creating container from $BASE"
enroot create --name "$NAME" "$BASE"

echo "[build] $(date +%T) preflight: repo visible at /workspace/psi0"
enroot start --root --rw "${MOUNTS[@]}" $NO_GPU "$NAME" \
  bash -c 'test -f /workspace/psi0/pyproject.toml && echo "[build] /workspace/psi0 mount OK"'

echo "[build] $(date +%T) running build inside container"
enroot start --root --rw "${MOUNTS[@]}" $NO_GPU -e PSI_MIRROR="$MIRROR" "$NAME" \
  bash /workspace/psi0/scripts/train/inner.sh

echo "[build] $(date +%T) exporting to $OUT"
rm -f "$OUT"
enroot export --output "$OUT" "$NAME"
ls -lh "$OUT"
echo "[build] $(date +%T) DONE"
echo "[build] install with: sudo cp $OUT /mnt/beegfs/containers/psix.sqsh"
