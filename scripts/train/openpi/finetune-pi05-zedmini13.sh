#!/bin/bash
# pi0.5 (openpi) baseline arm on the THIRTEEN zedmini packs @ 294x168.
#
# Companion to scripts/train/psi0/finetune-sonic-neck-zedmini.sh: same packs,
# same 80-D action / 45-D state semantics, same 30-step chunk @ 30 Hz, same
# schedule (80k steps, global batch 128, peak lr 5e-5, warmup 1000, cosine, seed
# 292285, checkpoint every 10k). Everything that differs is a pi0.5 property:
#
#   canvas      168x294 (H W) -- the camera's 7:4 frame scaled down proportionally
#               onto SigLIP's patch-14 grid: 12x21 = 252 tokens/image, 756 across
#               the three slots, within 2% of the 768 the earlier openpi-05 runs
#               spent at 224x224. Token count, not pixel count, is what costs:
#               pixels reach only the patch-embedding conv (0.15% of the tower).
#               224x224 would spend 43% of its tokens on the black bars a 7:4 frame
#               leaves behind; this crops and letterboxes nothing.
#   views       pi0.5 carries three image slots; the packs ship one egocentric
#               camera, so the two wrist slots are zero-filled and masked off --
#               the convention HfmInputs already established for G1 data, and what
#               the psi0-era pi05 benchmark ran with.
#   norm        pi0.5's own quantile (q01/q99) normalisation, not psi0's min/max
#               bounds.
#   frozen      the PaliGemma LANGUAGE tower, as in the stock openpi-05 recipe.
#               The vision tower and projector stay trainable: SigLIP is fed an
#               18x32 patch grid interpolated off its pretrained 16x16, and frozen
#               it could never adapt to that canvas. Stock openpi-05 freezes the
#               whole VLM -- one field, `pytorch_freeze_patterns` on the config.
#               Note psi0/psix both tune their VLM outright (`--model.tune-vlm`).
#   action ord  hand(14) ++ body_token(64) ++ neck(2)  -- the psix ordering.
#               NOTE the psi0 arm concatenates body_token ++ hand ++ neck, so
#               eval/deploy must slice per arm.
#
# Config lives in src/openpi/training/config.py as `pi05_zedmini13`; the pack
# roster, action width and token budget are constants at the top of that file.
#
# Requires .venv-openpi (see baselines/pi05/README.md; build with python 3.11,
# not the 3.10 the README names -- pyproject pins requires-python==3.11.*) and
# norm stats at .runs/openpi-05/assets/zedmini13/norm_stats.json.
#
# Usage:
#   bash scripts/train/openpi/finetune-pi05-zedmini13.sh [exp]
#   DRY_RUN=1 ...   parse config, verify packs/stats/canvas, exit
#   SMOKE=N   ...   run N steps on the visible GPUs, no checkpoints, no wandb

set -uo pipefail

TORCHRUN_PID=
CLEANUP_RUNNING=0

cleanup() {
    if [ "$CLEANUP_RUNNING" -eq 1 ]; then return; fi
    CLEANUP_RUNNING=1
    echo "Interrupted - stopping torchrun..."
    trap - INT TERM
    if [ -n "$TORCHRUN_PID" ] && kill -0 "$TORCHRUN_PID" 2>/dev/null; then
        kill -TERM "$TORCHRUN_PID" 2>/dev/null || true
        wait "$TORCHRUN_PID" 2>/dev/null || true
        kill -0 "$TORCHRUN_PID" 2>/dev/null && kill -KILL "$TORCHRUN_PID" 2>/dev/null || true
    fi
}
trap cleanup INT TERM

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
REPO_ROOT="$(pwd)"

[ -x .venv-openpi/bin/python ] || {
    echo "FATAL: .venv-openpi missing. See baselines/pi05/README.md." >&2; exit 1; }
source .venv-openpi/bin/activate

# jax is imported for jax.tree.map only on the PyTorch path, but jax[cuda12]
# preallocates most of the GPU on first device use and would starve torch.
export JAX_PLATFORMS=cpu

: "${OMP_NUM_THREADS:=32}"
export OMP_NUM_THREADS
ulimit -n 65535

CONFIG=pi05_zedmini13
EXP="${1:-${EXP:-pi05-zedmini13}}"
MAX_STEPS="${MAX_STEPS:-80000}"
BATCH="${BATCH:-128}"          # GLOBAL; train_pytorch divides by world size
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
CKPT_BASE="${CKPT_BASE:-.runs/openpi-05}"

NPROC_PER_NODE=$(echo "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" | tr ',' '\n' | wc -l)
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-$((29700 + ${SLURM_JOB_ID:-0} % 2000))}

# --- preflight: packs and norm stats -----------------------------------------
# The config resolves the roster; ask it rather than duplicating the list here.
python - <<'PY' || exit 1
import pathlib, sys
from dotenv import load_dotenv
# Explicit path: find_dotenv() walks the caller's stack frame and asserts when the
# script is fed on stdin, as it is here.
assert load_dotenv(".env"), "no .env"
import openpi.training.config as c

cfg = c.get_config("pi05_zedmini13")
dc = cfg.data.create(cfg.assets_dirs, cfg.model)

missing = [r for r in dc.repo_roots if not pathlib.Path(r).is_dir()]
if missing:
    print("FATAL: missing pack roots:\n  " + "\n  ".join(missing), file=sys.stderr)
    sys.exit(1)

stats = pathlib.Path(cfg.data.assets.assets_dir) / cfg.data.assets.asset_id / "norm_stats.json"
if not stats.is_file():
    print(f"FATAL: norm stats missing: {stats}\n"
          f"       run .logs/2026-08-06-Pi05_Zedmini13/compute_norm_stats.py first",
          file=sys.stderr)
    sys.exit(1)
if dc.norm_stats is None:
    print(f"FATAL: norm stats present but not loaded from {stats}", file=sys.stderr)
    sys.exit(1)

h, w = cfg.model.image_resolution
gh, gw = h // 14, w // 14
assert gh * 14 == h and gw * 14 == w, f"canvas {h}x{w} is not on the patch-14 grid"
print(f"[preflight] packs={len(dc.repo_roots)} canvas={h}x{w} -> {gh}x{gw}={gh*gw} tok/img "
      f"(x3 slots = {3*gh*gw}) action_dim={cfg.model.action_dim} horizon={cfg.model.action_horizon} "
      f"max_token_len={cfg.model.max_token_len} quantile_norm={dc.use_quantile_norm}")
print(f"[preflight] norm stats: state {dc.norm_stats['state'].mean.shape[-1]}-D "
      f"actions {dc.norm_stats['actions'].mean.shape[-1]}-D  <- {stats}")
PY

args=(
    "$CONFIG"
    --exp_name="$EXP"
    --batch_size="$BATCH"
    --num_train_steps="$MAX_STEPS"
    --save_interval="$SAVE_INTERVAL"
    --checkpoint_base_dir="$CKPT_BASE"
)

RUN_DIR="$CKPT_BASE/$CONFIG/$EXP"
latest_ckpt() {
    [ -d "$RUN_DIR" ] || return 0
    find "$RUN_DIR" -maxdepth 1 -type d -regextype posix-extended -regex '.*/[0-9]+' -printf '%f\n' 2>/dev/null \
        | sort -n | tail -1
}
LATEST="$(latest_ckpt)"

if [ -n "${RESUME:-}" ]; then
    # train_pytorch prints and carries on when --resume finds nothing, which would
    # silently restart from step 0 and burn the whole allocation. Fail here instead.
    [ -n "$LATEST" ] || {
        echo "FATAL: RESUME set but no checkpoint under $RUN_DIR" >&2; exit 1; }
    # Target already met: the previous leg finished on its own. This is the normal
    # outcome for a chained insurance leg, so exit clean rather than as a failure.
    [ "$LATEST" -lt "$MAX_STEPS" ] || {
        echo "Nothing to do: ckpt_$LATEST already meets MAX_STEPS ($MAX_STEPS)."; exit 0; }
    [ -s "$RUN_DIR/wandb_id.txt" ] || {
        echo "FATAL: $RUN_DIR/wandb_id.txt missing; wandb resume='must' would abort" >&2; exit 1; }
    args+=(--resume)
    echo "Resume: $RUN_DIR at ckpt_$LATEST -> $MAX_STEPS"
elif [ -n "$LATEST" ] && [ -z "${SMOKE:-}" ]; then
    echo "FATAL: $RUN_DIR already holds ckpt_$LATEST but RESUME is not set." >&2
    echo "       Set RESUME=1 to continue it, or pick a different exp name." >&2
    exit 1
fi

if [ -n "${SMOKE:-}" ]; then
    args=(
        "$CONFIG"
        --exp_name="${EXP}-smoke"
        --batch_size="$BATCH"
        --num_train_steps="$SMOKE"
        --save_interval=$((SMOKE + 1))
        --checkpoint_base_dir=".runs/openpi-05-smoke"
        --no-wandb_enabled
        --overwrite
    )
    echo "SMOKE: $SMOKE steps, global batch $BATCH on $NPROC_PER_NODE GPU(s), no ckpt, no wandb"
fi

echo "Config: $CONFIG | exp=$EXP | steps=$MAX_STEPS | global batch=$BATCH | GPUs=$NPROC_PER_NODE"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[dry-run] preflight passed; not launching."
    exit 0
fi

torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    src/openpi/train_pytorch.py \
    "${args[@]}" &

TORCHRUN_PID=$!
wait "$TORCHRUN_PID"
