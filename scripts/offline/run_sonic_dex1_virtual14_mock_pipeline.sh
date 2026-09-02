#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${PYTHON:-${SONIC_PYTHON:-python}}
SOURCE_DIR=${SOURCE_DIR:-/tmp/mock_sonic_dex1_virtual14_source}
OUT_ROOT=${OUT_ROOT:-/tmp/mock_sonic_dex1_virtual14_lerobot}
TASK=${TASK:-mock-dex1-virtual14}
MAPPING_STATS=${DEX1_VIRTUAL_STATS:-$ROOT_DIR/data/psidata_real/Put_dumpling_into_blanket_and_turn_around_and_pass_to_human/meta/stats.json}

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR"

"$PYTHON" scripts/data/make_mock_sonic_dataset.py \
    --output-dir "$SOURCE_DIR" --frames 6 --end-effector dex1_virtual14
"$PYTHON" scripts/data/preflight_sonic_dex1_1_dataset.py \
    --data-root "$SOURCE_DIR" --hand-layout dex1_virtual14
"$PYTHON" scripts/data/raw_sonic_to_psi_lerobot.py \
    --data-root "$SOURCE_DIR" --work-dir "$OUT_ROOT" --repo-id "$TASK" \
    --num-workers 1 --end-effector dex1_virtual14
"$PYTHON" scripts/data/calc_modality_stats.py --task-dir "$OUT_ROOT/$TASK"
cp "$OUT_ROOT/$TASK/meta/stats.json" "$OUT_ROOT/$TASK/meta/stats_psi0.json"
"$PYTHON" scripts/data/sanity_check_sonic_dex1_virtual14.py \
    --dataset-dir "$OUT_ROOT/$TASK" --mapping-stats "$MAPPING_STATS"
echo "Mock SONIC Dex1 virtual14 pipeline passed: $OUT_ROOT/$TASK"
