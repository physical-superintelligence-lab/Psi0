#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/sonic_conda_env.sh"

ROBOT_IP="${ROBOT_IP:-192.168.123.164}"
TASK="${TASK:-Pick up the corn plush toy and place it into the basket.}"
DATASET_NAME="${DATASET_NAME:-corn_plush_mobile_sonic_v1_1}"
FPS="${FPS:-30}"
LOG_DIR="${SONIC_LOG_DIR:-$PSI_ROOT/.collect_logs}"
ROOT_OUTPUT_DIR="${ROOT_OUTPUT_DIR:-$SONIC_DIR/outputs}"
mkdir -p "$LOG_DIR"

case "${1:-}" in
    sim)
        cd "$SONIC_DIR"
        exec "$SONIC_PYTHON" "$PSI_ROOT/real/SONIC/run_sim_loop_single_dds.py"
        ;;
    deploy)
        cd "$SONIC_DIR/gear_sonic_deploy"
        export PATH="$PSI_ROOT/real/SONIC/scripts/no_ros2_bin:$PATH"
        exec ./deploy.sh \
            --cp "$SONIC_CHECKPOINT" \
            --obs-config "$SONIC_OBS_CONFIG" \
            --input-type zmq_manager \
            --output-type zmq "${2:-real}"
        ;;
    pico)
        cd "$PSI_ROOT"
        exec "$SONIC_PYTHON" -u real/SONIC/run_pico_manager_dex1.py \
            --network "$G1_NETWORK_INTERFACE"
        ;;
    exporter)
        cd "$SONIC_DIR"
        exec "$SONIC_PYTHON" -u "$PSI_ROOT/real/SONIC/run_data_exporter_dex1.py" \
            --camera-host "$ROBOT_IP" \
            --camera-port 5555 \
            --task-prompt "$TASK" \
            --dataset-name "$DATASET_NAME" \
            --root-output-dir "$ROOT_OUTPUT_DIR" \
            --data-collection-frequency "$FPS" \
            --no-text-to-speech
        ;;
    viewer)
        cd "$SONIC_DIR"
        exec "$SONIC_PYTHON" gear_sonic/scripts/run_camera_viewer.py \
            --camera-host "$ROBOT_IP" --camera-port 5555
        ;;
    pico-view)
        cd "$PSI_ROOT"
        exec "$SONIC_PYTHON" -u real/teleop/pico_camera_view.py \
            --camera-protocol sonic \
            --camera-ip "$ROBOT_IP" --camera-port 5555 \
            --camera-key ego_view --encoder pyav \
            --listen-host "${PICO_VIDEO_HOST:-0.0.0.0}" \
            --listen-port "${PICO_VIDEO_PORT:-13579}"
        ;;
    *)
        echo "Usage: $0 {sim|deploy [sim]|pico|exporter|viewer|pico-view}" >&2
        exit 1
        ;;
esac
