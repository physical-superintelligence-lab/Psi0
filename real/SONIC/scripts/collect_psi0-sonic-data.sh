#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/sonic_conda_env.sh"

ROBOT_IP="${ROBOT_IP:-192.168.123.164}"
ROBOT_SSH_TARGET="${ROBOT_SSH_TARGET:-unitree@$ROBOT_IP}"
ROBOT_CAMERA_SCRIPT="${ROBOT_CAMERA_SCRIPT:-/home/unitree/cam_relay/run_cam.sh}"
SYSTEM_SSH="${SYSTEM_SSH:-/usr/bin/ssh}"
G1_CAMERA_WARMUP_SEC="${G1_CAMERA_WARMUP_SEC:-6}"
TASK="${TASK:-Pick up the corn plush toy and place it into the basket.}"
DATASET_NAME="${DATASET_NAME:-corn_plush_mobile_sonic_v1_1}"
DATASET_DIR="${DATASET_DIR:-}"
FPS="${FPS:-30}"
SESSION="${SONIC_TMUX_SESSION:-psi0_sonic_v1_1_collection}"
PICO_SESSION="${SONIC_PICO_TMUX_SESSION:-psi0_pico_ego_view}"
LOG_DIR="${SONIC_LOG_DIR:-$PSI_ROOT/.collect_logs}"
PICO_VIDEO="${PICO_VIDEO:-1}"
PICO_VIDEO_HOST="${PICO_VIDEO_HOST:-0.0.0.0}"
PICO_VIDEO_PORT="${PICO_VIDEO_PORT:-13579}"
DESKTOP_VIEWER="${DESKTOP_VIEWER:-0}"
RUNTIME_DIR="/tmp/psi0-sonic-${UID}-${SESSION}"

for COLLECTION_TMUX_NAME in "$SESSION" "$PICO_SESSION"; do
    case "$COLLECTION_TMUX_NAME" in
        *[!A-Za-z0-9_.-]*)
            echo "Invalid tmux session name: use only letters, digits, dot, underscore, or dash" >&2
            exit 2
            ;;
    esac
done

MODE="${1:-real}"
if [ "$#" -gt 0 ]; then
    shift
fi
if [ "$MODE" = "start" ]; then
    MODE="real"
    START_CLI=1
else
    START_CLI=0
fi
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage:
  collect_psi0-sonic-data.sh start --dataset-dir PATH --task TEXT [--dry-run]
  collect_psi0-sonic-data.sh prepare
  collect_psi0-sonic-data.sh check
  collect_psi0-sonic-data.sh sim
  collect_psi0-sonic-data.sh stop
  collect_psi0-sonic-data.sh view-start
  collect_psi0-sonic-data.sh view-stop

Legacy environment-variable form remains supported:
  TASK=... DATASET_NAME=... collect_psi0-sonic-data.sh real
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dataset-dir)
            [ "$#" -ge 2 ] || { echo "--dataset-dir requires a value" >&2; exit 2; }
            DATASET_DIR="$2"
            shift 2
            ;;
        --task)
            [ "$#" -ge 2 ] || { echo "--task requires a value" >&2; exit 2; }
            TASK="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

mkdir -p "$LOG_DIR"

cleanup_runtime() {
    if [ -d "$RUNTIME_DIR" ]; then
        rm -f "$RUNTIME_DIR/manager.exited" "$RUNTIME_DIR/start.incomplete"
        rmdir "$RUNTIME_DIR" 2>/dev/null || true
    fi
}

run_system_ssh() {
    # conda sonic ships OpenSSL libraries that are incompatible with Ubuntu's
    # system ssh. Keep the robotics library path everywhere except this child.
    env -u LD_LIBRARY_PATH "$SYSTEM_SSH" "$@"
}

camera_stream_preflight() {
    ROBOT_IP="$ROBOT_IP" timeout 12 "$SONIC_PYTHON" - <<'PY'
import hashlib
import os
from gear_sonic.camera.composed_camera import ComposedCameraClientSensor

camera = ComposedCameraClientSensor(server_ip=os.environ["ROBOT_IP"], port=5555)
try:
    frames = [camera.read(blocking=True), camera.read(blocking=True)]
    hashes = []
    for frame in frames:
        if not frame or not frame.get("images"):
            raise SystemExit("camera returned no images")
        image = frame["images"].get("ego_view")
        if image is None:
            raise SystemExit("camera returned no ego_view")
        hashes.append(hashlib.sha256(image.tobytes()).digest())
    if hashes[0] == hashes[1]:
        raise SystemExit("camera frames are frozen")
    print("camera is live")
finally:
    camera.close()
PY
}

discover_g1_realsense_rgb_index() {
    run_system_ssh -o BatchMode=yes -o ConnectTimeout=3 "$ROBOT_SSH_TARGET" '
        for sysnode in /sys/class/video4linux/video*; do
            node=${sysnode##*/}
            [ -r "$sysnode/name" ] || continue
            [ -r "$sysnode/index" ] || continue
            case "$(cat "$sysnode/name")" in
                *RealSense*) ;;
                *) continue ;;
            esac
            [ "$(cat "$sysnode/index")" = "0" ] || continue
            udevadm info -q property -n "/dev/$node" 2>/dev/null \
                | grep -qx "ID_USB_INTERFACE_NUM=03" || continue
            printf "%s\n" "${node#video}"
            exit 0
        done
        exit 1
    '
}

restart_g1_camera_service() {
    local camera_index
    camera_index=$(discover_g1_realsense_rgb_index) || {
        echo "No RealSense RGB node found (expected interface 03, stream index 0)" >&2
        return 1
    }
    case "$camera_index" in
        ''|*[!0-9]*)
            echo "Invalid RealSense RGB video index: $camera_index" >&2
            return 1
            ;;
    esac

    echo "Discovered G1 RealSense RGB node: /dev/video$camera_index"
    run_system_ssh -o BatchMode=yes -o ConnectTimeout=3 "$ROBOT_SSH_TARGET" \
        "if [ -r /home/unitree/cam_relay/cam.pid ]; then old_pid=\$(cat /home/unitree/cam_relay/cam.pid); case \"\$old_pid\" in *[!0-9]*|'') ;; *) kill \"\$old_pid\" 2>/dev/null || true ;; esac; fi; nohup env CAM_IDX='$camera_index' '$ROBOT_CAMERA_SCRIPT' </dev/null >/dev/null 2>&1 &"
    sleep "$G1_CAMERA_WARMUP_SEC"
}

ensure_g1_camera_service() {
    CAMERA_SERVICE_RESTARTED=0
    if camera_stream_preflight; then
        return 0
    fi

    echo "G1 camera missing or stale; discovering the current RealSense RGB node..."
    restart_g1_camera_service
    if ! camera_stream_preflight; then
        echo "G1 camera failed after one automatic restart at $ROBOT_IP:5555" >&2
        return 1
    fi
    CAMERA_SERVICE_RESTARTED=1
}

prepare_network_preflight() {
    if ! ip -4 addr show "$G1_NETWORK_INTERFACE" 2>/dev/null \
        | grep -q 'inet 192\.168\.123\.'; then
        echo "$G1_NETWORK_INTERFACE is not configured for the G1 192.168.123.x network" >&2
        return 1
    fi
    if ! ping -c 1 -W 1 "$ROBOT_IP" >/dev/null 2>&1; then
        echo "G1 is unreachable at $ROBOT_IP" >&2
        return 1
    fi
    echo "G1 network ready: $G1_NETWORK_INTERFACE -> $ROBOT_IP"
}

prepare_dds_preflight() {
    G1_NETWORK_INTERFACE="$G1_NETWORK_INTERFACE" "$SONIC_PYTHON" - <<'PY'
import os
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

interface = os.environ["G1_NETWORK_INTERFACE"]
ChannelFactoryInitialize(0, interface)
count = {"lowcmd": 0, "lowstate": 0, "dex_cmd": 0, "left": 0, "right": 0}
ChannelSubscriber("rt/lowcmd", LowCmd_).Init(
    lambda _: count.__setitem__("lowcmd", count["lowcmd"] + 1), 10
)
ChannelSubscriber("rt/lowstate", LowState_).Init(
    lambda _: count.__setitem__("lowstate", count["lowstate"] + 1), 10
)
for side in ("left", "right"):
    ChannelSubscriber(f"rt/dex1/{side}/cmd", MotorCmds_).Init(
        lambda _, key="dex_cmd": count.__setitem__(key, count[key] + 1), 10
    )
    ChannelSubscriber(f"rt/dex1/{side}/state", MotorStates_).Init(
        lambda _, key=side: count.__setitem__(key, count[key] + 1), 10
    )
time.sleep(2)
if count["lowstate"] == 0:
    raise SystemExit("G1 lowstate is unavailable")
if count["left"] == 0 or count["right"] == 0:
    raise SystemExit(
        f"Dex1 state unavailable: left={count['left']} right={count['right']}"
    )
if count["lowcmd"] or count["dex_cmd"]:
    raise SystemExit(
        f"control channel busy: lowcmd={count['lowcmd']} dex1_cmd={count['dex_cmd']}"
    )
print(
    "Read-only DDS ready: "
    f"lowstate={count['lowstate']} left={count['left']} right={count['right']} "
    "lowcmd=0 dex1_cmd=0"
)
PY
}

run_prepare() {
    prepare_network_preflight
    prepare_dds_preflight
    ensure_g1_camera_service
    if [ "$CAMERA_SERVICE_RESTARTED" = "1" ]; then
        stop_pico_view
    fi
    start_pico_view

    if pgrep -f '[R]oboticsServiceProcess' >/dev/null 2>&1; then
        echo "XRoboToolkit RoboticsServiceProcess is running"
        echo "PREPARE READY: open Pico Full Body Tracking and Remote Vision, then confirm Working."
        return 0
    fi
    echo "ACTION REQUIRED: launch XRoboToolkit; RoboticsServiceProcess is not running." >&2
    echo "Camera and Pico relay are prepared, but Full Body Tracking still requires user confirmation." >&2
    return 2
}

start_pico_view() {
    if [ "$PICO_VIDEO" != "1" ]; then
        echo "Pico video is disabled (PICO_VIDEO=$PICO_VIDEO)"
        return 0
    fi
    if tmux has-session -t "$PICO_SESSION" 2>/dev/null; then
        echo "Persistent Pico ego view already running: $PICO_SESSION"
        return 0
    fi

    tmux new-session -d -s "$PICO_SESSION" -c "$PSI_ROOT"
    tmux rename-window -t "$PICO_SESSION:0" ego_view
    local pane
    pane=$(tmux display-message -p -t "$PICO_SESSION:0.0" '#{pane_id}')
    tmux pipe-pane -o -t "$pane" "cat >> '$LOG_DIR/sonic_pico_view.log'"
    tmux send-keys -t "$pane" \
        "source '$SCRIPT_DIR/sonic_conda_env.sh'; set +e; while true; do '$SONIC_PYTHON' -u '$PSI_ROOT/real/teleop/pico_camera_view.py' --camera-protocol sonic --camera-ip '$ROBOT_IP' --camera-port 5555 --camera-key ego_view --encoder pyav --listen-host '$PICO_VIDEO_HOST' --listen-port '$PICO_VIDEO_PORT' || true; sleep 1; done" C-m
    echo "Persistent Pico ego view started: $PICO_SESSION ($PICO_VIDEO_HOST:$PICO_VIDEO_PORT)"
}

stop_pico_view() {
    if tmux has-session -t "$PICO_SESSION" 2>/dev/null; then
        tmux kill-session -t "$PICO_SESSION"
        echo "Persistent Pico ego view stopped: $PICO_SESSION"
    else
        echo "Persistent Pico ego view is not running: $PICO_SESSION"
    fi
}

verify_lowcmd_stopped() {
    "$SONIC_PYTHON" - <<'PY'
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
ChannelFactoryInitialize(0, "enp4s0")
count = {"n": 0}
ChannelSubscriber("rt/lowcmd", LowCmd_).Init(
    lambda _: count.__setitem__("n", count["n"] + 1), 10
)
time.sleep(2)
print(f"rt/lowcmd packets after stop: {count['n']}")
raise SystemExit(0 if count["n"] == 0 else 1)
PY
}

stop_session() {
    pkill -INT -f "[r]un_pico_manager_dex1.py" 2>/dev/null || true
    pkill -INT -f "[r]un_data_exporter_dex1.py" 2>/dev/null || true
    sleep 3
    pkill -TERM -f "[g]1_deploy_onnx_ref" 2>/dev/null || true
    sleep 2
    pkill -KILL -f "[g]1_deploy_onnx_ref" 2>/dev/null || true
    local verify_status=0
    verify_lowcmd_stopped || verify_status=$?
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    cleanup_runtime
    return "$verify_status"
}

if [ "$MODE" = "stop" ]; then
    stop_session
    exit $?
fi
if [ "$MODE" = "view-start" ]; then
    ensure_g1_camera_service
    if [ "$CAMERA_SERVICE_RESTARTED" = "1" ]; then
        stop_pico_view
    fi
    start_pico_view
    exit $?
fi
if [ "$MODE" = "view-stop" ]; then
    stop_pico_view
    exit $?
fi

require_file() {
    if [ ! -f "$1" ]; then
        echo "Missing required file: $1" >&2
        exit 1
    fi
}

require_file "$SONIC_DIR/gear_sonic_deploy/${SONIC_CHECKPOINT}_encoder.onnx"
require_file "$SONIC_DIR/gear_sonic_deploy/${SONIC_CHECKPOINT}_decoder.onnx"
require_file "$SONIC_DIR/gear_sonic_deploy/$SONIC_OBS_CONFIG"
require_file "$DEX1_VIRTUAL_STATS"

"$SONIC_PYTHON" - <<'PY'
import importlib
for name in ("av", "gear_sonic", "tyro", "lerobot", "msgpack_numpy", "xrobotoolkit_sdk"):
    importlib.import_module(name)
print("Python dependencies OK")
PY

if [ "$MODE" = "prepare" ]; then
    run_prepare
    exit $?
fi

if [ "$MODE" = "check" ]; then
    echo "SONIC_DIR=$SONIC_DIR"
    echo "SONIC_PYTHON=$SONIC_PYTHON"
    echo "SONIC_CHECKPOINT=$SONIC_CHECKPOINT"
    echo "SONIC_OBS_CONFIG=$SONIC_OBS_CONFIG"
    echo "DEX1_VIRTUAL_STATS=$DEX1_VIRTUAL_STATS"
    echo "Offline environment check passed"
    exit 0
fi

if [ "$MODE" = "sim" ]; then
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION" -c "$SONIC_DIR"
    SIM_PANE=$(tmux display-message -p -t "$SESSION:0.0" '#{pane_id}')
    tmux send-keys -t "$SIM_PANE" \
        "source '$SCRIPT_DIR/sonic_conda_env.sh' && '$SONIC_PYTHON' '$PSI_ROOT/real/SONIC/run_sim_loop_single_dds.py'" C-m
    DEPLOY_PANE=$(tmux split-window -h -P -F '#{pane_id}' -t "$SIM_PANE" -c "$SONIC_DIR/gear_sonic_deploy")
    tmux send-keys -t "$DEPLOY_PANE" \
        "source '$SCRIPT_DIR/sonic_conda_env.sh' && '$SCRIPT_DIR/collect_psi0-sonic-data-manual.sh' deploy sim" C-m
    MANAGER_PANE=$(tmux split-window -v -P -F '#{pane_id}' -t "$DEPLOY_PANE" -c "$PSI_ROOT")
    tmux send-keys -t "$MANAGER_PANE" \
        "source '$SCRIPT_DIR/sonic_conda_env.sh' && '$SONIC_PYTHON' -u real/SONIC/run_pico_manager_dex1.py --no-dex1-hardware" C-m
    tmux select-layout -t "$SESSION" tiled
    exec tmux attach -t "$SESSION"
fi

if [ "$MODE" != "real" ]; then
    usage >&2
    exit 1
fi

if [ "$START_CLI" -eq 1 ] && [ -z "$DATASET_DIR" ]; then
    echo "start requires --dataset-dir" >&2
    exit 2
fi
if [ -n "$DATASET_DIR" ]; then
    DATASET_DIR=$(realpath -m "$DATASET_DIR")
    DATASET_NAME=$(basename "$DATASET_DIR")
    ROOT_OUTPUT_DIR=$(dirname "$DATASET_DIR")
else
    ROOT_OUTPUT_DIR="$SONIC_DIR/outputs"
    DATASET_DIR="$ROOT_OUTPUT_DIR/$DATASET_NAME"
fi

DATASET_DIR="$DATASET_DIR" FPS="$FPS" "$SONIC_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["DATASET_DIR"])
if not root.exists():
    print(f"Dataset is new; next official episode index: 0 ({root})")
    raise SystemExit(0)
info_path = root / "meta/info.json"
episodes_path = root / "meta/episodes.jsonl"
modality_path = root / "meta/modality.json"
if not info_path.is_file() or not modality_path.is_file():
    raise SystemExit(f"Refusing to resume non-LeRobot directory: {root}")
info = json.loads(info_path.read_text(encoding="utf-8"))
total = int(info.get("total_episodes", -1))
if total > 0 and not episodes_path.is_file():
    raise SystemExit(f"Dataset metadata is incomplete: missing {episodes_path}")
episodes = (
    [
        json.loads(line)
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if episodes_path.is_file()
    else []
)
indices = [int(row["episode_index"]) for row in episodes]
expected = list(range(total))
if indices != expected:
    raise SystemExit(
        f"Dataset episode indices are not contiguous: found={indices}, expected={expected}. "
        "Write a cleaned dataset with official process_dataset.py before appending."
    )
stored_fps = int(info.get("fps", -1))
requested_fps = int(os.environ["FPS"])
if stored_fps != requested_fps:
    raise SystemExit(
        f"Dataset fps={stored_fps} does not match requested fps={requested_fps}"
    )
for episode_index in expected:
    name = f"episode_{episode_index:06d}"
    parquets = list((root / "data").glob(f"*/{name}.parquet"))
    videos = list((root / "videos").glob(f"*/observation.images.ego_view/{name}.mp4"))
    if len(parquets) != 1 or len(videos) != 1:
        raise SystemExit(
            f"Episode {episode_index} files are incomplete: "
            f"parquet={len(parquets)} ego_video={len(videos)}"
        )
print(f"Existing official episodes: {total}; next official episode index: {total}")
PY

echo "============================================================"
echo "SONIC v1.1 Full-body POSE collection"
echo "Task:       $TASK"
echo "Dataset:    $DATASET_DIR"
echo "Camera:     $ROBOT_IP:5555"
echo "Pico video: $PICO_VIDEO ($PICO_VIDEO_HOST:$PICO_VIDEO_PORT)"
echo "============================================================"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY RUN: no robot, tmux, DDS, or camera process was started."
    exit 0
fi

# Necessary real-robot preflight only: no competing body/Dex1 controller.
"$SONIC_PYTHON" - <<'PY'
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
ChannelFactoryInitialize(0, "enp4s0")
count = {"cmd": 0, "state": 0, "grip": 0, "left_grip_state": 0, "right_grip_state": 0}
ChannelSubscriber("rt/lowcmd", LowCmd_).Init(lambda _: count.__setitem__("cmd", count["cmd"] + 1), 10)
ChannelSubscriber("rt/lowstate", LowState_).Init(lambda _: count.__setitem__("state", count["state"] + 1), 10)
for side in ("left", "right"):
    topic = f"rt/dex1/{side}/cmd"
    ChannelSubscriber(topic, MotorCmds_).Init(lambda _: count.__setitem__("grip", count["grip"] + 1), 10)
    state_key = f"{side}_grip_state"
    state_topic = f"rt/dex1/{side}/state"
    ChannelSubscriber(state_topic, MotorStates_).Init(
        lambda _, key=state_key: count.__setitem__(key, count[key] + 1), 10
    )
time.sleep(2)
if count["state"] == 0:
    raise SystemExit("G1 lowstate is unavailable")
if count["cmd"] or count["grip"]:
    raise SystemExit(f"control channel busy: lowcmd={count['cmd']} dex1_cmd={count['grip']}")
if count["left_grip_state"] == 0 or count["right_grip_state"] == 0:
    raise SystemExit(
        "Dex1 state unavailable: "
        f"left={count['left_grip_state']} right={count['right_grip_state']}"
    )
print(
    "G1 online, control channels idle, Dex1 states live: "
    f"left={count['left_grip_state']} right={count['right_grip_state']}"
)
PY

# Necessary data preflight and one-shot recovery use stable RealSense USB properties.
ensure_g1_camera_service
if [ "$CAMERA_SERVICE_RESTARTED" = "1" ]; then
    stop_pico_view
fi

start_pico_view

read -r -p "Proceed with REAL SONIC deployment? [y/N]: " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

cleanup_runtime
mkdir -p "$RUNTIME_DIR"
touch "$RUNTIME_DIR/start.incomplete"
trap 'if [ -e "$RUNTIME_DIR/start.incomplete" ]; then tmux kill-session -t "$SESSION" 2>/dev/null || true; cleanup_runtime; fi' EXIT INT TERM

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -c "$SONIC_DIR/gear_sonic_deploy"
tmux rename-window -t "$SESSION:0" collection
tmux set-option -t "$SESSION" -g mouse on
tmux bind-key -T root 'C-\' run-shell "'$SCRIPT_DIR/collect_psi0-sonic-data.sh' stop"
DEPLOY_PANE=$(tmux display-message -p -t "$SESSION:0.0" '#{pane_id}')
tmux pipe-pane -o -t "$DEPLOY_PANE" "cat >> '$LOG_DIR/sonic_deploy.log'"

# Start order is intentional: deploy subscribes before the manager PUB exists.
tmux send-keys -t "$DEPLOY_PANE" \
    "source '$SCRIPT_DIR/sonic_conda_env.sh' && '$SCRIPT_DIR/collect_psi0-sonic-data-manual.sh' deploy real" C-m
# Queue the official deploy confirmation for its later read prompt.
tmux send-keys -t "$DEPLOY_PANE" Y C-m
sleep 3

EXPORTER_PANE=$(tmux split-window -h -P -F '#{pane_id}' -t "$DEPLOY_PANE" -c "$SONIC_DIR")
tmux pipe-pane -o -t "$EXPORTER_PANE" "cat >> '$LOG_DIR/sonic_exporter.log'"
printf -v TASK_Q '%q' "$TASK"
printf -v DATASET_NAME_Q '%q' "$DATASET_NAME"
printf -v ROOT_OUTPUT_DIR_Q '%q' "$ROOT_OUTPUT_DIR"
tmux send-keys -t "$EXPORTER_PANE" \
    "source '$SCRIPT_DIR/sonic_conda_env.sh' && set -o pipefail && TASK=$TASK_Q DATASET_NAME=$DATASET_NAME_Q ROOT_OUTPUT_DIR=$ROOT_OUTPUT_DIR_Q ROBOT_IP='$ROBOT_IP' FPS='$FPS' '$SCRIPT_DIR/collect_psi0-sonic-data-manual.sh' exporter 2>&1 | sed -u -e '/^Image latency for ego_view:/d' -e '/^\[Latency\] Sonic Pose:/d'" C-m

MANAGER_PANE=$(tmux split-window -v -P -F '#{pane_id}' -t "$DEPLOY_PANE" -c "$PSI_ROOT")
tmux pipe-pane -o -t "$MANAGER_PANE" "cat >> '$LOG_DIR/sonic_manager.log'"
tmux send-keys -t "$MANAGER_PANE" \
    "source '$SCRIPT_DIR/sonic_conda_env.sh'; set +e; '$SONIC_PYTHON' -u real/SONIC/run_pico_manager_dex1.py --network '$G1_NETWORK_INTERFACE'; touch '$RUNTIME_DIR/manager.exited'" C-m

if [ "$DESKTOP_VIEWER" = "1" ]; then
    VIEWER_PANE=$(tmux split-window -v -P -F '#{pane_id}' -t "$EXPORTER_PANE" -c "$SONIC_DIR")
    tmux pipe-pane -o -t "$VIEWER_PANE" "cat >> '$LOG_DIR/sonic_viewer.log'"
    tmux send-keys -t "$VIEWER_PANE" \
        "source '$SCRIPT_DIR/sonic_conda_env.sh' && '$SONIC_PYTHON' gear_sonic/scripts/run_camera_viewer.py --camera-host '$ROBOT_IP' --camera-port 5555" C-m
fi

SUPERVISOR_PANE=$(tmux new-window -d -P -F '#{pane_id}' -t "$SESSION" -n supervisor -c "$PSI_ROOT")
tmux pipe-pane -o -t "$SUPERVISOR_PANE" "cat >> '$LOG_DIR/sonic_supervisor.log'"
tmux send-keys -t "$SUPERVISOR_PANE" \
    "while [ ! -e '$RUNTIME_DIR/manager.exited' ]; do sleep 1; done; '$SCRIPT_DIR/collect_psi0-sonic-data.sh' stop" C-m

tmux select-window -t "$SESSION:collection"
tmux select-layout -t "$SESSION:collection" tiled
rm -f "$RUNTIME_DIR/start.incomplete"
trap - EXIT INT TERM

echo "SONIC v1.1 collection started in tmux session: $SESSION"
echo "Task: $TASK"
echo "Dataset: $DATASET_DIR"
echo "Policy: keep RIGHT stick centered and hold its click for 0.6s; A+X toggles PLANNER <-> POSE."
echo "Episode: align in PLANNER -> A+X POSE -> record/task/save -> A+X PLANNER."
echo "Recording: Left Grip+A starts/stops; Left Grip+Y discards. Physical B is Remote Vision only."
echo "A non-POSE episode is saved with the official discarded marker and must be re-recorded."
echo "Official recording logs are visible in the exporter pane."
echo "Pico ego view is persistent and stays live across policy/session shutdown."
echo "Stop the view only when intended: '$SCRIPT_DIR/collect_psi0-sonic-data.sh view-stop'"
echo "Emergency cleanup: Ctrl+\\"
exec tmux attach -t "$SESSION"
