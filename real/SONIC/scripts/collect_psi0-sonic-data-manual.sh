#!/bin/bash

# No-tmux version of collect_psi0-sonic-data.sh: run each component in its own
# terminal. Use this if tmux is not available.
#
# Real robot (start the camera server on the robot first):
#   bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy     # 1) C++ controller
#   bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico       # 2) PICO streamer
#   bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh exporter   # 3) data exporter (records)
#
# Simulation teleop test (no robot/camera, no recording):
#   bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh sim         # 1) MuJoCo sim
#   bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy sim  # 2) C++ controller (sim)
#   bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico        # 3) PICO streamer

ROBOT_IP=192.168.123.164
TASK="Pick bottle and turn and pour into cup."
FPS=30

SONIC_DIR="$(cd "$(dirname "$0")/../../../third_party/GR00T-WholeBodyControl" && pwd)"
cd "$SONIC_DIR"

case "$1" in
    sim)
        source .venv_teleop/bin/activate
        python gear_sonic/scripts/run_sim_loop.py
        ;;
    deploy)
        cd gear_sonic_deploy
        source scripts/setup_env.sh
        ./deploy.sh --input-type zmq_manager "${2:-real}"   # pass 'sim' as 2nd arg for sim mode
        ;;
    pico)
        source .venv_teleop/bin/activate
        python gear_sonic/scripts/pico_manager_thread_server.py --manager
        ;;
    exporter)
        source .venv_data_collection/bin/activate
        python gear_sonic/scripts/run_data_exporter.py \
            --camera-host "$ROBOT_IP" \
            --task-prompt "$TASK" \
            --data-collection-frequency "$FPS"
        ;;
    *)
        echo "Usage: $0 {sim|deploy [sim]|pico|exporter}   (run each in its own terminal)"
        exit 1
        ;;
esac
