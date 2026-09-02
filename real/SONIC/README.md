# Ψ₀ with SONIC v1.1 — G1 + Dex1-1 Data Collection

This integration keeps NVIDIA's official SONIC manager, exporter, camera, and
C++ controller unchanged.  Psi0 supplies thin wrappers for the installed
Dex1-1 grippers and orchestration. Set portable workspace paths once:

```bash
export PSI_ROOT="$(git rev-parse --show-toplevel)"
export SONIC_DIR="${SONIC_DIR:-$PSI_ROOT/third_party/GR00T-WholeBodyControl}"
export SONIC_CONDA_PREFIX="${SONIC_CONDA_PREFIX:-$HOME/miniconda3/envs/sonic}"
```

## Current code map

The supported path is full-body SONIC with a dense virtual-Dex3 hand14 and
physical Dex1-1 grippers:

| Stage | Canonical entry | Main implementation |
|---|---|---|
| Collect | `real/SONIC/scripts/collect_psi0-sonic-data.sh` | `run_pico_manager_dex1.py`, `run_data_exporter_dex1.py` |
| Convert | `scripts/data/raw_sonic_to_psi_lerobot.py` | `dex1_1_layout.py`, `dex3_to_dex1.py` |
| Validate | `scripts/data/sanity_check_sonic_dex1_virtual14.py` | dense virtual-hand manifold checks |
| Train | `scripts/train/psi0/finetune-real-sonic-psi0.sh` | `SonicTrainer`, `Psi0Model` |
| Serve | `scripts/deploy/serve_psi0-rtc-sonic.sh` | `psi_serve_rtc_token-sonic.py` |
| Deploy | `real/scripts/deploy_psi0-sonic-rtc-{robot,client}.sh` | official RTC client plus `psi0_vla_dex1_bridge.py` |

The earlier scalar-hand smoke pipeline and its mock adapter were removed after
the dense virtual14 path and production policy bridge covered the same
interfaces.  The separate action36/decoupled-WBC experiments remain historical
rollback code; they are not part of this SONIC action78 workflow.

## Workstation environment

All Python processes use the existing conda `sonic` environment.  Do not create
or activate SONIC's `.venv_teleop`, `.venv_data_collection`, or `.venv_sim`.
The environment must provide the official fixed LeRobot commit, camera message
support, and a Python 3.11 XRoboToolkit binding:

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  "$SONIC_CONDA_PREFIX/bin/python" -m pip install \
  msgpack-numpy==0.4.8 datasets==3.6.0 jsonlines==4.0.0 draccus==0.10.0

GIT_LFS_SKIP_SMUDGE=1 \
  "$SONIC_CONDA_PREFIX/bin/python" -m pip install --no-deps \
  'lerobot @ git+https://github.com/huggingface/lerobot.git@a445d9c9da6bea99a8972daa4fe1fdd053d711d2'

"$SONIC_CONDA_PREFIX/bin/python" -m pip install cmake pybind11
CMAKE_PREFIX_PATH=$("$SONIC_CONDA_PREFIX/bin/python" -m pybind11 --cmakedir) \
  "$SONIC_CONDA_PREFIX/bin/python" -m pip install \
  --no-build-isolation -e \
  "$SONIC_DIR/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
```

Use the matching v1.1 files together; do not mix them with `release/` or
`low_latency/`:

```text
gear_sonic_deploy/policy/sonic_v1_1/model_encoder.onnx
gear_sonic_deploy/policy/sonic_v1_1/model_decoder.onnx
gear_sonic_deploy/policy/sonic_v1_1/observation_config.yaml
```

## Camera server (on the robot)

Reuse the `vision` conda env created in the robot [Image Server setup](../README.md#image-server-robot-only) (it already has `pyrealsense2`, `opencv`, `pyzmq`); just add the three remaining packages:

```bash
conda activate vision
pip install msgpack msgpack-numpy tyro
```

Copy the SONIC camera module from the workstation (run from the submodule root; G1 default IP `192.168.123.164`):

```bash
ssh unitree@192.168.123.164 mkdir -p ~/SONIC_psi0_release/gear_sonic
scp gear_sonic/__init__.py gear_sonic/version.py unitree@192.168.123.164:~/SONIC_psi0_release/gear_sonic/
scp -r gear_sonic/camera unitree@192.168.123.164:~/SONIC_psi0_release/gear_sonic/
scp real/SONIC/realsense_server.py unitree@192.168.123.164:~/SONIC_psi0_release/
```

Start the server on the robot (keep it running):

```bash
conda activate vision
cd ~/SONIC_psi0_release
python -m gear_sonic.camera.composed_camera --ego-view-camera realsense --port 5555
```

## Run

Run from the Psi0 root. The wrapper starts the official SONIC deploy, manager,
and exporter with the Dex1 adapter, plus a persistent Pico ego-view relay. It
performs only the necessary real-robot preflight: no competing lowcmd/Dex1
publisher, an available lowstate, and two fresh camera frames.

After powering on the workstation and G1, prepare all passive services first:

```bash
bash real/SONIC/scripts/collect_psi0-sonic-data.sh prepare
```

`prepare` checks the G1 network, subscribes read-only to lowstate and both
Dex1 states, ensures the dynamically discovered RealSense RGB service, and
idempotently starts the Pico relay. It never starts deploy, manager, exporter,
policy, `lowcmd`, Dex1 commands, or a dataset. XRoboToolkit/Full Body Tracking
remain explicit user steps; the command reports when RoboticsServiceProcess is
missing and asks the user to confirm `Working`.

```bash
bash real/SONIC/scripts/collect_psi0-sonic-data.sh start \
  --dataset-dir "$HOME/data/corn_plush_sonic" \
  --task 'Pick up the corn plush toy and place it into the basket.'
```

Use `--dry-run` with the same command to validate paths and dependencies
without starting tmux, DDS, camera, or robot processes. Legacy `check`, `sim`,
`real`, and emergency `stop` entry points remain available.

The exporter resumes an existing dataset only when its official 0-based
episode indices are contiguous. Official messages such as `Started recording
3`, `Stopping recording, preparing to save`, and `Finished saving episode` are
shown live in the exporter pane and copied into `.collect_logs/` by tmux. The
official camera client's repetitive `Image latency for ego_view` and
`[Latency] Sonic Pose` lines are filtered from this pane so recording and
validation status remains readable; camera data and timing are otherwise
unchanged.
If a prior session stopped after creating metadata but before episode 0, the
Psi0 exporter wrapper creates only the three missing empty LeRobot JSONL files
so the local dataset resumes without a Hugging Face network request.

The simulation launcher uses a Psi0 entry-point that suppresses the duplicate
DDS initialization recorded in upstream issue #77.  The upstream SONIC source
remains unchanged; `BaseSimulator` still performs the required single channel
initialization on `lo`.

Simulation uses SONIC's official elastic-band startup.  After starting the
policy, click the MuJoCo viewer and press **`9`** to disable the elastic band
and drop the robot onto the ground.  Pressing `9` before policy control is
active leaves the robot unsupported.

Data is written directly by the official exporter at the requested path, for
example:

```text
$HOME/data/corn_plush_sonic
```

This launcher records official full-body **POSE** demonstrations. Align the
operator with the robot before entering POSE. The triggers control physical
Dex1 and the same virtual hand14 targets stored in the dataset. Official
VR_3PT remains available on Left Stick Click for diagnostics, but an episode
containing VR_3PT is saved with the official discarded marker and must be
re-recorded for this dataset.

Recording buttons are deliberately exclusive to avoid official chord overlap:

- `Left Grip + A-only`: start/stop recording.
- `Left Grip + Y-only`: discard the current episode. Psi0 maps this to
  SONIC's internal discard event because Remote Vision owns physical B.
- Physical `B`: Remote Vision mono/stereo only; it cannot discard data.
- Keep the right stick centered and hold **Right Stick Click for 0.6 seconds**:
  start/stop SONIC policy. Release it completely before the next toggle.
  Right-stick direction remains the official planner yaw control; only the
  otherwise-unused click is remapped.
- `A+B+X+Y`: ignored by SONIC because Remote Vision owns this chord for its
  mono/stereo display. Physical face buttons are accumulated for 150 ms before
  dispatch, so asynchronous cross-controller arrival cannot leak A+X, B+Y,
  A+B, or X+Y while the four-button chord is being formed.
- `A+X`, `B+Y`, `A+B`, and `X+Y`: official mode/locomotion controls only,
  even if Left Grip is accidentally held.

Follow Psi0's official order: align with the robot in Planner, press `A+X` to
enter POSE, then press `Left Grip + A` to start recording. Perform the
full-body task and press `Left Grip + A` again to stop and save while still in
POSE. After saving finishes, press `A+X` to return to Planner Idle. Recording
controls remain mode-independent as in the upstream SONIC exporter.

After saving all episodes, physically support the robot and hold Right Stick
Click for 0.6 seconds to switch policy OFF. C++ SONIC then sends a final
damping command (`kp=0`, `kd=8`, zero torque) and stops active balance. Manager
exit triggers automatic exporter/deploy cleanup and verifies that
`rt/lowcmd` has stopped.

Before a normal save, the exporter checks the buffered stream modes, non-zero
POSE SMPL data, 43-D state/WBC action, 64-D motion token, finite values, and
hand ranges. Planner frames are permitted only as the lead-in and lead-out.
Invalid recordings are preserved on disk with SONIC's official discarded
metadata instead of being deleted or overwritten. Partial stale SMPL is only
reported and is handled later by the official dataset processor.

## Pico ego view

The G1 `ego_view` runs in the independent tmux session
`psi0_pico_ego_view`. It stays visible across episode boundaries, policy OFF,
manager exit, and later collection restarts. Configure Remote Vision for H.264
and point it at `<workstation-wifi-ip>:13579`. The relay listens on all local
interfaces by default; override `PICO_VIDEO_HOST` only when a fixed bind address
is required.

The relay uses the same official camera server (`192.168.123.164:5555`) as the
exporter, PyAV from conda `sonic`, latest-frame-only buffering, reconnect
keyframes, and automatic process restart. The collection `start` command
idempotently ensures that the persistent relay exists; ordinary `stop` leaves
it running. Use the maintenance commands below only when intentionally
starting the view without a policy or shutting the view down:

```bash
bash real/SONIC/scripts/collect_psi0-sonic-data.sh view-start
bash real/SONIC/scripts/collect_psi0-sonic-data.sh view-stop
```

Set `DESKTOP_VIEWER=1` when a second workstation preview is also needed. The
robot-side camera server remains independent and running after both policy and
relay shutdown. Camera preflight has a bounded timeout. If the service is
missing or stale, the launcher discovers the current RealSense RGB node by
stable USB properties (interface 03, stream index 0), restarts the camera once,
and verifies two changing frames before any SONIC/DDS process starts. It never
depends on an unstable `/dev/videoN` number and leaves a healthy camera alone.
System SSH calls explicitly drop conda's `LD_LIBRARY_PATH`, preventing the
sonic environment's OpenSSL libraries from breaking Ubuntu OpenSSH.

The robot RGB source is 640x480 (4:3), while Remote Vision requests 1280x720
per eye (16:9). Each eye therefore displays the complete source at 960x720
with 160-pixel black bars on the left and right. This preserves field of view
and aspect ratio without changing the raw 640x480 image recorded for training.
If Remote Vision drops the video socket while switching mono/stereo, the relay
abandons the stale endpoint and performs a fresh `OPEN_CAMERA` handshake.

No tmux? Run each component in its own terminal instead:

```bash
# sim teleop test
bash real/SONIC/scripts/collect_psi0-sonic-data-manual.sh sim
bash real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy sim
bash real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico
```

```bash
# real robot
bash real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy
bash real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico
bash real/SONIC/scripts/collect_psi0-sonic-data-manual.sh exporter
bash real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico-view
# optional workstation preview:
bash real/SONIC/scripts/collect_psi0-sonic-data-manual.sh viewer
```

## Convert and validate

Dex1 is represented as a reversible dense virtual-Dex3 hand14.  Both state and
action use the same mapping; the physical hardware remains two scalar grippers.
The official RobotModel stores each 7-D hand block as
`index0,index1,middle0,middle1,thumb0,thumb1,thumb2`, while SONIC teleop uses
`thumb0,thumb1,thumb2,index0,index1,middle0,middle1`.  The converter applies
that inverse permutation before writing Psi0 `states[29:43]`.

For `dex1_virtual14`, action hand14 is reconstructed from the official
`teleop.left_hand_joints` and `teleop.right_hand_joints` columns.  This is
intentional: older Psi0 wrapper recordings patched physical state correctly,
but their `action.wbc` hand blocks could remain static in POSE mode.  Source
Parquet and video files are never rewritten.

```bash
"$SONIC_CONDA_PREFIX/bin/python" \
  scripts/data/preflight_sonic_dex1_1_dataset.py \
  --data-root "$SONIC_DIR/outputs/$DATASET_NAME" \
  --hand-layout dex1_virtual14

"$SONIC_CONDA_PREFIX/bin/python" \
  scripts/data/raw_sonic_to_psi_lerobot.py \
  --data-root "$SONIC_DIR/outputs/$DATASET_NAME" \
  --work-dir "$PSI_HOME/data/sonic/lerobot" \
  --repo-id "$DATASET_NAME" \
  --robot-type g1 \
  --end-effector dex1_virtual14

"$SONIC_CONDA_PREFIX/bin/python" \
  scripts/data/calc_modality_stats.py \
  --task-dir "$PSI_HOME/data/sonic/lerobot/$DATASET_NAME"
cp "$PSI_HOME/data/sonic/lerobot/$DATASET_NAME/meta/stats.json" \
   "$PSI_HOME/data/sonic/lerobot/$DATASET_NAME/meta/stats_psi0.json"

"$SONIC_CONDA_PREFIX/bin/python" \
  scripts/data/sanity_check_sonic_dex1_virtual14.py \
  --dataset-dir "$PSI_HOME/data/sonic/lerobot/$DATASET_NAME" \
  --mapping-stats "$DEX1_VIRTUAL_STATS" \
  --required-moving-hands right
```

`--required-moving-hands` defaults to `both`.  Use `right` for the retained
2026-08-20 episode because that task uses only the right gripper.  The left
Dex1 later passed a separate small-range forward/return test; this option
describes episode content, not current hardware health.  It is an offline
data-quality assertion, not an additional real-robot runtime gate.

All source changes and commits belong to Psi0.  The wbi SONIC checkout is used
as an upstream runtime and weight directory only.  The Psi0 launcher builds
the ZMQ-only controller with the official `HAS_ROS2=0` switch; ROS2 input is not
used by this collection path.  Its `just` wrapper also prioritizes the bundled
Unitree CycloneDDS libraries at runtime, preventing ROS Humble's different
`libddsc` ABI from being loaded into the Unitree controller.
