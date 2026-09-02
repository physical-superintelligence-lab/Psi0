import json
import os
from pathlib import Path
import subprocess
import tempfile

from real.SONIC.run_pico_manager_dex1 import (
    ControllerInputRouter,
    StartChordReleaseFilter,
    enter_sonic_runtime_dir,
    require_dex1_state_pair,
)


def test_manager_enters_sonic_runtime_directory(tmp_path):
    original = Path.cwd()
    try:
        enter_sonic_runtime_dir(tmp_path)
        assert Path.cwd() == tmp_path.resolve()
    finally:
        os.chdir(original)


def test_start_chord_subcombos_are_suppressed_until_full_release():
    clock = FakeClock()
    buttons = {"value": (True, False, False, False)}
    filtered = StartChordReleaseFilter(
        lambda _reader=None: buttons["value"],
        release_frames=3,
        settle_sec=0.15,
        clock=clock,
    )

    outputs = [filtered()]
    for now, sample in (
        (0.03, (True, False, True, False)),
        (0.06, (True, True, True, False)),
        (0.09, (True, True, True, True)),
        (0.12, (True, False, True, False)),
        (0.15, (False, False, False, False)),
        (0.18, (False, False, False, False)),
        (0.21, (False, False, False, False)),
    ):
        clock.now = now
        buttons["value"] = sample
        outputs.append(filtered())

    assert outputs == [(False, False, False, False)] * len(outputs)


def test_pair_chord_release_bounce_produces_one_rising_edge():
    clock = FakeClock()
    buttons = {"value": (True, True, False, False)}
    filtered = StartChordReleaseFilter(
        lambda _reader=None: buttons["value"],
        release_frames=3,
        settle_sec=0.15,
        clock=clock,
    )
    outputs = [filtered()]
    clock.now = 0.14
    outputs.append(filtered())
    clock.now = 0.16
    outputs.append(filtered())
    buttons["value"] = (False, False, False, False)
    outputs.append(filtered())
    buttons["value"] = (True, True, False, False)  # one-frame release dropout
    outputs.append(filtered())
    buttons["value"] = (False, False, False, False)
    outputs.extend(filtered() for _ in range(3))

    ab = [a and b for a, b, _x, _y in outputs]
    rising_edges = sum(now and not before for before, now in zip([False, *ab[:-1]], ab))

    assert outputs[:2] == [(False, False, False, False)] * 2
    assert rising_edges == 1


def test_recording_grip_is_only_visible_for_record_or_remapped_discard():
    def route(chord):
        router = ControllerInputRouter(
            read_buttons=lambda _reader=None: chord,
            read_inputs=lambda _reader=None: (False, 0.2, 0.3, 0.9, 0.8),
            button_settle_sec=0.0,
        )
        expected = (False, False, False, False) if all(chord) else chord
        assert router.get_abxy_buttons() == expected
        return router.get_controller_inputs()

    assert route((True, False, False, False))[3] == 0.9

    for mode_chord in (
        (True, False, True, False),
        (False, True, False, True),
        (True, True, False, False),
        (False, True, False, False),
        (True, True, True, True),
        (False, False, False, False),
    ):
        inputs = route(mode_chord)
        assert inputs[:3] == (False, 0.2, 0.3)
        assert inputs[3] == 0.0
        assert inputs[4] == 0.8


def test_left_grip_y_remaps_to_discard_without_physical_b():
    router = ControllerInputRouter(
        read_buttons=lambda _reader=None: (False, False, False, True),
        read_inputs=lambda _reader=None: (False, 0.2, 0.3, 0.9, 0.8),
        button_settle_sec=0.0,
    )

    assert router.get_abxy_buttons() == (False, True, False, False)
    assert router.get_controller_inputs()[3] == 0.9


def test_physical_b_cannot_discard_when_remote_vision_owns_b():
    router = ControllerInputRouter(
        read_buttons=lambda _reader=None: (False, True, False, False),
        read_inputs=lambda _reader=None: (False, 0.2, 0.3, 0.9, 0.8),
        button_settle_sec=0.0,
    )

    assert router.get_abxy_buttons() == (False, True, False, False)
    assert router.get_controller_inputs()[3] == 0.0


def test_four_button_chord_with_grip_cannot_record_or_discard():
    router = ControllerInputRouter(
        read_buttons=lambda _reader=None: (True, True, True, True),
        read_inputs=lambda _reader=None: (False, 0.0, 0.0, 1.0, 0.0),
        button_settle_sec=0.0,
    )

    assert router.get_abxy_buttons() == (False, False, False, False)
    assert router.get_controller_inputs()[3] == 0.0


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_right_stick_hold_generates_policy_toggle_only_after_threshold():
    clock = FakeClock()
    clicks = {"value": (False, True)}
    router = ControllerInputRouter(
        read_buttons=lambda _reader=None: (False, False, False, False),
        read_inputs=lambda _reader=None: (False, 0.0, 0.0, 0.0, 0.0),
        read_axis_clicks=lambda _reader=None: clicks["value"],
        policy_hold_sec=0.6,
        clock=clock,
    )

    assert router.get_abxy_buttons() == (False, False, False, False)
    clock.now = 0.59
    assert router.get_abxy_buttons() == (False, False, False, False)
    clock.now = 0.61
    assert router.get_abxy_buttons() == (True, True, True, True)
    clock.now = 1.5
    assert router.get_abxy_buttons() == (True, True, True, True)

    clicks["value"] = (False, False)
    assert router.get_abxy_buttons() == (False, False, False, False)
    clicks["value"] = (False, True)
    clock.now = 2.0
    assert router.get_abxy_buttons() == (False, False, False, False)
    clock.now = 2.61
    assert router.get_abxy_buttons() == (True, True, True, True)


def test_right_stick_policy_toggle_is_not_visible_to_same_thread_planner_read():
    clock = FakeClock()
    router = ControllerInputRouter(
        read_buttons=lambda _reader=None: (False, False, False, False),
        read_inputs=lambda _reader=None: (False, 0.0, 0.0, 0.0, 0.0),
        read_axis_clicks=lambda _reader=None: (False, True),
        policy_hold_sec=0.0,
        clock=clock,
    )
    assert router.get_abxy_buttons() == (True, True, True, True)
    with router.planner_input_scope():
        assert router.get_abxy_buttons() == (False, False, False, False)


def test_physical_locomotion_chords_remain_visible_in_planner_scope():
    buttons = {"value": (True, True, False, False)}
    router = ControllerInputRouter(
        read_buttons=lambda _reader=None: buttons["value"],
        read_inputs=lambda _reader=None: (False, 0.0, 0.0, 0.0, 0.0),
        read_axis_clicks=lambda _reader=None: (False, False),
        button_settle_sec=0.0,
    )

    with router.planner_input_scope():
        assert router.get_abxy_buttons() == (True, True, False, False)


def test_left_stick_click_does_not_toggle_policy():
    clock = FakeClock()
    router = ControllerInputRouter(
        read_buttons=lambda _reader=None: (False, False, False, False),
        read_inputs=lambda _reader=None: (False, 0.0, 0.0, 0.0, 0.0),
        read_axis_clicks=lambda _reader=None: (True, False),
        policy_hold_sec=0.6,
        clock=clock,
    )

    assert router.get_abxy_buttons() == (False, False, False, False)
    clock.now = 1.0
    assert router.get_abxy_buttons() == (False, False, False, False)


def test_collection_launcher_dry_run_uses_explicit_dataset_dir():
    repo = Path(__file__).resolve().parents[3]
    launcher = repo / "real/SONIC/scripts/collect_psi0-sonic-data.sh"
    with tempfile.TemporaryDirectory(prefix="psi0-sonic-launcher-") as tmp:
        dataset = Path(tmp) / "custom-dataset"
        result = subprocess.run(
            [
                "bash",
                str(launcher),
                "start",
                "--dataset-dir",
                str(dataset),
                "--task",
                "launcher dry run",
                "--dry-run",
            ],
            cwd=repo,
            text=True,
            capture_output=True,
            timeout=30,
        )

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Dataset:    {dataset}" in result.stdout
    assert "Full-body POSE" in result.stdout
    assert "next official episode index: 0" in result.stdout
    assert "no robot, tmux, DDS, or camera process was started" in result.stdout


def test_collection_launcher_filters_only_noisy_exporter_camera_latency():
    repo = Path(__file__).resolve().parents[3]
    launcher = repo / "real/SONIC/scripts/collect_psi0-sonic-data.sh"
    script = launcher.read_text(encoding="utf-8")

    assert "'/^Image latency for ego_view:/d'" in script
    assert "'/^\\[Latency\\] Sonic Pose:/d'" in script
    assert "Full-body POSE validation" not in script.split("sed -u", 1)[1].split(" C-m", 1)[0]
    assert "set -o pipefail" in script


def test_pico_view_uses_independent_persistent_tmux_session(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    launcher = repo / "real/SONIC/scripts/collect_psi0-sonic-data.sh"
    fake_prefix = tmp_path / "fake-conda"
    fake_bin = fake_prefix / "bin"
    fake_bin.mkdir(parents=True)
    fake_log = tmp_path / "tmux.log"

    fake_python = fake_bin / "python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)
    fake_tmux = fake_bin / "tmux"
    fake_tmux.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"$*\" >> \"$FAKE_TMUX_LOG\"\n"
        "if [ \"$1\" = has-session ]; then exit \"${FAKE_TMUX_HAS:-1}\"; fi\n"
        "if [ \"$1\" = display-message ]; then echo %9; fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_tmux.chmod(0o755)
    env = {
        **os.environ,
        "PSI_ROOT": str(repo),
        "SONIC_CONDA_PREFIX": str(fake_prefix),
        "SONIC_LOG_DIR": str(tmp_path / "logs"),
        "FAKE_TMUX_LOG": str(fake_log),
    }

    start = subprocess.run(
        ["bash", str(launcher), "view-start"],
        cwd=repo,
        env={**env, "FAKE_TMUX_HAS": "1"},
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert start.returncode == 0, start.stdout + start.stderr
    calls = fake_log.read_text(encoding="utf-8")
    assert "new-session -d -s psi0_pico_ego_view" in calls
    assert "while true" in calls
    assert "manager.exited" not in calls

    stop = subprocess.run(
        ["bash", str(launcher), "view-stop"],
        cwd=repo,
        env={**env, "FAKE_TMUX_HAS": "0"},
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert stop.returncode == 0, stop.stdout + stop.stderr
    assert "kill-session -t psi0_pico_ego_view" in fake_log.read_text(encoding="utf-8")


def test_view_start_recovers_stale_camera_with_discovered_rgb_index(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    launcher = repo / "real/SONIC/scripts/collect_psi0-sonic-data.sh"
    fake_prefix = tmp_path / "fake-conda"
    fake_bin = fake_prefix / "bin"
    fake_bin.mkdir(parents=True)
    python_count = tmp_path / "python-count"
    ssh_log = tmp_path / "ssh.log"
    tmux_log = tmp_path / "tmux.log"

    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "count=0\n"
        "[ ! -f \"$FAKE_PYTHON_COUNT\" ] || count=$(cat \"$FAKE_PYTHON_COUNT\")\n"
        "count=$((count + 1))\n"
        "echo \"$count\" > \"$FAKE_PYTHON_COUNT\"\n"
        "[ \"$count\" -gt 1 ]\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_ssh = fake_bin / "ssh"
    fake_ssh.write_text(
        "#!/usr/bin/env bash\n"
        "[ -z \"${LD_LIBRARY_PATH+x}\" ] || { echo LEAKED_LD_LIBRARY_PATH >&2; exit 3; }\n"
        "echo \"$*\" >> \"$FAKE_SSH_LOG\"\n"
        "case \"$*\" in *video4linux*) echo 5 ;; esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_ssh.chmod(0o755)
    fake_tmux = fake_bin / "tmux"
    fake_tmux.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"$*\" >> \"$FAKE_TMUX_LOG\"\n"
        "[ \"$1\" != has-session ] || exit 0\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_tmux.chmod(0o755)
    env = {
        **os.environ,
        "PSI_ROOT": str(repo),
        "SONIC_CONDA_PREFIX": str(fake_prefix),
        "SONIC_LOG_DIR": str(tmp_path / "logs"),
        "G1_CAMERA_WARMUP_SEC": "0",
        "SYSTEM_SSH": str(fake_ssh),
        "FAKE_PYTHON_COUNT": str(python_count),
        "FAKE_SSH_LOG": str(ssh_log),
        "FAKE_TMUX_LOG": str(tmux_log),
    }

    result = subprocess.run(
        ["bash", str(launcher), "view-start"],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Discovered G1 RealSense RGB node: /dev/video5" in result.stdout
    assert python_count.read_text(encoding="utf-8").strip() == "2"
    ssh_calls = ssh_log.read_text(encoding="utf-8")
    assert "ID_USB_INTERFACE_NUM=03" in ssh_calls
    assert "CAM_IDX='5'" in ssh_calls
    tmux_calls = tmux_log.read_text(encoding="utf-8")
    assert "kill-session -t psi0_pico_ego_view" in tmux_calls


def test_prepare_starts_only_passive_services_and_creates_no_dataset(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    launcher = repo / "real/SONIC/scripts/collect_psi0-sonic-data.sh"
    fake_prefix = tmp_path / "fake-conda"
    fake_bin = fake_prefix / "bin"
    fake_bin.mkdir(parents=True)
    tmux_log = tmp_path / "tmux.log"

    for name, body in {
        "python": "exit 0",
        "ping": "exit 0",
        "pgrep": "exit 0",
        "ip": "echo '2: enp4s0 inet 192.168.123.99/24'; exit 0",
        "tmux": (
            "echo \"$*\" >> \"$FAKE_TMUX_LOG\"; "
            "[ \"$1\" != has-session ] || exit 0; exit 0"
        ),
    }.items():
        path = fake_bin / name
        path.write_text(f"#!/usr/bin/env bash\n{body}\n", encoding="utf-8")
        path.chmod(0o755)

    data_root = tmp_path / "must-not-exist"
    result = subprocess.run(
        ["bash", str(launcher), "prepare"],
        cwd=repo,
        env={
            **os.environ,
            "PSI_ROOT": str(repo),
            "SONIC_CONDA_PREFIX": str(fake_prefix),
            "SONIC_LOG_DIR": str(tmp_path / "logs"),
            "DATASET_DIR": str(data_root),
            "FAKE_TMUX_LOG": str(tmux_log),
        },
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PREPARE READY" in result.stdout
    assert "Proceed with REAL SONIC deployment" not in result.stdout
    assert not data_root.exists()
    assert "has-session -t psi0_pico_ego_view" in tmux_log.read_text(encoding="utf-8")
    script = launcher.read_text(encoding="utf-8")
    prepare_dds = script.split("prepare_dds_preflight()", 1)[1].split(
        "run_prepare()", 1
    )[0]
    assert "ChannelSubscriber" in prepare_dds
    assert "ChannelPublisher" not in prepare_dds


def test_collection_launcher_rejects_non_contiguous_dataset():
    repo = Path(__file__).resolve().parents[3]
    launcher = repo / "real/SONIC/scripts/collect_psi0-sonic-data.sh"
    with tempfile.TemporaryDirectory(prefix="psi0-sonic-launcher-") as tmp:
        dataset = Path(tmp) / "gapped-dataset"
        (dataset / "meta").mkdir(parents=True)
        (dataset / "meta/info.json").write_text(
            json.dumps({"total_episodes": 1, "fps": 30}), encoding="utf-8"
        )
        (dataset / "meta/modality.json").write_text("{}", encoding="utf-8")
        (dataset / "meta/episodes.jsonl").write_text(
            json.dumps({"episode_index": 1, "length": 10}) + "\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                "bash",
                str(launcher),
                "start",
                "--dataset-dir",
                str(dataset),
                "--task",
                "reject gap",
                "--dry-run",
            ],
            cwd=repo,
            text=True,
            capture_output=True,
            timeout=30,
        )

    assert result.returncode != 0
    assert "episode indices are not contiguous" in result.stdout + result.stderr


def test_collection_launcher_accepts_initialized_empty_official_dataset():
    repo = Path(__file__).resolve().parents[3]
    launcher = repo / "real/SONIC/scripts/collect_psi0-sonic-data.sh"
    with tempfile.TemporaryDirectory(prefix="psi0-sonic-launcher-") as tmp:
        dataset = Path(tmp) / "initialized-empty"
        (dataset / "meta").mkdir(parents=True)
        (dataset / "meta/info.json").write_text(
            json.dumps({"total_episodes": 0, "fps": 30}), encoding="utf-8"
        )
        (dataset / "meta/modality.json").write_text("{}", encoding="utf-8")
        result = subprocess.run(
            [
                "bash",
                str(launcher),
                "start",
                "--dataset-dir",
                str(dataset),
                "--task",
                "resume empty",
                "--dry-run",
            ],
            cwd=repo,
            text=True,
            capture_output=True,
            timeout=30,
        )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Existing official episodes: 0" in result.stdout


class FakeDex1Controller:
    def __init__(self, state):
        self.state = state
        self.closed = False
        self.started = False

    def get_current_dual_gripper_q(self):
        return self.state

    def close(self):
        self.closed = True

    def start_publishing(self):
        self.started = True


def test_manager_accepts_finite_dual_dex1_state():
    controller = FakeDex1Controller((1.25, 2.5))

    assert require_dex1_state_pair(controller) == (1.25, 2.5)
    assert not controller.closed
    assert not controller.started


def test_manager_closes_controller_when_either_dex1_state_is_missing():
    for state in ((None, 1.0), (1.0, None), (None, None)):
        controller = FakeDex1Controller(state)
        try:
            require_dex1_state_pair(controller)
        except RuntimeError as exc:
            assert "both left and right" in str(exc)
        else:
            raise AssertionError(f"missing state was accepted: {state}")
        assert controller.closed
        assert not controller.started
