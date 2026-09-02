#!/usr/bin/env python3
"""Run SONIC's official exporter with physical Dex1 state semantics."""

from __future__ import annotations

import os
import sys
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np


PSI_ROOT = Path(__file__).resolve().parents[2]
SONIC_DIR = Path(
    os.environ.get("SONIC_DIR", PSI_ROOT / "third_party/GR00T-WholeBodyControl")
).resolve()
for path in (PSI_ROOT, SONIC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from real.SONIC.dex1_virtual_runtime import (  # noqa: E402
    DEFAULT_STATS,
    Dex1StateReader,
    Dex1VirtualMapper,
    patch_sonic_proprio_hands,
)


POSE_STREAM_MODES = frozenset((1, 4))
PLANNER_STREAM_MODE = 2
FULL_BODY_ALLOWED_STREAM_MODES = frozenset((*POSE_STREAM_MODES, PLANNER_STREAM_MODE))
STREAM_MODE_NAMES = {
    0: "OFF",
    1: "POSE",
    2: "PLANNER",
    3: "PLANNER_FROZEN_UPPER_BODY",
    4: "POSE_PAUSE",
    5: "VR_3PT",
}

POSE_REQUIRED_VECTORS = {
    "observation.state": 43,
    "action.wbc": 43,
    "action.motion_token": 64,
    "teleop.smpl_pose": 63,
}

EMPTY_DATASET_METADATA_FILES = (
    "tasks.jsonl",
    "episodes.jsonl",
    "episodes_stats.jsonl",
)


def repair_initialized_empty_dataset(dataset_dir: Path) -> list[Path]:
    """Make an interrupted zero-episode SONIC dataset locally resumable.

    LeRobot v2.1 expects three JSONL files even when they contain no rows. The
    official exporter creates them on the first save, so an interruption after
    initial metadata creation otherwise falls through to a Hugging Face pull.
    Existing files and every non-empty dataset are left untouched.
    """

    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        return []
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if int(info.get("total_episodes", -1)) != 0 or int(info.get("total_frames", -1)) != 0:
        return []

    created: list[Path] = []
    for name in EMPTY_DATASET_METADATA_FILES:
        path = dataset_dir / "meta" / name
        if not path.exists():
            path.touch()
            created.append(path)
    return created


def _stack_episode_vectors(
    episode_buffer: dict[str, Any], column: str, expected_width: int
) -> tuple[np.ndarray | None, str | None]:
    values = episode_buffer.get(column)
    if values is None:
        return None, f"missing {column}"
    try:
        rows = [np.asarray(value, dtype=np.float64).reshape(-1) for value in values]
    except Exception as exc:
        return None, f"cannot read {column}: {exc}"
    if not rows:
        return None, f"empty {column}"
    if any(row.shape != (expected_width,) for row in rows):
        shapes = sorted({tuple(row.shape) for row in rows})
        return None, f"{column} width mismatch: expected {expected_width}, got {shapes}"
    stacked = np.vstack(rows)
    if not np.all(np.isfinite(stacked)):
        return None, f"{column} contains NaN/Inf"
    return stacked, None


def validate_full_body_pose_episode(episode_buffer: dict[str, Any]) -> dict[str, Any]:
    """Validate an unsaved official exporter buffer for full-body POSE collection.

    Planner frames are allowed only as lead-in/lead-out transitions. Partial
    stale SMPL is reported but left for the official dataset processor.
    """

    errors: list[str] = []
    warnings: list[str] = []
    try:
        declared_frames = int(episode_buffer.get("size", 0))
    except (TypeError, ValueError):
        declared_frames = 0
    if declared_frames <= 0:
        errors.append("episode buffer is empty")

    vectors: dict[str, np.ndarray] = {}
    for column, width in POSE_REQUIRED_VECTORS.items():
        values, error = _stack_episode_vectors(episode_buffer, column, width)
        if error:
            errors.append(error)
        elif values is not None:
            vectors[column] = values

    modes_raw = episode_buffer.get("teleop.stream_mode")
    modes: np.ndarray | None = None
    if modes_raw is None:
        errors.append("missing teleop.stream_mode")
    else:
        try:
            mode_rows = [np.asarray(value).reshape(-1) for value in modes_raw]
            if not mode_rows or any(row.size != 1 for row in mode_rows):
                errors.append("teleop.stream_mode must contain one scalar per frame")
            else:
                modes = np.asarray([int(row[0]) for row in mode_rows], dtype=np.int64)
        except Exception as exc:
            errors.append(f"cannot read teleop.stream_mode: {exc}")

    observed_lengths = [len(values) for values in vectors.values()]
    if modes is not None:
        observed_lengths.append(len(modes))
    if observed_lengths:
        unique_lengths = sorted(set(observed_lengths))
        if len(unique_lengths) != 1:
            errors.append(f"episode columns have inconsistent lengths: {unique_lengths}")
        elif declared_frames > 0 and unique_lengths[0] != declared_frames:
            errors.append(
                f"episode size={declared_frames} but columns contain {unique_lengths[0]} frames"
            )

    mode_counts: dict[str, int] = {}
    pose_mask: np.ndarray | None = None
    if modes is not None and len(modes):
        counts = Counter(int(mode) for mode in modes)
        mode_counts = {
            STREAM_MODE_NAMES.get(mode, f"UNKNOWN_{mode}"): count
            for mode, count in sorted(counts.items())
        }
        unsupported = sorted(set(counts) - FULL_BODY_ALLOWED_STREAM_MODES)
        if unsupported:
            names = [STREAM_MODE_NAMES.get(mode, f"UNKNOWN_{mode}") for mode in unsupported]
            errors.append("non-POSE stream mode recorded: " + ", ".join(names))

        pose_mask = np.isin(modes, tuple(POSE_STREAM_MODES))
        if not np.any(pose_mask):
            errors.append("episode contains no POSE/POSE_PAUSE frames")

        family = ["pose" if mode in POSE_STREAM_MODES else "planner" for mode in modes]
        collapsed = [family[0]]
        for item in family[1:]:
            if item != collapsed[-1]:
                collapsed.append(item)
        if any(
            item == "planner" and index not in (0, len(collapsed) - 1)
            for index, item in enumerate(collapsed)
        ):
            errors.append("PLANNER frames occur inside the POSE trajectory")

    smpl_valid_fraction: float | None = None
    smpl = vectors.get("teleop.smpl_pose")
    if (
        smpl is not None
        and pose_mask is not None
        and len(smpl) == len(pose_mask)
        and np.any(pose_mask)
    ):
        pose_smpl_valid = np.linalg.norm(smpl[pose_mask], axis=1) > 1e-6
        smpl_valid_fraction = float(np.mean(pose_smpl_valid))
        if not np.any(pose_smpl_valid):
            errors.append("all POSE SMPL frames are zero/stale")
        elif smpl_valid_fraction < 1.0:
            warnings.append(
                f"POSE SMPL valid fraction={smpl_valid_fraction:.3f}; "
                "official processing will remove stale frames"
            )

    gripper_ranges: dict[str, float] = {}
    for column in ("observation.state", "action.wbc"):
        values = vectors.get(column)
        if values is None:
            continue
        gripper_ranges[f"{column}.left_hand"] = float(np.max(np.ptp(values[:, 22:29], axis=0)))
        gripper_ranges[f"{column}.right_hand"] = float(np.max(np.ptp(values[:, 36:43], axis=0)))

    return {
        "valid": not errors,
        "frames": declared_frames,
        "stream_mode_counts": mode_counts,
        "smpl_valid_fraction": smpl_valid_fraction,
        "gripper_ranges": gripper_ranges,
        "errors": errors,
        "warnings": warnings,
    }


def format_pose_episode_validation(report: dict[str, Any]) -> str:
    status = "PASS" if report["valid"] else "DISCARD"
    details = [
        f"status={status}",
        f"frames={report['frames']}",
        f"modes={report['stream_mode_counts']}",
        f"smpl_valid={report['smpl_valid_fraction']}",
        f"gripper_ranges={report['gripper_ranges']}",
    ]
    messages = [*report["errors"], *report["warnings"]]
    if messages:
        details.append("messages=" + " | ".join(messages))
    return "Full-body POSE validation: " + "; ".join(details)


def select_virtual_hand_targets(
    stream_mode: int,
    sonic_msg: dict | None,
    planner_msg: dict | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Select the same hand target source as SONIC's official exporter."""
    source = sonic_msg if stream_mode in POSE_STREAM_MODES else planner_msg
    if not isinstance(source, dict):
        return None
    left = np.asarray(source.get("left_hand_joints"), dtype=np.float64)
    right = np.asarray(source.get("right_hand_joints"), dtype=np.float64)
    if left.shape != (7,) or right.shape != (7,):
        return None
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return None
    return left.copy(), right.copy()


def main() -> None:
    import tyro
    from gear_sonic.scripts import run_data_exporter as official_exporter

    config = tyro.cli(official_exporter.SonicDataExporterConfig)
    if config.dataset_name is None:
        config.dataset_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    dataset_dir = Path(config.root_output_dir).expanduser().resolve() / config.dataset_name
    repaired = repair_initialized_empty_dataset(dataset_dir)
    if repaired:
        print(
            "[Psi0 Dataset] repaired interrupted zero-episode metadata: "
            + ", ".join(path.name for path in repaired)
        )

    mapper = Dex1VirtualMapper(os.environ.get("DEX1_VIRTUAL_STATS", str(DEFAULT_STATS)))
    network = os.environ.get("G1_NETWORK_INTERFACE", "enp4s0")
    state_reader = Dex1StateReader(network=network)
    left_q, right_q = state_reader.wait(timeout=5.0)

    official_collector = official_exporter.GrootDataCollector
    official_poll_robot_config = official_exporter.poll_robot_config_zmq

    def poll_robot_config_with_dex1_marker(*args, **kwargs):
        config = dict(official_poll_robot_config(*args, **kwargs))
        config["end_effector"] = "dex1_virtual14"
        config["dex1_virtual_stats"] = str(mapper.stats_path)
        return config

    official_exporter.poll_robot_config_zmq = poll_robot_config_with_dex1_marker

    class Dex1VirtualCollector(official_collector):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            left_open = mapper.q_to_openness(left_q)
            right_open = mapper.q_to_openness(right_q)
            self._dex1_last_left, self._dex1_last_right = mapper.hand7_pair(left_open, right_open)

        def _check_recording_commands(self):
            key = self._keyboard_listener.read_msg()

            if self._manager_toggle_da:
                key = "x"
                self._manager_toggle_da = False
            elif self._manager_toggle_dc:
                key = "c"
                self._manager_toggle_dc = False

            if key == "c":
                state = self._episode_state.get_state()
                if state == self._episode_state.RECORDING:
                    report = validate_full_body_pose_episode(self.data_exporter.episode_buffer)
                    self._print_and_say(format_pose_episode_validation(report), say=False)
                    if not report["valid"]:
                        episode_index = self.current_episode_index
                        self.data_exporter.save_episode_as_discarded()
                        self._episode_state.reset_state()
                        self._initial_yaw = None
                        self._print_and_say(
                            f"Episode {episode_index} saved and marked discarded; "
                            "re-record this demonstration in full-body POSE",
                            blocking=False,
                        )
                        return

                self._episode_state.change_state()
                state = self._episode_state.get_state()
                if state == self._episode_state.RECORDING:
                    self._initial_yaw = None
                    self._print_and_say(
                        f"Started recording {self.current_episode_index}", blocking=False
                    )
                elif state == self._episode_state.NEED_TO_SAVE:
                    self._print_and_say(
                        "Stopping recording, preparing to save", blocking=False
                    )
                elif state == self._episode_state.IDLE:
                    self._print_and_say(
                        "Saved episode and back to idle state", blocking=False
                    )
            elif key == "x" and self._episode_state.get_state() == self._episode_state.RECORDING:
                self.data_exporter.save_episode_as_discarded()
                self._episode_state.reset_state()
                self._initial_yaw = None
                self._print_and_say("Discarded episode", blocking=False)

        def _add_data_frame_sonic(self, t_start: float) -> bool:
            current_left_q, current_right_q = state_reader.get_q()
            targets = select_virtual_hand_targets(
                self.current_stream_mode,
                self.latest_sonic_msg,
                self.latest_planner_msg,
            )
            if targets is not None:
                self._dex1_last_left, self._dex1_last_right = targets

            original = self.latest_proprio_msg
            if original is None:
                return super()._add_data_frame_sonic(t_start)
            patched = patch_sonic_proprio_hands(
                original,
                left_q=float(current_left_q),
                right_q=float(current_right_q),
                left_action7=self._dex1_last_left,
                right_action7=self._dex1_last_right,
                mapper=mapper,
            )
            self.latest_proprio_msg = patched
            try:
                return super()._add_data_frame_sonic(t_start)
            finally:
                self.latest_proprio_msg = original

    official_exporter.GrootDataCollector = Dex1VirtualCollector
    print(
        f"[Psi0 Dex1] exporter virtual14 enabled: stats={mapper.stats_path} "
        f"network={network}"
    )
    try:
        official_exporter.main(config)
    finally:
        state_reader.close()


if __name__ == "__main__":
    main()
