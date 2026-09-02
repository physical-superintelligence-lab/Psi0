"""Compare real SONIC POSE and VR_3PT recordings.

The exporter writes the same LeRobot schema for both modes.  This tool keeps
the comparison on the raw recordings so that stream-mode, SMPL, planner, and
VR 3-point fields are still available.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


STATE_COLUMN = "observation.state"
WBC_COLUMN = "action.wbc"
STREAM_MODE_COLUMN = "teleop.stream_mode"
SMPL_COLUMN = "teleop.smpl_pose"
VR_POSITION_COLUMN = "teleop.vr_3pt_position"
VR_ORIENTATION_COLUMN = "teleop.vr_3pt_orientation"
PLANNER_MOVEMENT_COLUMN = "teleop.planner_movement"
PROJECTED_GRAVITY_COLUMN = "observation.projected_gravity"

STREAM_MODE_NAMES = {
    0: "OFF",
    1: "POSE",
    2: "PLANNER",
    3: "PLANNER_FROZEN_UPPER_BODY",
    4: "POSE_PAUSE",
    5: "VR_3PT",
}

# Official SONIC 43-D whole-body layout.
JOINT_GROUPS = {
    "legs": tuple(range(0, 12)),
    "waist": tuple(range(12, 15)),
    "arms": tuple(range(15, 22)) + tuple(range(29, 36)),
}


def _stack_vector_column(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        raise ValueError(f"missing required column: {column}")
    rows = [np.asarray(value, dtype=np.float64).reshape(-1) for value in df[column]]
    if not rows:
        raise ValueError(f"empty column: {column}")
    width = rows[0].shape[0]
    if any(row.shape != (width,) for row in rows):
        raise ValueError(f"inconsistent vector width in {column}")
    values = np.vstack(rows)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite values in {column}")
    return values


def _optional_vectors(df: pd.DataFrame, column: str) -> np.ndarray | None:
    if column not in df.columns:
        return None
    return _stack_vector_column(df, column)


def _activity_fraction(values: np.ndarray | None, epsilon: float = 1e-6) -> float | None:
    if values is None:
        return None
    return float(np.mean(np.linalg.norm(values, axis=1) > epsilon))


def _joint_metrics(values: np.ndarray, indices: Iterable[int]) -> dict[str, float]:
    selected = values[:, tuple(indices)]
    joint_ranges = np.ptp(selected, axis=0)
    diffs = np.diff(selected, axis=0)
    if len(diffs):
        rms_step = float(np.sqrt(np.mean(np.square(diffs))))
        max_abs_step = float(np.max(np.abs(diffs)))
    else:
        rms_step = 0.0
        max_abs_step = 0.0
    return {
        "mean_joint_range_rad": float(np.mean(joint_ranges)),
        "max_joint_range_rad": float(np.max(joint_ranges)),
        "rms_step_rad": rms_step,
        "max_abs_step_rad": max_abs_step,
    }


def _episode_index(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"cannot parse episode index from {path.name}") from exc


def _analyze_episode(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"empty episode: {path}")

    state = _stack_vector_column(df, STATE_COLUMN)
    wbc = _stack_vector_column(df, WBC_COLUMN)
    if state.shape[1] != 43 or wbc.shape[1] != 43:
        raise ValueError(
            f"expected 43-D state/action in {path}, got {state.shape[1]}/{wbc.shape[1]}"
        )

    if STREAM_MODE_COLUMN not in df.columns:
        raise ValueError(f"missing required column: {STREAM_MODE_COLUMN}")
    stream_modes = np.asarray(df[STREAM_MODE_COLUMN], dtype=np.int64).reshape(-1)
    mode_counts = Counter(int(value) for value in stream_modes)

    timestamps = np.asarray(df["timestamp"], dtype=np.float64) if "timestamp" in df else None
    duration = (
        float(max(0.0, timestamps[-1] - timestamps[0]))
        if timestamps is not None and len(timestamps) > 1
        else 0.0
    )

    smpl = _optional_vectors(df, SMPL_COLUMN)
    vr_position = _optional_vectors(df, VR_POSITION_COLUMN)
    vr_orientation = _optional_vectors(df, VR_ORIENTATION_COLUMN)
    planner_movement = _optional_vectors(df, PLANNER_MOVEMENT_COLUMN)
    gravity = _optional_vectors(df, PROJECTED_GRAVITY_COLUMN)

    gravity_horizontal = None
    if gravity is not None:
        if gravity.shape[1] != 3:
            raise ValueError(f"{PROJECTED_GRAVITY_COLUMN} must be 3-D")
        gravity_horizontal = np.linalg.norm(gravity[:, :2], axis=1)

    report = {
        "episode_index": _episode_index(path),
        "frames": int(len(df)),
        "duration_sec": duration,
        "stream_mode_counts": {
            STREAM_MODE_NAMES.get(mode, f"UNKNOWN_{mode}"): count
            for mode, count in sorted(mode_counts.items())
        },
        "stream_mode_fractions": {
            STREAM_MODE_NAMES.get(mode, f"UNKNOWN_{mode}"): float(count / len(df))
            for mode, count in sorted(mode_counts.items())
        },
        "state": {name: _joint_metrics(state, indices) for name, indices in JOINT_GROUPS.items()},
        "action_wbc": {name: _joint_metrics(wbc, indices) for name, indices in JOINT_GROUPS.items()},
        "smpl_active_fraction": _activity_fraction(smpl),
        "vr_3pt_position_active_fraction": _activity_fraction(vr_position),
        "vr_3pt_orientation_active_fraction": _activity_fraction(vr_orientation),
        "planner_movement_active_fraction": _activity_fraction(planner_movement, epsilon=1e-4),
        "projected_gravity_horizontal_rms": (
            float(np.sqrt(np.mean(np.square(gravity_horizontal))))
            if gravity_horizontal is not None
            else None
        ),
        "projected_gravity_horizontal_max": (
            float(np.max(gravity_horizontal)) if gravity_horizontal is not None else None
        ),
    }
    raw = {
        "state": state,
        "wbc": wbc,
        "stream_modes": stream_modes,
        "smpl": smpl,
        "vr_position": vr_position,
        "vr_orientation": vr_orientation,
        "planner_movement": planner_movement,
        "gravity_horizontal": gravity_horizontal,
        "duration_sec": duration,
    }
    return report, raw


def _concat_present(items: list[np.ndarray | None]) -> np.ndarray | None:
    present = [item for item in items if item is not None]
    return np.concatenate(present, axis=0) if present else None


def _aggregate_joint_metrics(
    raw_episodes: list[dict[str, Any]], key: str, indices: Iterable[int]
) -> dict[str, float]:
    selected = [episode[key][:, tuple(indices)] for episode in raw_episodes]
    joint_ranges = np.ptp(np.concatenate(selected, axis=0), axis=0)
    episode_diffs = [np.diff(values, axis=0) for values in selected if len(values) > 1]
    if episode_diffs:
        diffs = np.concatenate(episode_diffs, axis=0)
        rms_step = float(np.sqrt(np.mean(np.square(diffs))))
        max_abs_step = float(np.max(np.abs(diffs)))
    else:
        rms_step = 0.0
        max_abs_step = 0.0
    return {
        "mean_joint_range_rad": float(np.mean(joint_ranges)),
        "max_joint_range_rad": float(np.max(joint_ranges)),
        "rms_step_rad": rms_step,
        "max_abs_step_rad": max_abs_step,
    }


def analyze_dataset(dataset_dir: Path, expected_mode: int) -> dict[str, Any]:
    dataset_dir = dataset_dir.expanduser().resolve()
    parquets = sorted((dataset_dir / "data").rglob("episode_*.parquet"))
    if not parquets:
        raise ValueError(f"no episode parquet files under {dataset_dir / 'data'}")

    episodes: list[dict[str, Any]] = []
    raw_episodes: list[dict[str, Any]] = []
    for parquet in parquets:
        report, raw = _analyze_episode(parquet)
        episodes.append(report)
        raw_episodes.append(raw)

    stream_modes = np.concatenate([item["stream_modes"] for item in raw_episodes])
    total_frames = int(len(stream_modes))
    mode_counts = Counter(int(value) for value in stream_modes)

    def combine_activity(key: str, epsilon: float = 1e-6) -> float | None:
        return _activity_fraction(_concat_present([item[key] for item in raw_episodes]), epsilon)

    gravity_horizontal = _concat_present(
        [item["gravity_horizontal"] for item in raw_episodes]
    )
    errors: list[str] = []
    expected_count = mode_counts.get(expected_mode, 0)
    if expected_count == 0:
        errors.append(
            f"expected {STREAM_MODE_NAMES[expected_mode]} frames, but none were recorded"
        )
    if expected_mode == 1 and (combine_activity("smpl") or 0.0) == 0.0:
        errors.append("POSE dataset has no active teleop.smpl_pose frames")
    if expected_mode == 5 and (combine_activity("vr_position") or 0.0) == 0.0:
        errors.append("VR_3PT dataset has no active teleop.vr_3pt_position frames")

    return {
        "dataset_dir": str(dataset_dir),
        "expected_mode": STREAM_MODE_NAMES[expected_mode],
        "valid_for_comparison": not errors,
        "errors": errors,
        "episode_count": len(episodes),
        "frames": total_frames,
        "duration_sec": float(sum(item["duration_sec"] for item in raw_episodes)),
        "stream_mode_counts": {
            STREAM_MODE_NAMES.get(mode, f"UNKNOWN_{mode}"): count
            for mode, count in sorted(mode_counts.items())
        },
        "stream_mode_fractions": {
            STREAM_MODE_NAMES.get(mode, f"UNKNOWN_{mode}"): float(count / total_frames)
            for mode, count in sorted(mode_counts.items())
        },
        "state": {
            name: _aggregate_joint_metrics(raw_episodes, "state", indices)
            for name, indices in JOINT_GROUPS.items()
        },
        "action_wbc": {
            name: _aggregate_joint_metrics(raw_episodes, "wbc", indices)
            for name, indices in JOINT_GROUPS.items()
        },
        "smpl_active_fraction": combine_activity("smpl"),
        "vr_3pt_position_active_fraction": combine_activity("vr_position"),
        "vr_3pt_orientation_active_fraction": combine_activity("vr_orientation"),
        "planner_movement_active_fraction": combine_activity(
            "planner_movement", epsilon=1e-4
        ),
        "projected_gravity_horizontal_rms": (
            float(np.sqrt(np.mean(np.square(gravity_horizontal))))
            if gravity_horizontal is not None
            else None
        ),
        "projected_gravity_horizontal_max": (
            float(np.max(gravity_horizontal)) if gravity_horizontal is not None else None
        ),
        "episodes": episodes,
    }


def build_comparison(pose_dir: Path, vr3pt_dir: Path) -> dict[str, Any]:
    pose = analyze_dataset(pose_dir, expected_mode=1)
    vr3pt = analyze_dataset(vr3pt_dir, expected_mode=5)
    deltas: dict[str, dict[str, float]] = {}
    for source in ("state", "action_wbc"):
        for group in JOINT_GROUPS:
            key = f"{source}.{group}"
            deltas[key] = {
                metric: float(vr3pt[source][group][metric] - pose[source][group][metric])
                for metric in (
                    "mean_joint_range_rad",
                    "max_joint_range_rad",
                    "rms_step_rad",
                    "max_abs_step_rad",
                )
            }
    return {
        "pose": pose,
        "vr3pt": vr3pt,
        "vr3pt_minus_pose": deltas,
        "valid_for_comparison": pose["valid_for_comparison"] and vr3pt["valid_for_comparison"],
    }


def _print_dataset(name: str, report: dict[str, Any]) -> None:
    print(
        f"{name}: episodes={report['episode_count']} frames={report['frames']} "
        f"duration={report['duration_sec']:.1f}s modes={report['stream_mode_counts']}"
    )
    for source in ("state", "action_wbc"):
        for group in JOINT_GROUPS:
            metric = report[source][group]
            print(
                f"  {source}.{group}: range(mean/max)="
                f"{metric['mean_joint_range_rad']:.4f}/{metric['max_joint_range_rad']:.4f} rad "
                f"step(rms/max)={metric['rms_step_rad']:.5f}/{metric['max_abs_step_rad']:.5f} rad"
            )
    print(
        "  activity: "
        f"smpl={report['smpl_active_fraction']} "
        f"vr3pt={report['vr_3pt_position_active_fraction']} "
        f"planner_move={report['planner_movement_active_fraction']}"
    )
    print(
        "  projected gravity horizontal: "
        f"rms={report['projected_gravity_horizontal_rms']} "
        f"max={report['projected_gravity_horizontal_max']}"
    )
    for error in report["errors"]:
        print(f"  ERROR: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument("--vr3pt-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    comparison = build_comparison(args.pose_dir, args.vr3pt_dir)
    _print_dataset("POSE", comparison["pose"])
    _print_dataset("VR_3PT", comparison["vr3pt"])
    if args.output_json:
        output = args.output_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        print(f"JSON report: {output}")
    raise SystemExit(0 if comparison["valid_for_comparison"] else 2)


if __name__ == "__main__":
    main()
