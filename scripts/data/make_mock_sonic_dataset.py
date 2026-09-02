"""Create a tiny SONIC-style LeRobot dataset for offline Dex1-1 checks.

The generated source dataset mimics the fields consumed by
``raw_sonic_to_psi_lerobot.py``:

* observation.state: 43-D SONIC joint layout
* action.wbc: 43-D SONIC joint layout
* action.motion_token: 64-D token action
* observation.images.ego_view: one copied mp4 per episode
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.offline.dex1_1_layout import (
    LEFT_HAND_SOURCE_SLICE,
    MOTION_TOKEN_DIM,
    RIGHT_HAND_SOURCE_SLICE,
    SOURCE_VECTOR_DIM,
    source_full_hand14,
    source_hand7_from_actuated,
)
from scripts.offline.dex3_to_dex1 import load_from_stats_file

VIDEO_KEY = "observation.images.ego_view"
TELEOP_LEFT_HAND_KEY = "teleop.left_hand_joints"
TELEOP_RIGHT_HAND_KEY = "teleop.right_hand_joints"
DEFAULT_MAPPING_STATS = (
    REPO_ROOT / "real/SONIC/assets/dex1_virtual_mapping_stats.json"
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_video(path: Path, frames: int, width: int = 640, height: int = 480, fps: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size={width}x{height}:rate={fps}",
        "-frames:v",
        str(frames),
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    if shutil.which("ffmpeg"):
        subprocess.run(command, check=True)
        return
    path.write_bytes(b"mock video placeholder\n")


def build_source_vectors(frame_index: int, end_effector: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = np.zeros(SOURCE_VECTOR_DIM, dtype=np.float32)
    wbc = np.zeros(SOURCE_VECTOR_DIM, dtype=np.float32)
    token = np.linspace(0.0, 1.0, MOTION_TOKEN_DIM, dtype=np.float32) + frame_index * 0.01

    state[:SOURCE_VECTOR_DIM] = np.arange(SOURCE_VECTOR_DIM, dtype=np.float32) * 0.001
    wbc[:SOURCE_VECTOR_DIM] = np.arange(SOURCE_VECTOR_DIM, dtype=np.float32) * 0.002

    left_start, _left_end = LEFT_HAND_SOURCE_SLICE
    right_start, _right_end = RIGHT_HAND_SOURCE_SLICE
    if end_effector == "dex1_virtual14":
        mapper = load_from_stats_file(str(DEFAULT_MAPPING_STATS))
        state_hand = mapper.state_hand14(float(frame_index % 2), float((frame_index + 1) % 2))
        action_hand = mapper.state_hand14(float((frame_index + 1) % 2), float(frame_index % 2))
        state[left_start : left_start + 7] = source_hand7_from_actuated(
            state_hand[:7]
        )
        state[right_start : right_start + 7] = source_hand7_from_actuated(
            state_hand[7:]
        )
        wbc[left_start : left_start + 7] = source_hand7_from_actuated(
            action_hand[:7]
        )
        wbc[right_start : right_start + 7] = source_hand7_from_actuated(
            action_hand[7:]
        )
    else:
        state[left_start] = float(frame_index % 2)
        state[right_start] = float((frame_index + 1) % 2)
        wbc[left_start] = float((frame_index + 1) % 2)
        wbc[right_start] = float(frame_index % 2)
    return state, wbc, token


def make_dataset(output_dir: Path, frames: int, end_effector: str = "dex1_1") -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "data" / "chunk-000").mkdir(parents=True)
    (output_dir / "meta").mkdir(parents=True)

    rows = []
    for i in range(frames):
        state, wbc, token = build_source_vectors(i, end_effector)
        row = {
            "observation.state": state.tolist(),
            "action.wbc": wbc.tolist(),
            "action.motion_token": token.tolist(),
            "timestamp": i / 30.0,
            "frame_index": i,
            "episode_index": 0,
            "index": i,
            "task_index": 0,
            "next.done": i == frames - 1,
        }
        if end_effector == "dex1_virtual14":
            hand_target = source_full_hand14(wbc)
            row[TELEOP_LEFT_HAND_KEY] = hand_target[:7].tolist()
            row[TELEOP_RIGHT_HAND_KEY] = hand_target[7:].tolist()
        rows.append(row)
    pd.DataFrame(rows).to_parquet(output_dir / "data" / "chunk-000" / "episode_000000.parquet")
    write_video(output_dir / "videos" / "chunk-000" / VIDEO_KEY / "episode_000000.mp4", frames)

    write_jsonl(output_dir / "meta" / "tasks.jsonl", [{"task_index": 0, "task": "mock dex1-1 grasp"}])
    write_jsonl(
        output_dir / "meta" / "episodes.jsonl",
        [{"episode_index": 0, "tasks": ["mock dex1-1 grasp"], "length": frames}],
    )
    modality = {
        "end_effector": end_effector,
        "observation": {
            "state": {
                "start": 0,
                "end": SOURCE_VECTOR_DIM,
                "original_key": "observation.state",
            }
        },
        "action": {
            "wbc": {
                "start": 0,
                "end": SOURCE_VECTOR_DIM,
                "original_key": "action.wbc",
            },
            "motion_token": {
                "start": 0,
                "end": MOTION_TOKEN_DIM,
                "original_key": "action.motion_token",
            },
        },
        "video": {
            "ego_view": {"original_key": VIDEO_KEY},
        },
    }
    if end_effector == "dex1_virtual14":
        modality["teleop"] = {
            "left_hand_joints": {
                "start": 0,
                "end": 7,
                "original_key": TELEOP_LEFT_HAND_KEY,
            },
            "right_hand_joints": {
                "start": 0,
                "end": 7,
                "original_key": TELEOP_RIGHT_HAND_KEY,
            },
        }
    (output_dir / "meta" / "modality.json").write_text(json.dumps(modality, indent=4), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/mock_sonic_source")
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument(
        "--end-effector",
        choices=["dex1_1", "dex1_virtual14"],
        default="dex1_1",
    )
    args = parser.parse_args()
    make_dataset(
        Path(args.output_dir).expanduser().resolve(), args.frames, args.end_effector
    )
    print(f"Wrote mock SONIC source dataset to {Path(args.output_dir).expanduser().resolve()}")


if __name__ == "__main__":
    main()
