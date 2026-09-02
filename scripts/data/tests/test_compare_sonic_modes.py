from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data.compare_sonic_modes import build_comparison


def _write_episode(root: Path, mode: int) -> None:
    rows = []
    for index in range(6):
        state = np.zeros(43, dtype=np.float32)
        state[:12] = index * (0.02 if mode == 1 else 0.01)
        wbc = state * 2.0
        rows.append(
            {
                "observation.state": state,
                "action.wbc": wbc,
                "observation.projected_gravity": np.array(
                    [0.01 * index, 0.0, -1.0], dtype=np.float32
                ),
                "teleop.stream_mode": mode,
                "teleop.smpl_pose": np.ones(63, dtype=np.float32) if mode == 1 else np.zeros(63),
                "teleop.vr_3pt_position": np.ones(9, dtype=np.float32) if mode == 5 else np.zeros(9),
                "teleop.vr_3pt_orientation": np.ones(18, dtype=np.float32) if mode == 5 else np.zeros(18),
                "teleop.planner_movement": np.array([0.1, 0.0, 0.0]) if mode == 5 else np.zeros(3),
                "timestamp": index / 30.0,
            }
        )
    path = root / "data" / "chunk-000" / "episode_000000.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(path)


def test_build_comparison_validates_modes_and_reports_delta(tmp_path: Path) -> None:
    pose = tmp_path / "pose"
    vr3pt = tmp_path / "vr3pt"
    _write_episode(pose, mode=1)
    _write_episode(vr3pt, mode=5)

    result = build_comparison(pose, vr3pt)

    assert result["valid_for_comparison"] is True
    assert result["pose"]["stream_mode_counts"] == {"POSE": 6}
    assert result["vr3pt"]["stream_mode_counts"] == {"VR_3PT": 6}
    assert result["pose"]["smpl_active_fraction"] == 1.0
    assert result["vr3pt"]["vr_3pt_position_active_fraction"] == 1.0
    assert result["vr3pt_minus_pose"]["state.legs"]["max_joint_range_rad"] < 0.0
