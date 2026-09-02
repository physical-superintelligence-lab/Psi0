"""Validate dense virtual-Dex3 hand semantics in a Psi0 SONIC dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.offline.dex3_to_dex1 import load_from_stats_file


def load_column(dataset_dir: Path, column: str) -> np.ndarray:
    values = []
    for parquet in sorted((dataset_dir / "data").glob("*/*.parquet")):
        df = pd.read_parquet(parquet, columns=[column])
        values.extend(np.asarray(item, dtype=np.float32) for item in df[column])
    if not values:
        raise ValueError(f"no {column} values found")
    return np.stack(values)


def assert_virtual_manifold(
    name: str,
    hands: np.ndarray,
    mapper,
    required_moving_hands: str = "both",
) -> None:
    if hands.shape[1] != 14 or not np.all(np.isfinite(hands)):
        raise AssertionError(f"{name} must be finite (N,14), got {hands.shape}")
    commands = [mapper.hand14_to_command(hand) for hand in hands]
    rebuilt = np.stack([mapper.state_hand14(cmd.left, cmd.right) for cmd in commands])
    np.testing.assert_allclose(hands, rebuilt, atol=1e-4, rtol=0)
    openness = np.asarray([[cmd.left, cmd.right] for cmd in commands])
    moving = np.ptp(openness, axis=0) >= 0.1
    requirements = {
        "both": bool(np.all(moving)),
        "either": bool(np.any(moving)),
        "left": bool(moving[0]),
        "right": bool(moving[1]),
        "none": True,
    }
    if not requirements[required_moving_hands]:
        raise AssertionError(
            f"{name} does not satisfy required-moving-hands={required_moving_hands}; "
            f"openness spans={np.ptp(openness, axis=0).tolist()}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--mapping-stats", required=True)
    parser.add_argument("--stats-path", default=None)
    parser.add_argument(
        "--required-moving-hands",
        choices=["both", "either", "left", "right", "none"],
        default="both",
        help="Which physical gripper(s) must span at least 0.1 openness.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    states = load_column(dataset_dir, "states")
    actions = load_column(dataset_dir, "action")
    if states.shape[1] != 43 or actions.shape[1] != 78:
        raise AssertionError(f"unexpected shapes: states={states.shape} action={actions.shape}")
    mapper = load_from_stats_file(args.mapping_stats)
    assert_virtual_manifold(
        "states", states[:, 29:43], mapper, args.required_moving_hands
    )
    assert_virtual_manifold(
        "action", actions[:, 64:78], mapper, args.required_moving_hands
    )

    stats_path = Path(args.stats_path or dataset_dir / "meta/stats_psi0.json")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if len(stats["states"]["min"]) != 43 or len(stats["action"]["min"]) != 78:
        raise AssertionError("stats dimensions do not match state43/action78")
    print("SONIC Dex1 virtual14 sanity check passed")


if __name__ == "__main__":
    main()
