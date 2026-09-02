import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data.preflight_sonic_dex1_1_dataset import run_preflight
from scripts.offline.dex1_1_layout import MOTION_TOKEN_DIM, SOURCE_VECTOR_DIM


class SonicDex11PreflightTests(unittest.TestCase):
    def _write_source(self, root: Path, *, include_motion_token: bool) -> None:
        (root / "data" / "chunk-000").mkdir(parents=True)
        (root / "videos" / "chunk-000" / "observation.images.ego_view").mkdir(parents=True)
        (root / "meta").mkdir(parents=True)

        row = {
            "observation.state": np.zeros(SOURCE_VECTOR_DIM, dtype=np.float32).tolist(),
            "action.wbc": np.zeros(SOURCE_VECTOR_DIM, dtype=np.float32).tolist(),
        }
        if include_motion_token:
            row["action.motion_token"] = np.zeros(MOTION_TOKEN_DIM, dtype=np.float32).tolist()
        pd.DataFrame([row]).to_parquet(root / "data" / "chunk-000" / "episode_000000.parquet")
        (root / "videos" / "chunk-000" / "observation.images.ego_view" / "episode_000000.mp4").write_bytes(
            b"placeholder"
        )
        (root / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "mock"}) + "\n")
        (root / "meta" / "episodes.jsonl").write_text(
            json.dumps({"episode_index": 0, "tasks": ["mock"], "length": 1}) + "\n"
        )
        modality = {
            "end_effector": "dex1_1",
            "observation": {
                "state": {"start": 0, "end": SOURCE_VECTOR_DIM, "original_key": "observation.state"}
            },
            "action": {
                "wbc": {"start": 0, "end": SOURCE_VECTOR_DIM, "original_key": "action.wbc"},
                "motion_token": {
                    "start": 0,
                    "end": MOTION_TOKEN_DIM,
                    "original_key": "action.motion_token",
                },
            },
        }
        (root / "meta" / "modality.json").write_text(json.dumps(modality))

    def test_missing_motion_token_blocks_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_source(root, include_motion_token=False)
            report = run_preflight(root, chunks_size=1000, max_episodes=1)
        self.assertFalse(report["can_convert"])
        self.assertTrue(any("missing column: action.motion_token" in error for error in report["errors"]))

    def test_valid_schema_can_convert(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_source(root, include_motion_token=True)
            report = run_preflight(root, chunks_size=1000, max_episodes=1)
        self.assertTrue(report["can_convert"])
        self.assertEqual(report["episode_count"], 1)
        self.assertEqual(report["frame_count"], 1)

    def test_official_feature_metadata_can_replace_full_vector_original_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_source(root, include_motion_token=True)
            modality_path = root / "meta" / "modality.json"
            modality = json.loads(modality_path.read_text())
            modality["end_effector"] = "dex1_virtual14"
            modality["observation"]["state"].pop("original_key")
            modality["action"]["wbc"].pop("original_key")
            modality_path.write_text(json.dumps(modality))
            (root / "meta" / "info.json").write_text(
                json.dumps(
                    {
                        "features": {
                            "observation.state": {"shape": [SOURCE_VECTOR_DIM]},
                            "action.wbc": {"shape": [SOURCE_VECTOR_DIM]},
                            "action.motion_token": {"shape": [MOTION_TOKEN_DIM]},
                        },
                        "script_config": {"end_effector": "dex1_virtual14"},
                    }
                )
            )
            report = run_preflight(
                root,
                chunks_size=1000,
                max_episodes=1,
                hand_layout="dex1_virtual14",
            )
        self.assertTrue(report["can_convert"], report["errors"])
        self.assertTrue(all(report["metadata"]["required_schema_keys"].values()))


if __name__ == "__main__":
    unittest.main()
