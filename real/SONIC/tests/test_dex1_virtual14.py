from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real.SONIC.dex1_virtual_runtime import (
    Dex1VirtualMapper,
    Dex1VirtualTeleopBridge,
    patch_sonic_proprio_hands,
)
from real.SONIC.run_data_exporter_dex1 import (
    repair_initialized_empty_dataset,
    select_virtual_hand_targets,
    validate_full_body_pose_episode,
)
from scripts.data.make_mock_sonic_dataset import make_dataset
from scripts.data.raw_sonic_to_psi_lerobot import (
    END_EFFECTOR_TO_HAND_LAYOUT,
    SRC_TELEOP_LEFT_HAND,
    SRC_TELEOP_RIGHT_HAND,
    Sonic2LeRobotConverter,
)
from scripts.data.sanity_check_sonic_dex1_virtual14 import assert_virtual_manifold
from scripts.offline.dex1_1_layout import (
    MOTION_TOKEN_DIM,
    pack_psi_state,
    source_hand7_from_actuated,
)


class FakeController:
    def __init__(self):
        self.target = None
        self.started = False
        self.closed = False

    def get_current_dual_gripper_q(self):
        return 2.75, 2.75

    def start_publishing(self):
        self.started = True

    def ctrl_dual_gripper(self, left, right):
        self.target = (left, right)

    def close(self):
        self.closed = True


class Dex1Virtual14Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapper = Dex1VirtualMapper()

    def test_openness_roundtrip(self):
        for left in np.linspace(0.0, 1.0, 5):
            for right in np.linspace(0.0, 1.0, 5):
                hand = self.mapper.hand14(float(left), float(right))
                command = self.mapper.command_from_hand14(hand)
                self.assertAlmostEqual(command.left, left, places=5)
                self.assertAlmostEqual(command.right, right, places=5)

    def test_virtual_hands_are_dense_and_converter_preserves_them(self):
        left, right = self.mapper.hand7_pair(0.2, 0.8)
        self.assertGreater(np.count_nonzero(np.abs(left) > 1e-6), 1)
        self.assertGreater(np.count_nonzero(np.abs(right) > 1e-6), 1)
        source = np.zeros(43, dtype=np.float32)
        source[22:29] = source_hand7_from_actuated(left)
        source[36:43] = source_hand7_from_actuated(right)
        packed = pack_psi_state(source, hand_layout="full")
        np.testing.assert_allclose(packed[29:36], left)
        np.testing.assert_allclose(packed[36:43], right)
        self.assertEqual(END_EFFECTOR_TO_HAND_LAYOUT["dex1_virtual14"], "full")

    def test_converter_virtual_action_prefers_teleop_hand_target(self):
        converter = Sonic2LeRobotConverter()
        converter.end_effector = "dex1_virtual14"
        converter.hand_layout = "full"
        target = self.mapper.hand14(0.15, 0.85)
        wbc = np.zeros(43, dtype=np.float32)
        action = np.asarray(
            converter.build_act(np.zeros(MOTION_TOKEN_DIM), wbc, hand14_override=target)
        )
        np.testing.assert_allclose(action[MOTION_TOKEN_DIM:], target)

    def test_mock_virtual_dataset_matches_official_hand_columns_and_order(self):
        with tempfile.TemporaryDirectory(prefix="psi0-mock-virtual14-") as tmp:
            root = Path(tmp)
            make_dataset(root, frames=2, end_effector="dex1_virtual14")
            frame = pd.read_parquet(
                root / "data/chunk-000/episode_000000.parquet"
            ).iloc[0]

            expected_state = self.mapper.hand14(0.0, 1.0)
            expected_action = self.mapper.hand14(1.0, 0.0)
            packed_state = pack_psi_state(
                np.asarray(frame["observation.state"], dtype=np.float32),
                hand_layout="full",
            )
            np.testing.assert_allclose(packed_state[29:43], expected_state)
            np.testing.assert_allclose(
                np.concatenate(
                    [
                        np.asarray(frame[SRC_TELEOP_LEFT_HAND]),
                        np.asarray(frame[SRC_TELEOP_RIGHT_HAND]),
                    ]
                ),
                expected_action,
            )

    def test_exporter_pose_uses_sonic_target_and_planner_mode_uses_planner(self):
        sonic = {
            "left_hand_joints": np.full(7, 1.0),
            "right_hand_joints": np.full(7, 2.0),
        }
        planner = {
            "left_hand_joints": np.full(7, 3.0),
            "right_hand_joints": np.full(7, 4.0),
        }
        left, right = select_virtual_hand_targets(1, sonic, planner)
        np.testing.assert_allclose(left, 1.0)
        np.testing.assert_allclose(right, 2.0)
        left, right = select_virtual_hand_targets(5, sonic, planner)
        np.testing.assert_allclose(left, 3.0)
        np.testing.assert_allclose(right, 4.0)
        self.assertIsNone(select_virtual_hand_targets(1, None, planner))

    def test_sanity_can_require_only_the_task_gripper(self):
        hands = np.stack([self.mapper.hand14(0.5, 0.0), self.mapper.hand14(0.5, 1.0)])
        assert_virtual_manifold("action", hands, self.mapper.dex1_map, "right")
        with self.assertRaises(AssertionError):
            assert_virtual_manifold("action", hands, self.mapper.dex1_map, "both")

    @staticmethod
    def _pose_episode_buffer(modes=(2, 1, 1, 2), stale_pose_indices=()):
        frame_count = len(modes)
        smpl = []
        for index, mode in enumerate(modes):
            active = mode in (1, 4) and index not in stale_pose_indices
            smpl.append(
                np.ones(63, dtype=np.float32)
                if active
                else np.zeros(63, dtype=np.float32)
            )
        return {
            "size": frame_count,
            "observation.state": [np.zeros(43, dtype=np.float32) for _ in modes],
            "action.wbc": [np.zeros(43, dtype=np.float32) for _ in modes],
            "action.motion_token": [np.zeros(64, dtype=np.float32) for _ in modes],
            "teleop.smpl_pose": smpl,
            "teleop.stream_mode": [np.array([mode], dtype=np.int32) for mode in modes],
        }

    def test_full_body_pose_validation_accepts_planner_edges(self):
        report = validate_full_body_pose_episode(
            self._pose_episode_buffer(stale_pose_indices=(1,))
        )
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["stream_mode_counts"], {"POSE": 2, "PLANNER": 2})
        self.assertEqual(report["smpl_valid_fraction"], 0.5)
        self.assertTrue(report["warnings"])

    def test_full_body_pose_validation_accepts_official_pose_only_episode(self):
        report = validate_full_body_pose_episode(
            self._pose_episode_buffer(modes=(1, 1, 1, 1))
        )
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["stream_mode_counts"], {"POSE": 4})

    def test_full_body_pose_validation_rejects_vr3pt(self):
        report = validate_full_body_pose_episode(
            self._pose_episode_buffer(modes=(2, 1, 5, 2))
        )
        self.assertFalse(report["valid"])
        self.assertTrue(any("VR_3PT" in error for error in report["errors"]))

    def test_full_body_pose_validation_rejects_no_pose_and_all_stale(self):
        no_pose = validate_full_body_pose_episode(self._pose_episode_buffer(modes=(2, 2)))
        self.assertFalse(no_pose["valid"])
        self.assertTrue(any("no POSE" in error for error in no_pose["errors"]))

        all_stale = validate_full_body_pose_episode(
            self._pose_episode_buffer(modes=(2, 1, 1, 2), stale_pose_indices=(1, 2))
        )
        self.assertFalse(all_stale["valid"])
        self.assertTrue(any("all POSE SMPL" in error for error in all_stale["errors"]))

    def test_full_body_pose_validation_rejects_planner_in_middle(self):
        report = validate_full_body_pose_episode(self._pose_episode_buffer(modes=(1, 2, 1)))
        self.assertFalse(report["valid"])
        self.assertTrue(any("inside" in error for error in report["errors"]))

    def test_full_body_pose_validation_rejects_bad_shapes_and_nonfinite_values(self):
        bad_shape = self._pose_episode_buffer()
        bad_shape["action.wbc"][0] = np.zeros(42, dtype=np.float32)
        report = validate_full_body_pose_episode(bad_shape)
        self.assertFalse(report["valid"])
        self.assertTrue(any("width mismatch" in error for error in report["errors"]))

        nonfinite = self._pose_episode_buffer()
        nonfinite["action.motion_token"][0][0] = np.nan
        report = validate_full_body_pose_episode(nonfinite)
        self.assertFalse(report["valid"])
        self.assertTrue(any("NaN/Inf" in error for error in report["errors"]))

    def test_initialized_empty_dataset_repair_is_local_and_idempotent(self):
        with tempfile.TemporaryDirectory(prefix="psi0-empty-sonic-") as tmp:
            dataset = Path(tmp)
            meta = dataset / "meta"
            meta.mkdir()
            (meta / "info.json").write_text(
                json.dumps({"total_episodes": 0, "total_frames": 0}), encoding="utf-8"
            )
            (meta / "modality.json").write_text("{}", encoding="utf-8")

            created = repair_initialized_empty_dataset(dataset)
            self.assertEqual(
                {path.name for path in created},
                {"tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl"},
            )
            self.assertEqual(repair_initialized_empty_dataset(dataset), [])
            for path in created:
                self.assertEqual(path.read_text(encoding="utf-8"), "")

    def test_nonempty_dataset_repair_does_not_touch_metadata(self):
        with tempfile.TemporaryDirectory(prefix="psi0-nonempty-sonic-") as tmp:
            dataset = Path(tmp)
            meta = dataset / "meta"
            meta.mkdir()
            (meta / "info.json").write_text(
                json.dumps({"total_episodes": 1, "total_frames": 10}), encoding="utf-8"
            )

            self.assertEqual(repair_initialized_empty_dataset(dataset), [])
            self.assertFalse((meta / "tasks.jsonl").exists())

    def test_trigger_bridge_uses_trigger_not_recording_grip(self):
        controller = FakeController()
        bridge = Dex1VirtualTeleopBridge(self.mapper, controller, max_step_rad=10.0)
        left, right = bridge.compute_hand_joints_from_inputs(
            None, None, 0.0, 1.0, 1.0, 0.0
        )
        expected_left, expected_right = self.mapper.hand7_pair(1.0, 0.0)
        np.testing.assert_allclose(left, expected_left)
        np.testing.assert_allclose(right, expected_right)
        self.assertEqual(controller.target, (5.5, 0.0))
        bridge.close()
        self.assertTrue(controller.closed)

    def test_exporter_patch_uses_state_for_observation_and_target_for_action(self):
        left_action, right_action = self.mapper.hand7_pair(0.1, 0.9)
        original = {"body_q": np.arange(29), "left_hand_q": np.ones(7)}
        patched = patch_sonic_proprio_hands(
            original,
            left_q=5.5,
            right_q=0.0,
            left_action7=left_action,
            right_action7=right_action,
            mapper=self.mapper,
        )
        expected_left_state, expected_right_state = self.mapper.hand7_pair(1.0, 0.0)
        np.testing.assert_allclose(patched["left_hand_q"], expected_left_state)
        np.testing.assert_allclose(patched["right_hand_q"], expected_right_state)
        np.testing.assert_allclose(patched["last_left_hand_action"], left_action)
        np.testing.assert_allclose(patched["last_right_hand_action"], right_action)
        self.assertIsNot(patched, original)
        np.testing.assert_array_equal(original["left_hand_q"], np.ones(7))


if __name__ == "__main__":
    unittest.main()
