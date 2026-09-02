import unittest

import numpy as np

from scripts.offline.dex1_1_layout import (
    ACTION_LEFT_GRIPPER_INDEX,
    ACTION_PADDING_HAND_INDICES,
    ACTION_RIGHT_GRIPPER_INDEX,
    HAND_SOURCE_SLICES,
    PSI_ACTION_DIM,
    PSI_STATE_DIM,
    STATE_LEFT_GRIPPER_INDEX,
    STATE_PADDING_HAND_INDICES,
    STATE_RIGHT_GRIPPER_INDEX,
    extract_dex1_command,
    pack_psi_action,
    pack_psi_state,
    source_full_hand14,
    source_hand7_from_actuated,
    refill_dex1_state,
)


class Dex11LayoutTest(unittest.TestCase):
    def test_pack_state_keeps_only_dex1_slots(self):
        source = np.arange(43, dtype=np.float32)
        state = pack_psi_state(source)
        self.assertEqual(state.shape, (PSI_STATE_DIM,))
        np.testing.assert_allclose(state[:15], source[:15])
        np.testing.assert_allclose(state[15:22], source[15:22])
        np.testing.assert_allclose(state[22:29], source[29:36])
        self.assertEqual(state[STATE_LEFT_GRIPPER_INDEX], source[22])
        self.assertEqual(state[STATE_RIGHT_GRIPPER_INDEX], source[36])
        np.testing.assert_allclose(state[list(STATE_PADDING_HAND_INDICES)], 0.0)

    def test_pack_action_routes_token_and_dex1_slots(self):
        token = np.arange(64, dtype=np.float32)
        wbc = np.arange(100, 143, dtype=np.float32)
        action = pack_psi_action(token, wbc)
        self.assertEqual(action.shape, (PSI_ACTION_DIM,))
        np.testing.assert_allclose(action[:64], token)
        self.assertEqual(action[ACTION_LEFT_GRIPPER_INDEX], wbc[22])
        self.assertEqual(action[ACTION_RIGHT_GRIPPER_INDEX], wbc[36])
        np.testing.assert_allclose(action[list(ACTION_PADDING_HAND_INDICES)], 0.0)

    def test_full_hand_layout_preserves_dex3_hand_values(self):
        source = np.arange(43, dtype=np.float32)
        left = np.asarray([10, 11, 12, 13, 14, 15, 16], dtype=np.float32)
        right = np.asarray([20, 21, 22, 23, 24, 25, 26], dtype=np.float32)
        source[22:29] = source_hand7_from_actuated(left)
        source[36:43] = source_hand7_from_actuated(right)
        state = pack_psi_state(source, hand_layout="full")
        token = np.arange(64, dtype=np.float32)
        action = pack_psi_action(token, source, hand_layout="full")
        expected_hand = source_full_hand14(source)
        np.testing.assert_allclose(expected_hand, np.concatenate([left, right]))
        np.testing.assert_allclose(state[29:43], expected_hand)
        np.testing.assert_allclose(action[64:78], expected_hand)
        self.assertEqual(tuple(HAND_SOURCE_SLICES), ((22, 29), (36, 43)))

    def test_extract_and_refill_clamps(self):
        action = np.zeros(78, dtype=np.float32)
        action[64] = 1.5
        action[71] = -0.5
        command = extract_dex1_command(action)
        self.assertEqual(command.left, 1.0)
        self.assertEqual(command.right, 0.0)
        state = refill_dex1_state(np.zeros(29, dtype=np.float32), command.left, command.right)
        self.assertEqual(state[29], 1.0)
        self.assertEqual(state[36], 0.0)

    def test_rejects_bad_dimensions(self):
        with self.assertRaises(ValueError):
            pack_psi_state(np.zeros(42, dtype=np.float32))
        with self.assertRaises(ValueError):
            pack_psi_action(np.zeros(63, dtype=np.float32), np.zeros(43, dtype=np.float32))
        with self.assertRaises(ValueError):
            extract_dex1_command(np.zeros(77, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
