"""Shared Dex1-1 gripper layout helpers for SONIC -> Psi0 data and mock deploy.

Psi0 uses 43-D state vectors and 78-D action vectors for SONIC:

* state = qpos29 + hand14
* action = motion_token64 + hand14

Dex1-1 only consumes two gripper scalars. They live in the first slot of each
7-wide hand block: left slot 0, right slot 7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

SOURCE_VECTOR_DIM = 43
QPOS_DIM = 29
HAND_WIDTH = 14
MOTION_TOKEN_DIM = 64
PSI_STATE_DIM = QPOS_DIM + HAND_WIDTH
PSI_ACTION_DIM = MOTION_TOKEN_DIM + HAND_WIDTH

LEFT_GRIPPER_SLOT = 0
RIGHT_GRIPPER_SLOT = 7
STATE_HAND_OFFSET = QPOS_DIM
ACTION_HAND_OFFSET = MOTION_TOKEN_DIM

STATE_LEFT_GRIPPER_INDEX = STATE_HAND_OFFSET + LEFT_GRIPPER_SLOT
STATE_RIGHT_GRIPPER_INDEX = STATE_HAND_OFFSET + RIGHT_GRIPPER_SLOT
ACTION_LEFT_GRIPPER_INDEX = ACTION_HAND_OFFSET + LEFT_GRIPPER_SLOT
ACTION_RIGHT_GRIPPER_INDEX = ACTION_HAND_OFFSET + RIGHT_GRIPPER_SLOT

# Source SONIC 43-D layout:
# [0:15] lower | [15:22] left arm | [22:29] left hand |
# [29:36] right arm | [36:43] right hand
QPOS_SOURCE_SLICES = ((0, 15), (15, 22), (29, 36))
LEFT_HAND_SOURCE_SLICE = (22, 29)
RIGHT_HAND_SOURCE_SLICE = (36, 43)
HAND_SOURCE_SLICES = (LEFT_HAND_SOURCE_SLICE, RIGHT_HAND_SOURCE_SLICE)

# SONIC's RobotModel stores each hand in full-configuration order:
#   index0, index1, middle0, middle1, thumb0, thumb1, thumb2
# The teleop/exporter hand7 vectors use actuated order:
#   thumb0, thumb1, thumb2, index0, index1, middle0, middle1
# These permutations are inverses.  The sparse Dex1 layout intentionally keeps
# using source slot zero and is not affected by this dense virtual14 conversion.
ACTUATED_TO_SOURCE_HAND7 = (3, 4, 5, 6, 0, 1, 2)
SOURCE_TO_ACTUATED_HAND7 = (4, 5, 6, 0, 1, 2, 3)

ACTIVE_HAND_SLOTS = (LEFT_GRIPPER_SLOT, RIGHT_GRIPPER_SLOT)
STATE_ACTIVE_GRIPPER_INDICES = (STATE_LEFT_GRIPPER_INDEX, STATE_RIGHT_GRIPPER_INDEX)
ACTION_ACTIVE_GRIPPER_INDICES = (ACTION_LEFT_GRIPPER_INDEX, ACTION_RIGHT_GRIPPER_INDEX)
STATE_PADDING_HAND_INDICES = tuple(
    STATE_HAND_OFFSET + i for i in range(HAND_WIDTH) if i not in ACTIVE_HAND_SLOTS
)
ACTION_PADDING_HAND_INDICES = tuple(
    ACTION_HAND_OFFSET + i for i in range(HAND_WIDTH) if i not in ACTIVE_HAND_SLOTS
)


@dataclass(frozen=True)
class Dex11Command:
    left: float
    right: float


def _as_1d_array(name: str, values: Sequence[float] | np.ndarray, expected_dim: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.shape[0] != expected_dim:
        raise ValueError(f"{name} must have length {expected_dim}, got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf")
    return arr


def take_slices(values: np.ndarray, slices: Iterable[tuple[int, int]]) -> np.ndarray:
    return np.concatenate([values[start:end] for start, end in slices]).astype(np.float32)


def source_qpos29(source43: Sequence[float] | np.ndarray) -> np.ndarray:
    source = _as_1d_array("source43", source43, SOURCE_VECTOR_DIM)
    return take_slices(source, QPOS_SOURCE_SLICES)


def dex1_hand14(left: float, right: float) -> np.ndarray:
    hand = np.zeros(HAND_WIDTH, dtype=np.float32)
    hand[LEFT_GRIPPER_SLOT] = float(left)
    hand[RIGHT_GRIPPER_SLOT] = float(right)
    return hand


def source_dex1_hand14(source43: Sequence[float] | np.ndarray) -> np.ndarray:
    source = _as_1d_array("source43", source43, SOURCE_VECTOR_DIM)
    left_start, _left_end = LEFT_HAND_SOURCE_SLICE
    right_start, _right_end = RIGHT_HAND_SOURCE_SLICE
    return dex1_hand14(
        source[left_start + LEFT_GRIPPER_SLOT],
        source[right_start + (RIGHT_GRIPPER_SLOT - 7)],
    )


def source_full_hand14(source43: Sequence[float] | np.ndarray) -> np.ndarray:
    source = _as_1d_array("source43", source43, SOURCE_VECTOR_DIM)
    left = source[LEFT_HAND_SOURCE_SLICE[0] : LEFT_HAND_SOURCE_SLICE[1]]
    right = source[RIGHT_HAND_SOURCE_SLICE[0] : RIGHT_HAND_SOURCE_SLICE[1]]
    permutation = list(SOURCE_TO_ACTUATED_HAND7)
    return np.concatenate([left[permutation], right[permutation]]).astype(np.float32)


def source_hand7_from_actuated(hand7: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert exporter/teleop actuated hand7 order to SONIC state-vector order."""
    hand = _as_1d_array("hand7", hand7, 7)
    return hand[list(ACTUATED_TO_SOURCE_HAND7)].astype(np.float32)


def pack_psi_state(
    source43: Sequence[float] | np.ndarray,
    *,
    hand_layout: str = "dex1-1",
) -> np.ndarray:
    qpos = source_qpos29(source43)
    hand = source_dex1_hand14(source43) if hand_layout == "dex1-1" else source_full_hand14(source43)
    return np.concatenate([qpos, hand]).astype(np.float32)


def pack_psi_action(
    motion_token64: Sequence[float] | np.ndarray,
    source_wbc43: Sequence[float] | np.ndarray,
    *,
    hand_layout: str = "dex1-1",
) -> np.ndarray:
    token = _as_1d_array("motion_token64", motion_token64, MOTION_TOKEN_DIM)
    hand = source_dex1_hand14(source_wbc43) if hand_layout == "dex1-1" else source_full_hand14(source_wbc43)
    return np.concatenate([token, hand]).astype(np.float32)


def extract_dex1_command(action78: Sequence[float] | np.ndarray, *, clamp: bool = True) -> Dex11Command:
    action = _as_1d_array("action78", action78, PSI_ACTION_DIM)
    left = float(action[ACTION_LEFT_GRIPPER_INDEX])
    right = float(action[ACTION_RIGHT_GRIPPER_INDEX])
    if clamp:
        left = float(np.clip(left, 0.0, 1.0))
        right = float(np.clip(right, 0.0, 1.0))
    return Dex11Command(left=left, right=right)


def refill_dex1_state(
    qpos29_or_state43: Sequence[float] | np.ndarray,
    left: float,
    right: float,
) -> np.ndarray:
    arr = np.asarray(qpos29_or_state43, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] not in (QPOS_DIM, PSI_STATE_DIM):
        raise ValueError(f"state input must be length {QPOS_DIM} or {PSI_STATE_DIM}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("state input contains NaN or Inf")
    qpos = arr[:QPOS_DIM]
    return np.concatenate([qpos, dex1_hand14(left, right)]).astype(np.float32)
