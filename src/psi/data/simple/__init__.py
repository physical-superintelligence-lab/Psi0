
# 36-D action layout — matches what the policy emits and what's stored under
# `action` in the postprocess datasets.
#
#   [0:14]   hand qpos          absolute (L_thumb 3, L_index 2, L_middle 2, R_hand 7)
#   [14:28]  arm qpos           absolute (L_arm 7, R_arm 7)
#   [28:31]  waist roll/pitch/yaw  absolute
#   [31]     base_height        absolute (m; reference 0.75)
#   [32]     vx                 relative (m/s, base frame)
#   [33]     vy                 relative
#   [34]     turning_flag       discrete (0/1)
#   [35]     target_yaw         absolute (world heading, rad)

import numpy as np


ACTION_SLICES_36D = [
    ("left_arm",         14, 21),
    ("right_arm",        21, 28),
    ("left_hand_thumb",   0,  3),
    ("left_hand_index",   3,  5),
    ("left_hand_middle",  5,  7),
    ("right_hand",        7, 14),
]

def simple_freeze_action(action: np.ndarray) -> np.ndarray:
    """Construct a freeze-target 36-D action for end-of-chunk supervision padding for the SIMPLE dataset.

    Use as the target when padding training samples that overshoot the recorded
    episode end: the policy should learn to output this kind of action so that
    when AMO consumes it, the robot stays still at its end pose.

    Recipe: keep all *absolute* targets (joint qpos, waist, base height,
    target_yaw); zero the *relative*/discrete loco signals (vx, vy, turning_flag).

        kept:  [0:14] hand_qpos, [14:28] arm_qpos, [28:31] waist_rpy,
               [31] base_height, [35] target_yaw
        zeroed:[32] vx, [33] vy, [34] turning_flag

    Note: target_yaw [35] stays at the last commanded *world-frame* heading.
    For most end-of-task states the robot is already tracking that heading,
    so dyaw ≈ 0 and AMO holds. If the episode ended mid-turn the residual
    dyaw will still drive a small rotation; that's a separate question
    (would need current robot yaw, not available at supervision-target time).
    """
    assert action.shape[-1] == 36, f"expected last dim 36, got shape={action.shape}"
    frozen = action.copy()
    frozen[..., 32] = 0.0  # vx
    frozen[..., 33] = 0.0  # vy
    frozen[..., 34] = 0    # turning_flag
    return frozen