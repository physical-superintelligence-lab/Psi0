import os
import sys
import time
import json

import casadi
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin


parent2_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(parent2_dir)

class GetTauer:
    def __init__(self):
        robot = pin.RobotWrapper.BuildFromURDF("./assets/g1/g1_body29_hand14.urdf", "./assets/g1/")

        mixed_jointsToLockIDs = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_hand_thumb_0_joint",
            "left_hand_thumb_1_joint",
            "left_hand_thumb_2_joint",
            "left_hand_middle_0_joint",
            "left_hand_middle_1_joint",
            "left_hand_index_0_joint",
            "left_hand_index_1_joint",
            "right_hand_thumb_0_joint",
            "right_hand_thumb_1_joint",
            "right_hand_thumb_2_joint",
            "right_hand_index_0_joint",
            "right_hand_index_1_joint",
            "right_hand_middle_0_joint",
            "right_hand_middle_1_joint",
        ]

        reduced_robot = robot.buildReducedRobot(
            list_of_joints_to_lock=mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * robot.model.nq),
        )

        reduced_robot.model.addFrame(
            pin.Frame(
                "L_ee",
                reduced_robot.model.getJointId("left_wrist_yaw_joint"),
                pin.SE3(np.eye(3), np.array([0.05, 0, 0]).T),
                pin.FrameType.OP_FRAME,
            )
        )

        reduced_robot.model.addFrame(
            pin.Frame(
                "R_ee",
                reduced_robot.model.getJointId("right_wrist_yaw_joint"),
                pin.SE3(np.eye(3), np.array([0.05, 0, 0]).T),
                pin.FrameType.OP_FRAME,
            )
        )
        self.model = reduced_robot.model
        self.data = reduced_robot.data
    
    def __call__(self, q, qdot=None, qddot=None):
        return self._compute_tau(self.model, self.data, q, qdot, qddot)
    
    def _compute_tau(
        self,
        model: pin.Model,
        data: pin.Data,
        q: np.ndarray,
        qdot: np.ndarray = None,
        qddot: np.ndarray = None,
    ) -> np.ndarray:

        nv = model.nv
        if qdot is None:
            qdot = np.zeros(nv)
        if qddot is None:
            qddot = np.zeros(nv)
        tau = pin.rnea(model, data, q, qdot, qddot)
        return tau