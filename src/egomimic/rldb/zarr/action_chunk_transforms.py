"""
Embodiment-dependent action chunk transforms for ZarrDataset.

Replicates the prestacking transformations from aria_to_lerobot.py / eva_to_lerobot.py,
applied at load time instead of at data creation time. Raw action frames are loaded
as (action_horizon, action_dim) and interpolated to (chunk_length, action_dim).

Translation (xyz) and gripper dimensions use linear interpolation.
Rotation (euler ypr) dimensions use np.unwrap before interpolation and rewrap after,
matching the behaviour of egomimicUtils.interpolate_arr_euler.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Literal

import numpy as np
import torch
from projectaria_tools.core.sophus import SE3
from scipy.spatial.transform import Rotation as R

from egomimic.utils.pose_utils import (
    _interpolate_euler,
    _interpolate_linear,
    _interpolate_quat_wxyz,
    _interpolate_xyz,
    _matrix_to_xyz,
    _matrix_to_xyzwxyz,
    _matrix_to_xyzypr,
    _xyz_to_matrix,
    _xyzwxyz_to_matrix,
    _xyzypr_to_matrix,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)

# ---------------------------------------------------------------------------
# Base Transform
# ---------------------------------------------------------------------------


class Transform:
    """Base Class for all transforms."""

    @abstractmethod
    def transform(self, batch: dict) -> dict:
        """Transform the data."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Interpolation Transforms
# ---------------------------------------------------------------------------


class InterpolatePose(Transform):
    """Interpolate a pose chunk of shape (T, 6) or (T, 7)."""

    def __init__(
        self,
        new_chunk_length: int,
        action_key: str,
        output_action_key: str,
        stride: int = 1,
        mode: Literal["xyzwxyz", "xyzypr"] = "xyzwxyz",
    ):
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        self.new_chunk_length = new_chunk_length
        self.action_key = action_key
        self.output_action_key = output_action_key
        self.stride = int(stride)
        self.mode = mode

    def transform(self, batch: dict) -> dict:
        actions = np.asarray(batch[self.action_key])
        actions = actions[:: self.stride]
        if self.mode == "xyzwxyz":
            if actions.ndim != 2 or actions.shape[-1] != 7:
                raise ValueError(
                    f"InterpolatePose expects (T, 7) when is_quat=True, got "
                    f"{actions.shape} for key '{self.action_key}'"
                )
            batch[self.output_action_key] = _interpolate_quat_wxyz(
                actions, self.new_chunk_length
            )
        elif self.mode == "xyzypr":
            if actions.ndim != 2 or actions.shape[-1] != 6:
                raise ValueError(
                    f"InterpolatePose expects (T, 6), got {actions.shape} for key "
                    f"'{self.action_key}'"
                )
            batch[self.output_action_key] = _interpolate_euler(
                actions, self.new_chunk_length
            )
        else:
            if actions.shape[-1] != 3:
                raise ValueError(
                    f"InterpolatePose expects (T, 3) or (T, K, 3), got {actions.shape} for key "
                    f"'{self.action_key}'"
                )
            batch[self.output_action_key] = _interpolate_xyz(
                actions, self.new_chunk_length
            )
        return batch


class InterpolateLinear(Transform):
    """Interpolate any chunk of shape (T, D) with linear interpolation."""

    def __init__(
        self,
        new_chunk_length: int,
        action_key: str,
        output_action_key: str,
        stride: int = 1,
    ):
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        self.new_chunk_length = new_chunk_length
        self.action_key = action_key
        self.output_action_key = output_action_key
        self.stride = int(stride)

    def transform(self, batch: dict) -> dict:
        actions = np.asarray(batch[self.action_key])
        if actions.ndim != 2:
            raise ValueError(
                f"InterpolateLinear expects (T, D), got {actions.shape} for key "
                f"'{self.action_key}'"
            )
        actions = actions[:: self.stride]
        batch[self.output_action_key] = _interpolate_linear(
            actions, self.new_chunk_length
        )
        return batch


# ---------------------------------------------------------------------------
# Coordinate Transforms
# ---------------------------------------------------------------------------


class ActionChunkCoordinateFrameTransform(Transform):
    def __init__(
        self,
        target_world: str,
        chunk_world: str,
        transformed_key_name: str,
        extra_batch_key: dict = None,
        mode: Literal["xyz", "xyzwxyz", "xyzypr"] = "xyzwxyz",
        inverse: bool = True,
    ):
        """
        args:
            target_world:
            chunk_world:
            transformed_key_name:
            is_quat: if True, inputs are xyz + quat(wxyz); otherwise xyz + ypr.
        """
        self.target_world = target_world
        self.chunk_world = chunk_world
        self.transformed_key_name = transformed_key_name
        self.extra_batch_key = extra_batch_key
        self.mode = mode
        self.inverse = inverse

    def transform(self, batch):
        """
        args:
            batch:
                if is_quat=False, inputs are xyz + ypr.
                if is_quat=True, inputs are xyz + quat(wxyz).
                Input shape validation is delegated to the selected to-matrix helper.
                transformed_key_name: str, name of the new key to store the transformed chunk world in

        returns
            batch with new key containing transformed chunk world in target frame:
                if is_quat=False: (T, 6) xyz + ypr
                if is_quat=True: (T, 7) xyz + quat(wxyz)
        """
        # flatten to (T, D)
        # target world is head pose, chunk world is keypoints
        batch.update(self.extra_batch_key or {})
        target_world = np.asarray(batch[self.target_world])
        chunk_world = np.asarray(batch[self.chunk_world])
        chunk_world_shape = None

        if chunk_world.ndim > 2:
            chunk_world_shape = chunk_world.shape
            chunk_world = chunk_world.reshape(-1, chunk_world_shape[-1])

        to_matrix_fn = None
        if self.mode == "xyzwxyz":
            to_matrix_fn = _xyzwxyz_to_matrix
        elif self.mode == "xyzypr":
            to_matrix_fn = _xyzypr_to_matrix
        elif self.mode == "xyz":
            to_matrix_fn = _xyz_to_matrix
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        target_world_to_matrix_fn = (
            _xyzwxyz_to_matrix if target_world.shape[-1] == 7 else _xyzypr_to_matrix
        )
        # Convert to SE3 for transformation
        target_se3 = SE3.from_matrix(
            target_world_to_matrix_fn(target_world[None, :])[0]
        )  # (4, 4)
        chunk_se3 = SE3.from_matrix(to_matrix_fn(chunk_world))  # (T, 4, 4)

        # Compute relative transform and apply to chunk
        if self.inverse:
            chunk_in_target_frame = target_se3.inverse() @ chunk_se3
        else:
            chunk_in_target_frame = target_se3 @ chunk_se3
        chunk_mats = chunk_in_target_frame.to_matrix()
        if chunk_mats.ndim == 2:
            chunk_mats = chunk_mats[None, ...]

        if self.mode == "xyzwxyz":
            chunk_in_target_frame = _matrix_to_xyzwxyz(chunk_mats)
        elif self.mode == "xyzypr":
            chunk_in_target_frame = _matrix_to_xyzypr(chunk_mats)
        elif self.mode == "xyz":
            chunk_in_target_frame = _matrix_to_xyz(chunk_mats)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        if chunk_world_shape is not None:
            chunk_in_target_frame = chunk_in_target_frame.reshape(*chunk_world_shape)

        # Store transformed chunk back in batch
        batch[self.transformed_key_name] = chunk_in_target_frame

        return batch


class QuaternionPoseToYPR(Transform):
    """Convert a single pose from xyz + quat(x,y,z,w) to xyz + ypr."""

    def __init__(self, pose_key: str, output_key: str):
        self.pose_key = pose_key
        self.output_key = output_key

    def transform(self, batch: dict) -> dict:
        pose = np.asarray(batch[self.pose_key])
        if pose.shape != (7,):
            raise ValueError(
                f"QuaternionPoseToYPR expects shape (7,), got {pose.shape} for key "
                f"'{self.pose_key}'"
            )
        xyz = pose[:3]
        xyzw = wxyz_to_xyzw(pose[3:7])
        ypr = R.from_quat(xyzw).as_euler("ZYX", degrees=False)
        batch[self.output_key] = np.concatenate([xyz, ypr], axis=0)
        return batch


class YPRToQuaternionPose(Transform):
    """Convert a single pose from xyz + ypr to xyz + quat(x,y,z,w)."""

    def __init__(self, pose_key: str, output_key: str):
        self.pose_key = pose_key
        self.output_key = output_key

    def transform(self, batch: dict) -> dict:
        pose = np.asarray(batch[self.pose_key])
        if pose.shape != (6,):
            raise ValueError(
                f"YPRToQuaternionPose expects shape (6,), got {pose.shape} for key "
                f"'{self.pose_key}'"
            )
        xyz = pose[:3]
        quat = R.from_euler("ZYX", pose[3:6], degrees=False).as_quat()  # (x,y,z,w)
        quat = xyzw_to_wxyz(quat)
        batch[self.output_key] = np.concatenate([xyz, quat], axis=0)
        return batch


class BatchQuaternionPoseToYPR(Transform):
    """Convert a batch of poses from xyz + quat(x,y,z,w) to xyz + ypr."""

    def __init__(self, pose_key: str, output_key: str):
        self.pose_key = pose_key
        self.output_key = output_key

    def transform(self, batch: dict) -> dict:
        pose = np.asarray(batch[self.pose_key])
        if pose.ndim != 2 or pose.shape[-1] != 7:
            raise ValueError(
                f"BatchQuaternionPoseToYPR expects shape (N, 7), got {pose.shape} for key "
                f"'{self.pose_key}'"
            )
        xyz = pose[:, :3]
        xyzw = wxyz_to_xyzw(pose[:, 3:7])
        ypr = R.from_quat(xyzw).as_euler("ZYX", degrees=False)  # (N, 3)
        batch[self.output_key] = np.concatenate([xyz, ypr], axis=1)
        return batch


class BatchYPRToQuaternionPose(Transform):
    """Convert a batch of poses from xyz + ypr to xyz + quat(x,y,z,w)."""

    def __init__(self, pose_key: str, output_key: str):
        self.pose_key = pose_key
        self.output_key = output_key

    def transform(self, batch: dict) -> dict:
        pose = np.asarray(batch[self.pose_key])
        if pose.ndim != 2 or pose.shape[-1] != 6:
            raise ValueError(
                f"BatchYPRToQuaternionPose expects shape (N, 6), got {pose.shape} for key "
                f"'{self.pose_key}'"
            )
        xyz = pose[:, :3]
        quat = R.from_euler("ZYX", pose[:, 3:6], degrees=False).as_quat()  # (N, 4)
        quat = xyzw_to_wxyz(quat)
        batch[self.output_key] = np.concatenate([xyz, quat], axis=1)
        return batch


class PoseCoordinateFrameTransform(Transform):
    """Transform a single pose into a target frame pose."""

    def __init__(
        self,
        target_world: str,
        pose_world: str,
        transformed_key_name: str,
        mode: Literal["xyzwxyz", "xyzypr", "xyz"] = "xyzwxyz",
    ):
        self.target_world = target_world
        self.pose_world = pose_world
        self.transformed_key_name = transformed_key_name
        self.mode = mode
        self._chunk_transform = ActionChunkCoordinateFrameTransform(
            target_world=target_world,
            chunk_world=pose_world,
            transformed_key_name=transformed_key_name,
            mode=mode,
        )

    def transform(self, batch: dict) -> dict:
        pose_world = np.asarray(batch[self.pose_world])
        transformed = self._chunk_transform.transform(
            {
                self.target_world: batch[self.target_world],
                self.pose_world: pose_world[None, :],
            }
        )
        batch[self.transformed_key_name] = np.asarray(
            transformed[self.transformed_key_name]
        )[0]
        return batch


class DeleteKeys(Transform):
    def __init__(self, keys_to_delete):
        self.keys_to_delete = keys_to_delete

    def transform(self, batch):
        for key in self.keys_to_delete:
            batch.pop(key, None)
        return batch


class XYZWXYZ_to_XYZYPR(Transform):
    """Convert listed keys from xyz+quat(wxyz) to xyz+ypr in-place."""

    def __init__(self, keys: list[str]):
        self.keys = list(keys)

    def transform(self, batch: dict) -> dict:
        for key in self.keys:
            value = np.asarray(batch[key])
            if value.ndim == 1 and value.shape[0] == 7:
                batch[key] = _matrix_to_xyzypr(_xyzwxyz_to_matrix(value[None, :]))[0]
            elif value.ndim == 2 and value.shape[1] == 7:
                batch[key] = _matrix_to_xyzypr(_xyzwxyz_to_matrix(value))
            else:
                raise ValueError(
                    f"XYZWXYZ_to_XYZYPR expects key '{key}' to have shape (7,) "
                    f"or (T, 7), got {value.shape}"
                )
        return batch


class CartesianWithGripperCoordinateTransform(Transform):
    def __init__(
        self,
        left_target_world: str,
        right_target_world: str,
        chunk_world: str,
        transformed_key_name: str,
        extra_batch_key: dict = None,
    ):
        """
        args:
            left_target_world: string key for left target world pose in batch (6D: xyz + ypr)
            right_target_world: string key for right target world pose in batch (6D: xyz + ypr)
            chunk_world: string key for chunk world pose in batch (14D: xyz + ypr + gripper * 2 arms)
            transformed_key_name: string key to store transformed chunk world in batch (14D)
        """
        self.left_target_world = left_target_world
        self.right_target_world = right_target_world
        self.chunk_world = chunk_world
        self.transformed_key_name = transformed_key_name
        self.extra_batch_key = extra_batch_key

    def transform(self, batch):
        """
        args:
            batch:
                left_target_world: numpy(6): xyz + ypr
                right_target_world: numpy(6): xyz + ypr
                chunk_world: numpy(T, 14): [left xyz+ypr+gripper, right xyz+ypr+gripper]
                transformed_key_name: str, name of the new key to store the transformed chunk world in

        returns
            batch with new key containing transformed chunk world in target frame: (T, 14)
        """
        batch.update(self.extra_batch_key or {})
        left_target_world = batch[self.left_target_world]
        right_target_world = batch[self.right_target_world]
        chunk_world = batch[self.chunk_world]

        if left_target_world.shape != (6,):
            raise ValueError(
                f"Expected left_target_world shape (6,), got {left_target_world.shape}"
            )
        if right_target_world.shape != (6,):
            raise ValueError(
                f"Expected right_target_world shape (6,), got {right_target_world.shape}"
            )
        if chunk_world.ndim != 2 or chunk_world.shape[1] != 14:
            raise ValueError(
                f"Expected chunk_world shape (T, 14), got {chunk_world.shape}"
            )

        # Chunk layout: [left xyz+ypr+gripper, right xyz+ypr+gripper]
        left_pose_world = chunk_world[:, :6]
        right_pose_world = chunk_world[:, 7:13]

        left_target_se3 = SE3.from_matrix(
            _xyzypr_to_matrix(left_target_world[None, :])[0]
        )
        right_target_se3 = SE3.from_matrix(
            _xyzypr_to_matrix(right_target_world[None, :])[0]
        )
        left_target_inv = left_target_se3.inverse()
        right_target_inv = right_target_se3.inverse()

        left_pose_in_target = _matrix_to_xyzypr(
            (
                left_target_inv @ SE3.from_matrix(_xyzypr_to_matrix(left_pose_world))
            ).to_matrix()
        )
        right_pose_in_target = _matrix_to_xyzypr(
            (
                right_target_inv @ SE3.from_matrix(_xyzypr_to_matrix(right_pose_world))
            ).to_matrix()
        )

        chunk_in_target_frame = np.empty_like(chunk_world)
        chunk_in_target_frame[:, :6] = left_pose_in_target
        chunk_in_target_frame[:, 6] = chunk_world[:, 6]  # left gripper unchanged
        chunk_in_target_frame[:, 7:13] = right_pose_in_target
        chunk_in_target_frame[:, 13] = chunk_world[:, 13]  # right gripper unchanged

        batch[self.transformed_key_name] = chunk_in_target_frame
        return batch


# ---------------------------------------------------------------------------
# Shape Transforms
# ---------------------------------------------------------------------------
class SplitKeys(Transform):
    def __init__(self, input_key: str, output_key_list: list[(str, int)]):
        self.input_key = input_key
        self.output_key_list = list(output_key_list)

    def transform(self, batch: dict) -> dict:
        prev_end = 0
        for key, size in self.output_key_list:
            batch[key] = batch[self.input_key][..., prev_end : prev_end + size]
            prev_end += size
        return batch


class ConcatKeys(Transform):
    def __init__(self, key_list, new_key_name, delete_old_keys=False):
        self.key_list = list(key_list)
        self.new_key_name = new_key_name
        self.delete_old_keys = delete_old_keys

    def transform(self, batch):
        present = [k for k in self.key_list if k in batch]
        if not present:
            return batch
        arrays = [np.asarray(batch[k]) for k in present]
        try:
            batch[self.new_key_name] = np.concatenate(arrays, axis=-1)
        except ValueError as e:
            shapes = {k: np.asarray(batch[k]).shape for k in present}
            raise ValueError(
                f"ConcatKeys failed for keys {present} with shapes {shapes}"
            ) from e

        if self.delete_old_keys:
            for k in present:
                batch.pop(k, None)

        return batch


class Reshape(Transform):
    def __init__(self, input_key: str, output_key: str, shape: tuple):
        self.input_key = input_key
        self.output_key = output_key
        self.shape = shape

    def transform(self, batch: dict) -> dict:
        batch[self.output_key] = batch[self.input_key].reshape(*self.shape)
        return batch


# ---------------------------------------------------------------------------
# Type Transforms
# ---------------------------------------------------------------------------


class NumpyToTensor(Transform):
    def __init__(self, keys: list[str]):
        self.keys = keys

    def transform(self, batch: dict) -> dict:
        for key in self.keys:
            if isinstance(batch[key], np.ndarray):
                batch[key] = torch.from_numpy(batch[key])
            elif isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].clone()
            else:
                raise ValueError(
                    f"NumpyToTensor expects key '{key}' to be a numpy array or torch tensor, got {type(batch[key])}"
                )
        return batch
