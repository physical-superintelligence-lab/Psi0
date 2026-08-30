from __future__ import annotations

from typing import Literal

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.zarr.action_chunk_transforms import (
    ActionChunkCoordinateFrameTransform,
    BatchQuaternionPoseToYPR,
    ConcatKeys,
    DeleteKeys,
    InterpolatePose,
    PoseCoordinateFrameTransform,
    QuaternionPoseToYPR,
    Reshape,
    SplitKeys,
    Transform,
    XYZWXYZ_to_XYZYPR,
)
from egomimic.utils.type_utils import _to_numpy
from egomimic.utils.viz_utils import (
    ColorPalette,
    _viz_axes,
    _viz_keypoints,
    _viz_traj,
)


class Human(Embodiment):
    VIZ_INTRINSICS_KEY = "base"
    VIZ_IMAGE_KEY = "observations.images.front_img_1"
    ACTION_STRIDE = 3

    @classmethod
    def viz_transformed_batch(
        cls,
        batch,
        mode: Literal["traj", "axes", "keypoints"] = "traj",
        action_key="actions_cartesian",
        image_key=None,
        transform_list=None,
        **kwargs,
    ):
        if transform_list is not None:
            batch = cls.apply_transform(batch, transform_list)

        image_key = image_key or cls.VIZ_IMAGE_KEY
        action_key = action_key or "actions_cartesian"
        intrinsics_key = cls.VIZ_INTRINSICS_KEY
        mode = (mode or "traj").lower()
        images = _to_numpy(batch[image_key][0])
        actions = _to_numpy(batch[action_key][0])

        return cls.viz(
            images=images,
            actions=actions,
            mode=mode,
            intrinsics_key=intrinsics_key,
            **kwargs,
        )

    @classmethod
    def viz(
        cls,
        images,
        actions,
        mode: Literal["traj", "axes", "keypoints"] = "traj",
        intrinsics_key=None,
        **kwargs,
    ):
        intrinsics_key = intrinsics_key or cls.VIZ_INTRINSICS_KEY
        if mode == "traj":
            return _viz_traj(
                images=images,
                actions=actions,
                intrinsics_key=intrinsics_key,
                **kwargs,
            )
        if mode == "axes":
            return _viz_axes(
                images=images,
                actions=actions,
                intrinsics_key=intrinsics_key,
                **kwargs,
            )
        if mode == "keypoints":
            color = kwargs.get("color", None)
            if color is not None and ColorPalette.is_valid(color):
                n = len(cls.FINGER_COLORS)
                colors = {
                    finger: ColorPalette.to_rgb(color, value=(i + 1) / (n + 1))
                    for i, finger in enumerate(cls.FINGER_COLORS)
                }
                dot_color = ColorPalette.to_rgb(color, value=0.7)
            else:
                colors = cls.FINGER_COLORS
                dot_color = cls.DOT_COLOR
            return _viz_keypoints(
                images=images,
                actions=actions,
                intrinsics_key=intrinsics_key,
                edges=cls.FINGER_EDGES,
                edge_ranges=cls.FINGER_EDGE_RANGES,
                colors=colors,
                dot_color=dot_color,
                **kwargs,
            )
        raise ValueError(
            f"Unsupported mode '{mode}'. Expected one of: "
            f"('traj', 'axes', 'keypoints')."
        )

    @classmethod
    def get_keymap(cls, mode: Literal["cartesian", "keypoints"]):
        if mode == "cartesian":
            key_map = {
                cls.VIZ_IMAGE_KEY: {
                    "key_type": "camera_keys",
                    "zarr_key": "images.front_1",
                },
                "right.action_ee_pose": {
                    "key_type": "action_keys",
                    "zarr_key": "right.obs_ee_pose",
                    "horizon": 30,
                },
                "left.action_ee_pose": {
                    "key_type": "action_keys",
                    "zarr_key": "left.obs_ee_pose",
                    "horizon": 30,
                },
                "right.obs_ee_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "right.obs_ee_pose",
                },
                "left.obs_ee_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "left.obs_ee_pose",
                },
                "obs_head_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "obs_head_pose",
                },
            }
        elif mode == "keypoints":
            key_map = {
                cls.VIZ_IMAGE_KEY: {
                    "key_type": "camera_keys",
                    "zarr_key": "images.front_1",
                },
                "left.action_keypoints": {
                    "key_type": "action_keys",
                    "zarr_key": "left.obs_keypoints",
                    "horizon": 30,
                },
                "right.action_keypoints": {
                    "key_type": "action_keys",
                    "zarr_key": "right.obs_keypoints",
                    "horizon": 30,
                },
                "left.action_wrist_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "left.obs_wrist_pose",
                    "horizon": 30,
                },
                "right.action_wrist_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "right.obs_wrist_pose",
                    "horizon": 30,
                },
                "left.obs_keypoints": {
                    "key_type": "proprio_keys",
                    "zarr_key": "left.obs_keypoints",
                },
                "right.obs_keypoints": {
                    "key_type": "proprio_keys",
                    "zarr_key": "right.obs_keypoints",
                },
                "left.obs_wrist_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "left.obs_wrist_pose",
                },
                "right.obs_wrist_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "right.obs_wrist_pose",
                },
                "obs_head_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "obs_head_pose",
                },
            }
        else:
            raise ValueError(
                f"Unsupported mode '{mode}'. Expected one of: 'cartesian', 'keypoints'."
            )
        return key_map


class Aria(Human):
    VIZ_INTRINSICS_KEY = "base"
    ACTION_STRIDE = 3
    # Aria SDK 21-landmark ordering (from projectaria_tools HandLandmark enum):
    #   0: THUMB_FINGERTIP      1: INDEX_FINGERTIP     2: MIDDLE_FINGERTIP
    #   3: RING_FINGERTIP       4: PINKY_FINGERTIP     5: WRIST
    #   6: THUMB_INTERMEDIATE   7: THUMB_DISTAL
    #   8: INDEX_PROXIMAL       9: INDEX_INTERMEDIATE  10: INDEX_DISTAL
    #   11: MIDDLE_PROXIMAL     12: MIDDLE_INTERMEDIATE 13: MIDDLE_DISTAL
    #   14: RING_PROXIMAL       15: RING_INTERMEDIATE  16: RING_DISTAL
    #   17: PINKY_PROXIMAL      18: PINKY_INTERMEDIATE 19: PINKY_DISTAL
    #   20: PALM_CENTER
    FINGER_EDGES = [
        # thumb:  WRIST(5) → INTERMEDIATE(6) → DISTAL(7) → TIP(0)   [3 edges]
        (5, 6),
        (6, 7),
        (7, 0),
        # index:  WRIST(5) → PROXIMAL(8) → INTERMEDIATE(9) → DISTAL(10) → TIP(1)  [4 edges]
        (5, 8),
        (8, 9),
        (9, 10),
        (10, 1),
        # middle: WRIST(5) → PROXIMAL(11) → INTERMEDIATE(12) → DISTAL(13) → TIP(2)  [4 edges]
        (5, 11),
        (11, 12),
        (12, 13),
        (13, 2),
        # ring:   WRIST(5) → PROXIMAL(14) → INTERMEDIATE(15) → DISTAL(16) → TIP(3)  [4 edges]
        (5, 14),
        (14, 15),
        (15, 16),
        (16, 3),
        # pinky:  WRIST(5) → PROXIMAL(17) → INTERMEDIATE(18) → DISTAL(19) → TIP(4)  [4 edges]
        (5, 17),
        (17, 18),
        (18, 19),
        (19, 4),
    ]
    FINGER_COLORS = {
        "thumb": (255, 100, 100),  # red
        "index": (100, 255, 100),  # green
        "middle": (100, 100, 255),  # blue
        "ring": (255, 255, 100),  # yellow
        "pinky": (255, 100, 255),  # magenta
    }
    FINGER_EDGE_RANGES = [
        ("thumb", 0, 3),  # 3 edges (no CMC joint in Aria SDK)
        ("index", 3, 7),  # 4 edges
        ("middle", 7, 11),  # 4 edges
        ("ring", 11, 15),  # 4 edges
        ("pinky", 15, 19),  # 4 edges
    ]
    DOT_COLOR = (255, 165, 0)

    @classmethod
    def get_transform_list(
        cls,
        mode: Literal[
            "cartesian",
            "keypoints_headframe_ypr",
            "keypoints_headframe_quat",
            "keypoints_wristframe_ypr",
            "keypoints_wristframe_quat",
        ],
        chunk_length: int = 100,
    ) -> list[Transform]:
        if mode == "cartesian":
            return _build_aria_cartesian_bimanual_transform_list(
                stride=cls.ACTION_STRIDE
            )
        elif mode == "keypoints_headframe_ypr":
            return _build_aria_keypoints_bimanual_transform_list(
                stride=cls.ACTION_STRIDE, chunk_length=chunk_length, is_quat=False
            )
        elif mode == "keypoints_headframe_quat":
            return _build_aria_keypoints_bimanual_transform_list(
                stride=cls.ACTION_STRIDE, chunk_length=chunk_length, is_quat=True
            )
        elif mode == "keypoints_wristframe_ypr":
            return _build_aria_keypoints_eef_frame_transform_list(
                stride=cls.ACTION_STRIDE, chunk_length=chunk_length, is_quat=False
            )
        elif mode == "keypoints_wristframe_quat":
            return _build_aria_keypoints_eef_frame_transform_list(
                stride=cls.ACTION_STRIDE, chunk_length=chunk_length, is_quat=True
            )
        else:
            raise ValueError(
                f"Unsupported mode '{mode}'. Expected one of: 'cartesian', 'keypoints_headframe_ypr', 'keypoints_headframe_quat', 'keypoints_wristframe_ypr', 'keypoints_wristframe_quat'."
            )


# MediaPipe 21-landmark ordering (used by Mecka and Scale hand tracking):
#   0: WRIST
#   1-4:   thumb  (CMC, MCP, IP, TIP)
#   5-8:   index  (MCP, PIP, DIP, TIP)
#   9-12:  middle (MCP, PIP, DIP, TIP)
#   13-16: ring   (MCP, PIP, DIP, TIP)
#   17-20: pinky  (MCP, PIP, DIP, TIP)
_MEDIAPIPE_FINGER_EDGES = [
    # thumb (4 edges)
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    # index (4 edges)
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    # middle (4 edges)
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    # ring (4 edges)
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    # pinky (4 edges)
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]
_MEDIAPIPE_FINGER_EDGE_RANGES = [
    ("thumb", 0, 4),
    ("index", 4, 8),
    ("middle", 8, 12),
    ("ring", 12, 16),
    ("pinky", 16, 20),
]


class Scale(Aria):
    VIZ_INTRINSICS_KEY = "scale"
    ACTION_STRIDE = 1
    FINGER_EDGES = _MEDIAPIPE_FINGER_EDGES
    FINGER_EDGE_RANGES = _MEDIAPIPE_FINGER_EDGE_RANGES


class Mecka(Aria):
    VIZ_INTRINSICS_KEY = "mecka"
    ACTION_STRIDE = 1
    FINGER_EDGES = _MEDIAPIPE_FINGER_EDGES
    FINGER_EDGE_RANGES = _MEDIAPIPE_FINGER_EDGE_RANGES


# this works for quat and ypr since actionChunkCoordinateFrameTransform works for both
def _build_aria_keypoints_revert_eef_frame_transform_list(
    *,
    action_key: str = "actions_keypoints",
    obs_key: str = "observations.state.keypoints",
    left_keypoints_action_wristframe: str = "left.action_keypoints_wristframe",
    right_keypoints_action_wristframe: str = "right.action_keypoints_wristframe",
    left_wrist_obs_headframe: str = "left.obs_wrist_pose_headframe",
    right_wrist_obs_headframe: str = "right.obs_wrist_pose_headframe",
    left_wrist_action_headframe: str = "left.action_wrist_pose_headframe",
    right_wrist_action_headframe: str = "right.action_wrist_pose_headframe",
    left_wrist_action_wristframe: str = "left.action_wrist_pose_wristframe",
    right_wrist_action_wristframe: str = "right.action_wrist_pose_wristframe",
    left_keypoints_action_headframe: str = "left.action_keypoints_headframe",
    right_keypoints_action_headframe: str = "right.action_keypoints_headframe",
    left_keypoints_obs_wristframe: str = "left.obs_keypoints_wristframe",
    right_keypoints_obs_wristframe: str = "right.obs_keypoints_wristframe",
    is_quat: bool = True,
) -> list[Transform]:
    if is_quat:
        pose_shape = 7
    else:
        pose_shape = 6
    transform_list = [
        SplitKeys(
            input_key=obs_key,
            output_key_list=[
                (left_wrist_obs_headframe, pose_shape),
                (left_keypoints_obs_wristframe, 63),
                (right_wrist_obs_headframe, pose_shape),
                (right_keypoints_obs_wristframe, 63),
            ],
        ),
        SplitKeys(
            input_key=action_key,
            output_key_list=[
                (left_wrist_action_wristframe, pose_shape),
                (left_keypoints_action_wristframe, 63),
                (right_wrist_action_wristframe, pose_shape),
                (right_keypoints_action_wristframe, 63),
            ],
        ),
        Reshape(
            input_key=left_keypoints_action_wristframe,
            output_key=left_keypoints_action_wristframe,
            shape=(100, 21, 3),
        ),
        Reshape(
            input_key=right_keypoints_action_wristframe,
            output_key=right_keypoints_action_wristframe,
            shape=(100, 21, 3),
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=left_wrist_obs_headframe,
            chunk_world=left_keypoints_action_wristframe,
            transformed_key_name=left_keypoints_action_headframe,
            mode="xyz",
            inverse=False,
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=right_wrist_obs_headframe,
            chunk_world=right_keypoints_action_wristframe,
            transformed_key_name=right_keypoints_action_headframe,
            mode="xyz",
            inverse=False,
        ),
        Reshape(
            input_key=left_keypoints_action_headframe,
            output_key=left_keypoints_action_headframe,
            shape=(100, 63),
        ),
        Reshape(
            input_key=right_keypoints_action_headframe,
            output_key=right_keypoints_action_headframe,
            shape=(100, 63),
        ),
        ConcatKeys(
            key_list=[
                left_keypoints_action_headframe,
                right_keypoints_action_headframe,
            ],
            new_key_name=action_key,
            delete_old_keys=True,
        ),
    ]
    return transform_list


def _build_aria_keypoints_eef_frame_transform_list(
    *,
    target_world: str = "obs_head_pose",
    target_world_ypr: str = "obs_head_pose_ypr",
    target_world_is_quat: bool = True,
    left_keypoints_action_world: str = "left.action_keypoints",
    right_keypoints_action_world: str = "right.action_keypoints",
    left_keypoints_obs_pose: str = "left.obs_keypoints",
    right_keypoints_obs_pose: str = "right.obs_keypoints",
    left_keypoints_action_headframe: str = "left.action_keypoints_headframe",
    right_keypoints_action_headframe: str = "right.action_keypoints_headframe",
    left_keypoints_obs_headframe: str = "left.obs_keypoints_headframe",
    right_keypoints_obs_headframe: str = "right.obs_keypoints_headframe",
    left_wrist_action_world: str = "left.action_wrist_pose",
    right_wrist_action_world: str = "right.action_wrist_pose",
    left_keypoints_action_wristframe: str = "left.action_keypoints_wristframe",
    right_keypoints_action_wristframe: str = "right.action_keypoints_wristframe",
    left_wrist_action_wristframe: str = "left.action_wrist_pose_wristframe",
    right_wrist_action_wristframe: str = "right.action_wrist_pose_wristframe",
    left_wrist_obs_pose: str = "left.obs_wrist_pose",
    right_wrist_obs_pose: str = "right.obs_wrist_pose",
    left_wrist_action_headframe: str = "left.action_wrist_pose_headframe",
    right_wrist_action_headframe: str = "right.action_wrist_pose_headframe",
    left_wrist_obs_headframe: str = "left.obs_wrist_pose_headframe",
    right_wrist_obs_headframe: str = "right.obs_wrist_pose_headframe",
    left_keypoints_obs_wristframe: str = "left.obs_keypoints_wristframe",
    right_keypoints_obs_wristframe: str = "right.obs_keypoints_wristframe",
    delete_target_world: bool = True,
    chunk_length: int = 100,
    stride: int = 3,
    is_quat: bool = True,
) -> list[Transform]:
    transform_list = _build_aria_keypoints_bimanual_transform_list(
        target_world=target_world,
        target_world_ypr=target_world_ypr,
        target_world_is_quat=target_world_is_quat,
        delete_target_world=delete_target_world,
        chunk_length=chunk_length,
        stride=stride,
        concat_keys=False,
        is_quat=True,
    )
    delete_keys = [
        left_keypoints_action_world,
        right_keypoints_action_world,
        left_keypoints_obs_pose,
        right_keypoints_obs_pose,
        left_wrist_action_world,
        right_wrist_action_world,
        left_wrist_obs_pose,
        right_wrist_obs_pose,
        left_keypoints_action_headframe,
        right_keypoints_action_headframe,
        left_keypoints_obs_headframe,
        right_keypoints_obs_headframe,
        left_wrist_action_headframe,
        right_wrist_action_headframe,
    ]
    if delete_target_world:
        delete_keys.append(target_world)
        if target_world_is_quat:
            delete_keys.append(target_world_ypr)
    transform_list.extend(
        [
            Reshape(
                input_key=left_keypoints_action_headframe,
                output_key=left_keypoints_action_headframe,
                shape=(chunk_length, 21, 3),
            ),
            Reshape(
                input_key=right_keypoints_action_headframe,
                output_key=right_keypoints_action_headframe,
                shape=(chunk_length, 21, 3),
            ),
            ActionChunkCoordinateFrameTransform(
                target_world=left_wrist_obs_headframe,
                chunk_world=left_keypoints_action_headframe,
                transformed_key_name=left_keypoints_action_wristframe,
                mode="xyz",
            ),
            ActionChunkCoordinateFrameTransform(
                target_world=right_wrist_obs_headframe,
                chunk_world=right_keypoints_action_headframe,
                transformed_key_name=right_keypoints_action_wristframe,
                mode="xyz",
            ),
            Reshape(
                input_key=left_keypoints_action_wristframe,
                output_key=left_keypoints_action_wristframe,
                shape=(chunk_length, 63),
            ),
            Reshape(
                input_key=right_keypoints_action_wristframe,
                output_key=right_keypoints_action_wristframe,
                shape=(chunk_length, 63),
            ),
            Reshape(
                input_key=left_keypoints_obs_headframe,
                output_key=left_keypoints_obs_headframe,
                shape=(21, 3),
            ),
            Reshape(
                input_key=right_keypoints_obs_headframe,
                output_key=right_keypoints_obs_headframe,
                shape=(21, 3),
            ),
            PoseCoordinateFrameTransform(
                target_world=left_wrist_obs_headframe,
                pose_world=left_keypoints_obs_headframe,
                transformed_key_name=left_keypoints_obs_wristframe,
                mode="xyz",
            ),
            PoseCoordinateFrameTransform(
                target_world=right_wrist_obs_headframe,
                pose_world=right_keypoints_obs_headframe,
                transformed_key_name=right_keypoints_obs_wristframe,
                mode="xyz",
            ),
            Reshape(
                input_key=left_keypoints_obs_wristframe,
                output_key=left_keypoints_obs_wristframe,
                shape=(63,),
            ),
            Reshape(
                input_key=right_keypoints_obs_wristframe,
                output_key=right_keypoints_obs_wristframe,
                shape=(63,),
            ),
            ActionChunkCoordinateFrameTransform(
                target_world=left_wrist_obs_headframe,
                chunk_world=left_wrist_action_headframe,
                transformed_key_name=left_wrist_action_wristframe,
                mode="xyzwxyz",
            ),
            ActionChunkCoordinateFrameTransform(
                target_world=right_wrist_obs_headframe,
                chunk_world=right_wrist_action_headframe,
                transformed_key_name=right_wrist_action_wristframe,
                mode="xyzwxyz",
            ),
        ]
    )
    if not is_quat:
        transform_list.extend(
            [
                BatchQuaternionPoseToYPR(
                    pose_key=left_wrist_action_wristframe,
                    output_key=left_wrist_action_wristframe,
                ),
                BatchQuaternionPoseToYPR(
                    pose_key=right_wrist_action_wristframe,
                    output_key=right_wrist_action_wristframe,
                ),
                QuaternionPoseToYPR(
                    pose_key=left_wrist_obs_headframe,
                    output_key=left_wrist_obs_headframe,
                ),
                QuaternionPoseToYPR(
                    pose_key=right_wrist_obs_headframe,
                    output_key=right_wrist_obs_headframe,
                ),
            ]
        )
    transform_list.extend(
        [
            ConcatKeys(
                key_list=[
                    left_wrist_action_wristframe,
                    left_keypoints_action_wristframe,
                    right_wrist_action_wristframe,
                    right_keypoints_action_wristframe,
                ],
                new_key_name="actions_keypoints",
                delete_old_keys=True,
            ),
            ConcatKeys(
                key_list=[
                    left_wrist_obs_headframe,
                    left_keypoints_obs_wristframe,
                    right_wrist_obs_headframe,
                    right_keypoints_obs_wristframe,
                ],
                new_key_name="observations.state.keypoints",
                delete_old_keys=True,
            ),
            DeleteKeys(keys_to_delete=delete_keys),
        ]
    )
    return transform_list


def _build_aria_keypoints_bimanual_transform_list(
    *,
    target_world: str = "obs_head_pose",
    target_world_ypr: str = "obs_head_pose_ypr",
    target_world_is_quat: bool = True,
    left_keypoints_action_world: str = "left.action_keypoints",
    right_keypoints_action_world: str = "right.action_keypoints",
    left_keypoints_obs_pose: str = "left.obs_keypoints",
    right_keypoints_obs_pose: str = "right.obs_keypoints",
    left_keypoints_action_headframe: str = "left.action_keypoints_headframe",
    right_keypoints_action_headframe: str = "right.action_keypoints_headframe",
    left_keypoints_obs_headframe: str = "left.obs_keypoints_headframe",
    right_keypoints_obs_headframe: str = "right.obs_keypoints_headframe",
    left_wrist_action_world: str = "left.action_wrist_pose",
    right_wrist_action_world: str = "right.action_wrist_pose",
    left_wrist_obs_pose: str = "left.obs_wrist_pose",
    right_wrist_obs_pose: str = "right.obs_wrist_pose",
    left_wrist_action_headframe: str = "left.action_wrist_pose_headframe",
    right_wrist_action_headframe: str = "right.action_wrist_pose_headframe",
    left_wrist_obs_headframe: str = "left.obs_wrist_pose_headframe",
    right_wrist_obs_headframe: str = "right.obs_wrist_pose_headframe",
    delete_target_world: bool = True,
    chunk_length: int = 100,
    stride: int = 3,
    concat_keys: bool = True,
    is_quat: bool = True,
) -> list[Transform]:
    keys_to_delete = list(
        {
            left_keypoints_action_world,
            right_keypoints_action_world,
            left_keypoints_obs_pose,
            right_keypoints_obs_pose,
            left_wrist_action_world,
            right_wrist_action_world,
            left_wrist_obs_pose,
            right_wrist_obs_pose,
            left_keypoints_action_headframe,
            right_keypoints_action_headframe,
            left_keypoints_obs_headframe,
            right_keypoints_obs_headframe,
            left_wrist_action_headframe,
            right_wrist_action_headframe,
            left_wrist_obs_headframe,
            right_wrist_obs_headframe,
        }
    )
    if delete_target_world:
        keys_to_delete.append(target_world)
        if target_world_is_quat:
            keys_to_delete.append(target_world_ypr)
    transform_list: list[Transform] = [
        Reshape(
            input_key=left_keypoints_action_world,
            output_key=left_keypoints_action_world,
            shape=(30, 21, 3),
        ),
        Reshape(
            input_key=right_keypoints_action_world,
            output_key=right_keypoints_action_world,
            shape=(30, 21, 3),
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=target_world,
            chunk_world=left_keypoints_action_world,
            transformed_key_name=left_keypoints_action_headframe,
            mode="xyz",
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=target_world,
            chunk_world=right_keypoints_action_world,
            transformed_key_name=right_keypoints_action_headframe,
            mode="xyz",
        ),
        Reshape(
            input_key=left_keypoints_obs_pose,
            output_key=left_keypoints_obs_pose,
            shape=(21, 3),
        ),
        Reshape(
            input_key=right_keypoints_obs_pose,
            output_key=right_keypoints_obs_pose,
            shape=(21, 3),
        ),
        PoseCoordinateFrameTransform(
            target_world=target_world,
            pose_world=left_keypoints_obs_pose,
            transformed_key_name=left_keypoints_obs_headframe,
            mode="xyz",
        ),
        PoseCoordinateFrameTransform(
            target_world=target_world,
            pose_world=right_keypoints_obs_pose,
            transformed_key_name=right_keypoints_obs_headframe,
            mode="xyz",
        ),
        Reshape(
            input_key=left_keypoints_obs_headframe,
            output_key=left_keypoints_obs_headframe,
            shape=(63,),
        ),
        Reshape(
            input_key=right_keypoints_obs_headframe,
            output_key=right_keypoints_obs_headframe,
            shape=(63,),
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=left_keypoints_action_headframe,
            output_action_key=left_keypoints_action_headframe,
            stride=stride,
            mode="xyz",
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=right_keypoints_action_headframe,
            output_action_key=right_keypoints_action_headframe,
            stride=stride,
            mode="xyz",
        ),
        Reshape(
            input_key=left_keypoints_action_headframe,
            output_key=left_keypoints_action_headframe,
            shape=(chunk_length, 63),
        ),
        Reshape(
            input_key=right_keypoints_action_headframe,
            output_key=right_keypoints_action_headframe,
            shape=(chunk_length, 63),
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=target_world,
            chunk_world=left_wrist_action_world,
            transformed_key_name=left_wrist_action_headframe,
            mode="xyzwxyz",
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=target_world,
            chunk_world=right_wrist_action_world,
            transformed_key_name=right_wrist_action_headframe,
            mode="xyzwxyz",
        ),
        PoseCoordinateFrameTransform(
            target_world=target_world,
            pose_world=left_wrist_obs_pose,
            transformed_key_name=left_wrist_obs_headframe,
            mode="xyzwxyz",
        ),
        PoseCoordinateFrameTransform(
            target_world=target_world,
            pose_world=right_wrist_obs_pose,
            transformed_key_name=right_wrist_obs_headframe,
            mode="xyzwxyz",
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=left_wrist_action_headframe,
            output_action_key=left_wrist_action_headframe,
            stride=stride,
            mode="xyzwxyz",
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=right_wrist_action_headframe,
            output_action_key=right_wrist_action_headframe,
            stride=stride,
            mode="xyzwxyz",
        ),
    ]
    if not is_quat:
        transform_list.extend(
            [
                BatchQuaternionPoseToYPR(
                    pose_key=left_wrist_action_headframe,
                    output_key=left_wrist_action_headframe,
                ),
                BatchQuaternionPoseToYPR(
                    pose_key=right_wrist_action_headframe,
                    output_key=right_wrist_action_headframe,
                ),
                QuaternionPoseToYPR(
                    pose_key=left_wrist_obs_headframe,
                    output_key=left_wrist_obs_headframe,
                ),
                QuaternionPoseToYPR(
                    pose_key=right_wrist_obs_headframe,
                    output_key=right_wrist_obs_headframe,
                ),
            ]
        )
    if concat_keys:
        transform_list.extend(
            [
                ConcatKeys(
                    key_list=[
                        left_wrist_action_headframe,
                        left_keypoints_action_headframe,
                        right_wrist_action_headframe,
                        right_keypoints_action_headframe,
                    ],
                    new_key_name="actions_keypoints",
                    delete_old_keys=True,
                ),
                ConcatKeys(
                    key_list=[
                        left_wrist_obs_headframe,
                        left_keypoints_obs_headframe,
                        right_wrist_obs_headframe,
                        right_keypoints_obs_headframe,
                    ],
                    new_key_name="observations.state.keypoints",
                    delete_old_keys=True,
                ),
                DeleteKeys(keys_to_delete=keys_to_delete),
            ]
        )
    return transform_list


def _build_aria_cartesian_bimanual_transform_list(
    *,
    target_world: str = "obs_head_pose",
    target_world_ypr: str = "obs_head_pose_ypr",
    target_world_is_quat: bool = True,
    left_action_world: str = "left.action_ee_pose",
    right_action_world: str = "right.action_ee_pose",
    left_obs_pose: str = "left.obs_ee_pose",
    right_obs_pose: str = "right.obs_ee_pose",
    left_action_headframe: str = "left.action_ee_pose_headframe",
    right_action_headframe: str = "right.action_ee_pose_headframe",
    left_obs_headframe: str = "left.obs_ee_pose_headframe",
    right_obs_headframe: str = "right.obs_ee_pose_headframe",
    actions_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    chunk_length: int = 100,
    stride: int = 3,
    delete_target_world: bool = True,
) -> list[Transform]:
    """Canonical ARIA bimanual transform pipeline used by tests and notebooks.

    Aria human data does not have commanded ee poses; action chunks are built
    from stacked observed ee poses (typically with a horizon on
    ``left/right.action_ee_pose`` mapped from ``left/right.obs_ee_pose``).
    """
    keys_to_delete = list(
        {
            left_action_world,
            right_action_world,
            left_obs_pose,
            right_obs_pose,
        }
    )
    target_pose_key = target_world
    if delete_target_world:
        keys_to_delete.append(target_world)
        if target_world_is_quat:
            keys_to_delete.append(target_world_ypr)

    transform_list: list[Transform] = [
        ActionChunkCoordinateFrameTransform(
            target_world=target_pose_key,
            chunk_world=left_action_world,
            transformed_key_name=left_action_headframe,
            mode="xyzwxyz",
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=target_pose_key,
            chunk_world=right_action_world,
            transformed_key_name=right_action_headframe,
            mode="xyzwxyz",
        ),
        PoseCoordinateFrameTransform(
            target_world=target_pose_key,
            pose_world=left_obs_pose,
            transformed_key_name=left_obs_headframe,
            mode="xyzwxyz",
        ),
        PoseCoordinateFrameTransform(
            target_world=target_pose_key,
            pose_world=right_obs_pose,
            transformed_key_name=right_obs_headframe,
            mode="xyzwxyz",
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=left_action_headframe,
            output_action_key=left_action_headframe,
            stride=stride,
            mode="xyzwxyz",
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=right_action_headframe,
            output_action_key=right_action_headframe,
            stride=stride,
            mode="xyzwxyz",
        ),
    ]

    if target_world_is_quat:
        transform_list.append(
            XYZWXYZ_to_XYZYPR(
                keys=[
                    left_action_headframe,
                    right_action_headframe,
                    left_obs_headframe,
                    right_obs_headframe,
                ]
            )
        )

    transform_list.extend(
        [
            ConcatKeys(
                key_list=[left_action_headframe, right_action_headframe],
                new_key_name=actions_key,
                delete_old_keys=True,
            ),
            ConcatKeys(
                key_list=[left_obs_headframe, right_obs_headframe],
                new_key_name=obs_key,
                delete_old_keys=True,
            ),
            DeleteKeys(keys_to_delete=keys_to_delete),
        ]
    )
    return transform_list
