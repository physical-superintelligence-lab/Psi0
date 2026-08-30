import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from egomimic.utils.egomimicUtils import (
    INTRINSICS,
    cam_frame_to_cam_pixels,
    draw_actions,
)
from egomimic.utils.pose_utils import _split_action_pose, _split_keypoints


class ColorPalette:
    Blues = "Blues"
    Greens = "Greens"
    Reds = "Reds"
    Oranges = "Oranges"
    Purples = "Purples"
    Greys = "Greys"

    @classmethod
    def is_valid(cls, name: str) -> bool:
        return name in vars(cls).values()

    @classmethod
    def to_rgb(cls, cmap_name: str, value: float = 0.7) -> tuple[int, int, int]:
        """Convert a ColorPalette cmap name to an RGB tuple (0-255).
        value: 0-1, where higher = darker shade."""
        rgba = plt.get_cmap(cmap_name)(value)
        return tuple(int(c * 255) for c in rgba[:3])


def _prepare_viz_image(img):
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))

    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=-1)
    elif img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    return img


def _viz_traj(images, actions, intrinsics_key, **kwargs):
    color = kwargs.get("color", "Blues")
    alpha = kwargs.get("alpha", 1.0)
    if not ColorPalette.is_valid(color):
        raise ValueError(f"Invalid color palette: {color}")

    images = _prepare_viz_image(images)
    intrinsics = INTRINSICS[intrinsics_key]
    left_xyz, _, right_xyz, _ = _split_action_pose(actions)

    base = images.copy()
    overlay = draw_actions(
        base.copy(),
        type="xyz",
        color=color,
        actions=left_xyz,
        extrinsics=None,
        intrinsics=intrinsics,
        arm="left",
    )
    overlay = draw_actions(
        overlay,
        type="xyz",
        color=color,
        actions=right_xyz,
        extrinsics=None,
        intrinsics=intrinsics,
        arm="right",
    )
    if alpha < 1.0:
        vis = cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0)
    else:
        vis = overlay
    return vis


def _viz_axes(images, actions, intrinsics_key, axis_len_m=0.04, **kwargs):
    alpha = kwargs.get("alpha", 1.0)
    images = _prepare_viz_image(images)
    intrinsics = INTRINSICS[intrinsics_key]
    left_xyz, left_ypr, right_xyz, right_ypr = _split_action_pose(actions)
    base = images.copy()
    vis = base.copy()

    def _draw_axis_color_legend(frame):
        _, w = frame.shape[:2]
        x_right = w - 12
        y_start = 14
        y_step = 12
        line_len = 24
        axis_legend = [
            ("x", (255, 0, 0)),
            ("y", (0, 255, 0)),
            ("z", (0, 0, 255)),
        ]
        for i, (name, color) in enumerate(axis_legend):
            y = y_start + i * y_step
            x0 = x_right - line_len
            x1 = x_right
            cv2.line(frame, (x0, y), (x1, y), color, 3)
            cv2.putText(
                frame,
                name,
                (x0 - 12, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
        return frame

    def _draw_rotation_at_anchor(
        frame, xyz_seq, ypr_seq, label, anchor_color, **kwargs
    ):
        if len(xyz_seq) == 0 or len(ypr_seq) == 0:
            return frame

        palm_xyz = xyz_seq[0]
        palm_ypr = ypr_seq[0]
        rot = R.from_euler("ZYX", palm_ypr, degrees=False).as_matrix()

        axis_points_cam = np.vstack(
            [
                palm_xyz,
                palm_xyz + rot[:, 0] * axis_len_m,
                palm_xyz + rot[:, 1] * axis_len_m,
                palm_xyz + rot[:, 2] * axis_len_m,
            ]
        )

        px = cam_frame_to_cam_pixels(axis_points_cam, intrinsics)[:, :2]
        if not np.isfinite(px).all():
            return frame
        pts = np.round(px).astype(np.int32)

        h, w = frame.shape[:2]
        x0, y0 = pts[0]
        if not (0 <= x0 < w and 0 <= y0 < h):
            return frame

        cv2.circle(frame, (x0, y0), 4, anchor_color, -1)
        axis_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for i, color in enumerate(axis_colors, start=1):
            x1, y1 = pts[i]
            if 0 <= x1 < w and 0 <= y1 < h:
                cv2.line(frame, (x0, y0), (x1, y1), color, 2)
                cv2.circle(frame, (x1, y1), 2, color, -1)

        cv2.putText(
            frame,
            label,
            (x0 + 6, max(12, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            anchor_color,
            1,
            cv2.LINE_AA,
        )
        return frame

    vis = _draw_rotation_at_anchor(vis, left_xyz, left_ypr, "L rot", (255, 180, 80))
    vis = _draw_rotation_at_anchor(vis, right_xyz, right_ypr, "R rot", (80, 180, 255))
    vis = _draw_axis_color_legend(vis)
    if alpha < 1.0:
        vis = cv2.addWeighted(vis, alpha, base, 1.0 - alpha, 0)
    return vis


def _viz_keypoints(
    images,
    actions,
    intrinsics_key,
    edges,
    colors,
    edge_ranges,
    dot_color=None,
    **kwargs,
):
    """Visualize all 21 MANO keypoints per hand, projected onto the image."""
    alpha = kwargs.get("alpha", 1.0)
    images = _prepare_viz_image(images)

    intrinsics = INTRINSICS[intrinsics_key]

    base = images.copy()
    vis = base.copy()
    h, w = vis.shape[:2]

    if actions.shape[-1] == 140:
        _, _, left_keypoints, _, _, right_keypoints = _split_keypoints(
            actions, wrist_in_data=True
        )
        hands = ("left", "right")
    elif actions.shape[-1] == 138:
        _, _, left_keypoints, _, _, right_keypoints = _split_keypoints(
            actions, wrist_in_data=True, is_quat=False
        )
        hands = ("left", "right")
    elif actions.shape[-1] == 70:
        # unimanual quat: 7 (wrist) + 63 (keypoints)
        left_keypoints = actions[..., 7:70]
        right_keypoints = None
        hands = ("left",)
    elif actions.shape[-1] == 69:
        # unimanual ypr: 6 (wrist) + 63 (keypoints)
        left_keypoints = actions[..., 6:69]
        right_keypoints = None
        hands = ("left",)
    else:
        left_keypoints, right_keypoints = _split_keypoints(actions, wrist_in_data=False)
        hands = ("left", "right")
    keypoints = {}
    keypoints["left"] = left_keypoints.reshape(-1, 3)
    if right_keypoints is not None:
        keypoints["right"] = right_keypoints.reshape(-1, 3)
    _default_dot_colors = {"left": (0, 120, 255), "right": (255, 80, 0)}
    for hand in hands:
        hand_dot_color = (
            dot_color if dot_color is not None else _default_dot_colors[hand]
        )
        kps_cam = keypoints[hand]
        # Camera frame -> pixels
        kps_px = cam_frame_to_cam_pixels(kps_cam, intrinsics)  # (42, 3+) 21 per arm

        # Identify valid keypoints (z > 0 and in image bounds)
        valid = kps_cam[:, 2] > 0.01
        valid &= (kps_px[:, 0] >= 0) & (kps_px[:, 0] < w)
        valid &= (kps_px[:, 1] >= 0) & (kps_px[:, 1] < h)

        # Draw skeleton edges (colored by finger)
        for finger, start, end in edge_ranges:
            color = colors[finger]
            for edge_idx in range(start, end):
                i, j = edges[edge_idx]
                if valid[i] and valid[j]:
                    p1 = (int(kps_px[i, 0]), int(kps_px[i, 1]))
                    p2 = (int(kps_px[j, 0]), int(kps_px[j, 1]))
                    cv2.line(vis, p1, p2, color, 2)

        # Draw keypoint dots on top
        for k in range(21):
            if valid[k]:
                center = (int(kps_px[k, 0]), int(kps_px[k, 1]))
                cv2.circle(vis, center, 4, hand_dot_color, -1)
                cv2.circle(vis, center, 4, (255, 255, 255), 1)  # white border

        # Label wrist
        if valid[0]:
            wrist_px = (int(kps_px[0, 0]) + 6, int(kps_px[0, 1]) - 6)
            cv2.putText(
                vis,
                f"{hand[0].upper()}",
                wrist_px,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                hand_dot_color,
                2,
            )

    if alpha < 1.0:
        vis = cv2.addWeighted(vis, alpha, base, 1.0 - alpha, 0)
    return vis


def save_image(image: np.ndarray, path: str) -> None:
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
