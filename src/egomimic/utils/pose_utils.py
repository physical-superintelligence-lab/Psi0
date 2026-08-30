import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def xyzw_to_wxyz(xyzw):
    return np.concatenate([xyzw[..., 3:4], xyzw[..., :3]], axis=-1)


def wxyz_to_xyzw(wxyz):
    return np.concatenate([wxyz[..., 1:4], wxyz[..., 0:1]], axis=-1)


def _interpolate_euler(seq: np.ndarray, chunk_length: int) -> np.ndarray:
    """Euler-aware interpolation for a single (T, 6) or (T, 7) sequence."""
    T, D = seq.shape
    assert D in (6, 7), f"Expected 6 or 7 dims, got {D}"

    if np.any(seq >= 1e8):
        return np.full((chunk_length, D), 1e9)

    old_time = np.linspace(0, 1, T)
    new_time = np.linspace(0, 1, chunk_length)

    trans_interp = interp1d(old_time, seq[:, :3], axis=0, kind="linear")(new_time)

    rot_unwrapped = np.unwrap(seq[:, 3:6], axis=0)
    rot_interp = interp1d(old_time, rot_unwrapped, axis=0, kind="linear")(new_time)
    rot_interp = (rot_interp + np.pi) % (2 * np.pi) - np.pi

    if D == 6:
        return np.concatenate([trans_interp, rot_interp], axis=-1)

    grip_interp = interp1d(old_time, seq[:, 6:7], axis=0, kind="linear")(new_time)
    return np.concatenate([trans_interp, rot_interp, grip_interp], axis=-1)


def _interpolate_linear(seq: np.ndarray, chunk_length: int) -> np.ndarray:
    """Simple linear interpolation for arbitrary (T, D) arrays."""
    T, _ = seq.shape
    if T == chunk_length:
        return seq
    old_time = np.linspace(0, 1, T)
    new_time = np.linspace(0, 1, chunk_length)
    return interp1d(old_time, seq, axis=0, kind="linear")(new_time)


def _interpolate_quat_wxyz(seq: np.ndarray, chunk_length: int) -> np.ndarray:
    """Quaternion-aware interpolation for a single (T, 7) sequence."""
    T, D = seq.shape
    if D != 7:
        raise ValueError(f"Expected 7 dims for xyz+quat(wxyz), got {D}")

    if np.any(seq >= 1e8):
        return np.full((chunk_length, D), 1e9)

    old_time = np.linspace(0, 1, T)
    new_time = np.linspace(0, 1, chunk_length)

    trans_interp = interp1d(old_time, seq[:, :3], axis=0, kind="linear")(new_time)
    quat_wxyz = np.asarray(seq[:, 3:7], dtype=np.float64)
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]

    norms = np.linalg.norm(quat_xyzw, axis=1, keepdims=True)
    zero_mask = (norms <= 0).squeeze(1)
    if np.any(zero_mask):
        # Replace zero-norm quaternions with identity (w=1, x=y=z=0 → xyzw: 0,0,0,1)
        quat_xyzw[zero_mask] = np.array([0.0, 0.0, 0.0, 1.0])
        norms[zero_mask] = 1.0
    quat_xyzw = quat_xyzw / norms

    # Enforce sign continuity to avoid long-path interpolation.
    quat_contiguous = quat_xyzw.copy()
    for i in range(1, T):
        if np.dot(quat_contiguous[i - 1], quat_contiguous[i]) < 0:
            quat_contiguous[i] = -quat_contiguous[i]

    if T == 1:
        quat_interp_xyzw = np.repeat(quat_contiguous[:1], chunk_length, axis=0)
    else:
        slerp = Slerp(old_time, R.from_quat(quat_contiguous))
        quat_interp_xyzw = slerp(new_time).as_quat()

    quat_interp_wxyz = quat_interp_xyzw[:, [3, 0, 1, 2]]
    dtype = seq.dtype if np.issubdtype(seq.dtype, np.floating) else np.float64
    return np.concatenate([trans_interp, quat_interp_wxyz], axis=-1).astype(
        dtype, copy=False
    )


def _interpolate_xyz(seq: np.ndarray, chunk_length: int) -> np.ndarray:
    """Linear interpolation for arbitrary (T, 3) arrays or (T, K, 3) arrays."""
    T = seq.shape[0]
    if T == chunk_length:
        return seq
    old_time = np.linspace(0, 1, T)
    new_time = np.linspace(0, 1, chunk_length)
    return interp1d(old_time, seq, axis=0, kind="linear")(new_time)


def _matrix_to_xyzypr(mats: np.ndarray) -> np.ndarray:
    """
    args:
        mats: (B, 4, 4) array of SE3 transformation matrices
    returns:
        (B, 6) np.array of [[x, y, z, yaw, pitch, roll]]
    """
    if mats.ndim != 3 or mats.shape[-2:] != (4, 4):
        raise ValueError(f"Expected (B, 4, 4) array, got shape {mats.shape}")

    mats = np.asarray(mats)
    dtype = mats.dtype if np.issubdtype(mats.dtype, np.floating) else np.float64

    xyz = mats[:, :3, 3]
    ypr = R.from_matrix(mats[:, :3, :3]).as_euler("ZYX", degrees=False)

    return np.concatenate([xyz, ypr], axis=-1).astype(dtype, copy=False)


def _xyzypr_to_matrix(xyzypr: np.ndarray) -> np.ndarray:
    """
    args:
        xyzypr: (B, 6) np.array of [[x, y, z, yaw, pitch, roll]]
    returns:
        (B, 4, 4) array of SE3 transformation matrices
    """
    if xyzypr.ndim != 2 or xyzypr.shape[-1] != 6:
        raise ValueError(f"Expected (B, 6) array, got shape {xyzypr.shape}")
    B = xyzypr.shape[0]
    dtype = xyzypr.dtype if np.issubdtype(xyzypr.dtype, np.floating) else np.float64

    mats = np.broadcast_to(np.eye(4, dtype=dtype), (B, 4, 4)).copy()
    mats[:, :3, :3] = R.from_euler("ZYX", xyzypr[:, 3:6], degrees=False).as_matrix()
    mats[:, :3, 3] = xyzypr[:, :3]
    return mats


def _matrix_to_xyzwxyz(mats: np.ndarray) -> np.ndarray:
    """
    args:
        mats: (B, 4, 4) array of SE3 transformation matrices
    returns:
        (B, 7) np.array of [[x, y, z, qw, qx, qy, qz]]
    """
    if mats.ndim != 3 or mats.shape[-2:] != (4, 4):
        raise ValueError(f"Expected (B, 4, 4) array, got shape {mats.shape}")

    mats = np.asarray(mats)
    dtype = mats.dtype if np.issubdtype(mats.dtype, np.floating) else np.float64

    xyz = mats[:, :3, 3]
    quat_xyzw = R.from_matrix(mats[:, :3, :3]).as_quat()
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]

    return np.concatenate([xyz, quat_wxyz], axis=-1).astype(dtype, copy=False)


def _xyzwxyz_to_matrix(xyzwxyz: np.ndarray) -> np.ndarray:
    """
    args:
        xyzwxyz: (B, 7) np.array of [[x, y, z, qw, qx, qy, qz]]
    returns:
        (B, 4, 4) array of SE3 transformation matrices
    """
    if xyzwxyz.ndim != 2 or xyzwxyz.shape[-1] != 7:
        raise ValueError(f"Expected (B, 7) array, got shape {xyzwxyz.shape}")

    B = xyzwxyz.shape[0]
    dtype = xyzwxyz.dtype if np.issubdtype(xyzwxyz.dtype, np.floating) else np.float64

    mats = np.broadcast_to(np.eye(4, dtype=dtype), (B, 4, 4)).copy()
    quat_xyzw = xyzwxyz[:, [4, 5, 6, 3]]

    # Replace zero-norm quaternions with identity (0,0,0,1 in xyzw) so scipy doesn't raise.
    norms = np.linalg.norm(quat_xyzw, axis=1)
    quat_xyzw[norms <= 0] = np.array([0.0, 0.0, 0.0, 1.0])

    mats[:, :3, :3] = R.from_quat(quat_xyzw).as_matrix()
    mats[:, :3, 3] = xyzwxyz[:, :3]

    return mats


def T_rot_orientation(T: np.ndarray, rot_orientation: np.ndarray) -> np.ndarray:
    """
    Permute the rotation matrix of a SE(3) transformation.
    """
    rot = T[:3, :3]
    rot = rot @ rot_orientation
    T[:3, :3] = rot
    return T


def _xyz_to_matrix(xyz: np.ndarray) -> np.ndarray:
    """
    args:
        xyz: (B, 3) np.array of [[x, y, z]]
    returns:
        (B, 4, 4) array of SE3 transformation matrices
    """
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError(f"Expected (B, 3) array, got shape {xyz.shape}")
    B = xyz.shape[0]
    dtype = xyz.dtype if np.issubdtype(xyz.dtype, np.floating) else np.float64
    mats = np.broadcast_to(np.eye(4, dtype=dtype), (B, 4, 4)).copy()
    mats[:, :3, 3] = xyz
    return mats


def _matrix_to_xyz(mats: np.ndarray) -> np.ndarray:
    """
    args:
        mats: (B, 4, 4) array of SE3 transformation matrices
    returns:
        (B, 3) np.array of [[x, y, z]]
    """
    if mats.ndim != 3 or mats.shape[-2:] != (4, 4):
        raise ValueError(f"Expected (B, 4, 4) array, got shape {mats.shape}")
    mats = np.asarray(mats)
    dtype = mats.dtype if np.issubdtype(mats.dtype, np.floating) else np.float64
    return mats[:, :3, 3].astype(dtype, copy=False)


def _split_action_pose(actions):
    # 14D layout: [L xyz ypr g, R xyz ypr g]
    # 12D layout: [L xyz ypr, R xyz ypr]
    if actions.shape[-1] == 14:
        left_xyz = actions[..., :3]
        left_ypr = actions[..., 3:6]
        right_xyz = actions[..., 7:10]
        right_ypr = actions[..., 10:13]
    elif actions.shape[-1] == 12:
        left_xyz = actions[..., :3]
        left_ypr = actions[..., 3:6]
        right_xyz = actions[..., 6:9]
        right_ypr = actions[..., 9:12]
    else:
        raise ValueError(f"Unsupported action dim {actions.shape[-1]}")
    return left_xyz, left_ypr, right_xyz, right_ypr


def _split_keypoints(keypoints, wrist_in_data: bool = False, is_quat: bool = True):
    if wrist_in_data:
        xyz_size = 3
        if is_quat:
            angle_size = 4
        else:
            angle_size = 3
        left_xyz_index = xyz_size
        left_angle_index = left_xyz_index + angle_size
        left_keypoints_index = left_angle_index + 21 * 3
        right_xyz_index = left_keypoints_index + xyz_size
        right_angle_index = right_xyz_index + angle_size
        right_keypoints_index = right_angle_index + 21 * 3
        return (
            keypoints[..., :left_xyz_index],
            keypoints[..., left_xyz_index:left_angle_index],
            keypoints[..., left_angle_index:left_keypoints_index],
            keypoints[..., left_keypoints_index:right_xyz_index],
            keypoints[..., right_xyz_index:right_angle_index],
            keypoints[..., right_angle_index:right_keypoints_index],
        )
    else:
        xyz_size = 3
        left_keypoints = keypoints[..., :63]
        right_keypoints = keypoints[..., 63:]
        return left_keypoints, right_keypoints
