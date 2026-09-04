#!/usr/bin/env python3
"""
EgoDex dataset loader
Implements 48-dimensional hand action representation, single-view images and language embedding data loading

-- adapted from H-RDT
"""

import h5py
import numpy as np
import torch
import os
import cv2
from pathlib import Path
import warnings
import random
import json
from torchcodec.decoders import VideoDecoder
import time
import pickle
from scipy.spatial.transform import Rotation as R
from psi.data.egodex.utils.data_utils import convert_to_camera_frame

warnings.filterwarnings("ignore")

def d9_to_mat44(nine_d):
    """
    Convert 9D representation back to 4x4 transformation matrix
    
    Args:
        nine_d: 9D vector [position(3) + rotation_6d(6)]
               rotation_6d is first two columns of rotation matrix
    
    Returns:
        mat44: 4x4 transformation matrix
    """
    position = nine_d[:3]
    rot_col0 = nine_d[3:6]
    rot_col1 = nine_d[6:9]
    
    # Gram-Schmidt orthogonalization to ensure valid rotation matrix
    col0 = rot_col0 / (np.linalg.norm(rot_col0) + 1e-8)
    col1 = rot_col1 - np.dot(rot_col1, col0) * col0
    col1 = col1 / (np.linalg.norm(col1) + 1e-8)
    col2 = np.cross(col0, col1)
    
    mat44 = np.eye(4, dtype=nine_d.dtype)
    mat44[:3, 0] = col0
    mat44[:3, 1] = col1
    mat44[:3, 2] = col2
    mat44[:3, 3] = position
    return mat44

def delta_rpy_from_tfs(tfs):
    """
    Compute delta roll, pitch, yaw between each consecutive transformation matrix.

    Args:
        tfs: np.ndarray of shape (N, 4, 4)

    Returns:
        delta_rpy: np.ndarray of shape (N-1, 3), where each row is (d_roll, d_pitch, d_yaw)
    """
    N = tfs.shape[0]
    delta_rpy = np.zeros((N-1, 3), dtype=np.float32)
    for i in range(N-1):
        # Get rotation matrices
        R1 = tfs[i][:3, :3]
        R2 = tfs[i+1][:3, :3]
        # Relative rotation: R_rel = R2 * R1.T
        R_rel = R2 @ R1.T
        # Convert to euler angles (roll, pitch, yaw)
        rpy = R.from_matrix(R_rel).as_euler('xyz', degrees=False)
        delta_rpy[i] = rpy
    return delta_rpy

def points_to_camera(points_3d, cam_ext):
    # Convert to homogeneous coordinates
    points_3d_homo = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    # Transform to camera coordinates using extrinsics
    points_cam = (np.linalg.inv(cam_ext) @ points_3d_homo.T).T
    # Get 3D point sin camera frame
    points_cam_3d = points_cam[:, :3]
    return points_cam_3d

def convert_to_delta_actions(actions, chunk_size, cam_ext):
    left_wrist = np.empty((chunk_size,4,4), dtype=np.float32)
    right_wrist = np.empty((chunk_size,4,4), dtype=np.float32)

    left_hand_finger_tips = {
        "leftThumbTip": np.empty((chunk_size, 3), dtype=np.float32),
        "leftIndexFingerTip": np.empty((chunk_size, 3), dtype=np.float32),
        "leftMiddleFingerTip": np.empty((chunk_size, 3), dtype=np.float32),
        "leftRingFingerTip": np.empty((chunk_size, 3), dtype=np.float32),
        "leftLittleFingerTip": np.empty((chunk_size, 3), dtype=np.float32),
    }
    right_hand_finger_tips = {
        "rightThumbTip": np.empty((chunk_size, 3), dtype=np.float32),
        "rightIndexFingerTip": np.empty((chunk_size, 3), dtype=np.float32),
        "rightMiddleFingerTip": np.empty((chunk_size, 3), dtype=np.float32),
        "rightRingFingerTip": np.empty((chunk_size, 3), dtype=np.float32),
        "rightLittleFingerTip": np.empty((chunk_size, 3), dtype=np.float32),
    }

    for i in range(actions.shape[0]):
        # action = actions[i]
        left_wrist[i] = d9_to_mat44(actions[i, :9])
        left_hand_finger_tips["leftThumbTip"][i] = actions[i, 9:12]
        left_hand_finger_tips["leftIndexFingerTip"][i] = actions[i, 12:15]
        left_hand_finger_tips["leftMiddleFingerTip"][i] = actions[i, 15:18]
        left_hand_finger_tips["leftRingFingerTip"][i] = actions[i, 18:21]
        left_hand_finger_tips["leftLittleFingerTip"][i] = actions[i, 21:24]

        right_wrist[i] = d9_to_mat44(actions[i, 24:33])
        right_hand_finger_tips["rightThumbTip"][i] = actions[i, 33:36]
        right_hand_finger_tips["rightIndexFingerTip"][i] = actions[i, 36:39]
        right_hand_finger_tips["rightMiddleFingerTip"][i] = actions[i, 39:42]
        right_hand_finger_tips["rightRingFingerTip"][i] = actions[i, 42:45]
        right_hand_finger_tips["rightLittleFingerTip"][i] = actions[i, 45:48]

    left_wrist_tfs_in_cam = convert_to_camera_frame(left_wrist, cam_ext) # (N,4,4)
    right_wrist_tfs_in_cam = convert_to_camera_frame(right_wrist, cam_ext)

    # convert to deleta action representations
    delta_left_wrist_xyz = left_wrist_tfs_in_cam[1:, :3, 3] - left_wrist_tfs_in_cam[:-1, :3, 3]
    delta_left_wrist_rpy = delta_rpy_from_tfs(left_wrist_tfs_in_cam)

    delta_right_wrist_xyz = right_wrist_tfs_in_cam[1:, :3, 3] - right_wrist_tfs_in_cam[:-1, :3, 3]
    delta_right_wrist_rpy = delta_rpy_from_tfs(right_wrist_tfs_in_cam)

    leftThumbTip = points_to_camera(left_hand_finger_tips["leftThumbTip"], cam_ext)
    delta_left_thumbtip = leftThumbTip[1:] - leftThumbTip[:-1]
    leftIndexFingerTip = points_to_camera(left_hand_finger_tips["leftIndexFingerTip"], cam_ext)
    delta_left_indextip = leftIndexFingerTip[1:] - leftIndexFingerTip[:-1]
    leftMiddleFingerTip = points_to_camera(left_hand_finger_tips["leftMiddleFingerTip"], cam_ext)
    delta_left_middletip = leftMiddleFingerTip[1:] - leftMiddleFingerTip[:-1]
    leftRingFingerTip = points_to_camera(left_hand_finger_tips["leftRingFingerTip"], cam_ext)
    delta_left_ringtip = leftRingFingerTip[1:] - leftRingFingerTip[:-1]
    leftLittleFingerTip = points_to_camera(left_hand_finger_tips["leftLittleFingerTip"], cam_ext)
    delta_left_littletip = leftLittleFingerTip[1:] - leftLittleFingerTip[:-1]

    rightThumbTip = points_to_camera(right_hand_finger_tips["rightThumbTip"], cam_ext)
    delta_right_thumbtip = rightThumbTip[1:] - rightThumbTip[:-1]   
    rightIndexFingerTip = points_to_camera(right_hand_finger_tips["rightIndexFingerTip"], cam_ext)
    delta_right_indextip = rightIndexFingerTip[1:] - rightIndexFingerTip[:-1]
    rightMiddleFingerTip = points_to_camera(right_hand_finger_tips["rightMiddleFingerTip"], cam_ext)
    delta_right_middletip = rightMiddleFingerTip[1:] - rightMiddleFingerTip[:-1]
    rightRingFingerTip = points_to_camera(right_hand_finger_tips["rightRingFingerTip"], cam_ext)
    delta_right_ringtip = rightRingFingerTip[1:] - rightRingFingerTip[:-1]
    rightLittleFingerTip = points_to_camera(right_hand_finger_tips["rightLittleFingerTip"], cam_ext)
    delta_right_littletip = rightLittleFingerTip[1:] - rightLittleFingerTip[:-1]

    # re-construct the delta actions
    delta_actions = np.concatenate([
        delta_left_wrist_xyz,
        delta_left_wrist_rpy,
        np.zeros_like(delta_left_wrist_rpy),
        delta_left_thumbtip,
        delta_left_indextip,
        delta_left_middletip,
        delta_left_ringtip,
        delta_left_littletip,
        delta_right_wrist_xyz,
        delta_right_wrist_rpy,
        np.zeros_like(delta_right_wrist_rpy),
        delta_right_thumbtip,
        delta_right_indextip,
        delta_right_middletip,
        delta_right_ringtip,
        delta_right_littletip,
    ], axis=1)  # (N-1, 48)
    return delta_actions

def convert_to_abs_actions(actions, chunk_size, cam_ext):
    """Absolute ee pose in the chunk frame-0 camera frame for frames 1..N-1.
    48D output layout, per hand [wrist_xyz(3), wrist_6d(6), tips(15)] (both hands -> 48):
    the wrist pose is kept ABSOLUTE in the frame-0 camera frame (no subtraction / no
    relative rotation), with rotation as 6D (first two columns of the rotation matrix).
    NOTE: this is a different representation from convert_to_delta_actions (which uses
    consecutive-frame deltas with rpy + zero-pad), so it needs its own normalization stats."""
    inv_ext = np.linalg.inv(cam_ext)
    out = []
    for base in (0, 24):
        wrist = np.empty((chunk_size, 4, 4), dtype=np.float32)
        for i in range(actions.shape[0]):
            wrist[i] = d9_to_mat44(actions[i, base:base + 9])
        wrist_cam = inv_ext[None] @ wrist
        xyz = wrist_cam[1:, :3, 3]
        six_d = np.concatenate([wrist_cam[1:, :3, 0], wrist_cam[1:, :3, 1]], axis=1)
        tips = points_to_camera(
            actions[:, base + 9:base + 24].reshape(-1, 3), cam_ext
        ).reshape(chunk_size, 15)[1:]
        out += [xyz, six_d, tips]
    return np.concatenate(out, axis=1)  # (N-1, 48)


class EgoDexDataset:
    """EgoDex dataset loader"""

    def __init__(
        self,
        data_root=None,
        upsample_rate=3,
        val=False,
        # use_precomp_lang_embed=True,
        chunk_size=1,
        img_history_size=1,
        # require_image=True,
        viz=False,
        use_delta_actions=False,
        use_abs_actions=False,
        load_retarget=False,
        data_downsample=1,
    ):
        """
        Args:
            data_root: Data root directory (e.g., "/share/hongzhe/datasets/egodex")
            upsample_rate: Temporal data upsampling rate (frame sampling interval)
            val: Whether it's validation set (True for test, False for train)
            use_precomp_lang_embed: Whether to use precomputed language embeddings
        """
        self.DATASET_NAME = "egodex"
        self.data_root = Path(data_root)
        self.upsample_rate = upsample_rate
        self.val = val
        self.viz = viz
        self.use_delta_actions = use_delta_actions
        self.use_abs_actions = use_abs_actions
        self.load_retarget = load_retarget
        # self.use_precomp_lang_embed = use_precomp_lang_embed
        # self.require_image = require_image

        self.chunk_size = chunk_size
        # self.state_dim = 48
        self.img_history_size = img_history_size
        self.data_downsample = data_downsample

        # Load data file list
        self.data_files = self._load_file_list()
        split_name = "test" if self.val else "train"
        print(f"Loaded {len(self.data_files)} {split_name} data files")
        # if len(self.data_files) == 0:
        #     raise ValueError(f"No data files found in {self.data_root} for split {split_name}")

    def get_dataset_name(self):
        """Return dataset name"""
        return self.DATASET_NAME

    def _load_file_list(self):
        """Load data file list, using cached file lists if available."""
        data_files = []

        def cache_path(part_dir):
            if self.load_retarget:
                part = part_dir.name
                return part_dir.parent.parent / "egodex_retargeting" / f"egodex_{part}_filelist.pkl"
            else:
                return part_dir / "filelist_cache.pkl"

        if not self.val:
            # Training set: part1-part5 + extra
            if self.load_retarget:
                parts = ["part1", "part2", "part3", "part4", "part5"]
            else:
                parts = ["part1", "part2", "part3", "part4", "part5", "extra"]
            for part in parts:
                part_dir = self.data_root / part
                if part_dir.exists():
                    cache_file = cache_path(part_dir)
                    if cache_file.exists():
                        try:
                            with open(cache_file, "rb") as f:
                                part_files = pickle.load(f)
                        except Exception as e:
                            print(f"Warning: Failed to load cache for {part_dir}: {e}, regenerating.")
                            part_files = self._scan_directory(part_dir)
                            with open(cache_file, "wb") as f:
                                pickle.dump(part_files, f)
                    else:
                        part_files = self._scan_directory(part_dir)
                        with open(cache_file, "wb") as f:
                            pickle.dump(part_files, f)
                    data_files.extend(part_files[::self.data_downsample])  # ablation: use 10% of egodex data for pretraining
        else:
            # Test set: test
            test_dir = self.data_root / "test"
            if test_dir.exists():
                cache_file = cache_path(test_dir)
                if cache_file.exists():
                    try:
                        with open(cache_file, "rb") as f:
                            test_files = pickle.load(f)
                    except Exception as e:
                        print(f"Warning: Failed to load cache for {test_dir}: {e}, regenerating.")
                        test_files = self._scan_directory(test_dir)
                        with open(cache_file, "wb") as f:
                            pickle.dump(test_files, f)
                else:
                    test_files = self._scan_directory(test_dir)
                    with open(cache_file, "wb") as f:
                        pickle.dump(test_files, f)
                data_files.extend(test_files[::self.data_downsample]) # ablation: use 10% of egodex data for pretraining

        return data_files

    def _scan_directory(self, directory):
        """Scan files in directory"""
        files = []
        for task_dir in directory.iterdir():
            if task_dir.is_dir():
                # Collect all triplets: hdf5, mp4, pt
                hdf5_files = list(task_dir.glob("*.hdf5"))
                for hdf5_file in hdf5_files:
                    file_index = hdf5_file.stem  # Get filename without extension
                    mp4_file = task_dir / f"{file_index}.mp4"
                    # pt_file = task_dir / f"{file_index}.pt"

                    # Ensure all required files exist
                    if hdf5_file.exists() and mp4_file.exists():  # and pt_file.exists():
                        files.append(
                            {
                                "hdf5": hdf5_file,
                                "mp4": mp4_file,
                                # "pt": pt_file,
                                "task": task_dir.name,
                                "file_index": file_index,
                            }
                        )

                # break # FIXME for debugging
        return files

    def construct_48d_action(self, hdf5_file, frame_indices):
        """
        Directly extract precomputed 48-dimensional hand action representation

        Args:
            hdf5_file: HDF5 file object
            frame_indices: List of frame indices to extract

        Returns:
            actions: (T, 57) action sequence
        """
        if "actions_48d" not in hdf5_file:
            raise ValueError(
                "Missing precomputed actions_48d data in HDF5 file, please run precompute_48d_actions.py first"
            )

        # Directly read precomputed 48-dimensional action data
        precomputed_actions = hdf5_file["actions_48d"][:]

        # Extract actions for specified frames
        selected_actions = precomputed_actions[frame_indices]

        return selected_actions.astype(np.float32)

    def parse_img_data(self, mp4_path, idx):
        """
        Args:
            mp4_path: MP4 file path
            idx: Current frame index

        Returns:
            frames: (img_history_size, H, W, 3) image frames
        """
        # cap = cv2.VideoCapture(str(mp4_path))
        # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        decoder = VideoDecoder(mp4_path, device='cpu', dimension_order='NHWC')
        total_frames = len(decoder)

        # Calculate sampling range following cvpr_real_dataset.py logic
        start_i = max(idx - self.img_history_size * self.upsample_rate + 1, 0)
        num_frames = (idx - start_i) // self.upsample_rate + 1

        frames = []

        try:
            for i, frame_idx in enumerate(range(start_i, idx + 1, self.upsample_rate)):
                if frame_idx < total_frames:
                    frame = decoder[frame_idx]
                    if frame is not None:
                        # BGR to RGB
                        frame = frame.cpu().numpy()
                        frames.append(frame)
                    else:
                        print(f"Warning: Not enough frames in {mp4_path}")
                        break
                else:
                    # If frame index exceeds total frames, use last valid frame
                    print(f"Warning: Frame index exceeds total frames in {mp4_path}")
                    break
        except Exception as e:
            raise ValueError(f"Error loading image frames: {e}")


        # Convert to numpy array
        frames = np.array(frames) # type: ignore

        # Pad if necessary (following cvpr_real_dataset.py logic)
        if frames.shape[0] < self.img_history_size:
            pad_frames = np.repeat(
                frames[:1], self.img_history_size - frames.shape[0], axis=0
            )
            frames = np.concatenate([pad_frames, frames], axis=0)

        return frames


    def __len__(self):
        return len(self.data_files)

    def get_item(self, idx=None):
        """
        Get a data sample

        Returns:
            Data dictionary containing all required fields
        """
        if idx is None:
            idx = random.randint(0, len(self.data_files) - 1)

        file_info = self.data_files[idx % len(self.data_files)]

        def full_path(path):
            if os.path.isabs(path):
                return path
            else:
                return self.data_root / path
            
        def reorder(x):
            # reorder 28d actions to align with he raw dataset
            # [left hand, right hand, left arm, right arm]
            return np.concatenate([
                x[:, 7:10],  # left hand
                x[:, 12:14],
                x[:, 10:12],
                x[:, 21:24],   # right hand
                x[:, 26:28],
                x[:, 24:26],
                x[:, 0:7],   # left arm
                x[:, 14:21],  # right arm
            ], axis=1)
        
        try:
            # Load HDF5 data
            with h5py.File(full_path(file_info["hdf5"]), "r") as root:
                # Get language instruction
                if root.attrs['llm_type'] == 'reversible':
                    direction = root.attrs['which_llm_description']
                    instruction = root.attrs['llm_description' if direction == '1' else 'llm_description2'] 
                    if isinstance(instruction, bytes):
                        instruction = instruction.decode("utf-8")
                    instruction = instruction.strip()
                else:
                    instruction = root.attrs['llm_description'] 
                    if isinstance(instruction, bytes):
                        instruction = instruction.decode("utf-8")
                    instruction = instruction.strip()
                assert instruction is not None
                
                # Get total number of frames
                transforms_group = root["transforms"]
                total_frames = list(transforms_group.values())[0].shape[0]

                # Calculate random index following cvpr_real_dataset.py logic
                max_index = total_frames - 2
                if max_index < 0:
                    print(f"Warning: Not enough frames in {file_info['hdf5']}")
                    return None

                # Random index for sampling
                index = random.randint(0, max_index)

                if self.load_retarget:
                    # load retargeted 28 actions
                    if "retarget_npz" not in file_info:
                        raise ValueError("Missing retarget_npz info in file list")
                    npz_file = self.data_root.parent / "egodex_retargeting" / file_info["retarget_npz"]
                    retarget_npz = np.load(npz_file)
                    pseudo_qpos = retarget_npz["g1_qpos"]
                    pseudo_qpos[:, 7:14] = retarget_npz["left_hand_qpos"]
                    pseudo_qpos[:, 21:28] = retarget_npz["right_hand_qpos"]
                    assert pseudo_qpos.shape[0] == total_frames
                    current_action = reorder(pseudo_qpos[index:index+1]) # (1, 28)
                else:
                    # Construct 48-dimensional actions using current index
                    current_action = self.construct_48d_action(root, [index]) # (1, 48)
                
                window_shifted = self.use_delta_actions or self.use_abs_actions
                chunk_size = self.chunk_size + 1 if window_shifted else self.chunk_size
                # Future action sequence
                action_end = min(
                    index + chunk_size * self.upsample_rate, 
                    max_index + 1
                )
                action_indices = list(
                    range(index if window_shifted else index + self.upsample_rate,
                          action_end,
                          self.upsample_rate)
                )

                # If not enough action frames, repeat the last one
                while len(action_indices) < chunk_size:
                    action_indices.append(
                        action_indices[-1] if action_indices else index + 1
                    )
            
                if self.load_retarget:
                    # load retargeted 28 actions
                    actions = reorder(pseudo_qpos[action_indices[:chunk_size]])  # (chunk_size, 28)
                else:
                    # Extract action sequence
                    actions = self.construct_48d_action(
                        root, action_indices[: chunk_size]
                    )

                if self.viz:
                    # for visualizer hands
                    from psi.data.egodex.utils.skeleton_tfs import RIGHT_FINGERS, RIGHT_INDEX, RIGHT_THUMB, RIGHT_RING, RIGHT_MIDDLE, RIGHT_LITTLE
                    from psi.data.egodex.utils.skeleton_tfs import LEFT_FINGERS, LEFT_INDEX, LEFT_THUMB, LEFT_RING, LEFT_MIDDLE, LEFT_LITTLE
                    query_tfs = RIGHT_FINGERS + ['rightHand', 'rightForearm'] + LEFT_FINGERS + ['leftHand', 'leftForearm']
                    tfdtype = root['/transforms/camera'][0].dtype  # type: ignore
                    tfs = np.zeros([len(query_tfs), 4, 4], dtype=tfdtype)
                    for i, tf_name in enumerate(query_tfs):
                        tfs[i] = root['/transforms/' + tf_name][index] # type: ignore

                cam_ext = np.array(root['/transforms/camera'][index]) # type: ignore , extrinsics
                cam_int = root['/camera/intrinsic'][:] # # type: ignore , intrinsics

                if self.use_abs_actions:
                    actions = convert_to_abs_actions(actions, chunk_size, cam_ext)
                elif self.use_delta_actions:
                    if self.load_retarget:
                        actions = actions[1:] - actions[:-1]
                    else:
                        actions = convert_to_delta_actions(actions, chunk_size, cam_ext)

            # Load single-view image frames using new sampling logic
            image_frames = self.parse_img_data(full_path(file_info['mp4']), index)
                        
            # Take only the required history size
            image_frames = image_frames[-self.img_history_size:]

            # Load language embedding
            # lang_embed_path = file_info["pt"]

            assert actions.shape[0] == self.chunk_size
            result = {
                "states": current_action,  # (1, 48)
                "actions": actions,  # (chunk_size, 48)
                "action_norm": np.ones_like(actions),  # Action indicator
                "current_images": [
                    image_frames
                ],  # [(img_history_size, H, W, 3)] single view
                "current_images_mask": [
                    np.ones(self.img_history_size, dtype=bool)
                ],  # Image mask
                "instruction": instruction,  # str(lang_embed_path),  # Language embedding file path
                "dataset_name": self.DATASET_NAME,
                "task": file_info["task"],
                "file_info": {
                    "hdf5_path": str(file_info["hdf5"]),
                    "mp4_path": str(file_info["mp4"]),
                    # "pt_path": str(file_info["pt"]),
                    "total_frames": total_frames,
                    "selected_index": index,
                    "action_indices": action_indices,
                },
                "dataset_name": "egodex"
            }
            if self.viz:
                result['viz'] = { 
                    'tfs': tfs,  # (num_tfs, 4, 4) # type: ignore
                    'cam_ext': cam_ext,  # (4, 4) # type: ignore
                    'cam_int': cam_int,  # (3, 3) # type: ignore
                } 
            return result

        except Exception as e:
            import traceback, sys
            traceback.print_exc()
            sys.stderr.flush()
            raise e
            # print(f"Error loading data {file_info['hdf5']}: {e}")
            return None

    def __getitem__(self, idx):
        """PyTorch Dataset interface"""
        """ print(f"egodex dataset __getitem__ called with idx={idx}")
        if not os.path.exists(f"tmp/{idx}.pkl"):
            print(f"missing {idx}.pkl, regenerating")
            data = self.get_item(idx)
            pickle.dump(data, open(f"tmp/{idx}.pkl", "wb"))
        else:
            data = pickle.load(open(f"tmp/{idx}.pkl", "rb"))
        return data """
        return self.get_item(idx)


if __name__ == "__main__":
    # Test dataset
    dataset = EgoDexDataset(data_root="./data/egodex", val=True, upsample_rate=3)

    print(f"Dataset size: {len(dataset)}")

    # Test loading samples
    sample = dataset.get_item(0)
    print("Sample data structure:")
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
