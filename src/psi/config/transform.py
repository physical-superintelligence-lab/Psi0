from __future__ import annotations
import os
import re
import base64
import copy
import json
import math
import random
import time
from io import BytesIO
from typing import (TYPE_CHECKING, Annotated, Any, Dict, List, Optional, Tuple,
                    Union)
from typing_extensions import Self, Annotated

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from torchvision.transforms import v2
from psi.config.augmentation import *
from psi.utils import get_asset_dir, pt_to_pil, resolve_path, pad_to_len

IGNORE_INDEX = -100


class RepackTransform(BaseModel):
    dataset_name: str = "default"
    goal_sampling: bool = False
    goal_future_offsets: List[float] = Field(default_factory=lambda: [1.0, 2.0, 3.0, 4.0])
    goal_end_segment_prob: float = 0.5

    # When True and meta/paraphrases.json was loaded for this dataset, render the
    # task/subtask prompt from a random member of {canonical}+full+simple per sample.
    use_paraphrases: bool = False
    _para_lookup: dict | None = PrivateAttr(default=None)

    def set_paraphrases(self, table: dict | None) -> None:
        if not table:
            self._para_lookup = None
            return
        lut: dict[str, dict[str, list[str]]] = {}
        for section in ("tasks", "subtasks"):
            lut[section] = {
                str(canon).strip().lower(): [str(canon), *v.get("full", []), *v.get("simple", [])]
                for canon, v in (table.get(section) or {}).items()
            }
        self._para_lookup = lut

    def _maybe_paraphrase(self, text: str, section: str, is_training: bool) -> str:
        if not (self.use_paraphrases and is_training) or self._para_lookup is None or not text:
            return text
        cands = self._para_lookup.get(section, {}).get(str(text).strip().lower())
        return random.choice(cands) if cands else text

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        return data

class LerobotRepackTransform(RepackTransform):

    def delta_timestamps(self, fps: int) -> dict[str, list[float]]:
        ...  

class ModelTransform(BaseModel):
    def __call__(self, data: dict[str, Any], no_aug: bool = False, **kwargs) -> dict[str, Any]:
        return data

class FieldTransform(BaseModel):

    @staticmethod
    def _bounds(field_transform) -> Tuple[np.ndarray, np.ndarray]:
        los, his = [], []
        # modality_dims: dict[str, int] = {}
        for action_key in field_transform.action_transform.apply_to:
            mode = field_transform.action_transform.normalization_modes.get(action_key, "min_max")
            stats = field_transform.action_transform.normalization_statistics.get(action_key, {})
            if not stats:
                continue
            if mode in ("min_max", "bounds"):
                lo = np.array(stats["min"], dtype=np.float32)
                hi = np.array(stats["max"], dtype=np.float32)
            elif mode in ("q99", "bounds_q99"):
                lo = np.array(stats["q01"], dtype=np.float32)
                hi = np.array(stats["q99"], dtype=np.float32)
            elif mode == "mean_std":
                std = np.array(stats["std"], dtype=np.float32)
                mean = np.array(stats["mean"], dtype=np.float32)
                lo, hi = mean - std, mean + std
            else:
                raise NotImplementedError(f"Unsupported normalization mode for denormalization: {mode}")
            los.append(lo)
            his.append(hi)
            # modality_dims[action_key] = lo.shape[0]
        if not los:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        return np.concatenate(los), np.concatenate(his) #, modality_dims

    def denormalize_L1_action_err(self, L1_err, dataset_name: list[str]):
        # overwrite by subclass
        ...

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        return data

class DataTransform(BaseModel):
    repack: RepackTransform
    model: ModelTransform
    field: FieldTransform

    def __call__(self, data:dict[str, Any], **kwargs) -> dict[str, Any]:
        data = self.repack(data, **kwargs)
        data = self.field(data, **kwargs)
        data = self.model(data, **kwargs)
        return data
    
class EgodexRepackTransform(BaseModel):
    dataset_name: str = "egodex"
    stage:str = "pretrain"  # "pretrain" or "postpre"
    pad_action_dim: int | None = None
    pad_state_dim: int | None = None

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        states, _ = pad_to_len(data["states"], self.pad_state_dim) if self.pad_state_dim is not None else (data["states"], None)
        if self.pad_action_dim is not None:
            actions, mask = pad_to_len(data["actions"], self.pad_action_dim) 
        else:
            actions = data["actions"]
            mask = np.ones_like(actions, dtype=bool)

        return dict(
            observations = [Image.fromarray(img[0]) for img in data["current_images"]],  # list of PIL Image
            states=states,  # (To, Da)
            actions=actions,  # (Tp, Da)
            instruction=data["instruction"].lower(),
            dataset=data.get("dataset_name", self.dataset_name),
            actions_mask=mask, #(Tp, Da)
        )
        
class EgoverseRepackTransform(BaseModel):
    """Repack raw EgoVerse data into the standard Psi 48 DoF action format.

    Input keys (from EgoVerseDataset / MultiDataset):
        observations.images.front_img_1  : torch.Tensor [3, H, W] float32 in [0,1]
        actions_keypoints                : torch.Tensor [T, 140]
        observations.state.keypoints     : torch.Tensor [140]
        embodiment                       : str  "aria" | "mecka" | "scale"
        metadata.task_description : str instruction

    140-dim layout (keypoints_headframe_quat):
        [0:7]    left wrist  (xyz + wxyz quaternion)
        [7:70]   left  21 keypoints x 3 = 63 dims (head frame)
        [70:77]  right wrist (xyz + wxyz quaternion)
        [77:140] right 21 keypoints x 3 = 63 dims (head frame)

    Output 48-dim (24 per arm):
        [0:3]   wrist xyz
        [3:9]   wrist 6D rotation (first two cols of R)
        [9:24]  5 fingertip xyz (thumb, index, middle, ring, pinky)
    """

    dataset_name: str = "egoverse"
    pad_action_dim: int | None = None
    pad_state_dim: int | None = None

    @staticmethod
    def _to_numpy(t) -> np.ndarray:
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return np.asarray(t, dtype=np.float32)

    @staticmethod
    def _quat_wxyz_to_rot6d(q: np.ndarray) -> np.ndarray:
        """q: (..., 4) wxyz → (..., 6) first two columns of rotation matrix."""
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        col0 = np.stack([1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y)], axis=-1)
        col1 = np.stack([2*(x*y - w*z), 1 - 2*(x*x + z*z), 2*(y*z + w*x)], axis=-1)
        return np.concatenate([col0, col1], axis=-1)

    @staticmethod
    def _fingertip_indices(embodiment: str) -> list:
        """Return 5 fingertip joint indices within the 21-joint array."""
        e = embodiment.lower()
        if "mecka" in e or "scale" in e:
            # MediaPipe ordering: THUMB_TIP=4, INDEX_TIP=8, MIDDLE_TIP=12, RING_TIP=16, PINKY_TIP=20
            return [4, 8, 12, 16, 20]
        # Aria ordering: THUMB_TIP=0, INDEX_TIP=1, MIDDLE_TIP=2, RING_TIP=3, PINKY_TIP=4
        return [0, 1, 2, 3, 4]

    def _convert_140_to_48(self, kp: np.ndarray, embodiment: str) -> np.ndarray:
        """(..., 140) keypoints → (..., 48) compact representation."""
        left_wrist = kp[..., 0:7]
        left_kp21  = kp[..., 7:70].reshape(*kp.shape[:-1], 21, 3)
        right_wrist = kp[..., 70:77]
        right_kp21  = kp[..., 77:140].reshape(*kp.shape[:-1], 21, 3)

        tip_idx = self._fingertip_indices(embodiment)

        def arm_24(wrist, kp21):
            xyz   = wrist[..., 0:3]
            rot6d = self._quat_wxyz_to_rot6d(wrist[..., 3:7])
            tips  = kp21[..., tip_idx, :].reshape(*kp21.shape[:-2], 15)
            return np.concatenate([xyz, rot6d, tips], axis=-1)

        return np.concatenate([arm_24(left_wrist, left_kp21),
                                arm_24(right_wrist, right_kp21)], axis=-1)

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        embodiment = data["embodiment"]

        # Image: [3, H, W] float32 [0,1] → PIL RGB
        img_np = self._to_numpy(data["observations.images.front_img_1"])
        img_hw = (img_np * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
        observations = [Image.fromarray(img_hw)]

        # State: [140] → [1, 48]
        state_kp = self._to_numpy(data["observations.state.keypoints"]).astype(np.float32)
        state_48 = self._convert_140_to_48(state_kp, embodiment)[np.newaxis]  # [1, 48]

        # Actions: [T, 140] → [T, 48]
        action_kp = self._to_numpy(data["actions_keypoints"]).astype(np.float32)
        actions_48 = self._convert_140_to_48(action_kp, embodiment)  # [T, 48]

        # Optional padding
        states, _ = (pad_to_len(state_48, self.pad_state_dim)
                     if self.pad_state_dim is not None else (state_48, None))
        if self.pad_action_dim is not None:
            actions, mask = pad_to_len(actions_48, self.pad_action_dim)
        else:
            actions = actions_48
            mask = np.ones(actions.shape, dtype=bool)

        instruction = data.get("metadata.task_description", "")

        return dict(
            observations=observations,          # list of PIL Image
            states=states,                      # [1, 48]
            actions=actions,                    # [T, 48]
            instruction=instruction.lower(),
            dataset=data.get("dataset_name", self.dataset_name),
            actions_mask=mask,                  # [T, 48]
        )

class HEPretrainRepackTransform(RepackTransform):
    dataset_name: str = "humanoid-everyday"
    num_past_frames: int = 0
    action_chunk_size: int = 1
    use_delta_actions: bool = True
    robot_type: str = "g1"
    # pad_action_dim: int | None = None
    # pad_state_dim: int | None = None

    def _to_numpy(self, value, dtype=np.float32):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=dtype)

    def _ensure_2d(self, value, dim: int, name: str):
        arr = self._to_numpy(value)
        if arr.ndim == 1:
            if arr.shape[0] != dim:
                raise ValueError(f"{name} expected shape ({dim},), got {arr.shape}")
            return arr.reshape(1, dim)
        if arr.ndim == 2:
            if arr.shape[1] != dim:
                raise ValueError(f"{name} expected shape (T,{dim}), got {arr.shape}")
            return arr
        raise ValueError(f"{name} expected 1D or 2D, got {arr.shape}")

    def _rpy_to_rot6d(self, rpy):
        from scipy.spatial.transform import Rotation as R
        rot = R.from_euler("xyz", rpy).as_matrix()
        return rot[:, :2].reshape(6)

    def _delta_rpy(self, rpy_seq):
        from scipy.spatial.transform import Rotation as R
        rot = R.from_euler("xyz", rpy_seq).as_matrix()
        rel = rot[1:] @ np.transpose(rot[:-1], (0, 2, 1))
        return R.from_matrix(rel).as_euler("xyz", degrees=False).astype(np.float32)

    def _build_actions(self, data: dict[str, Any]) -> np.ndarray:
        left_wrist_xyz = self._ensure_2d(
            data["action.wrists.left.xyz"], 3, "action.wrists.left.xyz"
        )
        left_wrist_rpy = self._ensure_2d(
            data["action.wrists.left.rpy"], 3, "action.wrists.left.rpy"
        )
        right_wrist_xyz = self._ensure_2d(
            data["action.wrists.right.xyz"], 3, "action.wrists.right.xyz"
        )
        right_wrist_rpy = self._ensure_2d(
            data["action.wrists.right.rpy"], 3, "action.wrists.right.rpy"
        )

        def tip_xyz(keys):
            for key in keys:
                if key in data:
                    return self._ensure_2d(data[key], 3, key)
            return np.zeros_like(left_wrist_xyz)

        left_thumb = tip_xyz(["action.hands.left_thumb.xyz"])
        left_index = tip_xyz(["action.hands.left_index.xyz"])
        left_middle = tip_xyz(
            ["action.hands.left_middle.xyz", "action.hands.left_middle_finger.xyz"]
        )
        left_ring = tip_xyz(["action.hands.left_ring_finger.xyz"])
        left_little = tip_xyz(["action.hands.left_little_finger.xyz"])

        right_thumb = tip_xyz(["action.hands.right_thumb.xyz"])
        right_index = tip_xyz(["action.hands.right_index.xyz"])
        right_middle = tip_xyz(
            ["action.hands.right_middle.xyz", "action.hands.right_middle_finger.xyz"]
        )
        right_ring = tip_xyz(["action.hands.right_ring_finger.xyz"])
        right_little = tip_xyz(["action.hands.right_little_finger.xyz"])

        if self.use_delta_actions:
            if left_wrist_xyz.shape[0] < 2:
                raise ValueError(
                    "use_delta_actions requires at least 2 action frames; "
                    "increase action_chunk_size or disable delta."
                )
            left_wrist_xyz = left_wrist_xyz[1:] - left_wrist_xyz[:-1]
            right_wrist_xyz = right_wrist_xyz[1:] - right_wrist_xyz[:-1]
            left_wrist_rpy = self._delta_rpy(left_wrist_rpy)
            right_wrist_rpy = self._delta_rpy(right_wrist_rpy)

            left_thumb = left_thumb[1:] - left_thumb[:-1]
            left_index = left_index[1:] - left_index[:-1]
            left_middle = left_middle[1:] - left_middle[:-1]
            left_ring = left_ring[1:] - left_ring[:-1]
            left_little = left_little[1:] - left_little[:-1]

            right_thumb = right_thumb[1:] - right_thumb[:-1]
            right_index = right_index[1:] - right_index[:-1]
            right_middle = right_middle[1:] - right_middle[:-1]
            right_ring = right_ring[1:] - right_ring[:-1]
            right_little = right_little[1:] - right_little[:-1]

            left_rot_pad = np.zeros_like(left_wrist_rpy)
            right_rot_pad = np.zeros_like(right_wrist_rpy)
        else:
            left_rot6d = np.stack(
                [self._rpy_to_rot6d(rpy) for rpy in left_wrist_rpy], axis=0
            )
            right_rot6d = np.stack(
                [self._rpy_to_rot6d(rpy) for rpy in right_wrist_rpy], axis=0
            )
            left_wrist_rpy = left_rot6d
            right_wrist_rpy = right_rot6d
            left_rot_pad = None
            right_rot_pad = None

        left_wrist = (
            np.concatenate([left_wrist_xyz, left_wrist_rpy, left_rot_pad], axis=1) #type:ignore
            if self.use_delta_actions
            else np.concatenate([left_wrist_xyz, left_wrist_rpy], axis=1)
        )
        right_wrist = (
            np.concatenate([right_wrist_xyz, right_wrist_rpy, right_rot_pad], axis=1) #type:ignore
            if self.use_delta_actions
            else np.concatenate([right_wrist_xyz, right_wrist_rpy], axis=1)
        )

        actions = np.concatenate(
            [
                left_wrist,
                left_thumb,
                left_index,
                left_middle,
                left_ring,
                left_little,
                right_wrist,
                right_thumb,
                right_index,
                right_middle,
                right_ring,
                right_little,
            ],
            axis=1,
        ).astype(np.float32)
        return actions

    def __call__(
        self,
        data: dict[str, Any],
        metadata=None,
        **kwargs,
    ) -> dict[str, Any]:
        hand_states = self._to_numpy(data["observation.hand_joints"])
        arm_states = self._to_numpy(data["observation.arm_joints"])
        if hand_states.ndim == 2:
            hand_states = hand_states[-1]
        if arm_states.ndim == 2:
            arm_states = arm_states[-1]
        states = np.concatenate((hand_states, arm_states), axis=0).astype(np.float32)

        try:
            actions = self._build_actions(data)
        except Exception as e:
            print("Error building actions:")
            print(data)

            import traceback
            traceback.print_exc()
            raise e

        instruction = ""
        if metadata is not None and "episode_index" in data:
            episode_idx = int(self._to_numpy(data["episode_index"]).reshape(-1)[0])
            if (
                hasattr(metadata, "episodes")
                and metadata.episodes is not None
                and episode_idx < len(metadata.episodes)
            ):
                instruction = metadata.episodes[episode_idx].get("instruction", "")
        if not instruction and "task" in data:
            instruction = data["task"]
        if isinstance(instruction, bytes):
            instruction = instruction.decode("utf-8")
            instruction = self.nice_instruction(instruction.split("/")[1])

        img = data["observation.images.egocentric"]
        if hasattr(img, "dim") and img.dim() == 4:
            img = img[0]
        elif isinstance(img, np.ndarray) and img.ndim == 4:
            img = img[0]

        return dict(
            observations=[pt_to_pil(img, normalized=False)],  # list of PIL Image
            states=states,
            actions=actions,
            instruction=str(instruction).lower(),
            dataset=data.get("dataset_name", self.dataset_name),
        )

    def nice_instruction(self, instruction: str) -> str:
        instruction = instruction.replace("_", " ").replace("-", " ")
        instruction = re.sub(r"\s+", " ", instruction)
        return instruction.strip()

    def delta_timestamps(self, fps: int):
        delta = {}
        delta["observation.hand_joints"] = [
            -t / fps for t in range(self.num_past_frames, -1, -1)
        ]
        delta["observation.arm_joints"] = [
            -t / fps for t in range(self.num_past_frames, -1, -1)
        ]

        action_keys = [
            "action.joint_angles",
            "action.wrists.left.xyz",
            "action.wrists.left.rpy",
            "action.wrists.right.xyz",
            "action.wrists.right.rpy",
            "action.hands.left_thumb.xyz",
            "action.hands.left_index.xyz",
            "action.hands.right_thumb.xyz",
            "action.hands.right_index.xyz",
        ]
        if self.robot_type == "h1":
            action_keys += [
                "action.hands.left_middle_finger.xyz",
                "action.hands.left_ring_finger.xyz",
                "action.hands.left_little_finger.xyz",
                "action.hands.right_middle_finger.xyz",
                "action.hands.right_ring_finger.xyz",
                "action.hands.right_little_finger.xyz",
            ]
        else:
            action_keys += [
                "action.hands.left_middle.xyz",
                "action.hands.right_middle.xyz",
            ]
        action_len = (
            self.action_chunk_size + 1 if self.use_delta_actions else self.action_chunk_size
        )
        for key in action_keys:
            delta[key] = [t / fps for t in range(action_len)]
        return delta

class RealRepackTransform(LerobotRepackTransform):
    dataset_name: str = "real"

    conditions: list[str] = Field(default_factory=lambda: [])
    image_keys: List[str] = Field(default_factory=lambda: ["observation.images.egocentric"])
    state_key: str = "states"
    action_key: str = "action"
    instruction_key: str = "task"

    num_past_frames: int = 0
    action_chunk_size: int = 30

    pad_action_dim: int | None = None
    pad_state_dim: int | None = None

    def delta_timestamps(self, fps):
        delta = {}
        for image_key in self.image_keys:
            delta[image_key] = [-t/fps for t in range(self.num_past_frames, -1, -1)]

        delta[self.state_key] = [-t/fps for t in range(self.num_past_frames, -1, -1)]
        delta[self.action_key] = [t/fps for t in range(self.action_chunk_size)]
        return delta

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        states, _ = pad_to_len(data[self.state_key], self.pad_state_dim) if self.pad_state_dim is not None else (data[self.state_key], None)

        if self.pad_action_dim is not None:
            actions, mask = pad_to_len(data[self.action_key], self.pad_action_dim)
        else:
            actions = data[self.action_key]
            mask = np.ones_like(actions, dtype=bool)

        result = {
            "observations": [ 
                pt_to_pil(data[key], normalized=False)
                for key in self.image_keys
            ], # list of PIL Image
            "states": np.array(states, dtype=np.float32), # (To, Da)
            "actions": np.array(actions, dtype=np.float32),  # (Tp, Da)
            "instruction": data[self.instruction_key].lower(),
            "actions_mask": mask, #(Tp, Da)
        } 
        return result

class SimpleRepackTransform(LerobotRepackTransform):
    dataset_name: str = "simple"

    num_past_frames: int = 0 # single current frame
    action_chunk_size: int = 30

    pad_action_dim: int | None = None
    pad_state_dim: int | None = None

    def delta_timestamps(self, fps: int):
        return {
            "observation.images.egocentric": [-t/fps for t in range(self.num_past_frames, -1, -1)],
            "states": [-t/fps for t in range(self.num_past_frames, -1, -1)],
            "action": [t/fps for t in range(self.action_chunk_size)]
        }
 

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        image_key = "observation.images.egocentric"
        states = data["states"].numpy() # (To, Ds)
        actions = data["action"].numpy() # (Tp, Da)
        action_is_pad = data["action_is_pad"].numpy() # (Ta,) 
        pad_mask = np.ones_like(actions) * (1. - action_is_pad[...,None].astype(np.float32))

        states, _ = pad_to_len(states, self.pad_state_dim) if self.pad_state_dim is not None else (states, None)
        if self.pad_action_dim is not None:
            actions, _ = pad_to_len(actions, self.pad_action_dim) 
            mask,_ = pad_to_len(pad_mask, self.pad_action_dim)
        else:
            mask = np.ones_like(actions, dtype=np.float32)

        return {
            "observations": [pt_to_pil(data[image_key],normalized=False)], # single view
            "states": states.astype(np.float32), # (To, Do)
            "actions": actions.astype(np.float32), # (Tp, Da)
            # "action_is_pad": np.array(data["action_is_pad"], dtype=bool),
            "instruction": data["task"].lower(),
            "actions_mask": mask
        }

class IdentityTransform(FieldTransform):
    # boilerplate transform config
    # define primitive data types only! eg., int, float, str
    # container types are allowed eg., list[int], dict[str, float]
    # default no-op transform
    # passing external dependency through **kwargs, eg., tokenizer, processor
    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        return data

class ActionStateTransform(FieldTransform):
    
    stat_path: str
    action_norm_type: str = "bounds"  # "bounds_q99"
    stat_action_key: str = "action"
    stat_state_key: str = "states"
    use_norm_mask: bool = False
    action_norm_masks: List[bool] = [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]  # delta-eef + gripper
    action_min: Optional[List[float]] = None
    action_max: Optional[List[float]] = None
    state_min: Optional[List[float]] = None
    state_max: Optional[List[float]] = None
    normalize_state: bool = False  # whether to normalize states

    pad_action_dim: int | None = None
    pad_state_dim: int | None = None

    def model_post_init(self, __context: Any) -> None:
        if not os.path.exists(resolve_path(self.stat_path)):
            return
        with open(resolve_path(self.stat_path), "r") as f:
            stats = json.load(f)
            self.populate_stats(stats)
        
    def populate_stats(self, stats: dict[str, Any]):
        if self.action_norm_type == "bounds_q99":
            self.action_min = stats[self.stat_action_key]["q01"]
            self.action_max = stats[self.stat_action_key]["q99"]
            if self.normalize_state:
                self.state_min = stats[self.stat_state_key]["q01"]
                self.state_max = stats[self.stat_state_key]["q99"]
        elif self.action_norm_type == "bounds":
            self.action_min = stats[self.stat_action_key]["min"]
            self.action_max = stats[self.stat_action_key]["max"]
            if self.normalize_state:
                self.state_min = stats[self.stat_state_key]["min"]
                self.state_max = stats[self.stat_state_key]["max"]
        else:
            raise ValueError(f"Unsupported action normalization type: {self.action_norm_type}")

        if self.pad_action_dim is not None:
            self.action_min= pad_to_len(np.array(self.action_min, dtype=np.float32), self.pad_action_dim, dim=0)[0].tolist()
            self.action_max = pad_to_len(np.array(self.action_max, dtype=np.float32), self.pad_action_dim, dim=0)[0].tolist()

        if self.pad_state_dim is not None and self.normalize_state:
            self.state_min = pad_to_len(np.array(self.state_min, dtype=np.float32), self.pad_state_dim, dim=0)[0].tolist()
            self.state_max = pad_to_len(np.array(self.state_max, dtype=np.float32), self.pad_state_dim, dim=0)[0].tolist()

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        assert self.action_min is not None and self.action_max is not None, \
            f"{self.stat_path} is not loaded properly. Probably {resolve_path(self.stat_path)} does not exist."
        action_min = np.array(self.action_min, dtype=np.float32)
        action_max = np.array(self.action_max, dtype=np.float32)
        if self.normalize_state:
            data["states"] = self.normalize_state_func(data["states"])

        # tolerate near-zero, not just exact zero
        ill_mask = np.abs(action_max - action_min) < 1e-4 * (np.abs(action_max) + np.abs(action_min) + 1e-8) 
        action_max[ill_mask] = 1.0  # prevent division by zero
        actions_normalized = np.where(
            ill_mask, data["actions"], (data["actions"] - action_min) / (action_max - action_min) * 2 - 1
        )

        if self.use_norm_mask:
            actions = np.where(self.action_norm_masks, actions_normalized, data["actions"])
        else: 
            actions = actions_normalized

        # if (np.abs(actions) > 10.0).any():
        #     print("Error: values outside quantile range during normalization.")

        actions = np.clip(actions, -1, 1).astype(np.float32)
        # print(data["dataset"], actions.max(), actions.min(), actions.mean(), actions.std())
        data["raw_actions"] = data["actions"]
        data["actions"] = actions

        return data

    def normalize(self, action, **kwargs):
        data = {"actions": action}
        return self.__call__(data)["actions"]

    def normalize_state_func(self, states, **kwargs):
        state_min = np.array(self.state_min, dtype=np.float32)
        state_max = np.array(self.state_max, dtype=np.float32)
        # Normalize states, tolerate near-zero, not just exact zero
        ill_mask = np.abs(state_max - state_min) < 1e-4 * (np.abs(state_max) + np.abs(state_min) + 1e-8)
        state_max[ill_mask] = 1.0  # prevent division by zero
        current_state = np.where(
            ill_mask, 0, 
            (states - state_min) / (state_max - state_min) * 2 - 1
        )
        current_state = np.clip(current_state, -1, 1).astype(np.float32)
        if np.isnan(current_state).any():
            current_state = np.nan_to_num(current_state, nan=0.0)
        return current_state

    def reverse_call(self, array: Any, **kwargs) -> Any:
        assert self.action_min is not None and self.action_max is not None
        action_min = np.array(self.action_min)
        action_max = np.array(self.action_max)
        reversed_array = (array + 1) / 2 * (action_max - action_min) + action_min
        return reversed_array

    def denormalize_L1_action_err(self, L1_err, dataset_name: list[str] | None = None):
        """return denormalized L1 err loss"""
        
        array_class = torch.tensor if torch.is_tensor(L1_err) else np.array
        where = torch.where if torch.is_tensor(L1_err) else np.where
        data_type = L1_err.dtype

        if self.action_norm_type == "bounds" or \
            self.action_norm_type == "bounds_q99":
            low = array_class(self.action_min, dtype=data_type)  # type: ignore
            high = array_class(self.action_max, dtype=data_type)  # type: ignore
            
            if self.use_norm_mask:
                if torch.is_tensor(L1_err):
                    low = low.to(L1_err.device)
                    high = high.to(L1_err.device)
                    action_norm_masks = torch.tensor(
                        self.action_norm_masks, device=L1_err.device
                    )
                else:
                    action_norm_masks = self.action_norm_masks
                result = where(action_norm_masks, 0.5 * L1_err * (high - low), L1_err) # type:ignore
            else:
                result = 0.5 * L1_err * (high - low)
            
            if dataset_name is None:
                return result # backward compatibility
            
            # action L1 errors
            avg_action_errors_denormed = result.mean(0)  # (Da,) NOTE only if the error is L1 (linear)
            # hand_joints(14) + arm_joints(14) + rpy(3) + height(1) + vx + vy + vyaw + dyaw
            # NOTE: the per-modality split below is HARDCODED to the legacy Psi0+AMO 36-D whole-body action layout.
            assert avg_action_errors_denormed.shape[-1] == 36, (
                "denormalize_L1_action_err per-modality breakdown is hardcoded to the G1 "
                f"36-D whole-body action layout, but got action dim "
                f"{avg_action_errors_denormed.shape[-1]}. Pass dataset_name=None to skip the "
                "breakdown, or override denormalize_L1_action_err for this embodiment."
            )
            hand_joints_end = 14
            arm_joints_end = 28
            rpy_end = 31
            height_end = 32
            torso_vx_end = 33
            torso_vy_end = 34
            torso_vyaw_end = 35
        
            labels_denormed = [
                "err_l1_hand_joints",
                "err_l1_arm_joints",
                "err_l1_torso_rpy",
                "err_l1_height",
                "err_l1_vx",
                "err_l1_vy",
                "err_l1_vyaw",
                "err_l1_target_yaw",
            ]
        
            avg_lr_action_err_denormed = np.split(
                avg_action_errors_denormed, [
                    hand_joints_end, arm_joints_end, rpy_end, height_end, torso_vx_end, torso_vy_end, torso_vyaw_end
                ], axis=-1
            )
            dataset = dataset_name[0] if dataset_name is not None else "unknown"
            return {dataset:dict(zip(labels_denormed, avg_lr_action_err_denormed))}
            
        else:
            raise NotImplementedError
    
    
    def denormalize(self, normalized: np.ndarray|torch.Tensor) -> np.ndarray|torch.Tensor: 
        """ Denormalize the action. """
        array_class = torch.tensor if torch.is_tensor(normalized) else np.array
        where = torch.where if torch.is_tensor(normalized) else np.where
        data_type = normalized.dtype

        if self.action_norm_type == "bounds" or \
                self.action_norm_type == "bounds_q99":
            if self.action_norm_type == "bounds":
                low = array_class(self.action_min, dtype=data_type)  # type: ignore
                high = array_class(self.action_max, dtype=data_type)  # type: ignore
            elif self.action_norm_type == "bounds_q99":
                low = array_class(self.action_min, dtype=data_type)  # type: ignore
                high = array_class(self.action_max, dtype=data_type)  # type: ignore

            if not self.use_norm_mask:
                assert self.action_min is not None
                action_norm_masks = [True] * len(self.action_min)
            else:
                action_norm_masks = self.action_norm_masks

            if torch.is_tensor(normalized):
                low = low.to(normalized.device)  # type: ignore
                high = high.to(normalized.device)  # type: ignore
                action_norm_masks = torch.tensor(action_norm_masks, device=normalized.device)
            else:
                action_norm_masks = np.array(action_norm_masks)

            action = where(action_norm_masks, 0.5 * (normalized + 1) * (high - low) + low, normalized)  # type: ignore
        return action # type: ignore


class Qwen3vlModelTransform(ModelTransform):
    resize: ResizeImage = Field(default_factory=lambda: ResizeImage(size=(270, 480)))
    color_jitter: ColorJitter
    img_aug: bool = False
    # anchor_size: tuple[int, int] = (256, 256) # H, W

    # for mixed dataset with different image sizes
    adaptive_resize: bool = False
    img_sizes: dict[str, Any] = Field(default_factory=lambda: {
        "egodex": [270, 480],
        "he"    : [240, 320],
    })

    def __call__(
        self, 
        data: dict[str, Any], 
        no_aug: bool = False,
        vlm_processor = None, 
        action_tokenizer = None, 
        is_eval: bool = False,
        **kwargs
    ) -> dict[str, Any]:
        if vlm_processor is None or action_tokenizer is None:
            return {"raw_actions": data["actions"], **data}
        """
        data: dict with keys:
            - instruction: str
            - observations: List[Image] (To x PIL)
            - states: (To, Da) torch.Tensor
            - action: (Tp, Da) torch.Tensor
        """
        do_img_aug = False if no_aug else self.img_aug
        if self.adaptive_resize:
            assert data["dataset"] is not None
            match data["dataset"]:
                case "egodex":
                    target_size = self.img_sizes["egodex"]
                case "humanoid-everyday":
                    target_size = self.img_sizes["he"]
                case _:
                    target_size = (256, 256)
            resizer = ResizeImage(size=tuple(target_size))() # type: ignore
        else:
            resizer = self.resize()
        t1 = v2.Compose([
            resizer,
            self.color_jitter() if do_img_aug else v2.Identity(),
        ])
        
        images = [t1(img) for img in data["observations"]]
        instruction = data["instruction"]
        state = data["states"]
        action = data["actions"]
        inputs, num_answer_tokens_list = self.build_qwenvl_inputs(
            vlm_processor, action_tokenizer, [images], [instruction], [state], [action], is_eval=is_eval
        )
        labels = copy.deepcopy(inputs["input_ids"])
        # keep loss on the answer + EOS + formatting tokens
        labels[:, : -(num_answer_tokens_list[0] + 2)] = IGNORE_INDEX 
        inputs["labels"] = labels
        inputs["dataset_name"] = data.get("dataset", "unknown")
        inputs["raw_actions"] = action
        # inputs["raw_instruction"] = instruction
        inputs["observations"] = np.stack([np.array(img) for img in images])
        return inputs

    def build_qwenvl_inputs(
        self,
        vlm_processor,
        action_tokenizer,
        images,
        instructions,
        states,
        actions,
        is_eval: bool = False,
        **kwargs,
    ):
        """adapted from Qwen_VL_Interface.build_qwenvl_inputs"""
        # Create messages: one message per sample
        messages = []
        num_answer_tokens_list = []
        assert len(images) == len(
            instructions
        ), "Images and instructions must have the same length"

        for imgs, instruction, action in zip(images, instructions, actions):
            tokenized_action = action_tokenizer(action)
            raw_action_tokens = vlm_processor.tokenizer(tokenized_action)["input_ids"]
            num_answer_tokens = len(raw_action_tokens)
            num_answer_tokens_list.append(num_answer_tokens)

            message = []
            content = [{"type": "image", "image": img} for img in imgs]
            content.append({"type": "text", "text": instruction})
            user_msg = {"role": "user", "content": content}
            message.append(user_msg)
            if not is_eval:
                assistant_msg = {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": tokenized_action}
                    ],  # squeeze batch dim
                }
                message.append(assistant_msg)
            messages.append(message)

        # Prepare text prompts using processor
        # default process is json --> message --> texts --> input_ids
        texts = [
            vlm_processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=is_eval
            )
            for m in messages
        ]

        try:
            from qwen_vl_utils import process_vision_info
        except:
            raise ImportError("qwen_vl_utils not found, make sure to install it if using Qwen-VL model!")
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)  # type: ignore
        inputs = vlm_processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs, num_answer_tokens_list

class HEPosttrainRepackTransform(RepackTransform):
    dataset_name: str = "humanoid-everyday"
    num_past_frames: int = 0
    action_chunk_size: int = 16
    use_delta_actions: bool = True
    pad_action_dim: int | None = None
    pad_state_dim: int | None = None

    def __call__(
        self,
        data: dict[str, Any],
        **kwargs,
    ) -> dict[str, Any]:
        # HACK: different action format determines robot type
        if data["action.joint_angles"].shape[1] == 26:
            # H1 robot
            raw = data["action.joint_angles"]
            actions = np.concatenate([
                raw[:, 6:12][::-1], # from left thumb to little
                np.zeros((raw.shape[0], 1), dtype=np.float32), # pad finger to 7 dof
                raw[:, 0:6][::-1], # from right thumb to little
                np.zeros((raw.shape[0], 1), dtype=np.float32), # pad finger to 7 dof
                raw[:, 12:], # left arm + right arm (each from should to wrist)
            ], axis=1).astype(np.float32)
            hand = data["observation.hand_joints"]
            states = np.concatenate([
                hand[:, 6:12],
                np.zeros((hand.shape[0], 1), dtype=np.float32), # pad finger to 7 dof
                hand[:, 0:6],
                np.zeros((hand.shape[0], 1), dtype=np.float32), # pad finger to 7 dof 
                data["observation.arm_joints"]
            ], axis=1).astype(np.float32)
        else:
           actions = data["action.joint_angles"].astype(np.float32)
           states = np.concatenate((data["observation.hand_joints"], data["observation.arm_joints"]), axis=1).astype(np.float32)


        if self.use_delta_actions:
            actions = actions[1:] - actions[:-1]
            action_mask = data["action_mask"][1:]
        else:
            action_mask = data["action_mask"]

        # Expand action_mask to (T, Da) by repeating along last dimension
        if action_mask.ndim == 1:
            action_mask = np.repeat(action_mask[:, None], actions.shape[1], axis=1)

        states, _ = pad_to_len(states, self.pad_state_dim) if self.pad_state_dim is not None else (states, None)
        if self.pad_action_dim is not None:
            actions, _ = pad_to_len(actions, self.pad_action_dim)
            mask, _  = pad_to_len(action_mask, self.pad_action_dim)
        else:
            mask = np.ones_like(actions, dtype=bool)
            # mask = action_mask
        return {
            "observations": [pt_to_pil(img, normalized=False) for img in data["observation.images.egocentric"]],
            "states": states,
            "actions": actions,
            "instruction": str(data["task"]).lower(),
            "dataset": data.get("dataset_name", self.dataset_name),
            "actions_mask": mask,# torch.from_numpy(mask),
            "obs_mask": data["obs_mask"], #torch.from_numpy(data["obs_mask"]),
        }

class MixedRepackTransform(RepackTransform):
    dataset_name: str = "mixed"

    num_past_frames: int = 0
    action_chunk_size: int = 1
    use_delta_actions: bool = True
    robot_type: str = "mixed"
    stage: str = "pretrain"  # "pretrain" or "postpre"
    pad_action_dim: int | None = None
    pad_state_dim: int | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.stage == "pretrain":
            self._he_repacker = HEPretrainRepackTransform.model_validate(self.model_dump())
        else:
            self._he_repacker = HEPosttrainRepackTransform.model_validate(self.model_dump())
        self._egodex_repacker = EgodexRepackTransform.model_validate(self.model_dump())

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        dataset_name = data.get("dataset_name", "default")
        match(dataset_name):
            case "egodex":
                return self._egodex_repacker(data, **kwargs)
            case "humanoid-everyday":
                return self._he_repacker(data, **kwargs)
        return data
    

class Psi0ModelTransform(ModelTransform):
    resize: ResizeImage = Field(default_factory=lambda: ResizeImage(size=(224)))
    center_crop: CenterCrop = Field(default_factory=lambda: CenterCrop(size=224))
    color_jitter: ColorJitter = Field(default_factory=lambda: ColorJitter())
    gaussian_noise: GaussianNoise = Field(default_factory=lambda: GaussianNoise(mean=0, std=3, prob_skip=0.1))
    img_aug: bool = False
    
    # for mixed dataset with different image sizes
    adaptive_resize: bool = False
    img_sizes: dict[str, Any] = Field(default_factory=lambda: {
        "egodex": [270, 480],
        "he"    : [240, 320],
    })

    def __call__(
        self, data: dict[str, Any], vlm_processor=None, no_aug=False, **kwargs
    ) -> dict[str, Any]:
        """
            ...
            data["observations"]: List of PIL Images
        """
        do_img_aug = False if no_aug else self.img_aug
        if self.adaptive_resize:
            assert data["dataset"] is not None
            match data["dataset"]:
                case "egodex":
                    target_size = self.img_sizes["egodex"]
                case "humanoid-everyday":
                    target_size = self.img_sizes["he"]
                case _:
                    target_size = (256, 256)
            resizer = ResizeImage(size=tuple(target_size))() # type: ignore
            center_crop = CenterCrop(size=tuple(target_size))() # type: ignore
        else:
            resizer = self.resize()
            center_crop = self.center_crop()
            
        t1 = v2.Compose([
            resizer,
            center_crop,
            self.color_jitter() if do_img_aug else v2.Identity(),
        ])
        
        images = [t1(img) for img in data["observations"]]
        instruction = data["instruction"]

        inputs: dict = self.build_qwenvl_inputs(
            vlm_processor, images, instruction,
        ) # type: ignore
        inputs["dataset_name"] = data.get("dataset", "unknown")
        inputs["raw_actions"] = data["raw_actions"]
        if "actions_mask" in data:
            inputs["actions_mask"] = data["actions_mask"]

        inputs["observations"] = images
        inputs['actions'] = data["actions"]
        inputs['states'] = data["states"]
        inputs['instruction'] = data["instruction"]
        return inputs

    def build_qwenvl_inputs(
        self,
        vlm_processor,
        imgs,
        instruction,
        # state,
        # action,
        **kwargs,
    ) -> tuple[dict, int]:
        from qwen_vl_utils import process_vision_info

        """adapted from Qwen_VL_Interface.build_qwenvl_inputs"""
        messages = []

        content = [{"type": "image", "image": img} for img in imgs]
        content.append({"type": "text", "text": instruction})
        user_msg = {"role": "user", "content": content}
        messages.append([user_msg])

        # Prepare text prompts using processor
        # default process is json --> message --> texts --> input_ids
        texts = [
            vlm_processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)  # type: ignore
        inputs = vlm_processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs

class ACTModelTransform(ModelTransform):
    resize: ResizeImage
    center_crop: CenterCrop
    color_jitter: ColorJitter
    normalize: Normalize
    img_aug: bool = False
    
    def __call__(
        self,
        data: dict[str, Any],
        no_aug: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        do_img_aug = False if no_aug else self.img_aug
        
        # Image transform pipeline
        t = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                self.resize(),
                self.center_crop(),
                self.color_jitter() if do_img_aug else v2.Identity(),
                self.normalize(),
            ]
        )
        
        processed_images = torch.stack([t(img) for img in data["observations"]]) # (N*To, C, H, W)
        
        if "action_is_pad" in data:
            # LerobotRepackTransform format: (chunk_size,), True=padded
            action_is_pad = data["action_is_pad"]
        elif "actions_mask" in data:
            # WeGR00TRepackTransform format: (chunk_size, Da), True=original, False=padded
            actions_mask = data["actions_mask"]
            if isinstance(actions_mask, np.ndarray):
                actions_mask = actions_mask.astype(bool)
                # (chunk_size, Da) --> (chunk_size,)
                if actions_mask.ndim == 2:
                    action_is_pad = ~actions_mask[:, 0]  # Invert: True->False (original), False->True (padded)
                else:
                    action_is_pad = ~actions_mask
            else:
                actions_mask = actions_mask.bool() if hasattr(actions_mask, 'bool') else actions_mask
                action_is_pad = ~actions_mask[:, 0] if len(actions_mask.shape) == 2 else ~actions_mask
        else:
            action_is_pad = np.zeros(data["actions"].shape[0], dtype=bool)
        
        return {
            "observation.images": processed_images,  # (N*To, C, H, W)
            "observation.state": data["states"],  # (state_dim,) or (obs_horizon, state_dim)
            "action": data["actions"],  # (chunk_size, action_dim)
            "action_is_pad": action_is_pad,  # (chunk_size,), standardized: True=padded, False=original
        }
    
class DiffusionPolicyModelTransform(ModelTransform):
    resize: ResizeImage
    color_jitter: ColorJitter
    center_crop: CenterCrop
    normalize: Normalize
    img_aug: bool = False
    
    def __call__(
        self,
        data: dict[str, Any],
        no_aug: bool = False,
        **kwargs,
    ) -> dict[str, Any]:

        # import torch
        # try:
        # from torchvision.transforms import v2
        # except:
        #     from torchvision import transforms as v2
        do_img_aug = False if no_aug else self.img_aug
        t = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                self.resize(),
                self.center_crop(),
                self.color_jitter() if do_img_aug else v2.Identity(),
                self.normalize(),
            ]
        )
        processed_images = torch.stack([t(img) for img in data["observations"]]) # (To, 3, H, W)
        
        if "action_is_pad" in data:
            # LerobotRepackTransform format: (chunk_size,), True=padded
            action_is_pad = data["action_is_pad"]
        elif "actions_mask" in data:
            # WeGR00TRepackTransform format: (chunk_size, Da), True=original, False=padded
            actions_mask = data["actions_mask"]
            if isinstance(actions_mask, np.ndarray):
                actions_mask = actions_mask.astype(bool)
                # (chunk_size, Da) --> (chunk_size,)
                if actions_mask.ndim == 2:
                    action_is_pad = ~actions_mask[:, 0]  # Invert: True->False (original), False->True (padded)
                else:
                    action_is_pad = ~actions_mask
            else:
                actions_mask = actions_mask.bool() if hasattr(actions_mask, 'bool') else actions_mask
                action_is_pad = ~actions_mask[:, 0] if len(actions_mask.shape) == 2 else ~actions_mask
        else:
            action_is_pad = np.zeros(data["actions"].shape[0], dtype=bool)
        
        
        return dict(
            image=processed_images, # (To, 3, H, W)
            agent_pos=data["states"], # (To, Da)
            action=data["actions"], #(Tp, Da)
            action_is_pad=action_is_pad,  # (Tp,), standardized: True=padded, False=original
        )
        
        # return {
        #     "observation.images": processed_images,  # (To, C, H, W)
        #     "observation.state": data["states"],  # (To, state_dim)
        #     "action": data["actions"],  # (Tp, Da)
        # }
