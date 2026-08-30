import os
import re
import json
import torch
import numpy as np
from pydantic import BaseModel, Field
from typing import List, Any, Optional
from .transform import LerobotRepackTransform, FieldTransform
from psi.utils import get_asset_dir, pt_to_pil, resolve_path, pad_to_len

def parse_modality_key(key: str) -> tuple[str, slice | int | None]:
    """Parse a key like 'action[:14]' into ('action', slice(None, 16)) or 'action.neck' into ('action.neck', None)."""
    m = re.match(r'^(.*?)\[([^\]]*)\]$', key)
    if not m:
        return key, None
    base_key, slice_str = m.group(1), m.group(2)
    parts = slice_str.split(':')
    if len(parts) == 1:
        return base_key, int(parts[0])
    start = int(parts[0]) if parts[0] else None
    stop = int(parts[1]) if len(parts) > 1 and parts[1] else None
    step = int(parts[2]) if len(parts) > 2 and parts[2] else None
    return base_key, slice(start, stop, step)


class SonicRepackTransform(LerobotRepackTransform):
    dataset_name: str = "sonic"

    image_keys: List[str] = Field(default_factory=lambda: ["observation.images.egocentric"])
    state_keys: List[str] = Field(default_factory=lambda: ["observation.state"])
    action_keys: List[str] = Field(default_factory=lambda: [
        "action.body_token", "action[:14]", "action.neck"
    ])

    instruction_key: str = "task"

    num_past_frames: int = 0
    action_chunk_size: int = 30

    pad_action_dim: int | None = None
    pad_state_dim: int | None = None

    # Per-dim validity of the REPACKED action vector, same width and order as
    # `action_keys` concatenated. Merged multi-embodiment datasets use it to mark
    # zero-filled blocks (H1's padded finger slots, the neck block on sources with
    # no neck DoF) so they contribute nothing to the loss -- finetune.py does
    # `loss_action = (loss_action * mask).sum(1)`. None -> legacy behaviour.
    action_mask_key: str | None = None

    def delta_timestamps(self, fps):
        delta = {}
        for image_key in self.image_keys:
            delta[image_key] = [-t/fps for t in range(self.num_past_frames, -1, -1)]

        for a_key in self.action_keys:
            base_key, _ = parse_modality_key(a_key)
            delta[base_key] = [t/fps for t in range(self.action_chunk_size)]
        
        for s_key in self.state_keys:
            delta[s_key] = [-t/fps for t in range(self.num_past_frames, -1, -1)]

        if self.action_mask_key is not None:
            delta[self.action_mask_key] = [t/fps for t in range(self.action_chunk_size)]

        return delta
    
    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        state_parts = []
        for s_key in self.state_keys:
            base_key, idx = parse_modality_key(s_key)
            vals = np.array(data[base_key])
            if idx is not None:
                vals = vals[..., idx]
            state_parts.append(vals)
        states = np.concatenate(state_parts, axis=-1)
        if self.pad_state_dim is not None:
            states, _ = pad_to_len(states, self.pad_state_dim)

        action_parts = []
        for a_key in self.action_keys:
            base_key, idx = parse_modality_key(a_key)
            vals = np.array(data[base_key])
            if idx is not None:
                vals = vals[..., idx]
            action_parts.append(vals)
        actions = np.concatenate(action_parts, axis=-1)
        if self.pad_action_dim is not None:
            actions, mask = pad_to_len(actions, self.pad_action_dim)
        else:
            mask = np.ones_like(actions, dtype=bool)

        if self.action_mask_key is not None and self.action_mask_key in data:
            valid = np.asarray(data[self.action_mask_key], dtype=np.float32)
            mask = mask.astype(np.float32) * valid.reshape(mask.shape)

        result = {
            "observations": [
                pt_to_pil(data[key], normalized=False)
                for key in self.image_keys
            ], # list of PIL Image
            "states": states.astype(np.float32), # (To, Ds)
            "actions": actions.astype(np.float32),  # (Tp, Da)
            "instruction": data[self.instruction_key].lower(),
            "actions_mask": mask, #(Tp, Da)
            "dataset": self.dataset_name
        }
        return result
    
class SonicActionStateTransform(FieldTransform):
    stat_path: str
    action_norm_type: str = "bounds"  # "bounds_q99"
    
    stat_action_keys: list[str] = Field(default_factory=lambda: [
        "action.body_token", "action[:14]", "action.neck"
    ]) 
    stat_state_keys: list[str] = Field(default_factory=lambda: [
        "observation.state"
    ]) 

    pad_action_dim: int | None = None
    pad_state_dim: int | None = None

    normalize_state: bool = False  # whether to normalize states
    use_norm_mask: bool = False  # backward compatibility

    action_min: Optional[List[float]] = None
    action_max: Optional[List[float]] = None
    state_min: Optional[List[float]] = None
    state_max: Optional[List[float]] = None

    def model_post_init(self, __context: Any) -> None:
        if not os.path.exists(resolve_path(self.stat_path)):
            return
        with open(resolve_path(self.stat_path), "r") as f:
            stats = json.load(f)
            self.populate_stats(stats)

    def _resolve_stat_values(self, stats: dict[str, Any], keys: list[str], stat_key: str) -> list[float]:
        parts = []
        for key in keys:
            base_key, idx = parse_modality_key(key)
            values = np.array(stats[base_key][stat_key])
            if idx is not None:
                values = values[idx]
            parts.append(np.atleast_1d(values))
        return np.concatenate(parts).tolist()

    def populate_stats(self, stats: dict[str, Any]):
        if self.action_norm_type == "bounds_q99":
            self.action_min = self._resolve_stat_values(stats, self.stat_action_keys, "q01")
            self.action_max = self._resolve_stat_values(stats, self.stat_action_keys, "q99")
            if self.normalize_state:
                self.state_min = self._resolve_stat_values(stats, self.stat_state_keys, "q01")
                self.state_max = self._resolve_stat_values(stats, self.stat_state_keys, "q99")
        elif self.action_norm_type == "bounds":
            self.action_min = self._resolve_stat_values(stats, self.stat_action_keys, "min")
            self.action_max = self._resolve_stat_values(stats, self.stat_action_keys, "max")
            if self.normalize_state:
                self.state_min = self._resolve_stat_values(stats, self.stat_state_keys, "min")
                self.state_max = self._resolve_stat_values(stats, self.stat_state_keys, "max")
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

        actions = np.clip(actions_normalized, -1, 1).astype(np.float32)
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
        # where = torch.where if torch.is_tensor(L1_err) else np.where
        data_type = L1_err.dtype

        if self.action_norm_type == "bounds" or \
            self.action_norm_type == "bounds_q99":
            low = array_class(self.action_min, dtype=data_type)  # type: ignore
            high = array_class(self.action_max, dtype=data_type)  # type: ignore
            
            
            result = 0.5 * L1_err * (high - low)
            
            if dataset_name is None:
                return result # backward compatibility
            
            # action L1 errors
            avg_action_errors_denormed = result.mean(0)  # (Da,) NOTE only if the error is L1 (linear)
            # body_token(64) + hand_joints(14) + [neck(2)]
        
            labels_denormed = [
                "action.latent_sonic",
                "action.hand_joints",
            ]
            if self.pad_action_dim is not None and self.pad_action_dim > 78:
                labels_denormed.append("action.neck")
        
            avg_lr_action_err_denormed = np.split(
                avg_action_errors_denormed, [
                    64, 78
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
