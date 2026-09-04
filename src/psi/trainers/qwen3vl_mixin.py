from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, cast
import re
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
if TYPE_CHECKING:
    from psi.trainers import Trainer
    from psi.config.model_psi0 import Psi0ModelConfig
    from psi.config.model_qwen3vl import Qwen3VLModelConfig

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, Qwen2TokenizerFast, Qwen3VLProcessor
from accelerate import Accelerator
from torch.nn.utils.rnn import pad_sequence
# from psi.config.tokenizer import BinActionTokenizerConfig
from psi.utils import initialize_overwatch, str_to_tensor #, shorten, seed_everything
overwatch = initialize_overwatch(__name__)


def _pad_string_tensor_batch(strings: list[str], max_length: int = 128) -> torch.Tensor:
    encoded = [str_to_tensor(s)[:max_length] for s in strings]
    padded = pad_sequence(encoded, batch_first=True)
    if padded.shape[1] < max_length:
        padded = torch.nn.functional.pad(padded, (0, max_length - padded.shape[1]), value=0)
    return padded[:, :max_length]

def _stack_images(imgs: list[torch.Tensor], channels_first: bool) -> torch.Tensor:
    """Stack per-sample image tensors; if H/W differ, resize all to the first sample's H/W.
    channels_first: (..., C, H, W) else (..., H, W, C)."""
    ref = imgs[0].shape
    if all(im.shape == ref for im in imgs):
        return torch.stack(imgs)
    hw = ref[-2:] if channels_first else ref[-3:-1]
    out = []
    for im in imgs:
        x = im if channels_first else im.movedim(-1, -3)
        lead = x.shape[:-3]
        x = F.interpolate(x.reshape(-1, *x.shape[-3:]).float(), size=hw, mode="nearest").to(im.dtype)
        x = x.reshape(*lead, *x.shape[-3:])
        out.append(x if channels_first else x.movedim(-3, -1))
    return torch.stack(out)


class PaddedCollatorForActionPrediction:
    def __init__(self, model_max_length, pad_token_id, padding_side="right", pixel_values_dtype=torch.float32):
        self.model_max_length = model_max_length
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side
        self.pixel_values_dtype = pixel_values_dtype

    def __call__(self, instances):
        
        # Extract sequences
        input_ids = [instance["input_ids"].squeeze(0) if instance["input_ids"].dim() == 2 else instance["input_ids"] 
                    for instance in instances]
        labels = []
        for ids, instance in zip(input_ids, instances):
            try:
                lab = instance["labels"]
                labels.append(lab.squeeze(0) if lab.dim() == 2 else lab)
            except KeyError:  # eval/generation samples carry no labels
                labels.append(torch.full_like(ids, -100))
        pixel_values = [instance["pixel_values"] for instance in instances]
        dataset_names = [instance["dataset_name"] for instance in instances] if "dataset_name" in instances[0] else None
        # which goal source produced each sample (subgoal_jpg / real_future / wm_future,
        # or "none" when the goal image was dropped) -- used only to break the val
        # metric down by mode; training never reads it
        # gate on ANY instance: the transform omits the key entirely when the goal
        # image was dropped, so instances[0] alone is not a reliable probe
        goal_sources = ([str(instance.get("goal_source") or "none") for instance in instances]
                        if any("goal_source" in inst for inst in instances) else None)
        instructions = [instance["instruction"] for instance in instances] if "instruction" in instances[0] else None

        # Pad sequences
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id, padding_side=self.padding_side)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100, padding_side=self.padding_side)

        # Truncate if longer than model_max_length
        input_ids = input_ids[:, :self.model_max_length]
        labels = labels[:, :self.model_max_length]

        # Attention mask based on pad token
        attention_mask = input_ids.ne(self.pad_token_id)

        # Cat pixel_values to flat (total_patches, dim): variable #images/sample
        # (subgoal dropout) -> cat, not stack; matches PaddedCollatorForTogether.
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.cat(pixel_values, dim=0).to(dtype=self.pixel_values_dtype)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {k: torch.cat([pv[k] for pv in pixel_values], dim=0) for k in pixel_values[0]}
        else:
            raise ValueError(f"Unsupported pixel_values type: {type(pixel_values[0])}")

        # Cat image_grid_thw to flat (total_images, 3): matches the cat'd pixel_values.
        image_grid_thw = torch.cat([instance["image_grid_thw"] for instance in instances], dim=0)

        # Build output
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values, 
            "image_grid_thw": image_grid_thw,
        }
        # see also:  batch_str_to_tensor in src/psi/data/gear/transform/base.py for similar encoding of string metadata
        if dataset_names is not None:
            # encode as padded uint8 tensor so accelerate can concatenate across processes
            output["dataset_name_str"] = pad_sequence(
                [str_to_tensor(s) for s in dataset_names], batch_first=True
            )
        if goal_sources is not None:
            output["goal_source_str"] = pad_sequence(
                [str_to_tensor(s) for s in goal_sources], batch_first=True
            )
        if instructions is not None:
            output["instruction_str"] = _pad_string_tensor_batch(instructions, max_length=128)

        raw_actions = [instance["raw_actions"] for instance in instances] if "raw_actions" in instances[0] else None
        if raw_actions is not None:
            output["raw_actions"] = raw_actions

        actions_mask = torch.stack([instance["actions_mask"] for instance in instances]) if "actions_mask" in instances[0] else None
        if actions_mask is not None:
            output["actions_mask"] = actions_mask

        raw_images = torch.stack([torch.from_numpy(np.array(instance["observations"])) for instance in instances]) if "observations" in instances[0] else None
        if raw_images is not None:
            output["observations"] = raw_images

        for key in ("episode_index", "frame_index"):
            if key in instances[0]:
                output[key] = torch.tensor([instance[key] for instance in instances])

        return output

class PaddedCollatorForTogether:
    def __init__(self, model_max_length, pad_token_id, padding_side="right", pixel_values_dtype=torch.float32):
        self.model_max_length = model_max_length
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side
        self.pixel_values_dtype = pixel_values_dtype

    def __call__(self, instances):
        # Extract sequences
        input_ids = [instance["input_ids"].squeeze(0) if instance["input_ids"].dim() == 2 else instance["input_ids"]
                    for instance in instances]

        pixel_values = [instance["pixel_values"] for instance in instances]
        pixel_values_videos = [instance["pixel_values_videos"] for instance in instances] if "pixel_values_videos" in instances[0] else None
        dataset_names = [instance["dataset_name"] for instance in instances] if "dataset_name" in instances[0] else None
        # which goal source produced each sample (subgoal_jpg / real_future / wm_future,
        # or "none" when the goal image was dropped) -- used only to break the val
        # metric down by mode; training never reads it
        # gate on ANY instance: the transform omits the key entirely when the goal
        # image was dropped, so instances[0] alone is not a reliable probe
        goal_sources = ([str(instance.get("goal_source") or "none") for instance in instances]
                        if any("goal_source" in inst for inst in instances) else None)
        instructions = [instance["instruction"] for instance in instances] if "instruction" in instances[0] else None

        # Only support right padding for now
        assert self.padding_side == "right", f"Invalid Tokenizer padding_side={self.padding_side}"

        # Pad sequences
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)

        # Truncate if longer than model_max_length
        input_ids = input_ids[:, :self.model_max_length]

        # Attention mask based on pad token
        attention_mask = input_ids.ne(self.pad_token_id)

        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.cat(pixel_values, dim=0).to(dtype=self.pixel_values_dtype)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {k: torch.cat([pv[k] for pv in pixel_values], dim=0) for k in pixel_values[0]}
        else:
            raise ValueError(f"Unsupported pixel_values type: {type(pixel_values[0])}")

        B = len(instances)
        # cat -> (total_images, 3): flat form Qwen expects; matches cat'd pixel_values above.
        image_grid_thw = torch.cat([instance["image_grid_thw"] for instance in instances], dim=0)

        # Build output
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        # see also:  batch_str_to_tensor in src/psi/data/gear/transform/base.py for similar encoding of string metadata
        if dataset_names is not None:
            # encode as padded uint8 tensor so accelerate can concatenate across processes
            output["dataset_name_str"] = pad_sequence(
                [str_to_tensor(s) for s in dataset_names], batch_first=True
            )
        if goal_sources is not None:
            output["goal_source_str"] = pad_sequence(
                [str_to_tensor(s) for s in goal_sources], batch_first=True
            )

        if instructions is not None:
            output["instruction_str"] = _pad_string_tensor_batch(instructions, max_length=128)

        raw_actions = [instance["raw_actions"] for instance in instances] if "raw_actions" in instances[0] else None
        if raw_actions is not None:
            output["raw_actions"] = raw_actions

        # print(type(instances[0]["actions_mask"]), dataset_names)
        actions_mask = torch.stack([torch.from_numpy(instance["actions_mask"]) for instance in instances]) if "actions_mask" in instances[0] else None
        if actions_mask is not None:
            output["actions_mask"] = actions_mask

        actions = torch.stack([torch.from_numpy(np.array(instance["actions"])) for instance in instances]) if "actions" in instances[0] else None
        if actions is not None:
            output["actions"] = actions

        states = torch.stack([torch.from_numpy(np.array(instance["states"])) for instance in instances]) if "states" in instances[0] else None
        if states is not None:
            output["states"] = states

        # raw uint8 images (logging only): with mixed resolutions, resize to the first sample's size
        if "observations" in instances[0]:
            output["observations"] = _stack_images(
                [torch.as_tensor(np.array(instance["observations"])) for instance in instances], channels_first=True)

        # handle subgoal images if present, watch out for possible dropouts
        subgoal_image_shapes = [instance["subgoal"].shape for instance in instances if  "subgoal" in instance]
        if len(subgoal_image_shapes) > 0:
            zero_subgoal = torch.zeros(subgoal_image_shapes[0], dtype=torch.uint8)
            output["subgoal"] = _stack_images([
                torch.as_tensor(instance["subgoal"]) if "subgoal" in instance else zero_subgoal
                for instance in instances
            ], channels_first=False)

        for key in ("episode_index", "frame_index"):
            if key in instances[0]:
                output[key] = torch.tensor([instance[key] for instance in instances])

        if pixel_values_videos:
            output["pixel_values_videos"] = torch.cat(pixel_values_videos, dim=0).to(dtype=self.pixel_values_dtype)

        # Deferred video path: raw frames (B, T, C, H, W) shipped to GPU, where
        # pixel_values_videos is computed batched. Keep native dtype (uint8) to
        # minimize host->device transfer; the GPU kernel rescales/normalizes.
        if "raw_video_frames" in instances[0]:
            output["raw_video_frames"] = torch.stack(
                [torch.as_tensor(instance["raw_video_frames"]) for instance in instances]
            )

        if "video_grid_thw" in instances[0]:
            output["video_grid_thw"] = torch.cat([instance["video_grid_thw"] for instance in instances], dim=0)

        return output

class QwenBinActionTokenizer:

    def __init__(
        self, tokenizer: Qwen2TokenizerFast, bins: int = 256, min_action: int = -1, max_action: int = 1
    ) -> None:
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.action_token_begin_idx: int = len(tokenizer) # avoid conflict with existing tokens

    def __call__(self, action: np.ndarray) -> tuple[List[int], int]:
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)
        discretized_action = discretized_action.flatten()
        action_token_ids = (self.action_token_begin_idx + discretized_action).tolist()
        return action_token_ids
    
    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Returns continuous actions for discrete action token IDs.
        """
        discretized_actions = action_token_ids - self.action_token_begin_idx   #self.tokenizer.vocab_size - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.n_bins

class Qwen3vlMixin:

    def init_qwen3vl_models(self):
        trainer = cast("Trainer", self)

        model_cfg: Qwen3VLModelConfig = trainer.cfg.model  # type: ignore
        overwatch.info(f"Loading pretrained from {model_cfg.model_name_or_path}")

        try:
            # default: Load the model on the available device(s)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_cfg.model_name_or_path,
                attn_implementation="flash_attention_2",
                dtype=torch.bfloat16,
                local_files_only=True 
            )
            overwatch.info(f"Load VLM from {model_cfg.model_name_or_path} successfully.")
        except Exception as e:
            # safe for multi-node training, please download weights in advance
            overwatch.critical(f"Failed to load VLM from {model_cfg.model_name_or_path}. \n"
                               f"Please download weights in advance using: 'python scripts/predownload_qwen3vl.py'\n"
                               f"Auto downloading is disabled in multi-node training to prevent conflicts. \n")
            raise e

        if not hasattr(self.model.config, "hidden_size"):
            self.model.config.hidden_size = self.model.config.text_config.hidden_size

        self.vlm_processor = AutoProcessor.from_pretrained(model_cfg.model_name_or_path)
        self.tokenizer = self.vlm_processor.tokenizer

        if trainer.cfg.train.lora:
            from peft import LoraConfig, get_peft_model, TaskType
            print("LoRA enabled")

            for p in self.model.parameters():
                p.requires_grad = False

            lora_config = LoraConfig(
                r=model_cfg.lora_r or 64,
                lora_alpha=model_cfg.lora_alpha or 128,
                lora_dropout=model_cfg.lora_dropout or 0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lora_config)
        else:
            if model_cfg.tune_mm_vision:
                for n, p in self.model.visual.named_parameters():
                    p.requires_grad = True
            else:
                for n, p in self.model.visual.named_parameters():
                    p.requires_grad = False

            if model_cfg.tune_mm_mlp:
                for n, p in self.model.visual.merger.named_parameters():
                    p.requires_grad = True
            else:
                for n, p in self.model.visual.merger.named_parameters():
                    p.requires_grad = False

            if model_cfg.tune_mm_llm:
                for n, p in self.model.language_model.named_parameters():
                    p.requires_grad = True
                self.model.lm_head.requires_grad = True # type:ignore
            else:
                for n, p in self.model.language_model.named_parameters():
                    p.requires_grad = False
                self.model.lm_head.requires_grad = False # type:ignore
    
    def prepare_qwen3vl(self, accelerator: Accelerator):
        trainer = cast("Trainer", self)
        self.model, self.optimizer, self.lr_scheduler = accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler
        )

        if trainer.train_cfg.data_parallel == "deepspeed":
            # Gradient Checkpoint Setup
            if trainer.model_cfg.gradient_checkpointing: # type: ignore
                assert "DeepSpeedEngine" in trainer.model.__class__.__name__, "deepspeed is not properly initialized!"
                if hasattr(trainer.model, "enable_input_require_grads"): 
                    trainer.model.enable_input_require_grads()
                else:
                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)
                    trainer.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
        if trainer.cfg.train.overfit_single_batch:
            overwatch.warning("Overfitting a single batch: reusing first batch every step. set cfg.data.image_aug = False for true memorization.")
            first_batch = next(iter(self.train_dataloader))
            class SingleBatchLoader:
                def __iter__(self): 
                    while True:
                        yield first_batch
                def __len__(self):
                    return 1
            self.train_dataloader = SingleBatchLoader()
            
        self.accelerator = accelerator
        return self.train_dataloader


    def create_qwen3vl_optimizer(self):
        trainer = cast("Trainer", self)
        opt_model = self.model
        decay_parameters = self.get_decay_parameter_names(opt_model)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        model_cfg: Qwen3VLModelConfig | Psi0ModelConfig = trainer.cfg.model  # type: ignore
        if model_cfg.mm_projector_lr is not None and model_cfg.mm_projector_lr != 0:
            projector_parameters = [
                name for name, _ in opt_model.named_parameters() if "merger" in name
            ]
            if model_cfg.vision_tower_lr is not None and model_cfg.vision_tower_lr != 0:
                vision_tower_parameters = [
                    name for name, _ in opt_model.named_parameters() if "visual" in name
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": model_cfg.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": model_cfg.weight_decay,
                        "lr": model_cfg.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": model_cfg.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": model_cfg.weight_decay,
                        "lr": model_cfg.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": model_cfg.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": model_cfg.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": model_cfg.weight_decay,
                        "lr": model_cfg.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": model_cfg.mm_projector_lr,
                    },
                ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": model_cfg.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        return optimizer_grouped_parameters

    ####  adatped from transformers.Trainer  ####
    def get_decay_parameter_names(self, model) -> list[str]:
        """
        Get all parameter names that weight decay will be applied to.

        This function filters out parameters in two ways:
        1. By layer type (instances of layers specified in ALL_LAYERNORM_LAYERS)
        2. By parameter name patterns (containing 'bias', or variation of 'norm')
        """
        forbidden_name_patterns = [r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
        decay_parameters = self.get_parameter_names(model, [nn.LayerNorm], forbidden_name_patterns)
        return decay_parameters

    def get_parameter_names(self, model, forbidden_layer_types, forbidden_layer_names=None):
        """
        Returns the names of the model parameters that are not inside a forbidden layer.
        """
        forbidden_layer_patterns = (
            [re.compile(pattern) for pattern in forbidden_layer_names] if forbidden_layer_names is not None else []
        )
        result = []
        for name, child in model.named_children():
            child_params = self.get_parameter_names(child, forbidden_layer_types, forbidden_layer_names)
            result += [
                f"{name}.{n}"
                for n in child_params
                if not isinstance(child, tuple(forbidden_layer_types))
                and not any(pattern.search(f"{name}.{n}".lower()) for pattern in forbidden_layer_patterns)
            ]
        # Add model specific parameters that are not in any child
        result += [
            k for k in model._parameters if not any(pattern.search(k.lower()) for pattern in forbidden_layer_patterns)
        ]

        return result