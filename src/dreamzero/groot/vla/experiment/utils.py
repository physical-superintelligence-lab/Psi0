"""
Originally trinity.train.utils
"""

from dataclasses import dataclass
import os
import pathlib
from pathlib import Path
import re
import shutil

import torch
import torch.nn as nn
from transformers import PretrainedConfig, Trainer


def dtype_from_string(dtype_str):
    if dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "float32":
        return torch.float32
    else:
        raise ValueError(f"Unsupported dtype_str {dtype_str}")


def rprint(*args, **kwargs):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        return print(f"[dist-{rank}-of-{world_size}]", *args, **kwargs)
    else:
        return print(*args, **kwargs)


def mprint(*args, **kwargs):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        if rank == 0:
            return print(f"[dist-{rank}-of-{world_size}]", *args, **kwargs)
        else:
            return
    else:
        return print(*args, **kwargs)


def is_local(model_name_or_path: str) -> bool:
    return os.path.isdir(model_name_or_path)


def get_checkpoint_path(output_dir: str, checkpoint_prefix: str = "checkpoint") -> str | None:
    output_dir = os.path.abspath(output_dir)
    pathlib_dir = pathlib.Path(output_dir)

    if list(pathlib_dir.glob("config.json")):
        # training has been finished
        return output_dir, False
    else:
        try:
            ordering_and_checkpoint_path = []
            glob_checkpoints = [
                str(x)
                for x in pathlib.Path(output_dir).glob(f"{checkpoint_prefix}-*")
                if os.path.isdir(x) and (x / "trainer_state.json").is_file()
            ]
            for path in glob_checkpoints:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))
            checkpoints_sorted = sorted(ordering_and_checkpoint_path)
            return checkpoints_sorted[-1][1], True
        except IndexError:
            return None, True


def prepare_config_for_training(
    config: PretrainedConfig, model_args: dataclass, training_args: dataclass, data_args: dataclass
) -> None:
    ## set default dtype
    # config.model_dtype = "bfloat16" if training_args.bf16 else "float16"

    ## set tuning modules
    config.tune_language_model = training_args.tune_language_model
    config.tune_vision_tower = training_args.tune_vision_tower
    config.tune_mm_projector = training_args.tune_mm_projector


def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir, _internal_call=True)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def compute_grad_accum_to_match_global_bs(global_bs: int, bs: int):
    num_devices = torch.distributed.get_world_size()
    per_step_bs = bs * num_devices
    assert global_bs % per_step_bs == 0, f"{global_bs=}, {per_step_bs=}"
    num_grad_accum = global_bs // per_step_bs
    return num_grad_accum


def get_training_param_info(model):
    module_states = dict()
    for module_name, module in model.named_children():
        key = f"{module_name}({module.__class__.__name__})"
        if all([p.requires_grad for p in module.parameters()]):
            module_states[key] = "true"
        elif all([not p.requires_grad for p in module.parameters()]):
            module_states[key] = "false"
        else:
            module_states[key] = get_training_param_info(module)

    return module_states


def get_param_count_tree(model: nn.Module):
    """
    Calculate parameters for the model, structure them as a nested dictionary,
    and save the result as a formatted JSON file.
    """

    def format_param_count(count: int) -> str:
        """Format the count as a string in millions, e.g. 11M or 5.5M."""
        count_in_millions = count / 1e6
        # If the value is an integer, display without decimal places
        if count_in_millions.is_integer():
            return f"{int(count_in_millions)}M"
        else:
            return f"{count_in_millions:.2f}M"

    def module_to_dict(module: nn.Module, module_name: str) -> dict:
        """
        Recursively convert a module and its children into a nested dictionary.
        The key is formatted as "module_name (ClassName, param_count)".
        """

        total_count = sum(p.numel() for p in module.parameters())
        formatted_total = format_param_count(total_count)
        key = f"{module_name} ({module.__class__.__name__}, {formatted_total})"

        # Get immediate children modules
        children = list(module.named_children())
        if children:
            nested = {}
            for child_name, child_module in children:
                # Recursively convert child modules
                nested.update(module_to_dict(child_module, child_name))
            return {key: nested}
        else:
            # Leaf module: return empty dict as value
            return {key: {}}

    # Start from the top-level module (you can change the name "model" as needed)
    nested_dict = module_to_dict(model, "model")
    return nested_dict
