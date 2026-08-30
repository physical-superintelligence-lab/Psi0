"""Config schema for psi0 post-training on a merged sonic-v1 LeRobot pack.

Same data/model shape as `finetune_sonic_psi0_config` (LeRobot + SonicRepackTransform
+ SonicActionStateTransform); the trainer is chosen separately by `--train.name`,
so pairing this module with `--train.name=posttrain` runs PosttrainTrainer over the
merged `.data/psix_sonic_v1_{train,val}` pack built by scripts/data/merge_sonic_v1.py.

Unlike `posttrain_he_psi0_config` (which is built on HERawDataConfig and ingests raw
HE episodes), this one consumes an already-packed LeRobot dataset whose action /
state columns are in the native sonic layout:

    action_keys = ["action.body_token", "action[:14]", "action.neck"]  -> 64+14+2 = 80
    state_keys  = ["observation.state"]                                -> 45
"""
from typing import Any, Union, Annotated
from pydantic import BaseModel, Field, model_validator

from psi.config.config import LaunchConfig
from psi.config.data_lerobot import LerobotDataConfig
from psi.config.model_psi0 import Psi0ModelConfig
from psi.config.transform import DataTransform
from psi.config import transform as pt
from psi.config import transform_psi0_sonic as ps


class DynamicDataTransform(DataTransform):
    repack: ps.SonicRepackTransform
    field: ps.SonicActionStateTransform
    model: pt.Psi0ModelTransform


class DynamicDataConfig(LerobotDataConfig):
    transform: DynamicDataTransform


class DynamicLaunchConfig(LaunchConfig):
    data: DynamicDataConfig
    model: Psi0ModelConfig

    @model_validator(mode="after")
    def check_observation_dim(self, __context: Any) -> None:
        assert self.data.transform.repack.pad_action_dim == self.data.transform.field.pad_action_dim, "inconsistent action dim"
        assert self.data.transform.repack.pad_state_dim == self.data.transform.field.pad_state_dim, "inconsistent state dim"
        assert self.model.odim == self.data.transform.repack.pad_state_dim, "inconsitent odim"
        assert self.model.action_chunk_size == self.data.transform.repack.action_chunk_size, "inconsistent action chunk size"
        assert self.model.action_dim == self.data.transform.repack.pad_action_dim, "inconsistent model action dim"
        if self.model.vlm_layer_indices is not None:
            assert self.model.num_blocks == len(self.model.vlm_layer_indices), (
                f"inconsistent number of blocks: num_blocks={self.model.num_blocks} but "
                f"{len(self.model.vlm_layer_indices)} vlm_layer_indices (need one VLM layer per block)"
            )
        return self
