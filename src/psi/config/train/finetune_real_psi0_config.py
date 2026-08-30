from typing import Any, Union, Annotated
from pydantic import BaseModel, Field, model_validator

from psi.config.config import LaunchConfig
from psi.config.data_lerobot import LerobotDataConfig
from psi.config.model_psi0 import Psi0ModelConfig
from psi.config.transform import DataTransform
from psi.config import transform as pt

class DynamicDataTransform(DataTransform):
    repack: pt.RealRepackTransform
    field: pt.ActionStateTransform
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
        if self.data.transform.repack.pad_state_dim is not None:
            assert self.model.odim == self.data.transform.repack.pad_state_dim, "inconsitent odim"
        return self
