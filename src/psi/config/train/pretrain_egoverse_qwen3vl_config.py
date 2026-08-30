from typing import Union, Annotated
from pydantic import BaseModel, Field, model_validator
# from tyro.conf import subcommand as cmd
from psi.config.config import LaunchConfig
# from psi.config import model as pm
from psi.config.model_qwen3vl import Qwen3VLModelConfig
# from psi.config.data_egodex import EgoDexDataConfig
from psi.config.data_egoverse import EgoVerseDataConfig
from psi.config import transform as pt
from psi.config.transform import DataTransform

class DynamicDataTransform(DataTransform):
    repack: pt.EgoverseRepackTransform
    field: pt.ActionStateTransform
    model: pt.Qwen3vlModelTransform
    
class DynamicDataConfig(EgoVerseDataConfig):
    transform: DynamicDataTransform

class DynamicLaunchConfig(LaunchConfig):
    data: DynamicDataConfig
    model: Qwen3VLModelConfig
