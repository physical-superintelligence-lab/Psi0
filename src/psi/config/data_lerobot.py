from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
from typing import Any, Optional, Dict, List, TYPE_CHECKING
from psi.config.config import DataConfig
from pathlib import Path
from psi.utils import resolve_data_path
import os
import json
if TYPE_CHECKING:
    from psi.data.dataset import TransformableDataset

from psi.config.transform import ActionStateTransform
class LerobotDataConfig(DataConfig):
    root_dir: str
    train_repo_ids: List[str] = Field(default_factory=list)
    val_repo_ids: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _resolve_psi_home_paths(self) -> "LerobotDataConfig":
        psi_home = os.environ.get("PSI_HOME", "/psi")
        if psi_home is None:
            return self
        def _resolve(p: str | None) -> str | None:
            if p is None or Path(p).is_absolute():
                return p
            candidate = Path(psi_home) / p
            return str(candidate) if candidate.exists() else p
        self.root_dir = _resolve(self.root_dir)
        return self

    @model_validator(mode="after")
    def check_repo_ids(self) -> "LerobotDataConfig":
        if len(self.train_repo_ids) == 0:
            raise ValueError("train_repo_ids must be provided")
        if len(self.val_repo_ids) == 0:
            self.val_repo_ids = [self.train_repo_ids[0]]
        return self
    
    @model_validator(mode="after")
    def load_stats(self) -> "LerobotDataConfig":
        if not isinstance(self.transform.field, ActionStateTransform):
            return self
        if (
            not Path(self.transform.field.stat_path).is_absolute() and 
            self.transform.field.action_max is None
        ):
            fpath = resolve_data_path(
                Path(self.root_dir) / self.train_repo_ids[0] / self.transform.field.stat_path
            )
            if not os.path.exists(fpath):
                return self
            with open(fpath, "r") as f:
                stats = json.load(f)
                self.transform.field.populate_stats(stats)
        return self

    def __call__(self, split: str = "train", transform_kwargs={}, **kwargs) -> TransformableDataset:
        from psi.data.lerobot import LeRobotDatasetWrapper
        from psi.data.dataset import Dataset as MapStyleDataset

        train_dataset = LeRobotDatasetWrapper(self, split=split)
        return MapStyleDataset(self, train_dataset, transform_kwargs=transform_kwargs)

    def mock(self, split: str = "train", transform_kwargs={}, **kwargs) -> Any:
        dataset = self.__call__(split, transform_kwargs=transform_kwargs, **kwargs)
        return dataset[0]
