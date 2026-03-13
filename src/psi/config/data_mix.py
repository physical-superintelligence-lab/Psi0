from pydantic import BaseModel, Field, model_validator
from typing import Any, Optional, Dict, List, TYPE_CHECKING
from psi.config.config import DataConfig

class MixedDataConfig(DataConfig):
    use_delta_actions: bool = True
    chunk_size: int = 1
    upsample_rate: int = 3
    data_downsample: Optional[int] = 1 # use every N files of the original data, mainly for ablation

    egodex: dict[str, Any] = Field(default_factory=lambda: {
        "root_dir": "/hfm/data/egodex",
        "ratio": 0.5,
        "load_retarget": False
    })
    he: dict[str, Any] = Field(default_factory=lambda: {
        "root_dir": "/hfm/data/HE_RAW",
        "robot_type": "both", # g1, h1, or both
        "episodes": None,
        "num_past_frames": 0,
        "ratio": 0.5
    })
    sampler: str = "batch_mixture"  # or "token_mixture"
    tokens_per_device: int = 2048 # for token_mixture sampler

    def __call__(self, split: str = "train", transform_kwargs=None, **kwargs) -> Any:
        if transform_kwargs is None:
            transform_kwargs = {}
        from psi.data.egodex.egodex_dataset import EgoDexDataset
        from psi.data.humanoid.he_raw_dataset import HERawDataset
        from psi.data.dataset import MixtureDataset
        from psi.data.dataset import Dataset as MapStyleDataset

        if split == "val":
            he = HERawDataset(
                data_root=self.he["root_dir"],
                num_past_frames=self.he["num_past_frames"],
                action_chunk_size=self.chunk_size,
                upsample_rate=self.upsample_rate,
                use_delta_actions=self.use_delta_actions,
                robot_type=self.he["robot_type"],
                episodes=self.he["episodes"]
            )
            he_dataset = MapStyleDataset(self, he, transform_kwargs=transform_kwargs, **kwargs)
            return he_dataset

        egodex = EgoDexDataset(
            data_root=self.egodex["root_dir"],
            upsample_rate=self.upsample_rate,
            val=(split != "train"),
            chunk_size=self.chunk_size,
            img_history_size=1,
            use_delta_actions=self.use_delta_actions,
            load_retarget=self.egodex.get("load_retarget", False),
            data_downsample=self.data_downsample or 1
        )
        egodex_dataset = MapStyleDataset(
            self, egodex, transform_kwargs=transform_kwargs, **kwargs
        )
        he = HERawDataset(
            data_root=self.he["root_dir"],
            num_past_frames=self.he["num_past_frames"],
            action_chunk_size=self.chunk_size,
            upsample_rate=self.upsample_rate,
            use_delta_actions=self.use_delta_actions,
            robot_type=self.he["robot_type"],
            episodes=self.he["episodes"]
        )
        he_dataset = MapStyleDataset(self, he, transform_kwargs=transform_kwargs, **kwargs)

        mixed_dataset = MixtureDataset({
            egodex_dataset: self.egodex["ratio"],
            he_dataset: self.he["ratio"],
        }, num_samples_per_epoch=2*len(he_dataset))

        if self.sampler == "token_mixture":
            from psi.data.sampler import DatasetSpec
            dataset_specs = [
                DatasetSpec(
                    dataset_length=egodex_dataset.dataset_length,
                    prob=self.egodex["ratio"],
                    image_size=(270, 480), # H,W
                    tokens_per_image=18*30,
                ),
                DatasetSpec(
                    dataset_length=he_dataset.dataset_length,
                    prob=self.he["ratio"],
                    image_size=(240, 320), # H,W
                    tokens_per_image=16*20,
                ),
            ]
            mixed_dataset.specs = dataset_specs

        return mixed_dataset