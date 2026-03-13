from pydantic import BaseModel, Field, model_validator
from typing import Any, Dict, List, TYPE_CHECKING
from psi.config.config import DataConfig

class EgoDexDataConfig(DataConfig):
    root_dir: str
    upsample_rate: int = 3
    chunk_size: int = 16
    train_repo_ids: List[str] = []
    val_repo_ids: List[str] = []
    # val_chunk_size: int = 16 # original window size
    use_delta_actions: bool = True
    load_retarget: bool = False

    def __call__(self, split: str = "train", transform_kwargs=None, **kwargs) -> Any:
        if transform_kwargs is None:
            transform_kwargs = {}
        from psi.data.egodex.egodex_dataset import EgoDexDataset
        from psi.data.dataset import Dataset as MapStyleDataset
        # from psi.data.dataset import IterableDataset

        dataset = EgoDexDataset(
            data_root=self.root_dir,
            upsample_rate=self.upsample_rate,
            val=(split != "train"),
            chunk_size=self.chunk_size,
            img_history_size=1,
            use_delta_actions=self.use_delta_actions,
            load_retarget=self.load_retarget,
        )
        return MapStyleDataset(
            self, dataset, transform_kwargs=transform_kwargs, **kwargs
        )

    def mock(self, split: str = "train", transform_kwargs=None, **kwargs) -> Any:
        if transform_kwargs is None:
            transform_kwargs = {}
        dataset = self.__call__(split, transform_kwargs=transform_kwargs, **kwargs)
        # return dataset[0]
        n_samples = min(10, len(dataset))
        import random

        indices = random.sample(range(len(dataset)), n_samples)
        return [dataset[i] for i in indices]