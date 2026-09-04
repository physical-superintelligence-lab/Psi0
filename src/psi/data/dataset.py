from __future__ import annotations
from typing import Any, Dict, TYPE_CHECKING
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from psi.config.config import DataConfig
    from psi.data.sampler import DatasetSpec


from collections.abc import Iterator, Sized
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import IterableDataset as TorchIterableDataset


class TransformableDataset(TorchDataset):
    """Marker base class for all PSI datasets.

    All PSI datasets must apply transforms internally so that data is
    fully transformed on access. Downstream APIs should type-hint against
    this class instead of individual dataset types.
    """

    def iter_episode(
        self,
        eps_idx: int | None = None,
        stride: int = 30,
        dataset_index: int | None = None,
    ) -> Iterator[Any]:
        """Yield transformed steps for one episode, advancing ``stride`` frames.

        ``dataset_index`` is required for unambiguous mixture evaluation because
        LeRobot episode IDs are local to each pack and commonly overlap.
        """
        raw_dataset = getattr(self, "raw_dataset", None)
        transform = getattr(self, "transform", None)
        transform_kwargs = getattr(self, "transform_kwargs", {})

        assert raw_dataset is not None, "raw_dataset is not set on this dataset"

        # Mode 2: ShardedLeRobotMixtureDataset
        if hasattr(raw_dataset, "iter_episode"):
            selected_datasets = (
                [raw_dataset.datasets[dataset_index]]
                if dataset_index is not None
                else raw_dataset.datasets
            )
            all_episode_ids: list[int] = [
                tid
                for ds in selected_datasets
                for shard in ds.sharded_trajectories
                for tid in shard
            ]
            total_episodes = len(all_episode_ids)
            if eps_idx is None:
                eps_idx = int(all_episode_ids[np.random.randint(0, total_episodes)])
            print(f"Episode {eps_idx} / {total_episodes} total")
            for i, frame in enumerate(
                raw_dataset.iter_episode(eps_idx, dataset_index=dataset_index)
            ):
                if i % stride == 0: # TODO optimize by skipping frames in the raw dataset instead of post-hoc
                    yield transform(frame, **transform_kwargs) if transform is not None else frame

        # Mode 1: plain LeRobot dataset
        elif hasattr(raw_dataset, "meta") and hasattr(raw_dataset.meta, "total_episodes"):
            total_episodes = raw_dataset.meta.total_episodes
            if eps_idx is None:
                eps_idx = np.random.randint(0, total_episodes)
            start_frame = raw_dataset.base_dataset.episode_data_index["from"][eps_idx].item()
            end_frame   = raw_dataset.base_dataset.episode_data_index["to"][eps_idx].item()
            print(f"Episode {eps_idx} / {total_episodes}, frames {start_frame}..{end_frame} ({end_frame - start_frame} frames)")
            for idx in range(start_frame, end_frame, stride):
                raw = raw_dataset[idx]
                yield transform(raw, **transform_kwargs) if transform is not None else raw

        else:
            raise NotImplementedError(
                f"iter_episode is not supported for raw_dataset type {type(raw_dataset).__name__}"
            )


class Dataset(TransformableDataset, TorchDataset):
    def __init__(self, dataCfg: DataConfig, dataset: TorchDataset|Any, **kwargs) -> None:
        self.raw_dataset = dataset
        self.transform = dataCfg.transform
        self.transform_kwargs = kwargs.get("transform_kwargs", {})

    def __len__(self):
        return len(self.raw_dataset)  # type: ignore

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raw = self.raw_dataset[idx]
        data = self.transform(raw, **self.transform_kwargs)
        return data

    @property
    def dataset_length(self) -> int:
        return len(self.raw_dataset) # type: ignore

    @property
    def dataset_statistics(self) -> Dict[str, Any]:
        return getattr(self.raw_dataset, "dataset_statistics", {})
    

class IterableDataset(TransformableDataset, TorchIterableDataset):

    def __init__(
        self, dataCfg: DataConfig, dataset: TorchIterableDataset|Any, **kwargs
    ) -> None:
        self.raw_dataset = dataset
        self.transform = dataCfg.transform
        self.transform_kwargs = kwargs.get("transform_kwargs", {})

    def __iter__(self) -> Iterator[Any]:
        for rlds_batch in self.raw_dataset:
            try:
                yield self.transform(rlds_batch, **self.transform_kwargs)
            except Exception as e:
                print(f"Error processing batch: {e}")
                raise e

    @property
    def dataset_length(self) -> int:
        return len(self.raw_dataset) # type: ignore

    @property
    def dataset_statistics(self) -> Dict[str, Any]:
        return getattr(self.raw_dataset, "dataset_statistics", {})


class MixtureDataset(TransformableDataset, TorchDataset):

    specs: list[DatasetSpec]
    
    def __init__(self, datasets: dict[Dataset|IterableDataset, float], num_samples_per_epoch: int) -> None:
        self.datasets = list(datasets.keys())
        self.ratios = list(datasets.values())
        self.num_samples_per_epoch = num_samples_per_epoch

    def __getitem__(self, index):
        assert isinstance(index, tuple) and len(index) == 2, "Maybe forget to use batch sampler? see ../data/sampler.py"
        dataset_id, sample_id = index
        return self.datasets[dataset_id][sample_id]

    def __len__(self):
        # raise RuntimeError("Length is defined by the sampler")
        # return sum(d.dataset_length for d in self.datasets)
        return self.num_samples_per_epoch
