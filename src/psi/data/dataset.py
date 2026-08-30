from __future__ import annotations
from typing import Any, Dict, TYPE_CHECKING

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