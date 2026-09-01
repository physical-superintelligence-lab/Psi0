import math
import os
import random
from dataclasses import dataclass

import torch
from torch.utils.data import Sampler

from psi.utils import initialize_overwatch

overwatch = initialize_overwatch(__name__)

class BatchMixtureSampler(Sampler):
    """
    Multi-node safe dataset mixture sampler.
    Example: datasets = [ds1, ds2], ratio = [4, 1]
    """
    def __init__(self, dataset_lens, mixture_ratios, num_samples_per_epoch, batch_size, seed=42):
        self.dataset_lens = dataset_lens
        self.weights = torch.tensor(mixture_ratios, dtype=torch.double)
        self.weights /= self.weights.sum()
        self.num_samples = num_samples_per_epoch
        self.batch_size = batch_size
        self.seed = seed

        # Distributed settings
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))

        # each rank gets this many samples
        self.num_samples_rank = math.ceil(self.num_samples / self.world_size)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Sample dataset IDs for all ranks
        all_dataset_ids = torch.multinomial(
            self.weights,
            self.num_samples_rank * self.world_size,
            replacement=True,
            generator=g,
        )

        # Shard by rank
        dataset_ids = all_dataset_ids[self.rank::self.world_size]

        # Vectorized sampling of indices within each dataset for efficiency
        dataset_ids_tensor = dataset_ids if isinstance(dataset_ids, torch.Tensor) else torch.tensor(dataset_ids)
        dataset_lens_tensor = torch.tensor(self.dataset_lens)
        lens_for_ids = dataset_lens_tensor[dataset_ids_tensor]
        rand_indices = torch.floor(torch.rand(len(dataset_ids_tensor), generator=g) * lens_for_ids).long()
        indices = list(zip(dataset_ids_tensor.tolist(), rand_indices.tolist()))
        
        # Group indices into batches
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        return iter(batches)

    def __len__(self):
        return math.ceil(self.num_samples_rank / self.batch_size)

@dataclass
class DatasetSpec:
    # dataset: torch.utils.data.Dataset
    dataset_length: int
    prob: float                     # mixture ratio
    image_size: tuple[int, int]     # e.g. 224, 448
    tokens_per_image: int           # (image_size / patch)^2

class TokenMixtureSampler(Sampler):
    """
    Multi-node safe dataset mixture sampler that samples from datasets with different image resolutoion while
    keeping the total number of tokens per batch (approximately) fixed.
    """
    def __init__(
        self,
        specs: list[DatasetSpec],
        tokens_per_batch: int, # per rank
        num_batches_per_rank: int,
        seed: int = 0,
    ):  
        self.specs = specs
        self.tokens_per_batch = tokens_per_batch
        self.num_batches_per_rank = num_batches_per_rank
        self.seed = seed

        self.probs = [s.prob for s in specs]
        self.epoch = 0

        # Distributed settings
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))

    def set_epoch(self, epoch):
        self.epoch = epoch
        # overwatch.info(f"TokenMixtureSampler set to epoch {epoch}")

    def __iter__(self):
        # assert self.epoch is not None, "TokenMixtureSampler epoch not set. Please call set_epoch(epoch) before iterating."

        # Synchronize dataset selection for each logical batch across all ranks.
        for batch_idx in range(self.num_batches_per_rank):
            epoch_batch_idx = self.epoch * self.num_batches_per_rank + batch_idx
            dataset_rng = random.Random(f"{self.seed}:dataset:{epoch_batch_idx}")
            dataset_id = dataset_rng.choices(
                range(len(self.specs)), weights=self.probs, k=1
            )[0]
            spec = self.specs[dataset_id]
            batch_size = max(1, self.tokens_per_batch // spec.tokens_per_image)
            # Sample different examples on every rank without reusing the previous
            # epoch's random streams.
            local_batch_rng = random.Random(
                f"{self.seed}:samples:{epoch_batch_idx}:rank:{self.rank}"
            )
            indices = [
                (dataset_id, local_batch_rng.randrange(spec.dataset_length))
                for _ in range(batch_size)
            ]
            yield indices

    def __len__(self):
        return self.num_batches_per_rank
