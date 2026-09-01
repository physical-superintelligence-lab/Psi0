import pytest

from psi.data.sampler import DatasetSpec, TokenMixtureSampler

SPECS = [
    DatasetSpec(
        dataset_length=100_000,
        prob=0.5,
        image_size=(270, 480),
        tokens_per_image=540,
    ),
    DatasetSpec(
        dataset_length=100_000,
        prob=0.5,
        image_size=(240, 320),
        tokens_per_image=320,
    ),
]
TOKENS_PER_BATCH = 8_640
NUM_BATCHES = 8
WORLD_SIZE = 4


def collect_rank_batches(monkeypatch: pytest.MonkeyPatch, epoch: int):
    batches_by_rank = []
    for rank in range(WORLD_SIZE):
        monkeypatch.setenv("RANK", str(rank))
        monkeypatch.setenv("WORLD_SIZE", str(WORLD_SIZE))
        sampler = TokenMixtureSampler(
            specs=SPECS,
            tokens_per_batch=TOKENS_PER_BATCH,
            num_batches_per_rank=NUM_BATCHES,
            seed=0,
        )
        sampler.set_epoch(epoch)
        batches_by_rank.append(list(sampler))
    return batches_by_rank


def test_token_mixture_sampler_synchronizes_global_batches(
    monkeypatch: pytest.MonkeyPatch,
):
    batches_by_rank = collect_rank_batches(monkeypatch, epoch=0)
    selected_datasets = set()

    for batch_idx in range(NUM_BATCHES):
        rank_batches = [batches[batch_idx] for batches in batches_by_rank]
        dataset_ids = [batch[0][0] for batch in rank_batches]
        selected_datasets.add(dataset_ids[0])

        assert len(set(dataset_ids)) == 1
        assert len({len(batch) for batch in rank_batches}) == 1

        sample_indices = [tuple(index for _, index in batch) for batch in rank_batches]
        assert len(set(sample_indices)) == WORLD_SIZE

    assert selected_datasets == {0, 1}


def test_token_mixture_sampler_is_reproducible_without_reusing_epochs(
    monkeypatch: pytest.MonkeyPatch,
):
    epoch_zero = collect_rank_batches(monkeypatch, epoch=0)
    assert collect_rank_batches(monkeypatch, epoch=0) == epoch_zero

    epoch_one = collect_rank_batches(monkeypatch, epoch=1)
    epoch_zero_batches = {
        tuple(batch) for rank_batches in epoch_zero for batch in rank_batches
    }
    epoch_one_batches = {
        tuple(batch) for rank_batches in epoch_one for batch in rank_batches
    }

    assert epoch_zero_batches.isdisjoint(epoch_one_batches)
