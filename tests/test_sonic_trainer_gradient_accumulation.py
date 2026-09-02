from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from psi.trainers.sonic import SonicTrainer
from psi.utils.attention import qwen3vl_attn_implementation


class _FakeAccelerator:
    sync_gradients = False
    is_main_process = False

    def accumulate(self, _model):
        return nullcontext()

    def autocast(self):
        return nullcontext()

    def backward(self, loss):
        loss.backward()


class _FakeOptimizer:
    def step(self):
        pass

    def zero_grad(self):
        pass


class _FakeScheduler:
    def step(self):
        pass

    def get_last_lr(self):
        return [1e-4]


def test_first_accumulation_micro_step_initializes_grad_norm():
    trainer = SonicTrainer.__new__(SonicTrainer)
    trainer.cfg = SimpleNamespace(
        train=SimpleNamespace(data_parallel="ddp", max_grad_norm=1.0)
    )
    trainer.model = torch.nn.Linear(1, 1)
    trainer.accelerator = _FakeAccelerator()
    trainer.optimizer = _FakeOptimizer()
    trainer.lr_scheduler = _FakeScheduler()
    trainer.forward_and_loss = lambda _model, _batch: {
        "loss": torch.tensor(1.0, requires_grad=True)
    }

    sync_gradients, metrics = trainer.training_step({})

    assert sync_gradients is False
    assert metrics["grad_norm_act"] == 0.0


def test_dataloader_worker_counts_follow_environment():
    trainer = SonicTrainer.__new__(SonicTrainer)
    trainer.cfg = SimpleNamespace(
        seed=292285,
        train=SimpleNamespace(train_batch_size=8, val_batch_size=16),
    )
    trainer.tokenizer = SimpleNamespace(model_max_length=128, pad_token_id=0)

    with patch.dict(
        "os.environ",
        {"PSI_TRAIN_NUM_WORKERS": "12", "PSI_VAL_NUM_WORKERS": "2"},
    ):
        train_loader, val_loader = trainer.create_dataloaders(
            list(range(32)), list(range(16))
        )

    assert train_loader.num_workers == 12
    assert val_loader.num_workers == 2


def test_attention_backend_honors_explicit_override():
    with patch.dict("os.environ", {"PSI_ATTN_IMPLEMENTATION": "sdpa"}):
        assert qwen3vl_attn_implementation() == "sdpa"
