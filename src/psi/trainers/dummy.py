from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import TYPE_CHECKING, Any, Union
from transformers import AutoProcessor

from accelerate import Accelerator
from psi.trainers.trainer import Trainer, worker_init_fn
from psi.trainers.qwen3vl_mixin import PaddedCollatorForTogether
from psi.utils import initialize_overwatch

if TYPE_CHECKING:
    from psi.config.config import TrainConfig
    from psi.config.model_psix import PsixModelConfig

overwatch = initialize_overwatch(__name__)


class DummyTrainer(Trainer):
    """Trainer for dataloader testing only.

    Loads only the VLM processor/tokenizer (no model weights) so that the
    full data pipeline — transforms, collation, dataloader — runs exactly as
    in production without allocating GPU memory for the large model.

    Usage:
        trainer = DummyTrainer.instantiate(config, device_id)
        trainer.init_models()
        train_ds, val_ds = trainer.create_datasets()
        trainer.create_dataloaders(train_ds, val_ds)
        for batch in trainer.train_dataloader:
            ...
    """

    @property
    def task_run_name(self) -> str:
        return ".dummy"

    @property
    def task_cfg(self) -> TrainConfig:
        return self.cfg.train

    @property
    def model_cfg(self) -> PsixModelConfig:
        return self.cfg.model  # type: ignore

    # ------------------------------------------------------------------
    # init_models: skip VLM weights, load only processor + tokenizer
    # ------------------------------------------------------------------

    def init_models(self):
        overwatch.info(
            f"DummyTrainer: loading processor from {self.model_cfg.model_name_or_path}"
        )
        self.vlm_processor = AutoProcessor.from_pretrained(
            self.model_cfg.model_name_or_path
        )
        self.tokenizer = self.vlm_processor.tokenizer
        self.model = nn.Linear(1, 1)
        overwatch.info("DummyTrainer: model stub ready (nn.Linear(1,1))")

    # ------------------------------------------------------------------
    # create_datasets / create_dataloaders — mirrors PsixFinetuneTrainer
    # ------------------------------------------------------------------

    def create_datasets(self):
        transform_kwargs = dict(vlm_processor=self.vlm_processor)
        self.train_dataset = self.data_cfg(split="train", transform_kwargs=transform_kwargs)
        self.val_dataset = self.data_cfg(split="val", transform_kwargs=transform_kwargs)
        return self.train_dataset, self.val_dataset

    def create_dataloaders(self, train_dataset, val_dataset):
        is_psix_mixture_dataset_enabled = (
            hasattr(self.data_cfg, "mixture_spec") and self.data_cfg.mixture_spec is not None
        )

        cfg_num_workers = getattr(self.data_cfg, "num_workers", None)
        num_workers = cfg_num_workers if cfg_num_workers is not None else 12

        train_dataloader_kwargs = {
            "num_workers": num_workers,
            "drop_last": True,
            "persistent_workers": True,
            "pin_memory": True,
            "prefetch_factor": 2,
        }

        val_dataloader_kwargs = {
            "num_workers": 1,
            "drop_last": False,
            "persistent_workers": True,
        }

        if not is_psix_mixture_dataset_enabled:
            g = torch.Generator().manual_seed(self.cfg.seed or 42)
            train_dataloader_kwargs["shuffle"] = True
            train_dataloader_kwargs["generator"] = g
            train_dataloader_kwargs["worker_init_fn"] = worker_init_fn

        collator = PaddedCollatorForTogether(
            model_max_length=self.tokenizer.model_max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            padding_side="right",
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.task_cfg.train_batch_size,
            collate_fn=collator,
            **train_dataloader_kwargs,
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.task_cfg.val_batch_size,
            collate_fn=collator,
            **val_dataloader_kwargs,
        )
        return self.train_dataloader, self.val_dataloader

    # ------------------------------------------------------------------
    # Stubs for abstract methods not needed for dataloader testing
    # ------------------------------------------------------------------

    def prepare(self, accelerator: Accelerator) -> DataLoader:
        self.accelerator = accelerator
        return self.train_dataloader

    def training_step(self, batch: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        raise NotImplementedError("DummyTrainer is not for training")

    def evaluate(self):
        return None
