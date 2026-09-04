from __future__ import annotations
from typing import Optional, Any, TYPE_CHECKING
import re
import torch
import os
import torch.nn as nn
import numpy as np
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from psi.data.dataset import TransformableDataset
import accelerate
from accelerate import Accelerator
from psi.trainers import Trainer,worker_init_fn
if TYPE_CHECKING:
    from psi.config.config import LaunchConfig
    from psi.config.model_qwen3vl import Qwen3VLModelConfig
    from psi.config.data_egodex import EgoDexDataConfig
    from psi.config.data_he import HERawDataConfig
    
from psi.config.data_mix import MixedDataConfig

from psi.utils import initialize_overwatch, shorten, flatten
overwatch = initialize_overwatch(__name__)

from transformers.modeling_outputs import CausalLMOutputWithPast
from psi.tokenizer import FastActionTokenizer
from psi.config.tokenizer import FastActionTokenizerConfig
from psi.trainers.qwen3vl_mixin import Qwen3vlMixin, PaddedCollatorForActionPrediction
from psi.utils import move_to_device

class PretrainTrainer(Qwen3vlMixin, Trainer):

    def __init__(self, cfg: LaunchConfig, device: torch.device):
        super().__init__(cfg, device)
        overwatch.info("Initialized Pretrain Trainer")
        self.Da = 48 # action dimension
        self._grad_norm_vlm = None
        self.maxmin = self.data_cfg.transform.field

    @property
    def task_run_name(self):
        tokenizer = self.model_cfg.action_tokenizer.__class__.__name__.replace("ActionTokenizerConfig", "").lower()
        dataset_name = shorten(self.data_cfg.transform.repack.dataset_name) # type: ignore
        delta = "delta" if self.data_cfg.use_delta_actions else "abs"
        return (
            f".{tokenizer}"
            f".{dataset_name}"
            f".{delta}"
            f".c{self.data_cfg.chunk_size}"
            f".{shorten(self.train_cfg.lr_scheduler_type)}"
            f".lr{self.train_cfg.learning_rate:.1e}"
        )
    
    @property
    def model_cfg(self) -> Qwen3VLModelConfig:
        return self.cfg.model  # type: ignore
    
    @property
    def data_cfg(self) -> EgoDexDataConfig | HERawDataConfig | MixedDataConfig:
        return self.cfg.data  # type: ignore
    
    def init_models(self):
        self.init_qwen3vl_models()
        self.im_end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        if isinstance(self.model_cfg.action_tokenizer, FastActionTokenizerConfig):
            self.model.resize_token_embeddings(
                len(self.tokenizer) + self.model_cfg.action_tokenizer.bins, 
                pad_to_multiple_of = 192,
                mean_resizing = True
            )
            overwatch.info(f"Resized model token embeddings to {self.model.lm_head.weight.shape[0]}")
        
    def prepare(self, accelerator: Accelerator) -> DataLoader:
        val_dataloader = getattr(self, "val_dataloader", None)
        if val_dataloader is not None: 
            self.val_dataloader = accelerator.prepare(self.val_dataloader)

        # if not (hasattr(self.data_cfg, "sampler") and self.data_cfg.sampler is not None) :
        if not (isinstance(self.data_cfg, MixedDataConfig) and self.data_cfg.sampler is not None):
            # dont prepare dataloader for mixed dataset as custom sampler is used
            self.train_dataloader = accelerator.prepare(self.train_dataloader)

        return self.prepare_qwen3vl(accelerator) # type: ignore

    def create_optimizer(
        self, num_training_steps: int | None = None
    ):
        optimizer_grouped_parameters = self.create_qwen3vl_optimizer()

        betas = self.train_cfg.lr_scheduler_kwargs["betas"]
        decay = self.train_cfg.lr_scheduler_kwargs["weight_decay"]
        eps = self.train_cfg.lr_scheduler_kwargs["eps"]
        assert isinstance(betas, tuple) and len(betas) == 2
        assert isinstance(decay, float)
        assert isinstance(eps, float)
        optimizer_cls = (
            torch.optim.AdamW
            if self.accelerator.state.deepspeed_plugin is None
            or "optimizer" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
            else accelerate.utils.DummyOptim
        )
        overwatch.info(f"Creating optimizer: {optimizer_cls.__name__}")

        self.optimizer = optimizer_cls(
            optimizer_grouped_parameters,
            lr=self.train_cfg.learning_rate,
            betas=(betas[0], betas[1]),
            weight_decay=decay, # type:ignore
            eps=eps,
            **({"foreach": self.train_cfg.optimizer_foreach} if self.train_cfg.optimizer_foreach is not None else {}),
        ) # type:ignore

    def create_datasets(self) -> tuple[TransformableDataset, TransformableDataset | None]:
        if isinstance(self.model_cfg.action_tokenizer, FastActionTokenizerConfig):
            self.action_tokenizer = FastActionTokenizer(
                self.tokenizer,self.data_cfg.chunk_size, self.Da, 
                pretrained_checkpoint=self.model_cfg.action_tokenizer.pretrained_checkpoint or "physical-intelligence/fast",
                bins=self.model_cfg.action_tokenizer.bins
            )
        else:
            raise NotImplementedError
        
        transform_kwargs=dict(
            action_tokenizer=self.action_tokenizer,
            vlm_processor=self.vlm_processor,
        )

        self.train_dataset = self.data_cfg(split="train", transform_kwargs=transform_kwargs)
        self.val_dataset = self.data_cfg(split="val", transform_kwargs={**transform_kwargs, "is_eval": True})
        return self.train_dataset, self.val_dataset

    def create_dataloaders(
        self, train_dataset, val_dataset
    ) -> tuple[DataLoader, DataLoader | None]:
        from psi.data.dataset import MixtureDataset
        # print(self.train_dataset);print(self.data_cfg.sampler);exit(0)
        # if isinstance(self.train_dataset, MixtureDataset):
        is_psix_mixture_dataset_enabled = hasattr(self.data_cfg, "mixture_spec") and self.data_cfg.mixture_spec is not None

        if isinstance(self.data_cfg, MixedDataConfig) and self.data_cfg.sampler is not None:
            assert isinstance(self.train_dataset, MixtureDataset)
            if self.data_cfg.sampler == "batch_mixture":
                from psi.data.sampler import BatchMixtureSampler
                batch_sampler = BatchMixtureSampler(
                    dataset_lens=[d.dataset_length for d in self.train_dataset.datasets],
                    mixture_ratios=self.train_dataset.ratios,
                    num_samples_per_epoch=len(self.train_dataset),
                    batch_size=self.train_cfg.train_batch_size,
                )
            elif self.data_cfg.sampler == "token_mixture":
                from psi.data.sampler import TokenMixtureSampler
                batch_sampler = TokenMixtureSampler(
                    specs=self.train_dataset.specs,
                    tokens_per_batch=self.data_cfg.tokens_per_device,
                    num_batches_per_rank=self.num_steps_per_epoch,
                )
            else:
                raise ValueError(f"Unknown sampler type: {self.data_cfg.sampler}")
            # self.accelerator.even_batches = False
            g = shuffle = batch_size = drop_last = None
        else:
            batch_sampler = None
            batch_size = self.train_cfg.train_batch_size
            drop_last = True

            

        collator = PaddedCollatorForActionPrediction(
            model_max_length=self.tokenizer.model_max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            padding_side="right", # for training
        )
        cfg_num_workers = getattr(self.data_cfg, "num_workers", None)
        num_workers = cfg_num_workers if cfg_num_workers is not None else 12

        if batch_sampler is not None:
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=collator,
                num_workers=num_workers,
                worker_init_fn=worker_init_fn,
                persistent_workers=True
            )
            self.train_sampler = batch_sampler # !!important
        else:
            train_dataloader_kwargs = {
                "num_workers": num_workers,
                "drop_last": drop_last or False,
                "persistent_workers": True,
                "pin_memory": True,
                "prefetch_factor": 2,
                # fork (linux default): workers inherit the dataset via COW; spawn would
                # pickle the 335k-episode metadata to every worker and never finish
                "multiprocessing_context": "fork" if num_workers > 0 else None,
            }
            if not is_psix_mixture_dataset_enabled:
                train_dataloader_kwargs["generator"] = torch.Generator().manual_seed(self.cfg.seed or 42)
                train_dataloader_kwargs["worker_init_fn"] = worker_init_fn
                train_dataloader_kwargs["shuffle"] = shuffle

            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                collate_fn=collator,
                **train_dataloader_kwargs
            )
        # FIXME support mixed dataset evaluation
        self.val_dataloader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=self.train_cfg.val_batch_size,
            collate_fn=PaddedCollatorForActionPrediction(
                model_max_length=self.tokenizer.model_max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                padding_side="left", # for batch autoregressive generation
            ),
            num_workers=0,
            worker_init_fn=worker_init_fn,
            persistent_workers=False,
            drop_last=False
        )
        return self.train_dataloader, self.val_dataloader

    def training_step(
        self,
        batch: dict[str, torch.Tensor | Any],
    ) -> tuple[bool, dict[str, Any]]:
        
        with self.accelerator.autocast():
            batch = move_to_device(batch, self.device) # type: ignore
            output: CausalLMOutputWithPast = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
                image_grid_thw=batch["image_grid_thw"],
            )
            loss = output.loss

        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            # dont do grad clipping for deepspeed
            if self.train_cfg.max_grad_norm is not None and self.train_cfg.data_parallel != "deepspeed":
                self._grad_norm_vlm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.cfg.train.max_grad_norm
                )
            else:
                self._grad_norm_vlm = self.get_total_grad_norm(self.model.parameters())
        
        action_preds = output.logits.detach()[:, :-1, :].argmax(dim=2) # type:ignore
        action_gt = batch["labels"][:, 1:].to(action_preds.device)
        if isinstance(self.model_cfg.action_tokenizer, FastActionTokenizerConfig):
            mask_start = self.action_tokenizer.action_token_begin_idx
            mask = (action_gt >= mask_start) & (action_gt < mask_start +  self.action_tokenizer.n_bins) # mask to get only action tokens
        else:
            raise NotImplementedError

        # Compute Accuracy
        correct = (action_preds == action_gt) & mask
        action_accuracy = correct.sum().float() / mask.sum().float()

        if isinstance(self.model_cfg.action_tokenizer, FastActionTokenizerConfig):
            action_gt_token_ids: list[list[int]] = [
                a[m].tolist() for a, m in zip(action_gt, mask)
            ]
            action_preds_token_ids: list[list[int]] = [
                a[m].tolist() for a, m in zip(action_preds, mask)
            ]

        raw_actions = batch["raw_actions"]  # list of (T_b, Da)
        l1_per_sample = []
        for gt_ids, pred_ids, raw in zip(action_gt_token_ids, action_preds_token_ids, raw_actions):
            T = len(raw)
            pred = torch.tensor(
                self.action_tokenizer.decode_token_ids_to_actions([pred_ids], time_horizon=T)[0], # type: ignore
                dtype=torch.float32)
            gt = torch.tensor(
                self.action_tokenizer.decode_token_ids_to_actions([gt_ids], time_horizon=T)[0], # type: ignore
                dtype=torch.float32)
            denorm_pred = self.maxmin.denormalize(pred) # type:ignore
            denorm_gt = self.maxmin.denormalize(gt) # type:ignore
            l1_per_sample.append(torch.abs(denorm_pred - denorm_gt).mean())
        action_l1_loss = torch.stack(l1_per_sample).mean() if l1_per_sample else torch.tensor(float("nan"))

        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        # Log batch["raw_image"] to wandb less often
        if (
            hasattr(self, "global_step")
            and self.global_step % (self.cfg.log.log_freq * 10) == 0
            and "observations" in batch
            and self.accelerator.is_main_process
        ):
            raw_imgs = batch["observations"]
            # Support both torch.Tensor and numpy
            if isinstance(raw_imgs, torch.Tensor):
                raw_imgs = raw_imgs.detach().cpu().numpy()
            # Log up to 4 images for visualization: concat them horizontally into one image
            img_arrays = []
            for i in range(min(10, len(raw_imgs))):
                img = raw_imgs[i][0]  # t=0
                img_arrays.append(img)

            # Images are assumed to have same shape and 3 channels; concat directly
            concat_img = np.concatenate(img_arrays, axis=1)
            wandb.log({"observations": [wandb.Image(concat_img, caption=f"raw images {self.global_step}")]}, step=self.global_step)

        return self.accelerator.sync_gradients, {
            "loss": loss.item(), # type:ignore
            "action_accuracy": action_accuracy.item(),
            "action_l1_loss": action_l1_loss.item(),
            "lr": self.lr_scheduler.get_last_lr()[0],
        }

    def evaluate(
        self,
    ) -> Optional[dict[str, float]]:
        unwrapped_model = self.unwrap_model()
        try:
            n_batches = len(self.val_dataloader)
        except TypeError:  # iterable dataset (psix sharded mixture) has no len
            n_batches = None
        if self.train_cfg.val_num_batches == -1:
            total_val_batches = n_batches
        elif n_batches is None:
            total_val_batches = self.train_cfg.val_num_batches
        else:
            total_val_batches = min(self.train_cfg.val_num_batches, n_batches)
        val_progress_bar = tqdm(
            total=total_val_batches,
            disable=not self.accelerator.is_local_main_process,
            position=1,
            leave=False,
        )
        val_progress_bar.set_description(f"Eval at global step {self.global_step}")

        l1_sum = 0.0
        t_abs_err_sum = 0.0
        n_items = 0
        decoding_error_count = 0
        val_iter = iter(self.val_dataloader)
        exhausted = False
        i = 0
        while total_val_batches is None or i < total_val_batches:
            # fetch first, then agree globally: ranks have uneven batch counts, and any
            # rank leaving this loop early deadlocks the zero3 collectives in generate()
            batch = next(val_iter, None)
            exhausted = batch is None
            if torch.distributed.is_initialized():
                continue_flag = torch.tensor([0.0 if exhausted else 1.0], device=self.device)
                torch.distributed.all_reduce(continue_flag, op=torch.distributed.ReduceOp.MIN)
                if continue_flag.item() == 0:
                    break
            elif exhausted:
                break

            gt_actions = batch["raw_actions"]  # list of (T_b, Da), T varies per sample
            batch = move_to_device(batch, self.device)
            generated_ids = unwrapped_model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                image_grid_thw=batch["image_grid_thw"],
                max_new_tokens=min(self.data_cfg.chunk_size * self.Da, 640),
                do_sample=False,
                # zero3: every decode forward needs all ranks' param gathers; keep ranks in
                # lockstep until the globally-longest sequence finishes or NCCL desyncs
                synced_gpus=True,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(batch["input_ids"], generated_ids)
            ]

            for b, pred_ids in enumerate(generated_ids_trimmed):
                if pred_ids.numel() == 0:
                    decoding_error_count += 1
                    continue

                im_end_positions = (pred_ids == self.im_end_token_id).nonzero(as_tuple=True)[0]
                if im_end_positions.numel() == 0:
                    decoding_error_count += 1
                    continue

                end_idx = int(im_end_positions[0].item())
                if end_idx == 0:
                    decoding_error_count += 1
                    continue

                T = len(gt_actions[b])
                pred_id_list = pred_ids[:end_idx].cpu().numpy().tolist()
                action_pred = self.action_tokenizer.decode_token_ids_to_actions(
                    [pred_id_list], time_horizon=T
                )
                denorm_action_pred = self.maxmin.denormalize(action_pred[0]) # (T, Da)
                denorm_gt_action = self.maxmin.denormalize(gt_actions[b])

                l1_sum += float(np.abs(denorm_action_pred - denorm_gt_action).mean())
                # predicted horizon (emitted coeffs / Da) vs GT horizon
                t_abs_err_sum += abs(
                    self.action_tokenizer.decoded_coeff_count(pred_id_list) / self.Da - T
                )
                n_items += 1

            i += 1
            val_progress_bar.update()

        val_progress_bar.close()
        if not exhausted and hasattr(self.val_dataloader, "end"):
            self.val_dataloader.end() # type: ignore

        # Aggregate across all processes
        stats = torch.tensor([l1_sum, t_abs_err_sum, n_items, decoding_error_count], dtype=torch.float64, device=self.device)
        stats = self.accelerator.reduce(stats, reduction="sum")
        total_l1_sum, total_t_err_sum, total_n_items, total_errors = stats.tolist()

        mean_l1 = total_l1_sum / total_n_items if total_n_items > 0 else float("nan")
        mean_t_err = total_t_err_sum / total_n_items if total_n_items > 0 else float("nan")
        return {
            "action_l1_loss": mean_l1,
            "horizon_abs_error": mean_t_err,
            "n_items": int(total_n_items),
            "decoding_error_count": int(total_errors),
        }

    def log(self, metrics: dict[str, float], start_time: Optional[float] = None) -> None:
        grad_norm_vlm = self._grad_norm_vlm.item() if torch.is_tensor(self._grad_norm_vlm) else float(self._grad_norm_vlm or 0.0)
        metrics.update(
            flatten({
                "grad_norm_vlm": grad_norm_vlm,
                # "lr": self.lr_scheduler.get_last_lr()[0],
            }, parent_key="train")  # type: ignore
        )
        # super().log(metrics, start_time)
        self.accelerator.log(metrics, step=self.global_step)

    def save_checkpoint(self, global_step: int) -> str | None:
        save_dir = os.path.join(self.project_dir, "checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        ckpt_dir = os.path.join(save_dir, f"ckpt_{global_step}")
        
        self.accelerator.save_model(self.model, ckpt_dir)
        return super().save_checkpoint(global_step)
    
    ####  adatped from transformers.Trainer  ####
    def get_decay_parameter_names(self, model) -> list[str]:
        """
        Get all parameter names that weight decay will be applied to.

        This function filters out parameters in two ways:
        1. By layer type (instances of layers specified in ALL_LAYERNORM_LAYERS)
        2. By parameter name patterns (containing 'bias', or variation of 'norm')
        """
        forbidden_name_patterns = [r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
        decay_parameters = self.get_parameter_names(model, [nn.LayerNorm], forbidden_name_patterns)
        return decay_parameters

    def get_parameter_names(self, model, forbidden_layer_types, forbidden_layer_names=None):
        """
        Returns the names of the model parameters that are not inside a forbidden layer.
        """
        forbidden_layer_patterns = (
            [re.compile(pattern) for pattern in forbidden_layer_names] if forbidden_layer_names is not None else []
        )
        result = []
        for name, child in model.named_children():
            child_params = self.get_parameter_names(child, forbidden_layer_types, forbidden_layer_names)
            result += [
                f"{name}.{n}"
                for n in child_params
                if not isinstance(child, tuple(forbidden_layer_types))
                and not any(pattern.search(f"{name}.{n}".lower()) for pattern in forbidden_layer_patterns)
            ]
        # Add model specific parameters that are not in any child
        result += [
            k for k in model._parameters if not any(pattern.search(k.lower()) for pattern in forbidden_layer_patterns)
        ]

        return result
    
    def clip_grad_norm(self) -> None:
        self._grad_norm_vlm = self.accelerator.clip_grad_norm_(
            self.model.parameters(), self.cfg.train.max_grad_norm
        )
    
