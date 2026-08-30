from __future__ import annotations
import os
import json
import torch
import wandb
import pickle
import numpy as np
from tqdm import tqdm
from copy import copy, deepcopy
import contextlib
import torch.nn.functional as F
from typing import Dict, Optional, List, Union, Any, TYPE_CHECKING
from torch.utils.data import DataLoader
from psi.data.dataset import TransformableDataset
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor #, Qwen2TokenizerFast, Qwen3VLProcessor
from psi.trainers.qwen3vl_mixin import PaddedCollatorForTogether

if TYPE_CHECKING:    
    from psi.config.config import TrainConfig #, LaunchConfig
    from psi.config.model_psi0 import Psi0ModelConfig
    from psi.config.transform import Psi0ModelTransform
    from psi.config.data_he import HERawDataConfig

from psi.config.data_mix import MixedDataConfig

from accelerate import Accelerator

# from utils.overwatch import initialize_overwatch
from psi.utils import (
    initialize_overwatch,
    shorten,
    move_to_device,
    batch_str_to_tensor
)

overwatch = initialize_overwatch(__name__)

# from .base import Trainer
from .trainer import Trainer, worker_init_fn

from psi.utils import flatten, shorten, move_to_device, rmse, seed_everything
from psi.models.psi0 import Psi0Model

class PosttrainTrainer(Trainer):

    def __init__(self, cfg, device: torch.device):
        super().__init__(cfg, device)

        self.noise_scheduler_name = self.model_cfg.noise_scheduler
        self.train_diffusion_steps = self.model_cfg.train_diffusion_steps
        self.eval_diffusion_steps = self.model_cfg.eval_diffusion_steps
        self.ac_chunk = self.model_cfg.action_chunk_size
        self.ac_dim = self.model_cfg.action_dim
        self.maxmin = self.data_cfg.transform.field

        if self.model_cfg.action_dim == 7:
            w_xyz, w_rpy, w_gripper = self.model_cfg.loss_w
            self.loss_w = (
                torch.from_numpy(np.array([w_xyz] * 3 + [w_rpy] * 3 + [w_gripper]))
                .float()
                .to(self.device)
            )  # (7,)
        else:
            self.loss_w = torch.tensor([1.0 / self.model_cfg.action_dim] * self.model_cfg.action_dim, dtype=torch.float32).to(self.device)
        assert self.loss_w.sum() == 1.0, "Weights better sum to 1.0 to keep loss range consistent"

        if self.noise_scheduler_name == "ddpm":
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler
            self.noise_scheduler = DDIMScheduler(  # FIXME configure args
                num_train_timesteps=self.train_diffusion_steps,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                set_alpha_to_one=True,
                steps_offset=0,
                prediction_type="epsilon",
            )
        elif self.noise_scheduler_name == "flow":
            from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
                FlowMatchEulerDiscreteScheduler,
            )
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=self.train_diffusion_steps,  # MUST be 1000 as per pretrained SD3
            )
        assert self.train_cfg.mixed_precision == "bf16", "keep qwen3vl in bf16 while training action header in fp32"
        # assert self.model_cfg.n_conditions == len(self.data_cfg.transform.repack.conditions), "inconsistent confs" # type: ignore

    @property
    def data_cfg(self) -> HERawDataConfig | MixedDataConfig:
        return self.cfg.data # type: ignore
    
    @property
    def model_cfg(self) -> Psi0ModelConfig:
        return self.cfg.model # type: ignore
    
    def init_qwen3vl_models(self):
        vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_cfg.model_name_or_path,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16
        )

        self.vlm_processor = AutoProcessor.from_pretrained(self.model_cfg.model_name_or_path)
        self.tokenizer = self.vlm_processor.tokenizer

        assert self.cfg.train.lora == False
        return vlm_model

    def init_models(self):
        # init pretrained vlm
        vlm_model = self.init_qwen3vl_models()

        self.model = Psi0Model(
            model_cfg=self.model_cfg,
            vlm_model=vlm_model,
            # dtype=self.dtype
        )
        # print("L129", self.model.action_header.time_ins_embed.out_net[0].weight.dtype) # debug
        
        if self.train_cfg.data_parallel == "deepspeed":
            # HACK to set correct config for deepspeed
            self.model.config = vlm_model.config
            self.model.config.hidden_size = vlm_model.config.text_config.hidden_size

        # only support freeze vlm now
        assert self.model_cfg.tune_vlm == False
        for p in self.model.vlm_model.parameters():
            p.requires_grad = False

        num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        overwatch.info(f"Model has {num_parameters:,} trainable parameters")


    def create_optimizers(self):
        # Separate parameters into groups
        action_header_params = []
        vlm_model_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if name.startswith("action_header."):
                action_header_params.append(param)
            elif name.startswith("vlm_model."):
                if self.model_cfg.tune_vlm:
                    vlm_model_params.append(param)
            else:
                other_params.append(param)

        param_groups = []
        if action_header_params:
            param_groups.append({
                'params': action_header_params,
                'lr': self.cfg.train.learning_rate,
                'group_name': 'action_header'
            })
        if vlm_model_params:
            param_groups.append({
                'params': vlm_model_params,
                'lr': self.model_cfg.lang_backbone_lr,
                'group_name': 'vlm_model'
            })
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.cfg.train.learning_rate,  # Default group
                'group_name': 'other'
            })

        optimizer_kwargs = dict(self.cfg.train.lr_scheduler_kwargs)
        if self.cfg.train.optimizer_foreach is not None:
            optimizer_kwargs["foreach"] = self.cfg.train.optimizer_foreach
        self.optimizer = torch.optim.AdamW(
            param_groups,
            **optimizer_kwargs, # type: ignore
        )

    def create_lr_schedulers(
        self,
        num_training_steps: int | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ):
        from transformers.optimization import get_scheduler
        assert num_training_steps is not None
        self.lr_scheduler = get_scheduler(
            self.cfg.train.lr_scheduler_type,
            num_training_steps=num_training_steps * self.world_size,
            optimizer=optimizer if optimizer is not None else self.optimizer,
            num_warmup_steps=self.num_warmup_steps * self.world_size,
            scheduler_specific_kwargs=self.cfg.train.scheduler_specific_kwargs,
        )

    def create_optimizer_and_scheduler(self, num_training_steps: int | None = None):
        optimizer = self.create_optimizers()
        self.create_lr_schedulers(
            num_training_steps=num_training_steps, optimizer=optimizer
        )

    def create_datasets(self) -> tuple[TransformableDataset, TransformableDataset | None]:
        transform_kwargs=dict(
            vlm_processor=self.vlm_processor,
        )
        self.train_dataset = self.data_cfg(split="train", transform_kwargs=transform_kwargs)
        self.val_dataset = self.data_cfg(split="val", transform_kwargs=transform_kwargs)
        return self.train_dataset, self.val_dataset

    def create_dataloaders(self, train_dataset, val_dataset):
        from psi.data.dataset import MixtureDataset
        
        if isinstance(self.data_cfg, MixedDataConfig):
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
            
            g = shuffle = batch_size = drop_last = None
        else:
            batch_sampler = None
            shuffle = True
            batch_size = self.train_cfg.train_batch_size
            drop_last = True

            g = torch.Generator()
            g.manual_seed(self.cfg.seed or 42)
        
        collator = PaddedCollatorForTogether(
            model_max_length=self.tokenizer.model_max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            padding_side="right",
        )
        if batch_sampler is not None:
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=collator,
                num_workers=1,
                worker_init_fn=worker_init_fn,
                persistent_workers=True
            )
            self.train_sampler = batch_sampler # !!important 
        else:
            self.train_dataloader = DataLoader(
                train_dataset,
                shuffle=shuffle, # 
                batch_sampler=batch_sampler,
                generator=g, # 
                batch_size=batch_size, # 
                collate_fn=collator,
                num_workers=1,
                worker_init_fn=worker_init_fn,
                persistent_workers=True,
                drop_last=drop_last or False
            )

        val_dataloader_kwargs = {
            "num_workers": 1,
            "drop_last": False,
            # "pin_memory": True,
            "persistent_workers": True,
            "shuffle": True,
        }  # 1 small enough to fit im Mem. 2. no need distributed sampler
        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.train_cfg.val_batch_size,
            collate_fn=collator,
            **val_dataloader_kwargs,
        )
        return self.train_dataloader, self.val_dataloader

    @property
    def task_run_name(self):
        dataset_name = shorten(self.data_cfg.transform.repack.dataset_name) # type:ignore
        return (
            f".{dataset_name}"
            f".{self.noise_scheduler_name}{self.train_diffusion_steps}"
            f".{shorten(self.train_cfg.lr_scheduler_type)}"
            f".lr{self.train_cfg.learning_rate:.1e}"
        )

    def prepare(self, accelerator: Accelerator) -> DataLoader:
        # NOTE: if use deepspeed, model, optimizer, lr_scheduler must be prepared together !!! 
        # print("L301", self.model.action_header.time_ins_embed.out_net[0].weight.dtype) # debug
        self.model, self.optimizer, self.lr_scheduler = accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler
        )
        # self.model.action_header.to(torch.float32) # HACK keep action header in fp32
        # print("L307", self.model.module.action_header.time_ins_embed.out_net[0].weight.dtype) # debug

        if not (isinstance(self.data_cfg, MixedDataConfig) and self.data_cfg.sampler is not None):
            # dont prepare dataloader for mixed dataset as custom sampler is used
            self.train_dataloader = accelerator.prepare(self.train_dataloader)

        if self.train_cfg.data_parallel == "deepspeed":
            # Gradient Checkpoint Setup
            if self.model_cfg.gradient_checkpointing and self.model_cfg.tune_vlm:
                assert "DeepSpeedEngine" in self.model.__class__.__name__, "deepspeed is not properly initialized!"
                if hasattr(self.model, "enable_input_require_grads"): 
                    self.model.vlm_model.enable_input_require_grads()
                else:
                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)
                    self.model.vlm_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        val_dataloader = getattr(self, "val_dataloader", None)
        if val_dataloader is not None: # not using if self.val_dataloader to avoid DataLoader.__len__() being called on iterable dataset
            self.val_dataloader = accelerator.prepare(self.val_dataloader)

        if self.cfg.train.overfit_single_batch:
            overwatch.warning("Overfitting a single batch: reusing first batch every step. set cfg.data.image_aug = False for true memorization.")
            first_batch = next(iter(self.train_dataloader))
            class SingleBatchLoader:
                def __iter__(self): 
                    while True:
                        yield first_batch
                def __len__(self):
                    return 1
            self.train_dataloader = SingleBatchLoader()
            self.val_dataloader = SingleBatchLoader()
            
        self.accelerator = accelerator
        return self.train_dataloader # type: ignore

    def training_step(
        self,
        batch: dict[str, Union[torch.Tensor, Any]],
    ) -> tuple[bool, dict[str, Any]]:

        if self.train_cfg.data_parallel == "deepspeed":
            gac = contextlib.nullcontext()
        else:
            # otherwise gradient accumulation is managed by deepspeed
            gac = self.accelerator.accumulate(self.model)
        with gac:
            with self.accelerator.autocast():
                batch = move_to_device(batch, self.device) # type: ignore
                losses = self.forward_and_loss(self.model, batch)

            self.accelerator.backward(losses["loss"])
            # overwatch.debug(self.global_step, self.device, batch["dataset_name"])
            if self.accelerator.sync_gradients:
                if self.train_cfg.max_grad_norm is not None:
                    self._grad_norm_act = self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.cfg.train.max_grad_norm
                    )
                else:
                    self._grad_norm_act = self.get_total_grad_norm(self.model.parameters())

                if self.train_cfg.data_parallel == "ddp":
                    # NOTE: It is normal that param.grad is None for deepspeed and fsdp
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is None:
                            overwatch.critical(f"[Unused] {name} did not receive a gradient.")  

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        # Log batch["raw_image"] to wandb every 100 steps
        if (
            hasattr(self, "global_step")
            and self.global_step % self.cfg.log.log_freq == 0
            and "raw_images" in batch
            and self.accelerator.is_main_process
            and self.cfg.log.report_to == "wandb"
        ):
            raw_imgs = batch["raw_images"]
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
            wandb.log({"raw_images": [wandb.Image(concat_img, caption=f"raw images {self.global_step}")]}, step=self.global_step)

        self.train_loss_tracker = step_loss = self.accelerator.gather(losses["loss"]).mean().item() # type: ignore

        return (self.accelerator.sync_gradients, {
            "lr_act": self.lr, 
            "grad_norm_act": self._grad_norm_act, # type: ignore
            "loss": step_loss
        })

    @torch.no_grad()
    def inference(self, eval_model, repacked_batch):

        traj2ds = repacked_batch["traj2ds"] if "traj2ds" in repacked_batch else None  # (B, 3, H, W)
        obs = repacked_batch["states"]  # (B,1,M)

        # imgs_in = {"cam0": view_features}  # views # TODO support multi-view
        bsz = obs.shape[0]

        action_samples = torch.randn(
            bsz, self.ac_chunk, self.ac_dim, device=self.device
        )
        self.noise_scheduler.set_timesteps(self.eval_diffusion_steps)

        for timestep in self.noise_scheduler.timesteps:
            batched_timestep = timestep.expand(bsz).to(self.device)

            model_pred = eval_model(
                input_ids=repacked_batch["input_ids"],#####
                attention_mask=repacked_batch["attention_mask"], # vlm related
                pixel_values=repacked_batch["pixel_values"],
                image_grid_thw=repacked_batch["image_grid_thw"], ####
                action_samples=action_samples, # (B,Tp,Da)
                states=obs,
                timestep=batched_timestep,
                traj2ds=traj2ds,
                return_dict=True,
            ).action

            action_samples = self.noise_scheduler.step(
                model_output=model_pred, timestep=timestep, sample=action_samples # type: ignore
            ).prev_sample # type: ignore

        return action_samples
    
    def save_checkpoint(self, global_step: int) -> str | None:
        save_dir = os.path.join(self.project_dir, "checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        ckpt_dir = os.path.join(save_dir, f"ckpt_{global_step}")
        
        self.accelerator.save_model(self.model, ckpt_dir)
        return super().save_checkpoint(global_step)

    def evaluate(self) -> dict[str, float] | None:
        accelerator = self.accelerator
        global_step = self.global_step
        eval_model = self.unwrap_model()

        total_val_batches = (
            len(self.val_dataloader)
            if self.train_cfg.val_num_batches == -1
            else min(self.train_cfg.val_num_batches, len(self.val_dataloader))
        )
        val_progress_bar = tqdm(
            self.val_dataloader,
            total=total_val_batches,
            disable=not accelerator.is_local_main_process,
            position=1,
            leave=False,
        )
        val_progress_bar.set_description(f"Eval at global step {global_step}")

        val_loss_list = []
        action_l1_err_list = []

        for val_step, val_batch in enumerate(val_progress_bar):
            val_batch: dict = move_to_device(batch_str_to_tensor(val_batch), self.device) # type: ignore
            # mask = val_batch["mask"]
            mask = torch.ones_like(val_batch["actions"]) # FIXME
            gt_actions = val_batch["actions"]  # (B, Tp, Da)
            # gt_actions = self.data_cfg.data_transforms.normalize_action(repacked_batch[1])

            # Tp -> predicted action horizon, Da -> action dim
            B, Tp, Da = gt_actions.shape

            # validation loss
            with accelerator.autocast():
                val_loss = self.forward_and_loss(eval_model, val_batch)
                val_loss_list.append(accelerator.gather(val_loss["loss"]))

                # action prediction loss
                pred_actions = self.inference(eval_model, val_batch)
                err_action_l1 = pred_actions - gt_actions  # (B, Tp, Da)

                err_action_l1_all = accelerator.gather(
                    err_action_l1[:, :Tp].contiguous()
                )
                err_action_masks_all = accelerator.gather(mask[:, :Tp].contiguous())
                err_action_l1 = err_action_l1_all[
                    err_action_masks_all.to(torch.bool) # type: ignore
                ].abs()
                action_l1_err_list.append(err_action_l1.reshape(-1, Da).float().cpu().numpy())  # (B*world_size*Ta, 7)

            if val_step + 1 >= total_val_batches:
                if accelerator.is_local_main_process:
                    val_progress_bar.close()
                if hasattr(self.val_dataloader, "end"):
                    self.val_dataloader.end() # type: ignore
                break

        avg_val_loss = torch.cat(val_loss_list).mean().item()
        action_l1_err_list = np.concatenate(action_l1_err_list, axis=0)  # (len_val_dataset*Ta, Da)
        action_l1_err_list_denormed = self.maxmin.denormalize_L1_action_err( action_l1_err_list) # type: ignore

        # action L1 errors
        avg_action_errors_denormed = action_l1_err_list_denormed.mean(0)  # (Da,) NOTE only if the error is L1 (linear)
        labels_denormed = [
            "hand_left_l1",
            "hand_right_l1",
            "arm_left_l1",
            "arm_right_l1",
        ]
        avg_lr_action_err_denormed = np.split(
            avg_action_errors_denormed, [7, 14, 21], axis=-1
        )
        l1_action_errs = [
            float(np.mean(x)) for x in avg_lr_action_err_denormed
        ]

        # log metrics
        return {
            "loss": avg_val_loss,
            **dict(zip(labels_denormed, l1_action_errs))
        }

    def forward_and_loss(self, model, batch) -> dict[str, torch.Tensor]:
        bsz, *_, h, w = batch["actions"].shape
        # for debugging purposes
        # gen = torch.Generator(device=self.device)
        # gen.manual_seed(42)  # or any fixed seed you want
        gen = None
        if self.noise_scheduler_name == "ddpm":
            timesteps = torch.randint(
                low=0, high=self.train_diffusion_steps, size=(bsz,), device=self.device
            ).long()

        elif self.noise_scheduler_name == "flow":
            # For reproducibility/debugging
            sigmas = torch.rand((bsz,), device=self.device, generator=gen).float()
            timesteps = sigmas * self.train_diffusion_steps
            sigmas = sigmas.view(-1, 1, 1)
        else:
            raise ValueError(f"Unknown noise scheduler: {self.noise_scheduler_name}")

        # noise_action = torch.randn_like(batch["actions"])
        noise_action = torch.randn(batch["actions"].shape, device=self.device, generator=gen)
        if self.noise_scheduler_name == "ddpm":
            noisy_actions = self.noise_scheduler.add_noise(
                batch["actions"], noise_action, timesteps # type: ignore
            )
            target_action = noise_action  # ddpm epsilon-prediction
        elif self.noise_scheduler_name == "flow":
            sigmas_action = sigmas.view(-1, 1, 1)  # B, Tp, action_dim # type: ignore
            noisy_actions = (1 - sigmas_action) * batch["actions"] + sigmas_action * noise_action
            target_action = noise_action - batch["actions"] # flow-matching velocity prediction
        else:
            raise ValueError
        
        model_output = model(
            input_ids=batch["input_ids"],#####
            attention_mask=batch["attention_mask"], # vlm related
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"], ####
            action_samples=noisy_actions, # (B,Tp,Da)
            states=batch["states"],  # (B,1,M)
            timestep=timesteps,
            traj2ds=batch["traj2ds"] if "traj2ds" in batch else None,  # (B, C, 3, H, W)
            return_dict=True,
        )
        action_pred = model_output.action
        loss_action = F.mse_loss(
            action_pred.float(), target_action.float(), reduction="none"
        )  # (B, Tp, Da)
        
        # for post pre training, we only care about the overall loss
        # """  
        #     sum(1) -> to keep gradient consistent when training with different action chunks
        #     mean(0) -> to average over the batch
        #     sum() -> to sum over the weighted loss of each action dimension
        # """
        if "actions_mask" not in batch:
            batch["actions_mask"] = torch.ones_like(batch["actions"])
        loss_action = (loss_action * batch["actions_mask"]).sum(1)  # (B, Da)
        loss_action = (loss_action.mean(0) * self.loss_w).sum()
        return {"loss": loss_action}
