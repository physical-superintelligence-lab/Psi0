import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import Union, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from psi.config.config import TrainConfig, LaunchConfig
from psi.config.data_lerobot import LerobotDataConfig
from psi.config.model_act import ACTModelConfig
from act.models.act import ACTConfig, ACTPolicy
from psi.trainers import Trainer

from psi.utils import flatten, shorten, initialize_overwatch, rmse, seed_everything
from psi.utils.utils import batch_str_to_tensor

overwatch = initialize_overwatch(__name__)

from accelerate import Accelerator


class ActG1Trainer(Trainer):
    epoch_loss: list[float]
    _loss_cpu: float

    def __init__(self, cfg: LaunchConfig, device: torch.device):
        super().__init__(cfg, device)
        overwatch.info("Initialized ACT G1 Trainer")
        self._loss_cpu: float = 0.0

    @property
    def task_run_name(self):
        return (
            ".g1"
            f".{shorten(self.train_cfg.lr_scheduler_type)}"
            f".lr{self.train_cfg.learning_rate:.1e}"
        )

    @property
    def task_cfg(self) -> TrainConfig:
        return self.cfg.train

    @property
    def model_cfg(self) -> ACTModelConfig:
        return self.cfg.model  # type: ignore

    @property
    def data_cfg(self) -> LerobotDataConfig:
        return self.cfg.data  # type: ignore

    def init_models(self):
        """Initialize ACT model."""
    
        act_config = ACTConfig(
            chunk_size=self.model_cfg.chunk_size,
            n_action_steps=self.model_cfg.n_action_steps,
            action_dim=self.model_cfg.action_dim,
            state_dim=self.model_cfg.state_dim,
            dim_model=self.model_cfg.dim_model,
            n_heads=self.model_cfg.n_heads,
            dim_feedforward=self.model_cfg.dim_feedforward,
            feedforward_activation=self.model_cfg.feedforward_activation,
            n_encoder_layers=self.model_cfg.n_encoder_layers,
            n_decoder_layers=self.model_cfg.n_decoder_layers,
            pre_norm=self.model_cfg.pre_norm,
            dropout=self.model_cfg.dropout,
            use_vae=self.model_cfg.use_vae,
            latent_dim=self.model_cfg.latent_dim,
            n_vae_encoder_layers=self.model_cfg.n_vae_encoder_layers,
            kl_weight=self.model_cfg.kl_weight,
            temporal_ensemble_coeff=self.model_cfg.temporal_ensemble_coeff,
        )
        
        self.model = ACTPolicy(config=act_config, dataset_stats=None)  # type: ignore
        self.epoch_loss = list()

    def prepare(self, accelerator: Accelerator):
        self.optimizer, self.lr_scheduler = accelerator.prepare(
            self.optimizer, self.lr_scheduler
        )
        self.model = accelerator.prepare(self.model)
        return super().prepare(accelerator)

    def create_datasets(self) -> tuple[Dataset, Dataset | None]:
        self.train_dataset = self.data_cfg(split="train")
        self.val_dataset = self.data_cfg(split="val")
        return self.train_dataset, self.val_dataset

    def create_dataloaders(self, train_dataset, val_dataset):
        g = torch.Generator()
        g.manual_seed(self.cfg.seed or 42)

        cfg_num_workers = getattr(self.data_cfg, "num_workers", None)
        num_workers = cfg_num_workers if cfg_num_workers is not None else 12
        is_psix_mixture_dataset_enabled = hasattr(self.data_cfg, "mixture_spec") and self.data_cfg.mixture_spec is not None

        train_dataloader_kwargs = {
            "num_workers": num_workers,
            "drop_last": True,
            "persistent_workers": True,
            "pin_memory": True,
            "prefetch_factor": 2,
        }
        if not is_psix_mixture_dataset_enabled:
            train_dataloader_kwargs["shuffle"] = True
            train_dataloader_kwargs["generator"] = g
            train_dataloader_kwargs["worker_init_fn"] = lambda worker_id: seed_everything(
                self.cfg.seed + worker_id
            )

        val_dataloader_kwargs = {
            "num_workers": 4,
            "drop_last": False,
            "pin_memory": True,
            "persistent_workers": True,
        }

        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.train_cfg.train_batch_size,
            **train_dataloader_kwargs,
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.train_cfg.val_batch_size,
            **val_dataloader_kwargs,
        )
        return self.train_dataloader, self.val_dataloader

    def next_epoch(self, epoch):
        self.epoch_loss.append(self._loss_cpu)
        return super().next_epoch(epoch)

    def training_step(
        self,
        batch: dict[str, Union[torch.Tensor, Any]],
    ) -> tuple[bool, dict[str, Any]]:
        with self.accelerator.accumulate(self.model):
            batch = self._prepare_batch(batch)
            loss_dict = self.model(batch)
            loss = loss_dict["loss"]

            # optimize
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
            # step lr scheduler every batch
            # this is different from standard pytorch behavior
            self.lr_scheduler.step()

            # Logging
            loss_cpu = loss.item()
            self._loss_cpu = loss_cpu
            
            metrics = {"loss": loss_cpu, "l1_loss": loss_dict["l1_loss"]}
            if "kld_loss" in loss_dict:
                metrics["kld_loss"] = loss_dict["kld_loss"]
            
            return True, metrics

    def _prepare_batch(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        if "observation.state" in batch:
            state = batch["observation.state"]
            if state.dim() == 3:
                batch["observation.state"] = state.squeeze(1)

        if "action_is_pad" in batch:
            if isinstance(batch["action_is_pad"], np.ndarray):
                batch["action_is_pad"] = torch.from_numpy(batch["action_is_pad"])
        return batch

    def log(self, metrics: dict[str, float], start_time: Optional[float] = None) -> None:
        super().log(metrics, start_time)

    def save_checkpoint(self, global_step: int):
        save_path = os.path.join(self.project_dir, "checkpoints")
        if not os.path.exists(f"{save_path}/ckpt_{global_step}"):
            save_path = self.accelerator.save_state(f"{save_path}/ckpt_{global_step}")
            return save_path
        else:
            overwatch.warning(
                f"Checkpoint {global_step} already exists, skipping save."
            )
            return None

    @torch.no_grad()
    def inference(self, eval_model, batch) -> torch.Tensor:
        """Run inference to get predicted actions."""
        batch = self._prepare_batch(batch)
        
        # Use select_action for inference (handles temporal ensembling if enabled)
        eval_model.eval()
        
        # Get the underlying model
        model = eval_model.module if hasattr(eval_model, 'module') else eval_model
        
        # For batch inference, we need to call the model directly
        actions = model.predict_action(batch)
        
        return actions

    def evaluate(self):
        accelerator = self.accelerator
        global_step = self.global_step
        eval_model = self.unwrap_model()
        eval_model.eval()
        
        total_val_batches = (
            len(self.val_dataloader)
            if self.task_cfg.val_num_batches == -1
            else min(self.task_cfg.val_num_batches, len(self.val_dataloader))
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
        kld_loss_list = []

        for val_step, val_batch in enumerate(val_progress_bar):
            val_batch = batch_str_to_tensor(val_batch)
            
            gt_actions = val_batch["action"]  # (B, Tp, Da)
            
            # Tp -> predicted action horizon, Da -> action dim
            B, Tp, Da = gt_actions.shape

            with accelerator.autocast():
                # Compute validation loss
                val_batch = self._prepare_batch(val_batch)
                loss_dict = eval_model(val_batch)
                val_loss_list.append(accelerator.gather(loss_dict["loss"].detach()))
                
                if "kld_loss" in loss_dict:
                    kld_loss_list.append(loss_dict["kld_loss"])

                # action prediction loss
                pred_actions = self.inference(eval_model, val_batch)
                err_action_l1 = pred_actions - gt_actions  # (B, Tp, Da)
                
                # Handle padding mask
                if "action_is_pad" in val_batch:
                    mask = ~val_batch["action_is_pad"]
                    if isinstance(mask, np.ndarray):
                        mask = torch.from_numpy(mask).to(self.device)
                else:
                    mask = torch.ones(B, Tp, dtype=torch.bool, device=self.device)
                
                # Gather errors across processes
                err_action_l1_all = accelerator.gather(err_action_l1[:, :Tp].contiguous())
                err_action_masks_all = accelerator.gather(mask[:, :Tp].contiguous())
                
                err_action_l1 = err_action_l1_all[
                    err_action_masks_all.to(torch.bool) # type: ignore
                ].abs()
                action_l1_err_list.append(
                    err_action_l1.reshape(-1, Da).float().cpu().numpy()
                )  # (B*world_size*Ta, Da)

            if val_step + 1 >= total_val_batches:
                if accelerator.is_local_main_process:
                    val_progress_bar.close()
                if hasattr(self.val_dataloader, 'end'):
                    self.val_dataloader.end() # type: ignore
                break

        # Compute average metrics
        avg_val_loss = torch.stack(val_loss_list).mean().item()
        action_l1_err_list = np.concatenate(action_l1_err_list, axis=0)  # (N, Da)
        action_l1_err_list_denormed = (
            self.data_cfg.transform.field.denormalize_L1_action_err(
                action_l1_err_list
            )
        )

        # action L1 errors
        avg_action_errors_denormed = action_l1_err_list_denormed.mean(0)  # (Da,) NOTE only if the error is L1 (linear)
        # Define dimension splits: hand_joints(14) + arm_joints(14) + rpy(3) + height(1) = 32
        hand_joints_start, hand_joints_end = 0, 14
        arm_joints_start, arm_joints_end = 14, 28
        rpy_start, rpy_end = 28, 31
        height_start, height_end = 31, 32
        torso_vx_start, torso_vx_end = 32, 33
        torso_vy_start, torso_vy_end = 33, 34
        torso_vyaw_start, torso_vyaw_end = 34, 35
        torso_dyaw_start, torso_dyaw_end = 35, 36
    
        labels_denormed = [
            "val/denorm_err_l1_hand_joints",
            "val/denorm_err_l1_arm_joints",
            "val/denorm_err_l1_rpy",
            "val/denorm_err_l1_height",
            "val/denorm_err_l1_torso_vx",
            "val/denorm_err_l1_torso_vy",
            "val/denorm_err_l1_torso_vyaw",
            "val/denorm_err_l1_torso_target_yaw",
        ]
        
        avg_lr_action_err_denormed = np.split(
            avg_action_errors_denormed, [hand_joints_end, arm_joints_end, rpy_end, height_end, torso_vx_end, torso_vy_end, torso_vyaw_end, torso_dyaw_end], axis=-1
        )

        # log metrics
        accelerator.log(
            {
                "val/bc_loss": avg_val_loss,
                "val/kld_loss": np.mean(kld_loss_list),
                **dict(
                    zip(
                        labels_denormed, map(np.linalg.norm, avg_lr_action_err_denormed)
                    )
                ),
            },
            step=global_step + 1,
        )

    def resume_from_checkpoint(self):
        initial_global_step, load_path = super().resume_from_checkpoint()
        return initial_global_step, load_path

    def forward_and_loss(self, model, batch) -> dict[str, torch.Tensor]:
        """Compute forward pass and loss."""
        loss_dict = model(batch)
        return {"loss": torch.tensor(loss_dict["loss"])}

    def finalize(self) -> None:
        super().save_checkpoint(self.global_step)
        super().finalize()
        overwatch.info(f"Finalized ACT Trainer. Epoch losses: {shorten(self.epoch_loss)}")
