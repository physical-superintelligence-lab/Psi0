import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import Union, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset  # , IterableDataset
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from psi.config.config import TrainConfig
from diffusers.training_utils import EMAModel

from psi.config.config import LaunchConfig
from psi.config.data_lerobot import LerobotDataConfig #PushTDataConfig,
from psi.config.model_dp import DiffusionPolicyModelConfig
from dp.models.diffusion_policy import DiffusionPolicyModel #, ConditionalUnet1D, get_resnet, replace_bn_with_gn
from psi.trainers import Trainer

from psi.utils import flatten, shorten, initialize_overwatch,rmse, seed_everything
from psi.utils.utils import batch_str_to_tensor

overwatch = initialize_overwatch(__name__)

from accelerate import Accelerator
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class DiffusionPolicyG1Trainer(Trainer):
    epoch_loss: list[float]
    _loss_cpu: float

    def __init__(self, cfg: LaunchConfig, device: torch.device):
        super().__init__(cfg, device)
        overwatch.info("Initialized DP Trainer")

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
    def model_cfg(self) -> DiffusionPolicyModelConfig:
        return self.cfg.model  # type: ignore

    @property
    def data_cfg(self) -> LerobotDataConfig:
        return self.cfg.data  # type: ignore

    def init_models(self): 
        self.model = DiffusionPolicyModel(
            obs_horizon=self.model_cfg.obs_horizon,
            action_dim=self.model_cfg.action_dim,
            lowdim_obs_dim=self.model_cfg.obs_dim,
            pred_horizon=self.model_cfg.action_chunk_size,
            num_diffusion_iters=self.model_cfg.num_diffusion_iters,
        )

        # Exponential Moving Average
        # accelerates training and improves stability
        # holds a copy of the model weights
        self.ema = EMAModel(
            parameters=self.model.parameters(),
            power=0.75
        )
        self.ema.to(self.device)

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

        # create training and validation dataloaders
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
            # data normalized in dataset
            B = batch["agent_pos"].shape[0]
            nimage=batch["image"]
            nagent_pos = batch['agent_pos']
            naction = batch['action']
            # naction_pad = batch['action_is_pad']

            output = self.model(nimage, nagent_pos, naction)
            loss = output.loss

            # optimize
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
            # step lr scheduler every batch
            # this is different from standard pytorch behavior
            self.lr_scheduler.step()

            if overwatch.is_rank_zero() and self.accelerator.sync_gradients:
                # update Exponential Moving Average of the model weights
                mmodel_unwrapped = self.unwrap_model()
                self.ema.step(mmodel_unwrapped.parameters())

            # logging
            loss_cpu = loss.item()
            self._loss_cpu = loss_cpu
            return True, {"loss": loss_cpu}

    def log(self, metrics: dict[str, float], start_time: Optional[float] = None) -> None:
        super().log(metrics, start_time)

    def save_checkpoint(self, global_step: int) -> str:
        saved_path = super().save_checkpoint(global_step)

        if overwatch.is_rank_zero():
            ema_nets = self.unwrap_model()
            self.ema.copy_to(ema_nets.parameters())
            ema_nets_weights = ema_nets.state_dict()
            torch.save(ema_nets_weights, os.path.join(saved_path, "ema_net.pth"))

        return saved_path

    @torch.no_grad()
    def inference(self, eval_model, batch) -> torch.Tensor:
        nimage = batch["image"]  # (B, obs_horizon, C, H, W)
        nagent_pos = batch["agent_pos"]  # (B, obs_horizon, obs_dim)
        
        B = nimage.shape[0]
        device = nimage.device
        
        image_features = eval_model.vision_encoder(
            nimage.flatten(end_dim=1)  # (B*obs_horizon, C, H, W)
        )
        # (B, obs_horizon, 512)
        image_features = image_features.reshape(*nimage.shape[:2], -1)
        
        obs_features = torch.cat([image_features, nagent_pos], dim=-1)
        obs_cond = obs_features.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)
        
        pred_horizon = eval_model.pred_horizon
        action_dim = eval_model.action_dim
        noisy_action = torch.randn(
            (B, pred_horizon, action_dim), device=device
        )
        
        naction = noisy_action
        
        eval_model.noise_scheduler.set_timesteps(eval_model.num_diffusion_iters)
        
        for k in eval_model.noise_scheduler.timesteps:
            noise_pred = eval_model.noise_pred_net(
                sample=naction,
                timestep=k,
                global_cond=obs_cond
            )
            
            naction = eval_model.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample
        
        return naction  # (B, pred_horizon, action_dim)

    def evaluate(self):
        accelerator = self.accelerator
        global_step = self.global_step
        eval_model = self.unwrap_model()
        
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

        for val_step, val_batch in enumerate(val_progress_bar):
            val_batch = batch_str_to_tensor(val_batch)
            # # mask = val_batch["mask"]
            # mask = torch.ones_like(val_batch["action"])
            gt_actions = val_batch["action"]  # (B, Tp, Da)
            
            # Tp -> predicted action horizon, Da -> action dim
            B, Tp, Da = gt_actions.shape

            with accelerator.autocast():
                loss_dict = self.forward_and_loss(eval_model, val_batch)
                val_loss_list.append(accelerator.gather(loss_dict["loss"].detach()))

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
                err_action_l1_all = accelerator.gather(
                    err_action_l1[:, :Tp].contiguous()
                )
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

        avg_val_loss = torch.cat(val_loss_list).mean().item()
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

        if load_path is not None:
            self.ema = EMAModel(parameters=self.unwrap_model().parameters(),
                power=0.75)
            self.ema.load_state_dict(torch.load(os.path.join(load_path, "ema_net.pth")))

        return initial_global_step, load_path

    def forward_and_loss(self, model, batch) -> dict[str, torch.Tensor]:
        nimage = batch["image"]  # (B, obs_horizon, C, H, W)
        nagent_pos = batch["agent_pos"]  # (B, obs_horizon, obs_dim)
        naction = batch["action"]  # (B, pred_horizon, action_dim)
        # naction_pad = batch["action_is_pad"]  # (B, pred_horizon)

        loss_dict = model.forward(nimage, nagent_pos, naction)
        
        return {"loss": loss_dict["loss"]}

    def finalize(self) -> None:
        saved_path = super().save_checkpoint(self.global_step)

        if overwatch.is_rank_zero():
            ema_nets = self.unwrap_model()
            self.ema.copy_to(ema_nets.parameters())
            ema_nets_weights = ema_nets.state_dict()
            torch.save(ema_nets_weights, os.path.join(saved_path, "ema_net.pth"))
        
        super().finalize()
        overwatch.info(f"Finalized DP Trainer. Epoch losses: {shorten(self.epoch_loss)}")