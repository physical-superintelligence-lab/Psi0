from __future__ import annotations

from dotenv import load_dotenv
assert load_dotenv(), "Failed to load .env file. Make sure it exists and is properly formatted."
import os
import re
import torch
import gc
from tqdm import tqdm
import shutil
import json
import datetime
import sys
import tyro
import importlib
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from psi.config.config import LaunchConfig
from psi.utils import batch_str_to_tensor, seed_everything, initialize_overwatch, nice, flatten
from psi.trainers import Trainer

overwatch = initialize_overwatch(__name__)

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.data_loader import (
    DataLoaderStateMixin as AcceleratorDataLoaderStateMixin,
)
from accelerate.utils import ProjectConfiguration
import random, numpy as np

MAX_TRAINING_EPOCHS = 1_000_000

def _auto_tag_run(run_name: str):
    """commit the code and tag the run with the run name"""
    import subprocess

    subprocess.run(["git", "add", "."], check=False)
    subprocess.run(["git", "commit", "-m", f"Auto-commit for {run_name}"], check=False)
    subprocess.run(["git", "tag", run_name], check=False)
    # subprocess.run(["git", "push", "--tags"], check=True)

def _initialize_accelerator(trainer: Trainer) -> Accelerator:
    os.makedirs(trainer.project_dir, exist_ok=True)
    logging_dir = os.path.join(trainer.project_dir, trainer.cfg.log.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=trainer.project_dir, logging_dir=logging_dir
    )

    # setup FSDP training strategy if needed
    fsdp_plugin = None
    if trainer.cfg.train.data_parallel == "fsdp":
        fsdp_plugin = trainer.get_fsdp_plugin()
    
    deepspeed_plugin = None
    if trainer.cfg.train.data_parallel == "deepspeed":
        # SONGLIN: use accelerate launch config instead
        ds_config_path = trainer.cfg.train.deepspeed_config 
        deepspeed_plugin = DeepSpeedPlugin(zero_stage=3, hf_ds_config=ds_config_path)
    

    accelerator = Accelerator(
        gradient_accumulation_steps=trainer.cfg.train.gradient_accumulation_steps,
        mixed_precision=trainer.cfg.train.mixed_precision,
        # downcast_bf16=True,
        log_with=trainer.cfg.log.report_to,
        project_config=accelerator_project_config,
        fsdp_plugin=fsdp_plugin,
        deepspeed_plugin=deepspeed_plugin
    )

    if trainer.cfg.train.data_parallel == "deepspeed":
        # When using DeepSpeed, accelerate.prepare() requires you 
        # to pass at least one of training or evaluation dataloaders 
        # with batch_size attribute returning an integer value or alternatively 
        # set an integer value in train_micro_batch_size_per_gpu 
        # in the deepspeed config file or assign integer value to
        accelerator.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = trainer.device_train_batch_size # type: ignore

    # Initialize the trackers unless in eval or debug mode
    if overwatch.is_rank_zero() and not trainer.cfg.eval:
        if trainer.cfg.auto_tag_run:
            _auto_tag_run(trainer.run_name)

        if trainer.cfg.log.report_to == "wandb":
            tracker_config = (
                trainer.cfg.model_dump()
            )  # cfg.model_dump() # dict(vars(cfg.task))
            
            # Add environment variables to tracker config for wandb logging
            if "_ENV_INFO_JSON" in os.environ:
                env_info = json.loads(os.environ["_ENV_INFO_JSON"])
                tracker_config["environment_variables"] = env_info
            
            wandb_config = dict(
                trainer.cfg.wandb
            )  # cfg.wandb.model_dump() #vars(copy.deepcopy(cfg.wandb))
            wandb_config["name"] = trainer.run_name
            wandb_config["dir"] = os.path.abspath(
                os.path.expanduser(trainer.project_dir)
            )
            wandb_config["group"] = trainer.cfg.train.name
            project = wandb_config.pop("project", "default")

            # wandb resume with old hisotry
            if trainer.cfg.train.resume_from_checkpoint is not None and trainer.cfg.wandb.resume != "never":
                run_id = wandb_config.get("id") or getattr(trainer.cfg.wandb, "id", None)
                if not run_id :
                    run_dir = os.path.join(*trainer.cfg.train.resume_from_checkpoint.split("/")[:3])
                    if os.path.exists(os.path.join(run_dir, "run_config.json")):
                        with open(os.path.join(run_dir, "run_config.json")) as f:
                            run_id = (json.load(f).get("wandb") or {}).get("id")
                            overwatch.info(f"resume wandb with run id: {run_id}")
                        wandb_config["id"] = run_id
                        
            accelerator.init_trackers(project, tracker_config, {"wandb": wandb_config})
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    trainer.cfg.wandb.id = tracker.run.id
                    trainer.cfg.wandb.name = tracker.run.name
                    trainer.cfg.wandb.entity = tracker.run.entity
                    trainer.cfg.wandb.group = tracker.run.group

        # Log the configuration
        os.makedirs(trainer.project_dir, exist_ok=True)
        # draccus.dump(cfg, open(f'{project_dir}/run_config.yaml','w'))
        with open(f"{trainer.project_dir}/run_config.json", "w") as f:
             f.write(trainer.cfg.model_dump_json(indent=4))
        overwatch.info(f"Saved configuration to {trainer.project_dir}")

        # Write down the sys.argv to argv.txt
        argv_path = os.path.join(trainer.project_dir, "argv.txt")
        with open(argv_path, "w") as f:
            i = 0
            while i < len(sys.argv):
                arg = sys.argv[i]
                if arg.startswith('--'):
                    # Start of a key, check for list values
                    line = [arg]
                    j = i + 1
                    # Collect all following args that do not start with '--'
                    while j < len(sys.argv) and not sys.argv[j].startswith('--'):
                        line.append(sys.argv[j])
                        j += 1
                    f.write(' '.join(line) + '\n')
                    i = j
                else:
                    f.write(arg + '\n')
                    i += 1

        # write system envs to envs.txt
        if "_ENV_INFO_JSON" in os.environ:
            env_info = json.loads(os.environ["_ENV_INFO_JSON"])
            envs_path = os.path.join(trainer.project_dir, "envs.txt")
            with open(envs_path, "w") as f:
                for k, v in env_info.items():
                    f.write(f"{k}={v}\n")

        # copy dataset statistics file if exists
        if hasattr(trainer.cfg.data.transform.field, "stat_path"):
            from psi.utils import resolve_path
            stat_path = resolve_path(trainer.cfg.data.transform.field.stat_path) # type: ignore
            if os.path.exists(stat_path):
                dst_stat_path = os.path.join(trainer.project_dir, "dataset_statistics.json")
                shutil.copy2(stat_path, dst_stat_path)
    trainer.accelerator = accelerator
    return accelerator


def train(config: LaunchConfig):
    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()

    if config.seed:
        overwatch.info(f"Seed everything with {config.seed}")
        seed_everything(config.seed)
        
    trainer = Trainer.instantiate(config, device_id)
    overwatch.info("Initialize models ... ")
    trainer.init_models()
    accelerator = _initialize_accelerator(trainer)
    overwatch.info(f"Training configurations:")
    overwatch.info(f"training task: '{config.train.name}'", ctx_level=1)
    overwatch.info(f"run name: {trainer.run_name}", ctx_level=1)
    overwatch.info(f"seed: {config.seed}", ctx_level=1)
    overwatch.info(f"mixed precision: {trainer.dtype}", ctx_level=1)
    overwatch.info(f"warmup steps: {trainer.num_warmup_steps}", ctx_level=1)
    overwatch.info(f"validation steps: {config.train.validation_steps}", ctx_level=1)
    overwatch.info(f"checkpoint steps: {config.train.checkpointing_steps}", ctx_level=1)
    overwatch.info(f"max gradient norm: {config.train.max_grad_norm}", ctx_level=1)

    train_dataset, val_dataset = trainer.create_datasets()
    overwatch.info(f"Num training samples:", ctx_level=1)
    overwatch.info(f"Training dataset size: {trainer.len_train_dataset:,}", ctx_level=2)  # type: ignore
    if val_dataset:
        overwatch.info(f"Val dataset size: {trainer.len_val_dataset:,}", ctx_level=2)  # type: ignore
    trainer.create_dataloaders(train_dataset, val_dataset)
    
    overwatch.info("Initialize optimizers and schedulers...")
    trainer.create_optimizer_and_scheduler(trainer.max_training_steps)

    # fmt: off
    overwatch.info("***** Running training *****")
    overwatch.info(f"Num training examples = {trainer.len_train_dataset}", ctx_level=1)
    overwatch.info(f"Max training Epochs = {trainer.max_training_epochs}", ctx_level=1)
    overwatch.info(f"Total optimization steps = {trainer.max_training_steps}", ctx_level=2)
    overwatch.info(f"Num steps Per Epoch = {trainer.num_steps_per_epoch}", ctx_level=2)
    overwatch.info(f"Effective training epochs = {trainer.max_training_steps/trainer.num_steps_per_epoch*trainer.world_size:.2f}", ctx_level=2)
    overwatch.info(f"Global train batch size (w. parallel, distributed & accumulation) = {trainer.global_train_batch_size}", ctx_level=1)
    overwatch.info(f"Device train batch size = {trainer.device_train_batch_size}", ctx_level=2)
    overwatch.info(f"Gradient Accumulation steps = {trainer.gradient_accumulation_steps}", ctx_level=2)
    overwatch.info(f"Num processes (GPUs) = {trainer.world_size}", ctx_level=2)
    # fmt: on

    trainer.prepare(accelerator)
    global_step = initial_global_step = trainer.resume_from_checkpoint()[0]
    epoch_start = global_step // trainer.num_steps_per_epoch

    overwatch.info(f"Accelerator runs in: {trainer.project_dir}")
    trainer.set_train()
    
    progress_bar = tqdm(
        range(0, trainer.max_training_steps),
        initial=initial_global_step,
        desc="Training steps",
        disable=not overwatch.is_rank_zero(),
        position=0,
    )

    is_max_train_steps_reached = (initial_global_step >= trainer.max_training_steps)
    # skip_resumed_steps = False
    skip = 0

    for epoch in range(epoch_start, MAX_TRAINING_EPOCHS):
        trainer.next_epoch(epoch)
        accelerator.wait_for_everyone()
        for local_step, batch in enumerate(trainer.train_dataloader):
            if (
                config.train.skip_resumed_steps
                and skip < initial_global_step % trainer.num_steps_per_epoch
            ):
                # skip to inital global step
                skip += 1
                continue
            sync_gradients, losses = trainer.step(batch_str_to_tensor(batch), global_step, local_step)
            
            if sync_gradients:
                # log metrics
                trainer.log(flatten({**losses, "epoch": epoch}, parent_key="train")) 

                # save checkpoints
                if (global_step + 1) % config.train.checkpointing_steps == 0 or global_step == trainer.max_training_steps - 1:
                    accelerator.wait_for_everyone() # ensures fsdp checkpointing without deadlock
                    save_path = trainer.save_checkpoint(global_step+1)
                    accelerator.wait_for_everyone()
                    
                    if overwatch.is_rank_zero():
                        tqdm.write(f"Saved state to {save_path}")

                # validation
                if config.train.validation_steps > 0 and (
                        global_step % config.train.validation_steps == 0 or 
                        global_step == trainer.max_training_steps - 1
                    ) :
                    gc.collect()
                    torch.cuda.empty_cache()
                    trainer.set_eval()
                    with torch.no_grad(), torch.autocast(
                        device_type="cuda", dtype=trainer.dtype
                    ):
                        eval_losses = trainer.evaluate()
                        if eval_losses is not None: # FIXME
                            trainer.log(flatten(eval_losses, parent_key="eval")) 
                    trainer.set_train()
                    """ NOTE:
                        This is a workaround for iterating over val_dataloader in the middle of training_dataloader.
                        When val dataloader dost not end explicitly, eg., the val dataloader is wrapped in a tqdm progress bar,
                        and the bar is closed before the end of the dataloader, it will cause an inconsistent accelerate gradient state
                    """
                    if hasattr(trainer, "val_dataloader") and isinstance(
                        trainer.val_dataloader, AcceleratorDataLoaderStateMixin
                    ):
                        if (
                            accelerator.gradient_state.active_dataloader
                            != trainer.train_dataloader
                        ):
                            try:
                                trainer.val_dataloader.end()
                                overwatch.warning_once(
                                    "Warning only Once: "
                                    "Val dataloader does not end after evaluation. I killed it for you."
                                )
                            except ValueError:
                                overwatch.warning_once("Even I can not save you.")

                progress_bar.update()
                progress_bar.set_postfix(dict(loss=losses["loss"], lr=nice(trainer.lr)))
                global_step += 1

            if global_step >= trainer.max_training_steps:
                if overwatch.is_rank_zero():
                    tqdm.write("Training has reached maximum steps.")
                is_max_train_steps_reached = True  # set to break outer loop
                break

        if is_max_train_steps_reached:
            break

    accelerator.wait_for_everyone()
    trainer.finalize()
    overwatch.info("Happy Ending!")
    accelerator.end_training()
    
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def _is_slurm_job_process() -> bool:
    return "SLURM_JOB_ID" in os.environ and not os.isatty(sys.stdout.fileno())

if __name__ == "__main__":
    if _is_slurm_job_process():
        overwatch.info("SLURM job detected. Setting up distributed environment variables...")
        os.environ["LOCAL_WORLD_SIZE"] = str(
            int(os.environ["WORLD_SIZE"]) // int(os.environ["SLURM_JOB_NUM_NODES"])
        )

    # Print and log important environment variables
    env_vars_to_track = [
        "OMP_NUM_THREADS", "HF_HOME", "TORCH_HOME", "HF_TOKEN", "HF_LEROBOT_HOME",
        "DATA_HOME", "UV_CACHE_DIR", "WANDB_API_KEY",
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "CUDA_VISIBLE_DEVICES",
        "WORLD_SIZE", "LOCAL_WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"
    ]

    env_info = {}
    overwatch.info("=== Environment Variables ===")
    for var in env_vars_to_track:
        value = os.environ.get(var, "Not Set")
        # Mask sensitive tokens for logging
        if "TOKEN" in var or "KEY" in var:
            display_value = f"{value[:3]}...{value[-4:]}" if value != "Not Set" and len(value) > 7 else value
        else:
            display_value = value
        
        env_info[var] = display_value
        overwatch.info(f"{var}: {display_value}")
    
    # Store env info for wandb logging later
    os.environ["_ENV_INFO_JSON"] = json.dumps(env_info)

    start = datetime.datetime.now()
    overwatch.info("Parsing configuration...")
    try:
        # By convention, the first argument after the script name is the config module name, 
        # eg., {trainer}_{data}_{model}_config, which corresponds to psi.config.train.pretrain_egodex_qwen3vl_config
        module = importlib.import_module(f"psi.config.train.{sys.argv[1]}")
        DynamicLaunchConfigClass = getattr(module, "DynamicLaunchConfig")
        config = tyro.cli(DynamicLaunchConfigClass, config=(tyro.conf.ConsolidateSubcommandArgs,), args=sys.argv[2:])
    except Exception as e:
        overwatch.error(f"Failed to import config module 'psi.config.train.{sys.argv[1]}'")
        raise e

    end = datetime.datetime.now()
    overwatch.info(f"Config parsing took {(end - start).total_seconds():.2f}s") 
    train(config) # type: ignore
