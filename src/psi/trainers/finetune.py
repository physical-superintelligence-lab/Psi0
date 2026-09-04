from __future__ import annotations
import os
import json
import math
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
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, Qwen3VLProcessor
from psi.trainers.qwen3vl_mixin import PaddedCollatorForTogether

if TYPE_CHECKING:    
    from psi.config.config import TrainConfig
    from psi.config.model_psi0 import Psi0ModelConfig
    from psi.config.transform import Psi0ModelTransform

from accelerate import Accelerator

# from utils.overwatch import initialize_overwatch
from psi.utils import (
    initialize_overwatch,
    shorten
)
from psi.utils.utils import batch_str_to_tensor

overwatch = initialize_overwatch(__name__)

# from .base import Trainer
from .trainer import Trainer, worker_init_fn

from psi.utils import flatten, shorten, move_to_device, rmse, seed_everything
from psi.models.psi0 import Psi0Model

class FinetuneTrainer(Trainer):

    def __init__(self, cfg, device: torch.device):
        super().__init__(cfg, device)

        self._skipped_steps = 0
        self._grad_norms: dict[str, float] = {}
        self._grad_norm_act: float | None = None

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

        # Frozen text encoder that produces the `pooled_projections` global-conditioning
        # vector consumed by combined_temb (SD3-style). Kept OUT of self.model on purpose:
        # frozen, not DDP-wrapped, not in checkpoints, not in the optimizer.
        self.pooled_text_encoder = None
        self.pooled_text_tokenizer = None
        self._pooled_cache: dict[str, torch.Tensor] = {}  # instruction -> pooled emb (frozen => cacheable)
        if self.model_cfg.pooled_text_encoder == "clip":
            from transformers import CLIPTextModelWithProjection, CLIPTokenizer
            path = self.model_cfg.pooled_text_encoder_path
            self.pooled_text_tokenizer = CLIPTokenizer.from_pretrained(path)
            clip = CLIPTextModelWithProjection.from_pretrained(path, torch_dtype=torch.bfloat16)
            clip.requires_grad_(False)
            clip.eval()
            self.pooled_text_encoder = clip.to(self.device)
            assert clip.config.projection_dim == self.model_cfg.pooled_projection_dim, (
                f"pooled_projection_dim={self.model_cfg.pooled_projection_dim} but CLIP "
                f"projection_dim={clip.config.projection_dim}; set --model.pooled-projection-dim accordingly"
            )
            overwatch.info(
                f"Loaded FROZEN CLIP pooled text encoder from {path} "
                f"(projection_dim={clip.config.projection_dim})"
            )
            # Pre-compute the pooled embeddings once, up front, so the train/eval loop
            # never invokes CLIP: (1) load a persisted cache if configured, (2) warm the
            # cache for every task instruction in the packs. Any runtime miss still falls
            # back to lazy compute in compute_pooled_projections.
            #
            # pooled_cache_path is a RELATIVE filename -> stored in project_dir alongside
            # run_config.json (absolute paths, e.g. a shared offline cache, are used as-is).
            # accelerator isn't built yet here, so gate the write on global RANK.
            cache_path = self.model_cfg.pooled_cache_path
            if cache_path and not os.path.isabs(cache_path):
                cache_path = os.path.join(self.project_dir, cache_path)
            if cache_path and os.path.exists(cache_path):
                try:
                    self.load_pooled_cache(cache_path)
                except Exception as e:  # noqa: BLE001
                    overwatch.warning(
                        f"pooled cache {cache_path} unreadable ({type(e).__name__}: {e}); "
                        "recomputing from tasks"
                    )
                    self._pooled_cache.clear()
            self._warm_pooled_cache_from_tasks()
            if cache_path and int(os.environ.get("RANK", "0")) == 0:
                self.save_pooled_cache(cache_path)
        elif self.model_cfg.combined_temb:
            overwatch.warning(
                "combined_temb=True but pooled_text_encoder is None: the action header needs a "
                "pooled_projections vector and will crash. Set --model.pooled-text-encoder=clip."
            )
        # assert self.task_cfg.mixed_precision == "no", "other options not tested"
        # assert self.model_cfg.n_conditions == len(self.data_cfg.transform.repack.conditions), "inconsistent confs" # type: ignore

    @property
    def task_cfg(self) -> TrainConfig:
        return self.cfg.train

    @property
    def model_cfg(self) -> Psi0ModelConfig:
        return self.cfg.model # type: ignore
    
    @torch.no_grad()
    def precompute_pooled_cache(self, instructions, batch_size: int = 128) -> int:
        """Warm the pooled-embedding cache for a known instruction set (batched), so the
        train/eval loop never calls CLIP. Returns the number of newly encoded strings.
        No-op without a pooled text encoder."""
        if self.pooled_text_encoder is None:
            return 0
        uniq = [s for s in dict.fromkeys(instructions) if s and s not in self._pooled_cache]
        # Warming can take a while on a big pack (hundreds of task strings), and it
        # runs before the training bar exists, so show progress. Rank 0 only --
        # eight interleaved bars are unreadable.
        batches = range(0, len(uniq), batch_size)
        for i in tqdm(
            batches,
            total=(len(uniq) + batch_size - 1) // batch_size,
            desc="Precomputing CLIP pooled text cache",
            unit="batch",
            disable=not overwatch.is_rank_zero() or not uniq,
            leave=False,
        ):
            chunk = uniq[i:i + batch_size]
            toks = self.pooled_text_tokenizer(
                chunk, padding=True, truncation=True,
                max_length=self.pooled_text_tokenizer.model_max_length, return_tensors="pt",
            ).to(self.pooled_text_encoder.device)
            embs = self.pooled_text_encoder(**toks).text_embeds
            for s, e in zip(chunk, embs):
                self._pooled_cache[s] = e.clone()
        return len(uniq)

    def _warm_pooled_cache_from_tasks(self) -> None:
        """Best-effort precompute from each pack's meta/tasks.jsonl (lowercased to match
        the transform's `instruction`). Misses are still handled lazily at runtime."""
        instrs: list[str] = []
        repos = list(dict.fromkeys(
            list(self.data_cfg.train_repo_ids) + list(self.data_cfg.val_repo_ids)
        ))
        for repo in repos:
            tasks_path = os.path.join(self.data_cfg.root_dir, repo, "meta", "tasks.jsonl")
            try:
                with open(tasks_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            instrs.append(json.loads(line)["task"].lower())
            except Exception as e:  # noqa: BLE001 - warming is best-effort
                overwatch.warning(f"pooled cache warm: could not read {tasks_path}: {e}")
        if instrs:
            n = self.precompute_pooled_cache(instrs)
            overwatch.info(
                f"Precomputed pooled cache: +{n} new, {len(self._pooled_cache)} total "
                f"from {len(set(instrs))} task instructions"
            )

    def save_pooled_cache(self, path: str) -> None:
        if not self._pooled_cache:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Write-then-rename
        tmp = f"{path}.tmp.{os.getpid()}"
        torch.save({k: v.detach().cpu() for k, v in self._pooled_cache.items()}, tmp)
        os.replace(tmp, path)
        overwatch.info(f"Saved pooled cache ({len(self._pooled_cache)} entries) -> {path}")

    def load_pooled_cache(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        dev = self.pooled_text_encoder.device if self.pooled_text_encoder is not None else self.device
        for k, v in data.items():
            self._pooled_cache[k] = v.to(dev)
        overwatch.info(f"Loaded pooled cache ({len(data)} entries) <- {path}")

    @torch.no_grad()
    def compute_pooled_projections(self, batch) -> Optional[torch.Tensor]:
        """Frozen-CLIP pooled instruction embedding (B, pooled_projection_dim) for
        combined_temb. Returns None when no pooled text encoder is configured, so
        non-combined_temb runs are unaffected. `batch["instruction"]` is a list[str]
        (batch_str_to_tensor already decoded instruction_str -> instruction).

        The encoder is frozen and deterministic, so results are cached per unique
        instruction string (~50 tasks here). Only cache-missing instructions hit CLIP;
        in steady state the whole batch is a lookup. Each entry is ~1.5 KB, so the
        cache is negligible. Cache is per-process (each DDP rank has its own)."""
        if self.pooled_text_encoder is None:
            return None
        instructions = batch.get("instruction")
        if instructions is None:
            return None
        if isinstance(instructions, str):
            instructions = [instructions]
        instructions = list(instructions)

        # encode only instructions not already cached (dedup, preserve first-seen order)
        missing = [s for s in dict.fromkeys(instructions) if s not in self._pooled_cache]
        if missing:
            toks = self.pooled_text_tokenizer(
                missing,
                padding=True,
                truncation=True,
                max_length=self.pooled_text_tokenizer.model_max_length,
                return_tensors="pt",
            ).to(self.pooled_text_encoder.device)
            embs = self.pooled_text_encoder(**toks).text_embeds  # (len(missing), D)
            for s, e in zip(missing, embs):
                self._pooled_cache[s] = e.clone()
        return torch.stack([self._pooled_cache[s] for s in instructions], dim=0)  # (B, D)

    def vlm_param_dtype(self) -> torch.dtype:
        """Weight dtype for the VLM: fp32 master weights whenever it is tuned.

        Inferred from tune_vlm / tune_mm_* -- a tuned component always gets fp32.
        Except deepspeed, whose ZeRO optimizer already keeps its own fp32 master copy of bf16 params.
        """
        if not self.vlm_trainable_components():
            return torch.bfloat16
        if self.train_cfg.data_parallel == "deepspeed":
            return torch.bfloat16
        return torch.float32

    def init_qwen3vl_models(self):
        from huggingface_hub.errors import HFValidationError
        dtype = self.vlm_param_dtype()
        # flash_attention_2 with fp32 weights only logs a warning: its kernels see
        # bf16 tensors anyway because the projections run under autocast.
        overwatch.info(f"Loading VLM weights in {dtype}")
        try:
            vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_cfg.model_name_or_path,
                attn_implementation="flash_attention_2",
                dtype=dtype
            )
        except HFValidationError as e:
            overwatch.error(f"Failed to load pretrained VLM model from {self.model_cfg.model_name_or_path}")
            raise e
        
        overwatch.info(f"Load pretrained VLM model from {self.model_cfg.model_name_or_path}")
        self.vlm_processor = AutoProcessor.from_pretrained(self.model_cfg.model_name_or_path)
        self.tokenizer = self.vlm_processor.tokenizer

        assert self.cfg.train.lora == False

        return vlm_model

    def _reconcile_action_header_state_dict(self, state_dict):
        """Reconcile a pretrained action-header checkpoint with the current header.

        With layerwise VLM fusion every block is `context_pre_only`, so `norm1_obs`
        is an AdaLayerNormContinuous (dim -> 2*dim) rather than the AdaLayerNormZero
        (dim -> 6*dim) that a non-layerwise checkpoint carries. Warm-start the
        continuous layer from the msa half of the zero layer and skip anything else
        whose shape still does not line up, so `load_state_dict` cannot raise --
        `strict=False` suppresses missing/unexpected keys but not size mismatches.
        """
        model_sd = self.model.action_header.state_dict()
        filtered, notes = {}, []
        for k, v in state_dict.items():
            if k not in model_sd:
                continue  # strict=False drops it anyway
            want = model_sd[k].shape
            if v.shape == want:
                filtered[k] = v
            elif (k.endswith("norm1_obs.linear.weight") or k.endswith("norm1_obs.linear.bias")) \
                    and v.shape[0] == 3 * want[0] and v.shape[1:] == want[1:]:
                # zero:       shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
                # continuous: scale, shift
                d = want[0] // 2
                filtered[k] = torch.cat([v[d:2 * d], v[:d]], dim=0)
                notes.append(f"{k}: adaLN-zero -> adaLN-continuous (msa half)")
            else:
                notes.append(f"{k}: {tuple(v.shape)} != {tuple(want)}, skipped")
        if notes:
            overwatch.warning(
                f"[action header] {len(notes)} key(s) not loaded as-is:\n  " + "\n  ".join(notes)
            )
        return filtered

    def init_models(self):
        # init pretrained vlm
        vlm_model = self.init_qwen3vl_models()

        self.model = Psi0Model(
            model_cfg=self.model_cfg,
            vlm_model=vlm_model,
        )

        # TODO refac
        if self.model_cfg.pretrained_action_header_path is not None:
            # load pretrained action header ckpt
            from safetensors.torch import load_file
            ckpt_path = self.model_cfg.pretrained_action_header_path
            state_dict = load_file(f"{ckpt_path}/action_header.safetensors")
            if (state_dict["action_proj_in.dec_pos"].shape[0] != self.model_cfg.action_chunk_size or
                state_dict["action_proj_out.linear.weight"].shape[0] != self.model_cfg.action_dim):
                # only load transformer blocks
                reduced_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("transformer_blocks"):
                        reduced_state_dict[k] = v
                overwatch.info(f"Loading pretrained action header from {ckpt_path}")
                self.model.action_header.load_state_dict(
                    self._reconcile_action_header_state_dict(reduced_state_dict), strict=False
                )
                overwatch.warning("action header size mismatch, only loaded transformer blocks.")
            else:
                self.model.action_header.load_state_dict(
                    self._reconcile_action_header_state_dict(state_dict), strict=False
                )
            overwatch.info("loaded pretrained action header successfully.")
        else:
            overwatch.info("No pretrained action header specified, training from scratch.")
        
        if self.train_cfg.data_parallel == "deepspeed":
            # HACK to set correct config for deepspeed
            self.model.config = vlm_model.config
            self.model.config.hidden_size = vlm_model.config.text_config.hidden_size

        self.apply_vlm_trainability()

        num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        overwatch.info(f"Model has {num_parameters:,} trainable parameters")

    def apply_vlm_trainability(self) -> set[str]:
        """Freeze/unfreeze the VLM per component; returns the tuned components."""
        trainable_components = self.vlm_trainable_components()
        for name, p in self.model.vlm_model.named_parameters():
            p.requires_grad = self._vlm_group_of(name) in trainable_components

        if not trainable_components:
            overwatch.info("VLM parameters are frozen (tune_vlm=False, no tune_mm_* set)")
        else:
            overwatch.info(f"VLM trainable components: {sorted(trainable_components)}")
            # Freeze the final norm layer since we don't use it (we extract hidden_states[-1])
            if hasattr(self.model.vlm_model.model, 'language_model') and hasattr(self.model.vlm_model.model.language_model, 'norm'):
                for p in self.model.vlm_model.model.language_model.norm.parameters():
                    p.requires_grad = False
                overwatch.info("VLM final norm layer frozen (not used in forward pass)")
            # We never read the logits, so an UNTIED lm_head receives no gradient
            # and DDP errors on the reduction that never comes.
            lm_head = self.model.vlm_model.lm_head
            if lm_head.weight is not self.model.vlm_model.get_input_embeddings().weight:
                lm_head.weight.requires_grad = False
                overwatch.info("Untied VLM lm_head frozen (action loss does not use logits)")

        if trainable_components and self.model_cfg.gradient_checkpointing:
            self.model.vlm_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.model.vlm_model.enable_input_require_grads()
            self.model.vlm_model.config.use_cache = False
            if hasattr(self.model.vlm_model.config, "text_config"):
                self.model.vlm_model.config.text_config.use_cache = False
            overwatch.info("VLM gradient checkpointing enabled; KV cache disabled")
        return trainable_components


    @staticmethod
    def _vlm_group_of(name: str) -> str:
        """Which VLM sub-group a `vlm_model.*` parameter belongs to.

        Follows the naming convention already used by `create_qwen3vl_optimizer`
        in qwen3vl_mixin.py: the patch merger(s) are the multimodal projector,
        everything else under `visual` is the vision tower.
        """
        if "merger" in name:
            return "mm_projector"
        if "visual" in name:
            return "vision_tower"
        return "lang_backbone"

    def vlm_trainable_components(self) -> set[str]:
        """Which VLM components to finetune.

        `tune_mm_vision` / `tune_mm_mlp` / `tune_mm_llm` select components
        individually, and selecting any of them implies tuning the VLM -- so
        `--model.tune-mm-vision` alone finetunes only the vision tower, at
        `vision_tower_lr`. When none of them is set, `tune_vlm` keeps its legacy
        meaning of "train the whole VLM".
        """
        selected = set()
        if self.model_cfg.tune_mm_vision:
            selected.add("vision_tower")
        if self.model_cfg.tune_mm_mlp:
            selected.add("mm_projector")
        if self.model_cfg.tune_mm_llm:
            selected.add("lang_backbone")
        if selected:
            if not self.model_cfg.tune_vlm:
                overwatch.info(
                    f"tune_mm_* selected {sorted(selected)}; treating the VLM as tuned "
                    "even though tune_vlm=False"
                )
            return selected
        if self.model_cfg.tune_vlm:
            return {"vision_tower", "mm_projector", "lang_backbone"}
        return set()

    def create_optimizers(self):
        # Separate parameters into groups.  Frozen params are skipped entirely:
        # AdamW would silently no-op on them (grad is None), but DeepSpeed/ZeRO
        # does not tolerate frozen params inside an optimizer group, and they
        # distort the per-group parameter counts logged below.
        grouped: dict[str, list[torch.nn.Parameter]] = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("action_header."):
                group = "action_header"
            elif name.startswith("vlm_model."):
                # requires_grad above is the single source of truth: init_models
                # already applied tune_vlm / tune_mm_* per component, so a
                # vision-only run simply yields no lang_backbone group.
                group = self._vlm_group_of(name)
            else:
                group = "other"
            grouped.setdefault(group, []).append(param)

        base_lr = self.cfg.train.learning_rate
        lr_by_group = {
            "action_header": base_lr,
            "other": base_lr,  # Default group
            "lang_backbone": self.model_cfg.lang_backbone_lr,
            "vision_tower": self.model_cfg.vision_tower_lr,
            "mm_projector": self.model_cfg.mm_projector_lr,
        }
        # Fixed order so the optimizer state_dict layout is deterministic across
        # runs/resumes, and group 0 stays the action header (see get_lr()).
        group_order = [
            "action_header", "other",
            "lang_backbone", "vision_tower", "mm_projector",
        ]

        param_groups = []
        for group_name in group_order:
            params = grouped.get(group_name)
            if not params:
                continue
            param_groups.append({
                "params": params,
                "lr": lr_by_group[group_name],
                "group_name": group_name,
            })
            overwatch.info(
                f"optimizer group '{group_name}': lr={lr_by_group[group_name]:.2e}, "
                f"{sum(p.numel() for p in params):,} trainable params"
            )

        optimizer_kwargs = dict(self.cfg.train.lr_scheduler_kwargs)
        if self.cfg.train.optimizer_foreach is not None:
            optimizer_kwargs["foreach"] = self.cfg.train.optimizer_foreach
        # AdamW's *default* lr matters even though every group sets its own:
        # transformers' `cosine_with_min_lr` computes the decay floor as
        # min_lr / optimizer.defaults["lr"] (optimization.py:373). Without this
        # it silently used torch's 1e-3 default, so `min_lr` was off by
        # learning_rate/1e-3 for every group.
        optimizer_kwargs.setdefault("lr", base_lr)
        self.optimizer = torch.optim.AdamW(
            param_groups,
            **optimizer_kwargs, # type: ignore
        )
        return self.optimizer

    def create_lr_schedulers(
        self,
        num_training_steps: int | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ):
        from transformers.optimization import get_scheduler
        assert num_training_steps is not None
        opt = optimizer if optimizer is not None else self.optimizer
        min_lr = self.cfg.train.scheduler_specific_kwargs.get("min_lr")
        if min_lr is not None and len(opt.param_groups) > 1 and "min_lr" in self.cfg.train.lr_scheduler_type:
            # The schedule is one shared multiplicative factor, so an absolute
            # min_lr can only be honored for the group whose lr equals the
            # optimizer default; the other groups floor at the same *ratio*.
            overwatch.warning(
                f"min_lr={min_lr} is interpreted as min_lr/{self.cfg.train.learning_rate:.2e} "
                "of each group's own LR; non-default groups floor proportionally, "
                "not at the absolute value."
            )
        self.lr_scheduler = get_scheduler(
            self.cfg.train.lr_scheduler_type,
            num_training_steps=num_training_steps * self.world_size,
            optimizer=opt,
            num_warmup_steps=self.num_warmup_steps * self.world_size,
            scheduler_specific_kwargs=self.cfg.train.scheduler_specific_kwargs,
        )

    def create_optimizer_and_scheduler(self, num_training_steps: int | None = None):
        optimizer = self.create_optimizers()
        assert optimizer is not None, "create_optimizers must return the optimizer"
        self.create_lr_schedulers(
            num_training_steps=num_training_steps, optimizer=optimizer
        )

    def group_lrs(self) -> dict[str, float]:
        """Current LR of every optimizer group, keyed `lr_<group_name>`.

        `self.lr` only reports param_groups[0] (the action header), so without
        this a misconfigured VLM LR is invisible in wandb. Read straight off the
        optimizer since that is what AdamW actually steps with.
        """
        optimizer = getattr(self, "optimizer", None)
        param_groups = getattr(optimizer, "param_groups", None)
        if not param_groups:
            return {}
        return {
            f"lr_{group.get('group_name', i)}": float(group["lr"])
            for i, group in enumerate(param_groups)
        }

    def create_datasets(self):  # TODO use parent impl.
        transform_kwargs=dict(
            vlm_processor=self.vlm_processor,
        )
        val_data_cfg = self.data_cfg.model_copy(deep=True)
        self.train_dataset = self.data_cfg(split="train", transform_kwargs=transform_kwargs)
        self.val_dataset = val_data_cfg(split="val", transform_kwargs=transform_kwargs)
        return self.train_dataset, self.val_dataset

    def create_dataloaders(self, train_dataset, val_dataset):
        g = torch.Generator()
        g.manual_seed(self.cfg.seed or 42)
        is_psix_dataset_enabled = hasattr(self.data_cfg, "label") and self.data_cfg.label == "psix"

        cfg_num_workers = getattr(self.data_cfg, "num_workers", None)
        num_workers = cfg_num_workers if cfg_num_workers is not None else 12

        # prefetch_factor: batches buffered per worker. 2 is too shallow to ride
        # out a multi-step produce spike under CPU contention (buffer drains before
        # the lagging worker recovers); 4 gives ~2x the cushion for the round-robin.
        cfg_prefetch = getattr(self.data_cfg, "prefetch_factor", None)
        prefetch_factor = cfg_prefetch if cfg_prefetch is not None else 4

        train_dataloader_kwargs = {
            "num_workers": num_workers,
            "drop_last": True,
            # "shuffle": True,
            # "generator": g,
            # thread-capping worker_init_fn applied to BOTH paths below
            "persistent_workers": True,
            "pin_memory": True,
            "prefetch_factor": prefetch_factor,
            "worker_init_fn": worker_init_fn,
        }
        val_dataloader_kwargs = {
            "num_workers": 1,
            "drop_last": False,
            # "pin_memory": True,
            "persistent_workers": True,
            # "shuffle": True,
        }  # 1 small enough to fit im Mem. 2. no need distributed sampler

        if not is_psix_dataset_enabled:
            # for non-mixture dataset, we can use more workers to speed up data loading
            train_dataloader_kwargs["generator"] = g
            train_dataloader_kwargs["worker_init_fn"] = worker_init_fn
            train_dataloader_kwargs["shuffle"] = True

        collator = PaddedCollatorForTogether(
            model_max_length=self.tokenizer.model_max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            padding_side="right",
        )

        # create training and validation dataloaders
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.task_cfg.train_batch_size,
            collate_fn=collator,
            **train_dataloader_kwargs,
        )

        overwatch.info(f"Dataloader kwargs:")
        for k, v in train_dataloader_kwargs.items():
            overwatch.info(f"{k}: {v}", ctx_level=1)

        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.task_cfg.val_batch_size,
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
            f".{shorten(self.task_cfg.lr_scheduler_type)}"
            f".lr{self.task_cfg.learning_rate:.1e}"
        )

    def prepare(self, accelerator: Accelerator) -> DataLoader:
        # NOTE: if use deepspeed, model, optimizer, lr_scheduler must be prepared together !!! 
        self.model, self.optimizer, self.lr_scheduler, self.train_dataloader = accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler, self.train_dataloader
        )

        if self.train_cfg.data_parallel == "deepspeed":
            # Checkpointing itself is switched on in apply_vlm_trainability(); this
            # only re-asserts the input-require-grad hook after the DeepSpeed engine
            # wraps the model. Gate on the tuned components, not tune_vlm, so a
            # tune_mm_* run keeps it too.
            if self.model_cfg.gradient_checkpointing and self.vlm_trainable_components():
                assert "DeepSpeedEngine" in self.model.__class__.__name__, "deepspeed is not properly initialized!"
                if hasattr(self.model, "enable_input_require_grads"):
                    self.model.vlm_model.enable_input_require_grads()
                else:
                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)
                    self.model.vlm_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

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

        val_dataloader = getattr(self, "val_dataloader", None)
        if val_dataloader is not None: # not using if self.val_dataloader to avoid DataLoader.__len__() being called on iterable dataset
            self.val_dataloader = accelerator.prepare(self.val_dataloader)
            
        self.accelerator = accelerator
        return self.train_dataloader # type: ignore

    # VLM optimizer groups, i.e. everything that is NOT the action header.
    VLM_GROUPS = ("lang_backbone", "vision_tower", "mm_projector")

    def grad_norm_metrics(self) -> dict[str, float]:
        """Pre-clip gradient norms for logging: action header vs VLM backbone.
        """
        metrics = {
            f"grad_norm_{name}": value
            for name, value in self._grad_norms.items()
            if name != "action_header"  # already reported as grad_norm_act
        }
        vlm = [v for name, v in self._grad_norms.items() if name in self.VLM_GROUPS]
        if vlm:
            metrics["grad_norm_vlm"] = math.sqrt(sum(v * v for v in vlm))
        return metrics

    def clip_grad_norms(self) -> dict[str, float]:
        """Clip each optimizer group on its own budget; return pre-clip norms.

        Only for ddp: fsdp needs the sharded all-reduce inside
        model.clip_grad_norm_, which accelerate only takes when handed the full
        parameter list, and deepspeed clips inside its own engine.
        """
        max_norm = self.train_cfg.max_grad_norm
        if self.train_cfg.data_parallel != "ddp":
            if self.train_cfg.data_parallel == "deepspeed":
                return {}  # engine-side, via `gradient_clipping` in the ds config
            norm = (
                self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm)
                if max_norm is not None
                else self.get_total_grad_norm(self.model.parameters())
            )
            return {} if norm is None else {"global": float(norm)}

        norms: dict[str, float] = {}
        for group in self.optimizer.param_groups:
            params = group["params"]
            norm = (
                self.accelerator.clip_grad_norm_(params, max_norm)
                if max_norm is not None
                else self.get_total_grad_norm(params)
            )
            if norm is not None:
                norms[group.get("group_name", "params")] = float(norm)
        return norms

    def should_skip_step(self, grad_norm) -> bool:
        """Drop an optimizer step whose gradient is non-finite.

        A non-finite grad (inf/nan anywhere) poisons Adam's moment estimates and
        would corrupt every subsequent step, so it is never worth taking. Finite
        gradients are left to gradient clipping.
        """
        gn = float(grad_norm) if grad_norm is not None else 0.0
        if not math.isfinite(gn):
            self._skipped_steps += 1
            overwatch.warning(f"skip-step: non-finite grad norm ({gn}) at step {self.global_step}")
            return True
        return False

    def forward_autocast(self):
        """Autocast context for the model forward.

        DeepSpeed owns precision and (unlike accelerate DDP) does NOT autocast the
        forward -- `accelerator.autocast()` is a no-op there. The diffusion inputs
        built below (`torch.randn_like` noise, timesteps) stay fp32 and would hit the
        bf16 linears: "mat1 and mat2 must have the same dtype, but got Float and
        BFloat16". Mirror the DDP autocast explicitly.
        """
        if self.train_cfg.data_parallel != "deepspeed" or not torch.cuda.is_available():
            return self.accelerator.autocast()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(
            self.train_cfg.mixed_precision
        )
        if dtype is None:  # "no" -> fp32 params, nothing to cast
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=dtype)

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
            with self.forward_autocast():
                losses = self.forward_and_loss(self.model, batch)

            self.accelerator.backward(losses["loss"])
            skip_step = False
            if self.accelerator.sync_gradients:
                self._grad_norms = self.clip_grad_norms()
                # `grad_norm_act` keeps its name's meaning: the action header's own
                # norm, no longer contaminated by the backbone. Fall back to
                # whatever single norm exists (global / frozen-VLM runs).
                self._grad_norm_act = self._grad_norms.get(
                    "action_header", next(iter(self._grad_norms.values()), None)
                )
                # A non-finite gradient anywhere poisons the step, so the guard
                # sees the worst group, not just the header's.
                worst = max(self._grad_norms.values(), default=None) if all(
                    math.isfinite(v) for v in self._grad_norms.values()
                ) else float("inf")
                skip_step = self.should_skip_step(
                    self._grad_norm_act if worst != float("inf") else worst
                )

                if self.train_cfg.data_parallel == "ddp" and self.global_step == 0:
                    # NOTE: It is normal that param.grad is None for deepspeed and fsdp
                    # Only check on step 0 — iterating 2.6B params every step is expensive.
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is None:
                            overwatch.critical(f"[Unused] {name} did not receive a gradient.")

            if not skip_step:
                self.optimizer.step()
            # step the schedule on wall-step count either way, so a skipped step
            # does not shift the LR curve relative to global_step
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        # Log batch["raw_image"] to wandb every 100 steps
        if (
            hasattr(self, "global_step")
            and self.global_step % self.cfg.log.log_freq == 0
            and "observations" in batch
            and self.accelerator.is_main_process
            and self.cfg.log.report_to == "wandb"
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

        # step_loss = self.accelerator.gather(losses["loss"].detach()).mean().item() # type:ignore
        # self.train_loss_tracker = step_loss
        self.train_loss_tracker = step_loss = losses["loss"].detach().item()

        vlm_feat_norm = losses.get("vlm_feat_norm")
        metrics = {
            "lr_act": self.lr,
            **self.group_lrs(),
            "grad_norm_act": self._grad_norm_act, # type: ignore
            **self.grad_norm_metrics(),
            "loss": step_loss
        }
        if vlm_feat_norm is not None:
            metrics["vlm_feat_norm"] = vlm_feat_norm.detach().item()
        return (self.accelerator.sync_gradients, metrics)

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

        # instruction-only -> constant across diffusion steps, compute once
        pooled_projections = self.compute_pooled_projections(repacked_batch)

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
                pooled_projections=pooled_projections,  # (B, D) or None
                return_dict=True,
            ).action

            action_samples = self.noise_scheduler.step(
                model_output=model_pred, timestep=timestep, sample=action_samples # type: ignore
            ).prev_sample

        return action_samples
    
    def save_checkpoint(self, global_step: int) -> str | None:
        save_dir = os.path.join(self.project_dir, "checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        ckpt_dir = os.path.join(save_dir, f"ckpt_{global_step}")
        
        # For DDP, save_state() below already writes the complete model
        if self.train_cfg.data_parallel != "ddp":
            self.accelerator.save_model(self.model, ckpt_dir, max_shard_size="100GB")
        return super().save_checkpoint(global_step)

    def evaluate(self) -> dict[str, float] | None:
        assert self.val_dataloader is not None, "No validation dataloader available for evaluation."
        accelerator = self.accelerator
        global_step = self.global_step
        eval_model = self.unwrap_model()

        total_val_batches = (
            self.len_val_dataloader
            if self.task_cfg.val_num_batches == -1
            else min(self.task_cfg.val_num_batches, self.len_val_dataloader)
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
        dataset_name_list = []

        for val_step, val_batch in enumerate(val_progress_bar):
            val_batch = batch_str_to_tensor(val_batch)
            # mask = val_batch["mask"]
            mask = torch.ones_like(val_batch["actions"]) # FIXME
            gt_actions = val_batch["actions"]  # (B, Tp, Da)
            # gt_actions = self.data_cfg.data_transforms.normalize_action(repacked_batch[1])

            # Tp -> predicted action horizon, Da -> action dim
            B, Tp, Da = gt_actions.shape

            # validation loss
            with self.forward_autocast():
                val_loss = self.forward_and_loss(eval_model, val_batch)
                val_loss_list.append(val_loss["loss"].detach())

                # action prediction loss
                pred_actions = self.inference(eval_model, val_batch)
                err_action_l1 = (pred_actions - gt_actions[:, :Tp]).abs()  # (B, Tp, Da)
                action_l1_err_list.append(err_action_l1.reshape(-1, Da).float())
                if "dataset_name" in val_batch:
                    names = val_batch["dataset_name"] if isinstance(val_batch["dataset_name"], list) else [val_batch["dataset_name"]]
                    dataset_name_list.extend(n for n in names for _ in range(Tp))

            if val_step + 1 >= total_val_batches:
                if accelerator.is_local_main_process:
                    val_progress_bar.close()
                self.val_dataloader.end() # type: ignore
                break

        # Reduce scalars across ranks after all ranks have exited the loop.
        avg_val_loss = accelerator.reduce(
            torch.stack(val_loss_list).mean(), reduction="mean"
        ).item()

        # Gather action L1 errors and dataset names across all ranks.
        local_action_l1 = torch.cat(action_l1_err_list, dim=0)  # (N_local*Tp, Da)
        all_action_l1 = accelerator.gather(local_action_l1)  # (N_all*Tp, Da)

        if accelerator.num_processes > 1:
            import torch.distributed as dist
            all_names_per_rank: list[list[str]] = [[] for _ in range(accelerator.num_processes)]
            dist.all_gather_object(all_names_per_rank, dataset_name_list)
            all_dataset_names = [n for rank_names in all_names_per_rank for n in rank_names]
        else:
            all_dataset_names = dataset_name_list

        action_l1_err_arr = all_action_l1.cpu().numpy()  # (N_all*Tp, Da)
        action_l1_err_list_denormed = (
            self.maxmin.denormalize_L1_action_err( # type: ignore
                action_l1_err_arr,
                dataset_name=all_dataset_names or None,
            )
        )

        modality_metrics: dict[str, float] = {}
        for ds, modalities in action_l1_err_list_denormed.items():
            for mod_key, mod_l1 in modalities.items():
                modality_metrics[f"{ds}/{mod_key}"] = mod_l1.mean(0)

        return {
            "loss": avg_val_loss,
            **modality_metrics,
        }

    def forward_and_loss(self, model, batch) -> dict[str, torch.Tensor]:
        bsz, Tp, Da = batch["actions"].shape

        if self.noise_scheduler_name == "ddpm":
            timesteps = torch.randint(
                low=0, high=self.train_diffusion_steps, size=(bsz,), device=self.device
            ).long()

        elif self.noise_scheduler_name == "flow":
            # TODO check how pi_0 does this ?
            sigmas = torch.rand((bsz,), device=self.device)
            timesteps = sigmas * self.train_diffusion_steps

            if self.model_cfg.rtc:
                delay = torch.randint(
                    low=0,
                    high=self.model_cfg.max_delay,
                    size=(bsz,),
                    device=self.device,
                    dtype=torch.long
                )
                prefix_mask = torch.arange(Tp, device=self.device)[None, :] < delay[:, None]
                sigmas = torch.where(
                    prefix_mask,
                    torch.tensor(0.0, device=self.device, dtype=sigmas.dtype),
                    sigmas[:, None]
                )
                timesteps = sigmas * self.train_diffusion_steps

        else:
            raise ValueError(f"Unknown noise scheduler: {self.noise_scheduler_name}")

        noise_action = torch.randn_like(batch["actions"])
        if self.noise_scheduler_name == "ddpm":
            noisy_actions = self.noise_scheduler.add_noise(
                batch["actions"], noise_action, timesteps # type: ignore
            )
            target_action = noise_action  # ddpm epsilon-prediction
        elif self.noise_scheduler_name == "flow":
            sigmas_action = sigmas  # type: ignore
            while len(sigmas_action.shape) < len(batch["actions"].shape):
                sigmas_action = sigmas_action.unsqueeze(-1)  # Expand to B, Tp, action_dim 
            noisy_actions = (1 - sigmas_action) * batch["actions"] + sigmas_action * noise_action
            target_action = noise_action - batch["actions"] # flow-matching velocity prediction
        else:
            raise ValueError

        model_output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"], # vlm related
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"],
            action_samples=noisy_actions,  # (B,Tp,Da)
            states=batch["states"],  # (B,1,M)
            timestep=timesteps,
            traj2ds=batch["traj2ds"] if "traj2ds" in batch else None,  # (B, C, 3, H, W)
            pooled_projections=self.compute_pooled_projections(batch),  # (B, D) or None
            return_dict=True,
        )
        action_pred = model_output.action
        loss_action = F.mse_loss(
            action_pred.float(), target_action.float(), reduction="none"
        )  # (B, Tp, Da)
        """  
            sum(1) -> to keep gradient consistent when training with different action chunks
            mean(0) -> to average over the batch
            sum() -> to sum over the weighted loss of each action dimension
        """
        mask = batch["actions_mask"].float() if "actions_mask" in batch else torch.ones_like(batch["actions"])
        if self.model_cfg.rtc:
            postfix_mask = (~prefix_mask)[:, :, None].float() # type: ignore  (B, Tp, 1)  
            mask = mask * postfix_mask
        
        loss_action = (loss_action * mask).sum(1)  # (B, Da)
        loss_action = (loss_action.mean(0) * self.loss_w).sum()
        return {"loss": loss_action, "vlm_feat_norm": model_output.vlm_feat_norm}
