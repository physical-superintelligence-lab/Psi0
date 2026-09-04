from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from tqdm import tqdm


def resolve_num_eval_batches(trainer: Any, dataloader: Any) -> int:
    configured = getattr(trainer.train_cfg, "val_num_batches", -1)
    dataloader_len = len(dataloader)
    return dataloader_len if configured == -1 else min(configured, dataloader_len)


def run_eval_loop(
    *,
    trainer: Any,
    dataloader: Iterable[dict[str, Any]],
    update_fn: Callable[[int, dict[str, Any], Any], None],
    total_batches: int | None = None,
    description: str | None = None,
) -> Any:
    eval_model = trainer.unwrap_model()
    total_batches = total_batches if total_batches is not None else resolve_num_eval_batches(trainer, dataloader)
    accelerator = trainer.accelerator
    multi = accelerator.num_processes > 1
    data_iter = iter(dataloader)
    progress_bar = tqdm(
        range(total_batches),
        disable=not accelerator.is_local_main_process,
        position=1,
        leave=False,
    )
    progress_bar.set_description(
        description or f"Eval at global step {trainer.global_step}"
    )

    try:
        for index in progress_bar:
            # A rank-sharded IterableDataset (the PsiX HLP mixture loader) can hand different
            # ranks different batch COUNTS. If one rank exhausts before `total_batches` while
            # the others keep calling update_fn — which runs collective accelerator.gather() —
            # the ranks DEADLOCK (all GPUs spin sm=100%/mem=0%, eval frozen at step 0). Pull a
            # batch on every rank and MIN-reduce a "has a batch" flag so every rank stops on the
            # SAME iteration, keeping the collective gathers in lockstep.
            try:
                batch = next(data_iter)
                have = 1
            except StopIteration:
                batch, have = None, 0
            if multi:
                import torch
                import torch.distributed as dist
                flag = torch.tensor([have], device=accelerator.device)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                have = int(flag.item())
            if not have:
                break
            update_fn(index, batch, eval_model)
    finally:
        if accelerator.is_local_main_process:
            progress_bar.close()
        if hasattr(dataloader, "end"):
            dataloader.end()  # type: ignore[attr-defined]

    return eval_model
