import os
import argparse
import dotenv
dotenv.load_dotenv()

import torch
import numpy as np
from pathlib import Path
from psi.utils import parse_args_to_tyro_config, seed_everything, move_to_device
from psi.config.config import LaunchConfig
from psi.config.tokenizer import FastActionTokenizerConfig

from torch.utils.data import DataLoader
from psi.trainers.qwen3vl_mixin import PaddedCollatorForActionPrediction

DEFAULT_TRAINING_SCRIPT = "scripts/train/psi0/pretrain-egodex-psi0-fast.sh"
DEFAULT_EGODEX_DIR = "/mnt/beegfs/datasets/egodex"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_NUM_EVALS = 500
DEFAULT_BATCH_SIZE = 16
DEFAULT_LOADER_SEED = None

def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FAST pretrain checkpoint on EgoDex split"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path or Hugging Face id for the pretrained checkpoint",
    )
    parser.add_argument(
        "split",
        choices=["train", "val"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--training-script",
        type=str,
        default=DEFAULT_TRAINING_SCRIPT,
        help="Training script used to parse config via parse_args_to_tyro_config",
    )
    parser.add_argument(
        "--egodex-dir",
        type=str,
        default=DEFAULT_EGODEX_DIR,
        help="Path to EgoDex dataset root",
    )
    parser.add_argument(
        "--num-evals",
        type=int,
        default=DEFAULT_NUM_EVALS,
        help="Approximate number of samples to evaluate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Approximate number of samples to evaluate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Torch device for model inference, e.g., cuda:0 or cpu",
    )
    parser.add_argument(
        "--loader-seed",
        type=int,
        default=DEFAULT_LOADER_SEED,
        help="Seed for DataLoader shuffling. Defaults to config.seed when omitted.",
    )
    args = parser.parse_args()
    if args.num_evals < 1:
        raise ValueError("--num-evals must be >= 1")
    return args


args = parse_cli_args()
PRETRAIN_CHECKPOINT = args.checkpoint
SPLIT = args.split
TRAINING_SCRIPT = args.training_script
EGODEX_DIR = args.egodex_dir
NUM_EVALS = args.num_evals
DEVICE = args.device
LOADER_SEED = args.loader_seed
BATCH_SIZE = args.batch_size

config:LaunchConfig = parse_args_to_tyro_config(TRAINING_SCRIPT)
seed_everything(config.seed or 42)

if LOADER_SEED is None:
    LOADER_SEED = config.seed or 42

from psi.config.data_lerobot import LerobotDataConfig
data_cfg: LerobotDataConfig = config.data # type: ignore
data_cfg.root_dir = EGODEX_DIR

maxmin = data_cfg.transform.field

from psi.config.model_qwen3vl import Qwen3VLModelConfig
model_cfg: Qwen3VLModelConfig = config.model # type: ignore

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from psi.tokenizer import FastActionTokenizer
from qwen_vl_utils import process_vision_info

Da = 48
vlm_processor = AutoProcessor.from_pretrained(PRETRAIN_CHECKPOINT)
tokenizer = vlm_processor.tokenizer
tokenizer.padding_side = "left"
vlm_processor.tokenizer.padding_side = "left"

print(f"Using device: {DEVICE}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    PRETRAIN_CHECKPOINT,
    attn_implementation="flash_attention_2",
    dtype=torch.bfloat16,
    device_map={"": DEVICE},  # Load entire model to GPU 7
)

if isinstance(model_cfg.action_tokenizer, FastActionTokenizerConfig):
    model.resize_token_embeddings(
        len(tokenizer) + model_cfg.action_tokenizer.bins, 
        pad_to_multiple_of = 192,
        mean_resizing = True
    )
    print(f"Resized model token embeddings to {model.lm_head.weight.shape[0]}")

action_tokenizer = FastActionTokenizer(
    tokenizer, data_cfg.chunk_size, Da,
    pretrained_checkpoint=model_cfg.action_tokenizer.pretrained_checkpoint,
    bins=model_cfg.action_tokenizer.bins
)

transform_kwargs=dict(
    action_tokenizer=action_tokenizer,
    vlm_processor=vlm_processor,
    is_eval=True, 
)

train_dataset = data_cfg(split="train", transform_kwargs=transform_kwargs)
val_dataset = data_cfg(split="val", transform_kwargs=transform_kwargs)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
# exit(0)

from PIL import Image
import numpy as np
np.set_printoptions(precision=4, suppress=True)

raw_dataset = train_dataset if SPLIT == "train" else val_dataset

# print(type(raw_dataset))
if "Mixed" in data_cfg.__class__.__name__:
    raw_dataset = raw_dataset.datasets[1]
    print(raw_dataset)

dataloader = DataLoader(
    raw_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    generator=torch.Generator().manual_seed(LOADER_SEED),
    collate_fn=PaddedCollatorForActionPrediction(
        model_max_length=tokenizer.model_max_length,
        pad_token_id=tokenizer.pad_token_id,
        padding_side="left",
    )
)

print(f"DataLoader shuffle seed: {LOADER_SEED}")

decoding_error_count = 0
l1_xyz = []
evaluated_samples = 0
im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

for i, batch in enumerate(dataloader):
    if evaluated_samples >= NUM_EVALS:
        break

    gt_actions = batch["raw_actions"]  # (B, Tp, Da)
    batch_size = batch["input_ids"].shape[0]

    with torch.inference_mode():
        batch = move_to_device(batch, DEVICE) 
        generated_ids = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"],
            max_new_tokens=1024,
            do_sample=False,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(batch["input_ids"], generated_ids)
        ]

    # Iterate each sample to calcuate L1 loss
    for b, pred_ids in enumerate(generated_ids_trimmed):
        if pred_ids.numel() == 0:
            decoding_error_count += 1
            continue

        im_end_positions = (pred_ids == im_end_token_id).nonzero(as_tuple=True)[0]
        if im_end_positions.numel() == 0:
            print(f"Warning: Sample {b} output may be truncated. No <|im_end|> token found.")
            decoding_error_count += 1
            continue

        end_idx = int(im_end_positions[0].item())
        if end_idx == 0:
            print(f"Warning: Sample {b} has empty action prediction before <|im_end|>.")
            decoding_error_count += 1
            continue

        action_pred = action_tokenizer.decode_token_ids_to_actions(
            [pred_ids[:end_idx].cpu().numpy().tolist()]
        )
        denorm_action_pred = maxmin.denormalize(action_pred) # (1, Tp, Da)
        denorm_gt_action = maxmin.denormalize(gt_actions[b:b+1])

        errs = np.abs(denorm_action_pred - denorm_gt_action).mean()
        print(f"Batch {i}, sample {b} L1 error:", errs)
        l1_xyz.append(errs)
        evaluated_samples += 1
    
    break # debug

# print(f"L1 errors: {l1_xyz}")
print(f"ckpt={os.path.basename(PRETRAIN_CHECKPOINT)}, split={SPLIT}")
print("decoding_error_count:", decoding_error_count)
print(f"Mean L1 error over {evaluated_samples} samples: {np.mean(l1_xyz)}")
