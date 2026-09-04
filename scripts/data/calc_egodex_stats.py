"""Compute per-dim action stats (min/max/q01/q99/mean/std) for EgoDex delta actions.

Samples pre-normalization delta chunks through the same data config as the FAST
tokenizer fit / pretrain (so chunk_size / upsample_rate / use_delta_actions all
match), bypassing the field transform by reading from the raw dataset directly.

The resulting json feeds ActionStateTransform (bounds_q99 normalization) and must
be regenerated whenever the action timescale changes (e.g. upsample_rate 3 -> 1),
because delta magnitudes scale with the frame stride. Never overwrite
assets/stats/egodex_stat_all.json: the psi0 scripts/checkpoints depend on it.

Usage:
    python scripts/data/calc_egodex_stats.py \
        --training_script scripts/train/psix/pretrain-egodex-psix-fast.sh \
        --output assets/stats/egodex_stat_all_30hz.json
"""
import argparse
import json

import numpy as np
import tqdm

if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv(verbose=True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training_script', type=str,
                        default="scripts/train/psix/pretrain-egodex-psix-fast.sh")
    parser.add_argument('--num_samples', type=int, default=20_000,
                        help='number of action chunks to sample (quantiles converge fast; '
                             '20k chunks x 30 steps = 600k per-step samples)')
    parser.add_argument('--output', type=str, default="assets/stats/egodex_stat_all_30hz.json")
    parser.add_argument('--key', type=str, default="egodex",
                        help='stat_action_key the field transform will look up')
    parser.add_argument('--abs', action='store_true',
                        help='absolute camera-frame ee pose instead of relative-to-frame-0 delta')
    args = parser.parse_args()

    # Bootstrap: the training script's field transform loads args.output at config
    # construction time, but that file is exactly what we are about to compute.
    # Drop in the legacy stats as a placeholder (raw action collection below does
    # not depend on them), then overwrite with the real values at the end.
    import os
    import shutil
    if not os.path.exists(args.output):
        shutil.copyfile("assets/stats/egodex_stat_all.json", args.output)
        print(f"bootstrapped placeholder {args.output} from legacy stats")

    from psi.utils import parse_args_to_tyro_config, seed_everything
    from psi.config.config import LaunchConfig

    config: LaunchConfig = parse_args_to_tyro_config(args.training_script)  # type: ignore
    seed_everything(config.seed or 42)

    data_cfg = config.data
    assert args.abs or getattr(data_cfg, "use_delta_actions", False), \
        "these stats are meant for delta actions"

    # raw (pre-normalization) delta chunks: read the wrapped dataset directly so
    # the field transform (which would normalize with the OLD stats) is bypassed
    raw_dataset = data_cfg(split="train", transform_kwargs={}).raw_dataset
    if args.abs:
        raw_dataset.use_delta_actions = False
        raw_dataset.use_abs_actions = True

    chunks = []
    for _ in tqdm.tqdm(range(args.num_samples)):
        idx = np.random.randint(0, len(raw_dataset))
        chunks.append(np.asarray(raw_dataset[idx]["actions"], dtype=np.float32))
    flat = np.concatenate(chunks, axis=0)  # (num_samples * chunk_size, Da)
    print("collected per-step samples:", flat.shape)

    stats = {
        args.key: {
            "min": flat.min(axis=0).tolist(),
            "max": flat.max(axis=0).tolist(),
            "q01": np.quantile(flat, 0.01, axis=0).tolist(),
            "q99": np.quantile(flat, 0.99, axis=0).tolist(),
            "mean": flat.mean(axis=0).tolist(),
            "std": flat.std(axis=0).tolist(),
        }
    }
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)
    print("saved", args.output)
    print("q99[:6]:", np.round(np.array(stats[args.key]["q99"][:6]), 4))
    print("q01[:6]:", np.round(np.array(stats[args.key]["q01"][:6]), 4))
