from __future__ import annotations

try:
    from lerobot.common.datasets.lerobot_dataset import (  # type: ignore
        LeRobotDataset,
        LeRobotDatasetMetadata,
        MultiLeRobotDataset,
    )

    LEROBOT_LAYOUT = "common"
except ModuleNotFoundError:
    from lerobot.datasets.lerobot_dataset import (  # type: ignore
        LeRobotDataset,
        LeRobotDatasetMetadata,
        MultiLeRobotDataset,
    )

    LEROBOT_LAYOUT = "datasets"

__all__ = [
    "LEROBOT_LAYOUT",
    "LeRobotDataset",
    "LeRobotDatasetMetadata",
    "MultiLeRobotDataset",
]

# ---- datasets>=4 "List" feature compat ------------------------------------
# The re-split finetune packs were written by a NEWER `datasets`, whose parquet
# metadata serializes variable-length lists as {"_type": "List", "feature": X}.
# This venv pins datasets 3.6.0, which has no "List" and raises
#   ValueError: Feature type 'List' not found
# inside Features.from_arrow_schema during LeRobotDataset.load_hf_dataset (the
# sharded loader is unaffected: it reads parquet with its own pyarrow code).
# "List" is shape-identical to the Sequence serialization and the actual arrow
# column type is plain list<...>, which 3.6.0 maps to Sequence -- so alias it.
# generate_from_dict resolves classes through _FEATURE_TYPES at call time and
# its Sequence branch is `Sequence(feature=generate_from_dict(f), **obj)`,
# which the List nodes (no extra keys) satisfy.
try:  # pragma: no cover - depends on the installed datasets version
    from datasets.features import features as _dsf

    if "List" not in _dsf._FEATURE_TYPES:
        _dsf._FEATURE_TYPES["List"] = _dsf.Sequence
except Exception:  # newer datasets already know "List"; never block imports
    pass
