"""Lightweight SONIC metadata helpers used by offline tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json_if_exists(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def info_feature_width(info: Any, key: str) -> int | None:
    """Return a 1-D feature width from official LeRobot ``info.json`` metadata."""
    if not isinstance(info, dict):
        return None
    features = info.get("features")
    if not isinstance(features, dict):
        return None
    entry = features.get(key)
    if not isinstance(entry, dict):
        return None
    shape = entry.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 1:
        return None
    width = shape[0]
    return width if isinstance(width, int) and not isinstance(width, bool) else None


def contains_end_effector_marker(obj: Any, markers: tuple[str, ...]) -> bool:
    text = json.dumps(obj, ensure_ascii=False).lower()
    return any(marker in text for marker in markers)


def detect_end_effector(data_root: Path, modality: dict[str, Any]) -> str:
    """Detect source end-effector from explicit metadata markers.

    Defaults to ``dex3`` to preserve the original SONIC conversion path. Dex1-1
    conversion should be requested with ``--end-effector dex1_1`` or marked in
    source schema metadata with a dex1-1/dex1_1 token.
    """
    meta_dir = data_root / "meta"
    metadata = {
        "modality": modality,
        "info": read_json_if_exists(meta_dir / "info.json"),
    }
    if contains_end_effector_marker(metadata, ("dex1_virtual14", "dex1-virtual14")):
        return "dex1_virtual14"
    if contains_end_effector_marker(metadata, ("dex1_1", "dex1-1", "dex1 1")):
        return "dex1_1"
    if contains_end_effector_marker(metadata, ("dex3", "dex3_1", "dex3-1")):
        return "dex3"
    return "dex3"
