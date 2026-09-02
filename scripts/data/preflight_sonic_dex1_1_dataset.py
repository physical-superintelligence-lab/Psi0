"""Preflight checks for raw SONIC + Dex1-1 datasets before Psi0 conversion."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.offline.dex1_1_layout import (
    LEFT_GRIPPER_SLOT,
    LEFT_HAND_SOURCE_SLICE,
    MOTION_TOKEN_DIM,
    RIGHT_GRIPPER_SLOT,
    RIGHT_HAND_SOURCE_SLICE,
    SOURCE_VECTOR_DIM,
)
from scripts.offline.sonic_schema import (
    contains_end_effector_marker,
    info_feature_width,
    read_json_if_exists,
)

SRC_VIDEO_KEY = "observation.images.ego_view"
SRC_STATE = "observation.state"
SRC_ACTION_WBC = "action.wbc"
SRC_MOTION_TOKEN = "action.motion_token"
REQUIRED_COLUMNS = {
    SRC_STATE: SOURCE_VECTOR_DIM,
    SRC_ACTION_WBC: SOURCE_VECTOR_DIM,
    SRC_MOTION_TOKEN: MOTION_TOKEN_DIM,
}


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _json_has_original_key(obj: Any, key: str) -> bool:
    if isinstance(obj, dict):
        if obj.get("original_key") == key:
            return True
        return any(_json_has_original_key(value, key) for value in obj.values())
    if isinstance(obj, list):
        return any(_json_has_original_key(value, key) for value in obj)
    return False


def _metadata_report(data_root: Path) -> dict[str, Any]:
    meta_dir = data_root / "meta"
    modality = _load_json(meta_dir / "modality.json")
    info = read_json_if_exists(meta_dir / "info.json")
    metadata = {"modality": modality, "info": info}
    required_original_keys = {key: _json_has_original_key(metadata, key) for key in REQUIRED_COLUMNS}
    required_schema_keys = {
        key: required_original_keys[key] or info_feature_width(info, key) == expected_width
        for key, expected_width in REQUIRED_COLUMNS.items()
    }
    dex1_marker = contains_end_effector_marker(metadata, ("dex1_1", "dex1-1", "dex1 1"))
    virtual14_marker = contains_end_effector_marker(
        metadata, ("dex1_virtual14", "dex1-virtual14")
    )
    gripper_fields_located = (dex1_marker or virtual14_marker) and all(required_schema_keys.values())
    return {
        "tasks_jsonl": (meta_dir / "tasks.jsonl").is_file(),
        "episodes_jsonl": (meta_dir / "episodes.jsonl").is_file(),
        "modality_json": (meta_dir / "modality.json").is_file(),
        "info_json": (meta_dir / "info.json").is_file(),
        "required_original_keys": required_original_keys,
        "required_schema_keys": required_schema_keys,
        "dex1_1_marker": dex1_marker,
        "dex1_virtual14_marker": virtual14_marker,
        "gripper_fields_located": gripper_fields_located,
    }


def _discover_parquets(data_root: Path, max_episodes: int | None) -> list[Path]:
    parquets = sorted((data_root / "data").rglob("episode_*.parquet"))
    if max_episodes is not None:
        parquets = parquets[:max_episodes]
    return parquets


def _episode_index(parquet_path: Path) -> int:
    try:
        return int(parquet_path.stem.split("_")[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"cannot parse episode index from {parquet_path.name}") from exc


def _find_video(data_root: Path, parquet_path: Path, chunks_size: int) -> Path | None:
    episode_index = _episode_index(parquet_path)
    candidates = [
        data_root / "videos" / parquet_path.parent.name / SRC_VIDEO_KEY / f"episode_{episode_index:06d}.mp4",
        data_root / "videos" / f"chunk-{episode_index // chunks_size:03d}" / SRC_VIDEO_KEY / f"episode_{episode_index:06d}.mp4",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted((data_root / "videos").glob(f"*/{SRC_VIDEO_KEY}/episode_{episode_index:06d}.mp4"))
    return matches[0] if matches else None


def _probe_video(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return report

    if shutil.which("ffprobe"):
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ]
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            streams = json.loads(result.stdout).get("streams", [])
            if streams:
                stream = streams[0]
                report.update(
                    {
                        "width": stream.get("width"),
                        "height": stream.get("height"),
                        "fps": stream.get("avg_frame_rate"),
                        "frame_count": stream.get("nb_frames"),
                        "probe": "ffprobe",
                    }
                )
                return report
        except Exception as exc:  # pragma: no cover - best-effort diagnostics
            report["probe_error"] = str(exc)
            return report

    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(path))
        report.update(
            {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": float(cap.get(cv2.CAP_PROP_FPS)),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "probe": "cv2",
            }
        )
        cap.release()
    except Exception as exc:  # pragma: no cover - best-effort diagnostics
        report["probe_error"] = str(exc)
    return report


def _stack_vectors(df: pd.DataFrame, column: str) -> np.ndarray:
    values = []
    for item in df[column]:
        arr = np.asarray(item, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"{column} entries must be 1-D, got {arr.shape}")
        values.append(arr)
    if not values:
        raise ValueError(f"{column} has no rows")
    return np.vstack(values)


def _range(values: np.ndarray) -> dict[str, float]:
    return {"min": float(np.min(values)), "max": float(np.max(values))}


def _episode_report(data_root: Path, parquet_path: Path, chunks_size: int) -> dict[str, Any]:
    episode_index = _episode_index(parquet_path)
    report: dict[str, Any] = {
        "episode_index": episode_index,
        "parquet": _rel(parquet_path, data_root),
        "ok": True,
        "errors": [],
        "warnings": [],
    }
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:
        report["ok"] = False
        report["errors"].append(f"cannot read parquet: {exc}")
        return report

    report["frames"] = int(len(df))
    if len(df) == 0:
        report["ok"] = False
        report["errors"].append("empty episode")
        return report

    vectors: dict[str, np.ndarray] = {}
    for column, expected_dim in REQUIRED_COLUMNS.items():
        if column not in df.columns:
            report["ok"] = False
            report["errors"].append(f"missing column: {column}")
            continue
        try:
            vector = _stack_vectors(df, column)
            vectors[column] = vector
        except Exception as exc:
            report["ok"] = False
            report["errors"].append(f"{column}: {exc}")
            continue
        report[f"{column}.shape"] = list(vector.shape)
        if vector.shape[1] != expected_dim:
            report["ok"] = False
            report["errors"].append(f"{column} dim {vector.shape[1]} != {expected_dim}")
        if not np.all(np.isfinite(vector)):
            report["ok"] = False
            report["errors"].append(f"{column} contains NaN/Inf")

    video = _find_video(data_root, parquet_path, chunks_size)
    if video is None:
        report["ok"] = False
        report["errors"].append(f"missing video for {SRC_VIDEO_KEY}")
    else:
        video_report = _probe_video(video)
        video_report["path"] = _rel(video, data_root)
        report["video"] = video_report

    if SRC_MOTION_TOKEN in vectors and np.allclose(vectors[SRC_MOTION_TOKEN], 0.0):
        report["warnings"].append(f"{SRC_MOTION_TOKEN} is all zero in this episode")

    state = vectors.get(SRC_STATE)
    wbc = vectors.get(SRC_ACTION_WBC)
    left_state_index = LEFT_HAND_SOURCE_SLICE[0] + LEFT_GRIPPER_SLOT
    right_state_index = RIGHT_HAND_SOURCE_SLICE[0] + (RIGHT_GRIPPER_SLOT - 7)
    gripper_report: dict[str, Any] = {
        "source_indices": {
            "left": left_state_index,
            "right": right_state_index,
        }
    }
    if state is not None and state.shape[1] > max(left_state_index, right_state_index):
        gripper_report["observation.state"] = {
            "left": _range(state[:, left_state_index]),
            "right": _range(state[:, right_state_index]),
        }
    if wbc is not None and wbc.shape[1] > max(left_state_index, right_state_index):
        gripper_report["action.wbc"] = {
            "left": _range(wbc[:, left_state_index]),
            "right": _range(wbc[:, right_state_index]),
        }
    report["gripper_raw_range"] = gripper_report
    return report


def run_preflight(
    data_root: Path,
    chunks_size: int,
    max_episodes: int | None,
    hand_layout: str = "dex1-1",
) -> dict[str, Any]:
    data_root = data_root.expanduser().resolve()
    meta = _metadata_report(data_root)
    report: dict[str, Any] = {
        "data_root": str(data_root),
        "required_schema": {
            SRC_STATE: SOURCE_VECTOR_DIM,
            SRC_ACTION_WBC: SOURCE_VECTOR_DIM,
            SRC_MOTION_TOKEN: MOTION_TOKEN_DIM,
            SRC_VIDEO_KEY: "mp4",
        },
        "metadata": meta,
        "episodes": [],
        "errors": [],
        "warnings": [],
    }

    for key in ("tasks_jsonl", "episodes_jsonl"):
        if not meta[key]:
            report["errors"].append(f"missing meta/{key.replace('_', '.')}")
    if not (meta["modality_json"] or meta["info_json"]):
        report["errors"].append("missing meta/modality.json or equivalent metadata")
    missing_schema_keys = [key for key, ok in meta["required_schema_keys"].items() if not ok]
    if missing_schema_keys:
        report["errors"].append(
            "metadata cannot locate expected source features: " + ", ".join(missing_schema_keys)
        )
    expected_marker = (
        meta["dex1_virtual14_marker"]
        if hand_layout == "dex1_virtual14"
        else meta["dex1_1_marker"]
    )
    if not expected_marker:
        report["errors"].append(
            f"metadata does not explicitly mark requested hand layout: {hand_layout}"
        )

    parquets = _discover_parquets(data_root, max_episodes)
    if not parquets:
        report["errors"].append("no data/*/episode_*.parquet files found")

    episode_reports = [_episode_report(data_root, parquet_path, chunks_size) for parquet_path in parquets]
    report["episodes"] = episode_reports
    report["episode_count"] = len(episode_reports)
    report["frame_count"] = int(sum(ep.get("frames", 0) for ep in episode_reports))
    report["video_count"] = int(sum(1 for ep in episode_reports if ep.get("video", {}).get("exists")))

    for ep in episode_reports:
        if ep.get("warnings"):
            report["warnings"].extend(f"episode {ep['episode_index']}: {warning}" for warning in ep["warnings"])
        if not ep.get("ok", False):
            report["errors"].extend(f"episode {ep['episode_index']}: {error}" for error in ep.get("errors", []))

    report["can_convert"] = not report["errors"]
    return report


def print_report(report: dict[str, Any]) -> None:
    print("SONIC Dex1-1 preflight report")
    print(f"  data_root: {report['data_root']}")
    print(f"  episodes checked: {report.get('episode_count', 0)}")
    print(f"  frames checked: {report.get('frame_count', 0)}")
    print(f"  videos found: {report.get('video_count', 0)}")
    print(f"  can_convert: {report['can_convert']}")
    print("  required fields:")
    for key, dim in report["required_schema"].items():
        print(f"    {key}: {dim}")
    print("  metadata:")
    metadata = report["metadata"]
    print(f"    tasks.jsonl: {metadata['tasks_jsonl']}")
    print(f"    episodes.jsonl: {metadata['episodes_jsonl']}")
    print(f"    modality.json: {metadata['modality_json']}")
    print(f"    dex1_1 marker: {metadata['dex1_1_marker']}")
    print(f"    dex1_virtual14 marker: {metadata['dex1_virtual14_marker']}")
    print(f"    gripper fields located: {metadata['gripper_fields_located']}")

    for ep in report["episodes"]:
        print(f"  episode {ep['episode_index']}: frames={ep.get('frames')} ok={ep.get('ok')}")
        if "video" in ep:
            video = ep["video"]
            shape = (
                f"{video.get('height')}x{video.get('width')}"
                if video.get("height") is not None and video.get("width") is not None
                else "unknown"
            )
            print(
                f"    video: {video.get('path')} shape={shape} "
                f"fps={video.get('fps')} frames={video.get('frame_count')}"
            )
        ranges = ep.get("gripper_raw_range", {})
        for source_key in (SRC_STATE, SRC_ACTION_WBC):
            if source_key in ranges:
                left = ranges[source_key]["left"]
                right = ranges[source_key]["right"]
                print(
                    f"    {source_key} gripper range: "
                    f"L={left['min']:.6g}/{left['max']:.6g} "
                    f"R={right['min']:.6g}/{right['max']:.6g}"
                )

    if report["warnings"]:
        print("  warnings:")
        for warning in report["warnings"]:
            print(f"    - {warning}")
    if report["errors"]:
        print("  errors:")
        for error in report["errors"]:
            print(f"    - {error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--chunks-size", type=int, default=1000)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--hand-layout",
        choices=["dex1-1", "dex1_virtual14"],
        default="dex1-1",
    )
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args()
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive")

    report = run_preflight(
        Path(args.data_root), args.chunks_size, args.max_episodes, args.hand_layout
    )
    print_report(report)
    if args.report_json:
        report_path = Path(args.report_json).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=4), encoding="utf-8")
        print(f"Wrote report JSON: {report_path}")
    if not report["can_convert"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
