"""
ZarrDataset implementation for EgoVerse.

Mirrors the LeRobotDataset API while reading data from Zarr arrays
instead of parquet/HF datasets.

Directory structure (per-episode metadata):
    dataset_root/
    └── episode_{ep_idx}.zarr/
        ├── observations.images.{cam}  (JPEG compressed)
        ├── observations.state
        ├── actions_joints
        └── ...

Each episode is self-contained with its own metadata, enabling:
- Independent episode uploads to S3
- Parallel processing without global coordination
- Easy episode-level data management
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import simplejpeg
import torch
import zarr
from tqdm import tqdm

# from action_chunk_transforms import Transform
from egomimic.rldb.filters import DatasetFilter
from egomimic.utils.aws.aws_data_utils import load_env
from egomimic.utils.aws.aws_sql import (
    create_default_engine,
    episode_table_to_df,
)

logger = logging.getLogger(__name__)

SEED = 42


def _is_primary_process() -> bool:
    """Return True only for the global rank-0 process in distributed runs."""
    rank = os.environ.get("RANK")
    if rank is None:
        # Non-distributed or rank not set.
        return True
    try:
        return int(rank) == 0
    except ValueError:
        return True


def split_dataset_names(dataset_names, valid_ratio=0.2, seed=SEED):
    """
    Split a list of dataset names into train/valid sets.
    Args:
        dataset_names (Iterable[str])
        valid_ratio (float): fraction of datasets to put in valid.
        seed (int): for deterministic shuffling.


    Returns:
        train_set (set[str]), valid_set (set[str])
    """
    names = sorted(dataset_names)
    if not names:
        return set(), set()

    rng = random.Random(seed)
    rng.shuffle(names)

    if not (0.0 <= valid_ratio <= 1.0):
        raise ValueError(f"valid_ratio must be in [0,1], got {valid_ratio}")

    n_valid = int(len(names) * valid_ratio)
    if valid_ratio > 0.0:
        n_valid = max(1, n_valid)

    valid = set(names[:n_valid])
    train = set(names[n_valid:])
    return train, valid


def _ensure_dataset_filter(filters: DatasetFilter | None) -> DatasetFilter:
    if filters is None:
        return DatasetFilter()
    if isinstance(filters, DatasetFilter):
        return filters
    if isinstance(filters, dict):
        lambdas = [f"lambda row: row.get({k!r}) == {v!r}" for k, v in filters.items()]
        return DatasetFilter(filter_lambdas=lambdas)
    raise TypeError(
        f"filters must be a DatasetFilter, dict, or None, got {type(filters)}"
    )


def _is_missing_filter_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""

    try:
        missing = pd.isna(value)
    except Exception:
        return False

    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _first_present(*values: object) -> object | None:
    for value in values:
        if not _is_missing_filter_value(value):
            return value
    return None


def _normalize_filter_row(
    row: Mapping[str, Any],
    *,
    episode_hash: str | None = None,
) -> dict[str, Any]:
    normalized = dict(row)
    normalized["episode_hash"] = (
        episode_hash if episode_hash is not None else normalized.get("episode_hash")
    )

    if _is_missing_filter_value(normalized.get("is_deleted")):
        normalized["is_deleted"] = False

    robot_name = _first_present(
        normalized.get("robot_name"),
        normalized.get("robot_type"),
        normalized.get("embodiment"),
    )
    if robot_name is not None:
        normalized["robot_name"] = robot_name

    return normalized


class EpisodeResolver:
    """
    Base class for episode resolution utilities.
    Provides shared static/class helpers; subclasses implement resolve().
    """

    def __init__(
        self,
        folder_path: Path,
        key_map: dict | None = None,
        transform_list: list | None = None,
    ):
        self.folder_path = Path(folder_path)
        self.key_map = key_map
        self.transform_list = transform_list

    def _load_zarr_datasets(self, search_path: Path, valid_folder_names: set[str]):
        """
        Loads multiple Zarr datasets from the specified folder path, filtering only those whose hashes
        are present in the valid_folder_names set.

        Args:
            search_path (Path): The root directory to search for Zarr datasets.
            valid_folder_names (set[str]): A set of valid folder names (episode hashes without ".zarr") to filter datasets.
        Returns:
            dict[str, ZarrDataset]: a dictionary mapping string keys to constructed zarr datasets from valid filters.
        """
        all_paths = sorted(search_path.iterdir())
        datasets: dict[str, ZarrDataset] = {}
        skipped: list[str] = []
        for p in tqdm(
            all_paths,
            desc=f"Loading Zarr datasets (pid={os.getpid()})",
            disable=not _is_primary_process(),
        ):
            if not p.is_dir():
                logger.info(f"{p} is not a valid directory")
                skipped.append(p.name)
                continue
            name = p.name
            if name.endswith(".zarr"):
                name = name[: -len(".zarr")]
            if name not in valid_folder_names:
                skipped.append(p.name)
                continue
            try:
                ds_obj = ZarrDataset(
                    p, key_map=self.key_map, transform_list=self.transform_list
                )
                datasets[name] = ds_obj
            except Exception as e:
                logger.debug(f"Failed to load dataset at {p}: {e}")
                skipped.append(p.name)

        return datasets

    @classmethod
    def _episode_already_present(cls, local_dir: Path, episode_hash: str) -> bool:
        direct = local_dir / episode_hash
        if direct.is_dir():
            return True


class S3EpisodeResolver(EpisodeResolver):
    """
    Resolves episodes via SQL table and optionally syncs from S3.
    """

    def __init__(
        self,
        folder_path: Path,
        bucket_name: str = "rldb",
        main_prefix: str = "processed_v3",
        key_map: dict | None = None,
        transform_list: list | None = None,
        debug: bool = False,
    ):
        self.bucket_name = bucket_name
        self.main_prefix = main_prefix
        self.debug = debug
        super().__init__(folder_path, key_map=key_map, transform_list=transform_list)

    def resolve(
        self,
        filters: DatasetFilter | None = None,
    ) -> dict[str, "ZarrDataset"]:
        """
        Outputs a dict of ZarrDatasets with relevant filters.
        Syncs S3 paths to local_root before indexing.
        """
        filters = _ensure_dataset_filter(filters)

        if self.folder_path.is_dir():
            logger.info(f"Using existing directory: {self.folder_path}")
        if not self.folder_path.is_dir():
            self.folder_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Filters: {filters}")

        filtered_paths = self.sync_from_filters(
            bucket_name=self.bucket_name,
            filters=filters,
            local_dir=self.folder_path,
            debug=self.debug,
        )

        valid_hashes = {hashes for _, hashes in filtered_paths}
        if not valid_hashes:
            raise ValueError(
                "No valid collection names from _get_filtered_paths: "
                "filters matched no episodes in the SQL table."
            )

        datasets = self._load_zarr_datasets(
            search_path=self.folder_path,
            valid_folder_names=valid_hashes,
        )

        return datasets

    @staticmethod
    def _get_filtered_paths(
        filters: DatasetFilter | None = None,
        debug: bool = False,
        limit: int | None = None,
    ) -> list[tuple[str, str]]:
        """
        Filters episodes from the SQL episode table according to the criteria specified in `filters`
        and returns a list of (zarr_processed_path, episode_hash) tuples for episodes that match and
        have a non-null zarr_processed_path.

        Args:
            filters (DatasetFilter | None): Filter object applied row-by-row to the
                episode table.
            limit (int | None): If set, return at most this many episodes.

        Returns:
            list[tuple[str, str]]: List of tuples, each containing (zarr_processed_path, episode_hash)
                                   for episodes passing the filter criteria.
        """
        filters = _ensure_dataset_filter(filters)
        engine = create_default_engine()
        df = episode_table_to_df(engine)
        if df.empty:
            logger.info("Episode table is empty.")
            return []

        if "is_deleted" in df.columns:
            df = df[~df["is_deleted"].fillna(False)]

        mask = df.apply(
            lambda row: filters.matches(_normalize_filter_row(row.to_dict())),
            axis=1,
        )
        output = df.loc[mask, ["zarr_processed_path", "episode_hash"]]

        output = output[
            output["zarr_processed_path"].fillna("").astype(str).str.strip() != ""
        ]

        if debug:
            output = output.head(10)
        elif limit is not None:
            output = output.head(limit)

        return list(output.itertuples(index=False, name=None))

    @classmethod
    def _sync_s3_to_local(
        cls,
        bucket_name: str,
        s3_paths: list[tuple[str, str]],
        local_dir: Path,
        numworkers: int = 10,
    ):
        if not s3_paths:
            return

        # 0) Skip episodes already present locally
        to_sync = []
        already = []
        for processed_path, episode_hash in s3_paths:
            if cls._episode_already_present(local_dir, episode_hash):
                already.append(episode_hash)
            else:
                to_sync.append((processed_path, episode_hash))

        if already:
            logger.info("Skipping %d episodes already present locally.", len(already))

        if not to_sync:
            logger.info("Nothing to sync from S3 (all episodes already present).")
            return

        # 1) Build s5cmd batch script (one line per episode)
        local_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix="_s5cmd_sync_",
            suffix=".txt",
            delete=False,
        ) as tmp_file:
            batch_path = Path(tmp_file.name)

        lines = []
        for processed_path, episode_hash in to_sync:
            # processed_path like: s3://rldb/processed_v2/eva/<hash>/
            if processed_path.startswith("s3://"):
                src_prefix = processed_path.rstrip("/") + "/*"
            else:
                src_prefix = (
                    f"s3://{bucket_name}/{processed_path.lstrip('/').rstrip('/')}"
                    + "/*"
                )

            # Destination is the root local_dir; s5cmd will preserve <hash>/... under it
            dst = local_dir / episode_hash
            lines.append(f'sync "{src_prefix}" "{str(dst)}/"')

        try:
            batch_path.write_text("\n".join(lines) + "\n")

            load_env()
            rl2_endpoint_url = os.environ.get("R2_ENDPOINT_URL")
            access_key_id = os.environ.get("R2_ACCESS_KEY_ID")
            secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY")
            if not all([rl2_endpoint_url, access_key_id, secret_access_key]):
                raise ValueError(
                    "R2 credentials missing. Ensure ~/.egoverse_env has "
                    "R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY."
                )
            s5cmd_env = os.environ.copy()
            s5cmd_env["AWS_ACCESS_KEY_ID"] = access_key_id
            s5cmd_env["AWS_SECRET_ACCESS_KEY"] = secret_access_key
            s5cmd_env["AWS_DEFAULT_REGION"] = "auto"
            s5cmd_env["AWS_REGION"] = "auto"
            cmd = [
                "s5cmd",
                "--endpoint-url",
                rl2_endpoint_url,
                "--numworkers",
                str(numworkers),
                "run",
                str(batch_path),
            ]
            logger.info("Running s5cmd batch (%d lines): %s", len(lines), " ".join(cmd))
            subprocess.run(cmd, check=True, env=s5cmd_env)

        finally:
            try:
                batch_path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning("Failed to delete batch file %s: %s", batch_path, e)

    @classmethod
    def sync_from_filters(
        cls,
        *,
        bucket_name: str,
        filters: DatasetFilter | None = None,
        local_dir: Path,
        numworkers: int = 10,
        debug: bool = False,
        limit: int | None = None,
    ):
        """
        Public API:
        - resolves episodes from DB using filters
        - runs a single aws s3 sync with includes
        - downloads into local_dir

        Args:
            numworkers: Number of parallel workers for s5cmd.
            limit: If set, download at most this many episodes.

        Returns:
            List[(processed_path, episode_hash)]
        """
        filters = _ensure_dataset_filter(filters)

        # 1) Resolve episodes from DB
        filtered_paths = cls._get_filtered_paths(filters, debug=debug, limit=limit)
        if not filtered_paths:
            logger.warning("No episodes matched filters.")
            return []

        # 2) Logging
        logger.info(
            f"Syncing S3 datasets with filters {filters} to local directory {local_dir}..."
        )

        # 3) Sync
        cls._sync_s3_to_local(
            bucket_name=bucket_name,
            s3_paths=filtered_paths,
            local_dir=local_dir,
            numworkers=numworkers,
        )

        return filtered_paths


class LocalEpisodeResolver(EpisodeResolver):
    """
    Resolves episodes from local Zarr stores, filtering via local metadata.
    """

    def __init__(
        self,
        folder_path: Path,
        key_map: dict | None = None,
        transform_list: list | None = None,
        debug=False,
    ):
        super().__init__(folder_path, key_map, transform_list)
        self.debug = debug

    @staticmethod
    def _local_filters_match(
        metadata: dict,
        episode_hash: str,
        filters: DatasetFilter,
    ) -> bool:
        return filters.matches(
            _normalize_filter_row(metadata, episode_hash=episode_hash)
        )

    @classmethod
    def _get_local_filtered_paths(
        cls,
        search_path: Path,
        filters: DatasetFilter | None = None,
        debug: bool = False,
    ):
        filters = _ensure_dataset_filter(filters)
        if not search_path.is_dir():
            logger.warning("Local path does not exist: %s", search_path)
            return []

        filtered = []
        for p in sorted(search_path.iterdir()):
            if not p.is_dir():
                continue

            episode_hash = p.name[:-5] if p.name.endswith(".zarr") else p.name

            try:
                store = zarr.open_group(str(p), mode="r")
                metadata = dict(store.attrs)
                # Augment metadata with which keys have non-zero data (checked on first frame).
                # This lets filter lambdas distinguish e.g. scale_bimanual from scale_left_arm,
                # which both have the same zarr keys but zero-fill the unused arm.
                valid_data_keys = set()
                for key in metadata.get("features", {}):
                    try:
                        frame = store[key][0]
                        if np.any(np.asarray(frame) != 0):
                            valid_data_keys.add(key)
                    except Exception:
                        pass
                metadata["valid_data_keys"] = valid_data_keys
            except Exception as e:
                logger.warning("Failed to read metadata for %s: %s", p, e)
                continue

            if cls._local_filters_match(metadata, episode_hash, filters):
                filtered.append((str(p), episode_hash))

        if debug:
            logger.info("Debug mode: limiting to 10 datasets.")
            filtered = filtered[:10]

        # logger.info("Local filtered paths: %s", filtered)
        return filtered

    def resolve(
        self,
        sync_from_s3=False,
        filters: DatasetFilter | None = None,
    ) -> dict[str, "ZarrDataset"]:
        """
        Outputs a dict of ZarrDatasets with relevant filters from local data.
        """
        if sync_from_s3:
            logger.warning(
                "LocalEpisodeResolver does not sync from S3; ignoring sync_from_s3=True."
            )

        filters = _ensure_dataset_filter(filters)

        filtered_paths = self._get_local_filtered_paths(
            self.folder_path, filters, debug=self.debug
        )

        valid_folder_names = {folder_name for _, folder_name in filtered_paths}
        # logger.info(f"Valid folder names: {valid_folder_names}")
        if not valid_folder_names:
            logger.warning(
                "No valid collection names from local filtering: "
                "filters matched no episodes in the local directory."
            )
            return {}

        datasets = self._load_zarr_datasets(
            search_path=self.folder_path, valid_folder_names=valid_folder_names
        )

        return datasets


class MultiDataset(torch.utils.data.Dataset):
    """
    Self wrapping MultiDataset, can wrap zarr or multi dataset.

    """

    def __init__(
        self,
        datasets,
        mode="train",
        percent=0.1,
        valid_ratio=0.2,
        **kwargs,
    ):
        """
        Args:
            datasets (dict): Dictionary mapping unique dataset hashes (str) to dataset objects. Datasets can be individual Zarr datasets or other multi-datasets; mixing different types is supported.
            mode (str, optional): Split mode to use (e.g., "train", "valid"). Defaults to "train".
            percent (float, optional): Fraction of the dataset to use from each underlying dataset. Defaults to 0.1.
            valid_ratio (float, optional): Validation split ratio for datasets that support a train/valid split.
            **kwargs: Additional keyword arguments passed to underlying dataset constructors if needed.
        """
        self.train_collections, self.valid_collections = split_dataset_names(
            datasets.keys(), valid_ratio=valid_ratio, seed=SEED
        )

        if mode == "train":
            chosen = self.train_collections
        elif mode == "valid":
            chosen = self.valid_collections
        elif mode == "total":
            chosen = set(datasets.keys())
        elif mode == "percent":
            all_names = sorted(datasets.keys())
            rng = random.Random(SEED)
            rng.shuffle(all_names)

            n_keep = int(len(all_names) * percent)
            if percent > 0.0:
                n_keep = max(1, n_keep)
            chosen = set(all_names[:n_keep])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.datasets = {rid: ds for rid, ds in datasets.items() if rid in chosen}

        self.index_map = []
        for dataset_name, dataset in self.datasets.items():
            for local_idx in range(len(dataset)):
                self.index_map.append((dataset_name, local_idx))

        super().__init__()

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx):
        dataset_name, local_idx = self.index_map[idx]
        
        try:
            data = self.datasets[dataset_name][local_idx]
            zarr_ds = self.datasets[dataset_name]
            robot_name = zarr_ds.embodiment        # e.g. "aria_left_arm", "mecka_bimanual"
            embodiment = robot_name.split("_")[0]  # e.g. "aria", "mecka", "scale", "eva"
            data["metadata.robot_name"] = robot_name
            data["embodiment"] = embodiment
            data["metadata.task_name"] = zarr_ds.metadata.get("task_name", "")
            data["metadata.task_description"] = zarr_ds.metadata.get("task_description", "")

        except Exception as e: # debug
            print("Error reading data:", dataset_name, local_idx)
            raise e

        return data

    @classmethod
    def _from_resolver(cls, resolver: EpisodeResolver, **kwargs):
        """
        create a MultiDataset from an EpisodeResolver.

        Args:
            resolver (EpisodeResolver): The resolver instance to use for loading datasets.
            embodiment: The embodiment identifier to use for resolving datasets.
            **kwargs: Keyword args forwarded to resolver (e.g., filters,
                sync_from_s3) and MultiDataset constructor (e.g., mode, percent,
                key_map, valid_ratio).
        Returns:
            MultiDataset: The constructed multi-dataset.
        """
        # TODO add key_map and transform pass to children

        sync_from_s3 = kwargs.pop("sync_from_s3", False)
        filters = kwargs.pop("filters", None)

        if isinstance(resolver, LocalEpisodeResolver):
            resolved = resolver.resolve(
                sync_from_s3=sync_from_s3,
                filters=filters,
            )
        else:
            resolved = resolver.resolve(filters=filters)

        return cls(datasets=resolved, **kwargs)


class ZarrDataset(torch.utils.data.Dataset):
    """
    Base Zarr Dataset object, Just intializes as pass through to read from zarr episode
    """

    def __init__(
        self,
        Episode_path: Path,
        key_map: dict,
        transform_list: list | None = None,
    ):
        """
        Args:
            episode_path: just a path to the designated zarr episode
            key_map: dict mapping from dataset keys to zarr keys and horizon info, e.g. {"obs/image/front": {"zarr_key": "observations.images.front", "horizon": 4}, ...}
            transform_list: list of Transform objects to apply to the data after loading, e.g. for action chunk transformations. Should be in order of application.
        """
        self.episode_path = Episode_path
        self.metadata = None
        self._image_keys = None  # Lazy-loaded set of JPEG-encoded keys
        self._json_keys = None  # Lazy-loaded set of JSON-encoded keys
        self._annotations = None
        self.init_episode()

        self.key_map = key_map
        self.transform = transform_list
        super().__init__()

    def init_episode(self):
        """
        inits the zarr episode and all the metadata associated, as well as total_frames for len
        """
        self.episode_reader = ZarrEpisode(self.episode_path)
        self.metadata = self.episode_reader.metadata
        self.total_frames = self.metadata["total_frames"]
        self.embodiment = self.metadata["embodiment"]
        self.keys_dict = {k: (0, None) for k in self.episode_reader._collect_keys()}

        # Detect JPEG-encoded image keys from metadata
        self._image_keys = self._detect_image_keys()
        self._json_keys = self._detect_json_keys()

    def _detect_image_keys(self) -> set[str]:
        """
        Detect which keys contain JPEG-encoded image data from metadata.

        Returns:
            Set of keys containing JPEG data
        """
        features = self.metadata.get("features", {})
        return {key for key, info in features.items() if info.get("dtype") == "jpeg"}

    def _detect_json_keys(self) -> set[str]:
        """
        Detect keys containing JSON-encoded bytes from metadata.

        Returns:
            Set of keys containing JSON payloads.
        """
        features = self.metadata.get("features", {})
        return {key for key, info in features.items() if info.get("dtype") == "json"}

    @staticmethod
    def _decode_json_entry(value):
        if isinstance(value, np.void):
            value = value.item()
        if isinstance(value, memoryview):
            value = value.tobytes()
        if isinstance(value, bytearray):
            value = bytes(value)
        if isinstance(value, bytes):
            return json.loads(value.decode("utf-8"))
        if isinstance(value, str):
            return json.loads(value)
        return value

    def _load_annotations(self) -> list[dict]:
        """
        Load and cache decoded language annotations.

        Expected format per entry:
            {"text": str, "start_idx": int, "end_idx": int}
        """
        if self._annotations is not None:
            return self._annotations

        raw = self.episode_reader._store["annotations"][:]

        decoded = [self._decode_json_entry(x) for x in raw]
        self._annotations = [d for d in decoded if isinstance(d, dict)]
        return self._annotations

    def _annotation_text_for_frame(self, frame_idx: int) -> str:
        """
        Resolve language annotation text for a frame from span annotations.
        """
        annotations = self._load_annotations()
        for ann in annotations:
            start_idx = int(ann.get("start_idx", -1))
            end_idx = int(ann.get("end_idx", -1))
            if start_idx <= frame_idx <= end_idx:
                return str(ann.get("text", ""))
        return ""

    def __len__(self) -> int:
        return self.total_frames

    def _pad_sequences(self, data, horizon: int | None) -> dict:
        if horizon is None:
            return data

        # Note that k is zarr key
        for k in data:
            if isinstance(data[k], np.ndarray):
                seq_len = data[k].shape[0]
                if seq_len < horizon:
                    # Pad by repeating the last frame
                    pad_len = horizon - seq_len
                    last_frame = data[k][-1:]  # Keep dims: (1, action_dim)
                    padding = np.repeat(last_frame, pad_len, axis=0)
                    data[k] = np.concatenate([data[k], padding], axis=0)

        return data

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Build keys_dict with ranges based on whether action chunking is enabled
        data = {}
        for k in self.key_map:
            zarr_key = self.key_map[k]["zarr_key"]
            horizon = self.key_map[k].get("horizon", None)

            if zarr_key == "annotations":
                data[k] = self._annotation_text_for_frame(idx)
                continue

            if horizon is not None:
                end_idx = min(idx + horizon, self.total_frames)
                read_interval = (idx, end_idx)
            else:
                read_interval = (idx, None)
            read_dict = {zarr_key: read_interval}
            raw_data = self.episode_reader.read(read_dict)
            self._pad_sequences(raw_data, horizon)  # should be able to pad images
            data[k] = raw_data[zarr_key]

            # Decode JPEG-encoded image data and normalize to [0, 1]
            # print(f"Print the image_keys: {self._image_keys}")
            if zarr_key in self._image_keys:
                jpeg_bytes = data[k]
                # Decode JPEG bytes to numpy array (H, W, 3)
                decoded = simplejpeg.decode_jpeg(jpeg_bytes, colorspace="RGB")
                # data[k] = torch.from_numpy(np.transpose(decoded, (2, 0, 1))).to(torch.float32) / 255.0
                data[k] = np.transpose(decoded, (2, 0, 1)) / 255.0
            elif zarr_key in self._json_keys:
                if isinstance(data[k], np.ndarray):
                    data[k] = [self._decode_json_entry(v) for v in data[k]]
                else:
                    data[k] = self._decode_json_entry(data[k])

        # Convert all numpy arrays in data to torch tensors

        # TODO add the transform list code here
        if self.transform:
            for transform in self.transform or []:
                # Skip transforms whose input key(s) are absent (e.g. right-arm
                # transforms on a unimanual episode that only has left-arm keys).
                # Different transform classes use different attribute names for their
                # input keys — check all known single-key and multi-key attributes.
                primary_key = (
                    getattr(transform, "input_key", None)
                    or getattr(transform, "chunk_world", None)
                    or getattr(transform, "pose_world", None)
                    or getattr(transform, "action_key", None)
                    or getattr(transform, "pose_key", None)
                )
                if primary_key is not None and primary_key not in data:
                    continue
                try:
                    data = transform.transform(data)
                except Exception as e:
                    logger.error(f"Error transforming data: {e}")
                    logger.error(f"Data: {data}")
                    logger.error(f"Transform: {transform}")
                    logger.error(f"Error: {e}")
                    if idx == 0:
                        logger.error("Error in first frame")
                        raise e
                    else:
                        return self.__getitem__(0)

        for k, v in data.items():
            if isinstance(v, np.ndarray):
                data[k] = torch.from_numpy(v).to(torch.float32)

        return data

    def get_item_keys(self, idx: int, keys) -> dict[str, torch.Tensor]:
        requested = self._normalize_keys_arg(keys)
        out = {}

        for k in requested:
            if k not in self.key_map:
                raise KeyError(
                    f"Unknown key '{k}'. Available keys: {list(self.key_map.keys())}"
                )

            zarr_key = self.key_map[k]["zarr_key"]
            horizon = self.key_map[k].get("horizon", None)

            if horizon is not None:
                end_idx = min(idx + horizon, self.total_frames)
                interval = (idx, end_idx)
            else:
                interval = (idx, None)

            raw = self.episode_reader.read({zarr_key: interval})
            self._pad_sequences(raw, horizon)
            val = raw[zarr_key]

            if zarr_key in self._image_keys:
                if (
                    isinstance(val, np.ndarray)
                    and val.dtype == object
                    and val.ndim == 1
                ):
                    decoded_seq = []
                    for jpeg_bytes in val:
                        img = simplejpeg.decode_jpeg(jpeg_bytes, colorspace="RGB")
                        decoded_seq.append(np.transpose(img, (2, 0, 1)) / 255.0)
                    val = np.stack(decoded_seq, axis=0)
                else:
                    img = simplejpeg.decode_jpeg(val, colorspace="RGB")
                    val = np.transpose(img, (2, 0, 1)) / 255.0

            out[k] = val

        if self.transform:
            for transform in self.transform or []:
                try:
                    out = transform.transform(out)
                except Exception as e:
                    logger.error(f"Error transforming data: {e}")
                    # NOTE: avoid dumping full arrays into logs
                    logger.error(f"Data keys: {list(out.keys())}")
                    logger.error(f"Transform: {transform}")
                    logger.error(f"Error: {e}")
                    if idx == 0:
                        logger.error("Error in first frame")
                        raise e
                    else:
                        return self.get_item_keys(0, keys)

        for k, v in out.items():
            if isinstance(v, np.ndarray):
                out[k] = torch.from_numpy(v).to(torch.float32)

        return out

    def _normalize_keys_arg(self, keys):
        """
        Normalize keys argument:
          None -> all dataset keys
          str  -> single key
          Iterable[str] -> list of keys
        """
        if keys is None:
            return list(self.key_map.keys())
        if isinstance(keys, str):
            return [keys]
        if isinstance(keys, Iterable):
            return list(keys)
        raise TypeError(f"keys must be None, str, or iterable[str], got {type(keys)}")


class ZarrEpisode:
    """
    Lightweight wrapper around a single Zarr episode store.
    Designed for efficient PyTorch DataLoader usage with direct store access.
    """

    __slots__ = (
        "_path",
        "_store",
        "metadata",
        "keys",
    )

    def __init__(self, path: str | Path):
        """
        Initialize ZarrEpisode wrapper.
        Args:
            path: Path to the .zarr episode directory
        """
        self._path = Path(path)
        self._store = zarr.open_group(str(self._path), mode="r")
        self.metadata = dict(self._store.attrs)
        self.keys = self.metadata["features"]

    def read(
        self, keys_with_ranges: dict[str, tuple[int, int | None]]
    ) -> dict[str, np.ndarray]:
        """
        Read data for specified keys, each with their own index or range.
        Args:
            keys_with_ranges: Dictionary mapping keys to (start, end) tuples.
                - start: Starting frame index
                - end: Ending frame index (exclusive). If None, reads single frame at start.
        Returns:
            Dictionary mapping keys to numpy arrays
        Example:
            >>> episode.read({
            ...     "obs/image": (0, 10),      # Read frames 0-10
            ...     "actions": (5, 15),        # Read frames 5-15
            ...     "rewards": (20, None),     # Read single frame at index 20
            ... })
        """
        result = {}
        for key, (start, end) in keys_with_ranges.items():
            arr = self._store[key]
            if end is not None:
                data = arr[start:end]
            else:
                # Single frame read - use slicing to avoid 0D array issues with VariableLengthBytes
                # arr[start:start+1] gives us a 1D array, then [0] extracts the actual object
                data = arr[start : start + 1][0]
            result[key] = data
        return result

    def _collect_keys(self) -> list[str]:
        """
        Collect all array keys from the store.
        Returns:
            List of array keys (flat structure with dot-separated names)
        """
        if isinstance(self.keys, dict):
            return list(self.keys.keys())
        return list(self.keys)

    def __len__(self) -> int:
        """
        Get total number of frames in the episode.
        Returns:
            Number of frames
        """
        return self.metadata["total_frames"]

    def __repr__(self) -> str:
        """String representation of the episode."""
        return f"ZarrEpisode(path={self._path}, frames={len(self)})"
