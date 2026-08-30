import logging
from pathlib import Path
from typing import Optional
import os
import numpy as np
import torch.utils.data
import zarr
from egomimic.rldb.embodiment.human import Aria, Mecka, Scale
from egomimic.rldb.zarr.action_chunk_transforms import DeleteKeys, InterpolatePose
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import (
    LocalEpisodeResolver,
    MultiDataset,
    _ensure_dataset_filter,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


BIMANUAL_ROBOT_NAMES = ("mecka_bimanual", "scale_bimanual", "aria_bimanual")


class SQLLocalEpisodeResolver(LocalEpisodeResolver):
    """Resolves local episodes using the SQL database for metadata/filtering.

    1. Queries the SQL DB for bimanual robot episodes only (targeted WHERE clause).
    2. Caches the result to a parquet file; subsequent runs use the cache.
    3. Intersects with locally present episode directories and logs missed episodes.
    4. Applies any extra filter lambdas, then opens only the matched zarr stores.

    Falls back to the default ``LocalEpisodeResolver`` zarr scan when the SQL
    database is not accessible (no credentials / offline).
    """

    def resolve(
        self,
        sync_from_s3: bool = False,
        filters: DatasetFilter | None = None,
    ) -> dict:
        if sync_from_s3:
            logger.warning("SQLLocalEpisodeResolver does not sync from S3; ignoring.")

        filters = _ensure_dataset_filter(filters)

        if not self.folder_path.is_dir():
            logger.warning("Local path does not exist: %s", self.folder_path)
            return {}

        # --- 1. Scan local episode folders ---
        local_hashes: dict[str, Path] = {}
        for p in tqdm(self.folder_path.iterdir(), desc="Scanning local episodes"):
            if not p.is_dir() or p.name.startswith("."):
                continue
            h = p.name[:-5] if p.name.endswith(".zarr") else p.name
            local_hashes[h] = p

        print(f"data root    : {self.folder_path}")
        print(f"local folders: {len(local_hashes)}")

        # --- 2. Load or build bimanual-only SQL cache ---
        _psi_home = os.environ.get("PSI_HOME")
        _SQL_CACHE = (
            Path(_psi_home) / "cache/egoverse/bimanual_episodes.parquet"
            if _psi_home
            else Path.home() / ".cache/egoverse/bimanual_episodes.parquet"
        )

        import pandas as pd
        df = None
        if _SQL_CACHE.exists():
            try:
                df = pd.read_parquet(_SQL_CACHE)
                print(f"SQL cache hit : {len(df)} bimanual episodes from {_SQL_CACHE}")
                print(f"              Delete to refresh.")
            except Exception:
                df = None

        if df is None:
            print(
                f"Querying SQL for robot_name IN {BIMANUAL_ROBOT_NAMES} ...",
                flush=True,
            )
            try:
                from sqlalchemy import bindparam, text
                from egomimic.utils.aws.aws_sql import create_default_engine

                engine = create_default_engine()
                with engine.connect() as conn:
                    result = conn.execute(
                        text(
                            "SELECT * FROM app.episodes"
                            " WHERE robot_name IN :names"
                            "   AND is_deleted IS NOT TRUE"
                        ).bindparams(bindparam("names", expanding=True)),
                        {"names": list(BIMANUAL_ROBOT_NAMES)},
                    )
                    df = pd.DataFrame(result.fetchall(), columns=result.keys())

                print(f"SQL query complete: {len(df)} bimanual episodes.", flush=True)
                _SQL_CACHE.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(_SQL_CACHE, index=False)
                print(f"SQL cache saved to {_SQL_CACHE}")
            except Exception as exc:
                logger.warning(
                    "SQL database not accessible (%s); falling back to zarr scan.", exc
                )
                return super().resolve(filters=filters)

        # --- 3. Intersect SQL result with local folders; log missed episodes ---
        sql_hashes = set(df["episode_hash"])
        in_sql       = {h for h in local_hashes if h in sql_hashes}
        local_missed = {h for h in local_hashes if h not in sql_hashes}
        not_local    = sql_hashes - set(local_hashes)

        print(f"SQL bimanual : {len(sql_hashes)}")
        # print(f"local in SQL : {len(in_sql)}/{len(local_hashes)}")
        print(f"not local    : {len(not_local)} SQL episodes not present in local folder")
        if local_missed:
            print(f"local missed : {len(local_missed)} local episodes not found in SQL bimanual cache")
            for h in sorted(local_missed):
                logger.debug("  missed episode: %s", h)

        # --- 4. Apply extra filter lambdas against SQL metadata rows ---
        df_local = df[df["episode_hash"].isin(in_sql)].copy()

        filtered: list[tuple[str, str]] = []
        for _, row in df_local.iterrows():
            episode_hash = row["episode_hash"]
            metadata = row.to_dict()
            if self._local_filters_match(metadata, episode_hash, filters):
                filtered.append((str(local_hashes[episode_hash]), episode_hash))

        filtered.sort(key=lambda x: x[1])
        if self.debug:
            filtered = filtered[:10]

        logger.info(
            "SQLLocalEpisodeResolver: %d/%d local episodes passed filters.",
            len(filtered), len(local_hashes),
        )

        valid_folder_names = {h for _, h in filtered}
        if not valid_folder_names:
            logger.warning(
                "No episodes matched filters in %s.", self.folder_path
            )
            return {}

        print(f"loading      : {len(valid_folder_names)} episodes")
        return self._load_zarr_datasets(
            search_path=self.folder_path, valid_folder_names=valid_folder_names
        )


BIMANUAL_ROBOTS = {
    "aria_bimanual": Aria,
    "mecka_bimanual": Mecka,
    "scale_bimanual": Scale,
    "scale": Scale,  
}

def head_to_world(pts_head: np.ndarray, head_pose_7: np.ndarray) -> np.ndarray:
    """ Transform points from head frame to world frame using the head pose.
        head_pose_7 is [x, y, z, qw, qx, qy, qz]
    """
    t   = head_pose_7[:3].numpy() if hasattr(head_pose_7, 'numpy') else np.asarray(head_pose_7[:3])
    q   = head_pose_7[3:7].numpy() if hasattr(head_pose_7, 'numpy') else np.asarray(head_pose_7[3:7])
    pts = pts_head.numpy() if hasattr(pts_head, 'numpy') else np.asarray(pts_head)
    w, qv = float(q[0]), q[1:4]          # wxyz split
    t1 = np.cross(qv, pts)               # qv × v,  (N, 3)
    t2 = np.cross(qv, t1)                # qv × (qv × v)
    return pts + 2 * w * t1 + 2 * t2 + t


class EgoVerseDataset(torch.utils.data.Dataset):
    """Wrapper around MultiDataset for EgoVerse bimanual episodes.

    Args:
        local_dir: Path to the zarr dataset directory.
        transform_mode: Mode string passed to ``Aria.get_transform_list``.
            Defaults to ``"keypoints_headframe_quat"``.
        split: Which split to load — ``"train"``, ``"val"``, or ``"all"``
            (no split). Episodes are sorted, shuffled with a fixed seed (42),
            then the first ``valid_ratio`` fraction goes to val and the rest
            to train — identical to ``split_dataset_names``. Defaults to
            ``"all"``.
        valid_ratio: Fraction of episodes reserved for validation. At least 1
            episode is always placed in valid when ``valid_ratio > 0``. Ignored
            when ``split="total"``. Defaults to ``0.2``.
        robots: List of robot name strings to include, e.g.
            ``["aria_bimanual", "mecka_bimanual"]``. Must be keys of
            ``BIMANUAL_ROBOTS``. Defaults to all robots in ``BIMANUAL_ROBOTS``.
    """

    def __init__(
        self,
        local_dir: str | Path = ".egoverse",
        transform_mode: str = "keypoints_headframe_quat",
        split: str = "all",
        valid_ratio: float = 0.2,
        robots: Optional[list[str]] = None,
        chunk_size: int = 1,
        stride: Optional[int] = None,
    ) -> None:
        if split not in ("train", "val", "all"):
            raise ValueError(f"split must be 'train', 'val', or 'all', got {split!r}")
        _mode = {"val": "valid", "all": "total"}.get(split, split)
        self.local_dir = Path(local_dir)
        self.transform_mode = transform_mode
        self.split = split
        self.valid_ratio = valid_ratio
        self.chunk_size = chunk_size

        if robots is None:
            self.robots = list(BIMANUAL_ROBOTS.keys())
        else:
            if len(robots) != len(set(robots)):
                raise ValueError(f"Duplicate robot names in {robots}")
            unknown = set(robots) - BIMANUAL_ROBOTS.keys()
            if unknown:
                raise ValueError(f"Unknown robot names: {unknown}. Valid: {list(BIMANUAL_ROBOTS)}")
            self.robots = list(robots)

        reference_cls = BIMANUAL_ROBOTS[self.robots[0]]
        key_map = reference_cls.get_keymap(mode="keypoints")
        transform_list = reference_cls.get_transform_list(mode=transform_mode, chunk_length=chunk_size)
        for t in transform_list:
            if isinstance(t, DeleteKeys):
                t.keys_to_delete = [k for k in t.keys_to_delete
                                     if k not in ("obs_head_pose", "obs_head_pose_ypr")]
            if stride is not None and isinstance(t, InterpolatePose):
                t.stride = stride

        resolver = SQLLocalEpisodeResolver(
            folder_path=self.local_dir,
            key_map=key_map,
            transform_list=transform_list,
        )

        self._ds = MultiDataset._from_resolver(
            resolver, filters=None, mode=_mode, valid_ratio=valid_ratio
        )

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        return self._ds[idx]

    @property
    def datasets(self):
        return self._ds.datasets


if __name__ == "__main__":
    import time
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # --- Dataset loading test ---
    start_time = time.time()
    ds = EgoVerseDataset(
        local_dir="/mnt/beegfs/datasets/egoverse",
        split="all",
        chunk_size=100,
        robots=["aria_bimanual", "mecka_bimanual", "scale_bimanual"], # robots in sql data
        stride=3,
    )
    end_time = time.time()
    print(f"Time to create EgoVerseDataset: {end_time - start_time:.2f} seconds")
    # print(ds.datasets.keys())
    print(f"  Episodes : {len(ds.datasets.keys())}")
    print(f"  Frames   : {len(ds)}")
    print(f"  Robots   : {set(z.embodiment for z in ds.datasets.values())}") # robots in local meta, which may differ from SQL

    
