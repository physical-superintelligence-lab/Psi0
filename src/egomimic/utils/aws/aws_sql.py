import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import boto3
from botocore.config import Config as BotocoreConfig
from sqlalchemy import (
    URL,
    MetaData,
    Table,
    create_engine,
    delete,
    insert,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError

from egomimic.utils.aws.aws_data_utils import load_env

logger = logging.getLogger(__name__)
YELLOW = "\033[33m"
RESET = "\033[0m"


@dataclass
class TableRow:
    episode_hash: str
    operator: str
    lab: str
    task: str
    embodiment: str
    robot_name: str
    num_frames: int = -1  # Updateable
    task_description: str = ""
    scene: str = ""
    objects: str = ""
    processed_path: str = ""  # Updateable
    zarr_processed_path: str = ""  # Updateable
    zarr_mp4_path: str = ""  # Updateable
    processing_error: str = ""  # Updateable
    zarr_processing_error: str = ""  # Updateable
    mp4_path: str = ""  # Updateable
    is_deleted: bool = False
    is_eval: bool = False
    eval_score: float = -1
    eval_success: bool = True


def _socks5_proxy_ctx():
    """Context manager: route all sockets through SOCKS5_PROXY if set."""
    import contextlib, socket
    from urllib.parse import urlparse
    proxy_url = os.environ.get("EGOVERSE_SOCKS5")
    if not proxy_url:
        return contextlib.nullcontext()
    import socks
    p = urlparse(proxy_url)

    @contextlib.contextmanager
    def _ctx():
        # botocore.utils imports getproxies at module level and calls it inside
        # get_environ_proxies() which is what endpoint.py uses to resolve proxies.
        # Patch that reference directly so botocore sees no proxy.
        import botocore.utils as _botocore_utils
        _orig = _botocore_utils.getproxies
        _botocore_utils.getproxies = lambda: {}
        orig = socket.socket
        socks.set_default_proxy(socks.SOCKS5, p.hostname, p.port)
        socket.socket = socks.socksocket
        try:
            yield
        finally:
            socket.socket = orig
            socks.set_default_proxy()
            _botocore_utils.getproxies = _orig

    return _ctx()


_CREDS_CACHE = None  # module-level in-process cache

def _creds_cache_path():
    from pathlib import Path
    return Path(f"{os.environ.get('PSI_HOME')}/cache/egoverse/db_creds.json").expanduser()

def create_default_engine():
    global _CREDS_CACHE
    # Populate env from ~/.egoverse_env only when SECRETS_ARN is not already set.
    if not os.environ.get("SECRETS_ARN"):
        load_env()

    # Try to get credentials from Secrets Manager; fall back to disk cache.
    SECRETS_ARN = os.environ.get("SECRETS_ARN")
    if not SECRETS_ARN:
        raise RuntimeError(
            "SECRETS_ARN environment variable not set. Please run ./egomimic/utils/aws/setup_secret.sh."
        )

    cfg = None
    cache_path = _creds_cache_path()

    # Use cached credentials if available (avoids calling Secrets Manager every run).
    if cache_path.exists() and _CREDS_CACHE is None:
        try:
            _CREDS_CACHE = json.loads(cache_path.read_text())
        except Exception:
            pass

    if _CREDS_CACHE is not None:
        cfg = _CREDS_CACHE
        print("  Using cached DB credentials.", flush=True)
    else:
        print("  Fetching DB credentials from Secrets Manager...", flush=True)
        try:
            boto_cfg = BotocoreConfig(connect_timeout=30, read_timeout=60, retries={"max_attempts": 2})
            with _socks5_proxy_ctx():
                secrets = boto3.client("secretsmanager", config=boto_cfg)
                sec = secrets.get_secret_value(SecretId=SECRETS_ARN)["SecretString"]
            cfg = json.loads(sec)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cfg))
            _CREDS_CACHE = cfg
            print("  Credentials fetched and cached.", flush=True)
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach Secrets Manager and no credential cache at {cache_path}. Error: {e}"
            ) from e

    HOST = cfg.get("host", cfg.get("HOST"))
    DBNAME = cfg.get("dbname", cfg.get("DBNAME", "appdb"))
    USER = cfg.get("username", cfg.get("user", cfg.get("USER")))
    PASSWORD = cfg.get("password", cfg.get("PASSWORD"))
    PORT = cfg.get("port", 5432)

    # --- 1) connect via SQLAlchemy ---
    engine = create_engine(
        URL.create(
            "postgresql+psycopg",
            username=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            database=DBNAME,
            query={"sslmode": "require", "connect_timeout": "10"},
        ),
        pool_pre_ping=True,
    )

    return engine


def _episodes_table(engine):
    md = MetaData()
    return Table("episodes", md, autoload_with=engine, schema="app")


def add_episode(engine, episode) -> bool:
    """
    Insert one row into app.episodes.
    Raises sqlalchemy.exc.IntegrityError if the row violates a unique/PK constraint.
    """
    episodes_tbl = _episodes_table(engine)
    row = asdict(episode)

    try:
        with engine.begin() as conn:
            conn.execute(insert(episodes_tbl).values(**row))
        return True
    except IntegrityError as e:
        # Duplicate (or other constraint) → surface a clear error
        raise RuntimeError(f"Insert failed (likely duplicate episode_hash): {e}") from e


def update_episode(engine, episode: TableRow):
    """
    Update a row in a PostgreSQL table using SQLAlchemy Core (SQLAlchemy 2 compatible).

    Args:
        engine: SQLAlchemy Engine instance.
        episode (TableRow): TableRow object.
    """
    episodes_tbl = _episodes_table(engine)

    # Create a dict out of episode fields
    row = asdict(episode)
    episode_hash = row.pop("episode_hash")  # Remove episode_hash from the update values

    stmt = (
        update(episodes_tbl)
        .where(episodes_tbl.c.episode_hash == episode_hash)
        .values(**row)
    )

    with (
        engine.begin() as conn
    ):  # use engine.begin() for transactional context (SQLAlchemy 2 style)
        conn.execute(stmt)
    return True


def episode_hash_to_table_row(engine, episode_hash):
    t = _episodes_table(engine)
    fields = set(TableRow.__dataclass_fields__.keys())
    if {c.name for c in t.columns} != fields:
        raise ValueError("Schema mismatch between TableRow and app.episodes")

    stmt = select(t).where(t.c.episode_hash == episode_hash).limit(1)
    with engine.connect() as conn:
        rec = conn.execute(stmt).mappings().first()

    return None if rec is None else TableRow(**rec)


def delete_episodes(engine, episode_hashes: list[int]):
    episodes_tbl = _episodes_table(engine)
    with engine.begin() as conn:
        conn.execute(
            delete(episodes_tbl).where(episodes_tbl.c.episode_hash.in_(episode_hashes))
        )
    return True


def delete_all_episodes(engine):
    episodes_tbl = _episodes_table(engine)
    with engine.begin() as conn:
        conn.execute(delete(episodes_tbl))
    return True


def episode_table_to_df(engine):
    """
    Prints all rows in the 'episodes' table in a nicely formatted table.
    """
    metadata = MetaData()
    episodes_tbl = Table("episodes", metadata, autoload_with=engine, schema="app")

    import pandas as pd

    with engine.connect() as conn:
        result = conn.execute(select(episodes_tbl))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
        if not df.empty:
            return df
        else:
            print("No rows found in table 'episodes'.")
            return df


def episode_hash_to_timestamp_ms(timestamp_str):
    """
    Convert a string like "2026-01-12-03-47-29-664000" to UTC epoch milliseconds.
    """
    dt = datetime.strptime(timestamp_str, "%Y-%m-%d-%H-%M-%S-%f").replace(
        tzinfo=timezone.utc
    )
    return int(dt.timestamp() * 1000)


def timestamp_ms_to_episode_hash(timestamp_ms):
    """
    Convert UTC epoch milliseconds like 1769460905119 to
    "YYYY-MM-DD-HH-MM-SS-ffffff".
    """
    timestamp_ms = int(timestamp_ms)
    seconds, milliseconds = divmod(timestamp_ms, 1000)
    dt = datetime.fromtimestamp(seconds, tz=timezone.utc).replace(
        microsecond=milliseconds * 1000
    )
    return dt.strftime("%Y-%m-%d-%H-%M-%S-%f")
