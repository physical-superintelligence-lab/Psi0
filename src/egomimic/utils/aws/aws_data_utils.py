from __future__ import annotations

import os
from pathlib import Path

import boto3
import cloudpathlib
from boto3.s3.transfer import TransferConfig
from botocore.config import Config


def _uses_r2_endpoint(endpoint_url: str | None) -> bool:
    return bool(endpoint_url and "r2.cloudflarestorage.com" in endpoint_url)


def load_env(path="./src/egomimic/utils/aws/.egoverse_env"):
    p = Path(path).expanduser()
    if not p.exists():
        raise ValueError(
            f"Env file {p} does not exist, run ./egomimic/utils/aws/setup_secret.sh"
        )
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'").strip('"'))


def get_cloudpathlib_s3_client():
    load_env()
    endpoint_url = os.environ.get("R2_ENDPOINT_URL")
    r2_access_key_id = os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get(
        "AWS_ACCESS_KEY_ID"
    )
    r2_secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get(
        "AWS_SECRET_ACCESS_KEY"
    )
    r2_session_token = os.environ.get("R2_SESSION_TOKEN") or os.environ.get(
        "AWS_SESSION_TOKEN"
    )
    if _uses_r2_endpoint(endpoint_url):
        r2_session_token = None
    s3_boto3_session = boto3.session.Session(
        region_name="auto",
        aws_access_key_id=r2_access_key_id,
        aws_secret_access_key=r2_secret_access_key,
        aws_session_token=r2_session_token,
    )

    s3_client = cloudpathlib.S3Client(
        endpoint_url=endpoint_url,
        boto3_session=s3_boto3_session,
    )
    for key in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
    ):
        os.environ.pop(key, None)
    return s3_client


def get_boto3_s3_client():
    load_env()
    endpoint_url = os.environ.get("R2_ENDPOINT_URL")
    access_key_id = os.environ["R2_ACCESS_KEY_ID"]
    secret_access_key = os.environ["R2_SECRET_ACCESS_KEY"]
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name="auto",  # R2 ignores region; "auto" is common
        config=Config(signature_version="s3v4"),
    )
    return s3


def s3_sync_to_local(bucket: str, key_prefix: str, local_dir: str | Path) -> None:
    """
    Rough equivalent of: aws s3 sync s3://bucket/key_prefix/ local_dir/
    Downloads all objects under key_prefix into local_dir, preserving subpaths.
    Skips download if local file exists and size matches S3 object's size.
    """
    key_prefix = key_prefix.lstrip("/")
    if key_prefix and not key_prefix.endswith("/"):
        key_prefix += "/"

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # Create the client inside the function so this works cleanly on Ray workers.
    s3 = get_boto3_s3_client()

    config = TransferConfig(
        max_concurrency=16,
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        use_threads=True,
    )

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue

            rel = key[len(key_prefix) :] if key_prefix else key
            dest = local_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)

            if dest.exists() and dest.stat().st_size == obj["Size"]:
                continue

            s3.download_file(bucket, key, str(dest), Config=config)


def upload_dir_to_s3(
    local_dir: str, bucket: str, prefix: str = "", concurrency: int = 32
):
    s3 = get_boto3_s3_client()
    cfg = TransferConfig(
        max_concurrency=concurrency,
        multipart_threshold=64 * 1024 * 1024,  # 64MB
        multipart_chunksize=64 * 1024 * 1024,  # 64MB
        use_threads=True,
    )

    local_dir = Path(local_dir).resolve()
    prefix = prefix.strip("/")

    for root, _, files in os.walk(local_dir):
        rootp = Path(root)
        for f in files:
            lp = rootp / f
            rel = lp.relative_to(local_dir).as_posix()
            key = f"{prefix}/{rel}" if prefix else rel
            s3.upload_file(str(lp), bucket, key, Config=cfg)
            print(f"Uploaded files to S3 {prefix}")
