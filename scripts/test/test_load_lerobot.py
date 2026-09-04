import json
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

DATASET_ROOT = Path(".data/psix_he_anno_0512/manual_label_v1_0512_lerobot/g1")

# Load meta info
with open(DATASET_ROOT / "meta" / "info.json") as f:
    info = json.load(f)
print("=== Dataset info ===")
print(f"  total_episodes : {info.get('total_episodes')}")
print(f"  total_frames   : {info.get('total_frames')}")
print(f"  fps            : {info.get('fps')}")
print()

# Load first parquet file
parquet_file = sorted((DATASET_ROOT / "data" / "chunk-000").glob("episode_*.parquet"))[0]
print(f"=== Loading {parquet_file} ===")
table = pq.read_table(parquet_file)
df = table.to_pandas()

print(f"  rows : {len(df)}")
print(f"  columns ({len(df.columns)}):")
for col in df.columns:
    val = df[col].iloc[0]
    arr = np.array(val) if not isinstance(val, (int, float, str, bool)) else val
    if isinstance(arr, np.ndarray):
        print(f"    {col:<40s}  numpy  shape={arr.shape}  dtype={arr.dtype}")
    else:
        print(f"    {col:<40s}  {type(val).__name__:<8s}  sample={repr(val)[:60]}")
