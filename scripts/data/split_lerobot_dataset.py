"""Split a LeRobot v2.1 dataset into <src>-train / <src>-val by holding out N random
episodes per task. Episodes are re-indexed 0..N-1 in each split (parquet episode_index /
index columns, file names, meta/episodes*.jsonl, info.json); tasks.jsonl and the global
stats files are copied unchanged so both splits normalize exactly like the source.

usage: python scripts/data/split_lerobot_dataset.py SRC_DIR [--per-task 2] [--seed 0]
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def write_jsonl(p, rows):
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def pick_val(episodes, per_task, seed):
    groups = {}
    for ep in sorted(episodes, key=lambda e: e["episode_index"]):
        groups.setdefault(repr(ep["tasks"]), []).append(ep["episode_index"])
    rng = random.Random(seed)
    val = []
    for key in sorted(groups):
        eps = groups[key]
        assert len(eps) > per_task, f"task {key} has only {len(eps)} episodes"
        val.extend(rng.sample(eps, per_task))
    return sorted(val)


def write_split(src, dst, old_eps, info, episodes, ep_stats):
    assert not dst.exists(), f"{dst} exists"
    (dst / "meta").mkdir(parents=True)
    chunks = info["chunks_size"]
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    by_old = {e["episode_index"]: e for e in episodes}
    stats_by_old = {e["episode_index"]: e for e in ep_stats}
    new_eps, new_stats, start = [], [], 0
    for new, old in enumerate(old_eps):
        t = pq.read_table(src / info["data_path"].format(episode_chunk=old // chunks, episode_index=old))
        n = t.num_rows
        for col, vals in [("episode_index", [new] * n), ("index", list(range(start, start + n)))]:
            i = t.schema.get_field_index(col)
            t = t.set_column(i, t.schema.field(i), pa.array(vals, type=pa.int64()))
        out = dst / info["data_path"].format(episode_chunk=new // chunks, episode_index=new)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, out)
        for vk in video_keys:
            fmt = dict(episode_chunk=old // chunks, episode_index=old, video_key=vk)
            vout = dst / info["video_path"].format(**{**fmt, "episode_chunk": new // chunks, "episode_index": new})
            vout.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / info["video_path"].format(**fmt), vout)
        e = dict(by_old[old], episode_index=new, source_episode_index=old)
        assert e["length"] == n, (old, e["length"], n)
        if "dataset_from_index" in e:  # source convention: inclusive end
            e["dataset_from_index"], e["dataset_to_index"] = start, start + n - 1
        new_eps.append(e)
        new_stats.append(dict(stats_by_old[old], episode_index=new))
        start += n

    for f in (src / "meta").iterdir():
        if f.name not in ("info.json", "episodes.jsonl", "episodes_stats.jsonl"):
            shutil.copy2(f, dst / "meta" / f.name)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    write_jsonl(dst / "meta/episodes.jsonl", new_eps)
    write_jsonl(dst / "meta/episodes_stats.jsonl", new_stats)
    new_info = dict(info, total_episodes=len(old_eps), total_frames=start,
                    total_videos=len(old_eps) * len(video_keys),
                    total_chunks=(len(old_eps) - 1) // chunks + 1,
                    splits={"train": f"0:{len(old_eps)}"})
    json.dump(new_info, open(dst / "meta/info.json", "w"), indent=4)
    print(f"{dst}: {len(old_eps)} episodes, {start} frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--per-task", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    src = a.src.resolve()
    info = json.load(open(src / "meta/info.json"))
    episodes = read_jsonl(src / "meta/episodes.jsonl")
    ep_stats = read_jsonl(src / "meta/episodes_stats.jsonl")
    val = pick_val(episodes, a.per_task, a.seed)
    train = sorted(set(e["episode_index"] for e in episodes) - set(val))
    print(f"val episodes ({len(val)}): {val}")
    write_split(src, src.with_name(src.name + "-train"), train, info, episodes, ep_stats)
    write_split(src, src.with_name(src.name + "-val"), val, info, episodes, ep_stats)


if __name__ == "__main__":
    main()
