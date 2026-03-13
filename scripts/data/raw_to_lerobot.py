import argparse
import cProfile
import io
import json
import logging
import lzma
import math
import os
import pstats
import re
import shutil
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import imageio.v3 as iio
import numpy as np
# import open3d as o3d
import pandas as pd
import pyarrow.parquet as pq
from datasets import Array2D, Dataset, Features, Sequence, Value
from datasets.utils.logging import disable_progress_bar, set_verbosity_error
from huggingface_hub import create_repo, create_tag, upload_large_folder
from tqdm import tqdm

CODE_VERSION = "v2.1"
FPS = 30

# disable_progress_bar()
set_verbosity_error()

# Silence pyarrow/parquet messages
logging.getLogger("pyarrow").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)

def detect_robot_type(ep_dir: Path) -> str:
    return HE2LeRobotConverter.get_robot_type(ep_dir)


@dataclass
class InfoDict:
    codebase_version: str
    robot_type: str
    total_episodes: int
    total_frames: int
    total_tasks: int
    total_videos: int
    total_chunks: int
    chunks_size: int
    fps: int
    data_path: str
    video_path: str
    features: Dict[str, Any]

def load_done_episodes(meta_dir: Path) -> set[int]:
    done = set()
    stats_file = meta_dir / "episodes_stats.jsonl"
    if stats_file.exists():
        with open(stats_file, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add(rec["episode_index"])
                except Exception:
                    continue
    return done

def episode_complete(ep_index: int, work_dir: Path, chunks_size: int, done_eps: set[int]) -> bool:
    chunk_id = ep_index // chunks_size
    data_ok = (work_dir / "data" / f"chunk-{chunk_id:03d}" / f"episode_{ep_index:06d}.parquet").exists()
    vid_ok  = (work_dir / "videos" / f"chunk-{chunk_id:03d}" / "egocentric" / f"episode_{ep_index:06d}.mp4").exists()
    stats_ok = ep_index in done_eps
    return data_ok and vid_ok and stats_ok

def append_jsonl_line_atomic(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n"
    fd = os.open(str(path), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
    try:
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            try:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_EX)
            except Exception:
                pass
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
    finally:
        pass

def safe_write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            json.dump(r, f, separators=(",", ":"), ensure_ascii=False)
            f.write("\n")
    os.replace(tmp, path)

def read_json_list(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    assert isinstance(data, list), f"Expected a list in {path}, got {type(data)}"

    return data


def iter_tasks(data_root: Path, tasks:list[str]=None) -> Iterator[Tuple[str, Path, str, str]]:
    if tasks is None:
        tasks = []
    for cat_dir in sorted(
        [p for p in data_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()
    ):
        for task_dir in sorted([p for p in cat_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
            # skip processed tasks
            if task_dir.name not in tasks:
                continue
            yield f"{cat_dir.name}/{task_dir.name}", task_dir, cat_dir.name, task_dir.name


class HE2LeRobotConverter:
    def __init__(self):
        self.features = Features(
            {
                # "observation.depth.egocentric": Array2D(dtype="float32", shape=(480, 640)),
                # "observation.lidar": Sequence(Sequence(Value("float32"))),
                # "observation.imu.quaternion": Sequence(Value("float32")),
                # "observation.imu.accelerometer": Sequence(Value("float32")),
                # "observation.imu.gyroscope": Sequence(Value("float32")),
                # "observation.imu.rpy": Sequence(Value("float32")),
                # "observation.odometry.position": Sequence(Value("float32")),
                # "observation.odometry.velocity": Sequence(Value("float32")),
                # "observation.odometry.rpy": Sequence(Value("float32")),
                # "observation.odometry.quat": Sequence(Value("float32")),
                # "observation.arm_joints": Sequence(Value("float32")),
                # "observation.leg_joints": Sequence(Value("float32")),
                # "observation.hand_joints": Sequence(Value("float32")),
                # "observation.prev_rpy": Sequence(Value("float32")),
                # "observation.prev_height": Value("float32"),
                "states": Sequence(Value("float32")),
                # "observation.tactile": Array2D(dtype="float32", shape=(-1, 4)),
                "action": Sequence(Value("float32")),
                "timestamp": Value("float32"),
                "frame_index": Value("int64"),
                "episode_index": Value("int64"),
                "index": Value("int64"),
                "task_index": Value("int64"),
                "next.done": Value("bool"),
            }
        )

        self.task_description_dict: Dict[str, str] = {}
        self.kept_records: List[Tuple[int, int, Path, str, str]] = []
        self.lengths_by_episode: Dict[int, int] = {}
        self.tasks_meta: Dict[int, Dict[str, Any]] = {}
        self.chunks_size: int = 1000

    @staticmethod
    def get_robot_type(ep_dir: Path) -> str:
        data_list = read_json_list(ep_dir / "data.json")
        if not data_list:
            return "h1"
        for frame in data_list:
            st = frame.get("states", {})
            if isinstance(st, dict) and "robot_type" in st:
                try:
                    return str(st["robot_type"]).lower()
                except Exception:
                    return "h1"
            if "robot_type" in frame:
                try:
                    return str(frame["robot_type"]).lower()
                except Exception:
                    return "h1"
        return "h1"

    def load_depth(self, depth_lzma_path: Path) -> Optional[np.ndarray]:
        try:
            with open(depth_lzma_path, "rb") as f:
                decompressed = lzma.decompress(f.read())
            depth_u16 = np.frombuffer(decompressed, dtype=np.uint16).reshape((480, 640))
            return depth_u16.astype(np.float32)
        except Exception:
            return None

    def load_lidar(self, pcd_path: Path) -> Optional[np.ndarray]:
        try:

            def pad_to_six(m):
                whole, dec = m.group("whole"), m.group("dec")
                return f"{whole}.{dec.ljust(6, '0')}"

            pcd_path_str = re.sub(r"(?P<whole>\d+)\.(?P<dec>\d{1,6})(?=\.pcd$)", pad_to_six, str(pcd_path))
            pcd = o3d.io.read_point_cloud(pcd_path_str)
            pts = np.asarray(pcd.points, dtype=np.float32)
            pts = pts[~np.all(pts == 0, axis=1)]
            if pts.size == 0:
                return None
            return pts
        except Exception:
            return None

    def load_tactile(self, state: Dict[str, Any]) -> List[Any]:
        sensor_ids, values = [], []
        if state and isinstance(state, dict) and "hand_pressure_state" in state and state["hand_pressure_state"] is not None:
            for sensor in state["hand_pressure_state"]:
                sid = int(sensor.get("sensor_id", -1))
                raw_vals = [float(x) for x in sensor.get("usable_readings", [])]
                if len(raw_vals) < 4:
                    raw_vals += [math.nan] * (4 - len(raw_vals))
                elif len(raw_vals) > 4:
                    raw_vals = raw_vals[:4]
                sensor_ids.append(sid)
                values.append(raw_vals)
        return values

    def build_obs(self, prev_rpy_height:Dict[str, Any], frame: Dict[str, Any], depth_arr: np.ndarray, pts: np.ndarray) -> Dict[str, Any]:
        states = frame.get("states", {}) or {}

        imu_in = states.get("imu", {}) if isinstance(states, dict) else {}
        imu = {"quaternion": [np.nan] * 4, "accelerometer": [np.nan] * 3, "gyroscope": [np.nan] * 3, "rpy": [np.nan] * 3}
        for k, n in (("quaternion", 4), ("accelerometer", 3), ("gyroscope", 3), ("rpy", 3)):
            if k in imu_in and isinstance(imu_in[k], (list, tuple)):
                vals = [float(x) for x in imu_in[k][:n]]
                imu[k] = vals + [0.0] * (n - len(vals))

        odo_in = states.get("odometry", {}) if isinstance(states, dict) else {}
        odometry = {"position": [np.nan] * 3, "velocity": [np.nan] * 3, "rpy": [np.nan] * 3, "quat": [np.nan] * 4}
        for k, n in (("position", 3), ("velocity", 3), ("rpy", 3), ("quat", 4)):
            if k in odo_in and isinstance(odo_in[k], (list, tuple)):
                vals = [float(x) for x in odo_in[k][:n]]
                odometry[k] = vals + [0.0] * (n - len(vals))

        arm_joints = [float(x) for x in states.get("arm_state", [])]
        leg_joints = [float(x) for x in states.get("leg_state", [])]
        hand_joints = [float(x) for x in states.get("hand_state", [])]
        torso_rpy = [float(x) for x in prev_rpy_height["torso_rpy"]]
        torso_height = [float(prev_rpy_height["torso_height"])]

        tactile = self.load_tactile(states)

        # return {
        #     # "observation.depth.egocentric": depth_arr,
        #     # "observation.lidar": pts,
        #     # "observation.imu.quaternion": imu["quaternion"],
        #     # "observation.imu.accelerometer": imu["accelerometer"],
        #     # "observation.imu.gyroscope": imu["gyroscope"],
        #     # "observation.odometry.position": odometry["position"],
        #     # "observation.odometry.velocity": odometry["velocity"],
        #     # "observation.odometry.rpy": odometry["rpy"],
        #     # "observation.odometry.quat": odometry["quat"],
        #     "observation.prev_height": prev_rpy_height["torso_height"],
        #     "observation.prev_rpy": prev_rpy_height["torso_rpy"], 
        #     "observation.leg_joints": leg_joints,
        #     "observation.arm_joints": arm_joints,
        #     "observation.hand_joints": hand_joints,
        #     # "observation.tactile": tactile,
        # }
        return {
            "states": hand_joints + arm_joints + torso_rpy + torso_height
        }


    def build_act(self, frame: Dict[str, Any]) -> List[float]:
        hand_joints: List[float] = []
        actions = frame.get("actions", {}) or {}
        r = actions.get("right_angles")
        l = actions.get("left_angles")

        def convert_h1_hand(qpos: List[float]) -> List[float]:
            out = [1.7 - qpos[i] for i in [4, 6, 2, 0]]
            out.append(1.2 - qpos[8])
            out.append(0.5 - qpos[9])
            return [float(x) for x in out]

        if r is not None and l is not None:
            if len(r) == 12 and len(l) == 12:
                hand_joints.extend(convert_h1_hand(l))
                hand_joints.extend(convert_h1_hand(r))
            elif len(r) == 7 and len(l) == 7:
                hand_joints.extend([float(x) for x in l])
                hand_joints.extend([float(x) for x in r])

        sq = actions.get("sol_q")
        if sq is not None:
            body_joints = [float(x) for x in sq]

        leg_joints = body_joints[0:15]
        arm_joints = body_joints[15:29]

        rpy = frame["actions"]["torso_rpy"]
        height = frame["actions"]["torso_height"]
        rpy = [float(x) for x in rpy]
        height = [float(height)]

        torso_vx = actions.get("torso_vx")
        torso_vy = actions.get("torso_vy")
        torso_vyaw = actions.get("torso_vyaw")
        torso_dyaw = actions.get("torso_dyaw")
        target_yaw = actions.get("target_yaw")
        torso_vx = [float(torso_vx)]
        torso_vy = [float(torso_vy)]
        torso_vyaw = [float(torso_vyaw)]
        torso_dyaw = [float(torso_dyaw)]
        target_yaw = [float(target_yaw)]
        # assert False, "check here dyaw or target yaw"
        return hand_joints + arm_joints + rpy + height + torso_vx + torso_vy + torso_vyaw + target_yaw # torso_dyaw # 36

    def make_one_episode(
        self,
        task_index: int,
        episode_index: int,
        episode_dir: Path,
        out_base: Path,
        chunks_size: int,
    ) -> Tuple[int, int, Dict[str, Any]]:
        # pr = cProfile.Profile()
        # pr.enable()

        try:
            chunk_path = out_base / f"chunk-{episode_index // chunks_size:03d}"
            chunk_path.mkdir(parents=True, exist_ok=True)
            parquet_path = chunk_path / f"episode_{episode_index:06d}.parquet"

            vid_chunk_dir = out_base.parent / "videos" / f"chunk-{episode_index // chunks_size:03d}" / "egocentric"
            vid_chunk_dir.mkdir(parents=True, exist_ok=True)
            vid_path = vid_chunk_dir / f"episode_{episode_index:06d}.mp4"

            data_list = read_json_list(episode_dir / "data.json")
            assert len(data_list) > 0, f"data.json malformed in {episode_dir}"

            def safe_path(episode_dir, f, key):
                p = f.get(key)
                return (episode_dir / p).resolve() if p else None

            rgb_paths   = [safe_path(episode_dir, f, "image") for f in data_list]
            depth_paths = [safe_path(episode_dir, f, "depth") for f in data_list]
            lidar_paths = [safe_path(episode_dir, f, "lidar") for f in data_list]

            def iter_depths():
                for p in depth_paths:
                    yield self.load_depth(p) if p else np.full((480, 640), np.nan, np.float32)

            def iter_lidars():
                for p in lidar_paths:
                    yield self.load_lidar(p) if p else np.zeros((0, 3), np.float32)



            rows: List[Dict[str, Any]] = []
            prev_rpy_height = {
                "torso_rpy": [0.0,0.0,0.0],
                "torso_height": 0.75,
            }

            for i, (frame, depth_arr, lidar_pts) in enumerate(zip(data_list, iter_depths(), iter_lidars())):
                obs = self.build_obs(prev_rpy_height, frame, depth_arr, lidar_pts)
                act = self.build_act(frame)

                rows.append(
                    {
                        **obs,
                        "action": act,
                        "timestamp": i * (1.0 / FPS),
                        "frame_index": i,
                        "episode_index": episode_index,
                        "index": i,  # TODO: global index if needed
                        "task_index": task_index,
                        "next.done": (i == len(data_list) - 1),
                    }
                )

                prev_rpy_height["torso_rpy"] = frame["actions"]["torso_rpy"]
                prev_rpy_height["torso_height"] = frame["actions"]["torso_height"]

            assert rows, f"No valid rows in episode {episode_index}"

            stats = None
            for r in rows:
                a = np.array(r["action"], dtype=np.float32)
                if stats is None:
                    stats = {"min": a.copy(), "max": a.copy(), "sum": a.copy(), "sumsq": a**2, "count": 1}
                else:
                    stats["min"] = np.minimum(stats["min"], a)
                    stats["max"] = np.maximum(stats["max"], a)
                    stats["sum"] += a
                    stats["sumsq"] += a**2
                    stats["count"] += 1

            assert stats is not None, f"No valid actions in episode {episode_index}"
            stats = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in stats.items()}

            tmp_dir = (out_base / f"_tmp_ep_{episode_index:06d}")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            parquet_tmp = tmp_dir / "episode.parquet"
            video_tmp   = tmp_dir / "episode.mp4"

            ds = Dataset.from_list(rows, features=self.features)
            ds.to_parquet(str(parquet_tmp)) 

            def frame_iter():
                for p in rgb_paths:
                    yield iio.imread(p)

            iio.imwrite(video_tmp, list(frame_iter()), fps=FPS, codec="libx264")
            # iio.imwrite(
            #     video_tmp, 
            #     list(frame_iter()), 
            #     fps=FPS, 
            #     codec="libx264",
            #     output_params=[
            #         "-g", "1",              # GOP size = 1，每帧都是关键帧
            #         "-qp", "0",             # 量化参数 = 0，完全无损
            #         "-preset", "medium",    # 编码速度（可选：fast/medium/slow）
            #         "-pix_fmt", "yuv444p"   # 完整色彩采样
            #     ]
            # )
            os.replace(parquet_tmp, parquet_path)
            os.replace(video_tmp, vid_path)
            shutil.rmtree(tmp_dir)

            action_mean = (np.array(stats["sum"]) / stats["count"]).tolist()
            action_std = (
                np.sqrt(np.maximum(np.array(stats["sumsq"]) / stats["count"] - np.square(np.array(stats["sum"]) / stats["count"]), 0))
            ).tolist()

            episode_stats = {
                "episode_index": episode_index,
                "stats": {
                    "action": {
                        "min": stats["min"],
                        "max": stats["max"],
                        "mean": action_mean,
                        "std": action_std,
                        "count": [len(rows)],
                    },
                    "timestamp": {
                        "min": [0.0],
                        "max": [(len(rows) - 1) / FPS],
                        "mean": [((len(rows) - 1) / 2) / FPS],
                        "std": [len(rows) / (2 * FPS * math.sqrt(3))],
                        "count": [len(rows)],
                    },
                },
            }
            meta_dir = out_base.parent / "meta"
            meta_dir.mkdir(parents=True, exist_ok=True)
            append_jsonl_line_atomic(meta_dir / "episodes_stats.jsonl", episode_stats)
        except Exception as e:
            print(f"Error processing episode {episode_index} in {episode_dir}: {e}")
            exit(-1)
            # return episode_index, 0, {}


        # pr.disable()
        # s = io.StringIO()
        # ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
        # ps.print_stats(40)
        # print(f"PROFILE for episode {episode_index}\n", s.getvalue())
        return episode_index, len(rows), stats

    def run(self, data_root: Path, work_dir: Path, chunks_size: int, num_workers: int, robot_type: str, task:str|None=None):
        self.chunks_size = chunks_size
        tdd = Path(__file__).parent / "task_description_dict.json"
        assert tdd.is_file(), "task_description_dict.json not found"
        self.task_description_dict = json.load(open(tdd))

        # --- existing progress detection ---
        data_dir = work_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        # existing = {p.stem for p in data_dir.rglob("episode_*.parquet")}

        self.episode_sources: list[tuple[int, Path, str, str, str]] = []
        task_index = 0
        self.tasks_meta = {}
        ep_index = 0

        all_ep_dirs: List[Tuple[int, Path, str, str, str]] = []
        for task_name, task_dir, cat_name, leaf_name in iter_tasks(data_root, tasks=[task] if task else []):
            print("\n" + "="*40)
            print(f"Task:        {task_name}")
            print(f"Directory:   {task_dir}")
            print(f"Category:    {cat_name}")
            print(f"Leaf:        {leaf_name}")
            print("="*40 + "\n")
            desc = self.task_description_dict.get(leaf_name, "")
        
            assert desc, f"Task description is empty for {leaf_name}"
            ep_dirs = [p for p in task_dir.iterdir() if p.is_dir() and re.match(r"episode_\d+", p.name)]
            ep_dirs = sorted(ep_dirs, key=lambda p: int(re.findall(r"\d+", p.name)[0]))
            for ep_dir in ep_dirs:
                all_ep_dirs.append((task_index, ep_dir, desc, task_name, cat_name))
            self.tasks_meta[task_index] = {"name": task_name, "category": cat_name, "description": desc}
            task_index += 1

        print(f"Detecting robot types for {len(all_ep_dirs)} episodes...")
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            ep_dirs_only = [t[1] for t in all_ep_dirs]
            robot_types = list(
                tqdm(
                    ex.map(detect_robot_type, ep_dirs_only),
                    total=len(ep_dirs_only),
                    desc="Scanning robot types",
                    unit="ep",
                )
            )

        meta_dir = work_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        done_eps = load_done_episodes(meta_dir)


        filtered = []
        for (meta, rtype) in zip(all_ep_dirs, robot_types):
            task_idx, ep_dir, desc, tname, cat_name = meta
            if robot_type == "both" or rtype == robot_type:
                filtered.append((task_idx, ep_dir, ep_index, desc, tname))
                ep_index += 1

        filtered = [
            (task_idx, ep_dir, ep_index, desc, tname)
            for (task_idx, ep_dir, ep_index, desc, tname) in filtered
            if not episode_complete(ep_index, work_dir, chunks_size, done_eps)
        ]

        self.episode_sources = filtered

        if not self.episode_sources:
            print(f"No episodes matched robot type '{robot_type}'.")
            return

        print(f"Resuming: {len(filtered)} new episodes (skipped {len(done_eps)})")

        total = len(self.episode_sources)
        data_stats: list[tuple[int, int, dict[str, any]]] = []
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [
                ex.submit(self.make_one_episode, task_idx, i, ep_dir, data_dir, chunks_size)
                for i, (task_idx, ep_dir, _, _, _) in enumerate(self.episode_sources)
            ]
            for fut in tqdm(as_completed(futures), total=total, desc="Processing new episodes", unit="ep"):
                ep_idx, n_frames, stats = fut.result()
                if n_frames <= 0:
                    continue
                data_stats.append((ep_idx, n_frames, stats))

                # --- incremental meta update ---
                self.lengths_by_episode[ep_idx] = n_frames
                self.num_episodes = len(self.lengths_by_episode)
                self.total_frames = sum(self.lengths_by_episode.values())

        # TODO: use resume

        # self.kept_records = sorted(kept_records, key=lambda x: x[0])
        self.lengths_by_episode = {ep_idx: n for ep_idx, n, _ in data_stats}
        self.num_episodes = len(self.episode_sources)
        self.total_frames = sum(self.lengths_by_episode.values())

        print(f"Now total episodes: {self.num_episodes}, frames: {self.total_frames}")



    def write_meta(self, out_dir: Path):
        meta_dir = out_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        dataset_cursor = 0
        ep_rows_meta = []

        for (task_idx, ep_dir, ep_index, task_dsc, tname) in sorted(self.episode_sources, key=lambda x: x[0]):
            n = self.lengths_by_episode.get(ep_index, 0)
            if n <= 0:
                continue
            ep_rows_meta.append(
                {
                    "episode_index": ep_index,
                    "tasks": [task_idx],
                    "length": n,
                    "dataset_from_index": dataset_cursor,
                    "dataset_to_index": dataset_cursor + (n - 1),
                    "robot_type": self.get_robot_type(ep_dir),
                    "instruction": task_dsc
                }
            )
            dataset_cursor += n

        episodes_df = pd.DataFrame(ep_rows_meta).sort_values("episode_index").reset_index(drop=True)

        task_rows = []
        for ti, meta in self.tasks_meta.items():
            task_rows.append(
                {
                    "task_index": ti,
                    "task": meta.get("name", f"task_{ti:04d}"),
                    "category": meta.get("category", ""),
                    "description": meta.get("description", ""),
                }
            )
        tasks_df = pd.DataFrame(task_rows).sort_values("task_index").reset_index(drop=True)

        features_meta = {
            "observation.images.egocentric": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": float(FPS),
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            # "observation.depth.egocentric": {"dtype": "float32", "shape": [480, 640], "names": ["height", "width"]},
            # "observation.lidar": {"dtype": "float32", "shape": [-1, 3]},
            # "observation.imu.quaternion": {"dtype": "float32", "shape": [4]},
            # "observation.imu.accelerometer": {"dtype": "float32", "shape": [3]},
            # "observation.imu.gyroscope": {"dtype": "float32", "shape": [3]},
            # "observation.imu.rpy": {"dtype": "float32", "shape": [3]},
            # "observation.odometry.position": {"dtype": "float32", "shape": [3]},
            # "observation.odometry.velocity": {"dtype": "float32", "shape": [3]},
            # "observation.odometry.rpy": {"dtype": "float32", "shape": [3]},
            # "observation.odometry.quat": {"dtype": "float32", "shape": [4]},
            # "observation.arm_joints": {"dtype": "float32", "shape": [-1]},
            # "observation.leg_joints": {"dtype": "float32", "shape": [-1]},
            # "observation.hand_joints": {"dtype": "float32", "shape": [-1]},
            # "observation.prev_rpy": {"dtype": "float32", "shape": [3]},
            # "observation.prev_height": {"dtype": "float32", "shape": [1]},
            "states": {"dtype": "float32", "shape": [-1]},
            # "observation.tactile": {"dtype": "float32", "shape": [-1, -1]},
            "action": {"dtype": "float32", "shape": [-1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        }

        robot_types = set(episodes_df["robot_type"].tolist()) if not episodes_df.empty else {"h1"}
        global_robot_type = list(robot_types)[0] if len(robot_types) == 1 else "mixed"

        info = InfoDict(
            codebase_version=CODE_VERSION,
            robot_type=global_robot_type,
            total_episodes=self.num_episodes,
            total_frames=self.total_frames,
            total_tasks=len(self.tasks_meta),
            total_videos=self.num_episodes,
            total_chunks=math.ceil(self.num_episodes / self.chunks_size),
            chunks_size=self.chunks_size,
            fps=FPS,
            data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            video_path="videos/chunk-{episode_chunk:03d}/egocentric/episode_{episode_index:06d}.mp4",
            features=features_meta,
        )

        (meta_dir / "info.json").write_text(json.dumps(asdict(info), indent=4))

        with open(meta_dir / "tasks.jsonl", "w") as f_tasks:
            for row in tasks_df.to_dict(orient="records"):
                json.dump(row, f_tasks)
                f_tasks.write("\n")

        with open(meta_dir / "episodes.jsonl", "w") as f_eps:
            for row in episodes_df.to_dict(orient="records"):
                json.dump(row, f_eps)
                f_eps.write("\n")

        print(f"\nWrote meta (info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl) and {self.num_episodes} episode(s) into: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--work-dir", type=str, default="_lerobot_build")
    parser.add_argument("--repo-id", type=str)
    parser.add_argument("--chunks-size", type=int, default=1000)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--repo-exist-ok", action="store_true")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count(), help="Max parallel workers (default: all CPUs)")
    parser.add_argument("--robot-type", type=str, choices=["h1", "g1", "both"], default="both",
                        help="Filter episodes by robot type (h1, g1, or both)")
    parser.add_argument("--task", type=str, default=None, help="Process only this specific task (category/task_name)")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    work_dir = Path(args.work_dir).resolve()
    if args.task:
        print(f"Processing only task: {args.task}")
        work_dir = work_dir / args.task

    for d in [work_dir / "data", work_dir / "videos", work_dir / "meta"]:
        d.mkdir(parents=True, exist_ok=True)

    pipeline = HE2LeRobotConverter()
    pipeline.run(data_root, work_dir, args.chunks_size, args.num_workers, args.robot_type, args.task)
    pipeline.write_meta(work_dir)

    if args.push:
        if not args.repo_id:
            raise ValueError("--repo-id is required when --push is set")
        create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=args.repo_exist_ok)
        upload_large_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=str(work_dir))
        create_tag(args.repo_id, tag=CODE_VERSION, repo_type="dataset")
        print(f"\n✅ Uploaded to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
