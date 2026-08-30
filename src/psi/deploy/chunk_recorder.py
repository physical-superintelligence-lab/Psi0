"""Persist every predicted action chunk during a rollout.

The wire carries one action row per control tick, so the chunk the policy
actually planned never reaches the client and cannot be recovered afterwards --
diagnosing a stall meant re-running the checkpoint offline and hoping the frame
and state were reconstructed faithfully. This records the chunk where it is
produced instead, on the inference thread, behind a bounded queue so a slow disk
can never reach the control loop.

Enabled only when ``PSIX_CHUNK_DUMP_DIR`` is set; otherwise the factory returns
None and the call sites cost one attribute test per replan.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_SHARD = 200          # chunks per npz shard
_QUEUE = 512          # replans buffered before we start dropping


class ChunkRecorder:
    """Bounded-queue writer for predicted chunks. Drops rather than blocks."""

    _STOP = object()

    def __init__(self, out_dir: str | os.PathLike, shard_size: int = _SHARD):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._q: queue.Queue = queue.Queue(maxsize=_QUEUE)
        self._shard_size = int(shard_size)
        self._dropped = 0
        self._written = 0
        self._thread = threading.Thread(target=self._writer, name="chunk-recorder",
                                        daemon=True)
        self._thread.start()

    # -- producer side (inference thread) ------------------------------------
    def record(self, chunk: np.ndarray, *, chunk_id: int, replan_id: int = -1,
               infer_ms: float = 0.0, meta: dict[str, Any] | None = None) -> None:
        """Queue one predicted chunk. Never raises, never blocks."""
        try:
            item = dict(chunk=np.ascontiguousarray(chunk, dtype=np.float32),
                        chunk_id=int(chunk_id), replan_id=int(replan_id),
                        infer_ms=float(infer_ms), t_mono=time.monotonic(),
                        t_wall=datetime.now().isoformat(timespec="milliseconds"),
                        meta=meta or {})
            self._q.put_nowait(item)
        except queue.Full:
            self._dropped += 1
        except Exception:                       # telemetry must never break serving
            self._dropped += 1

    def close(self, timeout: float = 5.0) -> None:
        try:
            self._q.put_nowait(self._STOP)
        except queue.Full:
            return
        self._thread.join(timeout=timeout)

    # -- consumer side (writer thread) ---------------------------------------
    def _flush(self, buf: list[dict[str, Any]], index) -> None:
        if not buf:
            return
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        name = f"chunks_{buf[0]['chunk_id']:06d}_{stamp}.npz"
        np.savez_compressed(
            self.dir/name,
            chunk=np.stack([b["chunk"] for b in buf]),
            chunk_id=np.array([b["chunk_id"] for b in buf], np.int64),
            replan_id=np.array([b["replan_id"] for b in buf], np.int64),
            infer_ms=np.array([b["infer_ms"] for b in buf], np.float32),
            t_mono=np.array([b["t_mono"] for b in buf], np.float64),
        )
        for b in buf:
            index.write(json.dumps({k: v for k, v in b.items() if k != "chunk"},
                                   default=str) + "\n")
        index.flush()
        self._written += len(buf)

    def _writer(self) -> None:
        buf: list[dict[str, Any]] = []
        with open(self.dir/"chunks_index.jsonl", "a", encoding="utf-8") as index:
            while True:
                try:
                    item = self._q.get(timeout=1.0)
                except queue.Empty:
                    self._flush(buf, index); buf = []      # idle -> land what we have
                    continue
                if item is self._STOP:
                    self._flush(buf, index)
                    index.write(json.dumps({"event": "close", "written": self._written,
                                            "dropped": self._dropped}) + "\n")
                    return
                buf.append(item)
                if len(buf) >= self._shard_size:
                    self._flush(buf, index); buf = []


def maybe_chunk_recorder(env_var: str = "PSIX_CHUNK_DUMP_DIR") -> ChunkRecorder | None:
    """ChunkRecorder if the env var names a directory, else None."""
    d = os.environ.get(env_var, "").strip()
    if not d:
        return None
    try:
        return ChunkRecorder(d)
    except Exception as exc:                    # a bad path must not stop serving
        print(f"[chunk-recorder] disabled: {type(exc).__name__}: {exc}", flush=True)
        return None
