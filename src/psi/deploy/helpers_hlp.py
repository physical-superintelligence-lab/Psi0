"""Shared HLP-client helpers used by the mock VLA clients to talk to an HLP
server (real serve_psix_hlp.py or mock_psix_hlp_serve.py).

Keeps the mock clients' HLP integration in one place: poll the current subtask,
optionally step it, and a standalone reception test that needs NO VLA server.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import requests

from psi.deploy.helpers import convert_numpy_in_dict, numpy_serialize


def hlp_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def frame_ego_image(frame: dict, image_keys: list[str]) -> np.ndarray:
    """Current-observation egocentric RGB image (HxWx3 uint8) — what the HLP sees.

    Mirrors the clients' build_request image extraction:
      frame["observations"][-1:] -> (n_cam, C, H, W); pick the egocentric camera and
      transpose CHW -> HWC. Falls back to the first camera if none is named "ego*".
    """
    obs = frame["observations"][-1:]
    arr = obs.numpy() if hasattr(obs, "numpy") else np.asarray(obs)
    ego_idx = next((i for i, k in enumerate(image_keys) if "ego" in str(k).lower()), 0)
    img = np.asarray(arr[ego_idx]).transpose(1, 2, 0)  # (C,H,W) -> (H,W,C)
    return np.ascontiguousarray(img).astype(np.uint8)


def serialize_ego_image(frame: dict, image_keys: list[str]) -> dict:
    """Extract + numpy-serialize the ego image ONCE so the SAME blob can be reused in
    both the HLP poll and the VLA build_request — the ego frame is extracted and base64
    encoded a single time per frame instead of once per package. Returns a
    {"__numpy__": ...} dict; numpy_serialize / convert_numpy_in_dict pass it through
    unchanged (idempotent) wherever it is embedded.
    """
    return numpy_serialize(frame_ego_image(frame, image_keys))


def poll_hlp(base_url: str, *, ego_image=None, task: str = "",
             is_initial: bool = False, timestamp: Optional[float] = None,
             timeout: float = 10.0) -> Optional[dict]:
    """POST the current observation to /hlp and return the parsed HLP response.

    Sends the payload the REAL HLP server (serve_psix_hlp.py) consumes — the current
    egocentric obs image plus the task, an is_initial flag, and a wall-clock timestamp:
        {ego_image: uint8 HxWx3 RGB, task: str, is_initial: bool, timestamp: float}
    Memory is NOT sent: the HLP SERVER owns + renders memory itself (server-side
    subtask/memory management). The image is required by the real HLP. The numpy image
    is serialized with the same `__numpy__` transport the VLA path uses.

    `ego_image` may be a raw HxWx3 uint8 array OR an already-serialized blob from
    serialize_ego_image() — passing the pre-serialized blob lets the caller encode the
    ego frame once and share it with the VLA request (no duplicate base64 per frame).
    Returns None on error.
    """
    try:
        payload: dict = {"task": task, "is_initial": is_initial}
        if ego_image is not None:
            payload["ego_image"] = (ego_image
                                    if isinstance(ego_image, dict) and "__numpy__" in ego_image
                                    else np.ascontiguousarray(ego_image))
        if timestamp is not None:
            payload["timestamp"] = float(timestamp)
        payload = convert_numpy_in_dict(payload, numpy_serialize)
        r = requests.post(f"{base_url}/hlp", json=payload, timeout=timeout)
        return r.json()
    except Exception:
        return None


def advance_hlp(base_url: str, timeout: float = 10.0) -> Optional[dict]:
    try:
        return requests.post(f"{base_url}/advance", timeout=timeout).json()
    except Exception:
        return None


def reset_hlp(base_url: str, timeout: float = 10.0) -> Optional[dict]:
    try:
        return requests.post(f"{base_url}/reset", timeout=timeout).json()
    except Exception:
        return None


def hlp_subtask_test(base_url: str, *, auto_steps: int = 0, poll_hz: float = 5.0) -> list[str]:
    """Standalone reception test (NO VLA needed): poll the HLP, print each distinct
    subtask received, and step through the list.

    Advancement:
      - auto_steps > 0  : POST /advance up to that many times (unattended), with a
                          short pause so we observe each subtask.
      - auto_steps == 0 : press <Enter> here to advance one subtask (manual).
    Stops when the HLP returns decision == "done". Returns the ordered list of
    distinct subtasks that were received (for assertions/inspection).
    """
    reset_hlp(base_url)
    received: list[str] = []
    last: Optional[str] = None
    steps = 0
    interval = 1.0 / max(poll_hz, 0.1)
    print(f"[hlp-test] polling {base_url}/hlp  (auto_steps={auto_steps})")
    while True:
        resp = poll_hlp(base_url, is_initial=(last is None))
        if resp is None:
            print("[hlp-test] HLP poll failed — is the mock HLP server running?")
            return received
        decision = resp.get("decision")
        subtask = resp.get("next_subtask")
        if subtask != last:
            tag = "(initial)" if last is None else f"(decision={decision})"
            print(f"[hlp-test] received subtask {tag}: {subtask!r}")
            if subtask is not None:
                received.append(subtask)
            last = subtask
        if decision == "done":
            print(f"[hlp-test] done — received {len(received)} subtasks in order: {received}")
            return received
        # step to the next subtask
        if auto_steps > 0:
            if steps >= auto_steps:
                print("[hlp-test] reached auto_steps cap")
                return received
            time.sleep(interval)
            advance_hlp(base_url)
            steps += 1
        else:
            try:
                input("[hlp-test] <Enter> to advance to next subtask (Ctrl-C to stop) ... ")
            except (EOFError, KeyboardInterrupt):
                print()
                return received
            advance_hlp(base_url)
