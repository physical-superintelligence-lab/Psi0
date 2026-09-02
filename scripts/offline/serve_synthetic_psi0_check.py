"""Load a Psi0 run and issue one synthetic serve-side action request."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psi.deploy.helpers import RequestMessage, ResponseMessage
from psi.deploy.psi0_serve_real_sonic import Server
from scripts.offline.dex1_1_layout import (
    ACTION_LEFT_GRIPPER_INDEX,
    ACTION_RIGHT_GRIPPER_INDEX,
    PSI_ACTION_DIM,
    PSI_STATE_DIM,
)


def _response_body(response) -> dict:
    raw = response.body.decode("utf-8")
    payload = json.loads(raw)
    if isinstance(payload, str):
        raise RuntimeError(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--ckpt-step", default="latest")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-exec-horizon", type=int, default=30)
    parser.add_argument("--expect-action-dim", type=int, default=PSI_ACTION_DIM)
    parser.add_argument("--instruction", default="mock dex1-1 grasp")
    args = parser.parse_args()

    os.environ.setdefault("PSI_ATTN_IMPLEMENTATION", "sdpa")
    server = Server(
        policy="psi0",
        run_dir=Path(args.run_dir).expanduser().resolve(),
        ckpt_step=args.ckpt_step,
        device=args.device,
        enable_rtc=False,
        action_exec_horizon=args.action_exec_horizon,
    )
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    state = np.zeros((1, PSI_STATE_DIM), dtype=np.float32)
    request = RequestMessage(
        image={"observation.images.egocentric": image},
        instruction=args.instruction,
        history={"reset": True},
        state={"states": state},
        condition={},
        gt_action=np.zeros((args.action_exec_horizon, args.expect_action_dim), dtype=np.float32),
        dataset_name="synthetic",
        timestamp="offline",
    )
    response = server.predict_action(request.serialize())
    body = _response_body(response)
    message = ResponseMessage.deserialize(body)
    action = np.asarray(message.action, dtype=np.float32)
    if action.ndim != 2:
        raise AssertionError(f"action must be 2-D, got {action.shape}")
    if action.shape[1] != args.expect_action_dim:
        raise AssertionError(f"action dim {action.shape[1]} != {args.expect_action_dim}")
    if not np.all(np.isfinite(action)):
        raise AssertionError("action contains NaN/Inf")
    result = {
        "action_shape": list(action.shape),
        "left_slot_index": ACTION_LEFT_GRIPPER_INDEX,
        "right_slot_index": ACTION_RIGHT_GRIPPER_INDEX,
        "left_slot_first": float(action[0, ACTION_LEFT_GRIPPER_INDEX]),
        "right_slot_first": float(action[0, ACTION_RIGHT_GRIPPER_INDEX]),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
