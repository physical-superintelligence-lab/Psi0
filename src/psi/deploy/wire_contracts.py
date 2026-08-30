"""Versioned wire contracts for the HLP x WM x VLA deployment stack.

Single source of truth for the cross-process protocol pieces that MUST be
byte-identical on both ends of a socket:

  1. VLA condition provenance (rides in the existing, currently-unused
     ``RequestMessage.condition`` dict — helpers.py:70-108) and the canonical
     condition-hash serialization the server recomputes over its *decoded*
     inputs. A client-supplied hash is never trusted by echo; the server
     recomputes and mismatches are protocol errors.
  2. The uniform HLP response contract shared by serve_psix_hlp.py and
     mock_psix_hlp_serve.py, plus reply-shape validation the poller runs
     before adopting anything.
  3. Gate-state vocabulary (RUN / HOLD / ABORT_LATCHED / HOLD_PATH_FAILURE)
     shared by the robot-side orchestrator, tests, and dashboards.

The robot client (GR00T repo) vendors a verbatim copy of the hash + condition
helpers; cross-repo consistency is enforced by the frozen golden vectors in
``scripts/tests/wire_contracts_golden.json`` — regenerate them ONLY with a version
bump, never in place.

Gate-state semantics (plan .logs/2026-07-07-WM_Serve/plan_next.md §5):

  RUN                normal execution; every executed action's ack key equals
                     the active condition_key.
  HOLD               recoverable transient hold. WBC keeps running; the client
                     publishes a frozen pose action at control rate. Entered on
                     startup-waiting, condition mismatch, HLP/WM staleness,
                     done, or any watchdog. Exits only via a fresh
                     condition-ack match (or operator command).
  ABORT_LATCHED      a hard deadline expired while in HOLD. Hold keeps being
                     published, but all automatic retry/recovery stops and the
                     first failure cause is latched. Operator ``:ack`` moves
                     ABORT_LATCHED -> HOLD only; release back to RUN requires a
                     fresh HLP/WM/VLA handshake.
  HOLD_PATH_FAILURE  the hold path itself is unhealthy (stale robot state,
                     encoder or publisher failure). Not recoverable in-band:
                     transition to the G0-frozen terminal/emergency policy.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Versions. Bump any of these when the corresponding byte layout changes; the
# golden-vector test fails loudly if a change forgets the bump.
# ---------------------------------------------------------------------------
CONDITION_SCHEMA_VERSION = "wm-vla-condition/1"
CONDITION_HASH_VERSION = "wm-vla-condition-hash/1"
HLP_SCHEMA_VERSION = "hlp-reply/1"
WM_SCHEMA_VERSION = "wm-reply/1"


class GateState(str, Enum):
    RUN = "RUN"
    HOLD = "HOLD"
    ABORT_LATCHED = "ABORT_LATCHED"
    HOLD_PATH_FAILURE = "HOLD_PATH_FAILURE"


# ---------------------------------------------------------------------------
# Condition provenance (VLA leg)
# ---------------------------------------------------------------------------

def new_vla_session_id() -> str:
    """Minted by the CLIENT once per WebSocket connection attempt.

    A reconnect gets a fresh id, so a stale action from the previous session
    can never be accepted just because its condition_id coincides.
    """
    return uuid.uuid4().hex


def condition_hash(instruction: str, goal_rgb: np.ndarray, *,
                   preprocess_version: str = "raw-rgb/1") -> str:
    """Canonical hash over the (instruction, decoded goal image) pair.

    Layout (frozen under CONDITION_HASH_VERSION — see module docstring):
    version NUL instruction-utf8 NUL dtype NUL shape-csv NUL C-order-bytes NUL
    preprocess_version. The IMAGE argument is the DECODED array (after JPEG
    decode, before any model-side resize); both peers must hash the same
    decoded content, which is exactly what makes a truncated/re-encoded goal
    detectable.
    """
    if goal_rgb.dtype != np.uint8:
        raise ValueError(f"condition goal must be uint8, got {goal_rgb.dtype}")
    h = hashlib.sha256()
    h.update(CONDITION_HASH_VERSION.encode("utf-8"))
    h.update(b"\x00")
    h.update(instruction.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(goal_rgb.dtype).encode("ascii"))
    h.update(b"\x00")
    h.update(",".join(str(int(s)) for s in goal_rgb.shape).encode("ascii"))
    h.update(b"\x00")
    h.update(np.ascontiguousarray(goal_rgb).tobytes())
    h.update(b"\x00")
    h.update(preprocess_version.encode("utf-8"))
    return h.hexdigest()


def build_condition(vla_session_id: str, condition_id: int,
                    supplied_condition_hash: str) -> Dict[str, Any]:
    """The dict the client puts in RequestMessage.condition for every obs."""
    return {
        "schema_version": CONDITION_SCHEMA_VERSION,
        "vla_session_id": str(vla_session_id),
        "condition_id": int(condition_id),
        "supplied_condition_hash": str(supplied_condition_hash),
    }


def parse_condition(obj: Any) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Classify an incoming ``condition`` field.

    Returns (kind, parsed) where kind is:
      "legacy"   — absent / None / {} (today's clients). Serve as before,
                   mark provenance absent in responses.
      "ok"       — a well-formed provenance dict (returned normalized).
      "invalid"  — present but malformed → the caller must treat it as a
                   protocol error (fail closed), NOT as legacy.
    """
    if obj is None or obj == {}:
        return "legacy", None
    if not isinstance(obj, dict):
        return "invalid", None
    if obj.get("schema_version") != CONDITION_SCHEMA_VERSION:
        return "invalid", None
    sid = obj.get("vla_session_id")
    cid = obj.get("condition_id")
    sch = obj.get("supplied_condition_hash")
    if not (isinstance(sid, str) and sid):
        return "invalid", None
    if not isinstance(cid, int) or isinstance(cid, bool) or cid < 0:
        return "invalid", None
    if not (isinstance(sch, str) and len(sch) == 64):
        return "invalid", None
    return "ok", {
        "schema_version": CONDITION_SCHEMA_VERSION,
        "vla_session_id": sid,
        "condition_id": cid,
        "supplied_condition_hash": sch,
    }


def build_action_ack(*, vla_session_id: str, condition_id: int,
                     action_condition_hash: str, model_condition_hash: str,
                     action_version: int,
                     inference_time_ms: Optional[float] = None) -> Dict[str, Any]:
    """Provenance fields attached to EVERY action message the VLA server emits.

    ``action_condition_hash`` is the server's recompute over its decoded
    inputs; ``model_condition_hash`` additionally folds the model-side
    preprocess (resize/crop config + final tensor digest)."""
    ack = {
        "schema_version": CONDITION_SCHEMA_VERSION,
        "action_vla_session_id": str(vla_session_id),
        "action_condition_id": int(condition_id),
        "action_condition_hash": str(action_condition_hash),
        "model_condition_hash": str(model_condition_hash),
        "action_version": int(action_version),
    }
    if inference_time_ms is not None:
        ack["inference_time_ms"] = float(inference_time_ms)
    return ack


def ack_matches(ack: Any, *, vla_session_id: str, condition_id: int,
                supplied_condition_hash: str) -> bool:
    """Client-side release check: the executed action must carry EXACTLY the
    candidate condition key and the server-recomputed hash must equal what the
    client supplied (content equality, not echo — the server recomputed it)."""
    if not isinstance(ack, dict):
        return False
    return (
        ack.get("action_vla_session_id") == vla_session_id
        and ack.get("action_condition_id") == condition_id
        and ack.get("action_condition_hash") == supplied_condition_hash
    )


# ---------------------------------------------------------------------------
# HLP uniform response contract (prod server + mock MUST both satisfy this)
# ---------------------------------------------------------------------------
HLP_DECISIONS = ("continue", "switch", "done")

# Keys every well-formed HLP reply carries (predict AND control endpoints).
HLP_REQUIRED_KEYS = (
    "schema_version",
    "decision",
    "next_subtask",       # str | None
    "instruction",        # str | None
    "stage",              # int
    "done",               # bool
    "state_revision",     # int, monotonic per server lifetime
    "robot_episode_session_id",  # str | None (None only before /reset/acquire)
    "request_id",         # echo of the client's request_id (int | str | None)
)

# Optional keys (present when applicable).
HLP_OPTIONAL_KEYS = (
    "task_fingerprint",
    "history", "history_items", "memory",
    "model_decision", "model_next_subtask",
    "pending_candidate", "pending_count",
    "committed_event", "stale_generation_discarded",
    "capture_id", "clean_start", "resumed",
    "timing_ms", "inference_time_ms", "raw_text",
)


def validate_hlp_reply(obj: Any) -> Tuple[str, Optional[str]]:
    """Classify an HLP reply BEFORE the poller adopts anything from it.

    Returns (kind, why) with kind:
      "error"     — HTTP-200 {"error": ...} body (server predict exception).
                    Never touches is_initial / desired state; log throttled.
      "malformed" — missing/mistyped required keys. Same handling as error.
      "ok"        — safe to run through the adoption/confirmation logic.
    """
    if not isinstance(obj, dict):
        return "malformed", "not a dict"
    if "error" in obj:
        return "error", str(obj.get("error"))[:200]
    for k in HLP_REQUIRED_KEYS:
        if k not in obj:
            return "malformed", f"missing key {k!r}"
    if obj["schema_version"] != HLP_SCHEMA_VERSION:
        return "malformed", f"schema_version {obj['schema_version']!r}"
    if obj["decision"] not in HLP_DECISIONS:
        return "malformed", f"decision {obj['decision']!r}"
    if not (obj["next_subtask"] is None or isinstance(obj["next_subtask"], str)):
        return "malformed", "next_subtask type"
    if not (obj["instruction"] is None or isinstance(obj["instruction"], str)):
        return "malformed", "instruction type"
    if not isinstance(obj["stage"], int) or isinstance(obj["stage"], bool):
        return "malformed", "stage type"
    if not isinstance(obj["done"], bool):
        return "malformed", "done type"
    if not isinstance(obj["state_revision"], int) or isinstance(obj["state_revision"], bool):
        return "malformed", "state_revision type"
    return "ok", None


def canonical_json(obj: Any) -> str:
    """Deterministic JSON used wherever a dict feeds a hash (profiles,
    request digests). sort_keys + compact separators, UTF-8, no NaN."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def profile_hash(profile: Dict[str, Any]) -> str:
    """Hash of a semantic-profile JSON with its own hash field excluded."""
    body = {k: v for k, v in profile.items() if k != "profile_hash"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
