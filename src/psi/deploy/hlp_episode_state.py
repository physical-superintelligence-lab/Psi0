"""Server-side HLP episode state machine — shared by serve_psix_hlp.py and
mock_psix_hlp_serve.py so the prod/mock response contract cannot drift.

Implements plan_next.md §3.2/§3.3 (.logs/2026-07-07-WM_Serve/plan_next.md):

  * state_revision compare-and-swap: every control mutation and every commit
    increments a server-lifetime monotonic revision; a model result generated
    against an older revision is discarded (``stale_generation_discarded``)
    instead of clobbering newer state.
  * server-side confirmation BEFORE commit: establish/switch/done predictions
    become a pending candidate and only mutate current/memory/stage/done after
    N confirmations from DISTINCT, fresh captures. Until then the authoritative
    reply stays "continue". A client-side dwell cannot do this — by the time a
    client sees "switch" the old server had already committed (P0-1).
  * episode lease: /reset/acquire mints robot_episode_session_id +
    task_fingerprint and a TTL lease. While a lease is active, polls/controls
    carrying a different session id are rejected (409) without touching state
    and cannot contribute confirmations. With NO active lease, legacy
    session-less clients keep working (shadow/compat mode).
  * uniform response contract (wire_contracts.HLP_REQUIRED_KEYS) on predict AND
    every control endpoint.

Everything here is pure CPU state — model generate stays in the callers:

    snap = st.begin_poll(...)          # short-circuits override/done/session
    if snap.reply is not None: return snap.reply (+ snap.http_status)
    out = <model generate on snap.memory_items / snap.is_initial_effective>
    reply = st.finish_poll(snap, model_decision=..., next_subtask=..., ...)
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from psi.deploy.wire_contracts import HLP_SCHEMA_VERSION, canonical_json


def _norm_key(text: str) -> str:
    """Equality key for confirmation counting: whitespace-collapsed casefold.
    (Semantic canonicalization to the profile vocabulary is CLIENT-side.)"""
    return re.sub(r"\s+", " ", str(text).strip()).casefold()


def task_fingerprint(task: str, profile_hash: Optional[str] = None,
                     scene_manifest_hash: Optional[str] = None) -> str:
    body = {
        "task": _norm_key(task),
        "profile_hash": profile_hash or "",
        "scene_manifest_hash": scene_manifest_hash or "",
    }
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


@dataclass
class PollSnapshot:
    reply: Optional[Dict[str, Any]] = None   # short-circuit reply (return as-is)
    http_status: int = 200
    revision: int = -1
    task: str = ""
    is_initial_effective: bool = False
    memory_items: List[Dict[str, Any]] = field(default_factory=list)
    current: Optional[str] = None
    capture_key: Optional[str] = None
    capture_fresh: bool = True
    request_id: Any = None
    capture_id: Any = None


class HlpEpisodeState:

    def __init__(self, *, switch_confirmations: int = 2, done_confirmations: int = 2,
                 establish_confirmations: Optional[int] = None,
                 max_capture_age_s: float = 3.0, lease_ttl_s: float = 30.0,
                 clock=time.time):
        self._clock = clock
        self._lock = threading.RLock()
        self.switch_confirmations = int(switch_confirmations)
        self.done_confirmations = int(done_confirmations)
        self.establish_confirmations = int(establish_confirmations
                                           if establish_confirmations is not None
                                           else switch_confirmations)
        self.max_capture_age_s = float(max_capture_age_s)
        self.lease_ttl_s = float(lease_ttl_s)

        self._revision = 0
        self._last_task = ""
        self._session_id: Optional[str] = None
        self._task_fingerprint: Optional[str] = None
        self._lease_expires_at: float = 0.0
        self._clean_episode_state()
        # diagnostics for /state
        self._last_model_next: Optional[str] = None
        self._last_model_decision: Optional[str] = None
        self._last_server_decision: Optional[str] = None
        self._last_seconds: float = 0.0
        self._last_timing: Dict[str, Any] = {}
        self._last_raw: str = ""
        self._last_status: str = "reset"
        self._last_updated_at: float = self._clock()

    # ------------------------------------------------------------- internals

    def _clean_episode_state(self) -> None:
        self._memory: List[Tuple[str, float]] = []
        self._current: Optional[str] = None
        self._override: Optional[str] = None
        self._stage_idx = -1
        self._done = False
        self._clear_pending()

    def _clear_pending(self) -> None:
        self._pending_kind: Optional[str] = None
        self._pending_text: Optional[str] = None   # stripped original of latest vote
        self._pending_key: Optional[str] = None    # (normalized) equality key
        self._pending_count = 0
        self._pending_captures: set = set()

    def _bump(self) -> None:
        self._revision += 1

    def _stage(self) -> int:
        return max(0, self._stage_idx)

    def _instruction(self) -> Optional[str]:
        t = str(self._last_task).strip().lower()   # training lowercases task, NOT subtask
        return f"Task: {t}. Subtask: {self._current}" if self._current else f"Task: {t}"

    def _memory_items(self, rounded: bool = False) -> List[Dict[str, Any]]:
        now = self._clock()
        out = []
        for text, st in reversed(self._memory):
            ago = now - st
            out.append({"text": text, "seconds_ago": round(ago, 1) if rounded else ago})
        return out

    def _lease_active(self) -> bool:
        return self._session_id is not None and self._clock() < self._lease_expires_at

    def _renew_lease(self) -> None:
        if self._session_id is not None:
            self._lease_expires_at = self._clock() + self.lease_ttl_s

    def _pending_view(self) -> Optional[Dict[str, Any]]:
        if self._pending_kind is None:
            return None
        return {"kind": self._pending_kind, "next_subtask": self._pending_text,
                "count": self._pending_count}

    def _reply(self, decision: str, *, request_id: Any = None, capture_id: Any = None,
               raw: str = "", secs: Any = None, extras: Optional[Dict[str, Any]] = None,
               render_history: bool = True) -> Dict[str, Any]:
        """Uniform contract reply (caller holds the lock)."""
        obj: Dict[str, Any] = {
            "schema_version": HLP_SCHEMA_VERSION,
            "decision": decision,
            "next_subtask": self._current,
            "instruction": self._instruction(),
            "stage": self._stage(),
            "done": bool(self._done),
            "state_revision": self._revision,
            "robot_episode_session_id": self._session_id,
            "request_id": request_id,
            "capture_id": capture_id,
            "task_fingerprint": self._task_fingerprint,
            "seconds_to_subgoal": float(secs) if isinstance(secs, (int, float)) else 0.0,
            "pending_candidate": self._pending_view(),
            "pending_count": self._pending_count,
            "raw_text": raw,
        }
        if render_history:
            items = self._memory_items(rounded=True)
            obj["history_items"] = items
            # legacy string render is composed by the server layer (needs
            # memory_format); left to callers that have it. Provide items always.
        if extras:
            obj.update(extras)
        return obj

    def _note(self, obj: Dict[str, Any], *, model_next=None, model_decision=None,
              status: str = "live") -> None:
        self._last_model_next = model_next
        self._last_model_decision = model_decision
        self._last_server_decision = obj.get("decision")
        self._last_seconds = obj.get("seconds_to_subgoal", 0.0)
        self._last_raw = str(obj.get("raw_text") or "")
        self._last_status = status
        self._last_updated_at = self._clock()

    # ------------------------------------------------------------- session

    def acquire(self, *, task: str, profile_hash: Optional[str] = None,
                scene_manifest_hash: Optional[str] = None, force: bool = False,
                request_id: Any = None) -> Tuple[int, Dict[str, Any]]:
        """POST /reset/acquire — clean state + new session + lease. 409 while a
        different, unexpired lease is active (unless force)."""
        with self._lock:
            if self._lease_active() and not force:
                obj = self._reply("continue", request_id=request_id,
                                  raw="(lease held)",
                                  extras={"error": "lease_held",
                                          "lease_expires_at": self._lease_expires_at})
                return 409, obj
            self._clean_episode_state()
            self._last_task = str(task)
            self._session_id = uuid.uuid4().hex
            self._task_fingerprint = task_fingerprint(task, profile_hash,
                                                      scene_manifest_hash)
            self._lease_expires_at = self._clock() + self.lease_ttl_s
            self._bump()
            obj = self._reply("continue", request_id=request_id, raw="(acquire)",
                              extras={"clean_start": True,
                                      "lease_expires_at": self._lease_expires_at,
                                      "lease_ttl_s": self.lease_ttl_s})
            self._note(obj, status="acquired")
            return 200, obj

    def session_gate(self, session_id: Any, *, request_id: Any = None
                     ) -> Optional[Tuple[int, Dict[str, Any]]]:
        """None = allowed. (409, reply) when an active lease exists and the
        caller's session doesn't match. Legacy callers (session_id=None) are
        allowed only while NO lease is active."""
        with self._lock:
            if not self._lease_active():
                return None
            if session_id == self._session_id:
                self._renew_lease()
                return None
            obj = self._reply("continue", request_id=request_id,
                              raw="(session mismatch)",
                              extras={"error": "session_mismatch"})
            return 409, obj

    # ------------------------------------------------------------- polling

    def begin_poll(self, *, task: str, is_initial: bool, session_id: Any = None,
                   capture_id: Any = None, capture_age_s: Any = None,
                   request_id: Any = None) -> PollSnapshot:
        gate = self.session_gate(session_id, request_id=request_id)
        if gate is not None:
            status, obj = gate
            return PollSnapshot(reply=obj, http_status=status)
        with self._lock:
            self._last_task = str(task)
            if self._override is not None:
                obj = self._reply("continue", request_id=request_id,
                                  capture_id=capture_id, raw="(override active)")
                self._note(obj, status="override active")
                return PollSnapshot(reply=obj)
            if self._done:
                obj = self._reply("done", request_id=request_id,
                                  capture_id=capture_id, secs=0.0,
                                  raw="(done latched)")
                self._note(obj, status="done")
                return PollSnapshot(reply=obj)
            fresh = True
            if capture_age_s is not None:
                try:
                    fresh = float(capture_age_s) <= self.max_capture_age_s
                except (TypeError, ValueError):
                    fresh = False
            key = str(capture_id) if capture_id is not None else f"anon-{uuid.uuid4().hex}"
            return PollSnapshot(
                reply=None,
                revision=self._revision,
                task=str(task),
                is_initial_effective=bool(is_initial) or self._current is None,
                memory_items=self._memory_items(rounded=False),
                current=self._current,
                capture_key=key,
                capture_fresh=fresh,
                request_id=request_id,
                capture_id=capture_id,
            )

    def _threshold(self, kind: str) -> int:
        return {"establish": self.establish_confirmations,
                "switch": self.switch_confirmations,
                "done": self.done_confirmations}[kind]

    def finish_poll(self, snap: PollSnapshot, *, model_decision: Any,
                    next_subtask: Any, seconds: Any = None, raw_text: str = "",
                    timing_ms: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._lock:
            extras: Dict[str, Any] = {
                "model_decision": model_decision,
                "model_next_subtask": next_subtask,
            }
            if timing_ms:
                extras["timing_ms"] = dict(timing_ms)
                extras["inference_time_ms"] = timing_ms.get("total")

            # CAS: a control mutation or commit landed during the generate —
            # the model result was computed against dead state. Discard it.
            if snap.revision != self._revision:
                extras["stale_generation_discarded"] = True
                decision = "done" if self._done else "continue"
                obj = self._reply(decision, request_id=snap.request_id,
                                  capture_id=snap.capture_id,
                                  raw="(stale generation discarded)", extras=extras)
                self._note(obj, model_next=next_subtask,
                           model_decision=model_decision, status="stale-discarded")
                return obj

            nxt = str(next_subtask).strip() if isinstance(next_subtask, str) else None
            nxt = nxt or None
            cand: Optional[Tuple[str, Optional[str]]] = None
            if model_decision == "done":
                cand = ("done", None)
            elif self._current is None and nxt and snap.is_initial_effective \
                    and model_decision in ("switch", "continue"):
                cand = ("establish", nxt)
            elif model_decision == "switch" and nxt \
                    and _norm_key(nxt) != _norm_key(self._current or ""):
                cand = ("switch", nxt)

            if cand is None:
                # inconsistent / inapplicable prediction clears the candidate
                if self._pending_kind is not None:
                    self._clear_pending()
                obj = self._reply("continue", request_id=snap.request_id,
                                  capture_id=snap.capture_id, secs=seconds,
                                  raw=raw_text, extras=extras)
                self._note(obj, model_next=nxt, model_decision=model_decision)
                return obj

            kind, text = cand
            key = f"{kind}\x00{_norm_key(text) if text else ''}"
            if key != self._pending_key:
                self._pending_kind, self._pending_text = kind, text
                self._pending_key = key
                self._pending_count = 0
                self._pending_captures = set()
            # a vote counts only from a fresh capture not yet used for this candidate
            if snap.capture_fresh and snap.capture_key not in self._pending_captures:
                self._pending_captures.add(snap.capture_key)
                self._pending_count += 1
                if text is not None:
                    self._pending_text = text   # keep the latest raw spelling

            if self._pending_count < self._threshold(kind):
                extras["pending_candidate"] = self._pending_view()
                extras["pending_count"] = self._pending_count
                obj = self._reply("continue", request_id=snap.request_id,
                                  capture_id=snap.capture_id, secs=seconds,
                                  raw=raw_text, extras=extras)
                self._note(obj, model_next=nxt, model_decision=model_decision,
                           status=f"pending-{kind}")
                return obj

            # ---- commit ----
            committed = {"kind": kind, "next_subtask": self._pending_text,
                         "state_revision": self._revision + 1}
            if kind == "done":
                self._done = True
                decision = "done"
            else:
                self._current = self._pending_text
                self._memory.append((self._current, self._clock()))
                self._stage_idx = 0 if self._stage_idx < 0 else self._stage_idx + 1
                decision = "switch"
            self._clear_pending()
            self._bump()
            extras["committed_event"] = committed
            obj = self._reply(decision, request_id=snap.request_id,
                              capture_id=snap.capture_id, secs=seconds,
                              raw=raw_text, extras=extras)
            self._note(obj, model_next=nxt, model_decision=model_decision,
                       status=f"committed-{kind}")
            return obj

    # ------------------------------------------------------------- controls
    # Semantics mirror the pre-existing endpoints exactly; every mutation
    # bumps the revision and clears the pending candidate.

    def prev(self, *, request_id: Any = None) -> Dict[str, Any]:
        with self._lock:
            self._done = False
            ov = self._override
            self._override = None
            if ov is not None:
                if self._memory and self._memory[-1][0] == ov:
                    self._memory.pop()
                self._current = self._memory[-1][0] if self._memory else None
                clean = self._current is None
            elif self._stage_idx <= 0 or len(self._memory) <= 1:
                sid, fp, lease = self._session_id, self._task_fingerprint, self._lease_expires_at
                task = self._last_task
                self._clean_episode_state()
                self._last_task = task
                self._session_id, self._task_fingerprint, self._lease_expires_at = sid, fp, lease
                clean = True
            else:
                self._memory.pop()
                self._stage_idx -= 1
                self._current = self._memory[-1][0] if self._memory else None
                clean = False
            self._clear_pending()
            self._bump()
            obj = self._reply("continue", request_id=request_id, raw="(prev)",
                              extras={"clean_start": clean})
            self._note(obj, status="prev")
            return obj

    def reset(self, *, request_id: Any = None) -> Dict[str, Any]:
        with self._lock:
            sid, fp, lease = self._session_id, self._task_fingerprint, self._lease_expires_at
            task = self._last_task
            self._clean_episode_state()
            self._last_task = task
            self._session_id, self._task_fingerprint, self._lease_expires_at = sid, fp, lease
            self._bump()
            obj = self._reply("continue", request_id=request_id, raw="(reset)",
                              extras={"clean_start": True})
            self._note(obj, status="reset")
            return obj

    def override(self, text: str, *, request_id: Any = None) -> Dict[str, Any]:
        with self._lock:
            self._done = False
            self._override = str(text)
            self._current = self._override
            if not self._memory or self._memory[-1][0] != self._override:
                self._memory.append((self._override, self._clock()))
            self._clear_pending()
            self._bump()
            obj = self._reply("continue", request_id=request_id, raw="(override)")
            self._note(obj, status="override active")
            return obj

    def resume(self, *, request_id: Any = None) -> Dict[str, Any]:
        with self._lock:
            was = self._override is not None
            self._override = None
            self._done = False
            self._clear_pending()
            self._bump()
            obj = self._reply("continue", request_id=request_id, raw="(resume)",
                              extras={"resumed": was,
                                      "clean_start": self._current is None})
            self._note(obj, status="resumed")
            return obj

    # ------------------------------------------------------------- state

    def state_view(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "schema_version": HLP_SCHEMA_VERSION,
                "stage": self._stage(),
                "current": self._current,
                "override": self._override,
                "instruction": self._instruction(),
                "history_items": self._memory_items(rounded=True),
                "done": self._done,
                "state_revision": self._revision,
                "robot_episode_session_id": self._session_id,
                "task_fingerprint": self._task_fingerprint,
                "lease_active": self._lease_active(),
                "lease_expires_at": self._lease_expires_at or None,
                "pending_candidate": self._pending_view(),
                "pending_count": self._pending_count,
                "confirmations": {"establish": self.establish_confirmations,
                                  "switch": self.switch_confirmations,
                                  "done": self.done_confirmations},
                "model_next_subtask": self._last_model_next,
                "model_decision": self._last_model_decision,
                "server_decision": self._last_server_decision,
                "seconds_to_subgoal": self._last_seconds,
                "timing_ms": self._last_timing,
                "raw_text": self._last_raw,
                "status": self._last_status,
                "updated_at": self._last_updated_at,
            }
