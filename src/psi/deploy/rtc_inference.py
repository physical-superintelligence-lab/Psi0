"""Real-time chunking (RTC) inference engine for the psix RTC server.

Wraps a PsixModel and turns its chunk-predicting policy into a streaming per-tick
action source: a background daemon thread keeps the action chunk fresh via the RTC
training-time flow, while next_action() dispenses one action per control tick. See
serve_psix.py for the control loop that drives it.
"""
import os
import threading
import time
from collections import deque
from typing import Any, Dict

from psi.deploy.chunk_recorder import maybe_chunk_recorder

import numpy as np
import torch

from psi.models.psix import PsixModel
from psi.utils.overwatch import initialize_overwatch
from psi.deploy.dashboard import update_dashboard, log_every, shift_actions_with_zero_pad

overwatch = initialize_overwatch(__name__)


def _condition_key(meta: Dict[str, Any] | None):
    """Identity of a served condition: (vla_session_id, condition_id). None for
    legacy no-provenance obs — legacy never triggers the superseded discard."""
    if not meta:
        return None
    return (meta.get("vla_session_id"), meta.get("condition_id"))


class RealTimeChunkInference:
    # Horizons/delays below are counted in control-loop ticks (1 tick = 1/ctrl_hz s).
    def __init__(
        self,
        policy: PsixModel,
        initial_obs: Dict[str, Any],  # images, video, instruction, states, etc
        min_exec_horizon: int,  # min ticks consumed from a chunk before starting a new inference
        delay_buf_size: int,    # recent-delay window length; max() over it feeds next inference_delay
        initial_delay: int,     # delay estimate (ticks) for the first inference, before any is measured
        max_delay: int,         # max inference delay (ticks) the model was trained for (model_cfg.max_delay)
        cfg_scale: float = 1.0, # >1 enables goal-image classifier-free guidance
        cfg_uncond_strip_subtask: bool = False,  # coupled-dropout ckpt: uncond row is task-only text
        rtc_mode: str = "train",        # train = frozen-prefix inpainting (ckpt trained with rtc)
                                        # test_time = PiGDM guidance (works on a no-rtc ckpt)
        pig_guidance_weight: float = 1000.0,
        pig_mask_schedule: str = "exponential",
        pig_sigma_threshold: float = 0.15,
        pig_guidance_mode: str = "alpha",
        pig_guidance_alpha: float = 0.9,
    ):
        self.policy = policy
        self.cfg_scale = cfg_scale
        self.cfg_uncond_strip_subtask = cfg_uncond_strip_subtask
        self.rtc_mode = rtc_mode
        self.pig_guidance_weight = pig_guidance_weight
        self.pig_mask_schedule = pig_mask_schedule
        self.pig_sigma_threshold = pig_sigma_threshold
        self.pig_guidance_mode = pig_guidance_mode
        self.pig_guidance_alpha = pig_guidance_alpha
        self.min_exec_horizon = min_exec_horizon
        self.trained_max_delay = max_delay
        self.initial_delay = initial_delay
        self._delay_buf_size = delay_buf_size
        self.ticks_since_replan = 0
        self._max_infer_ms = 0.0  # peak forward-pass latency seen so far
        # When True, the next inference re-warms via the unconditioned _predict_action
        # (no prev-action RTC conditioning), as if a fresh episode just started.
        self._reset_requested = False
        # Publish it up front: the dashboard now shows this row only for servers that
        # HAVE a reset path, so the key has to exist before the first inference tick.
        update_dashboard(reset_requested=self._reset_requested)
        # Condition provenance (plan §6.2): every installed chunk is tagged with the
        # condition_meta of the obs it was computed FROM; next_action() hands it back
        # with each action so the wire can ack exactly which (instruction, goal) pair
        # produced the executing chunk. A forward whose obs condition KEY was
        # superseded mid-flight is DISCARDED (never installed) and replanned at once.
        self.chunk_condition: Dict[str, Any] | None = None
        self._force_replan = False
        # The wire carries one action row per tick, so the planned chunk exists
        # nowhere else. Off unless PSIX_CHUNK_DUMP_DIR is set.
        self._chunk_recorder = maybe_chunk_recorder()

        # make a blocking prediction to obtain the first action chunk
        overwatch.info("Warming up: predicting first action chunk (1/3)")
        update_dashboard(status="warming up: first chunk (1/3)")
        # If the client shipped an encoded current-pose pseudo prev-action, warm-start the
        # first chunk through the RTC flow (conditioned on a single prefix action, d=1)
        # instead of the unconditioned path, so the first chunk is consistent with all later chunks.
        _init_prev = initial_obs.get('init_prev_action')
        if _init_prev is not None:
            H, Da = self.policy.action_horizon, self.policy.action_dim
            prev = np.tile(np.asarray(_init_prev, dtype=np.float32).reshape(1, -1), (H, 1))  # (H, Da)
            d0 = 1
            overwatch.info(f"[init-prev] warm-starting first chunk via RTC: prev={prev.shape}, d={d0}")
            initial_action = self._predict_action_rtc(initial_obs, prev, d0)  # (H, D)
        else:
            initial_action = self._predict_action(initial_obs) # (H, D)

        # warm up the RTC inference path (predict_action_with_training_rtc_flow);
        # 2x: 1st compiles/autotunes, 2nd stabilizes.
        shifted_action = shift_actions_with_zero_pad(initial_action, self.min_exec_horizon)
        for i in range(2):
            overwatch.info(f"Warming up: RTC forward pass ({i + 2}/3)")
            update_dashboard(status=f"warming up: RTC pass {i + 1}/2 ({i + 2}/3)")
            _ = self._predict_action_rtc(initial_obs, shifted_action, initial_delay)
        overwatch.info("Model warmed up")
        update_dashboard(status="live")

        self.action_chunk = initial_action # (H, D)
        self.chunk_condition = initial_obs.get('condition_meta')  # provenance of the initial chunk
        # Lightweight wire telemetry: increment exactly when a newly inferred
        # chunk is installed. This distinguishes a live RTC rollout from the
        # dangerous "repeat the last row after chunk exhaustion" path.
        self.chunk_id = 0
        # replan_seq counts every forward attempt (installed or superseded);
        # chunk_replan_id/chunk_infer_ms describe the forward that produced the
        # currently installed chunk, so the wire can attribute each action to
        # one specific replan and its latency.
        self.replan_seq = 0
        self.chunk_replan_id = 0
        self.chunk_infer_ms = 0.0
        self.current_obs: Dict[str, Any] | None = None
        # Rolling window of recently measured inference delays, in control-loop ticks
        # (how many ticks elapsed while a forward pass ran). max() over it is fed as the
        # `inference_delay` to the next RTC prediction. Seeded with initial_delay for the first one.
        self.inference_delays = deque([initial_delay], maxlen=delay_buf_size)

        # Condition variable layered on top of a lock. It coordinates the producer/consumer
        # handoff between two threads that share self.ticks_since_replan, self.current_obs, self.action_chunk, self.inference_delays
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

        # Set before the thread starts; stop() flips it to break the inference loop.
        self._stop_event = threading.Event()
        self._infer_th = threading.Thread(target=self._inference_loop, daemon=True)
        self._infer_th.start()


    def stop(self, join_timeout: float = 5.0) -> None:
        """Signal the background inference thread to exit and wait for it to finish.

        Called when the client disconnects. Wakes the thread if it's parked in
        _cond.wait(); if a forward pass is in flight, the thread exits right after
        it completes (bounded by join_timeout)."""
        self._stop_event.set()
        with self._cond:
            self._cond.notify()
        self._infer_th.join(timeout=join_timeout)
        if self._infer_th.is_alive():
            overwatch.warning(f"Inference thread did not stop within {join_timeout}s")

    def reset_stats(self) -> None:
        """Reset accumulated telemetry (e.g. peak inference latency) on server reset."""
        self._max_infer_ms = 0.0

    def reset(self) -> None:
        with self._cond:
            self._reset_requested = True
            self.inference_delays = deque([self.initial_delay], maxlen=self._delay_buf_size)
            self._max_infer_ms = 0.0
            update_dashboard(reset_requested=self._reset_requested)
            self._cond.notify()

    def next_action(self, obs_next: Dict[str, Any]):
        """Consume a_(t-1) and provide o_t.

        Returns ``(action(1,D), chunk_condition|None, rtc_meta)``. ``rtc_meta``
        contains only constant-time scalar bookkeeping and is safe to attach to
        every action response.
        """
        with self._cond:
            self.ticks_since_replan += 1
            self.current_obs = obs_next
            self._cond.notify()
            chunk_tick = self.ticks_since_replan - 1
            repeat_last = chunk_tick >= len(self.action_chunk)
            if repeat_last:
                single_action = self.action_chunk[-1]
                # The metadata remains per-tick, but terminal/file I/O must not
                # become a 30 Hz storm in the exact degraded mode we diagnose.
                log_every(
                    "rtc_chunk_runout",
                    "Run out of action chunk! Repeating last action.",
                    freq=1.0,
                    level="critical",
                )
            else:
                single_action = self.action_chunk[chunk_tick]
            cond = dict(self.chunk_condition) if self.chunk_condition else None
            rtc_meta = {
                "rtc_chunk_id": int(self.chunk_id),
                "rtc_chunk_tick": int(chunk_tick),
                "rtc_chunk_ticks": int(len(self.action_chunk)),
                "rtc_repeat_last": bool(repeat_last),
                "rtc_replan_id": int(self.chunk_replan_id),
                "rtc_infer_ms": float(self.chunk_infer_ms),
            }
            return single_action[np.newaxis, :], cond, rtc_meta

    def _inference_loop(self):
        while not self._stop_event.is_set():
            with self._cond:
                try:
                    while (self.ticks_since_replan < self.min_exec_horizon
                           and not self._force_replan
                           and not self._stop_event.is_set()):
                        # wait until the current chunk has been executed
                        # at least min_exec_horizon ticks (a superseded-condition
                        # discard forces an immediate replan instead)
                        self._cond.wait()

                    if self._stop_event.is_set():  # woken by stop(): exit cleanly
                        overwatch.info("Inference loop stopping (client disconnected)")
                        return
                    self._force_replan = False

                    executed_ticks = self.ticks_since_replan
                    # Grab the reference only (no copy): published obs are never mutated in place
                    # (publish-by-replacement), so this snapshot stays stable across the forward pass.
                    current_obs = self.current_obs
                    snapshot_condition = (current_obs or {}).get('condition_meta')
                    max_delay = max(self.inference_delays)
                    # Re-warm unconditioned if a reset was requested or the chunk ran out.
                    # Capture+consume the flag under the lock so it's set exactly once.
                    rewarm = self._reset_requested or executed_ticks >= self.action_chunk.shape[0]
                    self._reset_requested = False
                    update_dashboard(reset_requested=self._reset_requested)
                    if max_delay >= self.trained_max_delay:
                        update_dashboard(warn=f"inference too slow: delay {max_delay} >= max {self.trained_max_delay} (clamped)")
                        log_every("delay_saturated",
                                  f"[inference] delay saturated: {max_delay} >= max_delay {self.trained_max_delay}; "
                                  f"inference slower than training-time max_delay, so RTC conditioning is unreliable; ",
                                  freq=1.0, level="warning")

                    self._cond.release()
                    inference_start = time.perf_counter()

                    if rewarm:
                        # trained_max_delay is a training-time budget; a no-rtc ckpt
                        # has none, so fall back to the execution horizon.
                        _rewarm_keep = (self.trained_max_delay if self.rtc_mode != "test_time"
                                        else self.min_exec_horizon)
                        shifted_prev_action = shift_actions_with_zero_pad(self.action_chunk, self.action_chunk.shape[0] - _rewarm_keep)  # (H, D)
                    else:
                        shifted_prev_action = shift_actions_with_zero_pad(self.action_chunk, executed_ticks)  # (H, D)

                    next_action_chunk = self._predict_action_rtc(current_obs, shifted_prev_action, max_delay)
                    update_dashboard(warn="") # back to normal state

                    infference_end = time.perf_counter()
                    self._cond.acquire()
                    infer_ms = (infference_end - inference_start) * 1000
                    # Every forward attempt gets a replan id, including forwards
                    # that the superseded check below discards.
                    self.replan_seq += 1
                    replan_id = self.replan_seq

                    # DEBUG
                    if infer_ms > self._max_infer_ms and infer_ms > 100: # log if latency exceeds 100ms, which is a common threshold for "too slow for real-time"
                        overwatch.info(f"[inference] debug latency: max!!! {infer_ms:.2f}ms (executed {executed_ticks} ticks, delay {max_delay})")
                    else:
                        overwatch.info(f"[inference] debug latency: {infer_ms:.2f}ms ")

                    self._max_infer_ms = max(self._max_infer_ms, infer_ms)

                    # Superseded-condition discard (plan §6.2-4): if the desired condition
                    # KEY changed while this forward ran, the result was computed for a
                    # dead (instruction, goal) pair — never install it as the action chunk
                    # (nor its tick/delay bookkeeping); replan immediately on the newest obs.
                    live_condition = (self.current_obs or {}).get('condition_meta')
                    if _condition_key(snapshot_condition) != _condition_key(live_condition):
                        overwatch.info(
                            f"[inference] superseded: chunk for condition "
                            f"{_condition_key(snapshot_condition)} discarded "
                            f"(now {_condition_key(live_condition)}); replanning")
                        update_dashboard(warn="superseded chunk discarded")
                        self._force_replan = True
                        continue

                    # The measured delay = control ticks that elapsed during the forward,
                    # in BOTH branches (the old code left infer_delay unbound on the
                    # rewarm path: UnboundLocalError on a first-iteration rewarm, a stale
                    # append afterwards).
                    measured_delay = self.ticks_since_replan - executed_ticks

                    # update the action chunk and reset the tick counter;
                    # the next_action calls will start consuming from the new chunk;
                    # skip "measured_delay" steps that happened during inference
                    self.action_chunk = next_action_chunk
                    self.chunk_condition = snapshot_condition
                    self.chunk_id += 1
                    self.chunk_replan_id = replan_id
                    self.chunk_infer_ms = infer_ms
                    if self._chunk_recorder is not None:
                        self._chunk_recorder.record(
                            next_action_chunk, chunk_id=self.chunk_id,
                            replan_id=replan_id, infer_ms=infer_ms,
                            meta={"measured_delay": int(measured_delay),
                                  "rewarm": bool(rewarm),
                                  "rtc_mode": self.rtc_mode})

                    if rewarm:
                        self.ticks_since_replan = self.trained_max_delay
                    else:
                        self.ticks_since_replan = measured_delay

                    self.inference_delays.append(measured_delay)

                    update_dashboard(infer_ms=infer_ms, max_infer_ms=self._max_infer_ms,
                                     infer_executed=executed_ticks,
                                     infer_delay=max_delay, infer_ticks=self.ticks_since_replan)
                    log_every("inference",
                              f"[inference] latency={infer_ms/1000:.4f}s  executed={executed_ticks}  d={max_delay}  ticks_since_replan={self.ticks_since_replan}",
                              freq=2.0)
                except Exception:
                    overwatch.critical("Inference loop crashed — stopping program", exc_info=True)
                    update_dashboard(status="server crashed")
                    time.sleep(0.5)  # let the dashboard render thread paint the status before we exit
                    os._exit(1)

    def _predict_action_rtc(self, observation, prev_actions, inference_delay):
        # Stack cameras time-major: V x (T,C,H,W) → (T*V,C,H,W); tensor_to_pil_list runs inside predict_action
        cameras = list(observation['video'].values())
        obs_tensor = torch.stack(cameras, dim=1).reshape(-1, *cameras[0].shape[1:])  # (T*V,C,H,W)
        goal_image = observation.get('goal_image')
        prev = torch.from_numpy(prev_actions[np.newaxis, :, :]).to(self.policy.device)  # (H,Da) -> (1,H,Da)
        goals = [goal_image] if goal_image is not None else None
        if self.rtc_mode == "off":
            # True open-loop: no continuity mechanism at all, same call the warm-up
            # seed uses. Without this branch "off" fell through to the frozen-prefix
            # path below, i.e. sigma-0 inpainting on whatever checkpoint is loaded --
            # the worst case for a no-rtc one, and not an open-loop baseline at all.
            return self._predict_action(observation)
        if self.rtc_mode == "test_time":
            # Inference-time guidance: no sigma-0 prefix, so a checkpoint trained
            # WITHOUT rtc stays in-distribution. trained_max_delay is meaningless
            # here (the ckpt never had a delay budget) and is not passed.
            # The goal image is a plain input on this path, not CFG-only.
            actions = self.policy.predict_action_with_test_time_rtc(
                observations=[obs_tensor],
                states=observation['states'].unsqueeze(0).to(self.policy.device),
                instructions=[observation['instruction']],
                num_inference_steps=8,
                prev_actions=prev,
                inference_delay=inference_delay,
                execution_horizon=self.min_exec_horizon,
                goal_images=goals,
                mask_schedule=self.pig_mask_schedule,
                guidance_weight=self.pig_guidance_weight,
                guidance_sigma_threshold=self.pig_sigma_threshold,
                guidance_mode=self.pig_guidance_mode,
                guidance_alpha=self.pig_guidance_alpha,
            )[0].float().detach().cpu().numpy()
        else:
            actions = self.policy.predict_action_with_training_rtc_flow(
                observations=[obs_tensor],
                states=observation['states'].unsqueeze(0).to(self.policy.device),  # (Ts,Ds) -> (1,Ts,Ds)
                instructions=[observation['instruction']],
                num_inference_steps=8,
                prev_actions=prev,
                inference_delay=inference_delay,
                max_delay=self.trained_max_delay,
                # Forward the goal ALWAYS. It is not a CFG-only input: _build_qwen_inputs
                # appends it to the prompt whenever it is not None, and cfg_scale only
                # decides whether a second unconditional row is built. Gating it on
                # cfg_scale != 1.0 (our standing config is 1.0) silently ran this
                # goal-conditioned checkpoint with no goal frame at all.
                goal_images=goals,
                cfg_scale=self.cfg_scale,
                cfg_uncond_strip_subtask=self.cfg_uncond_strip_subtask,
            )[0].float().detach().cpu().numpy()  # (1, H, Da) -> (H, Da)
        return actions

    def _predict_action(self, o):
        # Stack cameras time-major: V x (T,C,H,W) → (T*V,C,H,W); tensor_to_pil_list runs inside predict_action
        cameras = list(o['video'].values())
        obs_tensor = torch.stack(cameras, dim=1).reshape(-1, *cameras[0].shape[1:])  # (T*V,C,H,W)
        goal_image = o.get('goal_image')
        actions = self.policy.predict_action(
            observations=[obs_tensor],
            states=[o['states'].to(self.policy.device)],
            instructions=[o['instruction']],
            num_inference_steps=8,
            goal_images=[goal_image] if goal_image is not None else None,
            cfg_scale=self.cfg_scale,
            cfg_uncond_strip_subtask=self.cfg_uncond_strip_subtask,
        )[0].float().detach().cpu().numpy()  # (1, H, Da) -> (H, Da)
        return actions
