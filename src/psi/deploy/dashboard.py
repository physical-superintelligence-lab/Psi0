"""Live htop/nvtop-style terminal dashboard for the psix RTC server.

Producer threads (asyncio receive, control loop, inference loop) write metrics into a
single lock-guarded dict via update_dashboard(); one daemon render thread (dashboard_loop) redraws
them with rich.Live. The dashboard is opt-in (ServerConfig.dashboard) and only engages on a
TTY — headless/slurm runs fall back to the line-based log_every output untouched.
"""
import os
import threading
import time
from typing import Any, Dict

import numpy as np

from psi.utils.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)


# Throttled logging: emit `msg` at most `freq` times/sec per `key`, regardless of
# how often the call is hit. First call for a key always fires.
_log_throttle: Dict[str, float] = {}

def log_every(key: str, msg: str, freq: float = 1.0, level: str = "info") -> None:
    now = time.monotonic()
    if now - _log_throttle.get(key, -float("inf")) >= 1.0 / freq:
        _log_throttle[key] = now
        getattr(overwatch, level)(msg)


def redirect_logs_to_file(log_dir: str = ".logs", prefix: str = "serve_psix") -> str:
    """Detach console handlers and send all log records to a file instead.

    The live dashboard owns the terminal, so the Rich console handler would
    scribble over it. We swap it for a FileHandler so nothing is lost — the
    INFO lines that used to print now land in this file.
    """
    import logging

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.log")

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )

    root = logging.getLogger()
    for h in list(root.handlers):  # drop console (Rich) handlers
        root.removeHandler(h)
    root.addHandler(file_handler)
    root.setLevel(logging.INFO)
    return log_path


def shift_actions_with_zero_pad(actions: np.ndarray, shift: int) -> np.ndarray:
    """Drop the first `shift` action steps and zero-pad `shift` rows at the end,
    keeping the chunk length (H) fixed. Builds the prev_actions for RTC inference.
    Returns a fresh array (independent of `actions`)."""
    shift = min(shift, actions.shape[0])  # clamp so an overrun still returns length H
    pad = np.zeros((shift, actions.shape[1]), dtype=actions.dtype)
    return np.concatenate([actions[shift:, :], pad], axis=0)  # (H, D)


_stats: Dict[str, Any] = {}
_stats_lock = threading.Lock()


def update_dashboard(**kwargs: Any) -> None:
    with _stats_lock:
        _stats.update(kwargs)


# Persistent connection/config context kept across a reset; everything else (peak latency,
# warnings, counters, thumbnail, obs/history) is dropped.
_RESET_KEEP = {"connected", "status", "obs_hz_target", "ctrl_target_ms", "error", "conn_id"}

def reset_dashboard() -> None:
    """Drop accumulated/transient dashboard metrics, keeping connection + config context."""
    with _stats_lock:
        for k in list(_stats):
            if k not in _RESET_KEEP:
                _stats.pop(k, None)


def dashboard_loop(stop_event: "threading.Event", start_time: float, refresh_hz: float = 3.0,
                   log_path: str | None = None, title: str | None = None) -> None:
    """`title` overrides the panel heading, so a server other than serve_psix
    (e.g. serve_psi0_sonic) names itself; None keeps the psix titles."""
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel

    def _fmt(v, spec="{}", dash="—"):
        return dash if v is None else spec.format(v)

    def render() -> "Panel":
        with _stats_lock:
            s = dict(_stats)
        uptime = time.monotonic() - start_time
        conn = s.get("connected", False)
        status = s.get("status", "—")
        # Yellow while warming up, green once live.
        status_color = "green" if status == "live" else ("yellow" if "warm" in status else "white")

        t = Table.grid(padding=(0, 2))
        t.add_column(justify="right", style="cyan", no_wrap=True)
        t.add_column(style="white")
        if conn == "disconnected":
            conn_text = "[bold red]x disconnected[/bold red]"
        elif conn:
            conn_text = "[green]● connected[/green]"
        else:
            conn_text = "[red]○ waiting[/red]"
        conn_id = s.get("conn_id")
        if conn_id is not None:
            conn_text += f"  [dim]#{conn_id}[/dim]"
        t.add_row("connection", conn_text)
        t.add_row("status", f"[{status_color}]{status}[/{status_color}]")
        t.add_row("instruction", f"[italic]{_fmt(s.get('instruction'))}[/italic]")

        # basics shown for BOTH panel types (only rows we actually have data for)
        def row_if(label, val):
            if val is not None:
                t.add_row(label, str(val))
        row_if("device", s.get("device"))
        if s.get("hlp_backend") or s.get("hlp_ckpt"):
            t.add_row("backend / ckpt", f"{_fmt(s.get('hlp_backend'))} / {_fmt(s.get('hlp_ckpt'))}")
        row_if("memory format", s.get("mem_format"))
        row_if("image res", s.get("img_res"))

        is_hlp = any(k in s for k in ("hlp_current", "hlp_model_next", "hlp_history", "hlp_backend"))
        if is_hlp:
            t.add_row("current", _fmt(s.get("hlp_current")))
            t.add_row("model next", _fmt(s.get("hlp_model_next")))
            t.add_row("decision", f"model={_fmt(s.get('hlp_model_decision'))} / server={_fmt(s.get('hlp_server_decision'))}")
            t.add_row("stage / seconds", f"{_fmt(s.get('hlp_stage'))} / {_fmt(s.get('hlp_seconds'), '{:.1f}')}")
            t.add_row("HLP latency", f"{_fmt(s.get('hlp_total_ms'), '{:.1f}')} ms "
                                      f"(gen {_fmt(s.get('hlp_generate_ms'), '{:.1f}')} ms)")
            t.add_row("history", _fmt(s.get("hlp_history")))
        t.add_row("uptime", f"{uptime:6.0f} s")
        if not is_hlp:
            t.add_row("", "")
            t.add_row("obs in / target", f"{_fmt(s.get('obs_hz'), '{:.1f}')} / {_fmt(s.get('obs_hz_target'), '{:.1f}')} Hz")
            t.add_row("history img / state", f"{_fmt(s.get('hist_img'))} / {_fmt(s.get('hist_state'))}")
            # Image-feed liveness: how long since the last frame arrived (no thumbnail render).
            frame_t = s.get("frame_t")
            if frame_t is None:
                img_text = "[dim]—[/dim]"
            else:
                age = time.time() - frame_t
                cam = s.get("frame_cam", "?")
                img_text = (f"[green]● receiving[/green]  {cam}  ({age * 1000:.0f} ms ago)" if age < 1.0
                            else f"[red]○ stale[/red]  {cam}  ({age:.1f} s ago)")
            t.add_row("image feed", img_text)
            t.add_row("", "")
            t.add_row("infer latency", f"{_fmt(s.get('infer_ms'), '{:.1f}')} ms")
            t.add_row("executed / delay / ticks", f"{_fmt(s.get('infer_executed'))} / {_fmt(s.get('infer_delay'))} / {_fmt(s.get('infer_ticks'))}")
            t.add_row("max inference delay", f"{_fmt(s.get('max_infer_ms'), '{:.1f}')} ms")
            t.add_row("", "")
            t.add_row("ctrl interval", f"{_fmt(s.get('ctrl_ms'), '{:.1f}')} ms (target {_fmt(s.get('ctrl_target_ms'), '{:.1f}')})")
            t.add_row("ctrl missed ticks", f"{_fmt(s.get('ctrl_missed', 0))}")
            t.add_row("action version", f"{_fmt(s.get('action_version', 0))}")
            # Only servers that HAVE a reset path publish this; serve_psi0_sonic has
            # none (a disconnect rebuilds the controller), so the row stays hidden.
            row_if("reset requested", s.get("reset_requested"))
        warn = s.get("warn")
        if warn:
            t.add_row("", "")
            t.add_row("⚠ warn", f"[yellow]{warn}[/yellow]")
        error = s.get("error")
        if error:
            t.add_row("", "")
            t.add_row("✗ error", f"[bold red]{error}[/bold red]")

        # Frame the panel red once an exception has been surfaced so it's obvious at a glance.
        border = "red" if error else "green"
        subtitle = f"[dim]log: {log_path}[/dim]" if log_path else None
        panel_title = title or ("[bold]Psi-X HLP Server[/bold]" if is_hlp
                                else "[bold]Psi-X Policy Server[/bold]")
        return Panel(t, title=panel_title, subtitle=subtitle,
                     subtitle_align="left", border_style=border)

    with Live(render(), refresh_per_second=refresh_hz, screen=True) as live:
        while not stop_event.is_set():
            live.update(render())
            time.sleep(1.0 / refresh_hz)
