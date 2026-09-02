"""Run SONIC's MuJoCo loop with exactly one Unitree DDS initialization.

Upstream ``run_sim_loop`` initializes DDS before constructing ``BaseSimulator``.
``BaseSimulator`` initializes the same channel again, which makes the second
CycloneDDS domain construction fail (the upstream troubleshooting guide tracks
this as issue #77).  Keep the upstream simulator unchanged and suppress only
the redundant outer initialization.
"""

from __future__ import annotations

from typing import Any


def disable_outer_channel_init(run_sim_loop_module: Any) -> None:
    """Let ``BaseSimulator`` own the single DDS channel initialization."""

    run_sim_loop_module.init_channel = lambda *args, **kwargs: None


def main() -> None:
    import tyro
    from gear_sonic.scripts import run_sim_loop

    disable_outer_channel_init(run_sim_loop)
    print("[Psi0 SIM] using BaseSimulator's single DDS initialization", flush=True)
    run_sim_loop.main(tyro.cli(run_sim_loop.ArgsConfig))


if __name__ == "__main__":
    main()
