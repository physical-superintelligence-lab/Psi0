"""HTTP (POST /act) server for the sonic Psi0 checkpoints.

Thin entry point over `serve_psi0_simple.Server`, which is now format-agnostic: it
handles both the simple packs (SimpleRepackTransform / ActionStateTransform, 2-D
state history) and the sonic ones (SonicRepackTransform / SonicActionStateTransform,
flat state vector, body_token ++ hand action layout, e.g. the neckless 78-D
`sonic-wbcbox.*` finetunes), and it supports RTC on both:

  --rtc                     enable real-time chunking (needs --action-exec-horizon < Tp)
  --rtc-mode auto           train-time frozen prefix if the ckpt has --model.rtc,
                            otherwise test-time guidance  (also: off | train | test_time)

This file existed because the older server only spoke the simple format and had no
test-time RTC path; both gaps are closed in serve_psi0_simple now, so the two entry
points differ only in their default denoising step count (8 here, matching the RTC
WebSocket sonic server; 10 there).

Example:
    serve_psi0_sonic_http --policy psi0 --port 8014 --ckpt-step 40000 --rtc \
        --action-exec-horizon 15 \
        --run-dir .runs/finetune/sonic-wbcbox.neckle.flow1000.cosine.lr1.0e-04.b256.gpus8.2608260223
"""
import sys
from pathlib import Path

import tyro

from psi.config.config import ServerConfig
from psi.deploy.serve_psi0_simple import Server
from psi.utils.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)


def serve(cfg: ServerConfig) -> None:
    overwatch.info("Server :: Initializing Sonic Psi0 (HTTP)")
    assert cfg.policy is not None, "which policy to serve?"
    server = Server(
        cfg.policy,
        Path(cfg.run_dir),
        cfg.ckpt_step,
        cfg.device,
        cfg.rtc,
        cfg.action_exec_horizon,
        rtc_mode=cfg.rtc_mode,
        pig_mask_schedule=cfg.pig_mask_schedule,
        pig_guidance_alpha=cfg.pig_guidance_alpha,
        num_inference_steps=cfg.num_inference_steps,
        rtc_inference_delay=cfg.rtc_inference_delay,
        min_exec_horizon=cfg.min_exec_horizon,
    )
    overwatch.info("Server :: Spinning Up")
    server.run(cfg.host, cfg.port)


def main():
    overwatch.info("Start Serving from uv")
    overwatch.info(f"Args: {sys.argv}")
    from dotenv import load_dotenv
    load_dotenv()
    config = tyro.cli(ServerConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=sys.argv[1:])
    serve(config)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    config = tyro.cli(ServerConfig, config=(tyro.conf.ConsolidateSubcommandArgs,))
    serve(config)
