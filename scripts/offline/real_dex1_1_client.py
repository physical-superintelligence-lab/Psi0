"""真实 Dex1-1夹爪驱动:实现当前SONIC策略桥使用的send/close接口。

职责:把策略输出的归一化夹爪指令 Dex11Command(left/right ∈ [0,1])映射为夹爪空间
弧度 q,做钳制与限速,再交给 DDS 控制器持续发布。

安全设计:
* 默认 ``dry_run=True``,只计算与记录,**不向总线发布**;需显式 ``dry_run=False``。
* 每次 send 做限速(单步 q 变化不超过 ``max_step_rad``),避免突跳。
* DDS 控制器(robot_hand_dex1_1.Dex1_1_Controller)**懒加载**,因此本模块在缺少
  unitree_sdk2py 的环境(如 .venv-psi)也能 import,纯映射逻辑可单测。

方向(norm 的 0/1 对应张开还是闭合)在硬件 T9 阶段物理确认前未定,用 ``invert``
与 ``q_closed/q_open`` 配置;默认 norm=0->q_closed(闭合),norm=1->q_open(张开)。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.offline.dex1_1_layout import Dex11Command  # noqa: E402

# 夹爪空间安全范围(rad),与 robot_hand_dex1_1 保持一致
Q_CLOSED_DEFAULT = 0.0
Q_OPEN_DEFAULT = 5.5


@dataclass(frozen=True)
class Dex11Calibration:
    """归一化 [0,1] <-> 夹爪空间 q(rad) 的标定。"""

    q_closed: float = Q_CLOSED_DEFAULT
    q_open: float = Q_OPEN_DEFAULT
    invert: bool = False  # True: norm=0 对应张开(待 T9 物理确认后设定)

    def norm_to_q(self, norm: float) -> float:
        n = float(np.clip(norm, 0.0, 1.0))
        if self.invert:
            n = 1.0 - n
        return self.q_closed + n * (self.q_open - self.q_closed)

    def q_to_norm(self, q: float) -> float:
        span = self.q_open - self.q_closed
        if span == 0:
            return 0.0
        n = (float(q) - self.q_closed) / span
        if self.invert:
            n = 1.0 - n
        return float(np.clip(n, 0.0, 1.0))


def rate_limit(prev_q: float, target_q: float, max_step_rad: float) -> float:
    """把 target_q 限制在 prev_q ± max_step_rad 内(单步限速)。"""
    if max_step_rad is None or max_step_rad <= 0:
        return float(target_q)
    delta = float(np.clip(target_q - prev_q, -max_step_rad, max_step_rad))
    return float(prev_q + delta)


@dataclass
class RealDex11Driver:
    """实现 Dex11Driver 协议(send(Dex11Command) -> None)。"""

    network: str = "enp4s0"
    calib: Dex11Calibration = field(default_factory=Dex11Calibration)
    kp: float = 5.0
    kd: float = 0.05
    max_step_rad: float = 0.5
    dry_run: bool = True
    controller: object | None = None  # 可注入(测试用 fake)
    _last_q: dict = field(default_factory=lambda: {"left": None, "right": None}, init=False)
    last_command: dict | None = field(default=None, init=False)

    def _ensure_controller(self):
        if self.controller is not None or self.dry_run:
            return
        # 懒加载真实 DDS 控制器
        from real.teleop.robot_control.robot_hand_dex1_1 import Dex1_1_Controller

        self.controller = Dex1_1_Controller(
            network=self.network, kp=self.kp, kd=self.kd
        )
        self.controller.start_publishing()

    def _seed_last_q(self, side: str, target_q: float) -> float:
        """首次 send 时的限速基准:优先用真实回读状态,否则用目标本身。"""
        if self._last_q[side] is not None:
            return self._last_q[side]
        if self.controller is not None and hasattr(self.controller, "get_current_dual_gripper_q"):
            lq, rq = self.controller.get_current_dual_gripper_q()
            cur = lq if side == "left" else rq
            if cur is not None:
                return float(cur)
        return float(target_q)

    def send(self, command: Dex11Command) -> None:
        self._ensure_controller()
        target = {
            "left": self.calib.norm_to_q(command.left),
            "right": self.calib.norm_to_q(command.right),
        }
        out = {}
        for side in ("left", "right"):
            base = self._seed_last_q(side, target[side])
            q = rate_limit(base, target[side], self.max_step_rad)
            self._last_q[side] = q
            out[side] = q

        self.last_command = {
            "norm_left": float(command.left),
            "norm_right": float(command.right),
            "q_left": out["left"],
            "q_right": out["right"],
            "dry_run": self.dry_run,
        }

        if not self.dry_run and self.controller is not None:
            self.controller.ctrl_dual_gripper(out["left"], out["right"])

    def get_state_norm(self) -> tuple[float | None, float | None]:
        if self.controller is None or not hasattr(self.controller, "get_current_dual_gripper_q"):
            return None, None
        lq, rq = self.controller.get_current_dual_gripper_q()
        ln = self.calib.q_to_norm(lq) if lq is not None else None
        rn = self.calib.q_to_norm(rq) if rq is not None else None
        return ln, rn

    def close(self):
        if self.controller is not None and hasattr(self.controller, "close"):
            self.controller.close()


# ---------------------------------------------------------------------------
# 带运动的台架验证 CLI(T2/T9 方向验证用)。默认 dry-run;--live 才真正发指令。
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Dex1-1 真实驱动台架验证(默认 dry-run,不发指令)")
    p.add_argument("--network", default="enp4s0")
    p.add_argument("--side", choices=["left", "right", "both"], default="left")
    p.add_argument("--target", type=float, default=0.3, help="目标归一化 [0,1]")
    p.add_argument("--q-closed", type=float, default=Q_CLOSED_DEFAULT)
    p.add_argument("--q-open", type=float, default=Q_OPEN_DEFAULT)
    p.add_argument("--invert", action="store_true")
    p.add_argument("--kp", type=float, default=5.0)
    p.add_argument("--kd", type=float, default=0.05)
    p.add_argument("--max-step-rad", type=float, default=0.15, help="单步限速(台架默认更保守)")
    p.add_argument("--hz", type=float, default=20.0, help="台架 send 频率")
    p.add_argument("--ramp-sec", type=float, default=3.0, help="从当前位渐变到目标的时长")
    p.add_argument("--hold-sec", type=float, default=1.0)
    p.add_argument("--live", action="store_true", help="真正发指令让夹爪运动(否则仅 dry-run)")
    p.add_argument("--confirm", action="store_true", help="--live 时必须同时给出,二次确认安全")
    args = p.parse_args()

    calib = Dex11Calibration(q_closed=args.q_closed, q_open=args.q_open, invert=args.invert)
    target_norm = float(np.clip(args.target, 0.0, 1.0))

    if not args.live:
        # 纯映射演示
        drv = RealDex11Driver(calib=calib, dry_run=True, max_step_rad=args.max_step_rad)
        l = target_norm if args.side in ("left", "both") else 0.0
        r = target_norm if args.side in ("right", "both") else 0.0
        drv.send(Dex11Command(left=l, right=r))
        print(json.dumps({"mode": "dry_run", **drv.last_command,
                          "q_open": args.q_open, "q_closed": args.q_closed,
                          "invert": args.invert, "side": args.side}, indent=2, ensure_ascii=False))
        print("\n仅 dry-run。确认无误且现场安全(空载/无障碍/急停可用)后,加 --live --confirm 让夹爪运动。")
        return

    if not args.confirm:
        print("拒绝执行:--live 必须同时加 --confirm(确认空载、夹爪周围无障碍、急停可用)。")
        sys.exit(2)

    print("=== LIVE:将真正驱动夹爪运动 ===")
    print(f"side={args.side} target_norm={target_norm} -> q={calib.norm_to_q(target_norm):.3f} rad "
          f"(q_closed={args.q_closed}, q_open={args.q_open}, invert={args.invert})")
    drv = RealDex11Driver(network=args.network, calib=calib, kp=args.kp, kd=args.kd,
                          max_step_rad=args.max_step_rad, dry_run=False)
    try:
        # 斜坡起点取当前实测位;未选中的爪保持当前位(不动)
        drv._ensure_controller()
        lq0, rq0 = drv.controller.get_current_dual_gripper_q()
        start_l = calib.q_to_norm(lq0) if lq0 is not None else 0.0
        start_r = calib.q_to_norm(rq0) if rq0 is not None else 0.0
        tgt_l = target_norm if args.side in ("left", "both") else start_l
        tgt_r = target_norm if args.side in ("right", "both") else start_r
        print(f"start_norm  left={start_l:.3f}(q={lq0})  right={start_r:.3f}(q={rq0})")
        print(f"target_norm left={tgt_l:.3f}  right={tgt_r:.3f}  (未选中的爪保持不动)")

        n_steps = max(1, int(args.ramp_sec * args.hz))
        period = 1.0 / args.hz
        for i in range(n_steps):
            frac = (i + 1) / n_steps
            l = start_l + (tgt_l - start_l) * frac
            r = start_r + (tgt_r - start_r) * frac
            drv.send(Dex11Command(left=l, right=r))
            lq, rq = drv.controller.get_current_dual_gripper_q()
            print(f"  step {i+1}/{n_steps} cmd_q={drv.last_command['q_left']:.3f}/"
                  f"{drv.last_command['q_right']:.3f}  state_q={lq}/{rq}")
            time.sleep(period)
        print(f"hold {args.hold_sec}s ...")
        time.sleep(args.hold_sec)
    finally:
        print("停止发布(电机将 BRAKE)。")
        drv.close()


if __name__ == "__main__":
    main()
