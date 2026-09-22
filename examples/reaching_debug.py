"""Reaching-controller debug demo for mujoco_debugdraw.

Run:  python examples/reaching_debug.py
(macOS: use `mjpython examples/reaching_debug.py` — required by launch_passive)

A 3-DoF arm tracks a target moving along a planned circular path with a
Jacobian-transpose controller. Everything you see besides the arm is drawn
at runtime by DebugDraw; scene.xml contains no debug geometry.
"""

from __future__ import annotations

import pathlib
import time

import mujoco
import mujoco.viewer
import numpy as np

from mujoco_debugdraw import DebugDraw

HERE = pathlib.Path(__file__).resolve().parent
CENTER = np.array([0.40, 0.0, 0.30])
RADIUS = 0.12
PERIOD = 4.5  # seconds per lap


def planned_path(n: int = 64) -> np.ndarray:
    """The circle the target follows, tilted so it is visible in 3D."""
    t = np.linspace(0.0, 2 * np.pi, n)
    return CENTER + np.stack(
        [0.35 * RADIUS * np.sin(t), RADIUS * np.cos(t), RADIUS * np.sin(t)], axis=1
    )


def target_at(time_s: float) -> np.ndarray:
    a = 2 * np.pi * time_s / PERIOD
    return CENTER + np.array([0.35 * RADIUS * np.sin(a), RADIUS * np.cos(a), RADIUS * np.sin(a)])


class ReachController:
    """tau = J^T (Kp * e - Kd * v) + gravity/Coriolis compensation."""

    def __init__(self, model: mujoco.MjModel, kp: float = 300.0, kd: float = 25.0) -> None:
        self.site = model.site("gripper_site").id
        self.jacp = np.zeros((3, model.nv))
        self.kp, self.kd = kp, kd

    def __call__(self, model, data, target):
        mujoco.mj_jacSite(model, data, self.jacp, None, self.site)
        pos = data.site_xpos[self.site].copy()
        vel = self.jacp @ data.qvel
        err = target - pos
        force = self.kp * err - self.kd * vel
        data.ctrl[:] = np.clip(self.jacp.T @ force + data.qfrc_bias, -30, 30)
        return pos, vel, err


def draw_debug(draw: DebugDraw, data, ctrl: ReachController, target, pos, vel, err, path) -> None:
    draw.clear()
    draw.line(path, color="cyan", width=3.0, key="plan")
    draw.point(target, color="lime", size=0.022, label="target", key="target")
    draw.frame("end_effector", scale=0.14)
    draw.frame("upper_arm", scale=0.15)
    draw.arrow(pos, err, color="orange", width=0.006, key="err")  # position error, true length
    draw.arrow(pos, vel, color="yellow", scale=0.5, width=0.006, key="vel")
    draw.text(
        f"EE error : {np.linalg.norm(err) * 100:5.1f} cm\n"
        f"EE speed : {np.linalg.norm(vel):5.2f} m/s\n"
        f"sim time : {data.time:5.1f} s",
        position="top_left",
    )
    draw.text("cyan: plan   lime: target   yellow: EE velocity\norange: position error   RGB: body frames (x/y/z)", position="bottom_left")


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(HERE / "scene.xml"))
    data = mujoco.MjData(model)
    data.qpos[:] = [0.0, 0.6, 0.6]
    ctrl = ReachController(model)
    path = planned_path()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.3, 0.0, 0.25]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.05, 60, -20
        draw = DebugDraw(model, data, viewer)

        while viewer.is_running():
            frame_start = time.time()
            target = target_at(data.time)
            pos, vel, err = ctrl(model, data, target)
            for _ in range(8):  # 8 physics steps per rendered frame (~60 Hz)
                mujoco.mj_step(model, data)

            draw_debug(draw, data, ctrl, target, pos, vel, err, path)
            draw.sync()   # always before viewer.sync()
            viewer.sync()

            time.sleep(max(0.0, 8 * model.opt.timestep - (time.time() - frame_start)))


if __name__ == "__main__":
    main()
