"""Debug a real 7-DoF arm (Franka Panda or KUKA LBR iiwa 14) with DebugDraw.

The robot models come unmodified from MuJoCo Menagerie
(https://github.com/google-deepmind/mujoco_menagerie, Apache-2.0). They are
not vendored here. Point MUJOCO_MENAGERIE_PATH at a checkout, or let this
script fetch the two robot folders (a sparse git clone, ~55 MB) into
~/.cache/mujoco-debugdraw/.

Run:
    python examples/manipulator_demo.py                 # Franka Panda, passive viewer
    python examples/manipulator_demo.py --robot iiwa    # KUKA LBR iiwa 14
    MUJOCO_GL=egl python examples/manipulator_demo.py --robot panda --record assets/panda.gif
(macOS: use mjpython for the interactive viewer)

What's drawn (all at runtime; the Menagerie XML is loaded as-is):
  cyan      planned TCP path (figure-eight)
  magenta   actual TCP trail (last ~2 s), so tracking error is visible as a gap
  lime      current target point, with faint RGB axes for the desired orientation
  RGB       actual TCP frame (bold) and every link frame (small)
  yellow    TCP velocity
  HUD       position/orientation error, speed, and joint-limit usage per joint
"""

from __future__ import annotations

import argparse
import collections
import os
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from mujoco_debugdraw import DebugDraw

HERE = pathlib.Path(__file__).resolve().parent
MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie"


# ----------------------------------------------------------------------------- robots


@dataclass
class RobotConfig:
    name: str
    folder: str
    scene: str
    ee_body: str
    tcp_offset: np.ndarray  # TCP position in ee_body frame
    home: np.ndarray  # 7 arm joint angles
    path_center: np.ndarray
    path_size: tuple[float, float, float]  # (x wobble, y half-width, z amplitude)
    extra_ctrl: dict[int, float] = field(default_factory=dict)  # e.g. gripper open
    camera: tuple[float, float, float, list[float]] = (1.45, 200.0, -18.0, [0.40, 0.0, 0.45])
    home_R: np.ndarray = field(default_factory=lambda: np.eye(3))  # filled in by setup()


ROBOTS = {
    "panda": RobotConfig(
        name="Franka Emika Panda",
        folder="franka_emika_panda",
        scene="scene.xml",
        ee_body="hand",
        tcp_offset=np.array([0.0, 0.0, 0.1034]),  # flange -> between fingertips
        home=np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]),  # Menagerie "home" key
        path_center=np.array([0.50, 0.0, 0.35]),
        path_size=(0.06, 0.26, 0.14),
        extra_ctrl={7: 255.0},  # actuator8: gripper fully open
    ),
    "iiwa": RobotConfig(
        name="KUKA LBR iiwa 14",
        folder="kuka_iiwa_14",
        scene="scene.xml",
        ee_body="link7",
        tcp_offset=np.array([0.0, 0.0, 0.045]),  # Menagerie attachment_site
        home=np.array([0.0, 0.55, 0.0, -1.35, 0.0, 1.25, 0.0]),
        path_center=np.array([0.55, 0.0, 0.42]),
        path_size=(0.06, 0.28, 0.14),
        camera=(1.65, 200.0, -18.0, [0.42, 0.0, 0.52]),
    ),
}

PERIOD = 6.0  # seconds per figure-eight


def menagerie_dir() -> pathlib.Path:
    env = os.environ.get("MUJOCO_MENAGERIE_PATH")
    if env:
        return pathlib.Path(env).expanduser()
    cache = pathlib.Path.home() / ".cache" / "mujoco-debugdraw" / "mujoco_menagerie"
    if not (cache / "franka_emika_panda").exists() or not (cache / "kuka_iiwa_14").exists():
        print(f"Fetching robot models from {MENAGERIE_URL} into {cache} ...", file=sys.stderr)
        cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not cache.exists():
                subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none",
                                "--sparse", MENAGERIE_URL, str(cache)], check=True)
            subprocess.run(["git", "-C", str(cache), "sparse-checkout", "set",
                            "franka_emika_panda", "kuka_iiwa_14"], check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            sys.exit(f"Could not fetch Menagerie ({exc}). Clone {MENAGERIE_URL} yourself and "
                     "set MUJOCO_MENAGERIE_PATH to it.")
    return cache


# ------------------------------------------------------------------------- trajectory


def target_pose(cfg: RobotConfig, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Figure-eight TCP target; orientation = the robot's home TCP orientation, gently yawing.

    Using each robot's own home orientation (rather than a fixed "tool down"
    matrix) keeps the wrist away from its joint limits on both arms.
    """
    a = 2 * np.pi * t / PERIOD
    wx, wy, wz = cfg.path_size
    pos = cfg.path_center + np.array([wx * np.sin(2 * a), wy * np.sin(a), wz * np.sin(2 * a)])
    yaw = 0.35 * np.sin(a)
    c, s = np.cos(yaw), np.sin(yaw)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return pos, rz @ cfg.home_R


def planned_path(cfg: RobotConfig, n: int = 120) -> np.ndarray:
    return np.array([target_pose(cfg, t)[0] for t in np.linspace(0, PERIOD, n)])


# ------------------------------------------------------------------------- controller


class DiffIK:
    """Damped-least-squares differential IK with nullspace posture control.

    Output goes to Menagerie's position actuators as joint setpoints.
    """

    def __init__(self, model: mujoco.MjModel, cfg: RobotConfig) -> None:
        self.m, self.cfg = model, cfg
        self.body = model.body(cfg.ee_body).id
        self.arm_dofs = np.arange(7)
        self.jnt_range = model.jnt_range[:7].copy()
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        self.damping, self.k_pos, self.k_rot, self.k_null = 0.05, 20.0, 15.0, 2.0
        self.dt = model.opt.timestep
        self.q_cmd = cfg.home.copy()  # integrated joint setpoint sent to the servos

    def tcp(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        R = data.xmat[self.body].reshape(3, 3)
        return data.xpos[self.body] + R @ self.cfg.tcp_offset, R

    def __call__(self, data: mujoco.MjData, p_des: np.ndarray, R_des: np.ndarray,
                 v_ff: np.ndarray | None = None):
        m = self.m
        p, R = self.tcp(data)
        mujoco.mj_jac(m, data, self.jacp, self.jacr, p, self.body)
        J = np.vstack([self.jacp[:, :7], self.jacr[:, :7]])

        e_pos = p_des - p
        q_cur, q_des = np.zeros(4), np.zeros(4)
        mujoco.mju_mat2Quat(q_cur, R.flatten())
        mujoco.mju_mat2Quat(q_des, R_des.flatten())
        q_err, q_conj = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(q_conj, q_cur)
        mujoco.mju_mulQuat(q_err, q_des, q_conj)
        e_rot = np.zeros(3)
        mujoco.mju_quat2Vel(e_rot, q_err, 1.0)

        twist = np.concatenate([self.k_pos * e_pos, self.k_rot * e_rot])
        if v_ff is not None:  # feed-forward the target's own velocity
            twist[:3] += v_ff
        JJt = J @ J.T + (self.damping**2) * np.eye(6)
        J_pinv = J.T @ np.linalg.solve(JJt, np.eye(6))
        qpos = data.qpos[:7]
        dq = J_pinv @ twist + (np.eye(7) - J_pinv @ J) @ (self.k_null * (self.cfg.home - qpos))
        self.q_cmd = np.clip(self.q_cmd + dq * self.dt, self.jnt_range[:, 0], self.jnt_range[:, 1])

        data.ctrl[:7] = self.q_cmd
        for idx, val in self.cfg.extra_ctrl.items():
            data.ctrl[idx] = val
        v = self.jacp[:, :7] @ data.qvel[:7]
        return p, R, v, e_pos, e_rot


# ---------------------------------------------------------------------------- drawing


def limit_bar(frac: float, width: int = 10) -> str:
    n = int(round(min(max(frac, 0.0), 1.0) * width))
    return "#" * n + "." * (width - n)


def draw_debug(draw: DebugDraw, data, ik: DiffIK, cfg: RobotConfig, path, trail,
               p_des, R_des, p, R, v, e_pos, e_rot) -> None:
    draw.clear()
    draw.line(path, color="cyan", width=2.0, key="plan")
    if len(trail) >= 2:
        draw.line(np.array(trail), color="magenta", width=5.0, key="trail")

    # Target: point + desired orientation (faint axes built from arrows).
    draw.point(p_des, color="lime", size=0.014, label="target", key="target")
    for i, col in enumerate(([1, .3, .3, .45], [.3, 1, .3, .45], [.3, .5, 1, .45])):
        draw.arrow(p_des, R_des[:, i], scale=0.07, width=0.003, color=col)

    # Actual TCP frame (bold) and every link frame (small).
    for i, col in enumerate(("red", "green", "blue")):
        draw.arrow(p, R[:, i], scale=0.10, width=0.005, color=col)
    for k in range(1, 8):
        draw.frame(f"link{k}", scale=0.14, width=0.003)

    draw.arrow(p, v, scale=0.4, width=0.004, color="yellow", key="vel")

    rot_err = np.degrees(np.linalg.norm(e_rot))
    draw.text(
        f"{cfg.name}\n"
        f"pos err  : {np.linalg.norm(e_pos) * 1000:6.1f} mm\n"
        f"rot err  : {rot_err:6.2f} deg\n"
        f"TCP speed: {np.linalg.norm(v):6.2f} m/s",
        position="top_left",
    )
    lo, hi = ik.jnt_range[:, 0], ik.jnt_range[:, 1]
    q = data.qpos[:7]
    usage = np.abs(q - (lo + hi) / 2) / ((hi - lo) / 2)  # 0 = centered, 1 = at a limit
    rows = [f"j{i + 1} [{limit_bar(u)}] {u * 100:3.0f}%{'  !' if u > 0.9 else ''}"
            for i, u in enumerate(usage)]
    draw.text("joint-limit usage\n" + "\n".join(rows), position="top_right")
    draw.text("cyan plan | magenta actual | lime target | yellow velocity", position="bottom_left")


# ------------------------------------------------------------------------------- main


def setup(robot: str):
    cfg = ROBOTS[robot]
    xml = menagerie_dir() / cfg.folder / cfg.scene
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    data.qpos[:7] = cfg.home
    data.ctrl[:7] = cfg.home
    for idx, val in cfg.extra_ctrl.items():
        data.ctrl[idx] = val
    mujoco.mj_forward(model, data)
    ik = DiffIK(model, cfg)
    cfg.home_R = ik.tcp(data)[1].copy()
    return cfg, model, data, ik, planned_path(cfg)


def step_frame(model, data, ik, cfg, trail, n_steps):
    for _ in range(n_steps):
        p_des, R_des = target_pose(cfg, data.time)
        v_ff = (target_pose(cfg, data.time + 1e-3)[0] - p_des) / 1e-3
        out = ik(data, p_des, R_des, v_ff)
        mujoco.mj_step(model, data)
    trail.append(out[0].copy())
    return (p_des, R_des) + out


def run_viewer(robot: str) -> None:
    import mujoco.viewer

    cfg, model, data, ik, path = setup(robot)
    trail: collections.deque = collections.deque(maxlen=120)
    steps = max(1, int(round((1 / 60) / model.opt.timestep)))
    with mujoco.viewer.launch_passive(model, data) as viewer:
        dist, az, el, look = cfg.camera
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = dist, az, el
        viewer.cam.lookat[:] = look
        draw = DebugDraw(model, data, viewer)
        while viewer.is_running():
            t0 = time.time()
            state = step_frame(model, data, ik, cfg, trail, steps)
            draw_debug(draw, data, ik, cfg, path, trail, *state)
            draw.sync()
            viewer.sync()
            time.sleep(max(0.0, steps * model.opt.timestep - (time.time() - t0)))


def record(robot: str, out: pathlib.Path, width: int = 640, height: int = 440, fps: int = 15) -> None:
    from PIL import Image

    sys.path.insert(0, str(HERE))
    from record_demo import hud  # same Pillow HUD compositor as the README GIF

    cfg, model, data, ik, path = setup(robot)
    trail: collections.deque = collections.deque(maxlen=int(2.0 * fps))
    renderer = mujoco.Renderer(model, height, width)
    cam = mujoco.MjvCamera()
    dist, az, el, look = cfg.camera
    cam.distance, cam.azimuth, cam.elevation = dist, az, el
    cam.lookat[:] = look
    draw = DebugDraw(model, data, renderer)
    steps = int(round(1.0 / (fps * model.opt.timestep)))

    for _ in range(int(1.5 * fps)):  # settle onto the path before recording
        step_frame(model, data, ik, cfg, trail, steps)
    frames = []
    for _ in range(int(PERIOD * fps)):
        state = step_frame(model, data, ik, cfg, trail, steps)
        draw_debug(draw, data, ik, cfg, path, trail, *state)
        renderer.update_scene(data, camera=cam)
        draw.sync()
        frames.append(hud(renderer.render(), draw.texts, width, height, font_size=13))
    renderer.close()

    pal = [Image.fromarray(f).quantize(colors=256, method=Image.Quantize.MAXCOVERAGE,
                                       dither=Image.Dither.NONE) for f in frames]
    out.parent.mkdir(parents=True, exist_ok=True)
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=int(1000 / fps), loop=0,
                optimize=True)
    print(f"wrote {out} ({len(frames)} frames)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", choices=sorted(ROBOTS), default="panda")
    ap.add_argument("--record", type=pathlib.Path, help="render a GIF offscreen instead of opening the viewer")
    args = ap.parse_args()
    if args.record:
        record(args.robot, args.record)
    else:
        run_viewer(args.robot)


if __name__ == "__main__":
    main()
