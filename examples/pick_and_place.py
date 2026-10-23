"""Tabletop pick and place with a Franka Panda: a raspberry, a light bulb, a rubber duck and two rocks.

Run:
    python examples/pick_and_place.py              # passive viewer; loops with a new layout each round
    python examples/pick_and_place.py --seed 7     # a different first layout
    MUJOCO_GL=egl python examples/pick_and_place.py --record assets/pick_and_place.gif   # 3x speed
(macOS: use mjpython)

The Panda is MuJoCo Menagerie's model, loaded unmodified (see manipulator_demo.py
for how it is fetched). The table, tray and objects are added at load time with
MjSpec (MuJoCo >= 3.3). The objects are built from MuJoCo primitives, and the rocks
are random convex meshes, so nothing extra is downloaded.

What's drawn for the object being picked (all at runtime):
  yellow box   its bounding box: green while held, red if it slips in the grip   box()
  white        its center of mass, labelled with name and mass                   body_com()
  orange       the grip: the contact force on each finger (0.08 m per N)         contacts()
  cyan         the remaining waypoints for this object                           line()
  RGB          the planned grasp pose, the target pose (faint), the TCP frame    frame_at()
  magenta      its path while carried
  white label  each phase change, marked where it happened for 1.5 s             ttl_frames
  green box    the tray's drop zone (drawn once)
  HUD          phase, grip force, finger gap, slip since grasp, progress
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import pathlib
import time
from dataclasses import dataclass
from typing import Callable, Optional, Union

import mujoco
import numpy as np

from manipulator_demo import ROBOTS, DiffIK, menagerie_dir
from mujoco_debugdraw import DebugDraw

TABLE_TOP = 0.25  # m
TABLE_CENTER, TABLE_HALF = np.array([0.55, 0.0]), np.array([0.25, 0.42])
TRAY_CENTER, TRAY_INNER = np.array([0.48, -0.26]), np.array([0.11, 0.10])
TRAY_FLOOR = TABLE_TOP + 0.006  # top of the tray's floor
# Drop spots relative to the tray center, in picking order: the corners, then the center
# for the bulb. Objects are placed with the fingers closing along x, so the wide hand
# lines up with the rows, which are 14 cm apart.
TRAY_SPOTS = [(-0.07, -0.055), (0.07, -0.055), (-0.07, 0.055), (0.07, 0.055), (0.0, 0.0)]
SAFE_Z = TABLE_TOP + 0.18  # TCP height for moving over the table
DROP = 0.002  # m, how far above the tray floor an object is released
OPEN, CLOSED = 255.0, 0.0  # gripper actuator commands
FORCE_SCALE = 0.08  # m per newton for the grip arrows (0.4-1.4 N per finger)
SLIP_WARN = 0.015  # m of drift in the grip that turns the bounding box red

# ------------------------------------------------------------------------- objects


def quat_axis(axis, angle: float) -> list[float]:
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return [np.cos(angle / 2), *(np.sin(angle / 2) * axis)]


def solid(body, **kw):
    """A collision geom that is also what you see."""
    return body.add_geom(condim=4, friction=[1.0, 0.01, 0.0001], **kw)


def visual(body, **kw):
    """Decoration only: no collisions, no mass."""
    return body.add_geom(contype=0, conaffinity=0, density=0, group=2, **kw)


SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX, MESH = (
    mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
    mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_MESH)


@dataclass
class Thing:
    """An object to pick. The body origin is the grasp center unless grasp_dz moves it up."""

    name: str
    build: Callable  # (spec, body, rng) -> None
    rest_z: float  # body origin height above whatever it rests on
    bbox: tuple[np.ndarray, np.ndarray]  # (center, half-size) in the body frame
    # Local axis to close the fingers along, "narrowest" to measure the mesh, or None for any.
    finger_axis: Union[int, str, None] = None
    grasp_dz: float = 0.0
    random_yaw: bool = True
    bid: int = -1  # filled in after compile
    qadr: int = -1
    gid: int = -1  # the mesh geom, for "narrowest"


def build_raspberry(spec, body, rng) -> None:
    a, c = 0.0105, 0.0125
    # Collide as a box: a pointy ellipsoid would topple and roll away, and a cylinder
    # only touches the flat finger pads along two lines, so it can roll out of the grip.
    solid(body, type=BOX, size=[a - 0.002, a - 0.002, c], mass=0.006, rgba=[0.55, 0.02, 0.12, 1])
    n = 48  # drupelets on a Fibonacci spiral, leaving the top for the calyx and a flat bottom
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        if z > 0.8 or z < -0.8:
            continue
        r, phi = np.sqrt(1 - z * z), i * np.pi * (3 - np.sqrt(5))
        shade = rng.uniform(0.8, 1.0)
        visual(body, type=SPHERE, size=[0.0042, 0, 0], pos=[a * r * np.cos(phi), a * r * np.sin(phi), c * z],
               rgba=[0.85 * shade, 0.05, 0.18 * shade, 1])
    for k in range(5):  # calyx and stem
        th = 2 * np.pi * k / 5
        visual(body, type=ELLIPSOID, size=[0.0065, 0.0022, 0.0009], quat=quat_axis([0, 0, 1], th),
               pos=[0.0045 * np.cos(th), 0.0045 * np.sin(th), c + 0.0005], rgba=[0.25, 0.55, 0.15, 1])
    visual(body, type=CYLINDER, size=[0.0012, 0.003, 0], pos=[0, 0, c + 0.003], rgba=[0.3, 0.5, 0.15, 1])


def build_bulb(spec, body, rng) -> None:
    glass = [1.0, 0.97, 0.82, 0.35]
    solid(body, type=SPHERE, size=[0.028, 0, 0], mass=0.010, rgba=glass)
    solid(body, type=CYLINDER, size=[0.0155, 0.009, 0], pos=[0, 0, -0.031], mass=0.002, rgba=glass)
    solid(body, type=CYLINDER, size=[0.0135, 0.012, 0], pos=[0, 0, -0.050], mass=0.012,
          rgba=[0.72, 0.72, 0.75, 1])
    for z in (-0.043, -0.050, -0.057):  # screw threads
        visual(body, type=CYLINDER, size=[0.0145, 0.0011, 0], pos=[0, 0, z], rgba=[0.8, 0.8, 0.82, 1])
    solid(body, type=CYLINDER, size=[0.0095, 0.0035, 0], pos=[0, 0, -0.0655], mass=0.001,
          rgba=[0.08, 0.08, 0.08, 1])
    wire = [0.5, 0.5, 0.5, 1]
    for x in (-0.004, 0.004):  # filament supports and the coil
        visual(body, type=CAPSULE, size=[0.0006, 0.011, 0], pos=[x, 0, -0.016], rgba=wire)
    visual(body, type=CAPSULE, size=[0.0012, 0.004, 0], pos=[0, 0, -0.004],
           quat=quat_axis([0, 1, 0], np.pi / 2), rgba=[1.0, 0.6, 0.1, 1])


def build_duck(spec, body, rng) -> None:
    yellow = [1.0, 0.85, 0.1, 1]
    solid(body, type=ELLIPSOID, size=[0.034, 0.026, 0.021], mass=0.010, rgba=yellow)
    solid(body, type=SPHERE, size=[0.0165, 0, 0], pos=[0.019, 0, 0.027], mass=0.003, rgba=yellow)
    visual(body, type=ELLIPSOID, size=[0.010, 0.0075, 0.0035], pos=[0.036, 0, 0.024], rgba=[1.0, 0.45, 0.0, 1])
    for y in (-0.0085, 0.0085):
        visual(body, type=SPHERE, size=[0.0028, 0, 0], pos=[0.030, y, 0.034], rgba=[0.05, 0.05, 0.05, 1])
        visual(body, type=ELLIPSOID, size=[0.015, 0.004, 0.009], pos=[-0.004, 2.8 * y, 0.005],
               rgba=[0.95, 0.78, 0.05, 1])
    visual(body, type=ELLIPSOID, size=[0.011, 0.009, 0.006], pos=[-0.031, 0, 0.012],
           quat=quat_axis([0, 1, 0], -0.6), rgba=yellow)


def rock_vertices(seed: int, radii) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(24, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v *= np.asarray(radii) * rng.uniform(0.85, 1.15, size=(24, 1))
    return v - v.mean(axis=0)


def make_rock(name: str, seed: int, radii, rgba) -> Thing:
    verts = rock_vertices(seed, radii)

    def build(spec, body, rng) -> None:
        spec.add_mesh(name=name, uservert=verts.flatten().tolist())
        solid(body, type=MESH, meshname=name, density=2600, rgba=rgba)

    lo, hi = verts.min(axis=0), verts.max(axis=0)
    return Thing(name, build, rest_z=-lo[2], bbox=((lo + hi) / 2, (hi - lo) / 2), finger_axis="narrowest")


def mesh_points(m, d, gid: int) -> np.ndarray:
    """World-space vertices of a mesh geom."""
    mid = m.geom_dataid[gid]
    adr, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    return d.geom_xpos[gid] + m.mesh_vert[adr:adr + n] @ d.geom_xmat[gid].reshape(3, 3).T


def make_things() -> list[Thing]:
    """In picking order. The tall light bulb goes last, so no later placement can knock it over."""
    return [
        Thing("raspberry", build_raspberry, rest_z=0.0125, finger_axis=0,  # close on two box faces
              bbox=(np.array([0, 0, 0.002]), np.array([0.015, 0.015, 0.017]))),
        # Grasp 4 mm above center so the head stays clear of the palm.
        Thing("rubber duck", build_duck, rest_z=0.021, finger_axis=1, grasp_dz=0.004,
              bbox=(np.array([0.002, 0, 0.011]), np.array([0.044, 0.027, 0.033]))),
        make_rock("granite rock", 3, [0.024, 0.017, 0.014], [0.46, 0.44, 0.42, 1]),
        make_rock("sandstone rock", 11, [0.021, 0.015, 0.013], [0.62, 0.48, 0.33, 1]),
        Thing("light bulb", build_bulb, rest_z=0.069, random_yaw=False,
              bbox=(np.array([0, 0, -0.0205]), np.array([0.029, 0.029, 0.0485]))),
    ]


# Where each object starts on the table (x, y), before random jitter, in picking order.
LAYOUT = [(0.42, 0.22), (0.46, 0.06), (0.62, 0.08), (0.40, -0.06), (0.58, 0.26)]

# --------------------------------------------------------------------------- scene


def add_table_and_tray(spec) -> None:
    wb = spec.worldbody
    wood, tray = [0.55, 0.38, 0.22, 1], [0.35, 0.45, 0.30, 1]
    wb.add_geom(type=BOX, size=[*TABLE_HALF, 0.015], pos=[*TABLE_CENTER, TABLE_TOP - 0.015], rgba=wood)
    for sx in (-1, 1):
        for sy in (-1, 1):
            leg = TABLE_CENTER + np.array([sx, sy]) * (TABLE_HALF - 0.03)
            wb.add_geom(type=BOX, size=[0.02, 0.02, (TABLE_TOP - 0.03) / 2], pos=[*leg, (TABLE_TOP - 0.03) / 2],
                        rgba=wood, contype=0, conaffinity=0)
    wb.add_geom(type=BOX, size=[*(TRAY_INNER + 0.006), 0.003], pos=[*TRAY_CENTER, TABLE_TOP + 0.003], rgba=tray)
    for axis in (0, 1):
        for s in (-1, 1):
            half = TRAY_INNER + 0.006
            size = [0.003, half[1], 0.015] if axis == 0 else [half[0], 0.003, 0.015]
            pos = TRAY_CENTER.copy()
            pos[axis] += s * (TRAY_INNER[axis] + 0.003)
            wb.add_geom(type=BOX, size=size, pos=[*pos, TABLE_TOP + 0.015], rgba=tray)


def build_scene(things: list[Thing], seed: int):
    cfg = dataclasses.replace(ROBOTS["panda"], extra_ctrl={})  # we drive the gripper ourselves
    spec = mujoco.MjSpec.from_file(str(menagerie_dir() / cfg.folder / cfg.scene))
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC  # MuJoCo's advice for stable grasps
    spec.option.impratio = 10
    add_table_and_tray(spec)
    rng = np.random.default_rng(seed)
    for thing, (x, y) in zip(things, LAYOUT):
        body = spec.worldbody.add_body(name=thing.name, pos=[x, y, TABLE_TOP + thing.rest_z])
        body.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
        thing.build(spec, body, rng)
    for key in spec.keys:  # Menagerie's keyframe doesn't know about the objects
        if len(key.qpos):
            key.qpos = list(key.qpos) + [v for x, y in LAYOUT for v in (x, y, TABLE_TOP, 1, 0, 0, 0)]
    model = spec.compile()
    for thing in things:
        thing.bid = model.body(thing.name).id
        thing.qadr = model.jnt_qposadr[model.body_jntadr[thing.bid]]
        meshes = [g for g in range(model.ngeom) if model.geom_bodyid[g] == thing.bid and model.geom_type[g] == MESH]
        thing.gid = meshes[0] if meshes else -1
    return cfg, model


def reset(model, data, cfg, ik: DiffIK, things: list[Thing], rng) -> None:
    """Robot at home with the gripper open, objects back on the table with fresh jitter."""
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = data.ctrl[:7] = cfg.home
    data.ctrl[7] = OPEN
    for thing, (x, y) in zip(things, LAYOUT):
        yaw = rng.uniform(-np.pi, np.pi) if thing.random_yaw else 0.0
        x, y = x + rng.uniform(-0.015, 0.015), y + rng.uniform(-0.015, 0.015)
        data.qpos[thing.qadr:thing.qadr + 7] = [x, y, TABLE_TOP + thing.rest_z + 0.002, *quat_axis([0, 0, 1], yaw)]
    ik.q_cmd = cfg.home.copy()
    while data.time < 0.6:  # let the objects settle and the fingers open
        mujoco.mj_step(model, data)


# ---------------------------------------------------------------------------- motion


def smooth(s: float) -> float:
    """Minimum-jerk profile on [0, 1]."""
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


def yaw_matrix(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


@dataclass
class Segment:
    phase: str
    p0: np.ndarray
    p1: np.ndarray
    yaw0: float
    yaw1: float
    duration: float
    grip0: float
    grip1: float
    t0: float = 0.0

    def at(self, t: float):
        s = smooth((t - self.t0) / self.duration)
        return (self.p0 + s * (self.p1 - self.p0), self.yaw0 + s * (self.yaw1 - self.yaw0),
                self.grip0 + s * (self.grip1 - self.grip0))


class PickPlace:
    """Plans and runs approach -> descend -> grasp -> lift -> carry -> lower -> release -> retreat."""

    def __init__(self, model, data, ik: DiffIK, things: list[Thing], home_R: np.ndarray) -> None:
        self.m, self.d, self.ik, self.things, self.home_R = model, data, ik, things, home_R
        self.finger_x = home_R[:, 1]  # world direction the fingers close along at yaw 0
        self.restart()

    def restart(self) -> None:
        self.index, self.segments, self.k = -1, [], 0
        self.yaw, self.grip0_rel = 0.0, None
        self.events: list[tuple[str, np.ndarray]] = []  # phase changes, for drawing

    @property
    def thing(self) -> Optional[Thing]:
        return self.things[self.index] if 0 <= self.index < len(self.things) else None

    @property
    def phase(self) -> str:
        return self.segments[self.k].phase if self.segments else "DONE"

    def grasp_target(self, thing: Thing) -> tuple[np.ndarray, float]:
        """Where to put the TCP, and the wrist yaw that closes the fingers along the right axis."""
        d = self.d
        grasp = d.xpos[thing.bid] + np.array([0, 0, thing.grasp_dz])
        if thing.finger_axis is None:
            return grasp, self.yaw  # round object: don't turn the wrist
        if thing.finger_axis == "narrowest":
            # A rock can settle on any face, so measure its hull as it lies now and
            # close across the narrowest horizontal width, halfway up.
            pts = mesh_points(self.m, d, thing.gid)
            xy = pts[:, :2] - pts[:, :2].mean(axis=0)
            axes = np.linalg.eigh(xy.T @ xy)[1].T  # rows: narrowest, widest
            mid = [(xy @ a).min() / 2 + (xy @ a).max() / 2 for a in axes]
            center = pts[:, :2].mean(axis=0) + mid[0] * axes[0] + mid[1] * axes[1]
            z = max((pts[:, 2].min() + pts[:, 2].max()) / 2, TABLE_TOP + 0.011)  # fingertips clear the table
            grasp, axis = np.array([*center, z]), axes[0]
        else:
            axis = d.xmat[thing.bid].reshape(3, 3)[:2, thing.finger_axis]
        yaw = np.arctan2(axis[1], axis[0]) - np.arctan2(self.finger_x[1], self.finger_x[0])
        return grasp, (yaw + np.pi / 2) % np.pi - np.pi / 2  # the gripper is symmetric: stay within +-90 deg

    def plan(self, tcp: np.ndarray) -> None:
        thing = self.thing
        grasp, yg = self.grasp_target(thing)
        spot = TRAY_CENTER + np.array(TRAY_SPOTS[self.index])
        place = np.array([*spot, TRAY_FLOOR + (grasp[2] - TABLE_TOP) + DROP])
        above_obj, above_place = np.array([*grasp[:2], SAFE_Z]), np.array([*spot, SAFE_Z])
        y0 = self.yaw
        legs = [("APPROACH", tcp, above_obj, y0, yg, 1.8, OPEN, OPEN),
                ("DESCEND", above_obj, grasp, yg, yg, 1.0, OPEN, OPEN),
                ("GRASP", grasp, grasp, yg, yg, 0.8, OPEN, CLOSED),
                ("LIFT", grasp, above_obj, yg, yg, 1.0, CLOSED, CLOSED),
                ("CARRY", above_obj, above_place, yg, 0.0, 1.8, CLOSED, CLOSED),
                ("LOWER", above_place, place, 0.0, 0.0, 1.0, CLOSED, CLOSED),
                ("RELEASE", place, place, 0.0, 0.0, 0.5, CLOSED, OPEN),
                ("RETREAT", place, above_place, 0.0, 0.0, 0.8, OPEN, OPEN)]
        t = self.d.time
        self.segments, self.k = [], 0
        for leg in legs:
            self.segments.append(Segment(*leg, t0=t))
            t += leg[5]
        self.grasp_pose = (grasp, yg)
        self.grip0_rel = None

    def command(self, tcp: np.ndarray):
        """Target (position, yaw, velocity, gripper command) for the current time."""
        t = self.d.time
        while not self.segments or t >= self.segments[self.k].t0 + self.segments[self.k].duration:
            if self.segments and self.k + 1 < len(self.segments):
                self.k += 1
            elif self.index + 1 < len(self.things):
                self.index += 1
                self.plan(tcp)
            else:
                seg = self.segments[-1]  # all done: hold the last pose
                return seg.p1, seg.yaw1, np.zeros(3), seg.grip1
            self.events.append((self.phase, tcp.copy()))
        seg = self.segments[self.k]
        p, yaw, grip = seg.at(t)
        v = (seg.at(t + 1e-3)[0] - p) / 1e-3
        self.yaw = yaw
        return p, yaw, v, grip

    def slip(self, tcp: np.ndarray, R: np.ndarray) -> Optional[float]:
        """How far the object has moved in the hand since the grasp closed (m)."""
        if self.phase not in ("LIFT", "CARRY", "LOWER"):
            return None
        rel = R.T @ (self.d.xpos[self.thing.bid] - tcp)
        if self.grip0_rel is None:
            self.grip0_rel = rel
        return float(np.linalg.norm(rel - self.grip0_rel))

    @property
    def done(self) -> bool:
        return self.index == len(self.things) - 1 and self.k == len(self.segments) - 1 and \
            self.d.time >= self.segments[-1].t0 + self.segments[-1].duration


# --------------------------------------------------------------------------- drawing


def draw_static(draw: DebugDraw) -> None:
    """Drawn once; stays queued until the next round's clear()."""
    draw.box([*TRAY_CENTER, TRAY_FLOOR + 0.02], [*(TRAY_INNER - 0.005), 0.02], color=[0.2, 1.0, 0.3, 0.12],
             key="tray")


def draw_debug(draw: DebugDraw, task: PickPlace, tcp, R, p_des, yaw_des, carried: collections.deque,
               label_frames: int = 90) -> None:
    m, d, thing = task.m, task.d, task.thing
    if thing is None:
        return
    phase = task.phase
    slip = task.slip(tcp, R)

    # Plan: the remaining waypoints, the grasp pose until it's reached, target and TCP frames.
    waypoints = [tcp] + [s.p1 for s in task.segments[task.k:]]
    draw.line(np.array(waypoints), color="cyan", width=2.0, key="plan")
    if phase in ("APPROACH", "DESCEND"):
        grasp, yg = task.grasp_pose
        draw.frame_at(grasp, yaw_matrix(yg) @ task.home_R, scale=0.05, width=0.002, label="grasp", key="grasp")
    else:
        draw.remove("grasp")
    draw.frame_at(p_des, yaw_matrix(yaw_des) @ task.home_R, scale=0.06, width=0.002, alpha=0.35, key="target")
    draw.frame_at(tcp, R, scale=0.05, width=0.002, key="tcp")

    # The object: bounding box and COM. The grip is drawn as the force on each finger:
    # arrows start at the contact point, so the force on the object would point into
    # it and be hidden, while the force on a thin finger pokes out the other side.
    held = phase in ("LIFT", "CARRY", "LOWER")
    color = [1.0, 0.85, 0.1, 0.18]
    if held:
        color = [1.0, 0.15, 0.1, 0.3] if slip is not None and slip > SLIP_WARN else [0.2, 1.0, 0.3, 0.2]
    Rb = d.xmat[thing.bid].reshape(3, 3)
    center, half = thing.bbox
    draw.box(d.xpos[thing.bid] + Rb @ center, half + 0.003, rotation=Rb, color=color, key="bbox")
    draw.body_com(thing.bid, size=0.005, label=f"{thing.name} {m.body_mass[thing.bid] * 1000:.0f} g", key="com")
    for finger in ("left_finger", "right_finger"):
        draw.contacts(body=finger, normals=False, force_scale=FORCE_SCALE, force_color="orange", width=0.003,
                      key=f"grip_{finger}")
    if held:
        carried.append(d.xipos[thing.bid].copy())
    if len(carried) >= 2:
        draw.line(np.array(carried), color="magenta", width=3.0, key="carried")

    # Phase changes leave a short-lived label where they happened.
    # GRASP and RELEASE happen in place, so their labels would sit under the next one.
    while task.events:
        name, where = task.events.pop(0)
        if name not in ("GRASP", "RELEASE"):
            draw.point(where, color="white", size=0.004, label=name.lower(), ttl_frames=label_frames)

    grip = max(0.0, -float(d.actuator_force[7]))  # tendon force; negative while squeezing
    gap = (d.qpos[7] + d.qpos[8]) * 1000
    draw.text(
        f"Franka Panda pick and place\n"
        f"object : {thing.name} [{task.index + 1}/{len(task.things)}]\n"
        f"phase  : {phase}\n"
        f"grip   : {grip:4.1f} N   gap {gap:4.1f} mm\n"
        f"slip   : " + ("  --" if slip is None else f"{slip * 1000:4.1f} mm"),
        position="top_left", key="hud")
    rows = []
    for i, th in enumerate(task.things):
        mark = "x" if i < task.index or (i == task.index and task.done) else (">" if i == task.index else " ")
        rows.append(f"[{mark}] {th.name:<15s}{m.body_mass[th.bid] * 1000:3.0f} g")
    draw.text("\n".join(rows), position="top_right", key="progress")
    draw.text("orange finger contact force (8 cm/N) | box: yellow target, green held, red slipping\n"
              "cyan waypoints | magenta carried path | RGB grasp / target / TCP frames",
              position="bottom_left", key="legend")


# ------------------------------------------------------------------------------ main


def setup(seed: int):
    things = make_things()
    cfg, model = build_scene(things, seed)
    data = mujoco.MjData(model)
    ik = DiffIK(model, cfg)
    rng = np.random.default_rng(seed)
    reset(model, data, cfg, ik, things, rng)
    task = PickPlace(model, data, ik, things, home_R=ik.tcp(data)[1].copy())
    return things, cfg, model, data, ik, rng, task


def advance(model, data, ik: DiffIK, task: PickPlace, n_steps: int, draw: DebugDraw, carried) -> tuple:
    """Run n_steps of control and physics, then queue this frame's drawing."""
    index = task.index
    for _ in range(n_steps):
        tcp, _ = ik.tcp(data)
        p_des, yaw_des, v_des, grip = task.command(tcp)
        ik(data, p_des, yaw_matrix(yaw_des) @ task.home_R, v_des)
        data.ctrl[7] = grip
        mujoco.mj_step(model, data)
    if task.index != index:  # new object: forget the last one's carried path
        carried.clear()
        draw.remove("carried")
    return p_des, yaw_des


def run_viewer(seed: int) -> None:
    import mujoco.viewer

    things, cfg, model, data, ik, rng, task = setup(seed)
    steps = max(1, int(round((1 / 60) / model.opt.timestep)))
    carried: collections.deque = collections.deque(maxlen=180)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.52, 0.0, 0.30]  # facing the robot, so the hand works in front of the arm
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.1, 180, -30
        draw = DebugDraw(model, data, viewer)
        draw_static(draw)
        done_at = None
        while viewer.is_running():
            t0 = time.time()
            p_des, yaw_des = advance(model, data, ik, task, steps, draw, carried)
            tcp, R = ik.tcp(data)
            draw_debug(draw, task, tcp, R, p_des, yaw_des, carried, label_frames=90)  # 1.5 s at 60 fps
            draw.sync()
            viewer.sync()

            if task.done:  # pause, then start a new round with a new layout
                done_at = data.time if done_at is None else done_at
                if data.time - done_at > 2.0:
                    with viewer.lock():
                        reset(model, data, cfg, ik, things, rng)
                    task.restart()
                    draw.clear()
                    draw_static(draw)
                    done_at = None
            time.sleep(max(0.0, steps * model.opt.timestep - (time.time() - t0)))


def record(out: pathlib.Path, seed: int, speed: float, fps: int = 15, width: int = 640, height: int = 440) -> None:
    """Render one full round offscreen to a GIF, sped up by `speed`."""
    from PIL import Image

    from record_demo import hud  # same Pillow HUD compositor as the other GIFs

    things, cfg, model, data, ik, rng, task = setup(seed)
    renderer = mujoco.Renderer(model, height, width)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.47, -0.01, 0.33]  # facing the robot, so the hand works in front of the arm
    cam.distance, cam.azimuth, cam.elevation = 0.78, 180, -36
    draw = DebugDraw(model, data, renderer)
    draw_static(draw)
    carried: collections.deque = collections.deque(maxlen=180)
    steps = max(1, int(round(speed / (fps * model.opt.timestep))))
    label_frames = max(1, round(1.5 * fps / speed))  # 1.5 s of sim time
    frames, done_at = [], None
    while done_at is None or data.time - done_at < 1.0:
        p_des, yaw_des = advance(model, data, ik, task, steps, draw, carried)
        tcp, R = ik.tcp(data)
        draw_debug(draw, task, tcp, R, p_des, yaw_des, carried, label_frames=label_frames)
        renderer.update_scene(data, camera=cam)
        draw.sync()
        frames.append(hud(renderer.render(), draw.texts, width, height, font_size=12))
        if task.done and done_at is None:
            done_at = data.time
    renderer.close()

    pal = [Image.fromarray(f).quantize(colors=256, method=Image.Quantize.MAXCOVERAGE, dither=Image.Dither.NONE)
           for f in frames]
    out.parent.mkdir(parents=True, exist_ok=True)
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=int(1000 / fps), loop=0, optimize=True)
    print(f"wrote {out} ({len(frames)} frames, {speed:g}x speed)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0, help="seed for the object layout jitter")
    ap.add_argument("--record", type=pathlib.Path, help="render one round to a GIF offscreen instead of opening the viewer")
    ap.add_argument("--speed", type=float, default=3.0, help="playback speed-up for --record (default 3)")
    args = ap.parse_args()
    if args.record:
        record(args.record, args.seed, args.speed)
    else:
        run_viewer(args.seed)


if __name__ == "__main__":
    main()
