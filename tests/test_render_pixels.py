"""Pixel-level checks with an offscreen mujoco.Renderer.

Skipped automatically when no OpenGL backend is available. In CI set
MUJOCO_GL=egl (or osmesa) to run them headless.
"""

import mujoco
import numpy as np
import pytest

from mujoco_debugdraw import DebugDraw

XML = """<mujoco><visual><global fovy="20" offwidth="600" offheight="300"/></visual>
<worldbody><light pos="0 0 3" directional="true"/></worldbody></mujoco>"""


@pytest.fixture(scope="module")
def rig():
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    try:
        r = mujoco.Renderer(m, 300, 600)
    except Exception as exc:  # no GL context available
        pytest.skip(f"offscreen rendering unavailable: {exc}")
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.5, 0, 0]
    cam.distance, cam.azimuth, cam.elevation = 4.0, 90.0, 0.0  # looking along +y: world x -> screen right
    yield m, d, r, cam
    r.close()


def _render(rig, fn):
    m, d, r, cam = rig
    draw = DebugDraw(m, d, r)
    fn(draw)
    r.update_scene(d, camera=cam)
    draw.sync()
    return r.render()


def _yellow_cols(img):
    mask = (img[..., 0] > 120) & (img[..., 1] > 120) & (img[..., 2] < 90)
    return np.where(mask.any(axis=0))[0]


def _magenta_cols(img):
    mask = (img[..., 0] > 120) & (img[..., 1] < 90) & (img[..., 2] > 120)
    return np.where(mask.any(axis=0))[0]


@pytest.mark.parametrize("length", [0.4, 1.0])
def test_arrow_tip_lands_on_endpoint(rig, length):
    """Guards the mjGEOM_ARROW half-length compensation in debug_draw.py."""
    arrow = _render(rig, lambda d: d.arrow([0, 0, 0], [length, 0, 0], color="yellow", width=0.02))
    marks = _render(rig, lambda d: (d.point([0, 0, 0], color="magenta", size=0.01),
                                    d.point([length, 0, 0], color="magenta", size=0.01)))
    a, p = _yellow_cols(arrow), _magenta_cols(marks)
    assert a.size and p.size
    px_per_m = (p.max() - p.min()) / length
    tol = 0.03 * length * px_per_m + 3
    assert abs(a.min() - p.min()) <= tol, "arrow tail should start at origin"
    assert abs(a.max() - p.max()) <= tol, "arrow tip should end at origin + scale*vector"


def test_point_renders_and_clear_removes_it(rig):
    img = _render(rig, lambda d: d.point([0.5, 0, 0], color="magenta", size=0.05))
    assert _magenta_cols(img).size > 0
    img = _render(rig, lambda d: (d.point([0.5, 0, 0], color="magenta", size=0.05), d.clear()))
    assert _magenta_cols(img).size == 0
