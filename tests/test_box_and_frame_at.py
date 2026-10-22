import mujoco
import numpy as np
import pytest

from mujoco_debugdraw import DebugDraw
from mujoco_debugdraw.primitives import AXIS_COLORS
from mujoco_debugdraw.transforms import as_rotation

from conftest import assert_close


def rot_xyz(a, b, c):
    cx, sx, cy, sy, cz, sz = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


R = rot_xyz(0.3, -0.7, 1.1)


# ---------------------------------------------------------------- as_rotation


def test_as_rotation_accepts_matrix_and_flat_row_major():
    assert_close(as_rotation(R), R)
    assert_close(as_rotation(R.flatten()), R)


@pytest.mark.parametrize(
    "bad",
    [
        np.eye(2),
        np.zeros(8),
        2.0 * np.eye(3),  # not orthonormal
        np.diag([1.0, 1.0, -1.0]),  # reflection
        np.full((3, 3), np.nan),
    ],
)
def test_as_rotation_rejects(bad):
    with pytest.raises(ValueError, match="rotation"):
        as_rotation(bad)


def test_as_rotation_copies_input():
    src = R.copy()
    out = as_rotation(src)
    src[:] = 0
    assert_close(out, R)


# ------------------------------------------------------------------------ box


def test_box_spec_holds_center_half_size_and_rotation(model, data, scene):
    draw = DebugDraw(model, data, scene)
    draw.box([0.1, 0.2, 0.3], [0.05, 0.1, 0.2], rotation=R, color="#00ff0080", label="zone")
    (spec,) = draw.geom_specs()
    assert spec.kind == "box"
    assert_close(spec.start, [0.1, 0.2, 0.3])
    assert_close(spec.size, [0.05, 0.1, 0.2])
    assert_close(spec.mat, R.flatten())
    assert spec.label == "zone"
    assert spec.rgba[3] == pytest.approx(128 / 255)


def test_box_defaults_to_world_aligned(model, data, scene):
    draw = DebugDraw(model, data, scene)
    draw.box([0, 0, 0], [1, 1, 1])
    assert_close(draw.geom_specs()[0].mat, np.eye(3).flatten())


def test_box_flat_rotation_from_xmat(model, data, scene):
    draw = DebugDraw(model, data, scene)
    draw.box([0, 0, 0], [0.1, 0.1, 0.1], rotation=data.xmat[1])
    assert_close(draw.geom_specs()[0].mat, data.xmat[1])


def test_box_scene_geom(model, data, scene):
    draw = DebugDraw(model, data, scene)
    draw.box([0.1, 0.2, 0.3], [0.05, 0.1, 0.2], rotation=R)
    draw.sync()
    assert scene.ngeom == 1
    g = scene.geoms[0]
    assert g.type == mujoco.mjtGeom.mjGEOM_BOX
    assert_close(g.pos, [0.1, 0.2, 0.3], tol=1e-6)
    assert_close(g.size, [0.05, 0.1, 0.2], tol=1e-6)
    assert_close(g.mat.flatten(), R.flatten(), tol=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"half_size": [0.1, 0.0, 0.1]},
        {"half_size": [0.1, -0.1, 0.1]},
        {"half_size": [0.1, 0.1]},
        {"rotation": np.eye(2)},
        {"rotation": 2.0 * np.eye(3)},
        {"rotation": np.diag([1.0, 1.0, -1.0])},
        {"center": [0.0, np.inf, 0.0]},
    ],
)
def test_box_rejects(model, data, scene, kwargs):
    args = {"center": [0, 0, 0], "half_size": [0.1, 0.1, 0.1], **kwargs}
    with pytest.raises(ValueError):
        DebugDraw(model, data, scene).box(**args)


# ------------------------------------------------------------------- frame_at


def test_frame_at_axis_tips(model, data, scene):
    draw = DebugDraw(model, data, scene)
    pos = np.array([0.3, -0.2, 0.5])
    draw.frame_at(pos, R, scale=0.4)
    specs = draw.geom_specs()
    assert len(specs) == 3
    for i, spec in enumerate(specs):
        assert_close(spec.start, pos)
        assert_close(spec.end, pos + 0.4 * R[:, i])
        assert spec.rgba == AXIS_COLORS[i]


@pytest.mark.parametrize("body", ["link", "end_effector", "box"])
def test_frame_at_matches_frame_for_a_body(model, data, scene, body):
    bid = model.body(body).id
    a, b = DebugDraw(model, data, scene), DebugDraw(model, data, scene)
    a.frame(bid, scale=0.2, label="L")
    b.frame_at(data.xpos[bid], data.xmat[bid], scale=0.2, label="L")
    assert len(a.geom_specs()) == len(b.geom_specs()) == 3
    for sa, sb in zip(a.geom_specs(), b.geom_specs()):
        assert sa.kind == sb.kind and sa.rgba == sb.rgba and sa.label == sb.label
        assert sa.size == sb.size
        assert_close(sa.start, sb.start, tol=0)
        assert_close(sa.end, sb.end, tol=0)


def test_frame_at_alpha_fades_all_axes(model, data, scene):
    draw = DebugDraw(model, data, scene)
    draw.frame_at([0, 0, 0], np.eye(3), alpha=0.4)
    for i, spec in enumerate(draw.geom_specs()):
        assert spec.rgba[:3] == AXIS_COLORS[i][:3]
        assert spec.rgba[3] == pytest.approx(0.4)


def test_frame_at_is_a_snapshot(model, data, scene):
    draw = DebugDraw(model, data, scene)
    draw.frame_at(data.xpos[2], data.xmat[2])
    queued = draw.geom_specs()[0].end.copy()
    data.qpos[0] = -1.0
    mujoco.mj_forward(model, data)
    assert_close(draw.geom_specs()[0].end, queued)


@pytest.mark.parametrize(
    "kwargs",
    [{"alpha": 0.0}, {"alpha": 1.5}, {"alpha": float("nan")}, {"scale": 0}, {"width": -1},
     {"rotation": np.diag([1.0, -1.0, 1.0])}, {"position": [0, 0]}],
)
def test_frame_at_rejects(model, data, scene, kwargs):
    args = {"position": [0, 0, 0], "rotation": np.eye(3), **kwargs}
    with pytest.raises(ValueError):
        DebugDraw(model, data, scene).frame_at(**args)
