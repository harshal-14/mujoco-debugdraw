import mujoco
import numpy as np
import pytest

from mujoco_debugdraw import DebugDraw, UnknownTargetError
from mujoco_debugdraw.primitives import AXIS_COLORS
from mujoco_debugdraw.transforms import frame_axes

from conftest import assert_close


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def test_frame_axes_identity():
    axes = frame_axes(np.zeros(3), np.eye(3).flatten(), 2.0)
    for i, (a, b) in enumerate(axes):
        assert_close(a, 0)
        assert_close(b, 2.0 * np.eye(3)[i])


def test_frame_axes_are_columns_of_rotation():
    R = rot_z(np.pi / 2)  # local X -> world +Y, local Y -> world -X
    axes = frame_axes(np.array([1.0, 0, 0]), R.flatten(), 1.0)
    assert_close(axes[0][1], [1, 1, 0])
    assert_close(axes[1][1], [0, 0, 0])
    assert_close(axes[2][1], [1, 0, 1])


def _axis_tips(draw):
    return [s.end for s in draw.geom_specs()]


@pytest.mark.parametrize("target,kind", [("end_effector", None), ("tip_site", None), ("tip_site", "site")])
def test_frame_matches_mujoco_pose(model, data, scene, target, kind):
    draw = DebugDraw(model, data, scene)
    draw.frame(target, scale=0.5, kind=kind)
    if target == "tip_site":
        pos, R = data.site(target).xpos, data.site(target).xmat.reshape(3, 3)
    else:
        pos, R = data.body(target).xpos, data.body(target).xmat.reshape(3, 3)
    specs = draw.geom_specs()
    assert len(specs) == 3
    for i, spec in enumerate(specs):
        assert_close(spec.start, pos)
        assert_close(spec.end, pos + 0.5 * R[:, i])
        assert spec.rgba == AXIS_COLORS[i]


def test_frame_axes_are_orthonormal_and_right_handed(model, data, scene):
    draw = DebugDraw(model, data, scene)
    draw.frame("tip_site", scale=1.0)
    x, y, z = [s.end - s.start for s in draw.geom_specs()]
    assert_close([np.dot(x, y), np.dot(y, z), np.dot(x, z)], 0, tol=1e-9)
    assert_close(np.cross(x, y), z, tol=1e-9)


def test_frame_by_body_id(model, data, scene):
    draw = DebugDraw(model, data, scene)
    bid = model.body("end_effector").id
    draw.frame(bid, scale=1.0)
    assert_close(draw.geom_specs()[0].start, data.xpos[bid])


def test_frame_by_site_id(model, data, scene):
    draw = DebugDraw(model, data, scene)
    sid = model.site("tip_site").id
    draw.frame(sid, kind="site")
    assert_close(draw.geom_specs()[0].start, data.site_xpos[sid])


def test_frame_follows_motion(model, data, scene):
    draw = DebugDraw(model, data, scene)
    draw.frame("end_effector", key="ee")
    before = draw.geom_specs()[0].start.copy()
    data.qpos[0] = -0.8
    mujoco.mj_forward(model, data)
    draw.frame("end_effector", key="ee")  # same key replaces
    after = draw.geom_specs()[0].start
    assert draw.num_primitives == 1
    assert np.linalg.norm(after - before) > 0.1
    assert_close(after, data.body("end_effector").xpos)


def test_frame_snapshot_not_aliased_to_data(model, data, scene):
    draw = DebugDraw(model, data, scene)
    draw.frame("end_effector")
    queued = draw.geom_specs()[0].start.copy()
    data.qpos[0] = 1.2
    mujoco.mj_forward(model, data)
    assert_close(draw.geom_specs()[0].start, queued)


def test_unknown_name_suggests_close_match(model, data, scene):
    draw = DebugDraw(model, data, scene)
    with pytest.raises(UnknownTargetError, match="end_effector"):
        draw.frame("end_efector")


def test_unknown_name_is_key_error(model, data, scene):
    draw = DebugDraw(model, data, scene)
    with pytest.raises(KeyError):
        draw.frame("nothing_like_it")


def test_kind_restricts_lookup(model, data, scene):
    draw = DebugDraw(model, data, scene)
    with pytest.raises(UnknownTargetError, match="No site named"):
        draw.frame("end_effector", kind="site")


@pytest.mark.parametrize("bad", [999, -1])
def test_body_id_out_of_range(model, data, scene, bad):
    with pytest.raises(UnknownTargetError, match="out of range"):
        DebugDraw(model, data, scene).frame(bad)


def test_bad_kind(model, data, scene):
    with pytest.raises(ValueError):
        DebugDraw(model, data, scene).frame("end_effector", kind="joint")


def test_bool_target_rejected(model, data, scene):
    with pytest.raises(TypeError):
        DebugDraw(model, data, scene).frame(True)
