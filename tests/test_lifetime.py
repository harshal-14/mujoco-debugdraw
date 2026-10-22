import warnings

import mujoco
import pytest

from mujoco_debugdraw import CapacityWarning, DebugDraw

from conftest import FakeViewer, assert_close


@pytest.fixture
def viewer(model):
    return FakeViewer(model)


# ------------------------------------------------------------------ ttl_frames


@pytest.mark.parametrize("n", [1, 3])
def test_ttl_appears_in_exactly_n_syncs(model, data, viewer, n):
    draw = DebugDraw(model, data, viewer)
    draw.point([0, 0, 0], ttl_frames=n)
    drawn = []
    for _ in range(n + 2):
        draw.sync()
        drawn.append(viewer.user_scn.ngeom)
    assert drawn == [1] * n + [0, 0]
    assert draw.num_primitives == 0 and draw.num_geoms == 0


def test_ttl_frees_geom_count_on_expiry(model, data, viewer):
    draw = DebugDraw(model, data, viewer)
    draw.frame("end_effector", ttl_frames=2)
    draw.point([0, 0, 0])
    assert draw.num_geoms == 4
    draw.sync()
    assert draw.num_geoms == 4
    draw.sync()
    assert draw.num_geoms == 1


def test_ttl_redraw_with_same_key_restarts(model, data, viewer):
    draw = DebugDraw(model, data, viewer)
    draw.point([0, 0, 0], key="p", ttl_frames=2)
    draw.sync()
    draw.point([1, 0, 0], key="p", ttl_frames=2)  # restart: 2 more syncs
    draw.sync()
    assert viewer.user_scn.ngeom == 1
    draw.sync()
    assert viewer.user_scn.ngeom == 1
    assert_close(viewer.user_scn.geoms[0].pos, [1, 0, 0], tol=1e-6)
    draw.sync()
    assert viewer.user_scn.ngeom == 0


def test_clear_removes_ttl_primitive_early(model, data, viewer):
    draw = DebugDraw(model, data, viewer)
    draw.point([0, 0, 0], ttl_frames=10)
    draw.sync()
    draw.clear()
    draw.sync()
    assert viewer.user_scn.ngeom == 0 and draw.num_primitives == 0


def test_ttl_none_persists_without_clear(model, data, viewer):
    draw = DebugDraw(model, data, viewer)
    draw.line([[0, 0, 0], [1, 0, 0]], key="plan")
    for _ in range(50):
        draw.sync()
    assert viewer.user_scn.ngeom == 1


def test_ttl_on_text(model, data, viewer):
    draw = DebugDraw(model, data, viewer)
    draw.text("hello", ttl_frames=1)
    draw.sync()
    assert viewer.texts is not None
    assert draw.texts == {}
    draw.sync()
    assert viewer.texts is None and viewer.cleared == 1


def test_ttl_works_in_scene_mode(model, data, scene):
    draw = DebugDraw(model, data, scene)
    draw.point([0, 0, 0], ttl_frames=1)
    draw.sync()
    assert scene.ngeom == 1
    scene.ngeom = 0  # what mujoco.Renderer.update_scene does each frame
    draw.sync()
    assert scene.ngeom == 0


def test_no_clear_loop_does_not_accumulate(model, data, viewer):
    draw = DebugDraw(model, data, viewer)
    draw.line([[0, 0, 0], [1, 0, 0], [1, 1, 0]], key="plan")  # drawn once, stays
    counts = set()
    with warnings.catch_warnings():
        warnings.simplefilter("error", CapacityWarning)
        for _ in range(300):
            mujoco.mj_step(model, data)
            draw.frame("end_effector", key="ee")  # keyed: replaced each frame
            draw.contacts(ttl_frames=1)  # unkeyed: expires after one sync
            draw.point(data.xpos[1], ttl_frames=1)
            draw.sync()
            counts.add(draw.num_primitives)
    assert counts == {2}  # plan + ee survive each sync; the ttl=1 ones are gone


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "3"])
def test_ttl_rejects_invalid(model, data, scene, bad):
    draw = DebugDraw(model, data, scene)
    with pytest.raises(ValueError, match="ttl_frames"):
        draw.point([0, 0, 0], ttl_frames=bad)


@pytest.mark.parametrize(
    "call",
    [
        lambda d: d.point([0, 0, 0], ttl_frames=0),
        lambda d: d.line([[0, 0, 0], [1, 0, 0]], ttl_frames=0),
        lambda d: d.arrow([0, 0, 0], [1, 0, 0], ttl_frames=0),
        lambda d: d.frame("end_effector", ttl_frames=0),
        lambda d: d.frame_at([0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 1], ttl_frames=0),
        lambda d: d.box([0, 0, 0], [1, 1, 1], ttl_frames=0),
        lambda d: d.body_com("link", ttl_frames=0),
        lambda d: d.joint_axis("hinge", ttl_frames=0),
        lambda d: d.contacts(ttl_frames=0),
        lambda d: d.text("x", ttl_frames=0),
    ],
)
def test_every_method_validates_ttl(model, data, scene, call):
    with pytest.raises(ValueError, match="ttl_frames"):
        call(DebugDraw(model, data, scene))


# --------------------------------------------------------------------- remove


def test_remove_returns_true_then_false(model, data, scene):
    draw = DebugDraw(model, data, scene)
    draw.frame("end_effector", key="ee")
    assert draw.remove("ee") is True
    assert draw.remove("ee") is False
    assert draw.num_primitives == 0 and draw.num_geoms == 0


def test_remove_frees_capacity(model, data, scene):
    draw = DebugDraw(model, data, scene, max_geoms=1)
    draw.point([0, 0, 0], key="a")
    assert draw.remove("a")
    with warnings.catch_warnings():
        warnings.simplefilter("error", CapacityWarning)
        draw.point([0, 0, 0], key="b")
    assert draw.num_geoms == 1


def test_remove_only_touches_its_key(model, data, viewer):
    draw = DebugDraw(model, data, viewer)
    draw.point([0, 0, 0], key="a")
    draw.point([1, 0, 0], key="b")
    draw.remove("a")
    draw.sync()
    assert viewer.user_scn.ngeom == 1
    assert_close(viewer.user_scn.geoms[0].pos, [1, 0, 0], tol=1e-6)


@pytest.mark.parametrize("key", ["missing", None, 3, ["unhashable"]])
def test_remove_never_raises(model, data, scene, key):
    assert DebugDraw(model, data, scene).remove(key) is False
