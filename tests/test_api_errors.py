import warnings

import mujoco
import pytest

from mujoco_debugdraw import CapacityWarning, DebugDraw, IncompatibleViewerError


def test_rejects_non_model(data, scene):
    with pytest.raises(TypeError, match="MjModel"):
        DebugDraw("model.xml", data, scene)


def test_rejects_non_data(model, scene):
    with pytest.raises(TypeError, match="MjData"):
        DebugDraw(model, None, scene)


def test_rejects_mismatched_data(model, scene):
    other = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    with pytest.raises(ValueError, match="does not belong"):
        DebugDraw(model, mujoco.MjData(other), scene)


def test_rejects_unsupported_viewer(model, data):
    with pytest.raises(IncompatibleViewerError, match="launch_passive"):
        DebugDraw(model, data, object())


@pytest.mark.parametrize("bad", [0, -5, 2.5, "10"])
def test_rejects_bad_max_geoms(model, data, scene, bad):
    with pytest.raises(ValueError):
        DebugDraw(model, data, scene, max_geoms=bad)


@pytest.mark.parametrize(
    "call",
    [
        lambda d: d.point([0, 0, 0], size=0),
        lambda d: d.point([0, 0, 0], size=-1),
        lambda d: d.line([[0, 0, 0], [1, 1, 1]], width=0),
        lambda d: d.arrow([0, 0, 0], [1, 0, 0], width=-0.1),
        lambda d: d.arrow([0, 0, 0], [1, 0, 0], scale=float("nan")),
        lambda d: d.frame("end_effector", scale=0),
        lambda d: d.text("hi", position="center"),
        lambda d: d.point([0, 0]),
        lambda d: d.arrow([0, 0, 0], [1, 0]),
    ],
)
def test_bad_arguments_raise_value_error(model, data, scene, call):
    with pytest.raises(ValueError):
        call(DebugDraw(model, data, scene))


def test_max_geoms_warns_and_drops(model, data, scene):
    draw = DebugDraw(model, data, scene, max_geoms=5)
    with pytest.warns(CapacityWarning, match="max_geoms=5"):
        for i in range(8):
            draw.point([i, 0, 0])
    assert draw.num_geoms == 5
    assert draw.dropped == 3
    draw.sync()
    assert scene.ngeom == 5


def test_frame_that_does_not_fit_is_dropped_whole(model, data, scene):
    draw = DebugDraw(model, data, scene, max_geoms=4)
    draw.point([0, 0, 0])
    draw.point([0, 0, 0])
    with pytest.warns(CapacityWarning):
        draw.frame("end_effector")  # needs 3, only 2 free
    assert draw.num_geoms == 2


def test_clear_resets_capacity(model, data, scene):
    draw = DebugDraw(model, data, scene, max_geoms=3)
    for _ in range(3):
        draw.point([0, 0, 0])
    draw.clear()
    assert draw.num_geoms == 0 and draw.dropped == 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(3):
            draw.point([0, 0, 0])


def test_keyed_replacement_does_not_consume_capacity(model, data, scene):
    draw = DebugDraw(model, data, scene, max_geoms=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for i in range(100):
            draw.point([i, 0, 0], key="goal")
    assert draw.num_primitives == 1 and draw.num_geoms == 1


def test_scene_capacity_is_respected(model, data):
    tiny = mujoco.MjvScene(model, maxgeom=2)
    draw = DebugDraw(model, data, tiny)
    for i in range(4):
        draw.point([i, 0, 0])
    with pytest.warns(CapacityWarning, match="room for 2"):
        draw.sync()
    assert tiny.ngeom == 2


def test_clear_removes_keyed_and_unkeyed(model, data, scene):
    draw = DebugDraw(model, data, scene)
    draw.point([0, 0, 0], key="a")
    draw.point([0, 0, 0])
    draw.text("x", key="t")
    draw.clear()
    assert draw.num_primitives == 0 and draw.texts == {}


def test_label_is_truncated_to_mujoco_limit(model, data, scene):
    draw = DebugDraw(model, data, scene)
    draw.point([0, 0, 0], label="x" * 500)
    draw.sync()
    assert len(scene.geoms[0].label) <= 99
