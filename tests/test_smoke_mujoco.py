"""End-to-end checks against real MuJoCo scene structs (no OpenGL needed)."""

import mujoco
import numpy as np

from mujoco_debugdraw import DebugDraw

from conftest import XML, FakeViewer, assert_close


def draw_everything(draw, data):
    draw.clear()
    draw.point([0.4, 0.0, 0.3], color="lime", size=0.025, label="target", key="goal")
    draw.line([[0.2, -0.2, 0.05], [0.3, -0.1, 0.1], [0.4, 0.0, 0.3]], color="cyan")
    draw.arrow(data.site_xpos[0], [0.0, 0.0, 0.1], color="yellow", label="v")
    draw.frame("end_effector", scale=0.12, label="EE")
    draw.text(f"t = {data.time:.2f}")


def test_scene_geom_contents(model, data, scene):
    draw = DebugDraw(model, data, scene)
    draw_everything(draw, data)
    draw.sync()
    assert scene.ngeom == 1 + 2 + 1 + 3
    types = [scene.geoms[i].type for i in range(scene.ngeom)]
    assert types[0] == mujoco.mjtGeom.mjGEOM_SPHERE
    assert types[1:3] == [mujoco.mjtGeom.mjGEOM_LINE] * 2
    assert all(t == mujoco.mjtGeom.mjGEOM_ARROW for t in types[3:])
    sphere = scene.geoms[0]
    assert_close(sphere.pos, [0.4, 0.0, 0.3], tol=1e-6)
    assert sphere.size[0] == np.float32(0.025)
    assert sphere.label == "target"
    assert_close(sphere.rgba, (0.2, 1.0, 0.2, 1.0), tol=1e-6)


def test_line_geom_endpoints(model, data, scene):
    draw = DebugDraw(model, data, scene)
    draw.line([[0, 0, 0], [1, 0, 0]], width=4)
    draw.sync()
    g = scene.geoms[0]
    assert g.type == mujoco.mjtGeom.mjGEOM_LINE
    assert_close(g.pos, [0, 0, 0], tol=1e-6)
    assert_close(g.mat.reshape(3, 3)[:, 2] * g.size[2], [1, 0, 0], tol=1e-6)  # z-axis spans the segment


def test_no_accumulation_over_many_frames(model, data, scene):
    draw = DebugDraw(model, data, scene)
    counts = set()
    for _ in range(500):
        mujoco.mj_step(model, data)
        scene.ngeom = 0  # what mujoco.Renderer.update_scene does each frame
        draw_everything(draw, data)
        draw.sync()
        counts.add((scene.ngeom, draw.num_primitives))
    assert counts == {(7, 5)}


def test_viewer_mode_resets_each_sync(model, data):
    v = FakeViewer(model)
    draw = DebugDraw(model, data, v)
    for _ in range(200):
        draw_everything(draw, data)
        draw.sync()
    assert v.user_scn.ngeom == 7
    assert v.lock_calls == 200


def test_viewer_mode_preserves_users_own_user_scn_geoms(model, data):
    v = FakeViewer(model)
    g = v.user_scn.geoms[0]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, np.ones(3) * 0.1, np.zeros(3), np.eye(3).flatten(), np.ones(4, np.float32))
    v.user_scn.ngeom = 1
    draw = DebugDraw(model, data, v)
    for _ in range(10):
        draw.clear()
        draw.point([0, 0, 1])
        draw.sync()
    assert v.user_scn.ngeom == 2
    assert v.user_scn.geoms[0].type == mujoco.mjtGeom.mjGEOM_BOX


def test_viewer_texts_grouped_by_corner(model, data):
    v = FakeViewer(model)
    draw = DebugDraw(model, data, v)
    draw.text("line A")
    draw.text("line B")
    draw.text("status", position="bottom_right")
    draw.sync()
    by_pos = {pos: t1 for (_font, pos, t1, _t2) in v.texts}
    assert by_pos[mujoco.mjtGridPos.mjGRID_TOPLEFT] == "line A\nline B"
    assert by_pos[mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT] == "status"
    draw.clear()
    draw.sync()
    assert v.cleared == 1 and v.texts is None


def test_text_fallback_without_set_texts(model, data, capsys):
    class OldViewer(FakeViewer):
        set_texts = None  # simulates MuJoCo versions without the overlay API

    v = OldViewer(model)
    del v.texts
    draw = DebugDraw(model, data, v)
    draw.text("hello fallback")
    draw.sync()
    draw.sync()  # rate-limited: printed only once
    err = capsys.readouterr().err
    assert err.count("hello fallback") == 1


def _rollout(steps, draw_each_step):
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    d.qvel[0] = 1.0
    scn = mujoco.MjvScene(m, maxgeom=1000)
    draw = DebugDraw(m, d, scn) if draw_each_step else None
    for _ in range(steps):
        mujoco.mj_step(m, d)
        if draw:
            scn.ngeom = 0
            draw_everything(draw, d)
            draw.sync()
    return d


def test_dynamics_identical_with_and_without_drawing():
    control = _rollout(300, draw_each_step=False)
    drawn = _rollout(300, draw_each_step=True)
    assert np.all(np.isfinite(drawn.qpos))
    np.testing.assert_array_equal(control.qpos, drawn.qpos)
    np.testing.assert_array_equal(control.qvel, drawn.qvel)
    assert control.ncon == drawn.ncon


def test_model_is_not_mutated(model, data, scene):
    ngeom, nsite, nbody = model.ngeom, model.nsite, model.nbody
    draw = DebugDraw(model, data, scene)
    for _ in range(10):
        draw_everything(draw, data)
        draw.sync()
    assert (model.ngeom, model.nsite, model.nbody) == (ngeom, nsite, nbody)
