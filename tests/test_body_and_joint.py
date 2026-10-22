import mujoco
import numpy as np
import pytest

from mujoco_debugdraw import DebugDraw, UnknownTargetError

from conftest import assert_close

# One joint of each type, posed away from zero so anchors and axes are non-trivial.
JOINTS_XML = """
<mujoco>
  <worldbody>
    <body name="arm" pos="0 0 1" euler="0.2 0.3 0">
      <joint name="hinge_j" type="hinge" axis="0 1 0" pos="0.1 0 0"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
      <body name="slider" pos="0.2 0 0">
        <joint name="slide_j" type="slide" axis="1 0 1"/>
        <geom type="sphere" size="0.03"/>
      </body>
    </body>
    <body name="ball_body" pos="1 0 1">
      <joint name="ball_j" type="ball"/>
      <geom type="sphere" size="0.05"/>
    </body>
    <body name="free_body" pos="-1 0 1">
      <freejoint name="free_j"/>
      <geom type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def jmodel():
    return mujoco.MjModel.from_xml_string(JOINTS_XML)


@pytest.fixture
def jdata(jmodel):
    d = mujoco.MjData(jmodel)
    d.qpos[0] = 0.5  # hinge
    d.qpos[1] = 0.1  # slide
    mujoco.mj_forward(jmodel, d)
    return d


@pytest.fixture
def jdraw(jmodel, jdata):
    return DebugDraw(jmodel, jdata, mujoco.MjvScene(jmodel, maxgeom=100))


# ------------------------------------------------------------------- body_com


@pytest.mark.parametrize("body", ["link", "end_effector", "box"])
def test_body_com_matches_xipos(model, data, scene, body):
    draw = DebugDraw(model, data, scene)
    draw.body_com(body, size=0.03, color="magenta", label="com")
    (spec,) = draw.geom_specs()
    assert spec.kind == "sphere" and spec.size == 0.03 and spec.label == "com"
    assert_close(spec.start, data.xipos[model.body(body).id])


@pytest.mark.parametrize("body", ["link", "end_effector"])
def test_body_com_subtree_matches_subtree_com(model, data, scene, body):
    draw = DebugDraw(model, data, scene)
    draw.body_com(body, subtree=True)
    assert_close(draw.geom_specs()[0].start, data.subtree_com[model.body(body).id])


def test_world_subtree_com_is_whole_model_com(model, data, scene):
    draw = DebugDraw(model, data, scene)
    draw.body_com(0, subtree=True)
    mass = model.body_mass
    expected = (mass[:, None] * data.xipos).sum(axis=0) / mass.sum()
    assert_close(draw.geom_specs()[0].start, expected, tol=1e-9)


def test_body_com_by_name_world(model, data, scene):
    draw = DebugDraw(model, data, scene)
    draw.body_com("world", subtree=True)
    assert_close(draw.geom_specs()[0].start, data.subtree_com[0])


def test_body_com_unknown_name(model, data, scene):
    with pytest.raises(UnknownTargetError, match="end_effector"):
        DebugDraw(model, data, scene).body_com("end_efector")


def test_body_com_does_not_accept_sites(model, data, scene):
    with pytest.raises(UnknownTargetError, match="No body named"):
        DebugDraw(model, data, scene).body_com("tip_site")


def test_body_com_is_a_snapshot(model, data, scene):
    draw = DebugDraw(model, data, scene)
    draw.body_com("end_effector")
    queued = draw.geom_specs()[0].start.copy()
    data.qpos[0] = -1.0
    mujoco.mj_forward(model, data)
    assert_close(draw.geom_specs()[0].start, queued)


# ---------------------------------------------------------------- joint_axis


@pytest.mark.parametrize("joint", ["hinge_j", "slide_j"])
def test_joint_axis_endpoint(jmodel, jdata, jdraw, joint):
    jid = jmodel.joint(joint).id
    jdraw.joint_axis(joint, scale=0.3, label=joint)
    (spec,) = jdraw.geom_specs()
    assert spec.kind == "arrow" and spec.label == joint
    assert_close(spec.start, jdata.xanchor[jid])
    assert_close(spec.end, jdata.xanchor[jid] + 0.3 * jdata.xaxis[jid])


def test_joint_axis_by_id(jmodel, jdata, jdraw):
    jid = jmodel.joint("slide_j").id
    jdraw.joint_axis(jid)
    assert_close(jdraw.geom_specs()[0].start, jdata.xanchor[jid])


def test_joint_axis_follows_hinge_rotation(jmodel, jdata, jdraw):
    # The slide axis is carried by the hinge, so moving the hinge must move it.
    jdraw.joint_axis("slide_j", key="ax")
    before = jdraw.geom_specs()[0].end.copy()
    jdata.qpos[0] = -0.5
    mujoco.mj_forward(jmodel, jdata)
    jdraw.joint_axis("slide_j", key="ax")
    assert np.linalg.norm(jdraw.geom_specs()[0].end - before) > 1e-3


@pytest.mark.parametrize("joint,jtype", [("ball_j", "ball"), ("free_j", "free")])
def test_joint_axis_rejects_ball_and_free(jdraw, joint, jtype):
    with pytest.raises(ValueError, match=jtype):
        jdraw.joint_axis(joint)


def test_joint_axis_unknown_name_suggests(jdraw):
    with pytest.raises(UnknownTargetError, match="hinge_j"):
        jdraw.joint_axis("hinge_k")


def test_joint_axis_id_out_of_range(jdraw):
    with pytest.raises(UnknownTargetError, match="Joint ID 99 out of range"):
        jdraw.joint_axis(99)


@pytest.mark.parametrize("kwargs", [{"scale": 0}, {"scale": -0.1}, {"width": 0}])
def test_joint_axis_rejects_bad_sizes(jdraw, kwargs):
    with pytest.raises(ValueError):
        jdraw.joint_axis("hinge_j", **kwargs)


def test_joint_hinge_in_shared_fixture(model, data, scene):
    draw = DebugDraw(model, data, scene)
    draw.joint_axis("hinge", scale=0.2)
    jid = model.joint("hinge").id
    assert_close(draw.geom_specs()[0].end, data.xanchor[jid] + 0.2 * data.xaxis[jid])
