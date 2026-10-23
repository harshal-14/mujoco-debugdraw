import mujoco
import numpy as np
import pytest

from mujoco_debugdraw import DebugDraw, UnknownTargetError, parse_color

from conftest import assert_close

M_A, M_B = 1.0, 0.4  # different masses, so a sign or bookkeeping error cannot cancel out

# Box B rests on box A, which rests on the floor.
STACK_XML = f"""
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 .1"/>
    <body name="A" pos="0 0 0.05">
      <freejoint/>
      <geom name="a" type="box" size="0.05 0.05 0.05" mass="{M_A}"/>
    </body>
    <body name="B" pos="0 0 0.14">
      <freejoint/>
      <geom name="b" type="box" size="0.04 0.04 0.04" mass="{M_B}"/>
    </body>
    <body name="far" pos="2 0 1">
      <geom type="sphere" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""

NORMAL = parse_color("red")
FORCE = parse_color("yellow")


@pytest.fixture(scope="module")
def stack():
    m = mujoco.MjModel.from_xml_string(STACK_XML)
    d = mujoco.MjData(m)
    while d.time < 2.0:
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)  # contacts and forces for the current state
    assert d.ncon > 0
    return m, d


def _contacts(d):
    return [d.contact[i] for i in range(d.ncon)]


def _arrows(draw, rgba):
    return [(s.start, s.end - s.start) for s in draw.geom_specs() if s.rgba == rgba]


def _bodies(m, c):
    return {int(m.geom_bodyid[g]) for g in c.geom if g >= 0}


@pytest.mark.parametrize("body,mass", [("A", M_A), ("B", M_B)])
def test_forces_on_a_body_sum_to_its_weight(stack, body, mass):
    m, d = stack
    draw = DebugDraw(m, d, mujoco.MjvScene(m, maxgeom=200))
    draw.contacts(body=body, force_scale=0.01)
    total = sum(v for _, v in _arrows(draw, FORCE)) / 0.01
    weight = mass * -m.opt.gravity[2]
    np.testing.assert_allclose(total, [0, 0, weight], atol=0.02 * weight)


def test_body_a_sees_both_geom_orders(stack):
    # A is geom[1] in the floor contacts and geom[0] in the A-B contacts, so the
    # weight test above only passes if the A-B forces are flipped onto A.
    m, d = stack
    a = m.body("A").id
    sides = {k for c in _contacts(d) for k in (0, 1) if c.geom[k] >= 0 and m.geom_bodyid[c.geom[k]] == a}
    assert sides == {0, 1}


def test_unfiltered_arrows_show_force_on_geom1(stack):
    m, d = stack
    draw = DebugDraw(m, d, mujoco.MjvScene(m, maxgeom=200))
    draw.contacts(normal_scale=0.05)
    normals = _arrows(draw, NORMAL)
    assert len(normals) == d.ncon
    for (start, vec), c in zip(normals, _contacts(d)):
        assert_close(start, c.pos)
        assert_close(vec, 0.05 * c.frame[:3])
    forces = _arrows(draw, FORCE)
    assert forces
    for _, vec in forces:
        assert vec[2] > -1e-9  # everything at rest is pushed up onto geom[1]


def test_normals_have_normal_scale_length(stack):
    m, d = stack
    for body in (None, "A", "B"):
        draw = DebugDraw(m, d, mujoco.MjvScene(m, maxgeom=200))
        draw.contacts(body=body, normal_scale=0.07)
        lengths = [np.linalg.norm(v) for _, v in _arrows(draw, NORMAL)]
        assert lengths and np.allclose(lengths, 0.07)


def test_filtered_normals_point_into_the_body(stack):
    m, d = stack
    draw = DebugDraw(m, d, mujoco.MjvScene(m, maxgeom=200))
    draw.contacts(body="A")
    # A is pushed up by the floor and down by B.
    zs = sorted(np.sign(v[2]) for _, v in _arrows(draw, NORMAL))
    assert zs[0] < 0 < zs[-1]


def test_geom_count_at_most_two_per_contact(stack):
    m, d = stack
    draw = DebugDraw(m, d, mujoco.MjvScene(m, maxgeom=200))
    draw.contacts()
    assert 0 < draw.num_geoms <= 2 * d.ncon
    assert draw.num_primitives == 1


def test_body_filter_excludes_other_contacts(stack):
    m, d = stack
    b = m.body("B").id
    expected = [c.pos.copy() for c in _contacts(d) if b in _bodies(m, c)]
    assert 0 < len(expected) < d.ncon  # the floor-A contacts must be left out
    draw = DebugDraw(m, d, mujoco.MjvScene(m, maxgeom=200))
    draw.contacts(body="B")
    starts = [s for s, _ in _arrows(draw, NORMAL)]
    assert len(starts) == len(expected)
    for s, e in zip(starts, expected):
        assert_close(s, e)


def test_body_with_no_contacts_draws_nothing(stack):
    m, d = stack
    draw = DebugDraw(m, d, mujoco.MjvScene(m, maxgeom=200))
    draw.contacts(body="far")
    assert draw.num_geoms == 0 and draw.num_primitives == 1


def test_normals_only_and_forces_only(stack):
    m, d = stack
    draw = DebugDraw(m, d, mujoco.MjvScene(m, maxgeom=200))
    draw.contacts(forces=False)
    assert len(_arrows(draw, NORMAL)) == draw.num_geoms == d.ncon
    draw.clear()
    draw.contacts(normals=False)
    assert _arrows(draw, NORMAL) == [] and draw.num_geoms > 0


def test_contact_frame_works_with_frame_at(stack):
    m, d = stack
    draw = DebugDraw(m, d, mujoco.MjvScene(m, maxgeom=200))
    c = d.contact[0]
    draw.frame_at(c.pos, c.frame.reshape(3, 3).T, scale=0.1)
    x_axis = draw.geom_specs()[0]
    assert_close(x_axis.end - x_axis.start, 0.1 * c.frame[:3])


def test_contacts_are_a_snapshot(stack):
    m, settled = stack
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:] = settled.qpos, settled.qvel
    mujoco.mj_forward(m, d)
    draw = DebugDraw(m, d, mujoco.MjvScene(m, maxgeom=200))
    draw.contacts()
    queued = [(s.start.copy(), s.end.copy()) for s in draw.geom_specs()]
    d.qpos[0] += 0.3  # slide both boxes along x; every contact moves
    d.qpos[7] += 0.3
    mujoco.mj_forward(m, d)
    after = [(s.start, s.end) for s in draw.geom_specs()]
    assert len(after) == len(queued)
    for (a0, a1), (b0, b1) in zip(queued, after):
        assert np.array_equal(a0, b0) and np.array_equal(a1, b1)


# Contacts in the "gap" band are reported in data.contact but excluded from the
# solver, so they have efc_address == -1. Older MuJoCo puts the band at
# margin - gap < dist < margin; newer MuJoCo (3.15 at least) at
# margin < dist < margin + gap. With margin = gap = 2 cm, a box 1 cm above
# the floor is in the band on old versions and 3 cm above it is on new ones.
GAP_XML = """
<mujoco>
  <worldbody>
    <geom type="plane" size="1 1 .1" margin="0.02" gap="0.02"/>
    <body>
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


def _in_gap_state(m):
    for height in (0.01, 0.03):
        d = mujoco.MjData(m)
        d.qpos[2] = 0.05 + height  # box half-size + clearance
        mujoco.mj_forward(m, d)
        if d.ncon > 0 and all(c.efc_address < 0 for c in _contacts(d)):
            return d
    pytest.fail("could not place the box so that all its contacts are in the gap")


def test_contact_outside_solver_gets_normal_but_no_force():
    m = mujoco.MjModel.from_xml_string(GAP_XML)
    d = _in_gap_state(m)
    draw = DebugDraw(m, d, mujoco.MjvScene(m, maxgeom=100))
    draw.contacts()
    assert len(_arrows(draw, NORMAL)) == d.ncon
    assert _arrows(draw, FORCE) == []


def test_no_contacts_is_fine(model, scene):
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    draw = DebugDraw(model, d, scene)
    draw.contacts()
    draw.sync()
    assert scene.ngeom == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"normals": False, "forces": False},
        {"normal_scale": 0},
        {"force_scale": -1},
        {"width": 0},
        {"normal_color": "nope"},
    ],
)
def test_contacts_rejects(model, data, scene, kwargs):
    with pytest.raises(ValueError):
        DebugDraw(model, data, scene).contacts(**kwargs)


def test_contacts_unknown_body(model, data, scene):
    with pytest.raises(UnknownTargetError, match="Did you mean"):
        DebugDraw(model, data, scene).contacts(body="bx")
