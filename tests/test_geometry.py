import numpy as np
import pytest

from mujoco_debugdraw import DebugDraw
from mujoco_debugdraw.primitives import Primitive, expand
from mujoco_debugdraw.transforms import arrow_endpoint, as_points, as_vec3, segments

from conftest import assert_close


def test_arrow_endpoint_formula():
    o, v = np.array([1.0, 2.0, 3.0]), np.array([0.5, -1.0, 2.0])
    assert_close(arrow_endpoint(o, v, 0.2), o + 0.2 * v)


def test_arrow_direction_preserved_after_scaling():
    o, v = np.zeros(3), np.array([3.0, 4.0, 12.0])
    for s in (0.01, 0.5, 7.0):
        d = arrow_endpoint(o, v, s) - o
        assert_close(d / np.linalg.norm(d), v / np.linalg.norm(v))


def test_negative_scale_flips_arrow():
    o, v = np.zeros(3), np.array([1.0, 0, 0])
    assert_close(arrow_endpoint(o, v, -2.0), [-2.0, 0, 0])


def test_segments_connect_consecutive_points():
    pts = as_points([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]])
    segs = segments(pts)
    assert len(segs) == 3
    for i, (a, b) in enumerate(segs):
        assert_close(a, pts[i])
        assert_close(b, pts[i + 1])


@pytest.mark.parametrize("bad", [[1, 2], [1, 2, 3, 4], [[1, 2, 3]], "abc", [1, np.nan, 3], [np.inf, 0, 0]])
def test_as_vec3_rejects(bad):
    with pytest.raises(ValueError):
        as_vec3(bad)


@pytest.mark.parametrize("bad", [[[0, 0, 0]], [0, 0, 0], [[0, 0], [1, 1]], [[0, 0, 0], [np.nan, 0, 0]]])
def test_as_points_rejects(bad):
    with pytest.raises(ValueError):
        as_points(bad)


def test_line_with_one_point_message(model, data, scene):
    draw = DebugDraw(model, data, scene)
    with pytest.raises(ValueError, match="at least 2 points"):
        draw.line([[0, 0, 0]])


def test_zero_vector_arrow_draws_nothing(model, data, scene):
    draw = DebugDraw(model, data, scene)
    draw.arrow([0, 0, 0], [0, 0, 0])
    draw.arrow([0, 0, 0], [1, 0, 0], scale=0.0)
    assert draw.num_geoms == 0
    draw.sync()
    assert scene.ngeom == 0


def test_expand_counts(model, data, scene):
    draw = DebugDraw(model, data, scene)
    draw.point([0, 0, 0])
    draw.line([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    draw.arrow([0, 0, 0], [0, 0, 1])
    draw.frame("end_effector")
    draw.text("hello")
    assert draw.num_primitives == 5
    assert draw.num_geoms == 1 + 2 + 1 + 3 + 0


def test_arrow_spec_endpoints(model, data, scene):
    draw = DebugDraw(model, data, scene)
    draw.arrow([0.1, 0.2, 0.3], [1.0, 0.0, -1.0], scale=0.25)
    (spec,) = draw.geom_specs()
    assert_close(spec.start, [0.1, 0.2, 0.3])
    assert_close(spec.end, [0.35, 0.2, 0.05])


def test_unknown_primitive_kind():
    with pytest.raises(ValueError):
        expand(Primitive("blob", None, (1, 1, 1, 1), {}))
