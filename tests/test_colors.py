import math

import pytest

from mujoco_debugdraw import NAMED_COLORS, InvalidColorError, parse_color

SPEC_NAMES = ["red", "green", "blue", "yellow", "cyan", "magenta", "white", "black", "gray", "orange", "lime"]


@pytest.mark.parametrize("name", SPEC_NAMES)
def test_named_colors(name):
    rgba = parse_color(name)
    assert len(rgba) == 4 and rgba[3] == 1.0
    assert rgba == NAMED_COLORS[name]


def test_named_is_case_and_space_insensitive():
    assert parse_color("  Red ") == parse_color("red")


def test_hex_rgb():
    assert parse_color("#FF8000") == pytest.approx((1.0, 128 / 255, 0.0, 1.0))


def test_hex_rgba():
    assert parse_color("#00ff0080") == pytest.approx((0.0, 1.0, 0.0, 128 / 255))


def test_rgb_and_rgba_sequences():
    assert parse_color([0.1, 0.2, 0.3]) == pytest.approx((0.1, 0.2, 0.3, 1.0))
    assert parse_color((0.1, 0.2, 0.3, 0.4)) == pytest.approx((0.1, 0.2, 0.3, 0.4))


def test_numpy_input():
    np = pytest.importorskip("numpy")
    assert parse_color(np.array([1.0, 0.0, 0.0])) == (1.0, 0.0, 0.0, 1.0)


@pytest.mark.parametrize(
    "bad",
    ["purple-ish", "#12345", "#GGGGGG", "#1234567", "", [1, 0], [0, 0, 0, 0, 0],
     [1.5, 0, 0], [-0.1, 0, 0], [255, 0, 0], [math.nan, 0, 0], 3, None],
)
def test_rejects_invalid(bad):
    with pytest.raises(InvalidColorError):
        parse_color(bad)


def test_error_is_value_error():
    with pytest.raises(ValueError):
        parse_color("nope")


def test_255_hint_in_message():
    with pytest.raises(InvalidColorError, match="255"):
        parse_color([255, 0, 0])
