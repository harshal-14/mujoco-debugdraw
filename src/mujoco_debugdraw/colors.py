"""Color parsing: named colors, hex strings and normalized RGB/RGBA sequences."""

from __future__ import annotations

from typing import Sequence, Tuple, Union

from .exceptions import InvalidColorError

RGBA = Tuple[float, float, float, float]
ColorLike = Union[str, Sequence[float]]

NAMED_COLORS: dict[str, RGBA] = {
    "red": (1.0, 0.0, 0.0, 1.0),
    "green": (0.0, 0.6, 0.0, 1.0),
    "blue": (0.0, 0.3, 1.0, 1.0),
    "yellow": (1.0, 0.9, 0.0, 1.0),
    "cyan": (0.0, 0.9, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0, 1.0),
    "white": (1.0, 1.0, 1.0, 1.0),
    "black": (0.0, 0.0, 0.0, 1.0),
    "gray": (0.5, 0.5, 0.5, 1.0),
    "grey": (0.5, 0.5, 0.5, 1.0),
    "orange": (1.0, 0.55, 0.0, 1.0),
    "lime": (0.2, 1.0, 0.2, 1.0),
}


def parse_color(color: ColorLike) -> RGBA:
    """Convert a user-supplied color into a normalized ``(r, g, b, a)`` tuple.

    Accepts a named color (``"red"``), ``"#RRGGBB"``, ``"#RRGGBBAA"``, or a
    sequence of 3 or 4 floats in ``[0, 1]``. Invalid input raises
    :class:`InvalidColorError`; values are never silently clamped.
    """
    if isinstance(color, str):
        s = color.strip().lower()
        if s in NAMED_COLORS:
            return NAMED_COLORS[s]
        if s.startswith("#"):
            hexpart = s[1:]
            if len(hexpart) not in (6, 8):
                raise InvalidColorError(
                    f"Hex color {color!r} must be '#RRGGBB' or '#RRGGBBAA'."
                )
            try:
                vals = [int(hexpart[i : i + 2], 16) / 255.0 for i in range(0, len(hexpart), 2)]
            except ValueError:
                raise InvalidColorError(f"Hex color {color!r} contains non-hex digits.") from None
            if len(vals) == 3:
                vals.append(1.0)
            return (vals[0], vals[1], vals[2], vals[3])
        known = ", ".join(sorted(NAMED_COLORS))
        raise InvalidColorError(
            f"Unknown color name {color!r}. Use one of: {known}; a hex string; or [r, g, b(, a)]."
        )

    try:
        vals = [float(v) for v in color]  # type: ignore[union-attr]
    except TypeError:
        raise InvalidColorError(
            f"Color must be a string or a sequence of 3-4 floats, got {type(color).__name__}."
        ) from None
    if len(vals) not in (3, 4):
        raise InvalidColorError(f"Color sequence must have 3 or 4 values, got {len(vals)}.")
    for v in vals:
        if not (0.0 <= v <= 1.0):  # also rejects NaN
            raise InvalidColorError(
                f"Color components must be normalized to [0, 1], got {vals}. "
                "Divide 0-255 values by 255."
            )
    if len(vals) == 3:
        vals.append(1.0)
    return (vals[0], vals[1], vals[2], vals[3])
