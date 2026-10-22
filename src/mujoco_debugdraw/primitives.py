"""Backend-independent primitive records and their expansion into geometry specs.

A :class:`Primitive` is what the user asked for (a point, a line, ...). A
:class:`GeomSpec` is one MuJoCo visual geom needed to draw it. Keeping this
step free of MuJoCo scene objects makes the geometry fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .colors import RGBA
from .transforms import arrow_endpoint, frame_axes, is_zero, segments

AXIS_COLORS: tuple[RGBA, RGBA, RGBA] = (
    (1.0, 0.1, 0.1, 1.0),  # X: red
    (0.1, 0.85, 0.1, 1.0),  # Y: green
    (0.15, 0.35, 1.0, 1.0),  # Z: blue
)


@dataclass
class Primitive:
    kind: str  # "point" | "line" | "arrow" | "frame" | "box" | "contacts" | "text"
    key: Optional[str]
    color: RGBA
    payload: dict[str, Any] = field(default_factory=dict)
    ttl: Optional[int] = None  # sync() calls left before removal; None = until clear()


@dataclass
class GeomSpec:
    """One visual geom. ``kind`` is "sphere", "box", "line" or "arrow"."""

    kind: str
    rgba: RGBA
    start: np.ndarray  # sphere/box center, or connector start
    end: Optional[np.ndarray] = None  # connector end
    # sphere radius, box half-sizes (3,), line width (px), or arrow shaft radius
    size: Any = 0.0
    label: str = ""
    mat: Optional[np.ndarray] = None  # box orientation, row-major 3x3 as (9,)


def expand(prim: Primitive) -> list[GeomSpec]:
    """Return the geoms needed to draw ``prim`` (empty for text and zero arrows)."""
    p = prim.payload
    label = p.get("label") or ""

    if prim.kind == "point":
        return [GeomSpec("sphere", prim.color, p["position"], size=p["size"], label=label)]

    if prim.kind == "line":
        return [
            GeomSpec("line", prim.color, a, b, size=p["width"])
            for a, b in segments(p["points"])
        ]

    if prim.kind == "arrow":
        vec = p["scale"] * p["vector"]
        if is_zero(vec):
            return []  # Documented: zero-length arrows draw nothing.
        tip = arrow_endpoint(p["origin"], p["vector"], p["scale"])
        return [GeomSpec("arrow", prim.color, p["origin"], tip, size=p["width"], label=label)]

    if prim.kind == "frame":
        axes = frame_axes(p["position"], p["xmat"], p["scale"])
        width = p["width"]
        alpha = p.get("alpha", 1.0)
        out = []
        for i, (a, b) in enumerate(axes):
            rgba = AXIS_COLORS[i][:3] + (alpha,)
            out.append(GeomSpec("arrow", rgba, a, b, size=width, label=label if i == 0 else ""))
        return out

    if prim.kind == "box":
        return [
            GeomSpec(
                "box", prim.color, p["center"], size=p["half_size"], label=label,
                mat=p["rotation"].flatten(),
            )
        ]

    if prim.kind == "contacts":
        return [
            GeomSpec("arrow", rgba, start, end, size=p["width"])
            for start, end, rgba in p["arrows"]
            if not is_zero(end - start)
        ]

    if prim.kind == "text":
        return []

    raise ValueError(f"Unknown primitive kind {prim.kind!r}")


def geom_count(prim: Primitive) -> int:
    return len(expand(prim))
