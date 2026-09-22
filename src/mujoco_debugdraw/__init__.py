"""Draw targets, frames, vectors, paths and labels in the MuJoCo Python viewer."""

from .colors import NAMED_COLORS, parse_color
from .debug_draw import DebugDraw
from .exceptions import (
    CapacityWarning,
    DebugDrawError,
    IncompatibleViewerError,
    InvalidColorError,
    UnknownTargetError,
)

__version__ = "0.1.0"

__all__ = [
    "DebugDraw",
    "parse_color",
    "NAMED_COLORS",
    "DebugDrawError",
    "InvalidColorError",
    "UnknownTargetError",
    "IncompatibleViewerError",
    "CapacityWarning",
    "__version__",
]
