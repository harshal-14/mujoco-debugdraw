"""Exceptions raised by mujoco_debugdraw."""

from __future__ import annotations


class DebugDrawError(Exception):
    """Base class for all mujoco_debugdraw errors."""


class InvalidColorError(DebugDrawError, ValueError):
    """A color could not be parsed or was out of range."""


class UnknownTargetError(DebugDrawError, KeyError):
    """A body or site name/ID passed to ``frame()`` does not exist in the model."""

    def __str__(self) -> str:  # KeyError wraps messages in quotes; undo that.
        return str(self.args[0]) if self.args else ""


class IncompatibleViewerError(DebugDrawError, TypeError):
    """The object passed as ``viewer`` is not a supported drawing target."""


class CapacityWarning(UserWarning):
    """Emitted when primitives are dropped because ``max_geoms`` was reached."""
