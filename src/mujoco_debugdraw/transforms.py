"""Pure-NumPy geometry helpers (no MuJoCo or OpenGL required)."""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Vectors shorter than this are treated as zero-length and not drawn.
ZERO_EPS = 1e-9

# as_rotation accepts R when the Frobenius norm of R^T R - I is below this.
ROTATION_TOL = 1e-5


# The validators below copy their input, so a primitive built from a view
# into MjData (e.g. data.xpos[i]) keeps the value it had when it was drawn.


def as_vec3(value: object, name: str = "value") -> np.ndarray:
    """Return ``value`` as a finite float64 array of shape ``(3,)`` or raise ``ValueError``."""
    arr = np.array(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite numbers, got {arr.tolist()}.")
    return arr


def as_rotation(value: object, name: str = "rotation") -> np.ndarray:
    """Return ``value`` as a ``(3, 3)`` rotation matrix or raise ``ValueError``.

    The *columns* of a ``(3, 3)`` input are the local X/Y/Z axes in world
    coordinates, as in ``data.xmat[i].reshape(3, 3)``. A flat ``(9,)`` input
    is read row-major, like ``data.xmat[i]`` itself. Reflections and
    non-orthonormal matrices are rejected.
    """
    arr = np.array(value, dtype=np.float64)
    if arr.shape == (9,):
        arr = arr.reshape(3, 3)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3) or (9,), got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite numbers.")
    if np.linalg.norm(arr.T @ arr - np.eye(3)) >= ROTATION_TOL:
        raise ValueError(f"{name} must be orthonormal (R.T @ R == I).")
    if np.linalg.det(arr) <= 0:
        raise ValueError(f"{name} must be a proper rotation (det > 0); got a reflection.")
    return arr


def as_points(value: object, name: str = "points") -> np.ndarray:
    """Return ``value`` as a finite float64 array of shape ``(N, 3)`` with ``N >= 2``."""
    arr = np.array(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got shape {arr.shape}.")
    if arr.shape[0] < 2:
        raise ValueError(f"{name} needs at least 2 points to draw a line, got {arr.shape[0]}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite numbers.")
    return arr


def arrow_endpoint(origin: np.ndarray, vector: np.ndarray, scale: float) -> np.ndarray:
    """Tip of an arrow: ``origin + scale * vector``."""
    return np.asarray(origin, dtype=np.float64) + float(scale) * np.asarray(vector, dtype=np.float64)


def segments(points: np.ndarray) -> list[Tuple[np.ndarray, np.ndarray]]:
    """Consecutive ``(start, end)`` pairs of a polyline."""
    return [(points[i], points[i + 1]) for i in range(len(points) - 1)]


def frame_axes(
    position: np.ndarray, xmat: np.ndarray, scale: float
) -> list[Tuple[np.ndarray, np.ndarray]]:
    """World-space ``(origin, tip)`` for the X, Y and Z axes of a frame.

    ``xmat`` is a MuJoCo row-major 3x3 rotation (9 values, as in
    ``data.xmat`` / ``data.site_xmat``). Its *columns* are the local axes
    expressed in world coordinates.
    """
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    rot = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    return [(pos, pos + scale * rot[:, i]) for i in range(3)]


def is_zero(vector: np.ndarray) -> bool:
    return float(np.linalg.norm(vector)) < ZERO_EPS
