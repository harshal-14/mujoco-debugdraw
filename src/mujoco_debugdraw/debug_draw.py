"""The :class:`DebugDraw` class: runtime debug drawing for MuJoCo viewers."""

from __future__ import annotations

import difflib
import sys
import time
import warnings
from typing import Any, Optional, Union

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - exercised only without mujoco
    raise ImportError(
        "mujoco_debugdraw requires the official 'mujoco' Python package. "
        "Install it with: pip install mujoco"
    ) from exc

from .colors import ColorLike, parse_color
from .exceptions import CapacityWarning, IncompatibleViewerError, UnknownTargetError
from .primitives import GeomSpec, Primitive, expand, geom_count
from .transforms import as_points, as_vec3

TEXT_POSITIONS = {
    "top_left": mujoco.mjtGridPos.mjGRID_TOPLEFT,
    "top_right": mujoco.mjtGridPos.mjGRID_TOPRIGHT,
    "bottom_left": mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
    "bottom_right": mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
}

_GEOM_TYPES = {
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "line": mujoco.mjtGeom.mjGEOM_LINE,
    "arrow": mujoco.mjtGeom.mjGEOM_ARROW,
}

_IDENTITY_MAT = np.eye(3).flatten()

# MuJoCo renders mjGEOM_ARROW as a shaft of size[2]/3 plus a cone of size[2]/6,
# i.e. the visible arrow is half of size[2] (see render_gl3.c and the Filament
# renderable.cc). We pass a doubled endpoint to mjv_connector so the drawn tip
# lands exactly on origin + scale * vector. tests/test_render_pixels.py guards this.
_ARROW_LENGTH_FACTOR = 2.0

Target = Union[str, int]


class DebugDraw:
    """Draw temporary debug geometry and text in a MuJoCo viewer.

    Nothing is added to the model: primitives become *visual-only* geoms in
    the viewer's user scene, so they have no mass, cause no contacts, and do
    not change the simulation. ``model`` and ``data`` are only read.

    Typical loop (call ``draw.sync()`` **before** ``viewer.sync()``)::

        while viewer.is_running():
            mujoco.mj_step(model, data)
            draw.clear()
            draw.point(target, color="lime", label="target")
            draw.frame("end_effector")
            draw.sync()
            viewer.sync()

    Args:
        model: The ``mujoco.MjModel``.
        data: The ``mujoco.MjData`` for ``model``.
        viewer: A handle from ``mujoco.viewer.launch_passive()``. For
            offscreen rendering a ``mujoco.Renderer`` or a bare
            ``mujoco.MjvScene`` is also accepted (see README).
        max_geoms: Upper bound on debug geoms per frame. Extra primitives are
            dropped with a :class:`CapacityWarning`.
    """

    def __init__(self, model: Any, data: Any, viewer: Any, max_geoms: int = 500) -> None:
        if not isinstance(model, mujoco.MjModel):
            raise TypeError(f"model must be a mujoco.MjModel, got {type(model).__name__}.")
        if not isinstance(data, mujoco.MjData):
            raise TypeError(f"data must be a mujoco.MjData, got {type(data).__name__}.")
        if data.qpos.shape[0] != model.nq or data.xpos.shape[0] != model.nbody:
            raise ValueError("data does not belong to model (mismatched sizes).")
        if not isinstance(max_geoms, int) or max_geoms < 1:
            raise ValueError(f"max_geoms must be a positive integer, got {max_geoms!r}.")

        self.model = model
        self.data = data
        self.viewer = viewer
        self.max_geoms = max_geoms
        self._mode = self._detect_mode(viewer)
        self._prims: dict[str, Primitive] = {}
        self._auto_id = 0
        self._geoms_queued = 0
        self._base_ngeom: Optional[int] = None
        self._texts_active = False
        self._last_print = 0.0
        #: Number of geoms dropped in the current frame because of ``max_geoms``.
        self.dropped = 0

    # ------------------------------------------------------------------ setup

    @staticmethod
    def _detect_mode(viewer: Any) -> str:
        if isinstance(viewer, mujoco.MjvScene):
            return "scene"
        if hasattr(viewer, "user_scn") and hasattr(viewer, "lock"):
            return "viewer"
        if isinstance(getattr(viewer, "scene", None), mujoco.MjvScene):
            return "renderer"
        raise IncompatibleViewerError(
            "viewer must be the handle returned by mujoco.viewer.launch_passive() "
            f"(or a mujoco.Renderer / mujoco.MjvScene), got {type(viewer).__name__}."
        )

    # -------------------------------------------------------------- primitives

    def point(
        self,
        position: Any,
        color: ColorLike = "red",
        size: float = 0.02,
        label: Optional[str] = None,
        key: Optional[str] = None,
    ) -> None:
        """Draw a sphere marker of radius ``size`` (m) at a world-space position."""
        self._add(
            Primitive(
                "point",
                key,
                parse_color(color),
                {"position": as_vec3(position, "position"), "size": _positive(size, "size"), "label": label},
            )
        )

    def line(
        self,
        points: Any,
        color: ColorLike = "cyan",
        width: float = 2.0,
        key: Optional[str] = None,
    ) -> None:
        """Draw a polyline through ``points`` (shape ``(N, 3)``, ``N >= 2``).

        ``width`` is in screen pixels (MuJoCo line geoms).
        """
        self._add(
            Primitive(
                "line",
                key,
                parse_color(color),
                {"points": as_points(points, "points"), "width": _positive(width, "width")},
            )
        )

    def arrow(
        self,
        origin: Any,
        vector: Any,
        color: ColorLike = "yellow",
        scale: float = 1.0,
        width: float = 0.01,
        label: Optional[str] = None,
        key: Optional[str] = None,
    ) -> None:
        """Draw an arrow from ``origin`` to ``origin + scale * vector``.

        ``width`` is the shaft radius in meters. A zero-length result draws
        nothing (and does not raise).
        """
        scale = float(scale)
        if not np.isfinite(scale):
            raise ValueError(f"scale must be finite, got {scale}.")
        self._add(
            Primitive(
                "arrow",
                key,
                parse_color(color),
                {
                    "origin": as_vec3(origin, "origin"),
                    "vector": as_vec3(vector, "vector"),
                    "scale": scale,
                    "width": _positive(width, "width"),
                    "label": label,
                },
            )
        )

    def frame(
        self,
        target: Target,
        scale: float = 0.10,
        key: Optional[str] = None,
        label: Optional[str] = None,
        kind: Optional[str] = None,
        width: Optional[float] = None,
    ) -> None:
        """Draw X/Y/Z axes (red/green/blue) at a body or site's current pose.

        ``target`` is a body or site name, or an integer body ID. When a name
        exists as both a body and a site, the body wins; pass
        ``kind="site"`` (or ``kind="body"``) to choose explicitly. An integer
        with ``kind="site"`` is treated as a site ID.
        """
        scale = _positive(scale, "scale")
        pos, xmat = self._resolve_pose(target, kind)
        self._add(
            Primitive(
                "frame",
                key,
                (1.0, 1.0, 1.0, 1.0),
                {
                    "position": pos.copy(),
                    "xmat": xmat.copy(),
                    "scale": scale,
                    "width": width if width is not None else max(scale * 0.05, 1e-4),
                    "label": label,
                },
            )
        )

    def text(self, message: str, position: str = "top_left", key: Optional[str] = None) -> None:
        """Show status text in a screen corner of the passive viewer.

        ``position`` is one of ``top_left``, ``top_right``, ``bottom_left``,
        ``bottom_right``. Multiple messages in the same corner are stacked.
        """
        if position not in TEXT_POSITIONS:
            raise ValueError(
                f"position must be one of {sorted(TEXT_POSITIONS)}, got {position!r}."
            )
        self._add(Primitive("text", key, (1, 1, 1, 1), {"message": str(message), "position": position}))

    # --------------------------------------------------------------- lifecycle

    def clear(self) -> None:
        """Forget all queued primitives. Never touches the model or sim state."""
        self._prims.clear()
        self._geoms_queued = 0
        self.dropped = 0

    def sync(self) -> None:
        """Write the queued primitives into the viewer's scene.

        Passive viewer: call once per frame, before ``viewer.sync()``.
        ``mujoco.Renderer``: call after ``renderer.update_scene()`` and
        before ``renderer.render()``.
        """
        specs = [g for p in self._prims.values() for g in expand(p)]
        if self._mode == "viewer":
            with self.viewer.lock():
                scn = self.viewer.user_scn
                if self._base_ngeom is None:
                    self._base_ngeom = scn.ngeom
                scn.ngeom = self._base_ngeom
                self._write(scn, specs)
            self._sync_texts()
        else:
            scn = self.viewer if self._mode == "scene" else self.viewer.scene
            self._write(scn, specs)

    # ------------------------------------------------------------ introspection

    @property
    def num_primitives(self) -> int:
        return len(self._prims)

    @property
    def num_geoms(self) -> int:
        """Debug geoms that the next ``sync()`` will write."""
        return self._geoms_queued

    @property
    def texts(self) -> dict[str, list[str]]:
        """Queued text messages grouped by screen corner."""
        out: dict[str, list[str]] = {}
        for p in self._prims.values():
            if p.kind == "text":
                out.setdefault(p.payload["position"], []).append(p.payload["message"])
        return out

    def geom_specs(self) -> list[GeomSpec]:
        """The geoms the next ``sync()`` will write (useful for testing)."""
        return [g for p in self._prims.values() for g in expand(p)]

    # ---------------------------------------------------------------- internals

    def _add(self, prim: Primitive) -> None:
        key = prim.key
        if key is None:
            key = f"__auto_{self._auto_id}"
            self._auto_id += 1
        old = self._prims.pop(key, None)
        if old is not None:
            self._geoms_queued -= geom_count(old)
        n = geom_count(prim)
        if self._geoms_queued + n > self.max_geoms:
            self.dropped += n
            warnings.warn(
                f"mujoco_debugdraw: max_geoms={self.max_geoms} reached; dropping a "
                f"{prim.kind} primitive. Call clear() every frame or raise max_geoms.",
                CapacityWarning,
                stacklevel=3,
            )
            return
        self._geoms_queued += n
        self._prims[key] = prim

    def _write(self, scn: Any, specs: list[GeomSpec]) -> None:
        free = scn.maxgeom - scn.ngeom
        if len(specs) > free:
            warnings.warn(
                f"mujoco_debugdraw: scene has room for {free} more geoms; "
                f"dropping {len(specs) - free}.",
                CapacityWarning,
                stacklevel=3,
            )
            specs = specs[:free]
        for spec in specs:
            g = scn.geoms[scn.ngeom]
            rgba = np.asarray(spec.rgba, dtype=np.float32)
            gtype = _GEOM_TYPES[spec.kind]
            if spec.kind == "sphere":
                mujoco.mjv_initGeom(
                    g, gtype, np.array([spec.size, 0.0, 0.0]), spec.start, _IDENTITY_MAT, rgba
                )
            else:
                mujoco.mjv_initGeom(g, gtype, np.zeros(3), np.zeros(3), _IDENTITY_MAT, rgba)
                start = np.asarray(spec.start, dtype=np.float64)
                end = np.asarray(spec.end, dtype=np.float64)
                if spec.kind == "arrow":
                    end = start + _ARROW_LENGTH_FACTOR * (end - start)
                mujoco.mjv_connector(g, gtype, spec.size, start, end)
            g.label = spec.label[:99]
            scn.ngeom += 1

    def _sync_texts(self) -> None:
        grouped = self.texts
        set_texts = getattr(self.viewer, "set_texts", None)
        if set_texts is None:
            # Older MuJoCo without a text overlay API: rate-limited terminal fallback.
            now = time.monotonic()
            if grouped and now - self._last_print >= 1.0:
                self._last_print = now
                for pos, msgs in grouped.items():
                    print(f"[debugdraw {pos}] " + " | ".join(msgs), file=sys.stderr)
            return
        if grouped:
            font = mujoco.mjtFontScale.mjFONTSCALE_150
            set_texts([(font, TEXT_POSITIONS[pos], "\n".join(msgs), None) for pos, msgs in grouped.items()])
            self._texts_active = True
        elif self._texts_active:
            self.viewer.clear_texts()
            self._texts_active = False

    def _resolve_pose(self, target: Target, kind: Optional[str]) -> tuple[np.ndarray, np.ndarray]:
        m, d = self.model, self.data
        if kind not in (None, "body", "site"):
            raise ValueError(f"kind must be 'body', 'site' or None, got {kind!r}.")
        if isinstance(target, (bool, np.bool_)):
            raise TypeError("target must be a body/site name or an integer ID.")
        if isinstance(target, (int, np.integer)):
            idx = int(target)
            if kind == "site":
                if not 0 <= idx < m.nsite:
                    raise UnknownTargetError(f"Site ID {idx} out of range (model has {m.nsite} sites).")
                return d.site_xpos[idx], d.site_xmat[idx]
            if not 0 <= idx < m.nbody:
                raise UnknownTargetError(f"Body ID {idx} out of range (model has {m.nbody} bodies).")
            return d.xpos[idx], d.xmat[idx]
        if not isinstance(target, str):
            raise TypeError(f"target must be a str or int, got {type(target).__name__}.")

        if kind in (None, "body"):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, target)
            if bid >= 0:
                return d.xpos[bid], d.xmat[bid]
        if kind in (None, "site"):
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, target)
            if sid >= 0:
                return d.site_xpos[sid], d.site_xmat[sid]

        names = []
        if kind in (None, "body"):
            names += [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
        if kind in (None, "site"):
            names += [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)]
        names = [n for n in names if n]
        close = difflib.get_close_matches(target, names, n=3)
        what = {None: "body or site", "body": "body", "site": "site"}[kind]
        hint = f" Did you mean: {', '.join(close)}?" if close else f" Available: {', '.join(names[:10]) or '(none)'}."
        raise UnknownTargetError(f"No {what} named {target!r} in the model.{hint}")


def _positive(value: float, name: str) -> float:
    v = float(value)
    if not np.isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be a positive number, got {value!r}.")
    return v
