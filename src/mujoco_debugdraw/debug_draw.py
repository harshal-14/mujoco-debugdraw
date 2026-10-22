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
from .transforms import as_points, as_rotation, as_vec3

TEXT_POSITIONS = {
    "top_left": mujoco.mjtGridPos.mjGRID_TOPLEFT,
    "top_right": mujoco.mjtGridPos.mjGRID_TOPRIGHT,
    "bottom_left": mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
    "bottom_right": mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
}

_GEOM_TYPES = {
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "line": mujoco.mjtGeom.mjGEOM_LINE,
    "arrow": mujoco.mjtGeom.mjGEOM_ARROW,
}

# Object kinds that names and IDs can refer to: (mjtObj, MjModel count field, plural).
_OBJ_TYPES = {
    "body": (mujoco.mjtObj.mjOBJ_BODY, "nbody", "bodies"),
    "site": (mujoco.mjtObj.mjOBJ_SITE, "nsite", "sites"),
    "joint": (mujoco.mjtObj.mjOBJ_JOINT, "njnt", "joints"),
}

_JOINT_TYPE_NAMES = {
    int(mujoco.mjtJoint.mjJNT_FREE): "free",
    int(mujoco.mjtJoint.mjJNT_BALL): "ball",
    int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
    int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
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

    Every drawing method takes ``key`` (a repeated key replaces the old
    primitive) and ``ttl_frames`` (remove the primitive automatically after
    that many ``sync()`` calls), so a loop can also skip ``clear()``.

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
        ttl_frames: Optional[int] = None,
    ) -> None:
        """Draw a sphere marker of radius ``size`` (m) at a world-space position."""
        self._add(
            Primitive(
                "point",
                key,
                parse_color(color),
                {"position": as_vec3(position, "position"), "size": _positive(size, "size"), "label": label},
                ttl=_ttl(ttl_frames),
            )
        )

    def line(
        self,
        points: Any,
        color: ColorLike = "cyan",
        width: float = 2.0,
        key: Optional[str] = None,
        ttl_frames: Optional[int] = None,
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
                ttl=_ttl(ttl_frames),
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
        ttl_frames: Optional[int] = None,
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
                ttl=_ttl(ttl_frames),
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
        ttl_frames: Optional[int] = None,
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
                    "width": _frame_width(width, scale),
                    "label": label,
                },
                ttl=_ttl(ttl_frames),
            )
        )

    def frame_at(
        self,
        position: Any,
        rotation: Any,
        scale: float = 0.10,
        width: Optional[float] = None,
        alpha: float = 1.0,
        label: Optional[str] = None,
        key: Optional[str] = None,
        ttl_frames: Optional[int] = None,
    ) -> None:
        """Draw X/Y/Z axes (red/green/blue) at any world pose.

        ``rotation`` is a ``(3, 3)`` matrix whose columns are the local axes
        in world coordinates, or a row-major ``(9,)`` array like
        ``data.xmat[i]``. ``alpha`` in ``(0, 1]`` fades all three axes, e.g.
        to show a desired orientation next to the actual one.
        """
        scale = _positive(scale, "scale")
        alpha = float(alpha)
        if not 0.0 < alpha <= 1.0:  # also rejects NaN
            raise ValueError(f"alpha must be in (0, 1], got {alpha!r}.")
        self._add(
            Primitive(
                "frame",
                key,
                (1.0, 1.0, 1.0, 1.0),
                {
                    "position": as_vec3(position, "position"),
                    "xmat": as_rotation(rotation, "rotation").flatten(),
                    "scale": scale,
                    "width": _frame_width(width, scale),
                    "label": label,
                    "alpha": alpha,
                },
                ttl=_ttl(ttl_frames),
            )
        )

    def box(
        self,
        center: Any,
        half_size: Any,
        rotation: Any = None,
        color: ColorLike = "orange",
        label: Optional[str] = None,
        key: Optional[str] = None,
        ttl_frames: Optional[int] = None,
    ) -> None:
        """Draw a solid box with half-extents ``half_size`` (m) along its local axes.

        ``rotation`` follows the same convention as :meth:`frame_at`; ``None``
        aligns the box with the world axes. Use a color with alpha below 1 to
        see through it.
        """
        half_size = as_vec3(half_size, "half_size")
        if np.any(half_size <= 0):
            raise ValueError(f"half_size must be > 0 on every axis, got {half_size.tolist()}.")
        self._add(
            Primitive(
                "box",
                key,
                parse_color(color),
                {
                    "center": as_vec3(center, "center"),
                    "half_size": half_size,
                    "rotation": np.eye(3) if rotation is None else as_rotation(rotation, "rotation"),
                    "label": label,
                },
                ttl=_ttl(ttl_frames),
            )
        )

    def body_com(
        self,
        body: Target,
        subtree: bool = False,
        color: ColorLike = "white",
        size: float = 0.02,
        label: Optional[str] = None,
        key: Optional[str] = None,
        ttl_frames: Optional[int] = None,
    ) -> None:
        """Mark a body's center of mass with a sphere of radius ``size`` (m).

        ``subtree=False`` uses ``data.xipos`` (that body alone).
        ``subtree=True`` uses ``data.subtree_com`` (the body and all its
        descendants), so ``body_com(0, subtree=True)`` is the whole model's COM.
        """
        size = _positive(size, "size")
        _, bid = self._resolve_id(body, ("body",), "body")
        com = self.data.subtree_com if subtree else self.data.xipos
        self._add(
            Primitive(
                "point",
                key,
                parse_color(color),
                {"position": com[bid].copy(), "size": size, "label": label},
                ttl=_ttl(ttl_frames),
            )
        )

    def joint_axis(
        self,
        joint: Target,
        scale: float = 0.10,
        color: ColorLike = "orange",
        width: float = 0.005,
        label: Optional[str] = None,
        key: Optional[str] = None,
        ttl_frames: Optional[int] = None,
    ) -> None:
        """Draw a hinge or slide joint's axis as an arrow of length ``scale`` (m).

        The arrow starts at the joint anchor (``data.xanchor``) and points
        along ``data.xaxis``. Ball and free joints have no single axis and
        raise ``ValueError``.
        """
        scale = _positive(scale, "scale")
        _, jid = self._resolve_id(joint, ("joint",), "joint")
        jtype = _JOINT_TYPE_NAMES.get(int(self.model.jnt_type[jid]), "unknown")
        if jtype not in ("hinge", "slide"):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"#{jid}"
            raise ValueError(
                f"Joint {name!r} is a {jtype} joint, which has no single axis. "
                "joint_axis() supports hinge and slide joints."
            )
        self._add(
            Primitive(
                "arrow",
                key,
                parse_color(color),
                {
                    "origin": self.data.xanchor[jid].copy(),
                    "vector": self.data.xaxis[jid].copy(),
                    "scale": scale,
                    "width": _positive(width, "width"),
                    "label": label,
                },
                ttl=_ttl(ttl_frames),
            )
        )

    def contacts(
        self,
        body: Optional[Target] = None,
        normals: bool = True,
        forces: bool = True,
        normal_scale: float = 0.05,
        force_scale: float = 0.01,
        normal_color: ColorLike = "red",
        force_color: ColorLike = "yellow",
        width: float = 0.003,
        key: Optional[str] = None,
        ttl_frames: Optional[int] = None,
    ) -> None:
        """Draw contact normals and contact forces for the active contacts.

        Each contact gets a normal arrow of length ``normal_scale`` (m) and a
        force arrow scaled by ``force_scale`` (m per N), both starting at the
        contact point. Only the linear force is drawn; contacts that are not
        in the solver get a normal but no force.

        With ``body=None`` the arrows show the force acting on the contact's
        second geom (``geom[1]``). With ``body`` set, only contacts touching
        that body are drawn, and every arrow shows the force acting **on
        that body**.

        All arrows from one call are a single primitive, so ``key`` and
        ``ttl_frames`` apply to the whole set.
        """
        if not normals and not forces:
            raise ValueError("normals and forces are both False, so nothing would be drawn.")
        normal_scale = _positive(normal_scale, "normal_scale")
        force_scale = _positive(force_scale, "force_scale")
        normal_rgba, force_rgba = parse_color(normal_color), parse_color(force_color)
        width = _positive(width, "width")
        ttl = _ttl(ttl_frames)
        bid = None if body is None else self._resolve_id(body, ("body",), "body")[1]

        m, d = self.model, self.data
        arrows = []
        wrench = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            sign = 1.0  # +1: force on geom[1]; -1: force on geom[0]
            if bid is not None:
                g0, g1 = (int(g) for g in c.geom)  # -1 marks a flex side
                if g1 >= 0 and m.geom_bodyid[g1] == bid:
                    sign = 1.0
                elif g0 >= 0 and m.geom_bodyid[g0] == bid:
                    sign = -1.0
                else:
                    continue
            # contact.frame has the axes in its ROWS (data.xmat has them in
            # columns). Row 0 is the normal, pointing from geom[0] to geom[1].
            frame = np.array(c.frame).reshape(3, 3)
            pos = np.array(c.pos)
            if normals:
                arrows.append((pos, pos + sign * normal_scale * frame[0], normal_rgba))
            if forces and c.efc_address >= 0:
                mujoco.mj_contactForce(m, d, i, wrench)
                # wrench[:3] is in the contact frame; frame.T takes it to world.
                force = sign * (frame.T @ wrench[:3])
                arrows.append((pos, pos + force_scale * force, force_rgba))

        self._add(Primitive("contacts", key, (1.0, 1.0, 1.0, 1.0), {"arrows": arrows, "width": width}, ttl=ttl))

    def text(
        self,
        message: str,
        position: str = "top_left",
        key: Optional[str] = None,
        ttl_frames: Optional[int] = None,
    ) -> None:
        """Show status text in a screen corner of the passive viewer.

        ``position`` is one of ``top_left``, ``top_right``, ``bottom_left``,
        ``bottom_right``. Multiple messages in the same corner are stacked.
        """
        if position not in TEXT_POSITIONS:
            raise ValueError(
                f"position must be one of {sorted(TEXT_POSITIONS)}, got {position!r}."
            )
        self._add(
            Primitive(
                "text", key, (1, 1, 1, 1), {"message": str(message), "position": position},
                ttl=_ttl(ttl_frames),
            )
        )

    # --------------------------------------------------------------- lifecycle

    def clear(self) -> None:
        """Forget all queued primitives. Never touches the model or sim state."""
        self._prims.clear()
        self._geoms_queued = 0
        self.dropped = 0

    def remove(self, key: str) -> bool:
        """Remove the primitive stored under ``key``.

        Returns ``True`` if one was removed and ``False`` otherwise. Never raises.
        """
        try:
            prim = self._prims.pop(key, None)
        except TypeError:  # unhashable key
            return False
        if prim is None:
            return False
        self._geoms_queued -= geom_count(prim)
        return True

    def sync(self) -> None:
        """Write the queued primitives into the viewer's scene.

        Passive viewer: call once per frame, before ``viewer.sync()``.
        ``mujoco.Renderer``: call after ``renderer.update_scene()`` and
        before ``renderer.render()``.

        Afterwards, primitives drawn with ``ttl_frames`` have one frame less
        to live, and those that reach zero are removed.
        """
        specs = self.geom_specs()
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
        self._expire()

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

    def _expire(self) -> None:
        for key, prim in list(self._prims.items()):
            if prim.ttl is None:
                continue
            prim.ttl -= 1
            if prim.ttl <= 0:
                del self._prims[key]
                self._geoms_queued -= geom_count(prim)

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
            if spec.kind in ("sphere", "box"):
                if spec.kind == "sphere":
                    size = np.array([spec.size, 0.0, 0.0])
                else:
                    size = np.asarray(spec.size, dtype=np.float64)
                mat = _IDENTITY_MAT if spec.mat is None else np.asarray(spec.mat, dtype=np.float64)
                mujoco.mjv_initGeom(g, gtype, size, spec.start, mat, rgba)
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
        if kind not in (None, "body", "site"):
            raise ValueError(f"kind must be 'body', 'site' or None, got {kind!r}.")
        kind, idx = self._resolve_id(target, ("body", "site") if kind is None else (kind,))
        d = self.data
        if kind == "site":
            return d.site_xpos[idx], d.site_xmat[idx]
        return d.xpos[idx], d.xmat[idx]

    def _resolve_id(self, target: Target, kinds: tuple[str, ...], arg: str = "target") -> tuple[str, int]:
        """Look up a name or integer ID and return ``(kind, id)``.

        A name is tried against each of ``kinds`` in order. An integer is an
        ID of ``kinds[0]``. Unknown names raise :class:`UnknownTargetError`
        with close-match suggestions.
        """
        m = self.model
        if isinstance(target, (bool, np.bool_)):
            raise TypeError(f"{arg} must be a {'/'.join(kinds)} name or an integer ID.")
        if isinstance(target, (int, np.integer)):
            kind, idx = kinds[0], int(target)
            _, count_attr, plural = _OBJ_TYPES[kind]
            count = getattr(m, count_attr)
            if not 0 <= idx < count:
                raise UnknownTargetError(
                    f"{kind.capitalize()} ID {idx} out of range (model has {count} {plural})."
                )
            return kind, idx
        if not isinstance(target, str):
            raise TypeError(f"{arg} must be a str or int, got {type(target).__name__}.")

        for kind in kinds:
            idx = mujoco.mj_name2id(m, _OBJ_TYPES[kind][0], target)
            if idx >= 0:
                return kind, idx

        names = []
        for kind in kinds:
            obj, count_attr, _ = _OBJ_TYPES[kind]
            names += [mujoco.mj_id2name(m, obj, i) for i in range(getattr(m, count_attr))]
        names = [n for n in names if n]
        close = difflib.get_close_matches(target, names, n=3)
        hint = f" Did you mean: {', '.join(close)}?" if close else f" Available: {', '.join(names[:10]) or '(none)'}."
        raise UnknownTargetError(f"No {' or '.join(kinds)} named {target!r} in the model.{hint}")


def _positive(value: float, name: str) -> float:
    v = float(value)
    if not np.isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be a positive number, got {value!r}.")
    return v


def _frame_width(width: Optional[float], scale: float) -> float:
    return max(scale * 0.05, 1e-4) if width is None else _positive(width, "width")


def _ttl(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"ttl_frames must be None or a positive integer, got {value!r}.")
    return int(value)
