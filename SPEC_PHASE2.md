# MuJoCo DebugDraw — Phase 2 Specification

**Status:** Implemented in v0.2.0 (manual criterion 10 still to be checked in the viewer)  
**Builds on:** Phase 1 ([SPEC.md](SPEC.md)) as shipped in v0.1.0  
**Evidence used:** the current code in `src/mujoco_debugdraw/`, `examples/manipulator_demo.py`, the contact spike `src/mujoco_debugdraw/spike_contact_frame.py`, and the MuJoCo 3.15 headers (`mjdata.h`, `mjmodel.h`, `mujoco.h`) and `mujoco/viewer.py`.

## 1. Scope decisions

The Phase 2 roadmap in SPEC.md §10 is resolved as follows:

| Roadmap item | Decision | Reason |
|---|---|---|
| `sphere()` | **Dropped** | `point()` already draws a sphere of any radius and any RGBA (including alpha). A second method would duplicate it. |
| `box()` | **In** | No Phase 1 primitive can draw an oriented volume (workspace limits, bounding boxes, keep-out zones). |
| Contact normal and force arrows | **In**, as `contacts()` | The spike shows that this works with public APIs. It also shows that the contact-frame convention is easy to get wrong, so the library should handle it. |
| `body_com()` | **In** | MuJoCo already computes body and subtree COMs (`data.xipos`, `data.subtree_com`). |
| Joint-axis visualization | **In**, as `joint_axis()` | MuJoCo already computes world anchors and axes (`data.xanchor`, `data.xaxis`). |
| Better screen-space text | **Dropped** | Phase 1 `text()` already uses the passive viewer's real overlay (`Handle.set_texts`). No concrete gap is known. `set_texts` also supports a font scale and a right-hand text column, which can be exposed later if someone needs them. |
| `ttl_frames` | **In** | — |
| Persistent keyed objects | **Mostly done; finish it** | Primitives already persist until `clear()`, and a repeated `key` already replaces the old primitive. Missing pieces: deleting one key, and documenting a usage pattern that doesn't call `clear()`. |
| *(new)* `frame_at()` | **In** | `frame()` only accepts a body or site. The manipulator demo (`manipulator_demo.py:204-211`) and the contact spike both draw a frame from a raw rotation matrix by making three `arrow()` calls. |

## 2. Rules carried over from Phase 1

These apply to every new method unless a section says otherwise:

- Everything drawn is visual-only. Nothing is added to `MjModel`, and `model` and `data` are only read.
- Anything that comes from `data` is **sampled when the method is called**. Call the method after `mj_step`/`mj_forward`. `sync()` does not re-read `data`.
- `draw.sync()` is called before `viewer.sync()`. With a `mujoco.Renderer`, call it after `update_scene()`.
- `max_geoms` and `CapacityWarning` work exactly as they do in Phase 1.
- A zero-length arrow draws nothing and does not raise an error.
- Bad shapes, non-finite numbers and non-positive sizes raise a `ValueError` that names the argument. Unknown names raise `UnknownTargetError` with close-match suggestions.
- Every new drawing method accepts `key=None` and `ttl_frames=None` (§4).

## 3. New API

### 3.1 `box`

```python
draw.box(center, half_size, rotation=None, color="orange", label=None, key=None, ttl_frames=None)
```

| Parameter | Type | Description |
|---|---|---|
| `center` | `(3,)` | World-space center in meters |
| `half_size` | `(3,)` | Half-extents along the box's local X/Y/Z axes, in meters, all > 0. MuJoCo uses half-sizes too. |
| `rotation` | `(3, 3)`, `(9,)` or `None` | Orientation. See the rotation convention below. `None` means aligned with the world axes. |

- Draws one solid `mjGEOM_BOX`. To see through it, give `color` an alpha below 1.
- Uses 1 geom.

**Rotation convention** (shared by `box` and `frame_at`): a `(3, 3)` matrix's **columns** are the local X/Y/Z axes in world coordinates. This is the same convention as `data.xmat[i].reshape(3, 3)`. A flat `(9,)` input is read row-major, as `data.xmat[i]` is. The input must be finite and a proper rotation (`‖RᵀR − I‖ < 1e-5` and `det R > 0`). Anything else raises `ValueError`.

### 3.2 `frame_at`

```python
draw.frame_at(position, rotation, scale=0.10, width=None, alpha=1.0, label=None, key=None, ttl_frames=None)
```

- Draws the same red/green/blue axes as `frame()`, but at any world pose instead of a body or site.
- `rotation` follows the convention in §3.1. `width` defaults to the same value `frame()` uses.
- `alpha` in `(0, 1]` fades all three axes. The demo uses this to draw a "desired orientation" that looks different from the actual frame.
- Uses 3 geoms.
- For a body, `frame_at(data.xpos[i], data.xmat[i])` must produce the same geoms as `frame(i)`.

### 3.3 `body_com`

```python
draw.body_com(body, subtree=False, color="white", size=0.02, label=None, key=None, ttl_frames=None)
```

- `body` is a body name or body ID. Name lookup and errors work the same way as in `frame()`.
- `subtree=False` draws a sphere at `data.xipos[body]`, the center of mass of that body alone.
- `subtree=True` draws at `data.subtree_com[body]`, the center of mass of that body plus all its descendants. `body_com("world", subtree=True)` (body 0) therefore gives the COM of the whole model.
- Uses 1 geom.

### 3.4 `joint_axis`

```python
draw.joint_axis(joint, scale=0.10, color="orange", width=0.005, label=None, key=None, ttl_frames=None)
```

- `joint` is a joint name or joint ID. An unknown name raises `UnknownTargetError` with suggestions.
- For **hinge** and **slide** joints, draws an arrow from `data.xanchor[j]` to `data.xanchor[j] + scale * data.xaxis[j]`.
- **Ball** and **free** joints have no single axis, so they raise `ValueError` and the message names the joint type.
- Uses 1 geom.

### 3.5 `contacts`

```python
draw.contacts(body=None, normals=True, forces=True, normal_scale=0.05, force_scale=0.01,
              normal_color="red", force_color="yellow", width=0.003, key=None, ttl_frames=None)
```

For each active contact `i` in `data.contact[:data.ncon]`:

- **Normal:** an arrow at `c.pos` of length `normal_scale` (m) along the contact normal.
- **Force:** an arrow at `c.pos` along `R.T @ f[:3]`, scaled by `force_scale` (m per N). Here `R = c.frame.reshape(3, 3)` and `f` comes from `mj_contactForce(model, data, i, f)`.
- **Frame convention, which the library must hide:** `c.frame` stores the axes in its **rows**, and row 0 is the normal, pointing from `geom[0]` to `geom[1]` (`mjdata.h`). `data.xmat` stores them in **columns**. `mj_contactForce` returns the force in the contact frame, so `R.T` converts it to world coordinates.
- **Direction:** with `body=None`, the arrows show the force acting on `geom[1]`. When `body` is given, only contacts where `geom[0]` or `geom[1]` belongs to that body are drawn (checked with `model.geom_bodyid`). Each arrow is flipped when needed so that it shows the force acting **on that body**. A flex side (`geom[k] == -1`) never matches `body`.
- If a contact is not in the solver (`c.efc_address < 0`), its normal is drawn but its force is not.
- Only the linear force `f[:3]` is drawn. The torque part `f[3:6]` is ignored.
- `normals=False` or `forces=False` hides that arrow type. Setting both to `False` raises `ValueError`.
- All arrows from one call form **one primitive**, so `key` and `ttl_frames` apply to the whole set. The call uses up to 2 geoms per contact. With the default `max_geoms=500`, that is about 250 contacts before `CapacityWarning`.
- To show the full contact frame instead of just the normal, call `frame_at(c.pos, c.frame.reshape(3, 3).T)`. Document this in the README.

### 3.6 Lifetime: `ttl_frames`, `remove`, and running without `clear()`

**`ttl_frames`** (all drawing methods, Phase 1 methods included):

- `None` (the default) keeps the Phase 1 behavior: the primitive lives until `clear()`, until `remove(key)`, or until it is replaced by the same `key`.
- A positive integer `N` means the primitive is written by the next `N` calls to `sync()` and then removed automatically, with its geoms freed from the `max_geoms` count.
- A "frame" means one `sync()` call, not a span of seconds.
- Drawing again with the same `key` replaces the primitive and restarts its TTL.
- `clear()` still removes everything, whatever TTL is left.
- `0`, negative numbers, non-integers and `bool` raise `ValueError`.

**`remove`:**

```python
draw.remove(key) -> bool
```

Removes the primitive stored under `key` and frees its geoms. Returns `True` if something was removed and `False` otherwise. It never raises.

**Running without `clear()`** (document in the README): draw static things once, and update per-frame things in one of two ways:

```python
draw.line(plan, color="cyan", key="plan")              # drawn once, stays

while viewer.is_running():
    mujoco.mj_step(model, data)
    draw.frame("end_effector", key="ee")               # keyed: replaced each frame
    draw.contacts(ttl_frames=1)                        # unkeyed: expires after one sync
    draw.sync()
    viewer.sync()
```

If unkeyed primitives have no `ttl_frames` and `clear()` is never called, they pile up until `max_geoms` is reached. Phase 1 already behaves this way, and it stays that way.

## 4. Implementation notes

- **`primitives.py`:** `GeomSpec` gains a `"box"` kind, a `size` field that can also hold three values, and an optional `mat` field. `Primitive` gains `ttl: Optional[int]`. New primitive kinds are `"box"` and `"contacts"`. `frame_at` reuses the existing `"frame"` kind with its own `xmat` and per-axis alpha.
- **`transforms.py`:** add one `as_rotation(value, name)` validator, used by `box` and `frame_at`. All validators (`as_vec3`, `as_points`, `as_rotation`) copy their input. In Phase 1, `np.asarray` returned a view, so `point(data.xpos[i])` tracked `data` until `sync()`. That broke the §2 sampling rule, and it matters more now that primitives can outlive a frame.
- **`debug_draw.py`:**
  - Split the name→ID lookup out of `_resolve_pose` into one helper that handles bodies, sites and joints and keeps the `difflib` suggestions. `frame`, `body_com`, `joint_axis` and `contacts(body=...)` all use it.
  - `_write` handles `mjGEOM_BOX` by passing `size` and `mat` to `mjv_initGeom`.
  - `frame()` and `frame_at()` share the default-width rule, and an explicit non-positive `width` now raises `ValueError` (§2).
  - `sync()` lowers each TTL by one after writing and drops the primitives that reach 0.
- **`exceptions.py`:** broaden the `UnknownTargetError` docstring so it covers joints and is not limited to `frame()`.
- **The spike:** replace `src/mujoco_debugdraw/spike_contact_frame.py` with `examples/contact_forces.py`, which uses `contacts()`. Spike code must not stay inside the installed package.

## 5. Overlap with MuJoCo's built-in viewer flags

MuJoCo's viewer can already draw contact points, contact forces, COMs and joints through the visualization flags `mjVIS_CONTACTPOINT`, `mjVIS_CONTACTFORCE`, `mjVIS_COM` and `mjVIS_JOINT`. Those flags are all-or-nothing: they draw the feature for **every** object in the model, in fixed colors. The Phase 2 helpers exist for targeted debugging: one body's contacts, one joint, or one COM, with your own color, label, scale and lifetime. The README must say this so users can pick the right tool.

## 6. Non-goals for Phase 2

- Contact torques, friction-cone drawing, and anything specific to flex contacts beyond what §3.5 says.
- Wireframe boxes, inertia boxes, joint-range arcs.
- Lifetimes measured in seconds.
- Primitives that re-read `data` at `sync()` time ("live" frames). Call the method again every frame instead.
- Any change to `text()`.

## 7. Acceptance criteria

Phase 2 is done when all of these hold, and every Phase 1 test still passes.

**Unit and scene tests (no display needed):**

1. `box`: the `GeomSpec` holds the given center, half-sizes and rotation. It rejects non-positive half-sizes, a wrong rotation shape, a non-orthonormal matrix, and a reflection (`det = -1`).
2. `frame_at`: the axis tips equal `position + scale * R[:, i]`. Given `data.xpos[i]` and `data.xmat[i]`, it produces the same geoms as `frame(i)`.
3. `body_com`: matches `data.xipos[id]`. With `subtree=True`, it matches `data.subtree_com[id]`. For body 0 it equals `Σ mᵢ·xiposᵢ / Σ mᵢ`, computed from `model.body_mass`.
4. `joint_axis`: the endpoint equals `xanchor + scale * xaxis` for both a hinge and a slide. A ball or free joint raises `ValueError`. A misspelled name raises `UnknownTargetError` with a suggestion.
5. `contacts`, as a physics check: stack box B (mass `m_B`) on box A (mass `m_A`) on a plane and let them settle (about 2 s of sim time). With `body="A"` and with `body="B"`, the drawn force vectors, divided by `force_scale`, each sum to about `(0, 0, m·g)` for that box's own mass, within 2%. For A this only works if the A–B contact is flipped correctly (it pushes A down), so the test covers both geom orders. Every normal arrow has length `normal_scale`. The number of geoms is at most `2 * data.ncon`.
6. `contacts(body=...)`: contacts that don't touch the body are left out.
7. `ttl_frames=N`: the primitive appears in exactly `N` calls to `sync()`, and `num_geoms` goes down when it expires. Redrawing with the same `key` restarts the TTL. `clear()` removes it early. The invalid values from §3.6 raise `ValueError`.
8. `remove`: returns `True` and then `False` for the same key, and frees capacity under `max_geoms`.
9. **No side effects:** extend the existing "bit-identical `qpos`/`qvel`" test so that it also calls `box`, `frame_at`, `body_com`, `joint_axis` and `contacts` every step.

**Manual:**

10. `examples/contact_forces.py` runs in the passive viewer. The force arrows point up into the resting box and get shorter or longer as expected when the box is pushed.
11. The README API table lists every new method, the rotation convention, the "running without `clear()`" pattern, and the note about overlap with the built-in flags.

## 8. Work order

1. Refactor only, with no behavior change: the shared ID lookup and `as_rotation`. Phase 1 tests must stay green.
2. `box`, `frame_at`.
3. `body_com`, `joint_axis`.
4. `ttl_frames`, `remove`.
5. `contacts`, and `examples/contact_forces.py` (delete the spike).
6. README and version bump.
