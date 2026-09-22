# MuJoCo DebugDraw — Phase 1 Specification

## 1. Project summary

**Project name:** `mujoco-debugdraw`  
**Python import:** `mujoco_debugdraw`  
**Repository suggestion:** `harshalbhat/mujoco-debugdraw`  
**License:** MIT  
**Target duration:** 5–7 days  
**Primary language:** Python  
**Primary dependency:** Official `mujoco` Python package  
**Initial target:** MuJoCo Python passive viewer only

### Problem

MuJoCo roboticists often need to inspect targets, coordinate frames, vectors, paths, contacts, and controller state while a simulation runs. The standard Python passive viewer does not offer a simple, documented high-level interface for runtime debug drawing and overlays. Users currently resort to editing MJCF, manually manipulating low-level visualization structures, launching custom viewer code, or printing numerical data.

The project provides a small Python API for drawing debug primitives inside a running MuJoCo viewer:

- 3D points
- Lines and paths
- Direction arrows
- Coordinate frames
- Screen-space text
- Optional contact-force vectors later

This is a focused wrapper around MuJoCo’s visualization facilities, not a simulator, editor, dashboard, or robotics middleware.

### Core pitch

> Draw targets, coordinate frames, force vectors, paths, and labels in the MuJoCo Python viewer—without editing MJCF or writing OpenGL.

## 2. Goals

### Phase 1 goals

1. Draw a 3D point at an arbitrary world-space coordinate.
2. Draw a line segment or multi-point path.
3. Draw an arrow with a configurable origin, vector, color, and scale.
4. Draw RGB coordinate axes for a MuJoCo body or site.
5. Draw small status text in a simple heads-up display.
6. Work with `mujoco.viewer.launch_passive`.
7. Require no changes to the user’s MJCF/XML model.
8. Require no ROS 2, Isaac Sim, GPU-specific setup, external rendering framework, web server, or GUI framework.
9. Be easy to install and demonstrate in fewer than 30 lines of user code.
10. Ship one polished runnable example with a GIF/video.

### Non-goals

Do **not** include these in Phase 1:

- Isaac Sim, Gazebo, Webots, Unity, Unreal, or browser support.
- ROS 2 integration.
- rosbag/MCAP reading or replay.
- A robot-model validator, importer, converter, or exporter.
- Live plots, chart panes, sliders, menus, scene editors, or a React web UI.
- Persistent debug geometry saved into MJCF.
- Point-cloud visualization.
- Full contact-analysis tooling.
- Multi-user collaboration or cloud features.
- Guaranteed support for every MuJoCo release, OS, renderer, or viewer implementation.
- A replacement for MuJoCo’s native viewer or official visualization API.

## 3. User stories

### Robotics-control developer

> I am tuning an end-effector controller. I want to see the desired target, actual end-effector pose, and position-error vector in the MuJoCo window so I can spot a coordinate-frame mistake immediately.

### Motion-planning developer

> I have a generated 3D path. I want to render the planned path, current robot frame, and goal position without modifying model XML.

### Contact/debugging developer

> I want to display an arrow for a force or velocity vector at a known position so I can tell whether it points in the expected direction.

### RL/research user

> I want to add a small `Reward: 0.62` or `Episode: 41` label while my policy runs, without creating a custom UI.

## 4. Public API

The package must expose one primary class:

```python
from mujoco_debugdraw import DebugDraw
```

### Construction

```python
draw = DebugDraw(
    model,
    data,
    viewer,
    max_geoms=500,
)
```

Parameters:

| Parameter | Type | Required | Description |
|---|---:|---:|---|
| `model` | `mujoco.MjModel` | Yes | MuJoCo model |
| `data` | `mujoco.MjData` | Yes | MuJoCo runtime state |
| `viewer` | Passive viewer object | Yes | Object returned by `mujoco.viewer.launch_passive()` |
| `max_geoms` | `int` | No | Maximum number of temporary debug geometries; default 500 |

Requirements:

- Raise a helpful error if MuJoCo is missing or the provided model/data/viewer are incompatible.
- Do not mutate the user’s MJCF/XML source files.
- Do not alter simulation dynamics, contacts, actuators, or state except for reading data.

### `point`

```python
draw.point(
    position,
    color="red",
    size=0.02,
    label=None,
    key=None,
)
```

Draw a filled 3D marker at a world-space position.

Parameters:

| Parameter | Type | Default | Description |
|---|---:|---:|---|
| `position` | array-like shape `(3,)` | Required | World-space XYZ position in meters |
| `color` | string or RGBA | `"red"` | Named color, hex string, or normalized `[r, g, b, a]` |
| `size` | float | `0.02` | Radius or visual size in meters |
| `label` | `str \| None` | `None` | Optional text label near the point |
| `key` | `str \| None` | `None` | Optional stable ID for updating an existing primitive |

Examples:

```python
draw.point([0.45, 0.0, 0.30], color="lime", size=0.025)
draw.point(target_xyz, color="#00FF88", label="goal", key="goal")
```

### `line`

```python
draw.line(
    points,
    color="cyan",
    width=2.0,
    key=None,
)
```

Draw a connected sequence of line segments.

Parameters:

| Parameter | Type | Default | Description |
|---|---:|---:|---|
| `points` | array-like shape `(N, 3)` | Required | Two or more world-space points |
| `color` | string or RGBA | `"cyan"` | Line color |
| `width` | float | `2.0` | Rendered line width, subject to viewer behavior |
| `key` | `str \| None` | `None` | Optional stable ID |

Requirements:

- Reject fewer than two points with a useful `ValueError`.
- Support a simple planned path or trajectory.
- Use one visual segment per consecutive point pair.

Example:

```python
draw.line(planned_path_xyz, color="cyan", width=2.0, key="plan")
```

### `arrow`

```python
draw.arrow(
    origin,
    vector,
    color="yellow",
    scale=1.0,
    width=0.01,
    label=None,
    key=None,
)
```

Draw a directional vector from a world-space origin.

Parameters:

| Parameter | Type | Default | Description |
|---|---:|---:|---|
| `origin` | array-like shape `(3,)` | Required | Arrow tail in world coordinates |
| `vector` | array-like shape `(3,)` | Required | Direction and magnitude |
| `color` | string or RGBA | `"yellow"` | Arrow color |
| `scale` | float | `1.0` | Multiplier for vector length |
| `width` | float | `0.01` | Shaft radius/width in meters |
| `label` | `str \| None` | `None` | Optional label |
| `key` | `str \| None` | `None` | Optional stable ID |

Requirements:

- Draw a shaft and arrowhead.
- A zero-length vector must not crash.
- For zero vector, draw either a small point or nothing; document the chosen behavior.
- Vector direction must be preserved exactly after scaling.

Examples:

```python
draw.arrow(
    origin=ee_position,
    vector=desired_velocity,
    color="yellow",
    scale=0.20,
    label="v_des",
)
```

### `frame`

```python
draw.frame(
    target,
    scale=0.10,
    key=None,
    label=None,
)
```

Draw a standard coordinate frame:

- X axis: red
- Y axis: green
- Z axis: blue

`target` must support:

- Body name, for example `"end_effector"`.
- Site name, for example `"gripper_site"`.
- Integer body ID.
- Integer site ID only when explicitly specified through an optional future API; Phase 1 may accept names only if that keeps implementation simpler.

Parameters:

| Parameter | Type | Default | Description |
|---|---:|---:|---|
| `target` | `str` or `int` | Required | MuJoCo body or site identifier |
| `scale` | float | `0.10` | Axis length in meters |
| `key` | `str \| None` | `None` | Optional stable ID |
| `label` | `str \| None` | `None` | Optional label at frame origin |

Requirements:

- Resolve body/site names against `MjModel`.
- Read current world position and orientation from `MjData`.
- Give an actionable error if a name does not exist.
- Frame must update as the body/site moves.
- Axis orientation must match MuJoCo’s world-space orientation.

Example:

```python
draw.frame("end_effector", scale=0.12, label="EE")
```

### `text`

```python
draw.text(
    message,
    position="top_left",
    key=None,
)
```

Draw status text.

Allowed Phase 1 `position` values:

```python
"top_left"
"top_right"
"bottom_left"
"bottom_right"
```

Requirements:

- `message` may be a Python string.
- If true on-screen text overlay cannot be safely implemented through the selected public viewer path, Phase 1 may provide a clearly documented fallback:
  - Print updates in the terminal at a controlled rate, or
  - Render text as a 3D label near a fixed world-space anchor.
- The README must clearly say which behavior is implemented.
- Do not claim a HUD works unless it visibly works in the official passive viewer.

Example:

```python
draw.text(f"EE error: {position_error * 100:.1f} cm")
```

### `clear`

```python
draw.clear()
```

Remove all current debug primitives before the next render/update.

Requirements:

- Must not alter model geometry or simulation state.
- Must be safe to call each frame.
- Must remove primitives registered both with and without `key`.

### `sync`

```python
draw.sync()
```

Apply queued/current primitives to the viewer for the current simulation frame.

Expected loop:

```python
while viewer.is_running():
    mujoco.mj_step(model, data)

    draw.clear()
    draw.frame("end_effector", scale=0.12)
    draw.point(target_xyz, color="lime", label="target")
    draw.arrow(data.site_xpos[site_id], velocity, color="yellow")
    draw.text("Controller: TRACKING")

    draw.sync()
    viewer.sync()
```

Requirements:

- Document whether `draw.sync()` must be called before or after `viewer.sync()`.
- Choose one ordering and make every example use it consistently.
- Avoid unbounded geometry accumulation across frames.
- Enforce `max_geoms` with a readable error or warning.

## 5. Technical design

### Design principle

The package is an ergonomic wrapper around **temporary viewer-side visualization**. It must not add physics objects to the model, modify XML, or require the user to create `site`, `geom`, or `body` elements for debugging.

### Internal data model

Use a small internal representation:

```python
@dataclass
class Primitive:
    kind: str
    key: str | None
    color: tuple[float, float, float, float]
    payload: dict
```

Supported kinds:

```text
point
line
arrow
frame
text
```

`DebugDraw` stores either:

- A per-frame list of primitives, cleared with `clear()`, or
- A dictionary of keyed persistent primitives plus transient primitives.

For Phase 1, it is acceptable for `clear()` to reset everything and for the user to redraw each simulation frame.

### Rendering strategy

The implementation must first prove that it can render temporary geometry in the passive viewer without XML edits.

Preferred approach:

1. Use MuJoCo’s public visualization API and scene geometry allocation path.
2. Add temporary visualization geoms to the scene each frame.
3. Ensure debug geometry is reset/rebuilt every frame rather than permanently appended.
4. Use existing MuJoCo geometry types where possible:
   - Sphere or small geom for `point`.
   - Capsule/cylinder/line for `line`.
   - Cylinder/capsule plus cone/arrowhead for `arrow`.
   - Three arrows for `frame`.

### Important technical spike

Before building the package, complete this proof-of-concept:

```python
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Add one temporary visible marker at a fixed world coordinate
    # without modifying model XML.
    # Confirm it remains visible while stepping.
```

**Go/no-go rule:** If this cannot be implemented cleanly with stable public APIs in one day, revise scope. Do not build a fragile private-attribute hack as the public Phase 1 product.

MuJoCo users have asked for adding custom/debug geoms to the Python passive viewer, selectively displaying frames, and exposing higher-level overlay/UI controls. This project should use the least fragile supported route available. [github](https://github.com/google-deepmind/mujoco/issues/1023)

### Threading and viewer safety

MuJoCo’s passive viewer may run a rendering thread independently from the Python simulation loop.

Phase 1 rules:

- Treat `model` and `data` as read-only from debug-draw code.
- Do not write physics state.
- If viewer locking is required by the selected MuJoCo API path, use the viewer’s documented lock/context manager.
- Keep all temporary scene mutation synchronized with the viewer update cycle.
- Document any limitations, including unsupported multi-threaded patterns.

### Color parsing

Support:

```python
"red"
"green"
"blue"
"yellow"
"cyan"
"magenta"
"white"
"black"
"gray"
"orange"
"lime"
```

Also support:

```python
"#RRGGBB"
"#RRGGBBAA"
[r, g, b]
[r, g, b, a]
```

Normalized RGBA values must be in \([0, 1]\). Clamp or reject invalid values consistently; rejecting with a clear error is preferred.

## 6. Repository structure

```text
mujoco-debugdraw/
├── README.md
├── LICENSE
├── pyproject.toml
├── SPEC.md
├── src/
│   └── mujoco_debugdraw/
│       ├── __init__.py
│       ├── debug_draw.py
│       ├── primitives.py
│       ├── colors.py
│       ├── transforms.py
│       └── exceptions.py
├── examples/
│   ├── reaching_debug.py
│   ├── scene.xml
│   └── README.md
├── tests/
│   ├── test_colors.py
│   ├── test_geometry.py
│   ├── test_frames.py
│   ├── test_api_errors.py
│   └── test_smoke_mujoco.py
├── .github/
│   └── workflows/
│       └── test.yml
└── assets/
    └── demo.gif
```

## 7. Acceptance criteria

Phase 1 is complete only if every item below is true.

### Installation

```bash
pip install mujoco-debugdraw
```

- Installs successfully into a clean supported Python environment.
- Clearly documents supported Python and MuJoCo versions.
- Does not require ROS 2, Docker, CUDA, Isaac Sim, or a browser.

### Working demo

This example must work with no user-model changes:

```python
import mujoco
import mujoco.viewer

from mujoco_debugdraw import DebugDraw

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

target = [0.45, 0.00, 0.30]

with mujoco.viewer.launch_passive(model, data) as viewer:
    draw = DebugDraw(model, data, viewer)

    while viewer.is_running():
        mujoco.mj_step(model, data)

        draw.clear()
        draw.frame("end_effector", scale=0.12, label="EE")
        draw.point(target, color="lime", size=0.025, label="target")
        draw.line(
            [[0.20, -0.20, 0.05], [0.30, -0.10, 0.10], target],
            color="cyan",
        )
        draw.arrow(
            origin=data.site_xpos[0],
            vector=[0.0, 0.0, 0.10],
            color="yellow",
            label="desired velocity",
        )
        draw.text("MuJoCo DebugDraw active", position="top_left")

        draw.sync()
        viewer.sync()
```

The demo must visibly show:

- One green target point.
- One cyan multi-segment path.
- One yellow arrow.
- Red/green/blue coordinate axes attached to a moving body or site.
- Text or the explicitly documented Phase 1 fallback.
- No growing memory usage or primitive count while the simulation continues.

### Correctness checks

- `frame()` uses correct body/site world pose.
- X/Y/Z axes match the expected transformed local basis.
- Arrow endpoint equals:

\[
p_{\text{endpoint}} = p_{\text{origin}} + s \cdot v
\]

where \(s\) is `scale` and \(v\) is `vector`.

- `line()` connects each sequential pair of points.
- Invalid position/vector shapes raise clear `ValueError` messages.
- Unknown body/site names raise clear errors with available close matches if feasible.
- A zero vector does not crash.
- `clear()` does not modify simulation dynamics.
- The debug layer does not create model contacts or affect robot motion.

### Documentation

README must include:

1. A one-sentence project pitch.
2. An animated GIF or short video above the fold.
3. Installation command.
4. A copy-paste quickstart under 30 lines.
5. API table.
6. Supported versions and limitations.
7. A “Why this exists” section referencing debug frame/overlay/custom-geom workflow friction.
8. A clear Phase 1 roadmap and explicit non-goals.
9. Contribution instructions.
10. An issue template for bugs and feature requests.

## 8. Testing plan

### Unit tests

No OpenGL window required:

- Color parser accepts named colors, hex, RGB, RGBA.
- Color parser rejects malformed strings and values outside valid ranges.
- Point/vector shape validation.
- Arrow endpoint math.
- Axis basis transformation math.
- Frame target name resolution.
- Maximum geometry count behavior.
- `clear()` resets stored primitives.
- Keyed primitive behavior, if implemented.

### MuJoCo smoke test

Use a tiny XML fixture:

- One free body or hinged arm.
- One named body.
- One named site.

Tests:

1. Load `MjModel` and `MjData`.
2. Instantiate `DebugDraw`.
3. Add one primitive of each supported type.
4. Run `sync()` or equivalent scene-update path.
5. Verify no exception.
6. Verify expected internal primitive/geometry count.
7. Step 100 simulation frames.
8. Verify simulation state stays finite and dynamics are unchanged relative to a control run with no debug drawing.

If headless rendering is unreliable in GitHub Actions, unit-test transformations and primitive construction in CI, then document that manual viewer smoke testing is required locally for the release.

### Manual QA checklist

Before release:

- Linux test.
- macOS test if hardware/environment is available.
- A basic Windows test if practical.
- Confirm passive viewer opens.
- Confirm each primitive appears.
- Confirm `frame()` moves correctly with an articulated body.
- Confirm all primitive types disappear after `clear()`.
- Run the demo for at least five minutes and watch for increased memory/geometry count.
- Confirm the demo works from a clean clone using documented installation steps.

## 9. Milestones

### Day 1 — Technical spike

Deliverable:

- A local script proving one temporary sphere/point can be rendered in a passive viewer with no XML modification.

Decision:

- If stable public API path works, proceed.
- If not, reduce scope or change rendering target before writing the package.

### Day 2 — Primitive core

Deliver:

- `DebugDraw`
- Color parser
- `point()`
- `line()`
- Basic `clear()`
- Unit tests

### Day 3 — Arrows and frames

Deliver:

- `arrow()`
- `frame()`
- Name lookup
- Coordinate-transform tests
- One moving-arm demonstration

### Day 4 — Sync and ergonomics

Deliver:

- Stable frame update behavior
- `max_geoms`
- Helpful errors
- Public type hints/docstrings
- Begin polished demo

### Day 5 — Demo and docs

Deliver:

- `reaching_debug.py`
- GIF/video
- README quickstart
- API documentation
- GitHub Actions unit-test workflow

### Days 6–7 — Hardening

Deliver:

- Fix real environment/version issues.
- Test clean installation.
- Add examples and limitations.
- Prepare PyPI package.
- Publish the repository.

## 10. Future roadmap

These are **not** Phase 1 requirements.

### Phase 2

- `sphere()` and `box()` primitives.
- Contact normal and contact-force arrows.
- `body_com()` center-of-mass marker.
- Joint-axis visualization.
- Better fixed screen-space text overlays.
- Marker lifetime controls such as `ttl_frames`.
- Persistent keyed objects without calling `clear()` every frame.

### Phase 3

- Camera/image overlay support.
- Time-series plots.
- Optional keyboard toggle for debug layers.
- ROS 2 helper package.
- Path/trajectory visualization from NumPy or ROS messages.
- Isaac Sim equivalent package or adapter, only after MuJoCo API is stable.

## 11. LLM implementation handoff

Use this prompt after giving an LLM the repository and this specification:

```text
Implement Phase 1 of the project described in SPEC.md.

Constraints:
- Use Python and the official `mujoco` package only for MuJoCo integration.
- Do not add ROS 2, Isaac Sim, web UI, Docker, database, cloud, or model-editing features.
- Do not modify a user’s MJCF/XML file or simulation state.
- Do not use undocumented/private MuJoCo viewer attributes as the primary implementation strategy.
- Start with the Day 1 technical spike: prove that a temporary debug point can be drawn in `mujoco.viewer.launch_passive` without editing XML.
- If this cannot be done through a stable public API, stop and explain the limitation with a minimal reproducible script. Do not fabricate an API or build a fragile workaround.
- Once the spike succeeds, implement the smallest working package supporting:
  `point`, `line`, `arrow`, `frame`, `text`, `clear`, and `sync`.
- Keep the codebase small and typed. Add unit tests for coordinate math, input validation, and colors.
- Add one runnable example and a concise README.
- At each milestone, run tests and report exactly what works, what does not, and any API/version assumptions.
```

## 12. Success definition

Phase 1 succeeds if a robotics developer can clone/install the package and, in fewer than five minutes:

1. Launch a normal MuJoCo Python passive viewer.
2. Add a target marker.
3. Draw an end-effector/body coordinate frame.
4. Draw a planned path and a command/force vector.
5. See those objects update while the simulation runs.
6. Do all of this without editing MJCF or writing low-level MuJoCo visualization/OpenGL code.

The project is valuable precisely because it makes the most common simulation-debugging visualizations obvious and lightweight. It addresses repeated requests for custom passive-viewer geometry, frame inspection, and Python-side overlays without trying to become another simulator or robotics platform. [github](https://github.com/google-deepmind/mujoco/issues/1023)