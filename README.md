# mujoco-debugdraw

**Draw targets, coordinate frames, force vectors, paths and labels in the MuJoCo Python viewer without editing MJCF or writing OpenGL.**

![demo: a 3-DoF arm tracking a target, with its plan, frames, error and velocity vectors drawn at runtime](assets/demo.gif)

*Everything except the arm is drawn by `mujoco-debugdraw` at runtime. The model XML contains no debug geometry. ([`examples/reaching_debug.py`](examples/reaching_debug.py))*

### On real robot models: Franka Panda and KUKA LBR iiwa 14

| Franka Emika Panda | KUKA LBR iiwa 14 |
|---|---|
| ![Panda tracking a figure-eight](assets/panda.gif) | ![iiwa tracking a figure-eight](assets/iiwa.gif) |

[`examples/manipulator_demo.py`](examples/manipulator_demo.py) drives either 7-DoF arm along a figure-eight with damped-least-squares differential IK. The robot XML is loaded unmodified from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie). The demo draws:

- the plan (cyan) and the actual TCP trail (magenta)
- the target point, plus its desired orientation as faint axes
- the actual TCP frame, and a frame on every link
- the TCP velocity
- a HUD with position/orientation error and **joint-limit usage per joint**

```bash
python examples/manipulator_demo.py                # Franka Panda
python examples/manipulator_demo.py --robot iiwa   # KUKA LBR iiwa 14
```

The robot models are not redistributed here. On first run the script fetches them from Menagerie (Apache-2.0, ~55 MB), or it uses `MUJOCO_MENAGERIE_PATH` if you already have a checkout.

## Install

```bash
pip install git+https://github.com/harshalbhat/mujoco-debugdraw
```

Once the package is on PyPI, this becomes `pip install mujoco-debugdraw`. The only dependencies are `mujoco` and `numpy`. It needs no ROS, Docker, CUDA, or browser.

## Quickstart

```python
import mujoco
import mujoco.viewer
from mujoco_debugdraw import DebugDraw

model = mujoco.MjModel.from_xml_path("examples/scene.xml")
data = mujoco.MjData(model)
target = [0.45, 0.00, 0.30]

with mujoco.viewer.launch_passive(model, data) as viewer:
    draw = DebugDraw(model, data, viewer)

    while viewer.is_running():
        mujoco.mj_step(model, data)

        draw.clear()
        draw.frame("end_effector", scale=0.12, label="EE")
        draw.point(target, color="lime", size=0.025, label="target")
        draw.line([[0.20, -0.20, 0.05], [0.30, -0.10, 0.10], target], color="cyan")
        draw.arrow(data.site_xpos[0], [0.0, 0.0, 0.10], color="yellow", label="v_des")
        draw.text("MuJoCo DebugDraw active", position="top_left")

        draw.sync()    # always BEFORE viewer.sync()
        viewer.sync()
```

On macOS, launch passive-viewer scripts with `mjpython` instead of `python`. This is a MuJoCo requirement, not specific to this package.

## API

| Call | Draws | Notes |
|---|---|---|
| `DebugDraw(model, data, viewer, max_geoms=500)` | — | `viewer` is the `launch_passive()` handle. A `mujoco.Renderer` or `MjvScene` also works (see below). |
| `point(position, color="red", size=0.02, label=None, key=None)` | Sphere | `size` is the radius in meters. |
| `line(points, color="cyan", width=2.0, key=None)` | Polyline, one segment per consecutive pair | `points` has shape `(N, 3)` with N ≥ 2. `width` is in **pixels**. |
| `arrow(origin, vector, color="yellow", scale=1.0, width=0.01, label=None, key=None)` | Shaft + head from `origin` to `origin + scale * vector` | `width` is the shaft radius in meters. **A zero-length arrow draws nothing** and does not raise. |
| `frame(target, scale=0.10, key=None, label=None, kind=None, width=None)` | X/Y/Z axes in red/green/blue | `target` is a body or site name, or a body ID. If a name is both a body and a site, the body wins. Use `kind="site"` to force a site, or with an int to mean a site ID. |
| `text(message, position="top_left", key=None)` | Screen-corner HUD | `top_left`, `top_right`, `bottom_left`, `bottom_right`. Messages in the same corner stack. |
| `clear()` | — | Drops every queued primitive, keyed or not. Call it once per frame. |
| `sync()` | — | Writes the queue into the scene. Call it before `viewer.sync()`. |

**Colors:** `red green blue yellow cyan magenta white black gray orange lime`, `"#RRGGBB"`, `"#RRGGBBAA"`, `[r, g, b]` or `[r, g, b, a]` with values in `[0, 1]`. Out-of-range values raise `InvalidColorError` rather than being silently clamped.

**Keys:** passing `key="goal"` replaces any primitive already queued under that key. This lets you update a marker from several places in your loop without drawing it twice.

**Errors:** bad shapes, non-finite numbers, and non-positive sizes raise `ValueError` with a message that names the argument. An unknown body or site name raises `UnknownTargetError` and suggests close matches (`No body or site named 'end_efector'. Did you mean: end_effector?`).

## How it works (and what it doesn't touch)

- Primitives become **visual-only geoms in `viewer.user_scn`**, written under `viewer.lock()` with the public `mjv_initGeom` / `mjv_connector` functions. They have no mass and no collision, and they never enter `MjModel`.
- `model` and `data` are only **read**. The test suite steps a model 300 times with and without drawing and asserts bit-identical `qpos`, `qvel`, and contact count.
- Each `sync()` rewrites DebugDraw's slice of `user_scn` from scratch, so geometry never accumulates. Any geoms you put in `user_scn` yourself *before* the first `sync()` are preserved.
- Frames are sampled when you call `frame()`, so call it after `mj_step` / `mj_forward` each frame.

### Offscreen rendering (videos, CI, notebooks)

Pass a `mujoco.Renderer` instead of a viewer. Call `draw.sync()` **after** `renderer.update_scene()` and before `renderer.render()`:

```python
renderer = mujoco.Renderer(model, 480, 640)
draw = DebugDraw(model, data, renderer)
...
renderer.update_scene(data, camera=cam)
draw.sync()
pixels = renderer.render()
```

The Renderer has no text overlay, so `draw.text()` is not drawn in this mode. The queued messages are available as `draw.texts` (grouped by corner) if you want to composite them yourself. [`examples/record_demo.py`](examples/record_demo.py) does this with Pillow to produce the GIF above.

## Why this exists

Debugging a controller usually means answering questions like: *is my target where I think it is? Is this frame rotated the way I assumed? Which way is that force pointing?* In the MuJoCo Python viewer, getting those answers on screen has meant one of three things:

- adding throwaway `site`/`geom` elements to the MJCF
- hand-filling `mjvGeom` structs in `user_scn`
- printing numbers

People keep asking for an easier way to add custom geoms and frames to the passive viewer ([google-deepmind/mujoco#1023](https://github.com/google-deepmind/mujoco/issues/1023)). This package is a small, tested wrapper around the supported path. It is not a new viewer, simulator, or UI framework.

## Contributing

```bash
git clone https://github.com/harshalbhat/mujoco-debugdraw && cd mujoco-debugdraw
pip install -e ".[dev]"
pytest                                   # unit + scene tests, no display needed
MUJOCO_GL=egl pytest                     # also runs the offscreen pixel tests
xvfb-run -a pytest                       # also runs the real passive-viewer test (Linux)
python examples/reaching_debug.py        # interactive demo
```

## License

MIT
