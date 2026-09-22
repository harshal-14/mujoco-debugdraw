# Examples

| File | What it does |
|---|---|
| `scene.xml` | A plain 3-DoF arm. It contains no debug sites or geoms. |
| `reaching_debug.py` | An interactive passive-viewer demo. A Jacobian-transpose controller tracks a target on a circular plan, while DebugDraw shows the plan, target, body frames, error and velocity arrows, and a HUD. |
| `manipulator_demo.py` | A **Franka Panda** or **KUKA LBR iiwa 14** (unmodified MuJoCo Menagerie models) tracking a figure-eight with differential IK. It shows the plan vs. actual trail, target and TCP frames, per-link frames, velocity, and a joint-limit HUD. It fetches Menagerie (~55 MB) on first run unless `MUJOCO_MENAGERIE_PATH` is set. `--record out.gif` renders offscreen. |
| `record_demo.py` | Renders `assets/demo.gif` offscreen with `mujoco.Renderer`, using the same drawing code. |

```bash
python examples/reaching_debug.py                 # macOS: mjpython examples/reaching_debug.py
MUJOCO_GL=egl python examples/record_demo.py      # headless Linux
python examples/manipulator_demo.py --robot panda # or --robot iiwa
MUJOCO_GL=egl python examples/manipulator_demo.py --robot iiwa --record assets/iiwa.gif
```
