# Examples

| File | What it does |
|---|---|
| `scene.xml` | A plain 3-DoF arm. It contains no debug sites or geoms. |
| `reaching_debug.py` | An interactive passive-viewer demo. A Jacobian-transpose controller tracks a target on a circular plan, while DebugDraw shows the plan, target, body frames, error and velocity arrows, and a HUD. |
| `manipulator_demo.py` | A **Franka Panda** or **KUKA LBR iiwa 14** (unmodified MuJoCo Menagerie models) tracking a figure-eight with differential IK. It shows the plan vs. actual trail inside a workspace box, target and TCP frames, joint axes, the arm's COM, velocity, contact forces on a spring-loaded bumper the gripper brushes (`--no-bumper` removes it), and a joint-limit HUD. It fetches Menagerie (~55 MB) on first run unless `MUJOCO_MENAGERIE_PATH` is set. `--record out.gif` renders offscreen. |
| `pick_and_place.py` | The Franka Panda picks a raspberry, a rubber duck, two rocks and a light bulb off a table and drops them in a tray, round after round with a new layout. It shows each target's bounding box (green while held, red if it slips), COM and mass, the grip force on each finger, the remaining waypoints, the grasp, target and TCP frames, the carried path, and short-lived phase labels. The objects are built from MuJoCo primitives, and the rocks are random convex meshes. Needs MuJoCo ≥ 3.3. |
| `contact_forces.py` | Two stacked boxes with `contacts()` drawing each contact's normal and force, plus `body_com()` markers. The loop never calls `clear()`. `--frames` also draws every contact frame with `frame_at()`. |
| `record_demo.py` | Renders `assets/demo.gif` offscreen with `mujoco.Renderer`, using the same drawing code. |

```bash
python examples/reaching_debug.py                 # macOS: mjpython examples/reaching_debug.py
MUJOCO_GL=egl python examples/record_demo.py      # headless Linux
python examples/manipulator_demo.py --robot panda # or --robot iiwa
python examples/contact_forces.py --frames         # contact normals, forces and frames
python examples/pick_and_place.py                  # Panda tabletop pick and place
MUJOCO_GL=egl python examples/manipulator_demo.py --robot iiwa --record assets/iiwa.gif
```
