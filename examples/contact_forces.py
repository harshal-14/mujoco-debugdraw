"""Contact normals and contact forces on a stack of two boxes.

Run:  python examples/contact_forces.py            (macOS: mjpython ...)
      python examples/contact_forces.py --frames   # also draw every contact frame

Yellow arrows are contact forces (1 cm per newton) and red arrows are contact
normals. With no body filter, each arrow shows the force on the contact's
second geom, so at rest every arrow points up into the box above it. Push a
box (double-click it, then Ctrl + right-drag) and watch the arrows grow,
shrink and tilt.

The loop never calls clear(): keyed primitives are replaced each frame and
the per-contact frames expire after one sync (ttl_frames=1).
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer

from mujoco_debugdraw import DebugDraw

XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="1 1 .1"/>
    <body name="bottom" pos="0 0 0.05">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" mass="1.0" rgba=".5 .5 .55 1"/>
    </body>
    <body name="top" pos="0.01 0 0.14" euler="0 0 0.4">
      <freejoint/>
      <geom type="box" size="0.04 0.04 0.04" mass="0.4" rgba=".6 .55 .5 1"/>
    </body>
  </worldbody>
</mujoco>
"""

FORCE_SCALE = 0.01  # m per newton
STEPS_PER_FRAME = 8  # ~60 Hz with the default 2 ms timestep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", action="store_true", help="draw each contact's frame")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    g = -model.opt.gravity[2]
    weights = {name: model.body(name).mass[0] * g for name in ("bottom", "top")}

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.08]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.6, 135, -20
        draw = DebugDraw(model, data, viewer)

        while viewer.is_running():
            frame_start = time.time()
            for _ in range(STEPS_PER_FRAME):
                mujoco.mj_step(model, data)

            draw.contacts(force_scale=FORCE_SCALE, key="contacts")
            draw.body_com("top", color="magenta", size=0.008, key="com_top")
            draw.body_com("bottom", color="magenta", size=0.008, key="com_bottom")
            if args.frames:
                for i in range(data.ncon):
                    c = data.contact[i]
                    # contact.frame stores the axes in rows; frame_at wants columns.
                    draw.frame_at(c.pos, c.frame.reshape(3, 3).T, scale=0.03, width=0.002, ttl_frames=1)
            draw.text(
                f"contacts: {data.ncon}\n"
                f"weight: bottom {weights['bottom']:.2f} N, top {weights['top']:.2f} N",
                key="hud",
            )
            draw.text("yellow: contact force (1 cm/N)   red: contact normal   magenta: COM",
                      position="bottom_left", key="legend")

            draw.sync()  # always before viewer.sync()
            viewer.sync()
            time.sleep(max(0.0, STEPS_PER_FRAME * model.opt.timestep - (time.time() - frame_start)))


if __name__ == "__main__":
    main()
