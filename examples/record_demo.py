"""Render assets/demo.gif offscreen (no window needed).

Run:  MUJOCO_GL=egl python examples/record_demo.py      (Linux, headless)
      python examples/record_demo.py                    (desktop)

Uses the same DebugDraw calls as reaching_debug.py, but draws into a
mujoco.Renderer. The Renderer has no text overlay, so the HUD text from
draw.texts is composited with Pillow to show what the passive viewer
displays via viewer.set_texts().
"""

from __future__ import annotations

import pathlib

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mujoco_debugdraw import DebugDraw
from reaching_debug import HERE, PERIOD, ReachController, draw_debug, planned_path, target_at

W, H, FPS = 640, 420, 15


def hud(img: np.ndarray, texts: dict[str, list[str]], W: int = W, H: int = H, font_size: int = 15) -> np.ndarray:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    for corner, msgs in texts.items():
        body = "\n".join(msgs)
        box = d.multiline_textbbox((0, 0), body, font=font, spacing=4)
        w, h = box[2] + 16, box[3] + 12
        x = 10 if "left" in corner else W - w - 10
        y = 10 if "top" in corner else H - h - 10
        d.rectangle([x, y, x + w, y + h], fill=(0, 0, 0, 150))
        d.multiline_text((x + 8, y + 6), body, font=font, fill=(255, 255, 255, 255), spacing=4)
    return np.asarray(im)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(HERE / "scene.xml"))
    data = mujoco.MjData(model)
    data.qpos[:] = [0.0, 0.6, 0.6]
    ctrl = ReachController(model)
    path = planned_path()

    renderer = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.3, 0.0, 0.25]
    cam.distance, cam.azimuth, cam.elevation = 1.0, 60, -20
    draw = DebugDraw(model, data, renderer)

    steps_per_frame = int(round(1.0 / (FPS * model.opt.timestep)))
    frames = []
    total = int(PERIOD * FPS)  # one full lap, loops seamlessly-ish
    warmup = int(1.0 / model.opt.timestep)
    for _ in range(warmup):  # let the arm converge before recording
        ctrl(model, data, target_at(data.time))
        mujoco.mj_step(model, data)

    for _ in range(total):
        target = target_at(data.time)
        for _ in range(steps_per_frame):
            pos, vel, err = ctrl(model, data, target_at(data.time))
            mujoco.mj_step(model, data)
        draw_debug(draw, data, ctrl, target_at(data.time), pos, vel, err, path)
        renderer.update_scene(data, camera=cam)
        draw.sync()
        frames.append(hud(renderer.render(), draw.texts))

    out = HERE.parent / "assets" / "demo.gif"
    out.parent.mkdir(exist_ok=True)
    pal = [Image.fromarray(f).quantize(colors=256, method=Image.Quantize.MAXCOVERAGE, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=int(1000 / FPS), loop=0, optimize=True)
    print(f"wrote {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
