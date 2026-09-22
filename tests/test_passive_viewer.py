"""Runs against the real mujoco.viewer.launch_passive window.

Needs a display. On headless Linux: xvfb-run -a pytest tests/test_passive_viewer.py
Skipped when no display is available.
"""

import os
import sys

import mujoco
import numpy as np
import pytest

from mujoco_debugdraw import DebugDraw

from conftest import XML

pytestmark = pytest.mark.skipif(
    sys.platform.startswith("linux") and not os.environ.get("DISPLAY"),
    reason="passive viewer needs a display (use xvfb-run)",
)


def test_passive_viewer_end_to_end():
    viewer_mod = pytest.importorskip("mujoco.viewer")
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    try:
        ctx = viewer_mod.launch_passive(m, d, show_left_ui=False, show_right_ui=False)
    except Exception as exc:
        pytest.skip(f"cannot open passive viewer here: {exc}")
    with ctx as viewer:
        draw = DebugDraw(m, d, viewer)
        for _ in range(120):
            mujoco.mj_step(m, d)
            draw.clear()
            draw.frame("end_effector", scale=0.12, label="EE")
            draw.point([0.45, 0.0, 0.30], color="lime", label="target")
            draw.line([[0.2, -0.2, 0.05], [0.3, -0.1, 0.1], [0.45, 0, 0.3]])
            draw.arrow(d.site_xpos[0], [0, 0, 0.1], color="yellow")
            draw.text(f"t={d.time:.2f}")
            draw.sync()
            viewer.sync()
        assert viewer.user_scn.ngeom == 3 + 1 + 2 + 1
        assert np.all(np.isfinite(d.qpos))
