import contextlib

import mujoco
import numpy as np
import pytest

# A hinged arm with a named body and site, plus a free-falling box so that
# "dynamics unchanged" checks exercise contacts too.
XML = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="1 1 .1"/>
    <body name="link" pos="0 0 0.5" euler="0 0 0.7">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"/>
      <body name="end_effector" pos="0.3 0 0" euler="0.3 0 0">
        <geom type="sphere" size="0.03"/>
        <site name="tip_site" pos="0.05 0 0" euler="0 0.4 0"/>
      </body>
    </body>
    <body name="box" pos="0.5 0.3 0.4">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_string(XML)


@pytest.fixture
def data(model):
    d = mujoco.MjData(model)
    d.qpos[0] = 0.4
    mujoco.mj_forward(model, d)
    return d


@pytest.fixture
def scene(model):
    return mujoco.MjvScene(model, maxgeom=1000)


class FakeViewer:
    """Duck-types the passive viewer handle: user_scn, lock(), set_texts()."""

    def __init__(self, model, maxgeom=1000):
        self.user_scn = mujoco.MjvScene(model, maxgeom=maxgeom)
        self.texts = None
        self.cleared = 0
        self.lock_calls = 0

    @contextlib.contextmanager
    def lock(self):
        self.lock_calls += 1
        yield

    def set_texts(self, texts):
        self.texts = texts

    def clear_texts(self):
        self.cleared += 1
        self.texts = None


@pytest.fixture
def fake_viewer(model):
    return FakeViewer(model)


def assert_close(a, b, tol=1e-9):
    np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), atol=tol)
