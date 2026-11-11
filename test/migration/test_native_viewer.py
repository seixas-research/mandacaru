"""MG-V1: the viewer's geometry kernels run in Rust (``math::isosurface``) and reproduce the frozen
outputs of the Python kernels they replaced (``reference/viewer.json``, ``freeze_viewer.py``).

VQ-8: what the viewer derives before it plots (the weights and isovalue, the extent, the camera
paths, the cloud, the surface and the colors, sizes, bonds and limits it hands Matplotlib) runs
in Rust (``math::viewer``) and reproduces, bit for bit, what the Python route produced
(``reference/viewer_views.json``, ``freeze_viewer.py --views``)."""
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

from mandacaru import _native
from mandacaru.utils.viewer import enclosing_level, marching_tetrahedra

from .freeze_viewer import views

REFERENCE = json.loads((Path(__file__).parent / "reference" / "viewer.json").read_text())


@pytest.mark.parametrize("case", REFERENCE["surfaces"], ids=lambda c: c["name"])
def test_the_triangles_are_the_frozen_ones(case):
    shape = tuple(case["shape"])
    values, X, Y, Z = (np.asarray(case[k]).reshape(shape) for k in ("values", "x", "y", "z"))
    triangles = marching_tetrahedra(values, X, Y, Z, case["level"])
    assert triangles.shape == (case["count"], 3, 3)
    assert np.array_equal(triangles.ravel(), np.asarray(case["triangles"]))


@pytest.mark.parametrize("case", REFERENCE["levels"], ids=lambda c: str(c["fraction"]))
def test_the_enclosing_levels_are_the_frozen_ones(case):
    assert enclosing_level(REFERENCE["weights"], case["fraction"]) == case["level"]


VIEWS = json.loads((Path(__file__).parent / "reference" / "viewer_views.json").read_text())


@pytest.fixture(scope="module")
def current_views():
    return views()


# The views were frozen on Apple Arm with the system libm; the C library's cos, exp, log and pow
# differ in the last bit from one libm to another, so the bit-for-bit comparison runs there only.
@pytest.mark.skipif(not (sys.platform == "darwin" and platform.machine() == "arm64"),
                    reason="the references were frozen on Apple Arm (system libm)")
@pytest.mark.parametrize("name", list(VIEWS))
def test_the_views_are_the_frozen_ones_bit_for_bit(name, current_views):
    # json.dumps writes every float with its shortest round-trip form, NaN included.
    assert json.dumps(current_views[name], sort_keys=True) == json.dumps(VIEWS[name], sort_keys=True)


@pytest.mark.skipif(platform.machine() not in ("arm64", "aarch64"), reason="compares with NumPy's Arm kernels")
def test_the_lighting_angles_are_two_calls_of_the_c_library():
    """The sine and cosine of one angle must not fuse into ``sincos`` (a different last bit): with the
    axes as normals the face colors are ``0.35 + 0.65 |light_i|``, so every component of the light
    vector shows. The first two angles are ones where the fused form disagreed with NumPy."""
    rng = np.random.default_rng(3)
    angles = [(14.690531554752308, -368.81695589505773), (-0.6161108713485008, -693.9566312947725)]
    angles += [(float(e), float(a)) for e, a in zip(rng.uniform(-90, 90, 400), rng.uniform(-720, 720, 400))]
    normals = np.eye(3)
    base = np.ones((3, 3))
    for elev, azim in angles:
        e, a = np.radians(elev + 25.0), np.radians(azim - 30.0)
        light = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        shade = 0.35 + 0.65 * np.abs(normals @ light)
        expected = np.concatenate([base * shade[:, None], np.full((3, 1), 0.7)], axis=1)
        assert np.array_equal(_native.surface_rgba(normals, base, elev, azim, 0.7), expected)
