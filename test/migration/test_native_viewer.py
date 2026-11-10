"""MG-V1: the viewer's geometry kernels run in Rust (``math::isosurface``) and reproduce the frozen
outputs of the Python kernels they replaced (``reference/viewer.json``, ``freeze_viewer.py``)."""
import json
from pathlib import Path

import numpy as np
import pytest

from mandacaru.utils.viewer import enclosing_level, marching_tetrahedra

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
