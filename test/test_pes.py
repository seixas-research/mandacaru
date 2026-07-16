# -*- coding: utf-8 -*-
# file: test_pes.py

"""Smoke tests for the potential-energy-surface example helpers (examples/pes_utils.py)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "examples"))

pes_utils = pytest.importorskip("pes_utils")
from pes_utils import (GridSpec, atom_energy, commensurate_distances,  # noqa: E402
                       rhf_total_energy)
from carcara.basis import BasisSet  # noqa: E402
from carcara.integrals import Grid  # noqa: E402


class TestCommensurateDistances:
    def test_step_is_twice_the_spacing(self):
        d = commensurate_distances(1.0, 3.5, GridSpec(box_size=10.0, spacing=0.16))
        assert d[0] == pytest.approx(1.0)
        assert np.allclose(np.diff(d), 2 * 0.16)
        assert d[-1] <= 3.5 + 1e-9

    def test_single_point_when_range_below_step(self):
        d = commensurate_distances(1.0, 1.1, GridSpec(box_size=8.0, spacing=0.2))
        assert len(d) == 1


@pytest.fixture(scope="module")
def grid():
    return Grid(center=[0.0, 0.0, 0.0], box_size=6.0, h=0.22)


class TestEnergies:
    def test_h2_rhf_energy_is_bound(self, grid):
        sto3g = BasisSet.build("GTO", n_gaussians=3)
        pos = [np.array([0.0, 0.0, -0.37]), np.array([0.0, 0.0, 0.37])]
        e = rhf_total_energy(["H", "H"], pos, sto3g, grid)
        # H2 near equilibrium is well below two isolated H atoms (2 * -0.5).
        assert -1.3 < e < -1.0

    def test_isolated_hydrogen_atom(self, grid):
        sto3g = BasisSet.build("GTO", n_gaussians=3)
        e = atom_energy("H", sto3g, grid, position=[0.03, 0.02, -0.37])
        assert e == pytest.approx(-0.5, abs=0.06)
