# -*- coding: utf-8 -*-
# file: test_gaussian.py

import numpy as np
import pytest

from carcara.basis import GaussianOrbital
from carcara.basis.gaussian import primitive_norm
from carcara.integrals import Grid


def _radial_norm(orb, r_max=40.0, n=400000):
    r = np.linspace(0.0, r_max, n)
    return float(np.trapezoid(orb.radial(r) ** 2 * r * r, r))


class TestNormalization:
    @pytest.mark.parametrize("l", [0, 1, 2])
    def test_single_primitive_is_normalized(self, l):
        orb = GaussianOrbital(l, 0, [0.8], [1.0], units="bohr")
        assert np.isclose(_radial_norm(orb), 1.0, atol=1e-4)

    def test_contracted_is_normalized(self):
        orb = GaussianOrbital(0, 0, [18.731, 2.8254, 0.64012],
                              [0.033495, 0.234727, 0.813757], units="bohr")
        assert np.isclose(_radial_norm(orb), 1.0, atol=1e-4)

    def test_primitive_norm_matches_manual_integral(self):
        alpha, l = 1.3, 0
        N = primitive_norm(alpha, l)
        r = np.linspace(0.0, 40.0, 400000)
        prim = N * r ** l * np.exp(-alpha * r * r)
        assert np.isclose(np.trapezoid(prim * prim * r * r, r), 1.0, atol=1e-4)

    def test_3d_grid_normalization(self):
        orb = GaussianOrbital(0, 0, [1.3], [1.0], units="bohr")
        grid = Grid(center=[0, 0, 0], box_size=12.0, h=0.30, units="bohr")
        psi = orb.evaluate(grid.X, grid.Y, grid.Z)
        assert abs(float(np.sum(np.abs(psi) ** 2) * grid.dV) - 1.0) < 0.02


class TestGaussianOrbital:
    def test_broadcasts_over_grid_shape(self):
        orb = GaussianOrbital(1, 1, [0.8, 0.2], [0.5, 0.5], units="bohr")
        x = np.linspace(-2, 2, 5)
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        out = orb.evaluate(X, Y, Z)
        assert out.shape == X.shape and out.dtype == np.complex128

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            GaussianOrbital(0, 0, [1.0, 2.0], [1.0])
