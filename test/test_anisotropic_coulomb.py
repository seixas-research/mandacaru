# -*- coding: utf-8 -*-
# file: test/test_anisotropic_coulomb.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The FFT Coulomb path follows the voxel geometry, not just ``dx``.

Using one spacing for every axis made the same physical problem give different
energies on axis permutations of an anisotropic grid (a factor of four for
``(0.4, 0.2, 0.3)`` vs ``(0.2, 0.4, 0.3)`` Bohr).
"""

import numpy as np
import pytest

from mandacaru.integrals import Grid, PoissonFFTSolver
from mandacaru.integrals.poisson import (CUBE_SELF_CONSTANT, cell_self_potential,
                                         voxel_self_potential)


def gaussian_grid(spacings, half=4.0):
    """An isotropic ``exp(-r^2)`` sampled on a grid with the given spacings."""
    grid = Grid(center=[0.0, 0.0, 0.0], box_size=half, h=list(spacings),
                units="bohr")
    r2 = grid.X ** 2 + grid.Y ** 2 + grid.Z ** 2
    return grid, np.exp(-r2).reshape(-1).astype(complex)


def skewed_grid(cell, h=0.5):
    """A Gaussian sampled on the skewed lattice of ``cell``."""
    grid = Grid(center=[0.0, 0.0, 0.0], h=h, units="bohr",
                cell=np.asarray(cell, dtype=float), skew=True)
    r2 = grid.X ** 2 + grid.Y ** 2 + grid.Z ** 2
    return grid, np.exp(-r2).reshape(-1).astype(complex)


def self_integral(grid, rho):
    """``sum rho Phi dV`` -- the density's Coulomb self interaction."""
    phi = PoissonFFTSolver(grid.shape, step=grid.step).solve(rho)
    return float(np.real(np.sum(rho.conj() * phi) * grid.dV))


def direct_self_integral(grid, rho):
    """The same convolution summed in real space, with the same self term."""
    x, y, z = (c.reshape(-1) for c in (grid.X, grid.Y, grid.Z))
    d = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2
                + (z[:, None] - z[None, :]) ** 2)
    green = np.divide(1.0, d, out=np.zeros_like(d), where=d > 0)
    np.fill_diagonal(green, voxel_self_potential(grid.step) / grid.dV)
    return float(np.real(rho.conj() @ (green @ rho)) * grid.dV ** 2)


class TestCellSelfPotential:
    def test_cube_matches_the_tabulated_constant(self):
        assert cell_self_potential(1.0, 1.0, 1.0) == pytest.approx(
            CUBE_SELF_CONSTANT, abs=2e-6)
        # It scales as dx^2: the integral has units of length squared.
        assert cell_self_potential(0.3, 0.3, 0.3) == pytest.approx(
            0.09 * CUBE_SELF_CONSTANT, abs=2e-7)

    def test_matches_numerical_quadrature_for_a_flat_cell(self):
        rng = np.random.default_rng(0)
        dx, dy, dz = 0.4, 0.2, 0.3
        points = rng.uniform(-0.5, 0.5, size=(2_000_000, 3)) * [dx, dy, dz]
        numeric = dx * dy * dz * np.mean(1.0 / np.linalg.norm(points, axis=1))
        assert cell_self_potential(dx, dy, dz) == pytest.approx(numeric, rel=2e-3)

    def test_positive_spacings_required(self):
        with pytest.raises(ValueError, match="positive"):
            cell_self_potential(0.2, 0.0, 0.2)


class TestAnisotropicCoulomb:
    def test_axis_permutations_give_the_same_self_energy(self):
        """The reported defect: permuting the spacings changed the answer."""
        base = [0.4, 0.2, 0.3]
        reference = self_integral(*gaussian_grid(base))
        for permutation in ([0.2, 0.4, 0.3], [0.3, 0.2, 0.4], [0.2, 0.3, 0.4]):
            value = self_integral(*gaussian_grid(permutation))
            assert value == pytest.approx(reference, rel=2e-3), permutation

    def test_matches_an_independent_direct_sum(self):
        """Same convolution, summed in real space with the same cell self term."""
        grid, rho = gaussian_grid([0.5, 0.3, 0.4], half=1.5)
        assert self_integral(grid, rho) == pytest.approx(
            direct_self_integral(grid, rho), rel=1e-10)

    def test_isotropic_result_is_unchanged(self):
        """A cubic grid still integrates as before (pinned energies must hold)."""
        grid, rho = gaussian_grid([0.3, 0.3, 0.3])
        legacy = PoissonFFTSolver(grid.shape, grid.dx,
                                  self_const=CUBE_SELF_CONSTANT)
        phi = legacy.solve(rho)
        legacy_value = float(np.real(np.sum(rho.conj() * phi) * grid.dV))
        assert self_integral(grid, rho) == pytest.approx(legacy_value, rel=1e-7)


class TestVoxelSelfPotential:
    """The ``d = 0`` node integrates 1/r over its own parallelepiped."""

    @pytest.mark.parametrize("spacings", [(1.0, 1.0, 1.0), (0.4, 0.2, 0.3),
                                          (0.5, 0.5, 0.2)])
    def test_orthogonal_voxels_match_the_rectangular_formula(self, spacings):
        assert voxel_self_potential(np.diag(spacings)) == pytest.approx(
            cell_self_potential(*spacings), rel=1e-12)

    @pytest.mark.parametrize("step", [
        np.array([[0.5, 0.2, 0.0], [0.0, 0.45, 0.1], [0.0, 0.0, 0.4]]),
        np.array([[0.6, 0.3, 0.15], [0.1, 0.5, 0.2], [0.05, 0.1, 0.55]]),
    ])
    def test_skewed_voxels_match_numerical_quadrature(self, step):
        rng = np.random.default_rng(1)
        u = rng.uniform(-0.5, 0.5, size=(2_000_000, 3))
        points = u @ step.T
        numeric = (abs(np.linalg.det(step))
                   * np.mean(1.0 / np.linalg.norm(points, axis=1)))
        assert voxel_self_potential(step) == pytest.approx(numeric, rel=3e-3)

    def test_a_degenerate_voxel_is_refused(self):
        with pytest.raises(ValueError, match="zero volume"):
            voxel_self_potential(np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0]]))


class TestSkewedGrids:
    """A skewed lattice is a Bravais lattice: the convolution still holds."""

    CELL = np.array([[3.0, 0.0, 0.0], [1.2, 2.8, 0.0], [0.4, 0.5, 2.6]])

    def test_matches_an_independent_direct_sum(self):
        grid, rho = skewed_grid(self.CELL, h=0.9)
        assert not grid.is_orthogonal
        assert self_integral(grid, rho) == pytest.approx(
            direct_self_integral(grid, rho), rel=1e-10)

    @pytest.mark.parametrize("skew", [True, False])
    def test_fft_and_direct_agree_on_the_same_operator(self, skew):
        """Both paths now use the same voxel self-energy, so they must match.

        Before, the direct kernel clamped its own ``r12 = 0`` distance to
        1e-15 and returned ~1e12 for this integral.
        """
        from mandacaru.basis import FullAtomicOrbital
        from mandacaru.integrals import IntegralEngine

        grid = Grid(center=[0.0, 0.0, 0.0], h=0.6, units="bohr",
                    cell=self.CELL, skew=skew)
        orbital = FullAtomicOrbital(1, 0, 0, Z=1.0, center=[0.0, 0.0, 0.0],
                                    units="bohr")
        engine = IntegralEngine([orbital], grid)
        fft = engine.two_body(method="fft", energy_units="Ha")[0, 0, 0, 0].real
        direct = engine.two_body(method="direct", energy_units="Ha",
                                 softening=0.0)[0, 0, 0, 0].real
        assert fft == pytest.approx(direct, rel=1e-10)
        assert 0.0 < fft < 2.0                 # the exact 1s value is 0.625 Ha

    def test_a_skewed_cell_beats_its_bounding_box_on_volume(self):
        """Sanity: the skewed sampling really uses the skewed voxel volume."""
        grid, _rho = skewed_grid(self.CELL, h=0.9)
        assert grid.dV == pytest.approx(abs(np.linalg.det(grid.step)), rel=1e-12)
