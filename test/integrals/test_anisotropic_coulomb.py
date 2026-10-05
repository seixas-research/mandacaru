# -*- coding: utf-8 -*-
# file: test/test_anisotropic_coulomb.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The FFT Coulomb path follows the voxel geometry, not just ``dx``.

Using one spacing for every axis made the same physical problem give different
energies on axis permutations of an anisotropic grid (a factor of four for
``(0.4, 0.2, 0.3)`` vs ``(0.2, 0.4, 0.3)`` Bohr).  The solver is checked
against the closed-form self interaction of a Gaussian on anisotropic and
skewed samplings; the voxel self-potential -- the periodic truncated kernel's
``d = 0`` node -- against quadrature.
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


#: ``int int exp(-r1^2) exp(-r2^2) / r12`` -- two unit-exponent Gaussians,
#: ``pi^3 * 2 sqrt(mu/pi)`` with ``mu = 1/2``.
GAUSSIAN_SELF_INTERACTION = np.pi ** 2.5 * np.sqrt(2.0)


def self_integral(grid, rho):
    """``sum rho Phi dV`` -- the density's Coulomb self interaction."""
    phi = PoissonFFTSolver(grid.shape, step=grid.step).solve(rho)
    return float(np.real(np.sum(rho.conj() * phi) * grid.dV))


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
            assert value == pytest.approx(reference, rel=1e-10), permutation

    @pytest.mark.parametrize("spacings", [(0.4, 0.2, 0.3), (0.5, 0.3, 0.4)])
    def test_matches_the_closed_form(self, spacings):
        """Spectral at the singularity: exact to 1e-8 on an anisotropic
        sampling."""
        assert self_integral(*gaussian_grid(spacings)) == pytest.approx(
            GAUSSIAN_SELF_INTERACTION, rel=1e-8)


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
    """A skewed lattice is a Bravais lattice: the convolution still holds,
    and the spectral part's wave vectors come from the skewed padded cell."""

    CELL = 3.0 * np.array([[3.0, 0.0, 0.0], [1.2, 2.8, 0.0], [0.4, 0.5, 2.6]])

    @pytest.mark.parametrize("skew", [True, False])
    def test_matches_the_closed_form(self, skew):
        grid = Grid(center=[0.0, 0.0, 0.0], h=0.45, units="bohr",
                    cell=self.CELL, skew=skew)
        assert grid.is_orthogonal is not skew
        r2 = grid.X ** 2 + grid.Y ** 2 + grid.Z ** 2
        rho = np.exp(-r2).reshape(-1).astype(complex)
        assert self_integral(grid, rho) == pytest.approx(
            GAUSSIAN_SELF_INTERACTION, rel=1e-8)

    def test_a_skewed_cell_beats_its_bounding_box_on_volume(self):
        """Sanity: the skewed sampling really uses the skewed voxel volume."""
        grid = Grid(center=[0.0, 0.0, 0.0], h=0.9, units="bohr",
                    cell=self.CELL / 3.0, skew=True)
        assert grid.dV == pytest.approx(abs(np.linalg.det(grid.step)), rel=1e-12)
