# -*- coding: utf-8 -*-
# file: test/integrals/test_reciprocal.py

# This code is part of Mandacaru.
# MIT License

"""Reciprocal space of a real-space grid."""

import numpy as np
import pytest

from mandacaru.integrals import Grid
from mandacaru.integrals import reciprocal as rc

#: A skewed (hexagonal-like) cell, Angstrom: the convention must not assume
#: orthogonal axes.
SKEWED = np.array([[6.0, 0.0, 0.0], [3.0, 5.196, 0.0], [0.0, 0.0, 7.0]])


@pytest.fixture(scope="module")
def grid():
    return Grid(center=0.5 * SKEWED.sum(axis=0), box_size=0.0, h=0.25,
                units="angstrom", cell=SKEWED, periodic=True)


def _periodic_gaussian(grid, center, sigma):
    x = np.stack([grid.X.ravel(), grid.Y.ravel(), grid.Z.ravel()], axis=1)
    rho = np.zeros(len(x))
    for R in rc.lattice_translations(rc.lattice_vectors(grid), 24.0):
        d = x - center - R
        rho += np.exp(-np.sum(d * d, axis=1) / (2 * sigma ** 2)) \
            / (2 * np.pi * sigma ** 2) ** 1.5
    return rho


class TestConventions:
    def test_the_reciprocal_vectors_are_dual(self, grid):
        A = rc.lattice_vectors(grid)
        assert np.allclose(A.T @ rc.reciprocal_vectors(A), 2 * np.pi * np.eye(3))

    def test_a_gaussian_transforms_analytically(self, grid):
        """Normalized Gaussian: exp(-sigma^2 G^2 / 2) at its own phase.

        Wide enough (sigma = 1.1 Bohr) to carry nothing at the grid's Nyquist
        wave-vector, where a narrower one would alias.
        """
        center, sigma = np.array([4.0, 3.0, 6.0]), 1.1
        transform = rc.to_reciprocal(grid, _periodic_gaussian(grid, center,
                                                              sigma))
        G = rc.wavevectors(grid)
        expected = (np.exp(-0.5 * sigma ** 2 * np.sum(G * G, axis=0))
                    * rc.structure_factor(G, center))
        assert np.max(np.abs(transform - expected)) < 1e-8

    def test_the_inverse_is_exact(self, grid):
        values = _periodic_gaussian(grid, np.array([2.0, 1.0, 3.0]), 0.9)
        back = rc.to_real(grid, rc.to_reciprocal(grid, values))
        assert np.max(np.abs(back - values)) < 1e-12

    def test_the_coulomb_kernel_drops_g_zero(self, grid):
        kernel = rc.coulomb_kernel(rc.wavevectors(grid))
        assert kernel.flat[0] == 0.0 and np.all(kernel.ravel()[1:] > 0.0)


class TestTransforms:
    def test_a_multipole_transform_matches_the_fft(self, grid):
        """g_L Y_LM by the Bessel transform equals the FFT of its samples."""
        from mandacaru.basis._angular import spherical_harmonic
        L, M, center, a = 1, 1, np.array([5.0, 4.0, 6.0]), 0.5
        x = np.stack([grid.X.ravel(), grid.Y.ravel(), grid.Z.ravel()], axis=1)
        values = np.zeros(len(x), dtype=complex)
        for R in rc.lattice_translations(rc.lattice_vectors(grid), 14.0):
            d = x - center - R
            r = np.linalg.norm(d, axis=1)
            theta = np.arccos(np.clip(d[:, 2] / np.maximum(r, 1e-300), -1, 1))
            phi = np.arctan2(d[:, 1], d[:, 0])
            values += r * np.exp(-a * r * r) * spherical_harmonic(L, M, theta,
                                                                  phi)
        G = rc.wavevectors(grid)
        norm = rc.spherical(G)[0]
        r = np.linspace(0.0, 8.0, 4001)
        radial = rc.radial_transform(r, r * np.exp(-a * r * r), L, norm)
        analytic = rc.multipole_transform(G, center, radial, L, M)
        assert np.max(np.abs(rc.to_reciprocal(grid, values) - analytic)) < 1e-8

    def test_lattice_translations_are_complete_and_symmetric(self, grid):
        A = rc.lattice_vectors(grid)
        R = rc.lattice_translations(A, 20.0)
        assert np.all(np.linalg.norm(R, axis=1) <= 20.0 + 1e-12)
        keys = {tuple(np.round(r, 9)) for r in R}
        assert all(tuple(np.round(-r, 9)) in keys for r in R)
        # Brute force over a generous integer box finds nothing more.
        n = np.arange(-6, 7)
        n1, n2, n3 = np.meshgrid(n, n, n, indexing="ij")
        brute = np.stack([n1.ravel(), n2.ravel(), n3.ravel()], axis=1) @ A.T
        assert np.sum(np.linalg.norm(brute, axis=1) <= 20.0) == len(R)
