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


#: Cells whose grids alias differently (Angstrom), with the order of their
#: lattice's point group on these grids.
ALIASING_CELLS = {
    "orthorhombic": (np.diag([3.0, 3.4, 4.1]), 8),
    "hexagonal": (np.array([[3.11, 0.0, 0.0],
                            [-1.555, 1.555 * np.sqrt(3.0), 0.0],
                            [0.0, 0.0, 4.98]]), 24),
    "fcc": (2.715 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0],
                              [1.0, 1.0, 0.0]]), 48),
    "triclinic": (np.array([[3.0, 0.0, 0.0], [0.9, 3.2, 0.0],
                            [0.0, 0.4, 3.5]]), 2),
}


def _periodic_grid(cell, nodes):
    cell = np.asarray(cell, dtype=float)
    return Grid(center=0.5 * cell.sum(axis=0), box_size=0.0,
                h=np.linalg.norm(cell, axis=1) / np.asarray(nodes),
                units="angstrom", cell=cell, periodic=True)


def _frequencies(grid):
    return np.stack(np.meshgrid(*[np.fft.fftfreq(n, 1.0 / n)
                                  for n in grid.shape],
                                indexing="ij")).reshape(3, -1).astype(int)


class TestSymmetricAliases:
    @pytest.mark.parametrize("name", sorted(ALIASING_CELLS))
    def test_the_lattice_operations_are_the_point_group(self, name):
        cell, order = ALIASING_CELLS[name]
        operations = rc.lattice_operations(_periodic_grid(cell, (6, 6, 6)))
        assert len(operations) == order
        keys = {tuple(V.ravel()) for V in operations}
        assert all(tuple((P @ Q).ravel()) in keys
                   for P in operations for Q in operations)

    def test_a_box_cube_has_the_full_cubic_group(self):
        grid = Grid(center=np.zeros(3), box_size=6.0, h=0.4, units="bohr")
        assert len(rc.lattice_operations(grid)) == 48

    @pytest.mark.parametrize("name", sorted(ALIASING_CELLS))
    def test_each_frequency_gets_a_mean_of_its_aliases(self, name):
        """The averaged frequency differs from the box's by a mean of
        alias periods, and not at all where every operation keeps the
        frequency inside the box."""
        cell, _order = ALIASING_CELLS[name]
        grid = _periodic_grid(cell, (6, 6, 8))
        B = rc.reciprocal_vectors(rc.lattice_vectors(grid))
        m = _frequencies(grid)
        shift = np.linalg.solve(B, rc.symmetric_wavevectors(grid).reshape(
            3, -1)) - m
        count = len(rc.lattice_operations(grid))
        periods = shift * count / np.asarray(grid.shape)[:, None]
        assert np.allclose(periods, np.round(periods), atol=1e-9)
        inside = np.all(2 * np.abs(m) < 0.5 * np.asarray(grid.shape)[:, None],
                        axis=0)
        assert np.allclose(shift[:, inside], 0.0)

    @pytest.mark.parametrize("nodes", [(12, 12, 20), (11, 11, 19)])
    def test_a_hexagonal_rotation_maps_them_onto_themselves(self, nodes):
        """The 60-degree rotation permutes the frequencies of a hexagonal
        grid; the averaged wave-vectors rotate with it, and are odd, while
        the box's do neither."""
        grid = _periodic_grid(ALIASING_CELLS["hexagonal"][0], nodes)
        angle = np.pi / 3.0
        R = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                      [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
        B = rc.reciprocal_vectors(rc.lattice_vectors(grid))
        W = np.rint(np.linalg.solve(B, R @ B)).astype(int)
        shape = np.asarray(grid.shape)[:, None]
        m = _frequencies(grid)
        rotated = np.ravel_multi_index((W @ m) % shape, grid.shape)
        negated = np.ravel_multi_index((-m) % shape, grid.shape)
        vectors, norms = rc.symmetric_aliases(grid)
        vectors, norms = vectors.reshape(3, -1), norms.ravel()
        assert np.max(np.abs(vectors[:, rotated] - R @ vectors)) < 1e-12
        assert np.max(np.abs(vectors[:, negated] + vectors)) < 1e-12
        assert np.max(np.abs(norms[rotated] - norms)) < 1e-12
        box = rc.wavevectors(grid).reshape(3, -1)
        assert np.max(np.abs(box[:, rotated] - R @ box)) > 1.0

    def test_a_strain_moves_them_linearly(self):
        """The average is over integers, so a shear that keeps the
        operations changes only the reciprocal vectors it is multiplied
        by: the finite-strain stress sees the box's linear dependence."""
        cell = ALIASING_CELLS["hexagonal"][0]
        plain = _periodic_grid(cell, (12, 12, 20))
        reference = np.linalg.solve(
            rc.reciprocal_vectors(rc.lattice_vectors(plain)),
            rc.symmetric_wavevectors(plain).reshape(3, -1))
        for shear in (1e-3, -1e-3):
            strained = _periodic_grid(cell @ (np.eye(3) + shear * np.array(
                [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])),
                (12, 12, 20))
            assert len(rc.lattice_operations(strained)) == 24
            indices = np.linalg.solve(
                rc.reciprocal_vectors(rc.lattice_vectors(strained)),
                rc.symmetric_wavevectors(strained).reshape(3, -1))
            assert np.allclose(indices, reference, atol=1e-12)

    def test_on_an_orthorhombic_grid_only_the_nyquist_plane_differs(self):
        grid = _periodic_grid(ALIASING_CELLS["orthorhombic"][0], (6, 7, 8))
        vectors, norms = rc.symmetric_aliases(grid)
        G = rc.wavevectors(grid)
        assert np.allclose(norms, np.sqrt(np.sum(G * G, axis=0)))
        nyquist = np.zeros(grid.shape, dtype=bool)
        nyquist[3, :, :] = True
        nyquist[:, :, 4] = True
        assert np.allclose(vectors[:, ~nyquist], G[:, ~nyquist])
        assert np.allclose(vectors[0, 3], 0.0) and np.allclose(
            vectors[2, :, :, 4], 0.0)
