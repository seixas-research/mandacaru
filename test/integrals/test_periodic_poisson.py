# -*- coding: utf-8 -*-
# file: test/integrals/test_periodic_poisson.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The periodic Coulomb kernel: ``Phi(G) = 4 pi rho(G) / G^2``.

:class:`~mandacaru.integrals.poisson.PoissonFFTSolver` zero-pads so the Coulomb
tail cannot wrap the box, which is what makes it the potential of an *isolated*
density.  A crystal needs the opposite, and
:class:`~mandacaru.integrals.poisson.PeriodicPoissonSolver` provides it.

Two checks here are exact rather than approximate, which is the point of
choosing them:

* a single plane-wave density has the analytic solution
  ``Phi = (4 pi / G^2) cos(G . r)``, so the normalization and the
  reciprocal-lattice construction are pinned to machine precision;
* the periodic Hartree energy of narrow Gaussians must equal the **Ewald**
  energy of the corresponding point charges plus each Gaussian's own
  self-energy -- an independent implementation (``core.ewald``) reached by a
  completely different route.
"""

import numpy as np
import pytest

from mandacaru.core.ewald import ewald_energy
from mandacaru.integrals.poisson import (PeriodicPoissonSolver,
                                         PoissonFFTSolver, fft_g_squared)

BOX = 7.0        # Bohr
NODES = 24


@pytest.fixture(scope="module")
def solver():
    return PeriodicPoissonSolver(shape=(NODES,) * 3, spacing=BOX / NODES)


@pytest.fixture(scope="module")
def coordinates():
    axis = np.arange(NODES) * (BOX / NODES)
    return np.meshgrid(axis, axis, axis, indexing="ij")


class TestThePlaneWaveSolutionIsExact:
    """``rho = cos(G.r)`` has the analytic potential ``(4 pi / G^2) rho``."""

    @pytest.mark.parametrize("miller", [(1, 0, 0), (2, 1, 0), (3, -2, 1)])
    def test_a_single_reciprocal_vector(self, solver, coordinates, miller):
        x, y, z = coordinates
        wavevector = 2.0 * np.pi * np.array(miller) / BOX
        density = np.cos(wavevector[0] * x + wavevector[1] * y
                         + wavevector[2] * z)
        potential = solver.solve(density.reshape(-1)).real.reshape(density.shape)
        exact = (4.0 * np.pi / (wavevector @ wavevector)) * density
        assert potential == pytest.approx(exact, abs=1e-12)

    def test_a_uniform_density_gives_nothing(self, solver):
        """It *is* the ``G = 0`` component, which the background cancels."""
        potential = solver.solve(np.ones(NODES ** 3))
        assert np.abs(potential).max() == pytest.approx(0.0, abs=1e-14)

    def test_the_potential_has_zero_mean(self, solver, coordinates):
        """Dropping ``G = 0`` fixes the reference: the mean potential is zero."""
        x, _y, _z = coordinates
        density = np.cos(2.0 * np.pi * x / BOX) + 3.0     # a non-neutral density
        potential = solver.solve(density.reshape(-1)).real
        assert potential.mean() == pytest.approx(0.0, abs=1e-12)


class TestItAgreesWithEwald:
    """The independent check: a different implementation, a different method."""

    @pytest.mark.parametrize("sigma", [0.40, 0.25])
    def test_narrow_gaussians_reproduce_the_ewald_energy(self, sigma):
        box, nodes = 8.0, 72
        cell = np.eye(3) * box
        positions = np.array([[0.0, 0.0, 0.0], [3.1, 2.3, 1.7]])
        charges = np.array([1.0, -1.0])           # neutral, as Ewald assumes

        solver = PeriodicPoissonSolver(shape=(nodes,) * 3, spacing=box / nodes)
        axis = np.arange(nodes) * (box / nodes)
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        density = np.zeros((nodes,) * 3)
        for (px, py, pz), charge in zip(positions, charges):
            dx = (x - px + box / 2) % box - box / 2   # nearest image
            dy = (y - py + box / 2) % box - box / 2
            dz = (z - pz + box / 2) % box - box / 2
            density += charge * (2 * np.pi * sigma ** 2) ** -1.5 * np.exp(
                -(dx * dx + dy * dy + dz * dz) / (2 * sigma ** 2))

        potential = solver.solve(density.reshape(-1)).real.reshape(density.shape)
        hartree = 0.5 * float(np.sum(density * potential)) * (box / nodes) ** 3
        # A Gaussian of charge q and width sigma carries q^2 / (2 sigma sqrt(pi)).
        self_energy = float(np.sum(charges ** 2)) / (2 * sigma * np.sqrt(np.pi))
        assert hartree - self_energy == pytest.approx(
            ewald_energy(positions, charges, cell), abs=1e-9)


class TestItIsNotTheIsolatedSolver:
    """The distinction this class exists to make."""

    def test_the_two_disagree_on_a_compact_charge(self, coordinates):
        """A localized blob feels its own images periodically and not in vacuum."""
        x, y, z = coordinates
        center = BOX / 2
        density = np.exp(-((x - center) ** 2 + (y - center) ** 2
                           + (z - center) ** 2) / (2 * 0.5 ** 2))
        flat = density.reshape(-1)
        periodic = PeriodicPoissonSolver(shape=(NODES,) * 3, spacing=BOX / NODES)
        isolated = PoissonFFTSolver(shape=(NODES,) * 3, spacing=BOX / NODES)
        difference = np.abs(periodic.solve(flat).real
                            - isolated.solve(flat).real).max()
        assert difference > 1e-3, "the periodic kernel is behaving as isolated"

    def test_a_degenerate_voxel_is_refused(self):
        with pytest.raises(ValueError, match="zero volume"):
            PeriodicPoissonSolver(shape=(4, 4, 4), step=np.zeros((3, 3)))

    def test_it_needs_a_spacing_or_a_step(self):
        with pytest.raises(ValueError, match="spacing or a step"):
            PeriodicPoissonSolver(shape=(4, 4, 4))

class TestTheNyquistPlaneIsSymmetrized:
    r"""``|G|^2`` must not change when a shear flips sign.

    ``fftfreq`` enumerates ``m`` over ``{0, .., n/2-1, -n/2, .., -1}``, so for
    **even** ``n`` it holds ``-n/2`` but not ``+n/2`` -- the same discrete
    mode.  Writing ``|G|^2 = sum_ab Q_ab m_a m_b``, the two aliases differ by
    the terms linear in the Nyquist index, which vanish for an orthogonal cell
    (``Q_ab = 0`` off-diagonal) and do not for a sheared one.  That made the
    shear stress of a cubic crystal nonzero, at 6.4e-4 eV/Angstrom^3 on a
    10x10x10 grid, where symmetry forces it to vanish.
    """

    @staticmethod
    def spectrum(n, shear):
        a = 6.0
        cell = np.array([[a, shear * a, 0.0], [shear * a, a, 0.0],
                         [0.0, 0.0, a]])
        step = np.column_stack([cell[m] / n for m in range(3)])
        return np.sort(fft_g_squared((n, n, n), step @ np.diag([n] * 3)).ravel())

    @pytest.mark.parametrize("n", [9, 10, 11, 12])
    def test_the_spectrum_is_even_in_the_shear(self, n):
        """Held for odd ``n`` all along; 0.22 and 0.32 for n = 10 and 12."""
        plus = self.spectrum(n, +1e-3)
        minus = self.spectrum(n, -1e-3)
        assert np.abs(plus - minus).max() < 1e-12

    @pytest.mark.parametrize("shape", [(8, 8, 8), (10, 12, 14), (9, 11, 13)])
    def test_an_orthogonal_cell_is_untouched(self, shape):
        """The symmetrization drops cross terms, and there are none here.

        This is why no existing energy moved: every pinned periodic number was
        computed on an orthogonal cell.
        """
        step = np.diag([0.31, 0.27, 0.23])
        cell = step @ np.diag(shape)
        reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
        axes = [np.fft.fftfreq(n) * n for n in shape]
        m1, m2, m3 = np.meshgrid(*axes, indexing="ij")
        plain = np.zeros(shape)
        for row in range(3):
            component = (reciprocal[row, 0] * m1 + reciprocal[row, 1] * m2
                         + reciprocal[row, 2] * m3)
            plain += component * component
        assert fft_g_squared(shape, cell) == pytest.approx(plain, rel=1e-12)

    def test_a_skewed_cell_really_is_changed(self):
        """Or the test above would pass for the wrong reason."""
        n = 10
        a = 6.0
        cell = np.array([[a, 0.3 * a, 0.0], [0.3 * a, a, 0.0], [0.0, 0.0, a]])
        step = np.column_stack([cell[m] / n for m in range(3)])
        full = step @ np.diag([n] * 3)
        reciprocal = 2.0 * np.pi * np.linalg.inv(full).T
        axes = [np.fft.fftfreq(k) * k for k in (n, n, n)]
        m1, m2, m3 = np.meshgrid(*axes, indexing="ij")
        plain = np.zeros((n, n, n))
        for row in range(3):
            component = (reciprocal[row, 0] * m1 + reciprocal[row, 1] * m2
                         + reciprocal[row, 2] * m3)
            plain += component * component
        assert not np.allclose(fft_g_squared((n, n, n), full), plain)
