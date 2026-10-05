# -*- coding: utf-8 -*-
# file: test/integrals/test_poisson.py

# This code is part of Mandacaru.
# MIT License

"""The isolated Coulomb kernels of :mod:`mandacaru.integrals.poisson`: bare
and screened, against the closed-form interaction of two Gaussians."""

import numpy as np
import pytest
from scipy.special import erf

from scipy import fft as sfft

from mandacaru.integrals.poisson import (SPLIT_REACH, PoissonFFTSolver,
                                         short_range_coulomb_kernel)

H = 0.25
N = 40


def _gaussian(exponent, center):
    """A unit Gaussian charge on an N^3 grid of spacing H (Bohr)."""
    axis = (np.arange(N) - (N - 1) / 2) * H
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    r2 = (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
    return ((exponent / np.pi) ** 1.5 * np.exp(-exponent * r2)).reshape(-1)


def _interaction(solver, a, b):
    return float(np.real(np.sum(a * solver.solve(b))) * H ** 3)


class TestTheIsolatedSolver:
    @pytest.mark.parametrize("omega", [0.11, 0.6, 2.0])
    @pytest.mark.parametrize("a, b, distance", [(3.0, 2.0, 1.5),
                                                (1.0, 1.0, 0.0),
                                                (4.0, 4.0, 0.0)])
    def test_two_gaussians_interact_as_the_closed_form(self, omega, a, b,
                                                       distance):
        r"""Unit Gaussians of exponents ``a`` and ``b`` a distance ``R``
        apart interact through :math:`\operatorname{erfc}(\omega r)/r` as
        :math:`[\operatorname{erf}(\sqrt\mu R) - \operatorname{erf}(\sqrt p
        R)]/R`, :math:`\mu = ab/(a+b)`, :math:`1/p = 1/a + 1/b + 1/\omega^2`
        (:math:`2\sqrt{\mu/\pi} - 2\sqrt{p/\pi}` at ``R = 0``) -- to
        1e-8.  omega = 2 lies above the split, where the whole kernel is
        spectral."""
        rho_a = _gaussian(a, [0.0, 0.0, -0.5 * distance])
        rho_b = _gaussian(b, [0.0, 0.0, 0.5 * distance])
        solver = PoissonFFTSolver(N, spacing=H, omega=omega)
        assert (solver.split == omega) == (omega == 2.0)
        mu = a * b / (a + b)
        p = 1.0 / (1.0 / a + 1.0 / b + 1.0 / omega ** 2)
        exact = (2.0 * (np.sqrt(mu) - np.sqrt(p)) / np.sqrt(np.pi)
                 if distance == 0.0
                 else (erf(np.sqrt(mu) * distance)
                       - erf(np.sqrt(p) * distance)) / distance)
        assert _interaction(solver, rho_a, rho_b) == pytest.approx(exact,
                                                                   rel=1e-8)

    @pytest.mark.parametrize("a, b, distance", [(3.0, 2.0, 1.5),
                                                (1.0, 1.0, 0.0),
                                                (4.0, 4.0, 0.0)])
    def test_the_bare_kernel_is_exact(self, a, b, distance):
        """The default (omega = 0): the bare Coulomb interaction of two
        Gaussians, erf(sqrt(mu) R)/R (2 sqrt(mu/pi) at R = 0), to 1e-8,
        self interaction included."""
        rho_a = _gaussian(a, [0.0, 0.0, -0.5 * distance])
        rho_b = _gaussian(b, [0.0, 0.0, 0.5 * distance])
        mu = a * b / (a + b)
        exact = (2.0 * np.sqrt(mu / np.pi) if distance == 0.0
                 else erf(np.sqrt(mu) * distance) / distance)
        solver = PoissonFFTSolver(N, spacing=H)
        assert solver.omega == 0.0
        assert _interaction(solver, rho_a, rho_b) == pytest.approx(exact,
                                                                   rel=1e-8)

    def test_the_padding_does_not_depend_on_omega(self):
        """Nothing long-ranged is spectral, so a small omega costs what the
        bare kernel does: 2N - 1, rounded up to an FFT-friendly length."""
        solver = PoissonFFTSolver(N, spacing=H, omega=0.01)
        assert solver.L == PoissonFFTSolver(N, spacing=H).L
        assert solver.L == (sfft.next_fast_len(2 * N - 1),) * 3

    def test_a_small_grid_is_padded_past_the_spectral_reach(self):
        """On a grid shorter than the spectral part's reach (~14 nodes),
        2N - 1 would put its images inside the box."""
        solver = PoissonFFTSolver(6, spacing=H)
        reach = int(np.ceil(SPLIT_REACH / (solver.split * H)))
        assert all(L >= 6 + reach > 2 * 6 - 1 for L in solver.L)

    def test_omega_must_not_be_negative(self):
        with pytest.raises(ValueError, match="omega"):
            PoissonFFTSolver(8, spacing=H, omega=-0.1)


class TestKernels:
    def test_the_short_range_kernel_in_reciprocal_space(self):
        omega = 0.11
        g2 = np.array([0.0, 1e-12, 0.01, 1.0, 100.0])
        kernel = short_range_coulomb_kernel(g2, omega)
        assert kernel[0] == pytest.approx(np.pi / omega ** 2)
        assert kernel[1] == pytest.approx(np.pi / omega ** 2, rel=1e-9)
        expected = 4 * np.pi / g2[2:] * (1 - np.exp(-g2[2:] / (4 * omega ** 2)))
        np.testing.assert_allclose(kernel[2:], expected, rtol=1e-14)
