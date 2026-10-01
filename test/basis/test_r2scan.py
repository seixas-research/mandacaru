# -*- coding: utf-8 -*-
# file: test/basis/test_r2scan.py

# This code is part of Mandacaru.
# MIT License

"""The r2SCAN meta-GGA, pointwise."""

import numpy as np
import pytest

from mandacaru.basis import r2scan
from mandacaru.basis.atomic_solver import lda_exchange
from mandacaru.basis.xc import pw92_correlation

#: ``(rho, p, alpha) -> f`` from libxc 7 (MGGA_X_R2SCAN + MGGA_C_R2SCAN),
#: evaluated independently of this code.  libxc's Perdew-Wang constant
#: A = 0.0310907 differs from ours (0.031091) in the sixth digit, hence the
#: tolerance.
LIBXC = [
    (0.1, 0.05, 1.0, -0.03936177051867193),
    (1.0, 0.3, 0.5, -0.8503809695043372),
    (0.01, 1.0, 2.0, -0.0017940471007450876),
    (2.0, 0.01, 3.0, -1.7754644415342866),
]


def _state(rho, p, alpha):
    """``(sigma, tau)`` with reduced gradient ``p`` and indicator ``alpha``."""
    rho = np.asarray(rho, dtype=float)
    kf2 = (3.0 * np.pi ** 2 * rho) ** (2.0 / 3.0)
    sigma = p * 4.0 * kf2 * rho * rho
    tau_w = sigma / (8.0 * rho)
    tau = tau_w + alpha * (0.3 * kf2 * rho + r2scan.ETA * tau_w)
    return sigma, tau


class TestLimits:
    def test_the_uniform_gas_is_lda_exchange_plus_pw92(self):
        """p = 0 and alpha = 1: the enhancement is 1 and H vanishes."""
        rho = np.array([0.01, 0.1, 1.0, 10.0])
        sigma, tau = _state(rho, 0.0, 1.0)
        f = r2scan.energy_density(rho, sigma, tau)
        e_x, _vx = lda_exchange(rho)
        e_c, _vc = pw92_correlation(rho)
        assert np.allclose(f, rho * (e_x + e_c), rtol=1e-10)

    def test_a_tau_below_the_bound_stays_finite(self):
        rho = np.array([1e-6, 1e-3, 0.1])
        sigma, tau = _state(rho, 0.5, 0.0)
        values = r2scan.partials(rho, sigma, 0.5 * tau)
        assert all(np.all(np.isfinite(v)) for v in values)

    def test_below_the_floor_nothing_contributes(self):
        values = r2scan.partials(np.array([0.0, 1e-14]), np.zeros(2),
                                 np.zeros(2))
        assert all(np.all(v == 0.0) for v in values)


class TestPartials:
    def test_they_are_the_derivatives_of_the_energy(self):
        rng = np.random.default_rng(5)
        rho = 10 ** rng.uniform(-3, 1, 50)
        sigma, tau = _state(rho, 10 ** rng.uniform(-3, 0, 50),
                            rng.uniform(0.0, 3.0, 50))
        point = [rho, sigma, tau]
        analytic = r2scan.partials(*point)[1:]
        for index, derivative in enumerate(analytic):
            step = 1e-6 * point[index]
            plus, minus = list(point), list(point)
            plus[index] = point[index] + step
            minus[index] = point[index] - step
            numeric = (r2scan.energy_density(*plus)
                       - r2scan.energy_density(*minus)) / (2 * step)
            assert np.allclose(numeric, derivative, rtol=1e-5, atol=1e-10)


class TestAgainstAnIndependentImplementation:
    @pytest.mark.parametrize("rho, p, alpha, expected", LIBXC)
    def test_libxc_values(self, rho, p, alpha, expected):
        sigma, tau = _state(np.array([rho]), p, alpha)
        f = r2scan.energy_density(np.array([rho]), sigma, tau)[0]
        assert f == pytest.approx(expected, rel=2e-6)
