# -*- coding: utf-8 -*-
# file: test/integrals/test_vv10.py

# This code is part of Mandacaru.
# MIT License

"""rVV10 nonlocal correlation (:mod:`mandacaru.integrals.vv10`): the
analytic kernel transform, the interpolated energy against the exact double
sum, its partials against finite differences, and the uniform gas."""

from types import SimpleNamespace

import numpy as np
import pytest

from mandacaru.integrals import vv10


def _grid(shape, h, periodic):
    return SimpleNamespace(shape=shape, dV=h ** 3, step=np.diag([h] * 3),
                           periodic=periodic)


def _dimer(shape, h, separation, extra=0.0):
    """Two Ar-like Gaussian shells (and an optional perturbing one), with
    their analytic sigma = |grad n|^2."""
    axes = [(np.arange(m) - (m - 1) / 2) * h for m in shape]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    n = np.zeros_like(X)
    grad = [np.zeros_like(X) for _ in range(3)]
    parts = [(cx, amp, al) for cx in (-separation / 2, separation / 2)
             for amp, al in ((6.0, 1.1), (2.0, 4.0))] + [(0.4, extra, 0.7)]
    for cx, amp, al in parts:
        g = amp * (al / np.pi) ** 1.5 * np.exp(-al * ((X - cx) ** 2 + Y ** 2
                                                       + Z ** 2))
        n += g
        for i, coordinate in enumerate((X - cx, Y, Z)):
            grad[i] += -2.0 * al * coordinate * g
    return n.ravel(), sum(c * c for c in grad).ravel(), (X, Y, Z)


class TestTheKernel:
    @pytest.mark.parametrize("a, b", [(0.05, 0.05), (0.01, 0.3), (0.2, 0.07)])
    def test_the_transform_is_the_radial_fourier_integral(self, a, b):
        """The partial-fraction closed form against a brute-force radial
        transform (an adaptive quadrature fails on the oscillatory tail).  At
        k = 0 the R^-6 tail beyond the last point, 4 pi / (3 a b (a+b)
        R^3), is added; at k > 0 it oscillates away."""
        R = np.linspace(0.0, 400.0, 2_000_001)
        for k in (0.0, 0.4, 2.0):
            numeric = np.trapezoid(4 * np.pi * R * R * vv10.kernel_real(a, b, R)
                                   * np.sinc(k * R / np.pi), R)
            if k == 0.0:
                numeric += 4 * np.pi / (3 * a * b * (a + b) * R[-1] ** 3)
            assert vv10.kernel_transform(a, b, np.array([k]))[0] == \
                pytest.approx(numeric, rel=1e-6)

    def test_it_is_finite_and_smooth_at_zero(self):
        k = np.array([0.0, 1e-9, 1e-6])
        values = vv10.kernel_transform(0.01, 0.3, k)
        assert np.all(np.isfinite(values))
        assert values[1] == pytest.approx(values[0], rel=1e-8)

    def test_the_saturation_follows_q_then_caps_it(self):
        h, dh = vv10.saturate(np.array([1e-4, 0.01, 50.0]))
        assert h[0] == pytest.approx(1e-4, rel=1e-3) and dh[0] == pytest.approx(1.0, rel=1e-3)
        assert h[2] == pytest.approx(vv10.Q_CUT, rel=1e-9)

    def test_the_cardinal_splines_partition_unity(self):
        p, _dp = vv10.cardinal(np.geomspace(2e-4, 0.4, 50))
        assert np.allclose(p.sum(axis=0), 1.0, atol=1e-12)


class TestTheEnergy:
    def test_the_uniform_gas_has_none(self):
        """beta = (3/b^2)^(3/4)/32 cancels the kernel for a uniform
        density: what is left is the q interpolation's error (~1e-4 of
        beta N with :data:`~mandacaru.integrals.vv10.N_Q` = 16)."""
        for n0 in (0.001, 0.1):
            terms = vv10.nonlocal_correlation(_grid((12, 12, 12), 0.4, True),
                                              np.full(12 ** 3, n0),
                                              np.zeros(12 ** 3))
            n_total = n0 * 12 ** 3 * 0.4 ** 3
            assert abs(terms.energy) < 3e-4 * vv10.beta(vv10.B_R2SCAN) * n_total

    def test_it_is_the_exact_double_sum(self, monkeypatch):
        """The interpolated, spectrally applied energy against the O(N^2)
        sum of the rVV10 kernel with the same saturated q (the padding
        raised so that the kernel's images are out of the comparison)."""
        monkeypatch.setattr(vv10, "ISOLATED_PADDING", 4)
        h, shape = 0.35, (26, 26, 26)
        n, sigma, (X, Y, Z) = _dimer(shape, h, 3.2)
        rps = vv10.nonlocal_correlation(_grid(shape, h, False), n, sigma)
        b, C = vv10.B_R2SCAN, vv10.C_VV10
        on = n > vv10.DENSITY_CUTOFF
        kappa = vv10.kappa_prefactor(b) * n[on] ** (1 / 6)
        omega0 = np.sqrt(C * (sigma[on] / n[on] ** 2) ** 2
                         + 4 * np.pi * n[on] / 3)
        q, _ = vv10.saturate(omega0 / kappa)
        theta = n[on] * kappa ** -1.5
        points = np.stack([X.ravel()[on], Y.ravel()[on], Z.ravel()[on]], 1)
        total = 0.0
        for start in range(0, len(points), 1024):
            d2 = ((points[start:start + 1024, None]
                   - points[None, :]) ** 2).sum(-1)
            qa = q[start:start + 1024, None]
            phi = 1.0 / ((qa * d2 + 1) * (q * d2 + 1) * ((qa + q) * d2 + 2))
            total += theta[start:start + 1024] @ phi @ theta
        exact = (-0.75 * total * h ** 6
                 + vv10.beta(b) * float(np.sum(n)) * h ** 3)
        assert rps.energy == pytest.approx(exact, abs=2e-7)

    @pytest.mark.parametrize("periodic", [False, True])
    def test_the_partials_are_the_derivative(self, periodic):
        """d/de E[n + e dn] = int (dE/dn dn + dE/dsigma dsigma)."""
        h, shape = 0.35, (32, 24, 24)
        grid = _grid(shape, h, periodic)
        n, sigma, _ = _dimer(shape, h, 3.6)
        terms = vv10.nonlocal_correlation(grid, n, sigma)
        eps = 1e-4
        plus, minus = _dimer(shape, h, 3.6, eps), _dimer(shape, h, 3.6, -eps)
        numeric = (vv10.nonlocal_correlation(grid, *plus[:2]).energy
                   - vv10.nonlocal_correlation(grid, *minus[:2]).energy) / (2 * eps)
        dn = (plus[0] - minus[0]) / (2 * eps)
        ds = (plus[1] - minus[1]) / (2 * eps)
        analytic = float(np.sum(terms.d_density * dn + terms.d_sigma * ds)) * h ** 3
        assert analytic == pytest.approx(numeric, rel=1e-6)
