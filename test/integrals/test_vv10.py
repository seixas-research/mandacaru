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


def _grid(shape, grid_spacing, periodic):
    return SimpleNamespace(shape=shape, dV=grid_spacing ** 3, step=np.diag([grid_spacing] * 3),
                           periodic=periodic)


def _dimer(shape, grid_spacing, separation, extra=0.0):
    """Two Ar-like Gaussian shells (and an optional perturbing one), with
    their analytic sigma = |grad n|^2."""
    axes = [(np.arange(m) - (m - 1) / 2) * grid_spacing for m in shape]
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
            assert abs(terms.energy) < 3e-4 * (3/vv10.B_R2SCAN**2)**.75/32 * n_total

    def test_it_is_the_exact_double_sum(self, monkeypatch):
        """The interpolated, spectrally applied energy against the O(N^2)
        sum of the rVV10 kernel with the same saturated q (the padding
        raised so that the kernel's images are out of the comparison)."""
        monkeypatch.setattr(vv10, "ISOLATED_PADDING", 4)
        grid_spacing, shape = 0.35, (26, 26, 26)
        n, sigma, (X, Y, Z) = _dimer(shape, grid_spacing, 3.2)
        rps = vv10.nonlocal_correlation(_grid(shape, grid_spacing, False), n, sigma)
        b, C = vv10.B_R2SCAN, vv10.C_VV10
        on = n > vv10.DENSITY_CUTOFF
        kappa = (b*1.5*np.pi*(9*np.pi)**(-1/6)) * n[on] ** (1 / 6)
        omega0 = np.sqrt(C * (sigma[on] / n[on] ** 2) ** 2
                         + 4 * np.pi * n[on] / 3)
        q_raw = omega0 / kappa / vv10.Q_CUT
        q = -vv10.Q_CUT*np.expm1(-sum(q_raw**m/m for m in range(1,13)))
        theta = n[on] * kappa ** -1.5
        points = np.stack([X.ravel()[on], Y.ravel()[on], Z.ravel()[on]], 1)
        total = 0.0
        for start in range(0, len(points), 1024):
            d2 = ((points[start:start + 1024, None]
                   - points[None, :]) ** 2).sum(-1)
            qa = q[start:start + 1024, None]
            phi = 1.0 / ((qa * d2 + 1) * (q * d2 + 1) * ((qa + q) * d2 + 2))
            total += theta[start:start + 1024] @ phi @ theta
        exact = (-0.75 * total * grid_spacing ** 6
                 + (3/b**2)**.75/32 * float(np.sum(n)) * grid_spacing ** 3)
        assert rps.energy == pytest.approx(exact, abs=2e-7)

    @pytest.mark.parametrize("periodic", [False, True])
    def test_the_partials_are_the_derivative(self, periodic):
        """d/de E[n + e dn] = int (dE/dn dn + dE/dsigma dsigma)."""
        grid_spacing, shape = 0.35, (32, 24, 24)
        grid = _grid(shape, grid_spacing, periodic)
        n, sigma, _ = _dimer(shape, grid_spacing, 3.6)
        terms = vv10.nonlocal_correlation(grid, n, sigma)
        eps = 1e-4
        # Displace the positive-density fragments. Subtracting a Gaussian in
        # the far tail would produce a negative, unphysical trial density.
        plus, minus = _dimer(shape, grid_spacing, 3.6 + eps), _dimer(shape, grid_spacing, 3.6 - eps)
        numeric = (vv10.nonlocal_correlation(grid, *plus[:2]).energy
                   - vv10.nonlocal_correlation(grid, *minus[:2]).energy) / (2 * eps)
        dn = (plus[0] - minus[0]) / (2 * eps)
        ds = (plus[1] - minus[1]) / (2 * eps)
        analytic = float(np.sum(terms.d_density * dn + terms.d_sigma * ds)) * grid_spacing ** 3
        assert analytic == pytest.approx(numeric, rel=1e-6)
