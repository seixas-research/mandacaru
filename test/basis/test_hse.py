# -*- coding: utf-8 -*-
# file: test/basis/test_hse.py

# This code is part of Mandacaru.
# MIT License

"""The screened hole-model exchange of HSE06 (:mod:`mandacaru.basis.hse`)."""

import numpy as np
import pytest

from mandacaru.basis import hse
from mandacaru.basis.r2scan import _jax
from mandacaru.basis.xc_spin import spin_partials

jax, jnp = _jax()

#: Reference points: densities and sigma = |grad rho|^2 at reduced gradients
#: s = 0, 0.5, 1.3, 0.2, 3.0.
RHO = np.array([0.001, 0.05, 0.3, 2.0, 0.02])
SIGMA = np.array([0.0, 0.003247386729782121, 2.6094603880793112,
                  9.723306394337273, 0.010154607836794114])
#: libxc 7.0 (GGA_X_WPBEH through pyscf, recorded once): the energy per unit
#: volume and its two partials, unscreened (omega = 0) and at omega = 0.11.
LIBXC = {
    0.0: dict(
        f=[-7.38558762838078e-05, -0.014283947098389755, -0.1856221308658806,
           -1.8742852840047446, -0.006288046837605405],
        vrho=[-0.09847450171174371, -0.342581372093285, -0.6096075018207692,
              -1.2311749581992282, -0.3374302478847082],
        vsigma=[0.0, -0.22127725529444175, -0.009285529525105534,
                -0.0014153028604514332, -0.06039588820408929]),
    0.11: dict(
        f=[-2.9720256020136352e-05, -0.011325370204966038,
           -0.16707991205896358, -1.7524310598945108, -0.005064116810920185],
        vrho=[-0.04885466619928103, -0.28625870493827144,
              -0.5479121008612724, -1.170559213000167, -0.2761597010054849],
        vsigma=[0.0, -0.19659545322223956, -0.009273462663060486,
                -0.0013552464340068557, -0.06003212604464172]),
}
#: Points below s = 1, where the hole model passes s through unchanged and
#: agrees with libxc to its last digits, and above it, where the bend is
#: raised by S_BEND_OFFSET (5.2e-4) to be continuous: ~1e-4 relative there.
BELOW, ABOVE = [0, 1, 3], [2, 4]
#: libxc's HSE06 semilocal part (full-range hole-model exchange, PBE
#: correlation, minus a quarter of the short-range exchange).
LIBXC_HSE06 = [-9.136189381486298e-05, -0.013265168747538778,
               -0.14722519094807646, -1.5756973383286426,
               -0.005040686733946316]


def _partials(omega):
    def total(rho, sigma):
        return jnp.sum(hse.exchange_energy_density(jnp, rho, sigma, omega))
    f = np.asarray(hse.exchange_energy_density(
        jnp, jnp.asarray(RHO), jnp.asarray(SIGMA), omega))
    vrho, vsigma = jax.grad(total, argnums=(0, 1))(jnp.asarray(RHO),
                                                   jnp.asarray(SIGMA))
    return f, np.asarray(vrho), np.asarray(vsigma)


class TestAgainstLibxc:
    @pytest.mark.parametrize("omega", [0.0, 0.11])
    def test_the_energy_and_its_partials(self, omega):
        f, vrho, vsigma = _partials(omega)
        reference = {key: np.asarray(value)
                     for key, value in LIBXC[omega].items()}
        np.testing.assert_allclose(f[BELOW], reference["f"][BELOW],
                                   rtol=1e-9)
        np.testing.assert_allclose(vrho[BELOW], reference["vrho"][BELOW],
                                   rtol=1e-9)
        np.testing.assert_allclose(vsigma[BELOW], reference["vsigma"][BELOW],
                                   rtol=1e-8, atol=1e-14)
        np.testing.assert_allclose(f[ABOVE], reference["f"][ABOVE],
                                   rtol=2e-4)
        np.testing.assert_allclose(vrho[ABOVE], reference["vrho"][ABOVE],
                                   rtol=2e-4)

    def test_the_hse06_semilocal_part(self):
        """Full-range exchange + PBE correlation (the ``"hse06"`` spin kernel
        at zeta = 0) minus a quarter of the screened exchange.  The 1e-6 is
        the Perdew-Wang constants' last digits (libxc's are more precise),
        the same baseline as PBE."""
        q = 0.25 * SIGMA
        full = spin_partials("hse06", 0.5 * RHO, 0.5 * RHO, q, q, q)[0]
        short = hse.short_range_partials(RHO, SIGMA, hse.OMEGA)[0]
        mine = full - hse.EXACT_FRACTION * short
        reference = np.asarray(LIBXC_HSE06)
        np.testing.assert_allclose(mine[BELOW], reference[BELOW], rtol=1e-6)
        np.testing.assert_allclose(mine[ABOVE], reference[ABOVE], rtol=2e-4)

    def test_the_energy_is_continuous_where_the_reduced_gradient_bends(self):
        """The reference implementation's bend starts 5.2e-4 below s = 1; a
        force is the derivative of the energy only if the energy has no
        step, so the bend here starts at s = 1 itself."""
        kf = (3.0 * np.pi ** 2 * 0.1) ** (1.0 / 3.0)
        s = np.array([1.0 - 1e-9, 1.0 + 1e-9])
        sigma = (2.0 * kf * 0.1 * s) ** 2
        rho = np.full(2, 0.1)
        for omega in (0.0, hse.OMEGA):
            f = np.asarray(hse.exchange_energy_density(
                jnp, jnp.asarray(rho), jnp.asarray(sigma), omega))
            # The slope over 2e-9 in s is ~1e-10 relative; the old step was
            # ~1e-5 relative.
            assert abs(f[1] - f[0]) < 1e-8 * abs(f[0])


class TestLimits:
    def test_vanishing_screening_is_the_unscreened_exchange(self):
        s = jnp.asarray(np.linspace(0.0, 4.0, 41))
        full = np.asarray(hse.unscreened_enhancement(jnp, s))
        screened = np.asarray(hse.screened_enhancement(jnp, s, 1e-7 + 0 * s))
        np.testing.assert_allclose(screened, full, atol=1e-6)

    def test_strong_screening_removes_the_exchange(self):
        s = jnp.asarray(np.linspace(0.0, 4.0, 41))
        for nu in (20.0, 200.0):
            value = np.asarray(hse.screened_enhancement(jnp, s, nu + 0 * s))
            assert np.max(np.abs(value)) < 2e-3 * (20.0 / nu) ** 2

    def test_the_uniform_gas_limit(self):
        """At s = 0 the hole is the uniform gas's: F = 1 unscreened, and the
        screened factor follows the exact short-range exchange of the
        uniform gas (``a = omega / 2 k_F = nu / 2``) to the hole model's
        accuracy (measured 1.2e-3 at worst)."""
        from scipy.special import erf
        zero = jnp.zeros(1)
        assert float(hse.unscreened_enhancement(jnp, zero)[0]) == \
            pytest.approx(1.0, abs=1e-8)
        for nu in (0.01, 0.1, 0.5, 1.0, 3.0, 10.0):
            a = 0.5 * nu
            exact = 1.0 - 8.0 / 3.0 * a * (
                np.sqrt(np.pi) * erf(0.5 / a)
                + (2.0 * a - 4.0 * a ** 3) * np.exp(-0.25 / a ** 2)
                - 3.0 * a + 4.0 * a ** 3)
            model = float(hse.screened_enhancement(jnp, zero, nu)[0])
            assert model == pytest.approx(exact, abs=2e-3)

    def test_screening_only_lowers_the_exchange(self):
        s = jnp.asarray(np.linspace(0.0, 4.0, 21))
        previous = np.asarray(hse.unscreened_enhancement(jnp, s))
        for nu in (0.05, 0.2, 0.8, 3.0, 13.0, 15.0):
            value = np.asarray(hse.screened_enhancement(jnp, s, nu + 0 * s))
            assert np.all(value <= previous + 1e-12)
            previous = value


class TestPartials:
    def test_short_range_partials_are_the_derivatives(self):
        rng = np.random.default_rng(3)
        rho = 10 ** rng.uniform(-3, 0.5, 20)
        kf = (3 * np.pi ** 2 * rho) ** (1 / 3)
        sigma = (2 * kf * rho * rng.uniform(0.01, 3.0, 20)) ** 2
        f, vrho, vsigma = hse.short_range_partials(rho, sigma)
        step = 1e-6
        f_r = hse.short_range_partials(rho * (1 + step), sigma)[0]
        f_l = hse.short_range_partials(rho * (1 - step), sigma)[0]
        np.testing.assert_allclose((f_r - f_l) / (2 * step * rho), vrho,
                                   rtol=1e-6)
        f_r = hse.short_range_partials(rho, sigma * (1 + step))[0]
        f_l = hse.short_range_partials(rho, sigma * (1 - step))[0]
        np.testing.assert_allclose((f_r - f_l) / (2 * step * sigma), vsigma,
                                   rtol=1e-5)

    def test_below_the_floor_nothing_contributes(self):
        f, vrho, vsigma = hse.short_range_partials(np.array([0.0, 1e-14]),
                                                   np.array([0.0, 1e-30]))
        assert not np.any(f) and not np.any(vrho) and not np.any(vsigma)

    def test_the_gradient_derivative_is_smooth_as_s_vanishes(self):
        """The unscreened factor's two logarithms cancel analytically, so
        df/dsigma reaches its s -> 0 limit without round-off noise."""
        def vsigma(s):
            rho = jnp.full(1, 0.1)
            kf = (3 * np.pi ** 2 * 0.1) ** (1 / 3)
            sigma = jnp.full(1, (2 * kf * 0.1 * s) ** 2)
            return float(jax.grad(lambda g: jnp.sum(
                hse.exchange_energy_density(jnp, rho, g, 0.0)))(sigma)[0])
        values = [vsigma(s) for s in (1e-3, 1e-5, 1e-7, 1e-9)]
        assert np.ptp(values) < 1e-5 * abs(values[0])


class TestExponentialIntegral:
    def test_against_scipy(self):
        from scipy.special import exp1
        x = np.concatenate([np.geomspace(1e-8, 2.999, 40),
                            np.linspace(3.0, 600.0, 60)])
        reference = np.exp(np.minimum(x, 700)) * exp1(x)
        value = np.asarray(hse.e1_scaled(jnp, jnp.asarray(x)))
        np.testing.assert_allclose(value, reference, rtol=2e-14)
        combined = np.asarray(hse._e1_plus_log(jnp, jnp.asarray(x)))
        np.testing.assert_allclose(combined, reference + np.log(x),
                                   rtol=1e-12, atol=1e-13)
