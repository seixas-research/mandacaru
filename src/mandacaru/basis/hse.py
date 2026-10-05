# -*- coding: utf-8 -*-
# file: basis/hse.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""The screened semilocal exchange of the HSE06 hybrid.

HSE06 (Heyd, Scuseria and Ernzerhof, J. Chem. Phys. **118**, 8207 (2003);
erratum **124**, 219906 (2006); Krukau *et al.*, **125**, 224106 (2006))
splits the Coulomb operator with the error function,

.. math::

    \frac1r = \underbrace{\frac{\operatorname{erfc}(\omega r)}{r}}_{\rm SR}
            + \underbrace{\frac{\operatorname{erf}(\omega r)}{r}}_{\rm LR},
    \qquad \omega = 0.11\ a_0^{-1},

and replaces a quarter of the short-range semilocal exchange by exact
exchange:

.. math::

    E_{xc}^{\rm HSE06} = E_x^{\omega\rm PBE}(0)
        - \tfrac14 E_x^{\omega\rm PBE,SR}(\omega)
        + \tfrac14 E_x^{\rm HF,SR}(\omega) + E_c^{\rm PBE} .

This module holds the semilocal pieces: the exchange energy per unit volume
:math:`f = \rho\,\varepsilon_x^{\rm unif}(\rho)\,F_x(s, \nu)` with the
enhancement factor :math:`F_x` of the screened Ernzerhof-Perdew exchange
hole, :math:`s = |\nabla\rho|/2k_F\rho` and :math:`\nu = \omega/k_F`.
:math:`F_x(s, \nu)` is the hole integrated against
:math:`\operatorname{erfc}(\nu y)`: the Gaussian parts of the hole in closed
form, the slowly decaying part through HSE's expansion of the error function
(below :data:`NU_SWITCH`) or its asymptotic form (above).  :math:`F_x(s, 0)`
is this model's *full-range* PBE exchange, which is what HSE06 puts in its
unscreened part (it differs from the PBE enhancement factor by up to ~0.2 %);
written separately (:func:`unscreened_enhancement`) so that the
:math:`\ln s` singularities of the two terms that cancel at :math:`\nu = 0`
never meet in floating point.

The full-range part (with PBE correlation) is the ``"hse06"`` kernel of
:mod:`mandacaru.basis.xc_spin`; the screened part is
:func:`short_range_partials`.  The exact-exchange part is not here: it is
the short-range two-electron tensor of
:meth:`~mandacaru.core.hamiltonian.MolecularIntegrals.short_range_two_body`
contracted in the Kohn-Sham solver (:mod:`mandacaru.algorithms.dft`).

Everything is written in ``jax.numpy`` with elementary operations only (the
library exponential integral made the compiled functional take minutes to
build); the potentials come from automatic differentiation, as for
r\ :sup:`2`\ SCAN.  Against an independent implementation of the same hole
(``test/basis/test_hse.py``), below :math:`s = 1`: :math:`\omega = 0` to
1e-14 relative; :math:`\omega = 0.11` to 1e-10 in the energy and 4e-10 in
both partials.  Above :math:`s = 1` the reduced gradient's bend is made
continuous here (:data:`S_BEND_OFFSET`), so the two differ there by up to
2e-4 relative, by design: the reference's step makes forces disagree with
the energy.
Within a few percent below :data:`NU_SWITCH` the expansion cancels
catastrophically in both and they differ by up to 5e-6 relative -- at
:math:`\omega = 0.11` that is a density below 2e-8 Bohr\ :sup:`-3`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

#: HSE06's range-separation parameter (1/Bohr), Krukau et al. 2006.
OMEGA = 0.11
#: Fraction of short-range exact exchange.
EXACT_FRACTION = 0.25

# -- the Ernzerhof-Perdew exchange hole, as parameterized for HSE --------- #
HOLE_A = 1.0161144
HOLE_B = -0.37170836
HOLE_C = -0.077215461
HOLE_D = 0.57786348
HOLE_E = -0.051955731
#: ``H(s) = (a1 s^2 + a2 s^4) / (1 + a3 s^4 + a4 s^5 + a5 s^6)``.
H_COEFFICIENTS = (0.00979681, 0.0410834, 0.187440, 0.00120824, 0.0347188)
#: ``F(s) = c1 H(s) + c2``.
F_COEFFICIENTS = (6.4753871, 0.47965830)
#: ``E G(s)`` below :data:`EG_SWITCH`: ``a0 + a1 s^2 + a2 s^4``.
EG_SERIES = (-0.02628417880, -0.07117647788, 0.08534541323)
EG_SWITCH = 0.08
#: The reduced gradient is passed through unchanged below ``S_LINEAR`` and
#: bent towards ``S_MAX`` above it, which keeps the hole normalizable;
#: ``S_FLOOR`` keeps ``H(s)`` positive.  The bend is the reference
#: implementation's, ``s - ln(1 + e^(s - S_MAX))``, raised by
#: :data:`S_BEND_OFFSET` so that it starts at ``S_LINEAR`` itself: the
#: reference's starts 5.2e-4 below it (and caps at s = 15, a second step),
#: and those steps made the energy surface a staircase a finite-difference
#: force sees (2e-3 eV/Angstrom on H2O, 1e-2 on OH, 1e-3 on displaced Si,
#: HISTORY 2026-10-04).  Below ``S_LINEAR`` the two are identical.
S_LINEAR = 1.0
S_MAX = 8.572844
S_BEND_OFFSET = float(np.log1p(np.exp(S_LINEAR - S_MAX)))
S_FLOOR = 1e-15
#: Above this ``nu`` the long-ranged part of the hole is integrated with the
#: asymptotic form of the error function.
NU_SWITCH = 14.0
#: Gaussian exponents standing in for erfc in the long-ranged part: below and
#: above :data:`NU_SWITCH`.
ERFC_EXPONENT = (1.455915450052607, 2.0)

#: ``9 / 4A``.
_NINE_FOURTH_A = 9.0 / (4.0 * HOLE_A)
#: Coefficients (``nu^1, nu^3, nu^5, nu^7``) of the erfcx term and
#: (``nu^0 ... nu^8``, even) of the exponential-integral term of the
#: long-ranged part, HSE's error-function expansion folded with ``A``.
_ERFCX_POLY = (1.7059169152930056821, -4.1622705406440396562,
               4.2174370348694648999, -1.0676080470633097775)
_E1_POLY = (-HOLE_A, 3.26865659796668475, -4.8418398881417585092,
            2.723636568586566055, -0.20524577845574895866)
#: Below this argument the exponential integral is summed from its series,
#: above it from its continued fraction (both to round-off at this switch).
_SERIES_SWITCH = 3.0
_SERIES_TERMS = 40
_FRACTION_DEPTH = 40
_EULER_GAMMA = 0.57721566490153286061


def _special():
    from jax.scipy.special import erfcx
    return erfcx


def _ein(jnp, x):
    r""":math:`\mathrm{Ein}(x) = \sum_{k\ge1} (-1)^{k+1} x^k/(k\,k!)`,
    the entire part of :math:`E_1(x) = -\gamma - \ln x + \mathrm{Ein}(x)`."""
    term = x
    total = x
    for k in range(2, _SERIES_TERMS + 1):
        term = -term * x / k
        total = total + term / k
    return total


def _continued_fraction(jnp, x):
    r""":math:`e^x E_1(x) = 1/(x + 1 - 1^2/(x + 3 - 2^2/(x + 5 - \dots)))`,
    evaluated bottom-up at a fixed depth: 4e-16 relative for
    :math:`x \ge 3`, and no overflow at any :math:`x`."""
    f = x + 2.0 * _FRACTION_DEPTH + 1.0
    for k in range(_FRACTION_DEPTH, 0, -1):
        f = x + (2.0 * k - 1.0) - k * k / f
    return 1.0 / f


def e1_scaled(jnp, x):
    r""":math:`e^x E_1(x)` for :math:`x > 0`, finite where :math:`e^x`
    alone would overflow.

    Elementary operations only: with the library exponential integral the
    compiled hybrid functional took minutes to build.
    """
    small = x < _SERIES_SWITCH
    xs = jnp.where(small, x, 1.0)
    xl = jnp.where(small, 2.0 * _SERIES_SWITCH, x)
    series = jnp.exp(xs) * (_ein(jnp, xs) - _EULER_GAMMA - jnp.log(xs))
    return jnp.where(small, series, _continued_fraction(jnp, xl))


def _e1_plus_log(jnp, x):
    r""":math:`e^x E_1(x) + \ln x`, smooth down to :math:`x \to 0`.

    With the series of :func:`_ein` the logarithm cancels analytically:
    :math:`e^x(\mathrm{Ein}(x) - \gamma) - (e^x - 1)\ln x`.
    """
    small = x < _SERIES_SWITCH
    xs = jnp.where(small, x, 1.0)
    xl = jnp.where(small, 2.0 * _SERIES_SWITCH, x)
    series = (jnp.exp(xs) * (_ein(jnp, xs) - _EULER_GAMMA)
              - jnp.expm1(xs) * jnp.log(xs))
    return jnp.where(small, series,
                     _continued_fraction(jnp, xl) + jnp.log(xl))


def _regularized_s(jnp, s):
    """The reduced gradient as the hole model takes it: continuous, the bend
    written as ``S_MAX - ln(1 + e^(S_MAX - s))`` so that it neither
    overflows nor needs a cap (module constants)."""
    bent = S_MAX - jnp.logaddexp(0.0, S_MAX - s) + S_BEND_OFFSET
    return jnp.maximum(jnp.where(s < S_LINEAR, s, bent), S_FLOOR)


def _hole(jnp, s):
    """``(s^2, s^2 H, F, E G)`` of the hole at regularized ``s``."""
    erfcx = _special()
    a1, a2, a3, a4, a5 = H_COEFFICIENTS
    s2 = s * s
    s4 = s2 * s2
    H = (a1 * s2 + a2 * s4) / (1.0 + a3 * s4 + a4 * s4 * s + a5 * s4 * s2)
    Hs = s2 * H
    F = F_COEFFICIENTS[0] * H + F_COEFFICIENTS[1]
    P0 = HOLE_D + Hs
    sq_pi = np.sqrt(np.pi)
    # E G(s) from the normalization of the hole; its small-s series below
    # EG_SWITCH, where the closed form loses digits to cancellation.
    s2_safe = jnp.where(s > EG_SWITCH, s2, 1.0)
    P0_safe = jnp.where(s > EG_SWITCH, P0, 1.0)
    Hs_safe = jnp.where(s > EG_SWITCH, Hs, 1.0)
    root = P0_safe ** 3.5
    bracket = (0.75 * np.pi
               + sq_pi * (15.0 * HOLE_E
                          + 6.0 * HOLE_C * (F * s2_safe + 1.0) * P0_safe
                          + 4.0 * HOLE_B * P0_safe ** 2
                          + 8.0 * HOLE_A * P0_safe ** 3) / (16.0 * root)
               - 0.75 * np.pi * np.sqrt(HOLE_A)
               * erfcx(jnp.sqrt(_NINE_FOURTH_A * Hs_safe)))
    closed = -16.0 / 15.0 * bracket * root / (sq_pi * s2_safe)
    series = EG_SERIES[0] + EG_SERIES[1] * s2 + EG_SERIES[2] * s4
    EG = jnp.where(s > EG_SWITCH, closed, series)
    return s2, Hs, F, EG


def _gaussian_part(jnp, nu, s2, Hs, F, EG):
    r"""The Gaussian parts of the hole against :math:`\operatorname{erfc}(\nu
    y)`, in closed form (exact for every ``nu``)."""
    B, C, D, E = HOLE_B, HOLE_C, HOLE_D, HOLE_E
    P0 = D + Hs
    value = -4.0 / 9.0 * (B * P0 ** 2 + C * D + 2.0 * E + C * Hs
                          + C * P0 * s2 * F + 2.0 * s2 * EG) / P0 ** 3
    if nu is None:
        return value
    nu2 = nu * nu
    Pn = P0 + nu2
    root = Pn ** 2.5
    value = value + nu / 9.0 * (
        4.0 * B * Pn ** 2 + 6.0 * C * D + 15.0 * E + 6.0 * C * Hs
        + 6.0 * C * nu2 + 6.0 * C * Pn * s2 * F + 15.0 * s2 * EG
    ) / (P0 * root)
    value = value + 4.0 / 9.0 * nu * nu2 * (
        C * D + 5.0 * E + C * Hs + C * nu2 + C * Pn * s2 * F + 5.0 * s2 * EG
    ) / (P0 ** 2 * root)
    value = value + 8.0 / 9.0 * nu2 * nu2 * nu * (E + s2 * EG) / (
        P0 ** 3 * root)
    return value


def unscreened_enhancement(jnp, s):
    r""":math:`F_x(s, \nu = 0)`: the hole model's full-range exchange."""
    s2, Hs, F, EG = _hole(jnp, _regularized_s(jnp, s))
    A = HOLE_A
    long_ranged = 0.5 * A * (_e1_plus_log(jnp, _NINE_FOURTH_A * Hs)
                             - np.log(_NINE_FOURTH_A)
                             - jnp.log(HOLE_D + Hs))
    return -8.0 / 9.0 * long_ranged + _gaussian_part(jnp, None, s2, Hs, F,
                                                     EG)


def screened_enhancement(jnp, s, nu):
    r""":math:`F_x^{\rm SR}(s, \nu)` for :math:`\nu > 0`: the hole
    integrated against :math:`\operatorname{erfc}(\nu y)`.

    Tends to :func:`unscreened_enhancement` as :math:`\nu \to 0` and to zero
    as :math:`\nu \to \infty`.
    """
    erfcx = _special()
    s2, Hs, F, EG = _hole(jnp, _regularized_s(jnp, s))
    A, D = HOLE_A, HOLE_D
    sq_pi = np.sqrt(np.pi)
    low = nu < NU_SWITCH

    # -- below NU_SWITCH: HSE's error-function expansion ------------------- #
    w = jnp.minimum(nu, NU_SWITCH)
    w2 = w * w
    zeta = ERFC_EXPONENT[0]
    Q = Hs + zeta * w2
    P = D + Q
    X = _NINE_FOURTH_A * Q
    odd = sum(c * w ** (2 * k + 1) for k, c in enumerate(_ERFCX_POLY))
    # The constant of the even polynomial is -A: written as (poly + A) so the
    # A/2 E1 part joins ln Q in `_e1_plus_log` (both diverge as Q -> 0).
    even = sum(c * w2 ** k for k, c in enumerate(_E1_POLY) if k)
    sqP, sqQ = jnp.sqrt(P), jnp.sqrt(Q)
    expansion = (
        0.5 * np.pi * odd * erfcx(jnp.sqrt(X))
        - 0.5 * even * e1_scaled(jnp, X)
        + 0.5 * A * (_e1_plus_log(jnp, X) - np.log(_NINE_FOURTH_A)
                     - jnp.log(P))
        - 0.5732022993364590259 * sq_pi * w / sqP
        + 0.73807311952199090995 * w2 / P
        - 1.243162299390327 * sq_pi * (-9.0 / (8.0 * sqQ)
                                       + 0.2540286 / (P * sqP)) * w * w2
        + (-1.093302940630051125 / Q + 0.49374260512735112038 / P ** 2)
        * w2 * w2
        - 0.052484962540331303985 * sq_pi
        * (3.0 * P ** 2.5 * (9.0 * Q - 2.0322288)
           + 4.12995389554944 * Q * sqQ) / (P ** 2.5 * Q * sqQ)
        * w2 * w2 * w
        + (0.25085884618821050197 / P ** 3
           + 0.007715016088131 * (-36.0 + 79.715433616529792314 * Hs) / Q ** 2)
        * w2 ** 3
        + 0.0014762353927435135389 * sq_pi
        * (-41.96505624603881896 * Q ** 2.5
           + 9.0 * P ** 3.5 * (27.0 * Q ** 2 - 6.0966864 * Q
                               + 4.12995389554944))
        / (P ** 3.5 * Q ** 2.5) * w2 ** 3 * w
        + 0.0075666704254679261017
        * (81.278266164980202635 * zeta * P ** 4 * Q
           + 3.3847844843765416574 * Q ** 3
           + 0.008401793031216 * P ** 4 * (-729.0 * Q ** 2 + 329.2210656 * Q
                                           - 297.35668047955968))
        / (P ** 4 * Q ** 3) * w2 ** 4)

    # -- above NU_SWITCH: the asymptotic form ------------------------------ #
    big = jnp.maximum(nu, NU_SWITCH)
    Q2 = Hs + ERFC_EXPONENT[1] * big * big
    asymptotic = 0.5 * A * (e1_scaled(jnp, _NINE_FOURTH_A * Q2)
                            - jnp.log(D + Q2) + jnp.log(Q2))

    long_ranged = jnp.where(low, expansion, asymptotic)
    return -8.0 / 9.0 * long_ranged + _gaussian_part(jnp, nu, s2, Hs, F, EG)


def exchange_energy_density(jnp, rho, sigma, omega: float):
    r"""Screened hole-model exchange per unit volume,
    :math:`\rho\,\varepsilon_x^{\rm unif}(\rho)\,F_x(s, \omega/k_F)`.

    ``omega = 0`` is the unscreened (full-range) exchange.  ``rho`` must be
    above the density floor and ``sigma`` non-negative.
    """
    kf = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)
    ex_unif = -3.0 * kf / (4.0 * np.pi)
    # The square root's derivative is infinite at sigma = 0: such points sit
    # on the s floor, which `_regularized_s` would select anyway.
    live = sigma > (2.0 * kf * rho * S_FLOOR) ** 2
    s = jnp.sqrt(jnp.where(live, sigma, 1.0)) / (2.0 * kf * rho)
    s = jnp.where(live, s, 0.0)
    if float(omega) == 0.0:
        enhancement = unscreened_enhancement(jnp, s)
    else:
        enhancement = screened_enhancement(jnp, s, float(omega) / kf)
    return rho * ex_unif * enhancement


@lru_cache(maxsize=None)
def _compiled_short_range(omega: float):
    from .r2scan import _jax
    jax, jnp = _jax()

    def pointwise(rho, sigma, dense):
        safe = jnp.where(dense, rho, 1.0)
        value = exchange_energy_density(jnp, safe,
                                        jnp.where(dense, sigma, 0.0), omega)
        return jnp.where(dense, value, 0.0)

    def total(rho, sigma, dense):
        return jnp.sum(pointwise(rho, sigma, dense))

    return jax.jit(pointwise), jax.jit(jax.grad(total, argnums=(0, 1)))


def short_range_partials(rho, sigma, omega: float = OMEGA):
    r"""``(f, df/drho, df/dsigma)`` of the screened exchange
    :math:`f_x^{\rm SR}(\rho, \sigma; \omega)` at every point (unpolarized).

    The semilocal exchange a screened hybrid replaces: HSE06 subtracts
    ``fraction`` times it and adds as much short-range exact exchange.
    Points below :data:`~mandacaru.basis.xc.DENSITY_FLOOR` contribute nothing.
    """
    from .xc import DENSITY_FLOOR

    rho = np.asarray(rho, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    dense = rho > DENSITY_FLOOR
    pointwise, gradient = _compiled_short_range(float(omega))
    f = np.asarray(pointwise(rho, sigma, dense))
    df_drho, df_dsigma = (np.where(dense, np.asarray(g), 0.0)
                          for g in gradient(rho, sigma, dense))
    return f, df_drho, df_dsigma
