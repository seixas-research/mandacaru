# -*- coding: utf-8 -*-
# file: basis/xc_spin.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Spin-polarized exchange-correlation: LDA, PBE, r\ :sup:`2`\ SCAN and
the full-range semilocal part of HSE06.

The energy per unit volume as a function of the two spin densities, their
gradients' contractions :math:`\sigma_{\uparrow\uparrow} =
|\nabla\rho_\uparrow|^2`, :math:`\sigma_{\uparrow\downarrow} =
\nabla\rho_\uparrow\cdot\nabla\rho_\downarrow`,
:math:`\sigma_{\downarrow\downarrow}`, and (r\ :sup:`2`\ SCAN) the two
kinetic-energy densities.  Each functional is written once, in ``jax.numpy``,
and every partial derivative is taken by automatic differentiation of a
compiled ``value_and_grad`` -- so the potentials are the derivatives of the
energy by construction, and one trace serves a whole SCF.

**Exchange** of every functional follows from the unpolarized one by the
exact spin scaling :math:`E_x[\rho_\uparrow, \rho_\downarrow] =
\tfrac12\bigl(E_x[2\rho_\uparrow] + E_x[2\rho_\downarrow]\bigr)` -- with
:math:`\sigma \to 4\sigma_{\sigma\sigma}`, :math:`\tau \to 2\tau_\sigma` and
the relativistic factor of :mod:`mandacaru.basis.xc` evaluated at
:math:`2\rho_\sigma`.

**Correlation** depends on the total density and the polarization
:math:`\zeta = (\rho_\uparrow - \rho_\downarrow)/\rho`:

- LDA: Perdew-Zunger (1981), continuous at :math:`r_s = 1`
  (:data:`~mandacaru.basis.atomic_solver.PZ_PARA`), its paramagnetic and
  ferromagnetic fits joined
  by the von Barth-Hedin interpolation
  :math:`f(\zeta) = [(1+\zeta)^{4/3} + (1-\zeta)^{4/3} - 2]/(2^{4/3} - 2)`;
- PBE: Perdew-Wang (1992) -- the paramagnetic and ferromagnetic fits and the
  spin stiffness :math:`\alpha_c`, :math:`\varepsilon_c =
  \varepsilon_0 + \alpha_c f(\zeta)(1-\zeta^4)/f''(0) +
  (\varepsilon_1 - \varepsilon_0) f(\zeta)\zeta^4` -- plus the gradient term
  with :math:`\phi(\zeta) = [(1+\zeta)^{2/3} + (1-\zeta)^{2/3}]/2`;
- r\ :sup:`2`\ SCAN: the same uniform-gas limit, :math:`G_c(\zeta)` on the
  single-orbital limit, and :math:`d_s(\zeta) = [(1+\zeta)^{5/3} +
  (1-\zeta)^{5/3}]/2` in the iso-orbital indicator;
- ``"hse06"``: PBE correlation and the hole-model full-range exchange of
  :mod:`mandacaru.basis.hse` -- the part of HSE06 that is not screened.  The
  short-range part (semilocal and exact) is added by the caller
  (:mod:`mandacaru.integrals.exchange_correlation`).

At :math:`\zeta = 0` every function reduces to the unpolarized one of
:mod:`mandacaru.basis.xc` / :mod:`mandacaru.basis.r2scan` with the same
constants (``test/basis/test_xc_spin.py`` checks it to round-off).  At full
polarization :math:`(1 \pm \zeta)^{-1/3}` diverges; :math:`\zeta` is scaled
by :math:`1 -` :data:`ZETA_MARGIN` (not clipped, which would cut the empty
channel's potential), and a channel below
:data:`~mandacaru.basis.xc.DENSITY_FLOOR` contributes no exchange and gets
zero derivatives.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import hse
from .atomic_solver import PZ_FERRO, PZ_PARA
from . import r2scan as r2
from .xc import (DENSITY_FLOOR, PBE_BETA, PBE_GAMMA, PBE_KAPPA, PBE_MU,
                 PW92_A, PW92_ALPHA1, PW92_BETA, RELATIVISTIC_SERIES_BETA)
from .relativity import SPEED_OF_LIGHT

#: How far inside :math:`\pm1` the polarization is held.
ZETA_MARGIN = 1e-12

# Perdew-Wang (1992): (A, alpha1, beta1..beta4) of the paramagnetic and
# ferromagnetic fits and of minus the spin stiffness.
PW_PARA = (PW92_A, PW92_ALPHA1) + tuple(PW92_BETA)
PW_FERRO = (0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517)
PW_STIFFNESS = (0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671)
F_SECOND = 1.709921

FUNCTIONALS = ("lda", "pbe", "r2scan", "hse06")


def _jax():
    return r2._jax()


# --------------------------------------------------------------------------- #
# Spin interpolations.
# --------------------------------------------------------------------------- #

def _f_zeta(z):
    return ((1.0 + z) ** (4.0 / 3.0) + (1.0 - z) ** (4.0 / 3.0) - 2.0) \
        / (2.0 ** (4.0 / 3.0) - 2.0)


def _phi(z):
    return 0.5 * ((1.0 + z) ** (2.0 / 3.0) + (1.0 - z) ** (2.0 / 3.0))


def _ds(z):
    return 0.5 * ((1.0 + z) ** (5.0 / 3.0) + (1.0 - z) ** (5.0 / 3.0))


def _dx(z):
    return 0.5 * ((1.0 + z) ** (4.0 / 3.0) + (1.0 - z) ** (4.0 / 3.0))


# --------------------------------------------------------------------------- #
# Uniform-gas correlation.
# --------------------------------------------------------------------------- #

def _pz(jnp, rs, params):
    gamma, b1, b2, a, b, c, d = params
    high = jnp.minimum(rs, 1.0)
    low = jnp.maximum(rs, 1.0)
    log = jnp.log(high)
    value_high = a * log + b + c * high * log + d * high
    value_low = gamma / (1.0 + b1 * jnp.sqrt(low) + b2 * low)
    return jnp.where(rs < 1.0, value_high, value_low)


def _pw(jnp, rs, params):
    A, a1, b1, b2, b3, b4 = params
    sq = jnp.sqrt(rs)
    q1 = 2.0 * A * (b1 * sq + b2 * rs + b3 * rs * sq + b4 * rs * rs)
    return -2.0 * A * (1.0 + a1 * rs) * jnp.log1p(1.0 / q1)


def _pz_polarized(jnp, rs, z):
    para, ferro = _pz(jnp, rs, PZ_PARA), _pz(jnp, rs, PZ_FERRO)
    return para + _f_zeta(z) * (ferro - para)


def _pw_polarized(jnp, rs, z):
    para = _pw(jnp, rs, PW_PARA)
    ferro = _pw(jnp, rs, PW_FERRO)
    stiffness = -_pw(jnp, rs, PW_STIFFNESS)
    f, z4 = _f_zeta(z), z ** 4
    return (para + stiffness * f * (1.0 - z4) / F_SECOND
            + (ferro - para) * f * z4)


# --------------------------------------------------------------------------- #
# Unpolarized exchange, written once for the spin scaling.
# --------------------------------------------------------------------------- #

def _relativistic_factor(jnp, rho):
    """:math:`\\Phi_E(\\rho)` of :func:`~mandacaru.basis.xc.
    relativistic_exchange_factors`, in ``jax``."""
    beta = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0) / SPEED_OF_LIGHT
    small = beta < RELATIVISTIC_SERIES_BETA
    b = jnp.maximum(beta, RELATIVISTIC_SERIES_BETA)
    root = jnp.sqrt(1.0 + b * b)
    F = root / b - jnp.arcsinh(b) / (b * b)
    return jnp.where(small, 1.0 - (2.0 / 3.0) * beta ** 2
                     + (2.0 / 5.0) * beta ** 4, 1.0 - 1.5 * F * F)


def _exchange_lda(jnp, rho, sigma, tau, relativistic):
    f = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0) * rho ** (4.0 / 3.0)
    return f * _relativistic_factor(jnp, rho) if relativistic else f


def _exchange_pbe(jnp, rho, sigma, tau, relativistic):
    kf = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)
    ex_unif = -3.0 * kf / (4.0 * np.pi)
    s2 = sigma / (4.0 * kf * kf * rho * rho)
    fx = 1.0 + PBE_KAPPA - PBE_KAPPA / (1.0 + PBE_MU * s2 / PBE_KAPPA)
    f = rho * ex_unif * fx
    return f * _relativistic_factor(jnp, rho) if relativistic else f


def _exchange_r2scan(jnp, rho, sigma, tau, relativistic):
    kf2 = (3.0 * np.pi ** 2 * rho) ** (2.0 / 3.0)
    p = sigma / (4.0 * kf2 * rho * rho)
    p_safe = jnp.maximum(p, 1e-30)
    tau_w = sigma / (8.0 * rho)
    tau_unif = 0.3 * kf2 * rho
    alpha = r2._soft_floor(jnp, (tau - tau_w) / (tau_unif + r2.ETA * tau_w))
    damping = jnp.exp(-p * p / r2.DP2 ** 4)
    ex_lda = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0) * rho ** (4.0 / 3.0)
    x = (r2.C_ETA * r2.C2X_GE * damping + r2.MU_AK) * p
    h1x = 1.0 + r2.K1 - r2.K1 / (1.0 + x / r2.K1)
    fx = r2._interpolation(jnp, alpha, r2.CX, r2.C2X, r2.DX)
    gx = 1.0 - jnp.exp(-r2.A1 / p_safe ** 0.25)
    return ex_lda * (h1x + fx * (r2.H0X - h1x)) * gx


def _exchange_hse06(jnp, rho, sigma, tau, relativistic):
    return hse.exchange_energy_density(jnp, rho, sigma, 0.0)


_EXCHANGE = {"lda": _exchange_lda, "pbe": _exchange_pbe,
             "r2scan": _exchange_r2scan, "hse06": _exchange_hse06}


# --------------------------------------------------------------------------- #
# Correlation.
# --------------------------------------------------------------------------- #

def _correlation_lda(jnp, rho, z, sigma, tau):
    rs = (3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
    return rho * _pz_polarized(jnp, rs, z)


def _correlation_pbe(jnp, rho, z, sigma, tau):
    rs = (3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
    ec = _pw_polarized(jnp, rs, z)
    phi = _phi(z)
    phi3 = phi ** 3
    kf = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)
    ks = jnp.sqrt(4.0 * kf / np.pi)
    t2 = sigma / (4.0 * phi * phi * ks * ks * rho * rho)
    ratio = PBE_BETA / PBE_GAMMA
    A = ratio / jnp.expm1(-ec / (PBE_GAMMA * phi3))
    At2 = A * t2
    H = PBE_GAMMA * phi3 * jnp.log1p(
        ratio * t2 * (1.0 + At2) / (1.0 + At2 + At2 * At2))
    return rho * (ec + H)


def _correlation_r2scan(jnp, rho, z, sigma, tau):
    kf2 = (3.0 * np.pi ** 2 * rho) ** (2.0 / 3.0)
    p = sigma / (4.0 * kf2 * rho * rho)
    tau_w = sigma / (8.0 * rho)
    tau_unif = 0.3 * kf2 * rho
    ds = _ds(z)
    alpha = r2._soft_floor(jnp, (tau - tau_w)
                           / (tau_unif * ds + r2.ETA * tau_w))
    damping = jnp.exp(-p * p / r2.DP2 ** 4)
    rs = (3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
    phi = _phi(z)
    phi3 = phi ** 3
    # The uniform gas and its r_s slope (autodiff of the polarized fit).
    eps1 = _pw_polarized(jnp, rs, z)
    jax, _jnp = _jax()
    deps1 = jax.grad(lambda r: jnp.sum(_pw_polarized(jnp, r, z)))(rs)
    gc = (1.0 - 2.3631 * (_dx(z) - 1.0)) * (1.0 - z ** 12)
    denominator0 = 1.0 + r2.B2C * jnp.sqrt(rs) + r2.B3C * rs
    eps0_lda = -r2.B1C / denominator0
    deps0_lda = r2.B1C * (0.5 * r2.B2C / jnp.sqrt(rs) + r2.B3C) \
        / denominator0 ** 2
    w1 = jnp.expm1(-eps1 / (r2.GAMMA * phi3))
    beta = 0.066725 * (1.0 + 0.1 * rs) / (1.0 + 0.1778 * rs)
    k_s2 = 4.0 * (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0) / np.pi
    t2 = sigma / (4.0 * phi * phi * k_s2 * rho * rho)
    y = beta / (r2.GAMMA * w1) * t2
    delta_y = (r2.DELTA_FC2 / (27.0 * r2.GAMMA * ds * phi3 * w1)
               * (20.0 * rs * (deps0_lda * gc - deps1)
                  - 45.0 * r2.ETA * (eps0_lda * gc - eps1))
               * p * damping)
    g = (1.0 + 4.0 * (y - delta_y)) ** -0.25
    eps1_full = eps1 + r2.GAMMA * phi3 * jnp.log1p(w1 * (1.0 - g))
    w0 = jnp.expm1(-eps0_lda / r2.B1C)
    g_inf = (1.0 + 4.0 * r2.CHI_INF * p) ** -0.25
    eps0_full = (eps0_lda + r2.B1C * jnp.log1p(w0 * (1.0 - g_inf))) * gc
    fc = r2._interpolation(jnp, alpha, r2.CC, r2.C2C, r2.DC)
    return rho * (eps1_full + fc * (eps0_full - eps1_full))


_CORRELATION = {"lda": _correlation_lda, "pbe": _correlation_pbe,
                "r2scan": _correlation_r2scan, "hse06": _correlation_pbe}


# --------------------------------------------------------------------------- #
# The energy density and its partials.
# --------------------------------------------------------------------------- #

def _energy_density(jnp, functional, relativistic, up, dn, suu, sud, sdd,
                    tu, td):
    """``f`` at every point; channels and totals below the floor contribute
    nothing (the inputs there are already replaced by harmless values)."""
    exchange = _EXCHANGE[functional]
    live_u, live_d = up > DENSITY_FLOOR, dn > DENSITY_FLOOR
    safe_u = jnp.where(live_u, up, 1.0)
    safe_d = jnp.where(live_d, dn, 1.0)
    fx = 0.5 * (
        jnp.where(live_u, exchange(jnp, 2.0 * safe_u,
                                   4.0 * jnp.where(live_u, suu, 0.0),
                                   2.0 * jnp.where(live_u, tu, 0.3),
                                   relativistic), 0.0)
        + jnp.where(live_d, exchange(jnp, 2.0 * safe_d,
                                     4.0 * jnp.where(live_d, sdd, 0.0),
                                     2.0 * jnp.where(live_d, td, 0.3),
                                     relativistic), 0.0))
    total = up + dn
    live = total > DENSITY_FLOOR
    rho = jnp.where(live, total, 1.0)
    # Scaled rather than clipped: a clip has zero derivative, and the empty
    # channel of a fully polarized region would lose its correlation
    # potential (dE/drho_down is finite there even though rho_down = 0).
    z = (1.0 - ZETA_MARGIN) * jnp.where(live, (up - dn) / rho, 0.0)
    sigma = jnp.where(live, suu + 2.0 * sud + sdd, 0.0)
    tau = jnp.where(live, tu + td, 0.3)
    fc = jnp.where(live, _CORRELATION[functional](jnp, rho, z, sigma, tau),
                   0.0)
    return fx + fc


@lru_cache(maxsize=None)
def _compiled(functional: str, relativistic: bool):
    jax, jnp = _jax()

    def total(*args):
        return jnp.sum(_energy_density(jnp, functional, relativistic, *args))

    gradient = jax.jit(jax.grad(total, argnums=tuple(range(7))))
    pointwise = jax.jit(lambda *args: _energy_density(
        jnp, functional, relativistic, *args))
    return pointwise, gradient


def spin_partials(functional, rho_up, rho_dn, sigma_uu=None, sigma_ud=None,
                  sigma_dd=None, tau_up=None, tau_dn=None,
                  relativistic: bool = False):
    r"""``(f, df/drho_up, df/drho_dn, df/dsigma_uu, df/dsigma_ud,
    df/dsigma_dd, df/dtau_up, df/dtau_dn)`` at every point.

    The gradient and kinetic-energy entries are zero (and ignored) for the
    functionals that do not use them.  ``relativistic`` applies the
    relativistic exchange factor (LDA and PBE).
    """
    key = str(functional).strip().lower()
    if key not in FUNCTIONALS:
        raise ValueError(f"unknown functional {functional!r}; "
                         f"available: {FUNCTIONALS}")
    up = np.asarray(rho_up, dtype=float)
    dn = np.asarray(rho_dn, dtype=float)
    zero = np.zeros_like(up)

    def array(value, default):
        return default if value is None else np.asarray(value, dtype=float)

    args = (np.maximum(up, 0.0), np.maximum(dn, 0.0),
            array(sigma_uu, zero), array(sigma_ud, zero),
            array(sigma_dd, zero), array(tau_up, zero), array(tau_dn, zero))
    if key == "lda":
        args = args[:2] + (zero, zero, zero, zero, zero)
    elif key in ("pbe", "hse06"):
        args = args[:5] + (zero, zero)
    pointwise, gradient = _compiled(key, bool(relativistic and
                                              key in ("lda", "pbe")))
    f = np.asarray(pointwise(*args))
    partials = [np.asarray(g) for g in gradient(*args)]
    return (f, *partials)
