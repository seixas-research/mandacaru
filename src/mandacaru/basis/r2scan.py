# -*- coding: utf-8 -*-
# file: basis/r2scan.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""The r\ :sup:`2`\ SCAN meta-GGA, spin-unpolarized.

Furness, Kaplan, Ning, Perdew and Sun, J. Phys. Chem. Lett. **11**, 8208
(2020): the regularized-restored SCAN.  The energy per unit volume
:math:`f(\rho, \sigma, \tau)` depends on the density, :math:`\sigma =
|\nabla\rho|^2` and the kinetic-energy density :math:`\tau = \tfrac12
\sum_i f_i |\nabla\psi_i|^2` through

.. math::

    p = \frac{\sigma}{4(3\pi^2)^{2/3}\rho^{8/3}}, \qquad
    \bar\alpha = \frac{\tau - \tau_W}{\tau_{unif} + \eta\,\tau_W},

with :math:`\tau_W = \sigma/8\rho` and :math:`\tau_{unif} =
\tfrac{3}{10}(3\pi^2)^{2/3}\rho^{5/3}`.  The functional is written down once,
as :func:`energy_density`, in ``jax.numpy``; :func:`partials` takes its three
derivatives by automatic differentiation, so the potential is the derivative
of the energy by construction rather than by a second hand-derived formula.

The uniform-gas correlation it rests on is Perdew-Wang 1992 with the
constants of :mod:`mandacaru.basis.xc`, the same parameterization PBE uses
there.

Inputs violating :math:`\tau \ge \tau_W` (possible for a ``tau`` sampled on a
grid, never for an exact one) are tolerated down to :math:`\bar\alpha =`
:data:`ALPHA_FLOOR`, approached smoothly (:func:`_soft_floor`): a negative
:math:`\bar\alpha` is mapped into :math:`(\text{ALPHA\_FLOOR}, 0)` with the
value and slope continuous at zero.

Piecewise definitions (the interpolation functions switch form at
:math:`\bar\alpha = 2.5`) are evaluated with each branch's argument clamped
to its own domain, so neither branch's derivative can produce a NaN where it
is not selected.
"""

from __future__ import annotations

import numpy as np

from .xc import DENSITY_FLOOR, PW92_A, PW92_ALPHA1, PW92_BETA

#: Regularization of the iso-orbital indicator.
ETA = 0.001

# -- exchange ---------------------------------------------------------------- #
K1 = 0.065
H0X = 1.174
A1 = 4.9479
MU_AK = 10.0 / 81.0
DP2 = 0.361
#: Interpolation f_x(alpha) for alpha <= 2.5, coefficients of alpha^0..7.
CX = (1.0, -0.667, -0.4445555, -0.663086601049, 1.451297044490,
      -0.887998041597, 0.234528941479, -0.023185843322)
C2X = 0.8
DX = 1.24

# -- correlation ------------------------------------------------------------- #
B1C = 0.0285764
B2C = 0.0889
B3C = 0.125541
CHI_INF = 0.128026
CC = (1.0, -0.64, -0.4352, -1.535685604549, 3.061560252175,
      -1.915710236206, 0.516884468372, -0.051848879792)
C2C = 1.5
DC = 0.7
GAMMA = (1.0 - np.log(2.0)) / np.pi ** 2

#: Asymptote of the iso-orbital indicator below zero (see _soft_floor).  Physical
#: densities have alpha >= 0; the polynomial branch is still smooth and
#: bounded at -0.5 (f_x = 1.43).
ALPHA_FLOOR = -0.5

#: Where the interpolation functions change form.
ALPHA_SWITCH = 2.5


def _gradient_expansion_slope(coefficients) -> float:
    """``sum_i i c_i``: the slope at alpha = 1 of the polynomial branch."""
    return float(sum(i * c for i, c in enumerate(coefficients)))


#: Second-order gradient-expansion coefficients the restoration needs.
DELTA_FX2 = _gradient_expansion_slope(CX)
DELTA_FC2 = _gradient_expansion_slope(CC)
C2X_GE = -DELTA_FX2 * (1.0 - H0X)
C_ETA = 20.0 / 27.0 + 5.0 * ETA / 3.0


def _jax():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    return jax, jnp


def _soft_floor(jnp, alpha):
    """``alpha`` itself where it is >= 0; below, ``d tanh(alpha/d)`` with
    ``d = -ALPHA_FLOOR``: value, slope and curvature continuous at 0.

    A hard ``max`` would put a kink in the energy at every tail point near the
    floor, and the SCF then hops between both sides of it (an r2SCAN H2O
    oscillated at 1e-7 Ha for 200 iterations).
    """
    depth = -ALPHA_FLOOR
    negative = jnp.minimum(alpha, 0.0)
    return jnp.where(alpha >= 0.0, alpha, depth * jnp.tanh(negative / depth))


def _interpolation(jnp, alpha, coefficients, c2, d):
    """f(alpha): polynomial up to 2.5, ``-d exp(c2/(1-alpha))`` beyond."""
    low = jnp.minimum(alpha, ALPHA_SWITCH)
    high = jnp.maximum(alpha, ALPHA_SWITCH)
    poly = sum(c * low ** i for i, c in enumerate(coefficients))
    tail = -d * jnp.exp(c2 / (1.0 - high))
    return jnp.where(alpha <= ALPHA_SWITCH, poly, tail)


def _pw92(jnp, rs):
    """Perdew-Wang 1992 ``(eps_c, d eps_c / d r_s)``, unpolarized."""
    b1, b2, b3, b4 = PW92_BETA
    sq = jnp.sqrt(rs)
    q0 = -2.0 * PW92_A * (1.0 + PW92_ALPHA1 * rs)
    q1 = 2.0 * PW92_A * (b1 * sq + b2 * rs + b3 * rs * sq + b4 * rs * rs)
    dq1 = PW92_A * (b1 / sq + 2.0 * b2 + 3.0 * b3 * sq + 4.0 * b4 * rs)
    log = jnp.log1p(1.0 / q1)
    eps = q0 * log
    deps = -2.0 * PW92_A * PW92_ALPHA1 * log - q0 * dq1 / (q1 * q1 + q1)
    return eps, deps


def _energy_density(jnp, rho, sigma, tau):
    """``f = rho (e_x + e_c)`` for densities already above the floor."""
    kf2 = (3.0 * np.pi ** 2 * rho) ** (2.0 / 3.0)
    p = sigma / (4.0 * kf2 * rho * rho)
    p_safe = jnp.maximum(p, 1e-30)
    tau_w = sigma / (8.0 * rho)
    tau_unif = 0.3 * kf2 * rho
    # An exact tau is never below the von Weizsaecker bound sigma/8rho; one
    # sampled on a grid can be, and far below it in the tail, where a large
    # negative alpha sends the degree-7 interpolation polynomial to infinity.
    # The floor sits below zero on purpose: a one-orbital density has
    # tau = tau_W exactly, and a bound *at* zero would put every point of it
    # on a kink that round-off resolves, breaking dE = int v drho.
    alpha = _soft_floor(jnp, (tau - tau_w) / (tau_unif + ETA * tau_w))
    damping = jnp.exp(-p * p / DP2 ** 4)

    # -- exchange ----------------------------------------------------------- #
    ex_lda = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0) * rho ** (4.0 / 3.0)
    x = (C_ETA * C2X_GE * damping + MU_AK) * p
    h1x = 1.0 + K1 - K1 / (1.0 + x / K1)
    fx = _interpolation(jnp, alpha, CX, C2X, DX)
    gx = 1.0 - jnp.exp(-A1 / p_safe ** 0.25)
    exchange = ex_lda * (h1x + fx * (H0X - h1x)) * gx

    # -- correlation -------------------------------------------------------- #
    rs = (3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
    eps1, deps1 = _pw92(jnp, rs)
    denominator0 = 1.0 + B2C * jnp.sqrt(rs) + B3C * rs
    eps0_lda = -B1C / denominator0
    deps0_lda = B1C * (0.5 * B2C / jnp.sqrt(rs) + B3C) / denominator0 ** 2

    w1 = jnp.expm1(-eps1 / GAMMA)
    beta = 0.066725 * (1.0 + 0.1 * rs) / (1.0 + 0.1778 * rs)
    k_s2 = 4.0 * (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0) / np.pi
    t2 = sigma / (4.0 * k_s2 * rho * rho)
    y = beta / (GAMMA * w1) * t2
    delta_y = (DELTA_FC2 / (27.0 * GAMMA * w1)
               * (20.0 * rs * (deps0_lda - deps1)
                  - 45.0 * ETA * (eps0_lda - eps1))
               * p * damping)
    g = (1.0 + 4.0 * (y - delta_y)) ** -0.25
    eps1_full = eps1 + GAMMA * jnp.log1p(w1 * (1.0 - g))

    w0 = jnp.expm1(-eps0_lda / B1C)
    g_inf = (1.0 + 4.0 * CHI_INF * p) ** -0.25
    eps0_full = eps0_lda + B1C * jnp.log1p(w0 * (1.0 - g_inf))
    fc = _interpolation(jnp, alpha, CC, C2C, DC)
    correlation = rho * (eps1_full + fc * (eps0_full - eps1_full))
    return exchange + correlation


def energy_density(rho, sigma, tau) -> np.ndarray:
    r"""``f(rho, sigma, tau)``, the energy per unit volume (Hartree/Bohr^3).

    Zero below :data:`~mandacaru.basis.xc.DENSITY_FLOOR`.
    """
    return partials(rho, sigma, tau)[0]


def partials(rho, sigma, tau):
    r"""``(f, df/drho, df/dsigma, df/dtau)`` at every point.

    Points below :data:`~mandacaru.basis.xc.DENSITY_FLOOR` contribute
    nothing and have zero derivatives.
    """
    jax, jnp = _jax()
    rho = np.asarray(rho, dtype=float)
    dense = rho > DENSITY_FLOOR
    # Substitute a harmless density where the point is dropped, so neither the
    # value nor its derivatives can be NaN there (the double-where pattern).
    safe_rho = jnp.where(dense, rho, 1.0)
    safe_sigma = jnp.where(dense, np.asarray(sigma, dtype=float), 0.0)
    safe_tau = jnp.where(dense, np.asarray(tau, dtype=float), 0.3)

    def total(r, s, t):
        return jnp.sum(jnp.where(dense, _energy_density(jnp, r, s, t), 0.0))

    def pointwise(r, s, t):
        return jnp.where(dense, _energy_density(jnp, r, s, t), 0.0)

    grads = jax.grad(total, argnums=(0, 1, 2))(safe_rho, safe_sigma, safe_tau)
    f = pointwise(safe_rho, safe_sigma, safe_tau)
    return tuple(np.where(dense, np.asarray(v), 0.0) for v in (f, *grads))
