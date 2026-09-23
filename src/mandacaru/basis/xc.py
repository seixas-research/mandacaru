# -*- coding: utf-8 -*-
# file: basis/xc.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Exchange-correlation functionals for the spherical radial atom.

:mod:`mandacaru.basis.atomic_solver` carries the local density approximation it
has always used -- Slater exchange plus Perdew-Zunger (1981) correlation.  This
module adds the **generalized gradient approximation**, so a pseudopotential can
be generated from a GGA reference atom rather than an LDA one, and supplies the
dispatcher both paths go through (:func:`xc_potential`).

What is here
------------

- :func:`pw92_correlation` -- Perdew-Wang (1992) uniform-gas correlation.  PBE is
  built on it, *not* on Perdew-Zunger: the two parameterize the same Ceperley
  -Alder data and agree to about 1 mHa per electron, but PBE's :math:`H` term is
  defined with respect to PW92 and using PZ81 instead would leave the
  :math:`\nabla\rho\to0` limit inconsistent with the functional's own
  construction.
- :func:`pbe_exchange`, :func:`pbe_correlation` -- Perdew, Burke and Ernzerhof,
  Phys. Rev. Lett. **77**, 3865 (1996), spin-unpolarized.
- :func:`xc_potential` -- ``(e_xc, v_xc)`` for either functional, on a radial
  grid.

The gradient-corrected potential
--------------------------------

A GGA energy density depends on the density *and* its gradient, so its
functional derivative carries a divergence:

.. math::

    v_{xc} = \frac{\partial f}{\partial\rho}
           - \nabla\cdot\frac{\partial f}{\partial\nabla\rho} ,
    \qquad f = \rho\,\varepsilon_{xc}(\rho, \nabla\rho) .

For a spherical density the divergence is radial, so with
:math:`\sigma = d\rho/dr`

.. math::

    v_{xc} = \frac{\partial f}{\partial\rho}
           - \frac{1}{r^2}\frac{d}{dr}\Big(r^2\frac{\partial f}{\partial\sigma}\Big).

**The two partial derivatives are taken numerically**, by central differences of
:func:`xc_energy_density` in each of its two arguments, and only the radial
derivative of the resulting flux is taken on the grid.  Differentiating the
closed forms of :math:`F_x(s)` and :math:`H(r_s, t)` by hand is where a GGA
implementation normally goes wrong -- a dropped chain-rule factor gives a
potential that is not the derivative of the energy, which shows up as an SCF
that converges to the wrong answer rather than as a crash.  Doing it this way
makes the potential correct *by construction* given the energy, and
``test/basis/test_xc.py`` certifies it directly: for a random perturbation
:math:`\delta\rho`, :math:`E[\rho+\delta\rho]-E[\rho]` must equal
:math:`\int v_{xc}\,\delta\rho` to the order of the perturbation.

Below :data:`DENSITY_FLOOR` the gradient terms are switched off and the
functional falls back to its uniform-gas limit: :math:`s` and :math:`t` diverge
in the exponential tail, where the density carries no weight but the enhancement
factor would otherwise amplify round-off into the potential.
"""

from __future__ import annotations

import numpy as np

from .atomic_solver import lda_xc

#: The functionals :func:`xc_potential` accepts.
FUNCTIONALS = ("lda", "pbe")

#: Densities below this are treated as the uniform-gas limit (no gradient term).
DENSITY_FLOOR = 1e-12

#: Relative step of the numerical functional derivatives.
DERIVATIVE_STEP = 1e-6

# -- Perdew-Wang 1992 -------------------------------------------------------- #
#: ``(A, alpha1, beta1, beta2, beta3, beta4)`` of the unpolarized correlation
#: energy, Table I of Phys. Rev. B 45, 13244 (1992).
PW92_A = 0.031091
PW92_ALPHA1 = 0.21370
PW92_BETA = (7.5957, 3.5876, 1.6382, 0.49294)

# -- PBE --------------------------------------------------------------------- #
#: Exchange enhancement bound, PBE Eq. (14).  Fixed by the Lieb-Oxford bound.
PBE_KAPPA = 0.804
#: Gradient coefficient of the correlation hole, PBE Eq. (8).
PBE_BETA = 0.06672455060314922
#: ``beta pi^2 / 3``: recovers the correct linear response of the uniform gas.
PBE_MU = PBE_BETA * np.pi ** 2 / 3.0
#: ``(1 - ln 2) / pi^2``, PBE Eq. (8).
PBE_GAMMA = (1.0 - np.log(2.0)) / np.pi ** 2


def pw92_correlation(rho):
    r"""Perdew-Wang (1992) correlation: returns ``(e_c, v_c)`` (Hartree).

    The unpolarized limit of the parameterization,

    .. math::

        \varepsilon_c(r_s) = -2A(1+\alpha_1 r_s)
            \ln\Big[1 + \frac{1}{2A\,Q_1(r_s)}\Big],
        \quad Q_1 = \beta_1 r_s^{1/2} + \beta_2 r_s
                  + \beta_3 r_s^{3/2} + \beta_4 r_s^{2} ,

    with :math:`v_c = \varepsilon_c - (r_s/3)\,d\varepsilon_c/dr_s`, the
    derivative taken in closed form.
    """
    rho = np.maximum(np.asarray(rho, dtype=float), 1e-30)
    rs = (3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
    sqrt_rs = np.sqrt(rs)
    b1, b2, b3, b4 = PW92_BETA
    A, a1 = PW92_A, PW92_ALPHA1

    Q1 = b1 * sqrt_rs + b2 * rs + b3 * rs * sqrt_rs + b4 * rs * rs
    Q1_prime = (0.5 * b1 / sqrt_rs + b2 + 1.5 * b3 * sqrt_rs + 2.0 * b4 * rs)
    log_term = np.log1p(1.0 / (2.0 * A * Q1))

    ec = -2.0 * A * (1.0 + a1 * rs) * log_term
    # d/drs of the logarithm is -Q1' / (2A Q1^2 + Q1).
    dec = (-2.0 * A * a1 * log_term
           + 2.0 * A * (1.0 + a1 * rs) * Q1_prime / (2.0 * A * Q1 * Q1 + Q1))
    return ec, ec - (rs / 3.0) * dec


def _reduced_gradients(rho, gradient):
    """``(s, t)``: the exchange and correlation reduced density gradients."""
    rho = np.maximum(rho, DENSITY_FLOOR)
    abs_grad = np.abs(gradient)
    k_fermi = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)
    k_screen = np.sqrt(4.0 * k_fermi / np.pi)
    s = abs_grad / (2.0 * k_fermi * rho)
    t = abs_grad / (2.0 * k_screen * rho)
    return s, t


def pbe_exchange(rho, gradient):
    r"""PBE exchange energy per electron (Hartree).

    :math:`\varepsilon_x = \varepsilon_x^{\text{unif}} F_x(s)` with
    :math:`F_x = 1 + \kappa - \kappa/(1 + \mu s^2/\kappa)`, which is 1 at
    :math:`s = 0` and saturates at :math:`1+\kappa` -- the Lieb-Oxford bound.
    """
    rho = np.asarray(rho, dtype=float)
    dense = rho > DENSITY_FLOOR
    safe = np.maximum(rho, DENSITY_FLOOR)
    k_fermi = (3.0 * np.pi ** 2 * safe) ** (1.0 / 3.0)
    ex_unif = -3.0 * k_fermi / (4.0 * np.pi)
    s, _t = _reduced_gradients(safe, np.asarray(gradient, dtype=float))
    enhancement = 1.0 + PBE_KAPPA - PBE_KAPPA / (
        1.0 + PBE_MU * s * s / PBE_KAPPA)
    return ex_unif * np.where(dense, enhancement, 1.0)


def pbe_correlation(rho, gradient):
    r"""PBE correlation energy per electron (Hartree).

    :math:`\varepsilon_c = \varepsilon_c^{\text{PW92}} + H(r_s, t)` with

    .. math::

        H = \gamma \ln\Big[1 + \frac{\beta}{\gamma} t^2
            \frac{1 + A t^2}{1 + A t^2 + A^2 t^4}\Big],
        \qquad A = \frac{\beta}{\gamma}
            \Big[e^{-\varepsilon_c^{\text{PW92}}/\gamma} - 1\Big]^{-1} .
    """
    rho = np.asarray(rho, dtype=float)
    dense = rho > DENSITY_FLOOR
    safe = np.maximum(rho, DENSITY_FLOOR)
    ec_unif, _vc = pw92_correlation(safe)
    _s, t = _reduced_gradients(safe, np.asarray(gradient, dtype=float))

    ratio = PBE_BETA / PBE_GAMMA
    # expm1 keeps the small-argument limit accurate; the exponent is positive
    # because ec_unif < 0, so the denominator never vanishes.
    A = ratio / np.expm1(-ec_unif / PBE_GAMMA)
    t2 = t * t
    At2 = A * t2
    H = PBE_GAMMA * np.log1p(
        ratio * t2 * (1.0 + At2) / (1.0 + At2 + At2 * At2))
    return ec_unif + np.where(dense, H, 0.0)


def xc_energy_density(rho, gradient, functional: str = "pbe"):
    r"""The energy *per unit volume*, :math:`f = \rho\,\varepsilon_{xc}`.

    This is the quantity :func:`xc_potential` differentiates, so it is the one
    place the functional's form is written down.
    """
    key = _resolve(functional)
    rho = np.asarray(rho, dtype=float)
    if key == "lda":
        e_xc, _v = lda_xc(rho)
        return rho * e_xc
    return rho * (pbe_exchange(rho, gradient) + pbe_correlation(rho, gradient))


def _resolve(functional: str) -> str:
    key = str(functional).strip().lower().replace("-", "").replace("_", "")
    aliases = {"lda": "lda", "pz": "lda", "pz81": "lda", "ldapz": "lda",
               "pbe": "pbe", "gga": "pbe", "ggapbe": "pbe"}
    if key not in aliases:
        raise ValueError(
            f"unknown exchange-correlation functional {functional!r}; "
            f"available: {', '.join(FUNCTIONALS)}")
    return aliases[key]


def xc_potential(r, rho, functional: str = "lda", gradient=None):
    r"""``(e_xc, v_xc)`` on a radial grid: energy per electron and potential.

    Parameters
    ----------
    r : ndarray
        Uniform radial grid (Bohr), strictly positive.
    rho : ndarray
        Spherical density on that grid.
    functional : str
        ``"lda"`` (Slater + Perdew-Zunger, the historical default) or ``"pbe"``.
    gradient : ndarray, optional
        :math:`d\rho/dr`.  Computed from ``rho`` when omitted.

    Notes
    -----
    The LDA branch delegates to
    :func:`~mandacaru.basis.atomic_solver.lda_xc` unchanged, so switching a
    generation run to ``"pbe"`` and back reproduces the old numbers exactly.
    """
    key = _resolve(functional)
    r = np.asarray(r, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if key == "lda":
        return lda_xc(rho)

    if gradient is None:
        gradient = np.gradient(rho, r, edge_order=2)
    gradient = np.asarray(gradient, dtype=float)

    e_xc = pbe_exchange(rho, gradient) + pbe_correlation(rho, gradient)

    # Numerical partials of f(rho, sigma).  The steps are relative, with an
    # absolute floor so a vanishing density or a flat point still gets a
    # meaningful difference.
    d_rho = DERIVATIVE_STEP * np.maximum(np.abs(rho), DENSITY_FLOOR)
    scale = max(float(np.max(np.abs(gradient))), DENSITY_FLOOR)
    d_sigma = DERIVATIVE_STEP * np.maximum(np.abs(gradient),
                                           DERIVATIVE_STEP * scale)

    df_drho = (xc_energy_density(rho + d_rho, gradient, key)
               - xc_energy_density(rho - d_rho, gradient, key)) / (2.0 * d_rho)
    df_dsigma = (xc_energy_density(rho, gradient + d_sigma, key)
                 - xc_energy_density(rho, gradient - d_sigma, key)
                 ) / (2.0 * d_sigma)

    # -(1/r^2) d/dr (r^2 df/dsigma).
    flux = r * r * df_dsigma
    divergence = np.gradient(flux, r, edge_order=2) / (r * r)
    return e_xc, df_drho - divergence
