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

**The two partial derivatives are taken in closed form** (:func:`pbe_partials`),
and only the radial derivative of the resulting flux is taken on the grid.
They were first taken by central differences of :func:`xc_energy_density`,
which is correct by construction but not smooth: a relative step of
:math:`10^{-6}` leaves about :math:`10^{-10}` of cancellation noise in
:math:`\partial f/\partial\sigma`, the divergence amplifies it by
:math:`1/h` and :math:`1/r^2`, and on the 100500-point grid of a holmium
reference atom the SCF never converged -- it stalled at a residual of
:math:`10^{-6}` from an LDA start and wandered at 0.1-30 from the
Thomas-Fermi one.  A dropped chain-rule factor is the usual way a closed-form
GGA potential goes wrong, so ``test/basis/test_xc.py`` certifies it twice:
the partials against central differences of :func:`xc_energy_density`, and
the potential against the energy -- for a random perturbation
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

#: Relative spacing :math:`\Delta` of the points a GGA's radial derivatives are
#: taken on, :math:`\max(h, \Delta r)` (see :func:`derivative_nodes`).
GGA_DERIVATIVE_STEP = 0.01

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
    ec, dec = _pw92(rs)
    return ec, ec - (rs / 3.0) * dec


def _pw92(rs):
    """``(eps_c, d eps_c / d r_s)`` of Perdew-Wang (1992), unpolarized."""
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
    return ec, dec


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


def pbe_partials(rho, gradient):
    r"""``(df/drho, df/dsigma)`` of the PBE energy density, in closed form.

    :math:`f = \rho\,\varepsilon_{xc}(\rho, |\sigma|)` with
    :math:`\sigma = d\rho/dr`.  With :math:`s \propto |\sigma|\rho^{-4/3}`
    and :math:`t \propto |\sigma|\rho^{-7/6}`,

    .. math::

        \frac{\partial f_x}{\partial\rho} = \tfrac43\varepsilon_x^{\rm unif}
            (F_x - s F_x'), \qquad
        \frac{\partial f_x}{\partial|\sigma|} =
            \frac{\varepsilon_x^{\rm unif} F_x'}{2k_F},

    and for correlation :math:`\partial(\rho\varepsilon_c^{\rm PW92})/\partial\rho
    = v_c^{\rm PW92}` plus :math:`H + \rho\,\partial H/\partial\rho`, where
    :math:`H` depends on :math:`\rho` through :math:`t` and through
    :math:`A(\varepsilon_c^{\rm PW92})`.  Below :data:`DENSITY_FLOOR` both
    gradient terms vanish, as they do in the energy.
    """
    rho = np.asarray(rho, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    dense = rho > DENSITY_FLOOR
    safe = np.maximum(rho, DENSITY_FLOOR)
    g = np.abs(gradient)
    sign = np.sign(gradient)

    # -- exchange ----------------------------------------------------------- #
    k_fermi = (3.0 * np.pi ** 2 * safe) ** (1.0 / 3.0)
    ex_unif = -3.0 * k_fermi / (4.0 * np.pi)
    s = g / (2.0 * k_fermi * safe)
    denominator = 1.0 + PBE_MU * s * s / PBE_KAPPA
    fx = 1.0 + PBE_KAPPA - PBE_KAPPA / denominator
    dfx = 2.0 * PBE_MU * s / (denominator * denominator)
    dx_drho = np.where(dense, (4.0 / 3.0) * ex_unif * (fx - s * dfx),
                       (4.0 / 3.0) * ex_unif)
    dx_dg = np.where(dense, ex_unif * dfx / (2.0 * k_fermi), 0.0)

    # -- correlation -------------------------------------------------------- #
    rs = (3.0 / (4.0 * np.pi * safe)) ** (1.0 / 3.0)
    ec, dec_drs = _pw92(rs)
    vc = ec - (rs / 3.0) * dec_drs
    k_screen = np.sqrt(4.0 * k_fermi / np.pi)
    t = g / (2.0 * k_screen * safe)
    ratio = PBE_BETA / PBE_GAMMA
    growth = np.expm1(-ec / PBE_GAMMA)
    A = ratio / growth
    # dA/d eps_c, and d eps_c/d rho = d eps_c/d r_s * (-r_s / 3 rho).
    dA_dec = A * A * (growth + 1.0) / (ratio * PBE_GAMMA)
    dA_drho = dA_dec * dec_drs * (-rs / (3.0 * safe))
    u = t * t
    Au = A * u
    D = 1.0 + Au + Au * Au
    Q = u * (1.0 + Au) / D
    dQ_du = ((1.0 + 2.0 * Au) * D - u * (1.0 + Au) * (A + 2.0 * A * Au)) / (D * D)
    dQ_dA = (u * u * D - u * (1.0 + Au) * (u + 2.0 * Au * u)) / (D * D)
    X = ratio * Q
    H = PBE_GAMMA * np.log1p(X)
    dH_dX = PBE_GAMMA / (1.0 + X)
    # u = t^2 with t ~ g rho^(-7/6): du/drho = -(7/3) u / rho, du/dg = 2t dt/dg.
    dH_drho = dH_dX * ratio * (dQ_du * (-7.0 / 3.0) * u / safe
                               + dQ_dA * dA_drho)
    dH_dg = dH_dX * ratio * dQ_du * 2.0 * t / (2.0 * k_screen * safe)
    dc_drho = vc + np.where(dense, H + safe * dH_drho, 0.0)
    dc_dg = np.where(dense, safe * dH_dg, 0.0)

    return dx_drho + dc_drho, (dx_dg + dc_dg) * sign


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

    nodes = derivative_nodes(r)
    if gradient is None:
        gradient = _radial_derivative(r, rho, nodes)
    gradient = np.asarray(gradient, dtype=float)

    e_xc = pbe_exchange(rho, gradient) + pbe_correlation(rho, gradient)

    df_drho, df_dsigma = pbe_partials(rho, gradient)

    # -(1/r^2) d/dr (r^2 df/dsigma).
    flux = r * r * df_dsigma
    divergence = _radial_derivative(r, flux, nodes) / (r * r)
    return e_xc, df_drho - divergence


def derivative_nodes(r) -> np.ndarray | None:
    r"""Indices of the grid points the GGA's radial derivatives are taken on,
    or ``None`` for all of them.

    **Why not every point.**  A gradient-corrected potential differentiates
    the density twice, which multiplies whatever noise the density carries by
    about :math:`1/h^2`.  On the fine uniform grid of a heavy reference atom
    (100500 points for holmium, :math:`h = 3\times10^{-4}` Bohr) the orbitals
    are accurate only to the eigensolver's :math:`\epsilon\lVert T\rVert/
    \Delta \sim 10^{-8}`, and that broadband noise (not a checkerboard: its
    lag-1 autocorrelation is -0.33) came out of the two derivatives as a
    relative ripple of :math:`5\times10^{-3}` in :math:`v_{xc}` in the
    valence.  The scalar-relativistic Darwin term then differentiated it twice
    more: spurious classically allowed points out to 4.2 Bohr for the 4f, a
    PAW local potential whose derivative-matched polynomial dived to -4691
    Hartree at the origin for antimony, and an SCF that fed on its own ripple.

    So the derivatives are taken on a subset whose spacing is
    :math:`\max(h, \Delta\,r)` with :math:`\Delta` =
    :data:`GGA_DERIVATIVE_STEP` -- logarithmic where the grid is finer than
    the physics needs, every point near the nucleus where it is not -- and
    splined back.  :math:`\Delta` is set by the most demanding consumer, a
    PAW or ONCVPSP local potential matched through its *fourth* derivative at
    :math:`r_{cl}`.  For PBE antimony at :math:`r_{cl}` = 2.74 Bohr,
    :math:`V^{(4)}` read over windows of 0.01-0.2 Bohr scattered from +11 to
    -80 at :math:`\Delta` = 0.002 and 0.005, and agreed from 0.01 up (-2.1 to
    -4.4; LDA gives -1.64).  The price at 0.01: antimony's 5p level moves by
    5 microhartree and its total energy by 1.2 mHa out of 6493 Hartree.  On a
    grid already coarser than :math:`\Delta r` everywhere it matters, and on
    a non-uniform grid, every point is used.
    """
    r = np.asarray(r, dtype=float)
    if r.size < 8:
        return None
    step = r[1:] - r[:-1]
    h = float(step[0])
    if not np.allclose(step, h, rtol=1e-6, atol=0.0):
        return None
    stride = np.maximum(1, np.floor(GGA_DERIVATIVE_STEP * r / h).astype(int))
    if stride.max() <= 1:
        return None
    nodes = [0]
    while True:
        nxt = nodes[-1] + int(stride[nodes[-1]])
        if nxt >= r.size - 1:
            break
        nodes.append(nxt)
    nodes.append(r.size - 1)
    return np.asarray(nodes)


def _radial_derivative(r, y, nodes):
    """``dy/dr`` on the ``nodes`` subset, splined back onto ``r``."""
    if nodes is None:
        return np.gradient(y, r, edge_order=2)
    from scipy.interpolate import CubicSpline

    derivative = np.gradient(y[nodes], r[nodes], edge_order=2)
    return CubicSpline(r[nodes], derivative)(r)
