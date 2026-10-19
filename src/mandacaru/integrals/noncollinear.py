# -*- coding: utf-8 -*-
# file: integrals/noncollinear.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Noncollinear exchange-correlation through the local spin axis.

A two-component (spinor) Kohn-Sham state has, at each point, a :math:`2
\times 2` spin density matrix

.. math::

    \rho_{\sigma\tau}(\mathbf r) = \tfrac12\big(n\,\delta_{\sigma\tau}
        + \mathbf m\cdot\boldsymbol\sigma_{\sigma\tau}\big),

with the density :math:`n` and the magnetization vector :math:`\mathbf m`.
The collinear functionals of :mod:`.exchange_correlation` take two spin
densities.  Diagonalizing :math:`\rho` at each point gives them (Kübler et
al. 1988):

.. math::

    n_\pm = \tfrac12\big(n \pm |\mathbf m|\big), \qquad
    E_{xc}[n, \mathbf m] = E_{xc}^{\rm coll}[n_+, n_-],

the spin-polarized functional in the frame whose axis is
:math:`\hat{\mathbf m}(\mathbf r)`.  Its derivative is a potential and a
magnetic field, which enter the spinor Kohn-Sham matrix as :math:`v\,\mathbb
1 + \mathbf B\cdot\boldsymbol\sigma` (:func:`evaluate_noncollinear`).  For
LDA,

.. math::

    v = \tfrac12(v_+ + v_-), \qquad
    \mathbf B = \frac{\delta E_{xc}}{\delta\mathbf m}
        = \tfrac12(v_+ - v_-)\,\hat{\mathbf m}.

The gradient-corrected functional (PBE) takes the same local axis for the
gradients too: the spin densities' gradients are
:math:`\mathbf g_\pm = \tfrac12(\nabla n \pm \mathbf t)` with
:math:`t_k = \hat{\mathbf m}\cdot\partial_k\mathbf m`, the gradient of
:math:`\mathbf m` projected on the local axis -- which is
:math:`\nabla|\mathbf m|` wherever :math:`\mathbf m` is smooth and nonzero,
but, unlike the gradient of the field :math:`|\mathbf m|`, has no kink where
a collinear :math:`m_z` changes sign (there :math:`\hat{\mathbf m}` flips with
it, and :math:`n_\pm` and :math:`\mathbf g_\pm` are the collinear channels
swapped, which the functional cannot tell apart).  The field is the exact
derivative of that discrete energy:

.. math::

    \mathbf B = \tfrac12(f_+ - f_-)\hat{\mathbf m}
        - \sum_k \partial_k\big(C_k\hat{\mathbf m}\big)
        + \frac{1}{|\mathbf m|}\big(\mathbb 1 - \hat{\mathbf m}
          \hat{\mathbf m}^{T}\big)\sum_k C_k\,\partial_k\mathbf m ,
    \qquad
    \mathbf C = \tfrac12(\mathbf A_+ - \mathbf A_-),

with :math:`\mathbf A_\pm = 2 f_{\sigma_{\pm\pm}}\mathbf g_\pm +
f_{\sigma_{+-}}\mathbf g_\mp` and the divergence the adjoint of the grid's
gradient (:func:`~.exchange_correlation.divergence`).

The other functionals follow the same frame.  **r2SCAN** takes the
kinetic-energy density matrix :math:`\tau_{\sigma\sigma'} =
\tfrac12(\tau\,\delta_{\sigma\sigma'} + \boldsymbol\tau_m\cdot
\boldsymbol\sigma_{\sigma\sigma'})` along the axis, :math:`\tau_\pm =
\tfrac12(\tau \pm \hat{\mathbf m}\cdot\boldsymbol\tau_m)`, and returns the
identity and Pauli parts of its derivative (``tau_potential``,
``tau_field``); the turning of the axis through :math:`\tau_\pm` adds
:math:`(\mathbb 1 - \hat{\mathbf m}\hat{\mathbf m}^T)\tfrac12(e_{\tau+} -
e_{\tau-})\boldsymbol\tau_m/|\mathbf m|` to :math:`\mathbf B`.  (A
noncollinear meta-GGA is not gauge invariant without the spin current; the
local-axis form neglects it, as the collinear one does the paramagnetic
current.)  **rVV10** is a functional of the total density alone.  The
**HSE06** semilocal part is PBE's, and the short-range exchange it gives up
for the exact exchange is subtracted channel by channel along the same axis
with the same projected gradients (:func:`_short_range_noncollinear`).

Properties: the energy is invariant under a global spin rotation (the field
turns with it); a collinear state, whatever the sign pattern of its
:math:`m_z`, gives back the collinear functional and both channel potentials
to round-off; a time-reversal symmetric (Kramers-paired, closed-shell) state
has :math:`\mathbf m = 0` and the spin-unpolarized functional.  The
approximation, and its known limits:

- it is the collinear GGA applied in the local frame, not a noncollinear GGA
  in its own right: the transverse gradients of :math:`\mathbf m` (the
  gradient of the axis, :math:`\nabla\hat{\mathbf m}`) do not enter the
  energy, so the spin stiffness of a slowly twisting magnetization is not
  described;
- the last term of :math:`\mathbf B` (from the rotation of the axis) carries
  :math:`1/|\mathbf m|`: where a noncollinear magnetization passes through
  zero at a finite density it can be large, the known numerical instability
  of local-axis GGAs.  Where :math:`\mathbf m` vanishes -- exactly, or to
  round-off (:data:`MAGNETIZATION_ROUNDOFF` of the density) -- that term is
  set to zero, and the axis is taken from :math:`\boldsymbol\tau_m`
  (r2SCAN) or the gradient instead (:func:`gradient_axis`): on a grid node of a collinear moment's nodal
  plane the axis is otherwise undefined, and a zero axis would drop the
  spin gradient there (2.65 mHa on a test density with a nodal plane on the
  grid);
- the gradient terms are dropped below
  :data:`~.exchange_correlation.GRADIENT_DENSITY_FLOOR`, as in the collinear
  evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exchange_correlation import (GRADIENT_DENSITY_FLOOR, _nonlocal,
                                   _screening, divergence, evaluate_spin,
                                   gradient, resolve_functional,
                                   semilocal_part)

#: The functionals the local-axis construction is written for: LDA, PBE, the
#: semilocal part of HSE06 (the caller adds the spinor exact exchange),
#: r2SCAN with the kinetic-energy density along the same axis, and
#: r2SCAN+rVV10, whose nonlocal correlation sees the total density only.
NONCOLLINEAR_FUNCTIONALS = ("lda", "pbe", "hse06", "r2scan", "r2scan-rvv10")


def supports_noncollinear(functional: str) -> bool:
    """Whether ``functional`` has a two-component (noncollinear) form here."""
    return resolve_functional(functional) in NONCOLLINEAR_FUNCTIONALS


@dataclass(frozen=True)
class NoncollinearXCTerms:
    r"""The noncollinear functional on one :math:`(n, \mathbf m)`.

    ``energy`` (Hartree), ``potential`` :math:`\delta E/\delta n` (flat, the
    part of the operator proportional to the identity in spin) and ``field``
    :math:`\mathbf B = \delta E/\delta\mathbf m`, ``(3, ngrid)``, the part
    multiplying the Pauli matrices.  A meta-GGA adds ``tau_potential``
    :math:`\partial E/\partial\tau` and ``tau_field``
    :math:`\partial E/\partial\boldsymbol\tau_m` (``(3, ngrid)``), the
    identity and Pauli parts of the operator on the kinetic-energy density
    matrix; ``None`` otherwise.
    """

    energy: float
    potential: np.ndarray
    field: np.ndarray
    tau_potential: np.ndarray | None = None
    tau_field: np.ndarray | None = None


def magnetization_axis(magnetization) -> tuple[np.ndarray, np.ndarray]:
    r"""``(|m|, m_hat)`` of a ``(3, ngrid)`` magnetization: the length and
    the unit vector, zero where :math:`\mathbf m` vanishes."""
    m = np.asarray(magnetization, dtype=float)
    length = np.sqrt(np.sum(m * m, axis=0))
    axis = np.divide(m, length, out=np.zeros_like(m), where=length > 0.0)
    return length, axis


#: A magnetization below this fraction of the local density is round-off:
#: its direction is noise.  A collinear state turned to x carries 1e-16
#: components along y and z, which on a nodal plane of the spin density
#: (|m| ~ 1e-17 there) gave the axis a random direction -- r2SCAN's
#: tau_+- (tau_m does not vanish there) and the 1/|m| field term then
#: changed O2's energy by 1.4 mHa and stopped the SCF.
MAGNETIZATION_ROUNDOFF = 1e-12


def gradient_axis(axis, length, grad_m, *, zero=None,
                  tau_m=None) -> np.ndarray:
    r"""``axis`` with the points where :math:`\mathbf m = 0` exactly (or
    the mask ``zero``) given the direction of :math:`\boldsymbol\tau_m`
    where a meta-GGA passes one that does not vanish there, else of
    :math:`\partial_k\mathbf m` along the Cartesian ``k`` where it is
    largest (left zero where :math:`\nabla\mathbf m` vanishes too).

    For a collinear :math:`m_z` through a node on a grid point that is
    :math:`\pm\hat{\mathbf z}`, the axis on either side, so the projected
    gradient there is :math:`\pm\nabla m_z`, the collinear functional's.
    The sign is immaterial: with :math:`n_+ = n_-` it only swaps the two
    channels, which the functional cannot tell apart, and the field terms
    :math:`(f_+ - f_-)\hat{\mathbf m}` and :math:`C_k\hat{\mathbf m}`
    keep their sign; it is fixed by making the largest component positive.
    """
    zero = np.flatnonzero(length == 0.0 if zero is None else zero)
    if zero.size == 0:
        return axis
    g = grad_m[:, :, zero]                              # (3 k, 3 a, Z)
    row = np.argmax(np.sum(g * g, axis=1), axis=0)      # dominant k
    direction = g[row, :, np.arange(zero.size)].T       # (3 a, Z)
    if tau_m is not None:
        # A collinear tau_m points along the state's axis where m vanishes
        # (a nodal plane of the spin density, where tau_m does not).
        t = tau_m[:, zero]
        has = np.sum(t * t, axis=0) > 0.0
        direction = np.where(has[None, :], t, direction)
    size = np.sqrt(np.sum(direction * direction, axis=0))
    lead = direction[np.argmax(np.abs(direction), axis=0),
                     np.arange(zero.size)]
    direction = direction * np.where(lead < 0.0, -1.0, 1.0)
    axis = axis.copy()
    axis[:, zero] = np.divide(direction, size, out=np.zeros_like(direction),
                              where=size > 0.0)
    return axis


def evaluate_noncollinear(grid, density, magnetization,
                          functional: str = "lda", *,
                          relativistic: bool = False, tau=None,
                          tau_magnetization=None, screening=None,
                          exchange_density=None) -> NoncollinearXCTerms:
    r"""``E_xc``, :math:`\delta E/\delta n` and :math:`\mathbf B` of a
    noncollinear state on ``grid`` (the module docstring has the formulas).

    Parameters
    ----------
    grid : Grid
        The integration grid (Bohr).
    density : ndarray
        :math:`n`, flat, any (spin-unpolarized) partial core already added.
    magnetization : ndarray
        :math:`\mathbf m`, ``(3, ngrid)``.
    functional : str
        One of :data:`NONCOLLINEAR_FUNCTIONALS`.
    relativistic : bool
        The relativistic exchange factor, as for
        :func:`~.exchange_correlation.evaluate_spin`.
    tau, tau_magnetization : ndarray, optional
        r2SCAN's kinetic-energy density matrix
        :math:`\tau_{\sigma\sigma'} = \tfrac12(\tau\,\delta_{\sigma\sigma'} +
        \boldsymbol\tau_m\cdot\boldsymbol\sigma_{\sigma\sigma'})`: the
        trace :math:`\tau` (any partial core's added) and the vector
        :math:`\boldsymbol\tau_m` ``(3, ngrid)``.  The local axis gives the
        channels :math:`\tau_\pm = \tfrac12(\tau \pm \hat{\mathbf m}\cdot
        \boldsymbol\tau_m)` -- the collinear :math:`\tau_\uparrow,
        \tau_\downarrow` when :math:`\mathbf m` is along :math:`\pm z` --
        and the turning of the axis adds :math:`\tfrac{1}{|\mathbf m|}(1 -
        \hat{\mathbf m}\hat{\mathbf m}^T)\,\tfrac12(e_{\tau+} - e_{\tau-})
        \boldsymbol\tau_m` to the field (zero where :math:`\mathbf m = 0`).
    screening, exchange_density
        A hybrid's ``(omega, fraction)`` and the density its short-range
        semilocal exchange is subtracted on (the valence, no core), as for
        :func:`~.exchange_correlation.evaluate`: the channels
        :math:`\tfrac12(n_x \pm |\mathbf m|)` along the axis.
    """
    from ..basis.xc_spin import spin_partials

    full = resolve_functional(functional)
    if not supports_noncollinear(full):
        raise NotImplementedError(
            f"xc={full!r} has no two-component (noncollinear) form here; the "
            f"spinor Kohn-Sham functionals are "
            f"{', '.join(NONCOLLINEAR_FUNCTIONALS)}")
    key = semilocal_part(full)
    screening = _screening(key, screening)
    m = np.asarray(magnetization, dtype=float)
    n = np.asarray(density, dtype=float)
    length, axis = magnetization_axis(m)
    if key == "lda":
        terms = evaluate_spin(grid, 0.5 * (n + length), 0.5 * (n - length),
                              key, relativistic=relativistic)
        return NoncollinearXCTerms(
            energy=terms.energy,
            potential=0.5 * (terms.potential_up + terms.potential_dn),
            field=0.5 * (terms.potential_up - terms.potential_dn) * axis)

    meta = key == "r2scan"
    # The channels along the axis, clipped at zero as the collinear
    # evaluator clips its spin densities, and the clipped spin density
    # matrix rebuilt from them: n_c = c_+ + c_-, m_c = (c_+ - c_-) m_hat.
    # Its gradients are then those of the clipped channels (along z, the
    # collinear evaluator's), and stay free of the kink of |m|.
    raw_length = length
    up = np.maximum(0.5 * (n + length), 0.0)
    dn = np.maximum(0.5 * (n - length), 0.0)
    rho, length = up + dn, up - dn
    m = length * axis
    weighted = (up + dn) > GRADIENT_DENSITY_FLOOR
    grad_n = gradient(grid, rho)                       # (3 k, G)
    grad_m = gradient(grid, m)                         # (3 k, 3 a, G)
    # Round-off magnetizations count as zero (their direction is noise).
    zero = length <= MAGNETIZATION_ROUNDOFF * rho
    if meta:
        if tau is None or tau_magnetization is None:
            raise ValueError("a meta-GGA needs the kinetic-energy density "
                             "matrix (tau and tau_magnetization)")
        tau_m = np.asarray(tau_magnetization, dtype=float)
    axis = gradient_axis(axis, length, grad_m, zero=zero,
                         tau_m=tau_m if meta else None)
    t = np.einsum("ag,kag->kg", axis, grad_m)          # m_hat . d_k m
    g_up, g_dn = 0.5 * (grad_n + t), 0.5 * (grad_n - t)
    s_uu = np.where(weighted, np.sum(g_up * g_up, axis=0), 0.0)
    s_ud = np.where(weighted, np.sum(g_up * g_dn, axis=0), 0.0)
    s_dd = np.where(weighted, np.sum(g_dn * g_dn, axis=0), 0.0)
    # rVV10 sees the total density, before the meta-GGA's floor below (as
    # in the collinear evaluator).
    total = up + dn
    tau_up = tau_dn = None
    if meta:
        tau = np.asarray(tau, dtype=float)
        projected = np.sum(axis * tau_m, axis=0)
        tau_up, tau_dn = 0.5 * (tau + projected), 0.5 * (tau - projected)
        up, dn = np.where(weighted, up, 0.0), np.where(weighted, dn, 0.0)
    f, v_up, v_dn, d_uu, d_ud, d_dd, e_up, e_dn = spin_partials(
        key, up, dn, s_uu, s_ud, s_dd, tau_up, tau_dn,
        relativistic=relativistic)
    d_uu, d_ud, d_dd = (np.where(weighted, d, 0.0)
                        for d in (d_uu, d_ud, d_dd))
    e_nl, v_nl, s_nl = _nonlocal(grid, full, total,
                                 s_uu + 2.0 * s_ud + s_dd, weighted)
    v_up, v_dn = v_up + v_nl, v_dn + v_nl
    d_uu, d_ud, d_dd = d_uu + s_nl, d_ud + 2.0 * s_nl, d_dd + s_nl
    a_up = 2.0 * d_uu * g_up + d_ud * g_dn
    a_dn = 2.0 * d_dd * g_dn + d_ud * g_up
    potential = 0.5 * (v_up + v_dn) - divergence(grid, 0.5 * (a_up + a_dn))
    C = 0.5 * (a_up - a_dn)
    field = 0.5 * (v_up - v_dn) * axis - np.stack(
        [divergence(grid, C * axis[a]) for a in range(3)])
    # The rotation of the axis: (1 - m_hat m_hat^T) (sum_k C_k d_k m) / |m|,
    # and for a meta-GGA (e_tau+ - e_tau-)/2 tau_m through tau_+-.
    w = np.einsum("kg,kag->ag", C, grad_m)
    tau_potential = tau_field = None
    if meta:
        half = 0.5 * (np.asarray(e_up) - np.asarray(e_dn))
        w = w + half * tau_m
        tau_potential = 0.5 * (np.asarray(e_up) + np.asarray(e_dn))
        tau_field = half * axis
    w = w - axis * np.sum(axis * w, axis=0)
    field = field + np.divide(w, length, out=np.zeros_like(w),
                              where=~zero)
    # A channel the functional saw clipped at zero (n_+- < 0, possible for a
    # smooth PAW density with half a partial core) has no derivative there,
    # as the collinear evaluator's `_clipped`: the channel potentials along
    # the axis are zeroed, the transverse field kept.
    # The chain rule through the clipping: a clipped channel has no
    # derivative (the collinear `_clipped`), and the transverse field of
    # m_c = (c_+ - c_-) m_hat is that of m scaled by |m_c|/|m|.
    parallel = np.sum(field * axis, axis=0)
    transverse = field - parallel * axis
    channel_up = np.where(0.5 * (n + raw_length) < 0.0, 0.0,
                          potential + parallel)
    channel_dn = np.where(0.5 * (n - raw_length) < 0.0, 0.0,
                          potential - parallel)
    potential = 0.5 * (channel_up + channel_dn)
    ratio = np.divide(length, raw_length, out=np.ones_like(length),
                      where=raw_length > 0.0)
    field = 0.5 * (channel_up - channel_dn) * axis + ratio * transverse
    energy = float(np.sum(f) * grid.dV) + e_nl
    if screening is not None and screening[1] != 0.0:
        # The short-range semilocal exchange the exact exchange replaces.
        omega, fraction = screening
        x = rho if exchange_density is None else np.asarray(exchange_density,
                                                            dtype=float)
        e_sr, v_sr, b_sr = _short_range_noncollinear(grid, x, raw_length,
                                                     axis, omega, zero)
        energy -= fraction * e_sr
        potential = potential - fraction * v_sr
        field = field - fraction * b_sr
    return NoncollinearXCTerms(energy=energy,
                               potential=np.asarray(potential, dtype=float),
                               field=field, tau_potential=tau_potential,
                               tau_field=tau_field)


def _short_range_noncollinear(grid, density, length, axis, omega, zero):
    r"""``(E, v, B)`` of a screened hybrid's short-range semilocal exchange
    on a noncollinear state: the channels :math:`x_\pm = \tfrac12(x \pm
    |\mathbf m|)` along the local axis, clipped at zero and rebuilt as a
    clipped spin density matrix for their gradients (as the gradient
    functional's channels are), each channel's exchange :math:`\tfrac12
    E_x^{\rm SR}[2x_\pm]` (the exact spin scaling of
    :func:`~.exchange_correlation._short_range_exchange`, which a collinear
    state gives back), and the chain rule back through the clipping."""
    from ..basis import hse

    c_up = np.maximum(0.5 * (density + length), 0.0)
    c_dn = np.maximum(0.5 * (density - length), 0.0)
    clipped = c_up - c_dn
    grad_n = gradient(grid, c_up + c_dn)
    grad_m = gradient(grid, clipped * axis)
    t = np.einsum("ag,kag->kg", axis, grad_m)
    energy, df, A = 0.0, [], []
    for sign, channel in ((1.0, c_up), (-1.0, c_dn)):
        rho = 2.0 * channel
        grad = grad_n + sign * t                      # gradient of 2 x_+-
        weighted = rho > GRADIENT_DENSITY_FLOOR
        sigma = np.where(weighted, np.sum(grad * grad, axis=0), 0.0)
        f, df_drho, df_dsigma = hse.short_range_partials(rho, sigma, omega)
        df_dsigma = np.where(weighted, df_dsigma, 0.0)
        energy += 0.5 * float(np.sum(f) * grid.dV)
        df.append(df_drho)
        A.append(2.0 * df_dsigma * grad)              # dE/d(g_+-)
    potential = 0.5 * (df[0] + df[1]) - divergence(grid, 0.5 * (A[0] + A[1]))
    C = 0.5 * (A[0] - A[1])
    field = 0.5 * (df[0] - df[1]) * axis - np.stack(
        [divergence(grid, C * axis[a]) for a in range(3)])
    w = np.einsum("kg,kag->ag", C, grad_m)
    w = w - axis * np.sum(axis * w, axis=0)
    field = field + np.divide(w, clipped, out=np.zeros_like(w),
                              where=(clipped > 0.0) & ~zero)
    # The chain rule through the clipping, as for the gradient functional.
    parallel = np.sum(field * axis, axis=0)
    transverse = field - parallel * axis
    channel_up = np.where(0.5 * (density + length) < 0.0, 0.0,
                          potential + parallel)
    channel_dn = np.where(0.5 * (density - length) < 0.0, 0.0,
                          potential - parallel)
    ratio = np.divide(clipped, length, out=np.ones_like(length),
                      where=length > 0.0)
    return (energy, 0.5 * (channel_up + channel_dn),
            0.5 * (channel_up - channel_dn) * axis + ratio * transverse)
