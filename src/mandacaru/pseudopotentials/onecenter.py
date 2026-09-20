# -*- coding: utf-8 -*-
# file: pseudopotentials/onecenter.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""The PAW one-center electron-electron term.

Blöchl's total energy splits the Hartree energy into a smooth part evaluated
everywhere and a correction confined to each augmentation sphere,

.. math::

    E_H[n] = E_H[\tilde n + \hat n]
           + \sum_A \Big( E_H[n^1_A] - E_H[\tilde n^1_A + \hat n_A] \Big),

where :math:`n^1_A = \sum_{ij} D_{ij}\,\phi_i^*\phi_j` is the all-electron
one-center density and :math:`\tilde n^1_A` its smooth counterpart.  The first
term is what the grid and the compensation charges already compute
(:meth:`~mandacaru.pseudopotentials.paw.PAWIntegrals.two_body_augmentation`);
this module supplies the second.

It is **quadratic** in the one-center density matrix :math:`D`, and that is the
point.  Mandacaru previously linearized it about the isolated atom -- a fixed
:math:`D^{ion}` plus a per-species constant -- which is exact only while
:math:`D` stays at its atomic reference :math:`D^0`.  Forming a bond is
precisely what moves it: charge transfers and the s and p channels rehybridize.
Writing :math:`E(D) = \tfrac12 D D \Delta W` and expanding about :math:`D^0`,

.. math::

    E(D) = \underbrace{E(D^0) + (D - D^0)\,D^0 \Delta W}_{\text{the linearization}}
         + \underbrace{\tfrac12 (D-D^0)(D-D^0)\,\Delta W}_{\text{what was missing}},

so the omission is exactly the second-order term -- and because the Hartree
energy is exactly quadratic, adding it makes the one-center treatment *exact*
rather than merely better.  It costs no self-consistency: expanding the square
gives a two-body operator (folded into the ERI), a one-body operator (folded
into the nonlocal coupling) and a constant.

:math:`\Delta W` itself is
:math:`(\phi_a^*\phi_b|\phi_c^*\phi_d) - (\tilde\phi_a^*\tilde\phi_b + \hat n_{ab}
|\tilde\phi_c^*\tilde\phi_d + \hat n_{cd})` over one sphere, assembled from the
usual multipole expansion of :math:`1/r_{12}`: a radial double integral per
:math:`L` (:func:`radial_coulomb`) times the angular couplings of
:mod:`mandacaru.pseudopotentials.multipoles`.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson

from .multipoles import gaunt, multipole_range, partial_waves, shape_function

#: Radial points of the one-center quadrature (uniform, inside the sphere).
ONE_CENTER_POINTS = 600


def radial_coulomb(r: np.ndarray, rho_1: np.ndarray, rho_2: np.ndarray,
                   L: int) -> float:
    r"""``int int rho_1(r) rho_2(r') r_<^L / r_>^{L+1} r^2 r'^2 dr dr'``.

    The radial half of the :math:`L`-th term of the multipole expansion of
    :math:`1/r_{12}`; the angular half is a product of Gaunt coefficients.  The
    inner integral is accumulated once so the cost is linear in the grid.
    """
    L = int(L)
    r = np.asarray(r, dtype=float)
    weight_in = rho_2 * r ** (L + 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        weight_out = np.where(r > 0.0, rho_2 * r ** (1 - L), 0.0)
    below = np.concatenate([[0.0], cumulative_trapezoid(weight_in, r)])
    above = np.concatenate([[0.0], cumulative_trapezoid(weight_out, r)])
    above = above[-1] - above
    with np.errstate(divide="ignore", invalid="ignore"):
        potential = np.where(r > 0.0, below / r ** (L + 1), 0.0) + r ** L * above
    return float(simpson(rho_1 * potential * r * r, x=r))


def angular_coupling(L: int, a, b, c, d) -> complex:
    r"""``sum_M (-1)^M G(L,M;a,b) G(L,-M;c,d) * 4 pi/(2L+1)``.

    The angular factor multiplying :func:`radial_coulomb` for the pair
    densities :math:`\phi_a^*\phi_b` and :math:`\phi_c^*\phi_d`, from
    :math:`\int Y_{LM}Y_{L-M}d\Omega = (-1)^M`.
    """
    total = 0.0 + 0.0j
    for M in range(-int(L), int(L) + 1):
        left = gaunt(L, M, a.l, a.m, b.l, b.m)
        if left == 0:
            continue
        right = gaunt(L, -M, c.l, c.m, d.l, d.m)
        if right == 0:
            continue
        total += (-1.0) ** M * left * right
    return total * 4.0 * np.pi / (2 * int(L) + 1)


def _radial_tables(dataset, basis: str):
    """``(r, {l: (ae, ps)}, r_cut)`` on a uniform sphere grid."""
    channels = sorted(dataset.channels)
    r_cut = max(float(dataset.channels[l].r_cut) for l in channels)
    r = np.linspace(0.0, r_cut, ONE_CENTER_POINTS)
    source = np.asarray(dataset.r, dtype=float)
    tables = {}
    for l in channels:
        ae, ps = partial_waves(dataset, l)
        transform = None
        if basis == "raw":
            transform = np.linalg.inv(
                np.asarray(dataset.channels[l].vanderbilt, dtype=float))
        ae_r = [np.interp(r, source, w) for w in ae]
        ps_r = [np.interp(r, source, w) for w in ps]
        if transform is not None:
            ae_r = [sum(transform[i, k] * ae_r[k] for k in range(len(ae_r)))
                    for i in range(len(ae_r))]
            ps_r = [sum(transform[i, k] * ps_r[k] for k in range(len(ps_r)))
                    for i in range(len(ps_r))]
        tables[l] = (ae_r, ps_r)
    return r, tables, r_cut


def one_center_coulomb(dataset, projectors, basis: str = "raw") -> np.ndarray:
    r"""``DeltaW[a, b, c, d]`` for one atom, over its own ``projectors``.

    :math:`\Delta W = (\phi^*_a\phi_b|\phi^*_c\phi_d) -
    (\tilde\phi^*_a\tilde\phi_b + \hat n_{ab}|\tilde\phi^*_c\tilde\phi_d
    + \hat n_{cd})`, the all-electron minus smooth-plus-compensation
    electron-electron energy inside the augmentation sphere.  Both densities
    carry the same multipole moments by construction, so the difference is
    confined to the sphere and the result is finite and short-ranged.

    ``projectors`` are this atom's :class:`~mandacaru.pseudopotentials.orbitals.KBProjector`
    objects in the order the integrals index them; ``basis`` must match the one
    they were built in.
    """
    r, tables, _r_cut = _radial_tables(dataset, basis)
    r_g = float(dataset.compensation_radius)
    n = len(projectors)

    # Radial pair densities and their compensation moments, per projector pair
    # and per L: only (l_a, l_b) and the radial indices matter.
    pair_cache: dict = {}

    def pair(a, b, L):
        key = (a.l, a.index, b.l, b.index, int(L))
        if key not in pair_cache:
            ae_a, ps_a = tables[a.l]
            ae_b, ps_b = tables[b.l]
            ae = ae_a[a.index] * ae_b[b.index]
            ps = ps_a[a.index] * ps_b[b.index]
            moment = float(simpson(ae * r ** (int(L) + 2), x=r)
                           - simpson(ps * r ** (int(L) + 2), x=r))
            compensated = ps + moment * shape_function(r, r_g, int(L))
            pair_cache[key] = (ae, compensated)
        return pair_cache[key]

    out = np.zeros((n, n, n, n), dtype=complex)
    for a in range(n):
        for b in range(n):
            levels_ab = multipole_range(projectors[a].l, projectors[b].l)
            for c in range(n):
                for d in range(n):
                    levels = [L for L in multipole_range(projectors[c].l,
                                                         projectors[d].l)
                              if L in levels_ab]
                    total = 0.0 + 0.0j
                    for L in levels:
                        angular = angular_coupling(L, projectors[a],
                                                   projectors[b],
                                                   projectors[c], projectors[d])
                        if angular == 0:
                            continue
                        ae_ab, ps_ab = pair(projectors[a], projectors[b], L)
                        ae_cd, ps_cd = pair(projectors[c], projectors[d], L)
                        total += angular * (radial_coulomb(r, ae_ab, ae_cd, L)
                                            - radial_coulomb(r, ps_ab, ps_cd, L))
                    out[a, b, c, d] = total
    return out


def reference_density_matrix(dataset, projectors) -> np.ndarray:
    r"""``D^0`` -- the one-center density matrix of the isolated reference atom.

    The reference atom's smooth orbitals *are* the first (bound) partial wave of
    each channel, and the projectors are dual to them, so
    :math:`\langle\tilde p_i|\tilde\varphi_j\rangle = \delta_{ij}` and
    :math:`D^0` is diagonal: the channel's occupation shared equally over its
    ``2l+1`` magnetic states, on the bound wave only.
    """
    n = len(projectors)
    D0 = np.zeros((n, n), dtype=complex)
    for i, projector in enumerate(projectors):
        if projector.index != 0:
            continue                     # the scattering wave is unoccupied
        occupation = float(dataset.channels[projector.l].occupation)
        D0[i, i] = occupation / (2 * projector.l + 1)
    return D0
