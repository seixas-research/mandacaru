# -*- coding: utf-8 -*-
# file: pseudopotentials/onecenter.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""The PAW-LCAO one-center electron-electron terms.

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
point.  Mandacaru linearizes it about the isolated atom -- a fixed
:math:`D^{ion}` plus a per-species constant -- which is exact only while
:math:`D` stays at its atomic reference :math:`D^0`.  Forming a bond is
precisely what moves it: charge transfers and the s and p channels rehybridize.
Writing :math:`E(D) = \tfrac12 D D \Delta W` and expanding about :math:`D^0`,

.. math::

    E(D) = \underbrace{E(D^0) + (D - D^0)\,D^0 \Delta W}_{\text{the linearization}}
         + \underbrace{\tfrac12 (D-D^0)(D-D^0)\,\Delta W}_{\text{what was missing}},

so the omission is exactly the second-order term.  The Hartree version of it
(:func:`one_center_coulomb`) is measured machinery, **not** wired into the
Hamiltonian: 0.08-0.33 eV on oxygen, ruled out as the cause of an old binding
failure.

:math:`\Delta W` itself is
:math:`(\phi_a^*\phi_b|\phi_c^*\phi_d) - (\tilde\phi_a^*\tilde\phi_b + \hat n_{ab}
|\tilde\phi_c^*\tilde\phi_d + \hat n_{cd})` over one sphere, assembled from the
usual multipole expansion of :math:`1/r_{12}`: a radial double integral per
:math:`L` (:func:`radial_coulomb`) times the angular couplings of
:mod:`mandacaru.pseudopotentials.multipoles`.

The one-center exact exchange of a screened hybrid
------------------------------------------------------
HSE06 (:mod:`mandacaru.basis.hse`) replaces the fraction :math:`a` of the
short-range semilocal exchange of the valence density by as much short-range
exact exchange.  On PAW-LCAO both halves of that swap have a smooth part and a
one-center part, and **both** one-center parts are needed: either alone moves
the energy by ~0.1 Ha for H2O.

*Exact exchange.*  Exactly as for the Hartree energy, the all-electron exchange
is the smooth exchange of the **augmented** pair densities plus a one-center
correction per sphere,

.. math::

    E_x^{\rm HF,SR} = \tilde E_x^{\rm SR}\big[\tilde\rho_{pr} + \hat\rho_{pr}\big]
      - \frac14\sum_A \sum_{ijkl} D^A_{li} D^A_{jk}\,
        \Delta W^{\rm SR}_A[i,j,k,l]

(closed shell, both spins in :math:`D`; a spin channel alone carries
:math:`-\tfrac12`).  The first term is the grid tensor with the compensation
charges under the erfc kernel
(:meth:`~mandacaru.core.hamiltonian.MolecularIntegrals.short_range_two_body`,
:meth:`~mandacaru.pseudopotentials.paw.PAWIntegrals.long_range_augmentation`);
:math:`\Delta W^{\rm SR}` is :math:`\Delta W` with
:math:`\operatorname{erfc}(\omega r_{12})/r_{12}` (:func:`one_center_exchange_tensor`).
:math:`D^A = C_A^\dagger D C_A` is the projected density matrix,
:math:`C_{pi} = \langle\chi_p|\tilde p_i\rangle`, and the exchange matrix it
adds to the operator is :math:`C_A\,\Delta K\,C_A^\dagger` with
:math:`\Delta K_{il} = \sum_{jk} D^A_{jk}\,\Delta W^{\rm SR}[i,j,k,l]` -- the
one-center image of the molecular :math:`K_{pq} = \sum_{rs} D_{sr}
\langle pr|sq\rangle`.  It is kept **quadratic**, not linearized: an exchange
operator frozen at the spherical reference atom would shift occupied and empty
orbitals of a channel alike, which is the opposite of what exact exchange does.

*Semilocal exchange.*  The functional subtracts
:math:`a E_x^{\omega{\rm PBE,SR}}` of the valence density, whose PAW form is
:math:`E[\tilde n_v] + \sum_A (E[n^1_A] - E[\tilde n^1_A])` -- valence only,
no compensation charge, as for any semilocal term.  The grid takes the first
part.  The one-center part is exactly what the datasets **freeze**: their
one-center semilocal energies are linearized at the reference atom (the
constant ``one_center_energy`` and its derivative inside :math:`D^{ion}`).
The hybrid's share of them is frozen the same way,

.. math::

    \Delta X_A(D) = X^0_A + \sum_{ij} V^A_{ij}\,(D^A_{ij} - D^0_{ij}),
    \qquad
    X^0_A = E_x^{\rm SR}[n^1_0] - E_x^{\rm SR}[\tilde n^1_0],

    V^A_{ij} = \int v_x^{\rm SR}[n^1_0]\,\phi_i^*\phi_j
             - \int v_x^{\rm SR}[\tilde n^1_0]\,\tilde\phi_i^*\tilde\phi_j ,

with :math:`n^1_0` and :math:`\tilde n^1_0` the reference atom's all-electron
and smooth valence densities (spherical, so :math:`V` is diagonal in
:math:`lm`; the GGA term enters through :math:`2\,\partial f/\partial\sigma\,
n_0'\,(R_iR_j)'`), and :math:`D^0` its one-center density matrix (the
channel occupation on the bound wave).  The semilocal exchange is the hole
model of :mod:`mandacaru.basis.hse` (the piece HSE06 removes from PBE),
whatever functional the dataset was generated with, so that the change
HSE06 - PBE is the hybrid's own.

*Per atom, all together*, the hybrid adds to the frozen one-center energy

.. math::

    E^A_{\rm hyb}(D) = -a\big[X^0_A - V^A\!\cdot D^0\big]
      - a\,V^A\!\cdot D^A
      - \frac a4 \sum D^A_{li} D^A_{jk}\,\Delta W^{\rm SR}_A[i,j,k,l],

a per-species constant, a one-body term :math:`-a\,C_A V^A C_A^\dagger`
(both spins alike: the derivative of the spin-scaled functional at the
unpolarized reference) and the quadratic exchange.  :class:`OneCenterHybrid`
evaluates the three on the ``(P, P)`` projected density matrix, so a crystal
(:math:`D^A = \sum_k w_k C_A(k)^\dagger P^k C_A(k)`) uses it unchanged.
:math:`a \to 0` removes everything; :math:`\omega \to 0` turns the kernel into
:math:`1/r_{12}` and :math:`X^0` into the full hole-model exchange (a PBE0-like
hybrid); :math:`\omega \to \infty` removes everything.  The unitary datasets
(UPAW-LCAO, :math:`q = 0`) take the same path: their spheres carry no monopole
but still the higher multipoles and the full one-center difference.

The kernel.  :math:`\operatorname{erfc}(\omega r)/r = 1/r - \operatorname{erf}
(\omega r)/r`; the bare part keeps the closed-form radial kernel
:math:`r_<^L/r_>^{L+1}`, and the long-range part's multipole components
:math:`f_L(r, r') = \tfrac{2L+1}2\int_{-1}^1
\operatorname{erf}(\omega R)/R\,P_L(\mu)\,d\mu`,
:math:`R^2 = r^2 + r'^2 - 2rr'\mu`, are integrated by Gauss-Legendre in
:math:`\mu` (:func:`long_range_radial_kernel`).  :math:`\operatorname{erf}(x)/x`
is an entire function of :math:`x^2`, hence of :math:`\mu`, so the rule
converges exponentially (24 nodes: round-off, inside a sphere).  Inside an
augmentation sphere :math:`\omega r \lesssim 0.3`, where the series
:math:`2\omega/\sqrt\pi - (2\omega^3/3\sqrt\pi)R^2 + \dots` would also do:
its constant cancels between the two densities (they share every moment) and
the :math:`R^2` term is the whole correction to the bare :math:`\Delta W`
at the 1e-6 level (HISTORY.md, 2026-10-03).  The exact kernel costs nothing
measurable, so it is what is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson

from .multipoles import gaunt, multipole_range, partial_waves, shape_function

#: Radial points of the one-center quadrature (uniform, inside the sphere).
ONE_CENTER_POINTS = 600
#: Gauss-Legendre nodes of the long-range kernel's angular projection, at
#: least; the rule grows with omega times the largest radius
#: (:data:`KERNEL_POINTS_PER_OMEGA_R`).
KERNEL_POINTS = 24
#: Extra nodes per unit of omega r: near mu = 1 the integrand varies on a
#: scale ~ 1/(omega^2 r r'), which a fixed rule misses once omega r >> 1
#: (omega = 1000: the spheres' exact exchange was -7.6e-6 Ha instead of 0).
KERNEL_POINTS_PER_OMEGA_R = 8
#: Gauss-Legendre nodes over a compensation shape (its long-range potential).
SHAPE_POINTS = 48
#: Radial step (Bohr) of the long-range compensation potential table.
SHAPE_TABLE_STEP = 0.02


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


def long_range_radial_kernel(r1, r2, levels, omega: float,
                             points: int | None = None) -> dict:
    r"""``{L: f_L}`` -- the multipole components of
    :math:`\operatorname{erf}(\omega|\mathbf r - \mathbf r'|)/|\mathbf r -
    \mathbf r'|` between the radii ``r1`` and ``r2``.

    :math:`f_L(r, r') = \tfrac{2L+1}2\int_{-1}^1 \operatorname{erf}(\omega R)
    /R\,P_L(\mu)\,d\mu`, so that the kernel is :math:`\sum_L f_L P_L(\mu)` --
    the same normalization as :math:`r_<^L/r_>^{L+1}` in the expansion of
    :math:`1/r_{12}`, with the same angular factors.  Returns
    ``(len(r1), len(r2))`` arrays; ``omega = 0`` gives zeros.
    """
    from scipy.special import erf, eval_legendre

    r1 = np.atleast_1d(np.asarray(r1, dtype=float))
    r2 = np.atleast_1d(np.asarray(r2, dtype=float))
    levels = [int(L) for L in levels]
    omega = float(omega)
    out = {L: np.zeros((r1.size, r2.size)) for L in levels}
    if omega == 0.0 or not levels:
        return out
    if points is None:
        reach = omega * max(float(np.max(r1, initial=0.0)),
                            float(np.max(r2, initial=0.0)))
        points = KERNEL_POINTS + int(np.ceil(KERNEL_POINTS_PER_OMEGA_R
                                             * reach))
    mu, w = np.polynomial.legendre.leggauss(int(points))
    projections = np.stack([0.5 * (2 * L + 1) * w * eval_legendre(L, mu)
                            for L in levels], axis=1)          # (n_mu, n_L)
    limit = 2.0 * omega / np.sqrt(np.pi)
    rows = max(1, int(2e6 // max(1, r2.size * mu.size)))
    for start in range(0, r1.size, rows):
        a = r1[start:start + rows]
        R2 = (a[:, None, None] ** 2 + r2[None, :, None] ** 2
              - 2.0 * a[:, None, None] * r2[None, :, None] * mu[None, None, :])
        R = np.sqrt(np.clip(R2, 0.0, None))
        x = omega * R
        small = x < 1e-4
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(small, limit * (1.0 - x * x / 3.0),
                         erf(x) / np.where(small, 1.0, R))
        block = f @ projections                                  # (a, r2, n_L)
        for i, L in enumerate(levels):
            out[L][start:start + rows] = block[:, :, i]
    return out


_SHAPE_TABLES: dict = {}


def long_range_shape_potential(radius, r_g: float, L: int,
                               omega: float) -> np.ndarray:
    r"""Potential of the compensation multipole :math:`g_L Y_{LM}` under the
    long-range kernel :math:`\operatorname{erf}(\omega r)/r`, without the
    :math:`Y_{LM}` factor -- the screened counterpart of
    :func:`~.multipoles.shape_potential`.

    :math:`v_L(r) = \tfrac{4\pi}{2L+1}\int_0^{r_g} f_L(r, s)\,g_L(s)\,s^2\,ds`
    (:func:`long_range_radial_kernel`), Gauss-Legendre over the shape,
    tabulated every :data:`SHAPE_TABLE_STEP` Bohr and interpolated by a cubic
    spline (the potential is smooth on the scale :math:`1/\omega`).  The
    table grows to the largest radius ever asked for.
    """
    from scipy.interpolate import CubicSpline

    radius = np.asarray(radius, dtype=float)
    L, r_g, omega = int(L), float(r_g), float(omega)
    if omega == 0.0:
        return np.zeros_like(radius)
    reach = float(np.max(radius, initial=0.0)) + 1.0
    key = (round(r_g, 12), L, omega)
    cached = _SHAPE_TABLES.get(key)
    if cached is None or cached[0] < reach:
        r_max = max(reach, 2.0 * r_g, 0.0 if cached is None else cached[0])
        table = np.arange(0.0, r_max + SHAPE_TABLE_STEP, SHAPE_TABLE_STEP)
        x, wx = np.polynomial.legendre.leggauss(SHAPE_POINTS)
        s = 0.5 * r_g * (x + 1.0)
        weight = 0.5 * r_g * wx * s * s * shape_function(s, r_g, L)
        kernel = long_range_radial_kernel(table, s, [L], omega)[L]
        values = 4.0 * np.pi / (2 * L + 1) * (kernel @ weight)
        cached = (float(table[-1]), CubicSpline(table, values))
        _SHAPE_TABLES[key] = cached
    return cached[1](radius)


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
        ae_r = [np.interp(r, source, w) for w in ae]
        ps_r = [np.interp(r, source, w) for w in ps]
        transform = _basis_transform(dataset, l, basis)
        if transform is not None:
            ae_r = [sum(transform[i, k] * ae_r[k] for k in range(len(ae_r)))
                    for i in range(len(ae_r))]
            ps_r = [sum(transform[i, k] * ps_r[k] for k in range(len(ps_r)))
                    for i in range(len(ps_r))]
        tables[l] = (ae_r, ps_r)
    return r, tables, r_cut


def _basis_transform(dataset, l: int, basis: str):
    r"""The matrix taking the dual partial waves to ``basis``'s, or ``None``.

    In the ``"raw"`` projector basis the partial waves that multiply
    :math:`\langle\chi_k|` are :math:`\sum_i (B^{-1})_{ki}\phi_i`
    (:meth:`~.paw.PAWDataset.projector_set`)."""
    if basis == "raw":
        return np.linalg.inv(np.asarray(dataset.channels[int(l)].vanderbilt,
                                        dtype=float))
    if basis != "dual":
        raise ValueError(f"unknown projector basis {basis!r}")
    return None


def _one_center_tensor(dataset, projectors, basis: str, omega=None,
                       kernel=None) -> np.ndarray:
    r"""``Delta W[a, b, c, d]`` under :math:`1/r_{12}` (``omega=None``) or
    :math:`\operatorname{erfc}(\omega r_{12})/r_{12}`.

    ``kernel`` replaces the long-range radial kernel (a callable ``(r, L) ->
    (n, n)`` matrix), which is how the truncated series were measured against
    the exact one.  Radial integrals and angular factors are computed once per
    distinct radial pair and ``(l, m)`` pattern.
    """
    r, tables, _r_cut = _radial_tables(dataset, basis)
    r_g = float(dataset.compensation_radius)
    n = len(projectors)
    levels = sorted({L for a in projectors for b in projectors
                     for L in multipole_range(a.l, b.l)})
    long_range = None
    if kernel is not None:
        long_range = {L: kernel(r, L) for L in levels}
    elif omega is not None and float(omega) > 0.0:
        long_range = long_range_radial_kernel(r, r, levels, float(omega))
    weights = simpson(np.eye(r.size), x=r) * r * r if long_range else None

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

    radial_cache: dict = {}

    def radial(a, b, c, d, L):
        key = ((a.l, a.index, b.l, b.index), (c.l, c.index, d.l, d.index),
               int(L))
        if key not in radial_cache:
            ae_ab, ps_ab = pair(a, b, L)
            ae_cd, ps_cd = pair(c, d, L)
            value = (radial_coulomb(r, ae_ab, ae_cd, L)
                     - radial_coulomb(r, ps_ab, ps_cd, L))
            if long_range is not None:
                K = long_range[int(L)]
                value -= (float((ae_ab * weights) @ K @ (ae_cd * weights))
                          - float((ps_ab * weights) @ K @ (ps_cd * weights)))
            radial_cache[key] = value
        return radial_cache[key]

    angular_cache: dict = {}

    def angular(L, a, b, c, d):
        key = (int(L), a.l, a.m, b.l, b.m, c.l, c.m, d.l, d.m)
        if key not in angular_cache:
            angular_cache[key] = angular_coupling(L, a, b, c, d)
        return angular_cache[key]

    out = np.zeros((n, n, n, n), dtype=complex)
    for ia, a in enumerate(projectors):
        for ib, b in enumerate(projectors):
            levels_ab = multipole_range(a.l, b.l)
            for ic, c in enumerate(projectors):
                for id_, d in enumerate(projectors):
                    total = 0.0 + 0.0j
                    for L in multipole_range(c.l, d.l):
                        if L not in levels_ab:
                            continue
                        factor = angular(L, a, b, c, d)
                        if factor == 0:
                            continue
                        total += factor * radial(a, b, c, d, L)
                    out[ia, ib, ic, id_] = total
    return out


def _projector_basis(projectors, basis):
    if basis is not None:
        return str(basis)
    from .paw import DEFAULT_PROJECTOR_BASIS
    return str(getattr(projectors[0], "projector_basis",
                       DEFAULT_PROJECTOR_BASIS))


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
    return _one_center_tensor(dataset, projectors, basis)


def one_center_exchange_tensor(dataset, projectors, omega: float,
                               basis: str | None = None) -> np.ndarray:
    r"""``DeltaW^SR[a, b, c, d]`` -- :func:`one_center_coulomb` under the
    short-range kernel :math:`\operatorname{erfc}(\omega r_{12})/r_{12}`.

    The one-center correction of a screened hybrid's exact exchange (see the
    module docstring).  ``basis`` defaults to the projectors' own
    ``projector_basis``.  Cached on the dataset per ``(omega, basis)`` and
    projector layout; ``omega = 0`` is :func:`one_center_coulomb`,
    symmetrized.

    Symmetrized under the exchange of the two electrons,
    :math:`[ab|cd] = [cd|ab]`: the cumulative trapezoid inside
    :func:`radial_coulomb` breaks it at ~1e-5 relative (raw projector
    basis), and the exchange operator is the derivative of the energy only
    for a symmetric tensor.
    """
    omega = float(omega)
    if omega < 0.0:
        raise ValueError(f"omega must be >= 0, got {omega!r}")
    basis = _projector_basis(projectors, basis)
    layout = tuple((p.l, p.m, p.index) for p in projectors)
    cache = dataset.__dict__.setdefault("_one_center_exchange_cache", {})
    key = (omega, basis, layout)
    if key not in cache:
        W = _one_center_tensor(dataset, projectors, basis, omega=omega)
        cache[key] = 0.5 * (W + W.transpose(2, 3, 0, 1))
    return cache[key]


def _occupations(dataset, l: int) -> list[float]:
    """The reference atom's electrons in each partial wave of channel ``l``."""
    channel = dataset.channels[int(l)]
    return [float(o) for o in (getattr(channel, "occupations", None)
                               or [channel.occupation])]


def _radial_derivative(r, f):
    return np.gradient(np.asarray(f, dtype=float), np.asarray(r, dtype=float))


def frozen_exchange_terms(dataset, omega: float, basis: str = "raw"):
    r"""``(X0 - V.D0, {l: V_l})`` -- the reference atom's one-center
    short-range semilocal exchange, linearized.

    :math:`X^0 = E_x^{\rm SR}[n^1_0] - E_x^{\rm SR}[\tilde n^1_0]` and
    :math:`V_{ij} = \int v_x^{\rm SR}[n^1_0]\phi_i\phi_j -
    \int v_x^{\rm SR}[\tilde n^1_0]\tilde\phi_i\tilde\phi_j` per channel, in
    the hole model of :func:`~mandacaru.basis.hse.short_range_partials`
    (unpolarized, valence densities of the reference atom).  The first entry
    is the constant of the linearization, :math:`X^0 - \sum V D^0`; the blocks
    are in the projector ``basis`` (the same for every ``m`` of a channel).
    A hybrid with fraction :math:`a` adds :math:`-a` times both
    (:class:`OneCenterHybrid`).  Cached on the dataset per ``(omega, basis)``.
    """
    from ..basis.hse import short_range_partials

    omega = float(omega)
    cache = dataset.__dict__.setdefault("_frozen_exchange_cache", {})
    key = (omega, str(basis))
    if key in cache:
        return cache[key]
    r = np.asarray(dataset.r, dtype=float)
    waves = {l: partial_waves(dataset, l) for l in sorted(dataset.channels)}
    shell = 4.0 * np.pi * r * r
    densities = []
    for side in (0, 1):                       # all-electron, smooth
        n = np.zeros_like(r)
        for l, pair in waves.items():
            # Every occupied partial wave: two in a semicore channel.
            for i, occupation in enumerate(_occupations(dataset, l)):
                n += occupation * pair[side][i] ** 2 / (4.0 * np.pi)
        densities.append(n)
    energy, potentials = [], []
    for n in densities:
        slope = _radial_derivative(r, n)
        f, df_drho, df_dsigma = short_range_partials(n, slope * slope, omega)
        energy.append(float(np.trapezoid(f * shell, r)))
        potentials.append((df_drho, 2.0 * df_dsigma * slope))
    blocks, linear = {}, 0.0
    for l, pair in waves.items():
        size = len(pair[0])
        V = np.zeros((size, size))
        for side, sign in ((0, 1.0), (1, -1.0)):
            local, gradient = potentials[side]
            for i in range(size):
                for j in range(size):
                    product = pair[side][i] * pair[side][j]
                    integrand = (local * product + gradient
                                 * _radial_derivative(r, product)) * r * r
                    V[i, j] += sign * float(np.trapezoid(integrand, r))
        V = 0.5 * (V + V.T)
        linear += sum(occupation * V[i, i] for i, occupation
                      in enumerate(_occupations(dataset, l)))
        transform = _basis_transform(dataset, l, str(basis))
        blocks[l] = V if transform is None else transform @ V @ transform.T
    cache[key] = (energy[0] - energy[1] - linear, blocks)
    return cache[key]


@dataclass(frozen=True)
class OneCenterTerms:
    """The one-center hybrid terms at one projected density matrix.

    ``energy`` is everything the spheres add (Hartree): ``exact_exchange``,
    the quadratic one-center exact exchange, plus the linearized semilocal
    term (``semilocal``, constant included).  ``operators`` holds one
    ``(P, P)`` matrix per spin channel evaluated (one for a closed shell),
    :math:`\\partial E/\\partial D^T`, which enters a Hamiltonian as
    :math:`C\\,O\\,C^\\dagger`.
    """

    energy: float
    exact_exchange: float
    semilocal: float
    operators: tuple


class OneCenterHybrid:
    r"""The PAW-LCAO one-center terms of a screened hybrid, in projector space.

    Parameters
    ----------
    projectors : list
        Every :class:`~mandacaru.pseudopotentials.orbitals.KBProjector` of
        the system, in the order the projections ``C[:, p]`` index them (the
        integrals' ``kb_projectors``; the crystal's ``PeriodicPAW.projectors``).
    datasets : sequence
        One dataset per atom, indexed by ``projector.atom_index``.
    omega, fraction : float
        The hybrid's range separation (1/Bohr) and exact-exchange fraction.

    :meth:`evaluate` takes the ``(P, P)`` projected density matrix
    :math:`D = C^\dagger R\,C` -- for a crystal
    :math:`\sum_k w_k C(k)^\dagger P^k C(k)` -- and returns the energy and
    the ``(P, P)`` operator :math:`O`, block diagonal over atoms, that the
    Hamiltonian gains as :math:`C\,O\,C^\dagger` (at every k for a crystal).
    The bookkeeping is in the module docstring.  :attr:`constant` is the
    per-species frozen constant :math:`-a\sum_A (X^0_A - V^A\cdot D^0_A)`,
    already inside every :meth:`evaluate` energy.
    """

    def __init__(self, projectors, datasets, omega: float, fraction: float):
        self.omega = float(omega)
        self.fraction = float(fraction)
        self.n_projectors = len(projectors)
        positions: dict = {}
        for p, projector in enumerate(projectors):
            positions.setdefault(int(projector.atom_index), []).append(p)
        self.blocks = []
        constant = 0.0
        for atom, members in sorted(positions.items()):
            dataset = datasets[atom]
            own = [projectors[p] for p in members]
            basis = _projector_basis(own, None)
            W = one_center_exchange_tensor(dataset, own, self.omega, basis)
            frozen, per_l = frozen_exchange_terms(dataset, self.omega, basis)
            V = np.zeros((len(own), len(own)), dtype=complex)
            for i, a in enumerate(own):
                for j, b in enumerate(own):
                    if (a.l, a.m) == (b.l, b.m):
                        V[i, j] = per_l[a.l][a.index, b.index]
            constant += frozen
            self.blocks.append((np.asarray(members), W, V))
        #: The per-species frozen constant (Hartree), fraction included.
        self.constant = -self.fraction * constant

    def _exchange(self, D, W):
        """``Delta K_il = sum_jk D_jk W[i, j, k, l]``."""
        return np.einsum("jk,ijkl->il", D, W, optimize=True)

    def evaluate(self, density, density_down=None) -> OneCenterTerms:
        r"""The terms at the projected density matrix ``density``.

        One argument: a closed shell, ``density`` holding both spins; the
        exact exchange is :math:`-\tfrac a4 \sum D_{li}D_{jk}\Delta W` and
        the single operator :math:`-\tfrac a2\Delta K[D] - aV`.  Two: the
        spin channels, :math:`-\tfrac a2` per channel, operators
        :math:`-a\Delta K[D_\sigma] - aV` each.
        """
        a = self.fraction
        spins = ([np.asarray(density)] if density_down is None
                 else [np.asarray(density), np.asarray(density_down)])
        scale = 0.25 if density_down is None else 0.5
        total = sum(spins)
        operators = [np.zeros((self.n_projectors,) * 2, dtype=complex)
                     for _ in spins]
        exact = 0.0
        linear = 0.0
        for members, W, V in self.blocks:
            block = np.ix_(members, members)
            linear += float(np.real(np.sum(V * total[block].T)))
            for s, D in enumerate(spins):
                K = self._exchange(D[block], W)
                exact -= scale * a * float(np.real(np.sum(D[block].T * K)))
                operators[s][block] += -2.0 * scale * a * K - a * V
        semilocal = self.constant - a * linear
        return OneCenterTerms(energy=exact + semilocal, exact_exchange=exact,
                              semilocal=semilocal, operators=tuple(operators))
