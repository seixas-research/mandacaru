# -*- coding: utf-8 -*-
# file: pseudopotentials/local_split.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Range separation of the local pseudopotential, so its matrix leaves the grid.

Every integral of the real-space engine is a grid sum, so rigidly translating a
molecule by a fraction of the spacing :math:`h` changes the energy -- the
"egg-box" -- and the forces faithfully differentiate that artifact.  PAW-LCAO's
*nonlocal* integrals were cured by integrating them on atom-centered spherical
quadratures instead (:func:`~.paw.atom_centered_projections`), which depend only
on the separation of the two functions and are therefore exactly translation
invariant.  This module does the same for the **local potential matrix**

.. math::

    V_{pq} = \sum_A \int \tilde\phi^*_p(\mathbf r)\,\tilde\phi_q(\mathbf r)\,
             v_A(|\mathbf r - \mathbf R_A|)\, d^3r .

The obstacle is that :math:`v_A` is *long ranged*: it decays only as
:math:`-Z^{\rm ion}_A/r`, so it cannot be integrated inside a finite sphere.
It is therefore split into a long- and a short-range part, the potential of a
Gaussian ion plus the remainder,

.. math::

    v^{\rm lr}_A(r) = -Z^{\rm ion}_A\,\frac{\mathrm{erf}(r/\sqrt2\,\sigma)}{r},
    \qquad v^{\rm sr}_A = v_A - v^{\rm lr}_A .

* :math:`v^{\rm lr}` carries the whole tail and is **smooth**: its Fourier
  transform is :math:`-4\pi Z/G^2\,e^{-G^2\sigma^2/2}`, so choosing
  :math:`\sigma = ` :data:`SIGMA_FACTOR` :math:`\times h` puts a weight of
  :math:`e^{-(1.4\pi)^2/2} \approx 6\times10^{-5}` at the grid's Nyquist
  wave-vector :math:`\pi/h`.  The grid resolves it, so it stays on the grid --
  it simply replaces the full potential in what the engine samples.
* :math:`v^{\rm sr}` has **compact support**: beyond the dataset's local cutoff
  :math:`v_A = -Z/r` exactly, so there
  :math:`v^{\rm sr} = -Z\,\mathrm{erfc}(r/\sqrt2\,\sigma)/r`, which is
  :math:`7\times10^{-11}\,Z/r` at :data:`RADIUS_SIGMAS` :math:`= 6.5\,\sigma`.
  Its matrix is integrated on an atom-centered spherical product quadrature of
  that radius, exactly as the projectors' is.

The split is *exact*: :math:`v^{\rm sr}` is evaluated as the difference of the
dataset's own :meth:`~.paw.PAWDataset.local_potential` and :math:`v^{\rm lr}`,
never from the asymptotic form, so the only approximations are the quadrature
order and the truncation at :data:`RADIUS_SIGMAS`.

What it is worth, measured
--------------------------
Water in PAW-LCAO-SZ (10 Angstrom cell), shifted rigidly by fractions of :math:`h`
along :math:`(1,1,1)/\sqrt3` with the molecular orbitals and the density
frozen, one term at a time (peak-to-peak, meV).  :math:`h` is the grid
spacing (``grid_spacing``) in Angstrom, ``V_grid`` / ``V_split`` the local-potential term with
``exact_local_potential`` off and on, ``E_grid`` / ``E_split`` the total:

======  ======  ======  ========  ========  ======  ========  ========
     h       S       T    V_grid   V_split     eri    E_grid   E_split
======  ======  ======  ========  ========  ======  ========  ========
  0.30     288    1471      1102      2714    2390      3059      1470
  0.25     104      50       634       683     533        61        57
  0.20       8      86        55        14      26       105        44
  0.16       3      23        24        23      19        26        24
======  ======  ======  ========  ========  ======  ========  ========

Read that honestly, because it does not say what one would hope.

* The nonlocal and compensation terms are already 0.000 meV -- the atom-centered
  quadratures of :mod:`~.paw`.  ``V_loc`` is **not** the dominant remaining
  term: the finite-difference kinetic stencil (these numbers were measured
  with ``kinetic="fd"``) and the grid Coulomb tensor are
  of the same size and larger at some spacings, and the total is set by how the
  three happen to cancel.
* The half that moves to the sphere is exactly translation invariant
  (3e-16), but the long-range half **stays on the grid and keeps an egg-box of
  its own** -- larger than the full potential's at grid_spacing >= 0.25 Angstrom, because
  a Gaussian ion of width 1.4 h is a narrower well than the pseudized channel
  it replaces (-7.2 Ha at the origin against -5.4 Ha for oxygen at grid_spacing =
  0.25).
* Widening sigma shrinks that term monotonically but never to zero (oxygen at
  grid_spacing = 0.30 Angstrom: 3425 / 2714 / 2097 / 1501 / 1159 meV at
  ``SIGMA_FACTOR`` = 1 / 1.4 / 2 / 3 / 4), because what is left is the sampling
  of the **pair density**, not of the potential -- and it makes the *total*
  worse by breaking the cancellation against the electron-repulsion term
  (774 / 1470 / 2073 / 2654 / 2989 meV over the same sweep).

What it does buy: the total egg-box is 1.0-2.1x smaller at the default
``SIGMA_FACTOR`` and was not larger at any spacing measured; the net force on
water falls from 2.39 to 1.80 eV/Angstrom at grid_spacing = 0.25; and the analytic
gradient agrees with a central difference of the calculator's own energy 3.5x
better (2.6e-3 -> 7.5e-4 eV/Angstrom), because one more term is differentiated
analytically instead of through a grid sum.  The energy itself moves by
<= 4e-5 Hartree on the pinned H2 and LiH cases and the convergence with ``grid_spacing``
is unchanged (split minus grid is <= 1 meV for grid_spacing <= 0.16 Angstrom on H2 and
water).  Treat it as one term integrated exactly instead of approximately, not
as a cure for the egg-box.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

from ..integrals.sampling import sample_functions

#: Width of the Gaussian ion, in units of the grid spacing (Bohr).  At 1.4 the
#: long-range potential's Fourier weight at the Nyquist wave-vector ``pi/h`` is
#: ``exp(-(1.4 pi)^2/2) = 6e-5`` of its ``G -> 0`` value, so the grid resolves
#: it and the term that stays on the grid carries no egg-box worth measuring.
SIGMA_FACTOR = 1.4

#: Radius of the short-range sphere, in units of ``sigma``.  ``erfc(6.5/sqrt2)
#: = 7.3e-11``, so the neglected tail of ``v^sr`` is ~1e-10 Hartree/Bohr times
#: the ionic charge -- far below the 1e-6 Hartree the quadrature targets.
RADIUS_SIGMAS = 6.5

#: Hartree.  ``v^sr`` is taken to vanish where the dataset's local potential is
#: within this of the ionic tail ``-Z/r``; table noise beyond the tail is ~3e-7.
TAIL_TOLERANCE = 1e-5

#: Bohr.  The table is scanned out to here; beyond it ``local_potential`` is
#: ``-Z/r`` by construction.
TAIL_SCAN_MAX = 8.0

#: Gauss-Legendre points per **radial panel**.  The interval is split at the
#: dataset's local cutoff, where the pseudized polynomial joins the ionic tail
#: and the third derivative jumps; a single panel across that junction
#: converges algebraically instead of exponentially.
#:
#: The orders below are what a convergence study on the *energy* asked for --
#: water PAW-LCAO-DZ at grid_spacing = 0.25 Angstrom (12 orbitals, the oxygen sphere holding
#: both hydrogens), ``tr(D V^sr)`` against a 240x64x128 reference:
#: ``20x16x32`` +2.2e-6, ``32x24x48`` +7.9e-7, ``40x32x64`` +1.8e-7,
#: ``64x32x64`` ~3e-7, ``80x40x80`` -9.5e-8 Hartree.  Below ~32 angular points
#: the error is angular (a neighbor's orbital seen off-center); above it the
#: radial rule dominates.  PAW-LCAO-SZ water converges to ~1e-7 and H2 PAW-LCAO-DZP to
#: better than 1e-9, so these are sized by the hardest case tested.
RADIAL_POINTS = 64
#: Gauss-Legendre points in ``cos(theta)``.
POLAR_POINTS = 32
#: Equispaced azimuthal points (a trapezoid rule, spectrally accurate on the
#: periodic angle).
AZIMUTHAL_POINTS = 64

#: The crystal's short-range spheres (:meth:`~.periodic_paw.PeriodicPAW.
#: short_range_atom`): a sweep of the *self-consistent* results (HISTORY,
#: "Phase L: sphere quadrature sweep") showed the rule above 8x larger than
#: they need -- 32x16x32 moves displaced Si, displaced rocksalt LiH and water
#: by <= 6e-4 meV per atom, the forces by <= 2e-3 meV/Angstrom and the
#: eigenvalues by <= 3e-3 meV, below the 3e-7 Ha the larger rule carries
#: against the reference -- while a crystal's build and forces, which those
#: spheres dominate, run 2.2x faster (262k -> 33k points per atom).
#: Molecules retain their original nodes and weights to preserve numerical
#: results and orbital tracking. Their basis values now use the same native
#: shell evaluator as the crystal, sharing spline interpolation among m
#: components instead of evaluating every function separately in Python.
#: The azimuthal count is a multiple of 12, so the rule (polar axis z,
#: azimuths at half steps from x) maps onto itself under the rotations about
#: z by multiples of 30 and 90 degrees and the mirrors and two-fold axes
#: between them -- the hexagonal and tetragonal point groups with c along z.
#: With 32, the short-range matrices of one hcp Mg star split by 2.7e-8 Ha
#: and a wedge missed the full mesh by 1.2e-8 Ha (HISTORY 2026-10-11, K13b).
#: The cube's three-fold axes still move it.
CRYSTAL_RADIAL_POINTS = 32
CRYSTAL_POLAR_POINTS = 16
CRYSTAL_AZIMUTHAL_POINTS = 36

#: Radial shells evaluated at once.  The basis is sampled on
#: ``block x n_angular`` points at a time, which bounds the working set at a
#: few MB however large the sphere or the basis is.
SHELL_BLOCK = 8

#: Step (Bohr) of the central difference that moves a basis function in
#: :func:`short_range_gradients`, matching
#: :data:`~mandacaru.algorithms.forces.DEFAULT_ORBITAL_DELTA`.
DEFAULT_DELTA = 1e-3


def grid_spacing(grid) -> float:
    """The grid's coarsest node spacing in **Bohr**.

    The lattice-vector norms of :attr:`~mandacaru.integrals.Grid.step`, so an
    anisotropic or skewed grid is sized by its worst-resolved direction rather
    than by ``dx`` alone.
    """
    step = np.asarray(grid.step, dtype=float)
    return float(max(np.linalg.norm(step[:, i]) for i in range(3)))


def split_width(grid) -> float:
    """Gaussian width ``sigma`` (Bohr) of the split on this grid."""
    return SIGMA_FACTOR * grid_spacing(grid)


def long_range_potential(charge: float, sigma: float, radius) -> np.ndarray:
    r"""``-Z erf(r / (sqrt(2) sigma)) / r``, the potential of a Gaussian ion.

    Finite at the origin, where it tends to
    :math:`-Z\sqrt{2/\pi}/\sigma`; the ``r -> 0`` branch is taken explicitly
    rather than left to a ``0/0``.
    """
    radius = np.asarray(radius, dtype=float)
    charge = float(charge)
    sigma = float(sigma)
    limit = -charge * np.sqrt(2.0 / np.pi) / sigma
    safe = np.maximum(radius, 1e-300)
    value = -charge * erf(safe / (np.sqrt(2.0) * sigma)) / safe
    return np.where(radius < 1e-12 * sigma, limit, value)


def long_range_sampler(nuclei, datasets, sigma: float):
    """The ``V(x, y, z)`` callable the integral engine samples with the split on.

    ``nuclei`` are the ``(Z_ion, center)`` pairs of
    :class:`~mandacaru.integrals.potentials.Potentials` (centers in Bohr) and
    ``datasets`` the aligned pseudopotentials, whose ``valence_charge`` is the
    ionic charge whose tail the Gaussian reproduces.  A drop-in replacement for
    :meth:`~mandacaru.integrals.potentials.Potentials.pseudopotential`.
    """
    centers = [np.asarray(center, dtype=float) for _z, center in nuclei]
    charges = [float(dataset.valence_charge) for dataset in datasets]

    def sampler(x, y, z) -> np.ndarray:
        out = np.zeros_like(x, dtype=float)
        for charge, center in zip(charges, centers):
            radius = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2
                             + (z - center[2]) ** 2)
            out += long_range_potential(charge, sigma, radius)
        return out

    return sampler


def short_range_potential(dataset, sigma: float, radius) -> np.ndarray:
    """``v_A - v^lr_A`` of one dataset, by difference (never by an asymptote).

    Taking the difference of the dataset's own
    :meth:`~.paw.PAWDataset.local_potential` is what makes the split exact: the
    sum of what the grid samples and what the quadrature integrates is the
    potential the un-split calculation used, whatever the dataset's tail
    actually does (the shipped tables sit ~5e-7 Hartree off ``-Z/r``).
    """
    radius = np.asarray(radius, dtype=float)
    return (np.asarray(dataset.local_potential(radius), dtype=float)
            - long_range_potential(dataset.valence_charge, sigma, radius))


def long_range_potential_derivative(charge: float, sigma: float,
                                    radius) -> np.ndarray:
    r"""``d/dr`` of :func:`long_range_potential`; zero at the origin (the
    potential is even there)."""
    radius = np.asarray(radius, dtype=float)
    charge, sigma = float(charge), float(sigma)
    safe = np.maximum(radius, 1e-300)
    a = 1.0 / (np.sqrt(2.0) * sigma)
    value = -charge * (2.0 / np.sqrt(np.pi) * a * np.exp(-(a * safe) ** 2)
                       / safe - erf(a * safe) / safe ** 2)
    return np.where(radius < 1e-12 * sigma, 0.0, value)


def short_range_potential_derivative(dataset, sigma: float, radius) -> np.ndarray:
    """``d/dr`` of :func:`short_range_potential`, analytic (the dataset's
    spline derivative less the Gaussian ion's)."""
    radius = np.asarray(radius, dtype=float)
    return (np.asarray(dataset.local_potential_derivative(radius), dtype=float)
            - long_range_potential_derivative(dataset.valence_charge, sigma,
                                              radius))


def local_cutoff(dataset) -> float:
    """Radius (Bohr) beyond which the dataset's local potential is the ionic tail.

    The generation's ``r_cut_local`` when it is recorded (PAW-LCAO), else
    the largest channel cutoff -- only a panel boundary for the radial rule, so
    a loose value costs accuracy, never correctness.
    """
    cutoff = float(getattr(dataset, "r_cut_local", 0.0) or 0.0)
    if cutoff <= 0.0:
        cutoff = max([float(getattr(channel, "r_cut", 0.0) or 0.0)
                      for channel in getattr(dataset, "channels", {}).values()]
                     or [0.0])
    return cutoff


def tail_radius(dataset) -> float:
    """Radius (Bohr) beyond which the dataset's local potential is the ionic
    tail ``-Z/r`` to :data:`TAIL_TOLERANCE`, measured on its own table.

    ``r_cut_local`` is where the generation's polynomial ends, not where the
    tail begins: the shipped Cd table is still 2e-2 Ha off ``-Z/r`` just
    outside it and joins the tail near 2.2 Bohr.  Cached on the dataset.
    """
    cached = getattr(dataset, "_tail_radius", None)
    if cached is None:
        r = np.linspace(0.3, TAIL_SCAN_MAX, 7701)
        deviation = np.abs(dataset.local_potential(r)
                           + float(dataset.valence_charge) / r)
        outside = np.flatnonzero(deviation > TAIL_TOLERANCE)
        cached = float(r[outside[-1]] + 0.01) if outside.size else 0.0
        dataset._tail_radius = cached
    return cached


def short_range_radius(dataset, sigma: float) -> float:
    """Radius (Bohr) of the sphere that holds all of ``v^sr`` for this dataset.

    :data:`RADIUS_SIGMAS` widths of the Gaussian, but never less than the
    dataset's own local cutoff or its measured tail (:func:`tail_radius`): on a
    fine grid ``sigma`` shrinks with ``grid_spacing`` while the pseudized region
    does not, and truncating inside it throws away the part of ``v^sr`` that is
    *not* the erfc tail (Li's local cutoff is 2.34 Bohr against
    ``6.5 sigma = 1.7`` Bohr at grid_spacing = 0.10 Angstrom; Cd's table joins
    the tail at 2.2 Bohr, past its 1.98 Bohr cutoff, a staircase of
    0.1-0.4 eV/atom below grid_spacing 0.13 Angstrom).
    """
    return max(RADIUS_SIGMAS * float(sigma), local_cutoff(dataset),
               tail_radius(dataset))


# --------------------------------------------------------------------------- #
# The quadrature.
# --------------------------------------------------------------------------- #

def _radial_rule(radius: float, panel: float, points: int):
    """Gauss-Legendre nodes and ``w r^2`` weights on ``[0, radius]``.

    Split into two panels at ``panel`` when that lies strictly inside, so the
    kink where the pseudized polynomial meets the ionic tail falls on a panel
    boundary instead of inside one.
    """
    x, w = np.polynomial.legendre.leggauss(int(points))
    edges = ([0.0, float(panel), float(radius)]
             if 0.0 < float(panel) < float(radius) else [0.0, float(radius)])
    nodes, weights = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        half = 0.5 * (hi - lo)
        nodes.append(half * (x + 1.0) + lo)
        weights.append(half * w)
    r = np.concatenate(nodes)
    return r, np.concatenate(weights) * r * r


def _angular_rule(polar: int, azimuthal: int):
    """Unit directions ``(3, n)`` and their solid-angle weights ``(n,)``."""
    t, wt = np.polynomial.legendre.leggauss(int(polar))
    n_phi = int(azimuthal)
    azimuth = 2.0 * np.pi * (np.arange(n_phi) + 0.5) / n_phi
    sin_t = np.sqrt(1.0 - t * t)
    directions = np.stack([(sin_t[:, None] * np.cos(azimuth)[None, :]).ravel(),
                           (sin_t[:, None] * np.sin(azimuth)[None, :]).ravel(),
                           np.repeat(t, n_phi)])
    return directions, np.repeat(wt, n_phi) * (2.0 * np.pi / n_phi)


_SPHERE_OFFSET_CACHE: dict = {}


def _sorted_sphere_chunks(radii: np.ndarray, directions: np.ndarray,
                          chunk_size: int):
    key = (float(radii[0]), float(radii[-1]), int(radii.size),
           int(directions.shape[1]), float(directions[0, 0]), int(chunk_size))
    cached = _SPHERE_OFFSET_CACHE.get(key)
    if cached is None:
        ox = (radii[:, None] * directions[0][None, :]).ravel()
        oy = (radii[:, None] * directions[1][None, :]).ravel()
        oz = (radii[:, None] * directions[2][None, :]).ravel()
        octant = ((oz >= 0.0).astype(int) * 4
                  + (oy >= 0.0).astype(int) * 2
                  + (ox >= 0.0).astype(int))
        order = np.lexsort((np.floor(oz), np.floor(oy), np.floor(ox), octant))
        ox_s = np.ascontiguousarray(ox[order])
        oy_s = np.ascontiguousarray(oy[order])
        oz_s = np.ascontiguousarray(oz[order])
        chunks = []
        for start in range(0, ox_s.size, int(chunk_size)):
            stop = min(start + int(chunk_size), ox_s.size)
            cx = ox_s[start:stop]
            cy = oy_s[start:stop]
            cz = oz_s[start:stop]
            lo = np.array([cx.min(), cy.min(), cz.min()], dtype=float)
            hi = np.array([cx.max(), cy.max(), cz.max()], dtype=float)
            chunks.append((cx, cy, cz, order[start:stop], lo, hi))
        cached = chunks
        _SPHERE_OFFSET_CACHE[key] = cached
    return cached


def _atom_matrices(basis, center, weight, radii, directions, *,
                   gradients: bool, delta: float, block: int,
                   values: bool = True):
    """``I`` (and ``G``) of one sphere, accumulated block of shells at a time."""
    M = len(basis)
    # Real functions (a molecule's real harmonics) give real matrices.
    dtype = (float if all(getattr(fn, "real", False) for fn in basis)
             else complex)
    I = np.zeros((M, M), dtype=dtype) if values else None
    G = np.zeros((M, M, 3), dtype=dtype) if gradients else None
    center = np.asarray(center, dtype=float)
    # Only functions whose support reaches this sphere contribute; the rest
    # are exactly zero on every one of its points.  Without this every atom
    # evaluates every basis function, which made the quadrature N^2 (26100
    # evaluations, 32 s of a 56 s C8H18 setup).
    extra = float(delta) if gradients else 0.0
    reach = float(np.max(radii)) + extra
    near = [m for m, fn in enumerate(basis)
            if getattr(fn, "support_radius", None) is None
            or np.linalg.norm(np.asarray(fn.center, dtype=float) - center)
            <= fn.support_radius + reach]
    if not near:
        return I, G
    basis = [basis[m] for m in near]
    n_near = len(near)
    I_near = np.zeros((n_near, n_near), dtype=dtype) if values else None
    G_near = np.zeros((n_near, 3, n_near), dtype=dtype) if gradients else None
    rel_centers = np.array([np.asarray(fn.center, dtype=float) - center
                            for fn in basis], dtype=float)
    reach2 = np.array([
        np.inf if getattr(fn, "support_radius", None) is None
        else (float(fn.support_radius) + extra) ** 2
        for fn in basis
    ], dtype=float)
    chunk_size = max(16384, int(block) * int(directions.shape[1]))
    w_flat = weight.ravel()
    for ox, oy, oz, order_c, lo, hi in _sorted_sphere_chunks(
            radii, directions, chunk_size):
        gx = np.maximum(0.0, np.maximum(lo[0] - rel_centers[:, 0],
                                        rel_centers[:, 0] - hi[0]))
        gy = np.maximum(0.0, np.maximum(lo[1] - rel_centers[:, 1],
                                        rel_centers[:, 1] - hi[1]))
        gz = np.maximum(0.0, np.maximum(lo[2] - rel_centers[:, 2],
                                        rel_centers[:, 2] - hi[2]))
        active = np.flatnonzero(gx * gx + gy * gy + gz * gz <= reach2)
        if not active.size:
            continue
        basis_c = ([basis[i] for i in active]
                   if active.size < n_near else basis)
        flat = (center[0] + ox, center[1] + oy, center[2] + oz)
        w_block = w_flat[order_c]
        if not gradients:
            psi, _ = sample_functions(basis_c, flat)
            contrib = (np.conj(psi) * w_block) @ psi.T
            if active.size < n_near:
                I_near[np.ix_(active, active)] += contrib
            else:
                I_near += contrib
            continue
        # d phi / d R_k is minus the function's gradient (as
        # `atom_centered_projection_gradients` has it for the projectors):
        # analytic for the tabulated functions, a step-`delta` central
        # difference for the rest.
        psi, gradient = sample_functions(basis_c, flat, gradients=True,
                                         delta=float(delta))
        n_c = len(basis_c)
        weighted_psi_T = (psi * w_block).T
        if values:
            contrib_I = np.conj(psi) @ weighted_psi_T
            if n_c < n_near:
                I_near[np.ix_(active, active)] += contrib_I
            else:
                I_near += contrib_I
        contrib_G = (np.conj(gradient).reshape(3 * n_c, -1)
                     @ weighted_psi_T).reshape(n_c, 3, n_c)
        if n_c < n_near:
            for k in range(3):
                G_near[active[:, None], k, active[None, :]] -= contrib_G[:, k, :]
        else:
            G_near -= contrib_G
    block_ix = np.ix_(near, near)
    if values:
        I[block_ix] = I_near
    if gradients:
        for k in range(3):
            G[block_ix + (k,)] = G_near[:, k, :]
    return I, G


def short_range_matrices(basis, centers, datasets, sigma: float, *,
                         gradients: bool = False, delta: float = DEFAULT_DELTA,
                         radial: int = RADIAL_POINTS, polar: int = POLAR_POINTS,
                         azimuthal: int = AZIMUTHAL_POINTS,
                         block: int = SHELL_BLOCK,
                         values: bool = True):
    r"""Per-atom short-range local matrices, by atom-centered quadrature.

    Returns ``(I, G)`` with ``I[A][p, q]`` the ``(M, M)`` matrix

    .. math::

        I^A_{pq} = \int \tilde\phi^*_p\,\tilde\phi_q\,
                   v^{\rm sr}_A(|\mathbf r - \mathbf R_A|)\, d^3 r

    integrated over the sphere of :func:`short_range_radius` around
    :math:`\mathbf R_A`, and -- when ``gradients`` is set -- ``G[A][p, q, k]``
    the same integral with :math:`\phi_p` replaced by
    :math:`\partial\phi_p/\partial R_{p,k}`, i.e. **that function's own center**
    moving.  ``G`` is ``None`` otherwise.

    Because the integral depends only on the separations of the functions from
    the sphere's center, it is exactly translation invariant, and the two roles
    a displacement can play are both expressed through the one array ``G``:
    moving a basis function contributes :math:`+G`, moving the sphere (with its
    potential) contributes :math:`-G`.  Their sum over a rigid translation of
    everything is identically zero.

    ``centers`` are in Bohr, aligned with ``datasets``; ``sigma`` is the split
    width (:func:`split_width`).  The quadrature orders default to the module
    constants and are arguments only so a convergence study can raise them.
    """
    directions, w_angular = _angular_rule(polar, azimuthal)
    matrices, derivatives = ({} if values else None), ({} if gradients else None)
    quad_cache = {}
    for atom, dataset in enumerate(datasets):
        ds_key = (id(dataset), float(sigma), int(radial), int(polar), int(azimuthal))
        if ds_key in quad_cache:
            radii, weight = quad_cache[ds_key]
        else:
            radius = short_range_radius(dataset, sigma)
            radii, w_radial = _radial_rule(radius, local_cutoff(dataset), radial)
            potential = short_range_potential(dataset, sigma, radii)
            weight = (w_radial * potential)[:, None] * w_angular[None, :]
            quad_cache[ds_key] = (radii, weight)
        I, G = _atom_matrices(basis, centers[atom], weight, radii, directions,
                              gradients=gradients, delta=delta, block=block,
                              values=values)
        if values:
            matrices[atom] = I
        if gradients:
            derivatives[atom] = G
    return matrices, derivatives
