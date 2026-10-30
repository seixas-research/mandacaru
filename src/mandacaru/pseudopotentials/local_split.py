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

What lives here now
-------------------
The potentials of the split, the radii that size the short-range sphere, and
the radial and angular rules of its product quadrature.  Every number is
computed in Rust (``paw/local_split.rs``); this module hands it the dataset's
sampled local potential, which the dataset (a spline) evaluates itself.  The
sphere matrices (``short_range_matrices``) belonged to the molecular engine
that integrated the local potential this way; the native molecular path
(``paw/molecular``) owns that now, and the crystal builds its own spheres
(:mod:`.periodic_paw`) from the rules below.
"""

from __future__ import annotations

import numpy as np

from .. import _native
from .native_context import context

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

#: The crystal's short-range spheres (:meth:`~.periodic_paw.PeriodicPAW.
#: short_range_atom`): a sweep of the *self-consistent* results (HISTORY,
#: "Phase L: sphere quadrature sweep") showed the molecular rule 8x larger than
#: they need -- 32x16x32 moves displaced Si, displaced rocksalt LiH and water
#: by <= 6e-4 meV per atom, the forces by <= 2e-3 meV/Angstrom and the
#: eigenvalues by <= 3e-3 meV, below the 3e-7 Ha the larger rule carries
#: against the reference -- while a crystal's build and forces, which those
#: spheres dominate, run 2.2x faster (262k -> 33k points per atom).
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


def _flat(values) -> tuple:
    """``(contiguous 1-D float array, original shape)``."""
    values = np.asarray(values, dtype=float)
    return np.ascontiguousarray(values.ravel()), values.shape


def split_width(grid) -> float:
    """Gaussian width ``sigma`` (Bohr) of the split on this grid: the grid's
    coarsest node spacing (the largest lattice-vector norm of
    :attr:`~mandacaru.integrals.Grid.step`, so an anisotropic or skewed grid is
    sized by its worst-resolved direction) times :data:`SIGMA_FACTOR`."""
    return _native.paw_split_width(
        np.ascontiguousarray(grid.step, dtype=float), SIGMA_FACTOR)


def long_range_potential(charge: float, sigma: float, radius) -> np.ndarray:
    r"""``-Z erf(r / (sqrt(2) sigma)) / r``, the potential of a Gaussian ion.

    Finite at the origin, where it tends to
    :math:`-Z\sqrt{2/\pi}/\sigma`; the ``r -> 0`` branch is taken explicitly
    rather than left to a ``0/0``.
    """
    flat, shape = _flat(radius)
    return _native.paw_long_range_potential(
        context(), float(charge), float(sigma), flat).reshape(shape)[()]


def short_range_potential(dataset, sigma: float, radius) -> np.ndarray:
    """``v_A - v^lr_A`` of one dataset, by difference (never by an asymptote).

    Taking the difference of the dataset's own
    :meth:`~.paw.PAWDataset.local_potential` is what makes the split exact: the
    sum of what the grid samples and what the quadrature integrates is the
    potential the un-split calculation used, whatever the dataset's tail
    actually does (the shipped tables sit ~5e-7 Hartree off ``-Z/r``).
    """
    flat, shape = _flat(radius)
    sampled, _ = _flat(dataset.local_potential(flat))
    return _native.paw_short_range_potential(
        context(), sampled, float(dataset.valence_charge), float(sigma), flat,
        False).reshape(shape)[()]


def short_range_potential_derivative(dataset, sigma: float, radius) -> np.ndarray:
    """``d/dr`` of :func:`short_range_potential`, analytic (the dataset's
    spline derivative less the Gaussian ion's)."""
    flat, shape = _flat(radius)
    sampled, _ = _flat(dataset.local_potential_derivative(flat))
    return _native.paw_short_range_potential(
        context(), sampled, float(dataset.valence_charge), float(sigma), flat,
        True).reshape(shape)[()]


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
        r = _native.paw_linspace(context(), 0.3, TAIL_SCAN_MAX, 7701)
        cached = _native.paw_tail_radius(
            context(), r, _flat(dataset.local_potential(r))[0],
            float(dataset.valence_charge), TAIL_TOLERANCE)
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
    return _native.paw_short_range_radius(
        float(sigma), RADIUS_SIGMAS, local_cutoff(dataset),
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
    return _native.paw_radial_rule(float(radius), float(panel), int(points))


def _angular_rule(polar: int, azimuthal: int):
    """Unit directions ``(3, n)`` and their solid-angle weights ``(n,)``."""
    return _native.paw_angular_rule(int(polar), int(azimuthal))
