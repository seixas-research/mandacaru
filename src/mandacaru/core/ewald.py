# -*- coding: utf-8 -*-
# file: core/ewald.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Ewald summation: the electrostatic energy of point charges in a periodic cell.

A periodic lattice of point charges has a conditionally convergent Coulomb sum,
so it is split (Ewald 1921) into a short-ranged real-space part and a smooth
reciprocal-space part, each absolutely convergent:

.. math::

    E = \underbrace{\tfrac12 \sum_{i \ne j,\ \mathbf R}
            \frac{Z_i Z_j\,\mathrm{erfc}(\eta |\mathbf r_{ij} + \mathbf R|)}
                 {|\mathbf r_{ij} + \mathbf R|}}_{\text{real space}}
      + \underbrace{\frac{2\pi}{V} \sum_{\mathbf G \ne 0}
            \frac{e^{-G^2/4\eta^2}}{G^2}\,|S(\mathbf G)|^2}_{\text{reciprocal}}
      - \underbrace{\frac{\eta}{\sqrt\pi} \sum_i Z_i^2}_{\text{self}}
      - \underbrace{\frac{\pi}{2\eta^2 V}\Big(\sum_i Z_i\Big)^2}_{\text{background}},

with :math:`S(\mathbf G) = \sum_i Z_i e^{i\mathbf G\cdot\mathbf r_i}`.  The last
term is the neutralizing background: the same jellium convention the
plane-wave Hamiltonian uses when it drops the divergent :math:`\mathbf G = 0`
Hartree and electron-nuclear components
(:mod:`mandacaru.core.planewave`).  Only with *both* conventions in place do the
three divergences cancel and the total energy of a neutral cell come out right;
a molecular pair sum in a periodic Hamiltonian does not.

The splitting parameter :math:`\eta` only sets how the work is divided between
the two sums, never the result, and the cutoffs are chosen from the requested
accuracy.  Everything is in atomic units (Bohr, Hartree).
"""

from __future__ import annotations

import numpy as np

#: Terms smaller than this (relative to the charge scale) are dropped.
DEFAULT_ACCURACY = 1e-12


def _ewald_raw(positions, charges, cell, accuracy: float = DEFAULT_ACCURACY,
               eta: float | None = None, points=None, softening: float = 0.0) -> dict:
    from mandacaru import _native

    pos = np.ascontiguousarray(np.asarray(positions, dtype=float).reshape(-1, 3))
    q = np.ascontiguousarray(np.asarray(charges, dtype=float).reshape(-1))
    c = np.ascontiguousarray(np.asarray(cell, dtype=float).reshape(3, 3))
    if pos.shape[0] != q.size:
        raise ValueError(f"{pos.shape[0]} positions for {q.size} charges")
    volume = abs(float(np.linalg.det(c)))
    if volume <= 0.0:
        raise ValueError("the cell has zero volume")
    if q.size == 0:
        p_len = 0 if points is None else np.asarray(points, dtype=float).reshape(-1, 3).shape[0]
        return {
            "energy": 0.0,
            "forces": np.zeros_like(pos),
            "stress": np.zeros((3, 3), dtype=float),
            "potential": np.zeros(p_len, dtype=float),
        }
    pts = None
    if points is not None:
        pts = np.ascontiguousarray(np.asarray(points, dtype=float).reshape(-1, 3))
        if pts.size == 0:
            pts = None
    context = _native.Context(threads=1)
    return context.ewald(pos, q, c, accuracy=accuracy, eta=eta,
                         points_bohr=pts, softening_bohr=float(softening))


def ewald_energy(positions, charges, cell, accuracy: float = DEFAULT_ACCURACY,
                 eta: float | None = None) -> float:
    r"""Electrostatic energy per cell of point charges in a periodic lattice.

    Parameters
    ----------
    positions : (N, 3) array_like
        Cartesian coordinates of the charges, in Bohr.
    charges : (N,) array_like
        Point charges (nuclear charges ``Z``), in units of ``e``.
    cell : (3, 3) array_like
        Lattice vectors as **rows** (the ASE convention), in Bohr.
    accuracy : float
        Target truncation error; sets both cutoffs.
    eta : float, optional
        Splitting parameter (Bohr\ :sup:`-1`).  The default balances the two
        sums for the cell volume and charge count; the total does not depend on
        it.

    Returns
    -------
    float
        The energy in Hartree, including the neutralizing background.  For a
        charge-neutral cell it is the usual Ewald energy; for a charged one the
        background term keeps it finite (and physically it is then a
        jellium-compensated energy, not an isolated-system one).
    """
    return float(_ewald_raw(positions, charges, cell, accuracy=accuracy, eta=eta)["energy"])


def ewald_forces(positions, charges, cell, accuracy: float = DEFAULT_ACCURACY,
                 eta: float | None = None) -> np.ndarray:
    r"""``-dE/dR`` of :func:`ewald_energy`, analytically.

    Differentiating the two convergent sums term by term:

    .. math::

        \frac{\partial E}{\partial \mathbf R_i} =
          -Z_i \sum_{j,\mathbf R}' Z_j
             \left[\frac{\mathrm{erfc}(\eta r)}{r^2}
                   + \frac{2\eta}{\sqrt\pi}\frac{e^{-\eta^2 r^2}}{r}\right]
             \frac{\mathbf r}{r}
          - \frac{4\pi}{\Omega} Z_i \sum_{\mathbf G \neq 0}
             \frac{e^{-G^2/4\eta^2}}{G^2}\,\mathbf G\,
             \mathrm{Im}\!\left[e^{i\mathbf G \cdot \mathbf R_i} S^{*}(\mathbf G)\right] ,

    with :math:`S(\mathbf G) = \sum_j Z_j e^{i\mathbf G \cdot \mathbf R_j}`.
    The self and background terms do not move with the atoms and contribute
    nothing.

    Returns ``(N, 3)`` forces in Hartree/Bohr.  The sum over atoms is zero by
    translational invariance -- an exact identity here, unlike the grid-based
    terms, and worth checking.
    """
    return _ewald_raw(positions, charges, cell, accuracy=accuracy, eta=eta)["forces"]


def ewald_stress(positions, charges, cell, accuracy: float = DEFAULT_ACCURACY,
                 eta: float | None = None) -> np.ndarray:
    r"""The Ewald contribution to the stress tensor, in Hartree/Bohr\ :sup:`3`.

    Under a strain :math:`\mathbf r \to (1 + \varepsilon)\mathbf r` applied to
    the cell *and* the atoms together, every real-space separation scales and
    every reciprocal vector counter-scales.  Differentiating the two sums at
    :math:`\varepsilon = 0` and dividing by the volume,

    .. math::

        \sigma_{\alpha\beta} = \frac{1}{\Omega}\Bigl[
          -\tfrac12 \sum_{i,j,\mathbf R}' Z_i Z_j\,
             \Bigl(\frac{\mathrm{erfc}(\eta r)}{r^2}
             + \frac{2\eta}{\sqrt\pi}\frac{e^{-\eta^2 r^2}}{r}\Bigr)
             \frac{r_\alpha r_\beta}{r}
          + \frac{2\pi}{\Omega}\sum_{\mathbf G \neq 0} w(G)\,|S|^2
            \Bigl(\delta_{\alpha\beta}
              - 2\bigl(\tfrac{1}{G^2} + \tfrac{1}{4\eta^2}\bigr)
                G_\alpha G_\beta\Bigr)
          - \delta_{\alpha\beta} E_{\text{bg}} \Bigr] .

    The self term is volume-independent and drops; the background term scales
    as :math:`1/\Omega` and contributes its own isotropic piece.  Sign
    convention: :math:`\sigma = +\frac{1}{\Omega}\partial E/\partial
    \varepsilon`, which is what :meth:`ase.Atoms.get_stress` expects.
    """
    return _ewald_raw(positions, charges, cell, accuracy=accuracy, eta=eta)["stress"]


def madelung_constant(cell, accuracy: float = DEFAULT_ACCURACY,
                     eta: float | None = None) -> float:
    r"""Madelung potential of a unit charge in the cell (Hartree/electron).

    ``v_M = lim_{r->0} [v_E(r) - 1/r]``: what a unit charge feels from its own
    periodic images and the neutralizing background.  Equivalently twice the
    Ewald energy of a single unit charge in the cell, which is how it is
    computed here.

    This is the constant a periodic Coulomb kernel with ``G = 0`` dropped
    leaves out of every electron's self-interaction -- the Gygi-Baldereschi
    term, which for a Gamma-point supercell is exactly this number.
    """
    return 2.0 * ewald_energy([[0.0, 0.0, 0.0]], [1.0], cell,
                              accuracy=accuracy, eta=eta)


def ewald_potential(positions, charges, cell, points,
                    accuracy: float = DEFAULT_ACCURACY,
                    eta: float | None = None,
                    softening: float = 0.0) -> np.ndarray:
    r"""Zero-mean periodic electrostatic potential of point charges.

    The potential that :func:`ewald_energy` is the energy of: the lattice sum
    ``sum_{A,R} q_A / |r - tau_A - R|`` made finite by a uniform neutralizing
    background, evaluated at arbitrary ``points``.

    **Zero mean is the convention, and it has to be shared.**  The background
    term fixes ``(1/Omega) int v dr = 0`` over the cell, which is the same
    choice a reciprocal-space Coulomb kernel makes when it drops ``G = 0``.
    Mixing this potential with a Hartree term built any other way -- a finite
    window of bare ``-Z/r``, say -- leaves an arbitrary constant per electron
    that moves with the window.

    Parameters
    ----------
    positions : (N, 3) array_like
        Cartesian coordinates of the charges, in Bohr.
    charges : (N,) array_like
        Point charges in units of ``e``.
    cell : (3, 3) array_like
        Lattice vectors as **rows** (the ASE convention), in Bohr.
    points : (P, 3) array_like
        Where to evaluate, in Bohr.
    accuracy, eta
        As in :func:`ewald_energy`.
    softening : float
        Floor on the distance to a charge, in Bohr.  ``0`` keeps the true
        divergence; a grid sampling the potential needs a floor of about half a
        step, or a node landing on a nucleus returns infinity.  Only the
        short-range ``erfc`` term is affected -- the lattice sum itself is
        untouched.

    Returns
    -------
    (P,) ndarray
        The potential in Hartree.  At a charge's own site it diverges; the
        finite part there is :func:`madelung_constant`.
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.zeros(pts.reshape(-1, 3).shape[0], dtype=float)
    return _ewald_raw(positions, charges, cell, accuracy=accuracy, eta=eta,
                      points=pts, softening=softening)["potential"]
