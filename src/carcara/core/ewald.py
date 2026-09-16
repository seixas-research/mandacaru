# -*- coding: utf-8 -*-
# file: core/ewald.py

# This code is part of Carcará.
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
(:mod:`carcara.core.planewave`).  Only with *both* conventions in place do the
three divergences cancel and the total energy of a neutral cell come out right;
a molecular pair sum in a periodic Hamiltonian does not.

The splitting parameter :math:`\eta` only sets how the work is divided between
the two sums, never the result, and the cutoffs are chosen from the requested
accuracy.  Everything is in atomic units (Bohr, Hartree).
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc

#: Terms smaller than this (relative to the charge scale) are dropped.
DEFAULT_ACCURACY = 1e-12


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
        Splitting parameter (Bohr\\ :sup:`-1`).  The default balances the two
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
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    charges = np.asarray(charges, dtype=float).reshape(-1)
    cell = np.asarray(cell, dtype=float).reshape(3, 3)
    if positions.shape[0] != charges.size:
        raise ValueError(f"{positions.shape[0]} positions for {charges.size} "
                         f"charges")
    volume = abs(float(np.linalg.det(cell)))
    if volume <= 0.0:
        raise ValueError("the cell has zero volume")
    if charges.size == 0:
        return 0.0

    if eta is None:
        # The usual balance: real-space work ~ reciprocal-space work.
        eta = float(np.sqrt(np.pi) * (charges.size / volume ** 2) ** (1.0 / 6.0))
    eta = float(eta)

    # Cutoffs from the requested accuracy: erfc(eta r)/r and exp(-G^2/4 eta^2)/G^2.
    scale = float(np.max(np.abs(charges))) ** 2 or 1.0
    tolerance = accuracy / scale
    r_cut = _real_cutoff(eta, tolerance)
    g_cut = _reciprocal_cutoff(eta, tolerance, volume)

    real = _real_space(positions, charges, cell, eta, r_cut)
    reciprocal = _reciprocal_space(positions, charges, cell, eta, g_cut, volume)
    self_energy = eta / np.sqrt(np.pi) * float(np.sum(charges ** 2))
    background = np.pi / (2.0 * eta ** 2 * volume) * float(np.sum(charges)) ** 2
    return float(real + reciprocal - self_energy - background)


def _real_cutoff(eta: float, tolerance: float) -> float:
    """Radius where ``erfc(eta r)/r`` drops below ``tolerance``."""
    r = 1.0 / eta
    for _ in range(200):
        if erfc(eta * r) / r < tolerance:
            return r
        r *= 1.25
    return r


def _reciprocal_cutoff(eta: float, tolerance: float, volume: float) -> float:
    """Wave number where ``exp(-G^2/4 eta^2)/G^2`` drops below ``tolerance``."""
    g = 2.0 * eta
    for _ in range(200):
        if np.exp(-g * g / (4.0 * eta * eta)) / (g * g) < tolerance * volume:
            return g
        g *= 1.25
    return g


def _shell_range(cell: np.ndarray, cutoff: float) -> np.ndarray:
    """Integer ranges covering every lattice point within ``cutoff``."""
    # The spacing between lattice planes is 1/|b_i| with b the reciprocal rows.
    reciprocal = np.linalg.inv(cell).T          # rows: reciprocal vectors / 2pi
    spacing = 1.0 / np.linalg.norm(reciprocal, axis=1)
    return np.ceil(cutoff / spacing).astype(int) + 1


def _lattice_points(cell: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """All lattice vectors ``n @ cell`` with ``|n_i| <= limits_i``."""
    ranges = [np.arange(-int(n), int(n) + 1) for n in limits]
    grid = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3) @ cell


def _real_space(positions, charges, cell, eta, r_cut) -> float:
    """``1/2 sum_{i,j,R}' Z_i Z_j erfc(eta r)/r`` over images within ``r_cut``."""
    vectors = _lattice_points(cell, _shell_range(cell, r_cut))
    total = 0.0
    for i, (ri, zi) in enumerate(zip(positions, charges)):
        deltas = ri - positions                              # (N, 3)
        for j, (delta, zj) in enumerate(zip(deltas, charges)):
            separations = np.linalg.norm(delta + vectors, axis=1)
            if i == j:
                separations = separations[separations > 1e-12]   # skip R = 0
            inside = separations[separations < r_cut]
            if inside.size:
                total += zi * zj * float(np.sum(erfc(eta * inside) / inside))
    return 0.5 * total


def _reciprocal_space(positions, charges, cell, eta, g_cut, volume) -> float:
    """``2 pi / V sum_{G != 0} exp(-G^2/4 eta^2)/G^2 |S(G)|^2``."""
    reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T          # rows: b_i
    vectors = _lattice_points(reciprocal, _shell_range(reciprocal, g_cut))
    g_sq = np.einsum("ij,ij->i", vectors, vectors)
    keep = (g_sq > 1e-12) & (g_sq < g_cut * g_cut)
    vectors, g_sq = vectors[keep], g_sq[keep]
    if not g_sq.size:
        return 0.0
    phases = np.exp(1j * (vectors @ positions.T))             # (nG, N)
    structure = phases @ charges                              # S(G)
    weights = np.exp(-g_sq / (4.0 * eta * eta)) / g_sq
    return float(2.0 * np.pi / volume
                 * np.sum(weights * np.abs(structure) ** 2))
