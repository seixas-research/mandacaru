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


def _ewald_setup(positions, charges, cell, accuracy, eta):
    """The shared preamble of every Ewald quantity: shapes, volume, cutoffs."""
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    charges = np.asarray(charges, dtype=float).reshape(-1)
    cell = np.asarray(cell, dtype=float).reshape(3, 3)
    if positions.shape[0] != charges.size:
        raise ValueError(f"{positions.shape[0]} positions for {charges.size} "
                         f"charges")
    volume = abs(float(np.linalg.det(cell)))
    if volume <= 0.0:
        raise ValueError("the cell has zero volume")
    if eta is None:
        eta = float(np.sqrt(np.pi) * (max(charges.size, 1) / volume ** 2)
                    ** (1.0 / 6.0))
    eta = float(eta)
    scale = float(np.max(np.abs(charges))) ** 2 if charges.size else 1.0
    tolerance = accuracy / (scale or 1.0)
    return (positions, charges, cell, volume, eta,
            _real_cutoff(eta, tolerance),
            _reciprocal_cutoff(eta, tolerance, volume))


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
    (positions, charges, cell, volume, eta, r_cut,
     g_cut) = _ewald_setup(positions, charges, cell, accuracy, eta)
    gradient = np.zeros_like(positions)
    if charges.size == 0:
        return gradient

    # -- real space --------------------------------------------------------- #
    vectors = _lattice_points(cell, _shell_range(cell, r_cut))
    for i, (ri, zi) in enumerate(zip(positions, charges)):
        for rj, zj in zip(positions, charges):
            separations = (ri - rj) + vectors                 # (nR, 3)
            distance = np.linalg.norm(separations, axis=1)
            keep = (distance > 1e-12) & (distance < r_cut)
            if not np.any(keep):
                continue
            d = distance[keep]
            weight = (erfc(eta * d) / d ** 2
                      + (2.0 * eta / np.sqrt(np.pi)) * np.exp(-(eta * d) ** 2) / d)
            gradient[i] -= zi * zj * (separations[keep]
                                      * (weight / d)[:, None]).sum(axis=0)

    # -- reciprocal space ---------------------------------------------------- #
    reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T          # rows: b_i
    gvectors = _lattice_points(reciprocal, _shell_range(reciprocal, g_cut))
    g_sq = np.einsum("ij,ij->i", gvectors, gvectors)
    keep = (g_sq > 1e-12) & (g_sq < g_cut * g_cut)
    gvectors, g_sq = gvectors[keep], g_sq[keep]
    if g_sq.size:
        phases = np.exp(1j * (gvectors @ positions.T))        # (nG, N)
        structure = phases @ charges                          # S(G)
        weights = np.exp(-g_sq / (4.0 * eta * eta)) / g_sq
        # Im[e^{iG.R_i} S*] per (G, i).
        imaginary = np.imag(phases * np.conj(structure)[:, None])
        gradient -= (4.0 * np.pi / volume) * charges[:, None] * (
            (weights[:, None] * gvectors).T @ imaginary).T

    return -gradient


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
    (positions, charges, cell, volume, eta, r_cut,
     g_cut) = _ewald_setup(positions, charges, cell, accuracy, eta)
    stress = np.zeros((3, 3))
    if charges.size == 0:
        return stress

    vectors = _lattice_points(cell, _shell_range(cell, r_cut))
    for i, (ri, zi) in enumerate(zip(positions, charges)):
        for rj, zj in zip(positions, charges):
            separations = (ri - rj) + vectors
            distance = np.linalg.norm(separations, axis=1)
            keep = (distance > 1e-12) & (distance < r_cut)
            if not np.any(keep):
                continue
            d = distance[keep]
            sep = separations[keep]
            weight = (erfc(eta * d) / d ** 2
                      + (2.0 * eta / np.sqrt(np.pi)) * np.exp(-(eta * d) ** 2) / d)
            stress -= 0.5 * zi * zj * np.einsum(
                "n,na,nb->ab", weight / d, sep, sep)

    reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
    gvectors = _lattice_points(reciprocal, _shell_range(reciprocal, g_cut))
    g_sq = np.einsum("ij,ij->i", gvectors, gvectors)
    keep = (g_sq > 1e-12) & (g_sq < g_cut * g_cut)
    gvectors, g_sq = gvectors[keep], g_sq[keep]
    if g_sq.size:
        phases = np.exp(1j * (gvectors @ positions.T))
        structure = np.abs(phases @ charges) ** 2
        weights = np.exp(-g_sq / (4.0 * eta * eta)) / g_sq
        prefactor = (2.0 * np.pi / volume) * weights * structure
        # 1/V contributes -delta_ab; w(G) contributes +2 w (1/G^2 + 1/4eta^2)
        # G_a G_b, because G counter-scales with the strain.  Both signs were
        # inverted first, which left a residual the size of the stress itself.
        stress -= np.einsum("n,ab->ab", prefactor, np.eye(3))
        factor = 2.0 * (1.0 / g_sq + 1.0 / (4.0 * eta * eta))
        stress += np.einsum("n,n,na,nb->ab", prefactor, factor,
                            gvectors, gvectors)

    # E carries the background with a minus, and it scales as 1/V, so its
    # contribution to dE/d(strain) is +delta_ab times the term itself.
    background = np.pi / (2.0 * eta ** 2 * volume) * float(np.sum(charges)) ** 2
    stress += np.eye(3) * background
    return stress / volume


#: Point-block size for the reciprocal sum (points x G vectors stays bounded).
POTENTIAL_BLOCK = 4_000_000


def madelung_constant(cell, accuracy: float = DEFAULT_ACCURACY,
                      eta: float | None = None) -> float:
    r"""The lattice's Madelung potential ``v_M``, in Hartree per unit charge.

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
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    charges = np.asarray(charges, dtype=float).reshape(-1)
    cell = np.asarray(cell, dtype=float).reshape(3, 3)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if positions.shape[0] != charges.size:
        raise ValueError(f"{positions.shape[0]} positions for {charges.size} "
                         f"charges")
    volume = abs(float(np.linalg.det(cell)))
    if volume <= 0.0:
        raise ValueError("the cell has zero volume")
    if charges.size == 0 or points.size == 0:
        return np.zeros(points.shape[0], dtype=float)

    if eta is None:
        eta = float(np.sqrt(np.pi) * (charges.size / volume ** 2) ** (1.0 / 6.0))
    eta = float(eta)
    scale = float(np.max(np.abs(charges))) ** 2 or 1.0
    tolerance = accuracy / scale
    r_cut = _real_cutoff(eta, tolerance)
    g_cut = _reciprocal_cutoff(eta, tolerance, volume)

    # -- real space: erfc over the images within the cutoff ----------------- #
    translations = _lattice_points(cell, _shell_range(cell, r_cut))
    out = np.zeros(points.shape[0], dtype=float)
    for centre, charge in zip(positions, charges):
        for shift in translations:
            separation = np.linalg.norm(points - (centre + shift), axis=1)
            near = separation < r_cut
            if not np.any(near):
                continue
            distance = separation[near]
            if softening > 0.0:
                distance = np.maximum(distance, softening)
                contribution = erfc(eta * distance) / distance
            else:
                # A point sitting exactly on a charge is a true divergence.
                with np.errstate(divide="ignore", invalid="ignore"):
                    contribution = np.where(
                        distance > 1e-12, erfc(eta * distance) / distance,
                        np.inf)
            out[near] += charge * contribution

    # -- reciprocal space: the smooth Gaussian part ------------------------- #
    reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
    vectors = _lattice_points(reciprocal, _shell_range(reciprocal, g_cut))
    g_sq = np.einsum("ij,ij->i", vectors, vectors)
    keep = (g_sq > 1e-12) & (g_sq < g_cut * g_cut)
    vectors, g_sq = vectors[keep], g_sq[keep]
    if g_sq.size:
        weights = 4.0 * np.pi / volume * np.exp(-g_sq / (4.0 * eta * eta)) / g_sq
        structure = np.exp(-1j * (vectors @ positions.T)) @ charges   # S(G)
        real_part = weights * structure.real
        imag_part = weights * structure.imag
        block = max(1, int(POTENTIAL_BLOCK // max(g_sq.size, 1)))
        for start in range(0, points.shape[0], block):
            chunk = points[start:start + block]
            phase = chunk @ vectors.T
            out[start:start + block] += (np.cos(phase) @ real_part
                                         - np.sin(phase) @ imag_part)

    # -- the background: what makes the mean zero --------------------------- #
    out -= np.pi * float(np.sum(charges)) / (eta * eta * volume)
    return out


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
