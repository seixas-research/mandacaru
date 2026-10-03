# -*- coding: utf-8 -*-
# file: algorithms/local_correlation.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Local orbitals for local correlation: Foster-Boys orbitals, PAOs, domains.

Canonical orbitals spread over the whole molecule, so every occupied pair
couples to every virtual orbital and the work of a correlation method grows
steeply with size.  Electron correlation is short-ranged, though, and in
**localized** orbitals that shows: a localized occupied orbital correlates with
virtual functions near it, and two far-apart orbitals barely correlate at all.
This module provides the three ingredients a local method (DLPNO-MP2) builds
on.

**Foster-Boys orbitals.**  The occupied orbitals are rotated among themselves
to maximize :math:`\sum_i |\langle i|\mathbf r|i\rangle|^2`, the spread of
their centroids, which is the same as minimizing each orbital's own spread
(Foster and Boys, Rev. Mod. Phys. 32, 300 (1960)).  A rotation inside the
occupied space leaves the determinant, the density and every mean-field
quantity unchanged.

**Projected atomic orbitals (PAOs).**  The virtual space is represented by
the basis functions with the occupied space projected out,
:math:`|\tilde\mu\rangle = (1 - P_\text{occ})|\mu\rangle` (Pulay, Chem.
Phys. Lett. 100, 151 (1983)).  They are redundant (there are :math:`M` of them
for :math:`M - o` virtual orbitals) and non-orthogonal, but each one stays on
the atom of its basis function, which is what makes a domain possible.

**Domains.**  An occupied orbital's domain is the set of atoms that carry a
non-negligible share of it (its Loewdin population on the atom); its PAO
domain is the PAOs on those atoms.  A pair's domain is the union of the two.

Everything here works in the orthonormal (Loewdin) basis the Hamiltonian is
built in, where basis function :math:`\mu` is itself atom-centered.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["position_matrices", "boys_localize", "projected_atomic_orbitals",
           "orbital_domains", "LocalOrbitals"]

#: Convergence of the Foster-Boys Jacobi sweeps: the largest rotation angle.
BOYS_TOLERANCE = 1e-10

#: Largest number of Jacobi sweeps.
BOYS_MAX_SWEEPS = 200

#: Loewdin population an atom must carry for it to be in an orbital's domain.
DOMAIN_POPULATION = 0.02


def position_matrices(integrals) -> np.ndarray:
    r"""``(3, M, M)`` matrices of :math:`x, y, z` in the orthonormal basis.

    Sampled on the grid (the same samples the integrals use), then
    transformed with the Loewdin :math:`X`.  A PAW-LCAO basis adds no
    augmentation here: the matrices serve localization, which needs the
    orbitals' centroids, not an exact dipole.
    """
    psi = integrals._engine._psi
    grid = integrals.grid
    X = (integrals._lowdin_x() if integrals.orthogonalize
         else np.eye(psi.shape[0]))
    coords = (grid.X.ravel(), grid.Y.ravel(), grid.Z.ravel())
    out = []
    for r in coords:
        R = (np.conj(psi) * r) @ psi.T * grid.dV
        R = X.conj().T @ (0.5 * (R + R.conj().T)) @ X
        out.append(R)
    return np.array(out)


def _boys_functional(centroids) -> float:
    return float(np.sum(np.abs(centroids) ** 2))


def boys_localize(position, orbitals, *, tolerance: float = BOYS_TOLERANCE,
                  max_sweeps: int = BOYS_MAX_SWEEPS):
    r"""Foster-Boys localization of ``orbitals`` (columns, orthonormal basis).

    Jacobi sweeps over pairs: each 2x2 rotation is the exact maximizer of
    :math:`\sum_i |\langle i|\mathbf r|i\rangle|^2` for that pair (the
    classic Edmiston-Ruedenberg-style angle).  The orbitals must be real
    combinations of real functions or conjugation-real ones, for which the
    rotation is real.

    Returns ``(localized, U)`` with ``localized = orbitals @ U`` and ``U``
    orthogonal.
    """
    C = np.asarray(orbitals)
    n = C.shape[1]
    U = np.eye(n)
    if n < 2:
        return C.copy(), U
    # r_ij in the orbital basis, kept real (conjugation-real orbitals).
    R = np.real(np.array([C.conj().T @ r @ C for r in position]))
    for _sweep in range(max_sweeps):
        largest = 0.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                a = R[:, i, j]
                b = R[:, i, i] - R[:, j, j]
                A = float(a @ a - 0.25 * (b @ b))
                B = float(a @ b)
                if abs(A) < 1e-14 and abs(B) < 1e-14:
                    continue
                theta = 0.25 * np.arctan2(B, -A)
                if abs(theta) < tolerance:
                    continue
                largest = max(largest, abs(theta))
                c, s = np.cos(theta), np.sin(theta)
                # i' = c i + s j and j' = -s i + c j: the convention the
                # angle above maximizes.  Only rows and columns i, j change,
                # so the update is O(n) per rotation, not O(n^3).
                ri, rj = R[:, i, :].copy(), R[:, j, :].copy()
                R[:, i, :], R[:, j, :] = c * ri + s * rj, -s * ri + c * rj
                ci, cj = R[:, :, i].copy(), R[:, :, j].copy()
                R[:, :, i], R[:, :, j] = c * ci + s * cj, -s * ci + c * cj
                ui, uj = U[:, i].copy(), U[:, j].copy()
                U[:, i], U[:, j] = c * ui + s * uj, -s * ui + c * uj
        if largest < tolerance:
            break
    return C @ U, U


def projected_atomic_orbitals(occupied) -> np.ndarray:
    r"""PAO coefficients :math:`(1 - P_\text{occ})` in the orthonormal basis.

    Column :math:`\mu` is basis function :math:`\mu` with every occupied
    orbital projected out, normalized; a PAO whose norm falls below
    :math:`10^{-8}` (a function the occupied space already spans) is left
    zero.
    """
    C = np.asarray(occupied)
    M = C.shape[0]
    P = C @ C.conj().T
    paos = np.eye(M) - P
    norms = np.sqrt(np.real(np.sum(np.abs(paos) ** 2, axis=0)))
    keep = norms > 1e-8
    paos[:, keep] = paos[:, keep] / norms[keep]
    paos[:, ~keep] = 0.0
    return paos


def orbital_domains(orbitals, atom_of, *,
                    threshold: float = DOMAIN_POPULATION) -> list[tuple]:
    r"""The atoms each orbital lives on: Loewdin population above
    ``threshold``.

    ``atom_of[mu]`` is the atom of orthonormal basis function :math:`\mu`.
    An orbital's population on atom :math:`A` is :math:`\sum_{\mu\in A}
    |C_{\mu i}|^2` (its Loewdin population, exact in an orthonormal basis).
    The largest atom is always included.
    """
    C = np.asarray(orbitals)
    atom_of = np.asarray(atom_of)
    atoms = np.unique(atom_of)
    weight = np.abs(C) ** 2
    domains = []
    for i in range(C.shape[1]):
        population = np.array([weight[atom_of == A, i].sum() for A in atoms])
        chosen = set(atoms[population >= threshold].tolist())
        chosen.add(int(atoms[int(np.argmax(population))]))
        domains.append(tuple(sorted(chosen)))
    return domains


@dataclass
class LocalOrbitals:
    """Localized occupied orbitals, PAOs and their domains.

    Attributes
    ----------
    occupied : ndarray
        ``(M, o)`` Foster-Boys orbitals of the active occupied space.
    rotation : ndarray
        ``(o, o)`` orthogonal: ``occupied = canonical @ rotation``.
    paos : ndarray
        ``(M, M)`` projected atomic orbitals (all occupied projected out).
    atom_of : ndarray
        The atom of each basis function / PAO.
    domains : list of tuple
        Atoms of each localized orbital.
    centroids : ndarray
        ``(o, 3)`` orbital centroids, Bohr.
    """

    occupied: np.ndarray
    rotation: np.ndarray
    paos: np.ndarray
    atom_of: np.ndarray
    domains: list
    centroids: np.ndarray

    def pao_domain(self, i: int) -> list[int]:
        """PAO indices on the atoms of orbital ``i``'s domain."""
        return [mu for mu, A in enumerate(self.atom_of)
                if A in set(self.domains[i])]

    def pair_domain(self, i: int, j: int) -> list[int]:
        """PAO indices of the union of two orbitals' domains."""
        atoms = set(self.domains[i]) | set(self.domains[j])
        return [mu for mu, A in enumerate(self.atom_of) if A in atoms]


def basis_atoms(integrals) -> np.ndarray:
    """The atom index of every basis function, from the function centers."""
    index: dict[tuple, int] = {}
    out = []
    for f in integrals.basis:
        key = tuple(np.round(np.asarray(f.center, dtype=float), 6))
        out.append(index.setdefault(key, len(index)))   # atom order
    return np.array(out)


def localize(integrals, canonical_occupied, all_occupied, *,
             domain_threshold: float = DOMAIN_POPULATION) -> LocalOrbitals:
    """Foster-Boys orbitals of ``canonical_occupied``, PAOs with every orbital
    of ``all_occupied`` (core included) projected out, and the domains."""
    position = position_matrices(integrals)
    occupied, U = boys_localize(position, canonical_occupied)
    centroids = np.real(np.array([[occupied[:, i].conj() @ r @ occupied[:, i]
                                   for r in position]
                                  for i in range(occupied.shape[1])]))
    atom_of = basis_atoms(integrals)
    return LocalOrbitals(
        occupied=occupied, rotation=U,
        paos=projected_atomic_orbitals(all_occupied), atom_of=atom_of,
        domains=orbital_domains(occupied, atom_of,
                                threshold=domain_threshold),
        centroids=centroids)
