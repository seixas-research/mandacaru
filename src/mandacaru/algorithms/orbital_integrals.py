# -*- coding: utf-8 -*-
# file: algorithms/orbital_integrals.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Where the active-space selection gets its integrals from.

Choosing an active space needs a handful of molecular-orbital quantities --
the Fock diagonal, the pair-excitation couplings :math:`\langle ii|aa\rangle`
of :func:`~mandacaru.algorithms.active_space.correlating_partners`, and for
canonical MP2 the whole tensor.  Two providers answer the same questions:

* :class:`TensorOrbitalIntegrals` holds :math:`h` and :math:`\langle pq|rs\rangle`
  in the molecular orbitals, as the standard build always has;
* :class:`DirectOrbitalIntegrals` holds only the orbitals and computes each
  quantity on the grid (:class:`~mandacaru.integrals.direct.DirectCoulomb`),
  so a selection never forms the :math:`M^4` tensor.

Every method takes an optional ``rotation`` (the selector's ranked basis,
columns over the incoming orbitals), so a caller asks in whichever basis its
indices refer to.  Both providers agree to round-off.
"""

from __future__ import annotations

import numpy as np

__all__ = ["TensorOrbitalIntegrals", "DirectOrbitalIntegrals"]


class _LocalBasis:
    """What a local correlation method needs besides integrals: the basis
    functions and their positions, expressed in the molecular orbitals."""

    integrals = None
    orbitals = None

    def _require_basis(self):
        if self.integrals is None or self.orbitals is None:
            raise ValueError(
                "a local correlation method needs the basis behind the "
                "molecular orbitals (integrals= and orbitals=)")

    def basis_in_mo(self) -> np.ndarray:
        r"""``(M, M)``: column ``mu`` is orthonormal basis function ``mu`` in
        the molecular orbitals, :math:`C^\dagger e_\mu`."""
        self._require_basis()
        return self.orbitals.conj().T

    def position_mo(self) -> np.ndarray:
        """``(3, M, M)`` position matrices in the molecular orbitals."""
        from .local_correlation import position_matrices

        self._require_basis()
        C = self.orbitals
        return np.array([C.conj().T @ r @ C
                         for r in position_matrices(self.integrals)])

    def differential_overlap(self, occupied) -> np.ndarray:
        r"""Differential overlap of each orbital with each real basis part.

        :math:`\mathrm{DOI}_{i\mu} = (\int \phi_i^2 \chi_\mu^2)^{1/2}` on
        the grid, for MO-basis orbital columns ``occupied`` and the real and
        imaginary parts of every orthonormal basis function (in that order:
        the ``M`` real parts, then the ``M`` imaginary parts), the functions
        a local method builds its PAOs from.
        """
        self._require_basis()
        ints = self.integrals
        psi = ints._engine._psi
        X = (ints._lowdin_x() if ints.orthogonalize
             else np.eye(psi.shape[0]))
        phi = (X @ self.orbitals @ np.asarray(occupied)).T @ psi
        chi = X.T @ psi
        dV = ints.grid.dV
        dens = np.abs(phi) ** 2                                # (n, G)
        parts = np.concatenate([np.real(chi), np.imag(chi)]) ** 2
        return np.sqrt(np.maximum(dens @ parts.T * dV, 0.0))

    def atom_of(self) -> np.ndarray:
        """The atom of each orthonormal basis function."""
        from .local_correlation import basis_atoms

        self._require_basis()
        return basis_atoms(self.integrals)


class TensorOrbitalIntegrals(_LocalBasis):
    """Molecular-orbital integrals held as tensors.

    ``integrals`` and ``orbitals`` (the MO coefficients over the orthonormal
    basis) are needed only by a local correlation method, which works with
    atom-centered functions.
    """

    def __init__(self, h_mo, eri_mo, integrals=None, orbitals=None):
        self.h_mo = np.asarray(h_mo)
        self.eri_mo = np.asarray(eri_mo)
        self.integrals = integrals
        self.orbitals = None if orbitals is None else np.asarray(orbitals)

    def fock(self, weights) -> np.ndarray:
        """The full Fock matrix in the molecular orbitals for occupations
        ``weights``."""
        h, eri = np.asarray(self.h_mo), np.asarray(self.eri_mo)
        F = h.copy().astype(complex)
        for k, n_k in enumerate(weights):
            if n_k:
                F += 0.5 * n_k * (2.0 * eri[:, k, :, k] - eri[:, k, k, :])
        return 0.5 * (F + F.conj().T)

    def exchange_blocks(self, occupied, functions, pairs) -> dict:
        r""":math:`\langle ij|ab\rangle` for local pairs, MO-basis columns
        ``occupied`` / ``functions`` (as
        :meth:`~mandacaru.integrals.direct.DirectCoulomb.exchange_blocks`)."""
        occupied, functions = np.asarray(occupied), np.asarray(functions)
        half = np.einsum("pi,qj,pqrs->ijrs", occupied.conj(), occupied.conj(),
                         self.eri_mo, optimize=True)
        out = {}
        for (i, j), domain in pairs.items():
            F = functions[:, list(domain)]
            out[(i, j)] = F.T @ half[i, j] @ F
        return out

    @property
    def n_orbitals(self) -> int:
        return int(self.h_mo.shape[0])

    def tensors(self):
        """``(h_mo, eri_mo)``: what canonical MP2 needs."""
        return self.h_mo, self.eri_mo

    def _in(self, rotation):
        if rotation is None:
            return np.real(self.h_mo), np.real(self.eri_mo)
        from .mp2 import rotate_integrals

        h, eri = rotate_integrals(self.h_mo, self.eri_mo, rotation)
        return np.real(h), np.real(eri)

    def fock_diagonal(self, weights, rotation=None) -> np.ndarray:
        r"""Diagonal of the mean-field Fock matrix for occupations ``weights``.

        :math:`F_{pp} = h_{pp} + \sum_k \tfrac{n_k}{2}(2\langle pk|pk\rangle
        - \langle pk|kp\rangle)`; ``n_k`` is 2 for a doubly and 1 for a singly
        occupied orbital (the spin average of an open shell).
        """
        h, eri = self._in(rotation)
        fock = np.diag(h).copy()
        for k, n_k in enumerate(weights):
            if n_k:
                fock += 0.5 * n_k * (2.0 * np.diag(eri[:, k, :, k])
                                     - np.diag(eri[:, k, k, :]))
        return fock

    def pair_exchange(self, occupied, virtual, rotation=None) -> np.ndarray:
        r""":math:`|\langle ii|aa\rangle|` over ``occupied`` x ``virtual``."""
        _h, eri = self._in(rotation)
        return np.abs(np.array([[eri[i, i, a, a] for a in virtual]
                                for i in occupied]))


class DirectOrbitalIntegrals(_LocalBasis):
    """Molecular-orbital integrals computed on demand, without the tensor.

    Parameters
    ----------
    integrals : MolecularIntegrals
        The basis, its grid and its orthonormalization.
    orbitals : ndarray
        ``(M, M)`` molecular orbitals as columns over the orthonormal (Loewdin)
        basis the Hamiltonian is built in -- ``integrals.mo_coefficients``.
    """

    def __init__(self, integrals, orbitals):
        self.integrals = integrals
        self.orbitals = np.asarray(orbitals)
        self.direct = integrals.direct_coulomb()
        self.X = (integrals._lowdin_x() if integrals.orthogonalize
                  else np.eye(self.orbitals.shape[0]))
        self.h = integrals.one_body()                 # orthonormal basis

    @property
    def n_orbitals(self) -> int:
        return int(self.orbitals.shape[1])

    def tensors(self):
        """``None``: this provider never holds the tensor."""
        return None

    def _orbitals(self, rotation):
        return (self.orbitals if rotation is None
                else self.orbitals @ np.asarray(rotation))

    def two_electron(self, D):
        r""":math:`J[D] - \tfrac12 K[D]` in the orthonormal basis, for an
        orthonormal-basis density ``D``."""
        X = self.X
        return X.conj().T @ self.direct.fock_two_electron(
            X @ D @ X.conj().T, exact=True) @ X

    def fock_diagonal(self, weights, rotation=None) -> np.ndarray:
        """As :meth:`TensorOrbitalIntegrals.fock_diagonal`."""
        C = self._orbitals(rotation)
        weights = np.asarray(weights, dtype=float)
        D = (C * weights[None, :]) @ C.conj().T
        F = C.conj().T @ (self.h + self.two_electron(D)) @ C
        return np.real(np.diag(F))

    def pair_exchange(self, occupied, virtual, rotation=None) -> np.ndarray:
        """As :meth:`TensorOrbitalIntegrals.pair_exchange`."""
        C = self.X @ self._orbitals(rotation)
        return np.abs(self.direct.pair_exchange(C[:, list(occupied)],
                                                C[:, list(virtual)]))

    def fock(self, weights) -> np.ndarray:
        """As :meth:`TensorOrbitalIntegrals.fock`."""
        C = self.orbitals
        weights = np.asarray(weights, dtype=float)
        D = (C * weights[None, :]) @ C.conj().T
        F = C.conj().T @ (self.h + self.two_electron(D)) @ C
        return 0.5 * (F + F.conj().T)

    def exchange_blocks(self, occupied, functions, pairs) -> dict:
        """As :meth:`TensorOrbitalIntegrals.exchange_blocks`, on the grid."""
        to_ao = self.X @ self.orbitals
        return self.direct.exchange_blocks(to_ao @ np.asarray(occupied),
                                           to_ao @ np.asarray(functions),
                                           pairs)
