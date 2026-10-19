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
canonical MP2 the :math:`\langle ij|ab\rangle` block.  Two providers answer
the same questions:

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

from ..integrals.packed import is_packed

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
        the grid, for MO-basis orbital columns ``occupied`` and every
        orthonormal basis function (real, as the real harmonics make them),
        the functions a local method builds its PAOs from.
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
        return np.sqrt(np.maximum(dens @ (np.abs(chi) ** 2).T * dV, 0.0))

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
        # A packed tensor stays packed: every query below reads it so.
        self.eri_mo = eri_mo if is_packed(eri_mo) else np.asarray(eri_mo)
        self.integrals = integrals
        self.orbitals = None if orbitals is None else np.asarray(orbitals)

    def _mean_field(self, weights, rotation=None) -> np.ndarray:
        r""":math:`h + J[N] - \tfrac12 K[N]` of the occupations ``weights``
        (``N = diag(weights)``) in the (rotated) orbitals, from the packed
        tensor: the density is rotated, never the tensor."""
        h, eri = self.h_mo, self.eri_mo
        R = np.eye(h.shape[0]) if rotation is None else np.asarray(rotation)
        D = (R * np.asarray(weights, dtype=float)[None, :]) @ R.conj().T
        F = h + eri.coulomb(D) - 0.5 * eri.exchange(D)
        return R.conj().T @ F @ R

    def fock(self, weights) -> np.ndarray:
        """The full Fock matrix in the molecular orbitals for occupations
        ``weights``."""
        if is_packed(self.eri_mo):
            F = self._mean_field(weights)
            return 0.5 * (F + F.conj().T)
        h, eri = np.asarray(self.h_mo), np.asarray(self.eri_mo)
        F = h.astype(np.result_type(h, eri, float))
        for k, n_k in enumerate(weights):
            if n_k:
                F += 0.5 * n_k * (2.0 * eri[:, k, :, k] - eri[:, k, k, :])
        return 0.5 * (F + F.conj().T)

    def exchange_blocks(self, occupied, functions, pairs) -> dict:
        r""":math:`\langle ij|ab\rangle` for local pairs, MO-basis columns
        ``occupied`` / ``functions`` (as
        :meth:`~mandacaru.integrals.direct.DirectCoulomb.exchange_blocks`)."""
        occupied, functions = np.asarray(occupied), np.asarray(functions)
        if is_packed(self.eri_mo):
            identity = np.eye(self.n_orbitals)
            half = self.eri_mo.block(occupied, occupied, identity, identity)
        else:
            half = np.einsum("pi,qj,pqrs->ijrs", occupied.conj(),
                             occupied.conj(), self.eri_mo, optimize=True)
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
        if is_packed(self.eri_mo):
            return np.real(np.diag(self._mean_field(weights, rotation)))
        h, eri = self._in(rotation)
        fock = np.diag(h).copy()
        for k, n_k in enumerate(weights):
            if n_k:
                fock += 0.5 * n_k * (2.0 * np.diag(eri[:, k, :, k])
                                     - np.diag(eri[:, k, k, :]))
        return fock

    def pair_exchange(self, occupied, virtual, rotation=None) -> np.ndarray:
        r""":math:`|\langle ii|aa\rangle|` over ``occupied`` x ``virtual``."""
        if is_packed(self.eri_mo):
            R = (np.eye(self.n_orbitals) if rotation is None
                 else np.asarray(rotation))
            Ro, Rv = R[:, list(occupied)], R[:, list(virtual)]
            block = self.eri_mo.block(Ro, Ro, Rv, Rv)        # <ij|ab>
            o, v = Ro.shape[1], Rv.shape[1]
            return np.abs(block[np.arange(o)[:, None], np.arange(o)[:, None],
                                np.arange(v)[None, :], np.arange(v)[None, :]])
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
        # The grid's DirectCoulomb, or the Cholesky factors when the
        # integrals are factored: the same interface.
        self.direct = integrals.two_electron_operator()
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

    def mp2_natural_orbitals(self, n_occ: int):
        r"""Closed-shell MP2 frozen natural orbitals
        (:func:`~mandacaru.algorithms.mp2.mp2_natural_orbitals`) from the
        :math:`\langle ij|ab\rangle` block alone.

        The Fock matrix comes from the direct Coulomb and exchange, its
        semicanonical orbitals from :func:`~mandacaru.algorithms.mp2.semicanonical_rotation`,
        and the :math:`o^2 v^2` integrals from
        :meth:`~mandacaru.integrals.direct.DirectCoulomb.exchange_blocks`
        (:math:`o v` Poisson solves), so the :math:`M^4` tensor the canonical
        route reads is never formed.  Equal to that route to round-off.
        """
        from .mp2 import mp2_natural_orbitals_from_block, semicanonical_rotation

        n_occ = int(n_occ)
        M = self.n_orbitals
        if not 0 < n_occ < M:
            raise ValueError(
                f"MP2 needs at least one occupied and one virtual orbital, "
                f"got {n_occ} occupied of {M}")
        weights = [2.0] * n_occ + [0.0] * (M - n_occ)
        canonical, energies = semicanonical_rotation(self.fock(weights), n_occ)
        C = self.X @ self.orbitals @ canonical
        virtual = range(M - n_occ)
        pairs = {(i, j): virtual for i in range(n_occ) for j in range(n_occ)}
        blocks = self.direct.exchange_blocks(C[:, :n_occ], C[:, n_occ:], pairs)
        oovv = np.array([[blocks[(i, j)] for j in range(n_occ)]
                         for i in range(n_occ)])
        return mp2_natural_orbitals_from_block(oovv, energies, n_occ, canonical)

    def open_shell_mp2_natural_orbitals(self, n_alpha: int, n_beta: int):
        r"""Open-shell MP2 frozen natural orbitals
        (:func:`~mandacaru.algorithms.mp2.open_shell_mp2_natural_orbitals`)
        without the tensor.

        The two spin Fock matrices of the determinant filling the lowest
        ``n_alpha`` and ``n_beta`` orbitals come from the direct Coulomb and
        exchange, :math:`F^\sigma = h + J[D^\alpha + D^\beta] - K[D^\sigma]`
        (:func:`~mandacaru.algorithms.mp2.spin_fock_matrices`), and the
        same-spin and alpha-beta :math:`\langle ij|ab\rangle` blocks in each
        spin's semicanonical orbitals from
        :meth:`~mandacaru.integrals.direct.DirectCoulomb.ovov` (:math:`o v`
        Poisson solves per block), so the :math:`M^4` tensor the canonical
        route transforms twice is never formed.  Equal to that route to
        round-off.
        """
        from .mp2 import open_shell_mp2_from_blocks

        C, X = self.orbitals, self.X
        densities = []
        for n in (int(n_alpha), int(n_beta)):
            occupied = X @ C[:, :n]
            densities.append(occupied @ occupied.conj().T)     # AO basis
        J = self.direct.coulomb(densities[0] + densities[1])
        focks = []
        for D in densities:
            G = X.conj().T @ (J - self.direct.exchange(D)) @ X
            F = C.conj().T @ (self.h + G) @ C
            focks.append(0.5 * (F + F.conj().T))
        to_ao = X @ C

        def oovv(c1, c2, c3, c4):
            # <ij|ab> = (ia|jb): electron 1 on (i, a), electron 2 on (j, b).
            block = self.direct.ovov(to_ao @ c1, to_ao @ c3, to_ao @ c2,
                                     to_ao @ c4)
            return block.transpose(0, 2, 1, 3)

        return open_shell_mp2_from_blocks(focks, n_alpha, n_beta, oovv)

    def exchange_blocks(self, occupied, functions, pairs) -> dict:
        """As :meth:`TensorOrbitalIntegrals.exchange_blocks`, on the grid."""
        to_ao = self.X @ self.orbitals
        return self.direct.exchange_blocks(to_ao @ np.asarray(occupied),
                                           to_ao @ np.asarray(functions),
                                           pairs)
