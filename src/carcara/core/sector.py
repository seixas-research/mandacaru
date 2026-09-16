# -*- coding: utf-8 -*-
# file: core/sector.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Particle-number sector of a qubit register.

A molecular Hamiltonian conserves the numbers of alpha and beta electrons, so
its ground state lives in the span of the determinants with
:math:`(n_\alpha, n_\beta)` electrons: :math:`\binom{M}{n_\alpha}\binom{M}{n_\beta}`
basis states out of :math:`2^{2M}`.  For H\ :sub:`2` or LiH in a DZP basis
(20 qubits, one electron of each spin) that is 100 states instead of
1,048,576, and the difference decides feasibility: the sparse Hamiltonian over
the full register has billions of non-zeros, its restriction to the sector a
few thousand.

:class:`ParticleSector` enumerates the sector's computational basis states in
the fermion-to-qubit encoding the Hamiltonian uses (Jordan-Wigner, parity or
Bravyi-Kitaev, optionally tapered by the parity two-qubit reduction) and
restricts :class:`~carcara.core.mapping.PauliSum` operators to it without ever
forming a full-register matrix.  A Pauli string sends a basis state
:math:`|x\rangle` to a phase times :math:`|x \oplus f\rangle`, and that map is
evaluated for every sector state at once with integer bit operations.
Restricting a number-conserving operator (the Hamiltonian, an excitation
generator) is exact; :func:`apply_pauli_sum` applies an arbitrary operator,
such as a ladder operator, to a sparse state and returns wherever it lands.

Bit convention: qubit ``k`` is bit ``n - 1 - k`` of the basis-state index
(qubit 0 is the most significant bit), the same as
:meth:`~carcara.core.mapping.PauliSum.to_sparse_matrix`.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np

from .mapping import (PauliSum, _canonical_method, _encoding_matrix,
                      parity_tapered_qubits)

#: Largest sector the enumeration accepts (states).
MAX_SECTOR_DIMENSION = 5_000_000
#: Widest register a 64-bit basis-state index can address.
MAX_SECTOR_QUBITS = 62


def pauli_masks(label: str) -> tuple[int, int, int]:
    """``(flip, phase, n_y)`` bit masks of a Pauli string.

    ``flip`` marks the qubits an ``X`` or ``Y`` flips, ``phase`` the qubits
    whose bit contributes a sign (``Y`` and ``Z``), ``n_y`` counts the ``Y``s
    (each contributes a factor ``i``).
    """
    n = len(label)
    flip = phase = n_y = 0
    for k, letter in enumerate(label):
        bit = 1 << (n - 1 - k)
        if letter == "X":
            flip |= bit
        elif letter == "Y":
            flip |= bit
            phase |= bit
            n_y += 1
        elif letter == "Z":
            phase |= bit
    return flip, phase, n_y


def _term_action(label: str, coeff, states: np.ndarray):
    """Images and matrix elements of ``coeff * P`` on basis ``states``."""
    flip, phase, n_y = pauli_masks(label)
    images = states ^ np.int64(flip)
    parity = np.bitwise_count(states & np.int64(phase)) & 1
    values = complex(coeff) * (1j ** n_y) * (1 - 2 * parity.astype(np.int64))
    return images, values


def apply_pauli_sum(operator: PauliSum, indices, amplitudes):
    r"""Apply ``operator`` to the sparse state ``sum_x a_x |x>``.

    Parameters
    ----------
    operator : PauliSum
        Any qubit operator; it need not conserve particle number.
    indices, amplitudes : array_like
        Basis-state indices (int) and their amplitudes.

    Returns
    -------
    (indices, amplitudes)
        The image as a sparse state, with duplicate indices summed and exact
        zeros dropped; indices sorted ascending.
    """
    indices = np.asarray(indices, dtype=np.int64).ravel()
    amplitudes = np.asarray(amplitudes, dtype=complex).ravel()
    if indices.size == 0 or not operator.terms:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=complex)
    images, values = [], []
    for label, coeff in operator.terms.items():
        if coeff == 0:
            continue
        out, factor = _term_action(label, coeff, indices)
        images.append(out)
        values.append(factor * amplitudes)
    if not images:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=complex)
    images = np.concatenate(images)
    values = np.concatenate(values)
    unique, inverse = np.unique(images, return_inverse=True)
    summed = np.zeros(unique.size, dtype=complex)
    np.add.at(summed, inverse, values)
    keep = summed != 0
    return unique[keep], summed[keep]


class ParticleSector:
    r"""The :math:`(n_\alpha, n_\beta)` sector of a spin-blocked register.

    Parameters
    ----------
    n_qubits : int
        Register width (after tapering, when ``two_qubit_reduction``).
    num_particles : (int, int)
        ``(n_alpha, n_beta)``.
    mapping : str
        ``"jordan_wigner"``, ``"parity"`` or ``"bravyi_kitaev"``.
    two_qubit_reduction : bool
        The register is the parity mapping with its two particle-number
        qubits tapered.

    Attributes
    ----------
    indices : numpy.ndarray
        Sorted basis-state indices of the sector (``int64``).
    occupations : numpy.ndarray
        ``(dim, n_modes)`` spin-orbital occupations, aligned with
        :attr:`indices`.
    """

    def __init__(self, n_qubits: int, num_particles, mapping: str = "jordan_wigner",
                 two_qubit_reduction: bool = False):
        self.n_qubits = int(n_qubits)
        n_alpha, n_beta = (int(v) for v in num_particles)
        self.num_particles = (n_alpha, n_beta)
        self.mapping = _canonical_method(mapping)
        self.two_qubit_reduction = bool(two_qubit_reduction)
        if self.two_qubit_reduction and self.mapping != "parity":
            raise ValueError("two-qubit reduction requires mapping='parity'")
        if self.n_qubits > MAX_SECTOR_QUBITS:
            raise ValueError(f"a {self.n_qubits}-qubit register exceeds the "
                             f"{MAX_SECTOR_QUBITS}-qubit index range")
        self.n_modes = self.n_qubits + (2 if self.two_qubit_reduction else 0)
        if self.n_modes % 2:
            raise ValueError(f"a spin-blocked register needs an even number of "
                             f"spin-orbitals, got {self.n_modes}")
        M = self.n_modes // 2
        if not (0 <= n_alpha <= M and 0 <= n_beta <= M):
            raise ValueError(f"num_particles {self.num_particles} does not fit "
                             f"{M} spatial orbitals")
        dim = comb(M, n_alpha) * comb(M, n_beta)
        if dim > MAX_SECTOR_DIMENSION:
            raise ValueError(
                f"the {self.num_particles} sector of {M} orbitals has {dim} "
                f"states (limit {MAX_SECTOR_DIMENSION})")

        alpha = np.zeros((comb(M, n_alpha), M), dtype=np.int64)
        for i, occupied in enumerate(combinations(range(M), n_alpha)):
            alpha[i, list(occupied)] = 1
        beta = np.zeros((comb(M, n_beta), M), dtype=np.int64)
        for i, occupied in enumerate(combinations(range(M), n_beta)):
            beta[i, list(occupied)] = 1
        occupations = np.concatenate(
            [np.repeat(alpha, beta.shape[0], axis=0),
             np.tile(beta, (alpha.shape[0], 1))], axis=1)

        # Qubit register q = beta_matrix . x (mod 2), as reference_qubit_bits.
        encoding = _encoding_matrix(self.mapping, self.n_modes).astype(np.int64)
        bits = (occupations @ encoding.T) % 2
        if self.two_qubit_reduction:
            drop = set(parity_tapered_qubits(self.n_modes))
            bits = bits[:, [k for k in range(self.n_modes) if k not in drop]]
        weights = np.int64(1) << (np.int64(self.n_qubits) - 1
                                  - np.arange(self.n_qubits, dtype=np.int64))
        index = bits @ weights
        order = np.argsort(index)
        self.indices = index[order]
        self.occupations = occupations[order]
        if np.unique(self.indices).size != self.indices.size:
            raise RuntimeError("two determinants share a basis state; the "
                               "encoding is not invertible on this sector")

    # -- lookup ----------------------------------------------------------- #

    @property
    def dim(self) -> int:
        """Number of basis states in the sector."""
        return int(self.indices.size)

    def positions(self, indices):
        """Sector positions of basis-state ``indices`` (``-1`` if outside)."""
        indices = np.asarray(indices, dtype=np.int64)
        where = np.searchsorted(self.indices, indices)
        where = np.clip(where, 0, max(self.dim - 1, 0))
        inside = self.indices[where] == indices
        return np.where(inside, where, -1)

    def basis_vector(self, index: int) -> np.ndarray:
        """Sector vector of the basis state with full-register ``index``."""
        position = int(self.positions(np.array([index]))[0])
        if position < 0:
            raise ValueError(f"basis state {index} is not in the "
                             f"{self.num_particles} sector")
        vector = np.zeros(self.dim, dtype=complex)
        vector[position] = 1.0
        return vector

    # -- operators -------------------------------------------------------- #

    def restrict(self, operator: PauliSum):
        """``operator`` restricted to the sector, as a ``(dim, dim)`` CSR matrix.

        Exact for an operator that conserves ``(n_alpha, n_beta)``; the
        components of a non-conserving one that leave the sector are dropped.
        """
        import scipy.sparse as sp

        if operator.num_qubits != self.n_qubits:
            raise ValueError(f"operator acts on {operator.num_qubits} qubits, "
                             f"the sector register on {self.n_qubits}")
        columns = np.arange(self.dim, dtype=np.int64)
        rows_all, cols_all, data_all = [], [], []
        for label, coeff in operator.terms.items():
            if coeff == 0:
                continue
            images, values = _term_action(label, coeff, self.indices)
            rows = self.positions(images)
            inside = rows >= 0
            rows_all.append(rows[inside])
            cols_all.append(columns[inside])
            data_all.append(values[inside])
        if not data_all:
            return sp.csr_matrix((self.dim, self.dim), dtype=complex)
        matrix = sp.coo_matrix(
            (np.concatenate(data_all),
             (np.concatenate(rows_all), np.concatenate(cols_all))),
            shape=(self.dim, self.dim))
        return matrix.tocsr()

    # -- states ----------------------------------------------------------- #

    def conserves(self, operator: PauliSum, atol: float = 1e-12) -> bool:
        r"""Whether ``operator`` maps the sector into itself.

        :meth:`restrict` silently drops whatever leaves the sector, so a
        leaking operator would be *replaced* by its projection: for an ansatz
        that matters, because :math:`e^{PAP} \neq P e^{A} P` in general.  A
        term-by-term image test would reject operators whose leaking parts
        cancel, so the amplitudes of each image are summed first (per source
        state) and only a surviving amplitude outside the sector counts.
        """
        if not operator.terms:
            return True
        if operator.num_qubits != self.n_qubits:
            raise ValueError(f"operator acts on {operator.num_qubits} qubits, "
                             f"the sector register on {self.n_qubits}")
        for index in self.indices:
            images, amplitudes = apply_pauli_sum(operator, [index], [1.0])
            if images.size == 0:
                continue
            outside = self.positions(images) < 0
            if np.any(np.abs(amplitudes[outside]) > atol):
                return False
        return True

    def embed(self, vector) -> np.ndarray:
        """The full-register state vector (only for small registers)."""
        if self.n_qubits > 24:
            raise ValueError(f"refusing to allocate a 2^{self.n_qubits} vector")
        vector = np.asarray(vector, dtype=complex).ravel()
        if vector.size != self.dim:
            raise ValueError(f"expected {self.dim} amplitudes, got {vector.size}")
        full = np.zeros(2 ** self.n_qubits, dtype=complex)
        full[self.indices] = vector
        return full

    def project(self, full) -> np.ndarray:
        """Sector amplitudes of a full-register state vector."""
        full = np.asarray(full, dtype=complex).ravel()
        return full[self.indices].copy()

    def __repr__(self) -> str:
        return (f"ParticleSector(n_qubits={self.n_qubits}, "
                f"num_particles={self.num_particles}, mapping={self.mapping!r}, "
                f"dim={self.dim})")
