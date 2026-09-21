# -*- coding: utf-8 -*-
# file: core/sector.py

# This code is part of Mandacaru.
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
restricts :class:`~mandacaru.core.mapping.PauliSum` operators to it without ever
forming a full-register matrix.  A Pauli string sends a basis state
:math:`|x\rangle` to a phase times :math:`|x \oplus f\rangle`, and that map is
evaluated for every sector state at once with integer bit operations.
Restricting a number-conserving operator (the Hamiltonian, an excitation
generator) is exact; :func:`apply_pauli_sum` applies an arbitrary operator,
such as a ladder operator, to a sparse state and returns wherever it lands.

Bit convention: qubit ``k`` is bit ``n - 1 - k`` of the basis-state index
(qubit 0 is the most significant bit), the same as
:meth:`~mandacaru.core.mapping.PauliSum.to_sparse_matrix`.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np

from .mapping import (PauliSum, _encoding_matrix, parity_tapered_qubits,
                      resolve_mapping)

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


def flip_groups(operator: PauliSum) -> dict:
    r"""``{flip mask: [(phase mask, coefficient * i**n_y), ...]}``.

    The grouping :meth:`ParticleSector.restrict` is built on: a Pauli string
    acts as :math:`|x\rangle \mapsto s(x)\,|x \oplus f\rangle`, so terms
    sharing :math:`f` share every image and differ only in the sign
    :math:`s(x)`, which is cheap.  Zero coefficients are dropped -- they
    contribute nothing and would cost a pass over the sector.
    """
    groups: dict = {}
    for label, coeff in operator.terms.items():
        if coeff == 0:
            continue
        flip, phase, n_y = pauli_masks(label)
        groups.setdefault(flip, []).append(
            (phase, complex(coeff) * (1j ** n_y)))
    return groups


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


#: Sector entries held before being folded into the running matrix by
#: :meth:`ParticleSector.restrict`.  Each is 32 bytes (a complex value and two
#: 64-bit indices), so the default bounds that staging buffer at ~256 MB; the
#: result itself is far smaller than the un-summed total.
RESTRICT_BATCH_ENTRIES = 8_000_000


class ParticleSector:
    r"""The :math:`(n_\alpha, n_\beta)` sector of a spin-blocked register.

    Parameters
    ----------
    n_qubits : int
        Register width after any tapering selected by ``mapping``.
    num_particles : (int, int)
        ``(n_alpha, n_beta)``.
    mapping : str
        One of ``"jordan_wigner"``, ``"parity"``, ``"parity_reduced"`` or
        ``"bravyi_kitaev"``.
    Attributes
    ----------
    indices : numpy.ndarray
        Sorted basis-state indices of the sector (``int64``).
    occupations : numpy.ndarray
        ``(dim, n_modes)`` spin-orbital occupations, aligned with
        :attr:`indices`.
    """

    def __init__(self, n_qubits: int, num_particles,
                 mapping: str = "jordan_wigner"):
        self.n_qubits = int(n_qubits)
        n_alpha, n_beta = (int(v) for v in num_particles)
        self.num_particles = (n_alpha, n_beta)
        self.mapping = resolve_mapping(mapping)
        if self.n_qubits > MAX_SECTOR_QUBITS:
            raise ValueError(f"a {self.n_qubits}-qubit register exceeds the "
                             f"{MAX_SECTOR_QUBITS}-qubit index range")
        self.n_modes = self.n_qubits + (
            2 if self.mapping == "parity_reduced" else 0)
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
        if self.mapping == "parity_reduced":
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

    def restrict(self, operator: PauliSum, max_entries: int | None = None):
        r"""``operator`` restricted to the sector, as a ``(dim, dim)`` CSR matrix.

        Exact for an operator that conserves ``(n_alpha, n_beta)``; the
        components of a non-conserving one that leave the sector are dropped.

        **Terms are grouped by their flip mask.**  A Pauli string sends
        :math:`|x\rangle` to a phase times :math:`|x \oplus f\rangle`, so every
        term with the same :math:`f` sends column ``j`` to the *same* row --
        and the search that finds those rows, which dominates the cost, is
        paid once per group instead of once per term.  The grouping is not a
        lucky coincidence: ``X`` and ``Y`` both set a flip bit, so all eight
        Pauli strings of a Jordan-Wigner double excitation share one mask, the
        two of a single share one, and every number-conserving (Z-only) term
        lands in :math:`f = 0`.  A molecular Hamiltonian therefore has ~9x
        fewer masks than terms, and the ratio is flat in the basis size.

        The group's terms are summed into **one** value array before anything
        is emitted, which shrinks the staged entries by the same factor.  That
        matters because every term used to contribute one entry per sector
        state: for OH in PAW-DZ (14,707 terms, 25,200 states) that was 3.7e8
        entries, about 12 GB, for a result needing a small fraction of it.
        What is left is folded into the running CSR in batches of at most
        ``max_entries`` (default :data:`RESTRICT_BATCH_ENTRIES`), which bounds
        the peak at the batch plus the result and leaves the answer unchanged
        -- the sum is linear in the terms.
        """
        import scipy.sparse as sp

        if operator.num_qubits != self.n_qubits:
            raise ValueError(f"operator acts on {operator.num_qubits} qubits, "
                             f"the sector register on {self.n_qubits}")
        budget = int(max_entries or RESTRICT_BATCH_ENTRIES)
        columns = np.arange(self.dim, dtype=np.int64)
        total = None
        rows_all, cols_all, data_all, pending = [], [], [], 0

        def fold(total):
            """Sum the batch into ``total`` and release it."""
            if not data_all:
                return total
            batch = sp.coo_matrix(
                (np.concatenate(data_all),
                 (np.concatenate(rows_all), np.concatenate(cols_all))),
                shape=(self.dim, self.dim)).tocsr()
            rows_all.clear(), cols_all.clear(), data_all.clear()
            return batch if total is None else total + batch

        for flip, members in flip_groups(operator).items():
            rows = self.positions(self.indices ^ np.int64(flip))
            inside = rows >= 0
            if not inside.any():
                continue
            if inside.all():
                # The common case for a number-conserving operator: nothing
                # leaves the sector, so no array has to be compacted.
                kept_rows, kept_cols, states = rows, columns, self.indices
            else:
                kept_rows = rows[inside]
                kept_cols = columns[inside]
                states = self.indices[inside]
            values = np.zeros(states.size, dtype=complex)
            counts = np.empty(states.size, dtype=np.uint8)
            scratch = np.empty(states.size, dtype=np.int64)
            for phase, factor in members:
                if not phase:
                    # No Y and no Z: the sign is +1 on every state.
                    values += factor
                    continue
                np.bitwise_and(states, phase, out=scratch)
                np.bitwise_count(scratch, out=counts)
                values += factor * (1 - 2 * (counts & 1).astype(np.int8))
            rows_all.append(kept_rows)
            cols_all.append(kept_cols)
            data_all.append(values)
            pending += values.size
            if pending >= budget:
                total, pending = fold(total), 0
        total = fold(total)
        if total is None:
            return sp.csr_matrix((self.dim, self.dim), dtype=complex)
        return total.tocsr()

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
