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

from functools import lru_cache
from itertools import combinations
from math import comb

import numpy as np

from .mapping import (PauliSum, _encoding_matrix, parity_tapered_qubits,
                      resolve_mapping)

#: Largest sector the enumeration accepts (states).
MAX_SECTOR_DIMENSION = 5_000_000
#: Widest register a 64-bit basis-state index can address.
MAX_SECTOR_QUBITS = 62
#: Largest out-of-sector amplitude :meth:`ParticleSector.conserves` accepts.
#: The qubit mappings drop every Pauli string at or below 1e-12
#: (:meth:`~mandacaru.core.mapping.PauliSum.simplify`), one string at a time,
#: so a number-conserving term near that scale (a grid-anisotropy integral of
#: a degenerate pair in a rotated gauge, ~1e-11 Ha) keeps only part of its flip
#: group and leaks a few 1e-12.  A genuine leak is the size of a coefficient.
LEAK_ATOL = 1e-10


@lru_cache(maxsize=65536)
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


@lru_cache(maxsize=4096)
def _mask_bit_positions(mask: int) -> tuple[tuple[int, ...], np.ndarray]:
    positions = []
    m = int(mask)
    while m:
        lsb = m & -m
        positions.append(lsb.bit_length() - 1)
        m ^= lsb
    pos_tuple = tuple(positions)
    n_buckets = 1 << len(pos_tuple)
    buckets = np.arange(n_buckets, dtype=np.int64)
    rep = np.zeros(n_buckets, dtype=np.int64)
    for k, pos in enumerate(pos_tuple):
        rep |= ((buckets >> k) & 1) << pos
    rep.flags.writeable = False
    return pos_tuple, rep


def _extract_bits(states: np.ndarray,
                  bit_positions: tuple[int, ...]) -> np.ndarray:
    bucket = (states >> bit_positions[0]) & 1
    if len(bit_positions) == 1:
        return bucket
    bucket = bucket.copy()
    tmp = np.empty_like(states)
    for k in range(1, len(bit_positions)):
        np.right_shift(states, bit_positions[k], out=tmp)
        tmp &= 1
        tmp <<= k
        bucket |= tmp
    return bucket


def _group_lut_info(members):
    if len(members) <= 1:
        return None
    base_phase = members[0][0]
    union_dz = 0
    for phase, _ in members[1:]:
        union_dz |= phase ^ base_phase
    n_bits = union_dz.bit_count()
    if n_bits == 0 or n_bits > 6:
        return None
    bit_positions, rep_states = _mask_bit_positions(union_dz)
    lut = np.zeros(rep_states.size, dtype=complex)
    for phase, factor in members:
        dz = phase ^ base_phase
        if not dz:
            lut += factor
        else:
            p = np.bitwise_count(rep_states & np.int64(dz)) & 1
            lut += factor * (1 - 2 * p.astype(np.int8))
    lut[np.abs(lut) < 1e-15] = 0.0
    nz_lut = lut != 0
    return base_phase, bit_positions, lut, nz_lut


def group_amplitudes(members, states: np.ndarray) -> np.ndarray:
    r"""Summed amplitude of one :func:`flip_groups` group on each of
    ``states``: :math:`\sum_t c_t\, s_t(x)`, the coefficient of
    :math:`|x \oplus f\rangle` in the group applied to :math:`|x\rangle`."""
    if states.size == 0:
        return np.zeros(0, dtype=complex)
    if len(members) == 1:
        phase, factor = members[0]
        if not phase:
            return np.full(states.size, factor, dtype=complex)
        parity = np.bitwise_count(states & np.int64(phase)) & 1
        return factor * (1 - 2 * parity.astype(np.int8))
    lut_info = _group_lut_info(members)
    if lut_info is not None:
        base_phase, bit_positions, lut, nz_lut = lut_info
        if not np.any(nz_lut):
            return np.zeros(states.size, dtype=complex)
        bucket = _extract_bits(states, bit_positions)
        vals = lut[bucket]
        if base_phase:
            if not np.all(nz_lut):
                active = nz_lut[bucket]
                parity = np.bitwise_count(
                    states[active] & np.int64(base_phase)) & 1
                vals[active] *= (1 - 2 * parity.astype(np.int8))
            else:
                parity = np.bitwise_count(states & np.int64(base_phase)) & 1
                vals *= (1 - 2 * parity.astype(np.int8))
        return vals
    values = np.zeros(states.size, dtype=complex)
    counts = np.empty(states.size, dtype=np.uint8)
    scratch = np.empty(states.size, dtype=np.int64)
    signs = np.empty(states.size, dtype=np.int8)
    for phase, factor in members:
        if not phase:
            # No Y and no Z: the sign is +1 on every state.
            values += factor
            continue
        np.bitwise_and(states, phase, out=scratch)
        np.bitwise_count(scratch, out=counts)
        counts &= 1
        np.subtract(1, 2 * counts, out=signs, casting="unsafe")
        values += factor * signs
    return values


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
    indices : ndarray
        Image basis indices, sorted ascending, after summing duplicates and
        dropping exact zeros.
    amplitudes : ndarray
        Complex amplitudes corresponding to the returned indices.
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


#: Widest register whose basis-state positions are looked up in a table of
#: ``2^n`` ``int32`` (64 MiB at 24 qubits) instead of searched for.
LOOKUP_MAX_QUBITS = 24


def csr_from_flip_groups(groups: dict, states: np.ndarray, locate, dim: int):
    r"""The CSR matrix of grouped Pauli terms over the basis ``states``.

    ``groups`` is :func:`flip_groups`; ``locate(images)`` gives the position of
    each image state in ``states`` (``-1`` if it is not in the basis).  A
    group's strings send state ``j`` to the one image ``states[j] ^ f``, so a
    group fills at most one entry per column and per row, and two groups never
    reach the same ``(row, column)`` (the flip mask is ``states[row] ^
    states[col]``).  The matrix therefore has exactly ``sum`` of the groups'
    entries and **no duplicate to merge**: a first pass counts each row's
    entries, the second writes them into place, and nothing larger than the
    result is ever held (the older coordinate-format fold staged 32 bytes per
    entry, then summed copies of the running matrix: three times the result
    at its peak).  Rows come out sorted by column; exact zeros (terms that
    cancelled) are dropped.
    """
    import scipy.sparse as sp

    if not groups or dim == 0:
        return sp.csr_matrix((dim, dim), dtype=complex)

    columns = np.arange(dim, dtype=np.int64)

    def reach(flip):
        rows = locate(states ^ np.int64(flip))
        inside = rows >= 0
        if inside.all():
            return rows, columns, states
        return rows[inside], columns[inside], states[inside]

    if len(groups) == 1:
        (flip, members), = groups.items()
        rows, cols, kept = reach(flip)
        if rows.size == 0:
            return sp.csr_matrix((dim, dim), dtype=complex)
        vals = group_amplitudes(members, kept)
        nz = np.flatnonzero(vals)
        if nz.size == 0:
            return sp.csr_matrix((dim, dim), dtype=complex)
        if nz.size != vals.size:
            rows = rows[nz]
            cols = cols[nz]
            vals = vals[nz]
        if flip and rows.size > 1:
            order = np.argsort(rows)
            rows = rows[order]
            cols = cols[order]
            vals = vals[order]
        index_type = np.int32 if dim < np.iinfo(np.int32).max else np.int64
        counts = np.zeros(dim, dtype=index_type)
        counts[rows] = 1
        indptr = np.empty(dim + 1, dtype=index_type)
        indptr[0] = 0
        np.cumsum(counts, out=indptr[1:])
        matrix = sp.csr_matrix((vals, cols.astype(index_type), indptr),
                               shape=(dim, dim))
        matrix.has_sorted_indices = True
        return matrix

    lut_infos = {flip: _group_lut_info(members)
                 for flip, members in groups.items()}
    counts = np.zeros(dim, dtype=np.int64)
    for flip in groups:
        info = lut_infos[flip]
        if info is not None and not np.any(info[3]):
            continue
        rows = locate(states ^ np.int64(flip))
        inside = rows >= 0
        if not inside.all():
            rows = rows[inside]
            kept = states[inside]
        else:
            kept = states
        if rows.size == 0:
            continue
        if info is not None and not np.all(info[3]):
            active = info[3][_extract_bits(kept, info[1])]
            rows = rows[active]
        counts[rows] += 1
    indptr = np.zeros(dim + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    nnz = int(indptr[-1])
    if nnz == 0:
        return sp.csr_matrix((dim, dim), dtype=complex)
    index_type = np.int32 if nnz < np.iinfo(np.int32).max else np.int64
    indices = np.empty(nnz, dtype=index_type)
    data = np.empty(nnz, dtype=complex)
    fill = indptr[:-1].copy()
    for flip, members in groups.items():
        info = lut_infos[flip]
        if info is not None and not np.any(info[3]):
            continue
        rows, cols, kept = reach(flip)
        if rows.size == 0:
            continue
        if info is not None:
            base_phase, bit_positions, lut, nz_lut = info
            bucket = _extract_bits(kept, bit_positions)
            if not np.all(nz_lut):
                active = nz_lut[bucket]
                rows = rows[active]
                cols = cols[active]
                kept = kept[active]
                bucket = bucket[active]
            vals = lut[bucket]
            if base_phase and kept.size:
                parity = np.bitwise_count(kept & np.int64(base_phase)) & 1
                vals = np.where(parity, -vals, vals)
        else:
            vals = group_amplitudes(members, kept)
        slot = fill[rows]
        indices[slot] = cols
        data[slot] = vals
        fill[rows] += 1
    matrix = sp.csr_matrix((data, indices, indptr.astype(index_type)),
                           shape=(dim, dim))
    matrix.sort_indices()
    matrix.eliminate_zeros()
    return matrix


_EMPTY_INT64 = np.zeros(0, dtype=np.int64)


@lru_cache(maxsize=32)
def _sector_basis_cached(n_qubits: int, num_particles: tuple[int, int],
                         mapping: str, spin_conserving: bool):
    n_alpha, n_beta = num_particles
    n_modes = n_qubits + (2 if mapping == "parity_reduced" else 0)
    M = n_modes // 2
    n_total = n_alpha + n_beta
    dim = (comb(M, n_alpha) * comb(M, n_beta) if spin_conserving
           else comb(2 * M, n_total))
    if spin_conserving:
        alpha = np.zeros((comb(M, n_alpha), M), dtype=np.int64)
        for i, occupied in enumerate(combinations(range(M), n_alpha)):
            alpha[i, list(occupied)] = 1
        beta = np.zeros((comb(M, n_beta), M), dtype=np.int64)
        for i, occupied in enumerate(combinations(range(M), n_beta)):
            beta[i, list(occupied)] = 1
        occupations = np.concatenate(
            [np.repeat(alpha, beta.shape[0], axis=0),
             np.tile(beta, (alpha.shape[0], 1))], axis=1)
    else:
        occupations = np.zeros((dim, 2 * M), dtype=np.int64)
        for i, occupied in enumerate(combinations(range(2 * M), n_total)):
            occupations[i, list(occupied)] = 1

    # Qubit register q = beta_matrix . x (mod 2), as reference_qubit_bits.
    encoding = _encoding_matrix(mapping, n_modes).astype(np.int64)
    bits = (occupations @ encoding.T) % 2
    if mapping == "parity_reduced":
        drop = set(parity_tapered_qubits(n_modes))
        bits = bits[:, [k for k in range(n_modes) if k not in drop]]
    weights = np.int64(1) << (np.int64(n_qubits) - 1
                              - np.arange(n_qubits, dtype=np.int64))
    index = bits @ weights
    order = np.argsort(index)
    indices = index[order]
    occupations = occupations[order]
    if np.unique(indices).size != indices.size:
        raise RuntimeError("two determinants share a basis state; the "
                           "encoding is not invertible on this sector")
    indices.flags.writeable = False
    occupations.flags.writeable = False
    lookup_table = None
    if n_qubits <= 16:
        lookup_table = np.full(1 << n_qubits, -1, dtype=np.int32)
        lookup_table[indices] = np.arange(indices.size, dtype=np.int32)
        lookup_table.flags.writeable = False
    return indices, occupations, lookup_table


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
    spin_conserving : bool
        ``False`` for a Hamiltonian that conserves the particle number but not
        :math:`S_z` (spin-orbit coupling): the sector is then every state with
        :math:`N = n_\alpha + n_\beta` electrons in the :math:`2M`
        spin-orbitals, :math:`\binom{2M}{N}` of them, and ``num_particles``
        only fixes that total.  The parity reduction tapers the alpha-parity
        qubit and is refused.
    Attributes
    ----------
    indices : numpy.ndarray
        Sorted basis-state indices of the sector (``int64``).
    occupations : numpy.ndarray
        ``(dim, n_modes)`` spin-orbital occupations, aligned with
        :attr:`indices`.
    """

    def __init__(self, n_qubits: int, num_particles,
                 mapping: str = "jordan_wigner",
                 spin_conserving: bool = True):
        self.n_qubits = int(n_qubits)
        n_alpha, n_beta = (int(v) for v in num_particles)
        self.num_particles = (n_alpha, n_beta)
        self.mapping = resolve_mapping(mapping)
        self.spin_conserving = bool(spin_conserving)
        if not self.spin_conserving and self.mapping == "parity_reduced":
            raise ValueError(
                "the parity reduction tapers the alpha-parity qubit, which a "
                "Hamiltonian that does not conserve S_z (spin-orbit coupling) "
                "does not leave fixed; use 'jordan_wigner', 'parity' or "
                "'bravyi_kitaev'")
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
        n_total = n_alpha + n_beta
        dim = (comb(M, n_alpha) * comb(M, n_beta) if self.spin_conserving
               else comb(2 * M, n_total))
        if dim > MAX_SECTOR_DIMENSION:
            raise ValueError(
                f"the {self.num_particles} sector of {M} orbitals has {dim} "
                f"states (limit {MAX_SECTOR_DIMENSION})")

        indices, occupations, lookup_table = _sector_basis_cached(
            self.n_qubits, self.num_particles, self.mapping,
            self.spin_conserving)
        self.indices = indices
        self.occupations = occupations
        if lookup_table is not None:
            self.__dict__["_lookup_table"] = lookup_table

    # -- lookup ----------------------------------------------------------- #

    @property
    def dim(self) -> int:
        """Number of basis states in the sector."""
        return int(self.indices.size)

    def _lookup(self):
        """``position of every register state`` as an ``int32`` table (``-1``
        outside the sector), or ``None`` on a register too wide for one."""
        if self.n_qubits > LOOKUP_MAX_QUBITS:
            return None
        table = self.__dict__.get("_lookup_table")
        if table is None:
            table = np.full(1 << self.n_qubits, -1, dtype=np.int32)
            table[self.indices] = np.arange(self.dim, dtype=np.int32)
            self.__dict__["_lookup_table"] = table
        return table

    def positions(self, indices):
        """Sector positions of basis-state ``indices`` (``-1`` if outside)."""
        indices = np.asarray(indices, dtype=np.int64)
        table = self._lookup()
        if table is not None and indices.size and indices.min() >= 0 \
                and indices.max() < table.size:
            return table[indices]
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
        r"""``operator`` restricted to the sector, as a ``(dim, dim)`` CSR matrix.

        Exact for an operator that conserves ``(n_alpha, n_beta)``; the
        components of a non-conserving one that leave the sector are dropped.

        **Terms are grouped by their flip mask.**  A Pauli string sends
        :math:`|x\rangle` to a phase times :math:`|x \oplus f\rangle`, so every
        term with the same :math:`f` sends column ``j`` to the *same* row --
        and the lookup that finds those rows, which dominates the cost, is
        paid once per group instead of once per term.  The grouping is not a
        lucky coincidence: ``X`` and ``Y`` both set a flip bit, so all eight
        Pauli strings of a Jordan-Wigner double excitation share one mask, the
        two of a single share one, and every number-conserving (Z-only) term
        lands in :math:`f = 0`.  A molecular Hamiltonian therefore has ~9x
        fewer masks than terms, and the ratio is flat in the basis size.

        The group's terms are summed into **one** value array, and the groups
        are written straight into the CSR arrays
        (:func:`csr_from_flip_groups`): the peak is the result plus a few
        vectors of ``dim`` entries.  (Every term used to contribute one entry
        per sector state -- for OH in PAW-LCAO-DZ, 14,707 terms and 25,200
        states, 3.7e8 entries, about 12 GB, for a result needing a small
        fraction of it -- and the batched fold that replaced it still held
        about three times the result.)
        """
        if operator.num_qubits != self.n_qubits:
            raise ValueError(f"operator acts on {operator.num_qubits} qubits, "
                             f"the sector register on {self.n_qubits}")
        return csr_from_flip_groups(flip_groups(operator), self.indices,
                                    self.positions, self.dim)

    # -- states ----------------------------------------------------------- #

    def _outside_states(self, flip: int) -> np.ndarray:
        cache = self.__dict__.get("_outside_cache")
        if cache is not None:
            cached = cache.get(flip)
            if cached is not None:
                return cached
        outside = self.positions(self.indices ^ np.int64(flip)) < 0
        states = self.indices[outside] if outside.any() else _EMPTY_INT64
        if self.dim <= 10_000:
            if cache is None:
                cache = {}
                self.__dict__["_outside_cache"] = cache
            if len(cache) < 2048:
                cache[flip] = states
        return states

    def conserves(self, operator: PauliSum, atol: float = LEAK_ATOL) -> bool:
        r"""Whether ``operator`` maps the sector into itself.

        :meth:`restrict` silently drops whatever leaves the sector, so a
        leaking operator would be *replaced* by its projection: for an ansatz
        that matters, because :math:`e^{PAP} \neq P e^{A} P` in general.  A
        term-by-term image test would reject operators whose leaking parts
        cancel, so the amplitudes of each image are summed first (per source
        state) and only a surviving amplitude outside the sector counts.

        From one source state, terms with different flip masks reach
        different images, so only a :func:`flip_groups` group can cancel:
        each group is tested on every sector state at once, as
        :meth:`restrict` builds it, rather than state by state.
        """
        if not operator.terms:
            return True
        if operator.num_qubits != self.n_qubits:
            raise ValueError(f"operator acts on {operator.num_qubits} qubits, "
                             f"the sector register on {self.n_qubits}")
        for flip, members in flip_groups(operator).items():
            if not flip:
                continue                  # diagonal: every image is its source
            outside_states = self._outside_states(flip)
            if outside_states.size == 0:
                continue
            values = group_amplitudes(members, outside_states)
            if np.any(np.abs(values) > atol):
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
