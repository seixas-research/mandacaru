# -*- coding: utf-8 -*-
# file: core/matrix_free.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Qubit operators applied to a state vector without storing their matrix.

A Pauli string :math:`P = \bigotimes_k \sigma_k` is a product of ``2x2``
blocks, and on a computational basis state it acts as a single permutation
with a sign:

.. math::

    P\,|x\rangle = i^{n_Y}\,(-1)^{|x \wedge z|}\;|x \oplus f\rangle ,

with ``f`` the bits an ``X`` or ``Y`` flips, ``z`` the bits a ``Y`` or ``Z``
reads and :math:`|x \wedge z|` a popcount (:func:`~mandacaru.core.sector.pauli_masks`).
Applying the string to a vector therefore needs one XOR, one popcount and one
gather per amplitude -- the Kronecker product of the ``2x2`` blocks is never
formed, neither dense (:math:`4^n` entries) nor sparse (one entry per
amplitude *per string*).

:class:`PauliOperator` applies a whole :class:`~mandacaru.core.mapping.PauliSum`
this way.  Its terms are grouped by flip mask, as
:meth:`~mandacaru.core.sector.ParticleSector.restrict` does: every string of a
group sends :math:`|x\rangle` to the same :math:`|x \oplus f\rangle`, so a
group is one gather weighted by the group's summed signs
:math:`d_f(x) = \sum_k c_k (-1)^{|x \wedge z_k|}`.  A molecular Hamiltonian has
about nine times fewer groups than terms.  For a large group on the full
register (the ``f = 0`` group of number operators and ``ZZ`` pairs) the signed
sum is a Walsh-Hadamard transform of the coefficients placed at their ``z``
masks, :math:`O(n\,2^n)` instead of :math:`O(T_f\,2^n)`.

**Memory.**  The operator holds its groups as integer masks and coefficients
-- polynomial in the register -- plus a cache bounded by ``cache_bytes``: the
first groups, as many as fit, stored as one CSR block.  With every group
cached the product is a CSR product; past the bound, each product recomputes
the remaining groups and only a few vectors of the state's length are
allocated for them.  The state vector
itself is the one exponential object left, :math:`16 \cdot 2^n` bytes on the
full register (4 GB at 28 qubits, 1 TB at 36) or :math:`16 \cdot \dim` in a
:class:`~mandacaru.core.sector.ParticleSector`.  No representation of the
operator removes that bound: a 100-qubit register can be built, mapped and
differentiated symbolically (:mod:`mandacaru.core.pauli_algebra`) but not
simulated.
"""

from __future__ import annotations

import numpy as np

from .mapping import PauliSum
from .sector import flip_groups

#: Default bound (bytes) on the CSR block a :class:`PauliOperator` keeps
#: between products.  Groups past it are recomputed on every product: slower,
#: never larger.
MATRIX_FREE_CACHE_BYTES = 256 * 2 ** 20
#: Widest full register a 64-bit basis-state index addresses.
MAX_MATRIX_FREE_QUBITS = 62
#: The ``sparse=`` value that selects this module's operators.
MATRIX_FREE = "matrix-free"
#: Share of the physical memory the stored (CSR) operators may claim before
#: ``sparse="auto"`` switches to matrix-free products.
STORED_MEMORY_FRACTION = 0.25
#: State vectors an adaptive run holds at once: the state, ``H psi``, one
#: generator image, the optimizer's gradient images and the product's
#: temporaries.  A register whose vectors cannot fit this many times is
#: refused before anything is allocated.
STATE_VECTOR_COPIES = 8


def statevector_bytes(dim: int, columns: int = 1) -> int:
    """Bytes of ``columns`` complex state vectors of ``dim`` amplitudes."""
    return 16 * int(dim) * int(columns)


def physical_memory_bytes() -> int:
    """Installed memory of this machine (``sysconf``), or 16 GiB if unknown."""
    import os

    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return 16 * 2 ** 30


def format_bytes(n: float) -> str:
    """``n`` bytes in binary units, e.g. ``1.5 TiB``."""
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"):
        if value < 1024.0 or unit == "EiB":
            return f"{value:.3g} {unit}"
        value /= 1024.0
    return f"{value:.3g} EiB"


def stored_operator_bytes(operator: PauliSum, dim: int) -> int:
    """Upper bound on the CSR matrix of ``operator`` over ``dim`` states.

    One entry per state per flip group (a complex value and a column index);
    a group whose images leave a sector stores fewer, so the real matrix is
    never larger.  Counting the groups needs only the ``x`` bit table
    (:class:`~mandacaru.core.pauli_algebra.PauliTable`), not the matrix.
    """
    from .pauli_algebra import PauliTable

    if not operator.terms:
        return 0
    flips = PauliTable.from_pauli_sum(operator).x
    groups = int(np.unique(flips, axis=0).shape[0])
    index_bytes = 4 if dim < 2 ** 31 else 8
    return groups * int(dim) * (16 + index_bytes)


def refuse_unaffordable_state(n_qubits: int, dim: int,
                              in_sector: bool = False) -> None:
    """Raise :class:`MemoryError` when the run's state vectors cannot fit.

    No representation of the operators helps then: the state itself is the
    exponential object.  The message gives the arithmetic, so a 100-qubit
    request fails with a number rather than an allocation error deep in
    NumPy.
    """
    needed = statevector_bytes(dim, STATE_VECTOR_COPIES)
    available = physical_memory_bytes()
    if needed <= available:
        return
    space = (f"its particle-number sector has {dim:,} states" if in_sector
             else f"its state vector has 2^{n_qubits} amplitudes")
    raise MemoryError(
        f"a {n_qubits}-qubit simulation cannot fit: {space}, "
        f"{format_bytes(statevector_bytes(dim))} each and ~{STATE_VECTOR_COPIES} "
        f"held at once ({format_bytes(needed)}) against "
        f"{format_bytes(available)} of memory.  The Hamiltonian, the pool and "
        f"their commutators are still available symbolically (PauliSum, "
        f"PauliSum.commutator); exact state-vector simulation ends near 30 "
        f"qubits on the full register")


def walsh_hadamard_signs(n_qubits: int, phases, factors) -> np.ndarray:
    r""":math:`d(x) = \sum_k c_k (-1)^{|x \wedge z_k|}` for every ``x < 2^n``.

    The unnormalized Walsh-Hadamard transform of the vector holding
    :math:`c_k` at index :math:`z_k`: :math:`n` butterfly passes of the
    ``2x2`` block :math:`\begin{pmatrix}1&1\\1&-1\end{pmatrix}`, each over the
    whole vector, instead of one popcount pass per term.
    """
    dim = 1 << int(n_qubits)
    values = np.zeros(dim, dtype=complex)
    np.add.at(values, np.asarray(phases, dtype=np.int64),
              np.asarray(factors, dtype=complex))
    half = 1
    while half < dim:
        # Pair every index with the one differing in bit log2(half).
        view = values.reshape(-1, 2, half)
        upper = view[:, 0, :].copy()
        view[:, 0, :] += view[:, 1, :]
        view[:, 1, :] = upper - view[:, 1, :]
        half *= 2
    return values


def _signed_sum(states: np.ndarray, phases, factors) -> np.ndarray:
    r""":math:`\sum_k c_k (-1)^{|x \wedge z_k|}` for the given basis ``states``."""
    values = np.zeros(states.size, dtype=complex)
    counts = np.empty(states.size, dtype=np.uint8)
    scratch = np.empty(states.size, dtype=np.int64)
    for phase, factor in zip(phases, factors):
        if not phase:
            # No Y and no Z: the sign is +1 on every state.
            values += factor
            continue
        np.bitwise_and(states, phase, out=scratch)
        np.bitwise_count(scratch, out=counts)
        values += factor * (1 - 2 * (counts & 1).astype(np.int8))
    return values


class PauliOperator:
    r"""A :class:`~mandacaru.core.mapping.PauliSum` as a matrix-free linear map.

    Supports ``op @ vector`` for a single state (``(dim,)``) or a stack of
    column states (``(dim, k)``), which is all the state-vector drivers ask
    of their Hamiltonian and generator matrices.

    Parameters
    ----------
    operator : PauliSum
        The qubit operator.
    sector : ParticleSector, optional
        Act on the sector's basis states only
        (:class:`~mandacaru.core.sector.ParticleSector`): the vector then has
        ``sector.dim`` amplitudes, and a component that leaves the sector is
        dropped -- the same restriction as
        :meth:`~mandacaru.core.sector.ParticleSector.restrict`.
    hermitian : bool
        Keep only the Hermitian part: the real part of every coefficient
        (Pauli strings are Hermitian).  What ``0.5 * (H + H^dagger)`` does to
        a stored matrix, removing rounding noise from a Hamiltonian.
    cache_bytes : int, optional
        Bound on what is kept between products (default
        :data:`MATRIX_FREE_CACHE_BYTES`): the groups, in order, are stored as
        one CSR block while their entries fit, and the rest are recomputed on
        every product.  ``0`` keeps nothing.
    """

    def __init__(self, operator: PauliSum, sector=None, hermitian: bool = False,
                 cache_bytes: int | None = None):
        n = int(operator.num_qubits)
        if sector is None and n > MAX_MATRIX_FREE_QUBITS:
            raise ValueError(
                f"a {n}-qubit state vector has 2^{n} amplitudes; the "
                f"matrix-free product addresses at most "
                f"{MAX_MATRIX_FREE_QUBITS} qubits")
        if sector is not None and sector.n_qubits != n:
            raise ValueError(f"operator acts on {n} qubits, the sector "
                             f"register on {sector.n_qubits}")
        if hermitian:
            operator = PauliSum({label: complex(coeff.real)
                                 for label, coeff in operator.terms.items()},
                                num_qubits=n)
        self.operator = operator
        self.n_qubits = n
        self.sector = sector
        self.dim = sector.dim if sector is not None else 1 << n
        self.shape = (self.dim, self.dim)
        self.dtype = np.dtype(complex)
        # (flip, phase masks, factors): integers and coefficients only.
        self._groups = []
        for flip, members in flip_groups(operator).items():
            phases = np.array([phase for phase, _ in members], dtype=np.int64)
            factors = np.array([factor for _, factor in members], dtype=complex)
            self._groups.append((int(flip), phases, factors))
        self._budget = int(MATRIX_FREE_CACHE_BYTES if cache_bytes is None
                           else cache_bytes)
        self._states = (sector.indices if sector is not None
                        else np.arange(self.dim, dtype=np.int64))
        # Built on the first product: the cached groups as one CSR block,
        # and the groups past the budget, recomputed on every product.
        self._stored = None
        self._recomputed: list[int] | None = None

    # -- the product ------------------------------------------------------ #

    @property
    def num_groups(self) -> int:
        """Distinct flip masks: gathers per product."""
        return len(self._groups)

    def stored_bytes_bound(self) -> int:
        """Most the CSR block can take with every group stored."""
        index_bytes = 4 if self.dim < 2 ** 31 else 8
        return (index_bytes * (self.dim + 1)
                + len(self._groups) * self.dim * (16 + index_bytes))

    @property
    def cached_bytes(self) -> int:
        """Bytes kept between products (the CSR block of the cached groups)."""
        if self._stored is None:
            return 0
        return int(self._stored.data.nbytes + self._stored.indices.nbytes
                   + self._stored.indptr.nbytes)

    def _group_entries(self, index: int):
        r"""``(rows, columns, weights)`` of group ``index``.

        ``out[rows] += weights * vector[columns]`` applies the group; on the
        full register the columns are every state and the rows their images
        :math:`x \oplus f`.  A large group on the full register gets its
        weights from one Walsh-Hadamard transform.
        """
        flip, phases, factors = self._groups[index]
        if self.sector is None:
            if phases.size > self.n_qubits:
                weights = walsh_hadamard_signs(self.n_qubits, phases, factors)
            else:
                weights = _signed_sum(self._states, phases, factors)
            return self._states ^ np.int64(flip), self._states, weights
        rows = self.sector.positions(self._states ^ np.int64(flip))
        columns = np.flatnonzero(rows >= 0)
        return (rows[columns], columns,
                _signed_sum(self._states[columns], phases, factors))

    def _prepare(self) -> None:
        """Store groups, in order, while their CSR entries fit the budget."""
        import scipy.sparse as sp

        index_bytes = 4 if self.dim < 2 ** 31 else 8
        used = index_bytes * (self.dim + 1)          # the CSR row pointer
        rows, columns, weights, recomputed = [], [], [], []
        for index in range(len(self._groups)):
            if recomputed or used > self._budget:
                recomputed.append(index)
                continue
            r, c, w = self._group_entries(index)
            size = w.size * (16 + index_bytes)
            if used + size > self._budget:
                recomputed.append(index)
                continue
            rows.append(r), columns.append(c), weights.append(w)
            used += size
        if weights:
            # Groups never share an entry: distinct flips send a column to
            # distinct rows, so the COO -> CSR conversion sums nothing.
            self._stored = sp.coo_matrix(
                (np.concatenate(weights),
                 (np.concatenate(rows), np.concatenate(columns))),
                shape=self.shape).tocsr()
        self._recomputed = recomputed

    def matvec(self, vector) -> np.ndarray:
        """``operator @ vector`` for ``(dim,)`` or ``(dim, k)`` input."""
        vector = np.asarray(vector)
        if vector.shape[0] != self.dim:
            raise ValueError(f"expected {self.dim} amplitudes, "
                             f"got {vector.shape[0]}")
        if self._recomputed is None:
            self._prepare()
        out = (np.asarray(self._stored @ vector, dtype=complex)
               if self._stored is not None
               else np.zeros(vector.shape, dtype=complex))
        column = (slice(None),) + (None,) * (vector.ndim - 1)
        for index in self._recomputed:
            rows, columns, weights = self._group_entries(index)
            if self.sector is None:
                # x -> x ^ f is an involution of the whole index range, so the
                # scatter is a gather: out[y] += w(y ^ f) v(y ^ f).
                out += (weights[column] * vector)[rows]
            else:
                out[rows] += weights[column] * vector[columns]
        return out

    __matmul__ = matvec

    def expectation(self, vector) -> float:
        r"""Real part of :math:`\langle\psi|O|\psi\rangle`."""
        vector = np.asarray(vector, dtype=complex)
        return float(np.real(np.vdot(vector, self.matvec(vector))))

    # -- algebra the solvers need ----------------------------------------- #

    def adjoint(self) -> PauliOperator:
        """The adjoint: every coefficient conjugated (Pauli strings are Hermitian)."""
        conjugate = PauliSum({label: complex(coeff).conjugate()
                              for label, coeff in self.operator.terms.items()},
                             num_qubits=self.n_qubits)
        return PauliOperator(conjugate, sector=self.sector,
                             cache_bytes=self._budget)

    def trace(self) -> complex:
        """Trace over the vector space the operator acts on."""
        for index, (flip, _, _) in enumerate(self._groups):
            if flip == 0:
                return complex(np.sum(self._group_entries(index)[2]))
        return 0j

    def as_linear_operator(self):
        """A :class:`scipy.sparse.linalg.LinearOperator` (for ``expm_multiply``)."""
        from scipy.sparse.linalg import LinearOperator

        adjoint = self.adjoint()
        return LinearOperator(self.shape, matvec=self.matvec,
                              matmat=self.matvec, rmatvec=adjoint.matvec,
                              rmatmat=adjoint.matvec, dtype=complex)

    def __repr__(self) -> str:
        where = (f"sector dim={self.dim}" if self.sector is not None
                 else f"2^{self.n_qubits}")
        return (f"PauliOperator({len(self.operator.terms)} terms, "
                f"{self.num_groups} flip groups, {where})")
