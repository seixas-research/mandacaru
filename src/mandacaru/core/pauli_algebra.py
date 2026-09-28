# -*- coding: utf-8 -*-
# file: core/pauli_algebra.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Pauli-string algebra on bit tables, at any register width.

A Pauli string on :math:`n` qubits is two bit vectors, ``x`` (the qubits
carrying an ``X`` or ``Y``) and ``z`` (a ``Z`` or ``Y``), with

.. math::

    \sigma(x, z) = i^{|x \wedge z|}\, X^{x} Z^{z},
    \qquad Y = i\,XZ .

Products and commutators then reduce to integer operations on those vectors,
never to a matrix:

.. math::

    \sigma_1 \sigma_2 = i^{\,a_1 + a_2 - a_3 + 2|z_1 \wedge x_2|}\;
    \sigma(x_1 \oplus x_2,\; z_1 \oplus z_2),
    \qquad a_k = |x_k \wedge z_k| ,

and two strings anticommute exactly when
:math:`|x_1 \wedge z_2| + |z_1 \wedge x_2|` is odd, in which case
:math:`[\sigma_1, \sigma_2] = 2\,\sigma_1\sigma_2`; otherwise the commutator
vanishes.

:class:`PauliTable` stores an operator's strings as ``(terms, words)`` arrays
of 64-bit words, so one pass of vectorized bit operations handles every term,
and the width is bounded by nothing but memory -- a 100-qubit string is two
words.  :func:`commutator` is what an adaptive solver needs to state its pool
gradients as observables, :math:`g_i = \langle\psi|[H, A_i]|\psi\rangle`,
without the :math:`2^n`-dimensional state the analytic formula
:math:`2\,\mathrm{Re}\langle H\psi | A_i\psi\rangle` requires.

The cost is set by the term counts: :math:`O(T_A T_B)` word operations for
:math:`[A, B]`.  For an excitation generator (eight strings) against a
Hamiltonian that is eight passes over the Hamiltonian's table.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .mapping import PauliSum

#: ``x + 2 z`` of a Pauli letter -> its character.
_LETTERS = np.frombuffer(b"IXZY", dtype=np.uint8)
#: ``i**k`` for ``k = 0..3``, exact.
_I_POWERS = np.array([1, 1j, -1, -1j], dtype=complex)


def _pack(bits: np.ndarray) -> np.ndarray:
    """``(T, n)`` booleans -> ``(T, words)`` uint64, bit ``k`` = qubit ``k``."""
    terms, n = bits.shape
    words = max(1, -(-n // 64))
    padded = np.zeros((terms, 64 * words), dtype=bool)
    padded[:, :n] = bits
    return np.packbits(padded, axis=1, bitorder="little").view(np.uint64)


def _unpack(words: np.ndarray, n: int) -> np.ndarray:
    """Inverse of :func:`_pack`: ``(T, words)`` uint64 -> ``(T, n)`` uint8."""
    raw = np.ascontiguousarray(words).view(np.uint8)
    return np.unpackbits(raw, axis=1, bitorder="little")[:, :n]


def _popcount(words: np.ndarray) -> np.ndarray:
    """Per-row popcount of a ``(T, words)`` uint64 table."""
    return np.bitwise_count(words).sum(axis=1, dtype=np.int64)


class PauliTable:
    r"""A :class:`~mandacaru.core.mapping.PauliSum` as symplectic bit tables.

    Attributes
    ----------
    x, z : numpy.ndarray
        ``(terms, words)`` uint64; bit ``k`` of the row is qubit ``k``.
    coeffs : numpy.ndarray
        Complex coefficients of :math:`\sigma(x, z)`, the Pauli strings as
        written (``Y``, not ``iXZ``).
    num_qubits : int
    """

    def __init__(self, x: np.ndarray, z: np.ndarray, coeffs: np.ndarray,
                 num_qubits: int):
        self.x = x
        self.z = z
        self.coeffs = coeffs
        self.num_qubits = int(num_qubits)

    @classmethod
    def from_pauli_sum(cls, operator: PauliSum) -> PauliTable:
        n = int(operator.num_qubits)
        labels = list(operator.terms)
        coeffs = np.fromiter((complex(operator.terms[label]) for label in labels),
                             dtype=complex, count=len(labels))
        if not labels:
            empty = np.zeros((0, max(1, -(-n // 64))), dtype=np.uint64)
            return cls(empty, empty.copy(), coeffs, n)
        letters = np.frombuffer("".join(labels).encode("ascii"),
                                dtype=np.uint8).reshape(len(labels), n)
        is_x = (letters == ord("X")) | (letters == ord("Y"))
        is_z = (letters == ord("Z")) | (letters == ord("Y"))
        return cls(_pack(is_x), _pack(is_z), coeffs, n)

    def to_pauli_sum(self, atol: float = 0.0) -> PauliSum:
        """Back to labels, summing repeated strings and dropping ``|c| <= atol``."""
        n = self.num_qubits
        if not self.coeffs.size:
            return PauliSum(num_qubits=n)
        # Merge duplicates: one row of 2*words uint64 per string.
        keys = np.ascontiguousarray(np.concatenate([self.x, self.z], axis=1))
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        summed = np.zeros(unique.shape[0], dtype=complex)
        np.add.at(summed, inverse.ravel(), self.coeffs)
        keep = np.abs(summed) > atol
        unique, summed = unique[keep], summed[keep]
        if not summed.size:
            return PauliSum(num_qubits=n)
        words = self.x.shape[1]
        codes = (_unpack(unique[:, :words], n)
                 + 2 * _unpack(unique[:, words:], n))
        text = _LETTERS[codes].tobytes().decode("ascii")
        labels = (text[i * n:(i + 1) * n] for i in range(summed.size))
        return PauliSum._from_valid_terms(dict(zip(labels, summed.tolist())), n)

    def __len__(self) -> int:
        return int(self.coeffs.size)


def commutator(a: PauliSum, b: PauliSum, atol: float = 1e-12) -> PauliSum:
    r""":math:`[A, B] = AB - BA` of two qubit operators, term by term.

    Only anticommuting string pairs contribute, each :math:`2\,\sigma_i\sigma_j`
    (module docstring).  The loop runs over the operator with fewer terms, and
    every step is one vectorized pass over the other's table, so the
    commutator of a :math:`10^6`-term Hamiltonian with an excitation generator
    takes a handful of passes and no matrix.  Strings that cancel, and
    coefficients at or below ``atol``, are dropped.
    """
    width = a.num_qubits
    if b.num_qubits != width:
        raise ValueError(f"commutator of a {width}-qubit and a "
                         f"{b.num_qubits}-qubit operator")
    swap = len(a.terms) < len(b.terms)
    big = PauliTable.from_pauli_sum(b if swap else a)
    small = PauliTable.from_pauli_sum(a if swap else b)
    # Written as [big, small]; [a, b] = -[b, a] undoes the swap at the end.
    phase_big = _popcount(big.x & big.z)
    xs, zs, cs = [], [], []
    for row in range(len(small)):
        x2, z2 = small.x[row], small.z[row]
        symplectic = _popcount(big.x & z2) + _popcount(big.z & x2)
        odd = (symplectic & 1).astype(bool)
        if not odd.any():
            continue
        x1, z1 = big.x[odd], big.z[odd]
        x3, z3 = x1 ^ x2, z1 ^ z2
        exponent = (phase_big[odd] + int(_popcount((x2 & z2)[None, :])[0])
                    - _popcount(x3 & z3) + 2 * _popcount(z1 & x2)) % 4
        xs.append(x3)
        zs.append(z3)
        cs.append(2.0 * big.coeffs[odd] * small.coeffs[row]
                  * _I_POWERS[exponent])
    if not cs:
        return PauliSum(num_qubits=width)
    table = PauliTable(np.concatenate(xs), np.concatenate(zs),
                       np.concatenate(cs), width)
    if swap:
        table.coeffs = -table.coeffs
    return table.to_pauli_sum(atol=atol)


# -- fermion-to-qubit mapping ------------------------------------------------ #

#: Fermion terms mapped per vectorized batch (bounds the bit tables' memory:
#: a two-body batch holds 16 strings per term).
MAPPING_BATCH_TERMS = 200_000


@lru_cache(maxsize=16)
def _ladder_tables(method: str, n: int):
    r"""Bit tables of every ladder operator of an encoding on ``n`` modes.

    Returns ``(x, z, phase, coeff)`` with ``x``, ``z`` of shape
    ``(n, 2, words)`` (mode, string), ``phase`` the :math:`|x \wedge z|` of
    each string and ``coeff`` of shape ``(n, 2, 2)`` (mode, string, dagger).
    The strings come from :func:`~mandacaru.core.mapping._ladder_pauli`
    itself, so the letters are exactly the ones the term-by-term map writes.
    Cached: a pool maps thousands of small operators on one register.
    """
    from .mapping import _ladder_pauli, _mapping_sets

    update, parity, remainder = _mapping_sets(method, n)
    words = max(1, -(-n // 64))
    x = np.zeros((n, 2, words), dtype=np.uint64)
    z = np.zeros((n, 2, words), dtype=np.uint64)
    coeff = np.zeros((n, 2, 2), dtype=complex)
    for j in range(n):
        for dagger in (False, True):
            table = PauliTable.from_pauli_sum(_ladder_pauli(
                n, j, dagger, update[j], parity[j], remainder[j]))
            x[j], z[j] = table.x, table.z
            coeff[j, :, int(dagger)] = table.coeffs
    phase = np.bitwise_count(x & z).sum(axis=2, dtype=np.int64)
    return x, z, phase, coeff


def _map_products(modes: np.ndarray, daggers: np.ndarray, coeffs: np.ndarray,
                  tables) -> PauliTable:
    r"""Qubit strings of ``coeffs[t] * prod_l a_{modes[t, l]}^{(dagger)}``.

    Each ladder operator is a sum of two strings, so a product of ``k`` of
    them expands into :math:`2^k` strings per term.  All terms and all
    :math:`2^k` choices are one table of rows, multiplied left to right with
    the symplectic product rule (module docstring): ``k`` vectorized steps
    whatever the number of terms.
    """
    lx, lz, lphase, lcoeff = tables
    terms, k = modes.shape
    words = lx.shape[2]
    choices = 1 << k
    # Row (t, c): term t, string choice c (bit l of c picks ladder l's string).
    pick = (np.arange(choices)[:, None] >> np.arange(k)[None, :]) & 1
    x = np.zeros((terms * choices, words), dtype=np.uint64)
    z = np.zeros_like(x)
    phase = np.zeros(terms * choices, dtype=np.int64)
    value = np.repeat(coeffs, choices)
    for step in range(k):
        mode = np.repeat(modes[:, step], choices)
        string = np.tile(pick[:, step], terms)
        dagger = np.repeat(daggers[:, step], choices).astype(np.int64)
        x_step, z_step = lx[mode, string], lz[mode, string]
        x_new, z_new = x ^ x_step, z ^ z_step
        phase_new = _popcount(x_new & z_new)
        exponent = (phase + lphase[mode, string] - phase_new
                    + 2 * _popcount(z & x_step)) % 4
        value = value * lcoeff[mode, string, dagger] * _I_POWERS[exponent]
        x, z, phase = x_new, z_new, phase_new
    return PauliTable(x, z, value, 0)


def map_fermion_terms(terms, n: int, method: str) -> PauliSum:
    r"""Map ``{fermion term: coefficient}`` to a qubit operator on ``n`` qubits.

    ``method`` is a base encoding (``"jordan_wigner"``, ``"parity"`` or
    ``"bravyi_kitaev"``).  The same operator the term-by-term
    :meth:`~mandacaru.core.mapping.Fermion.map_to_qubits` built with one
    Python ``compose`` per ladder operator, now from bit tables: terms are
    batched by length and mapped :data:`MAPPING_BATCH_TERMS` at a time, so a
    :math:`10^7`-term Hamiltonian on 100 spin-orbitals is a few thousand
    vectorized passes rather than hours of string manipulation.  Strings whose
    coefficients sum to at most ``1e-12`` are dropped, as
    :meth:`~mandacaru.core.mapping.PauliSum.simplify` does.
    """
    tables = _ladder_tables(method, int(n))
    by_length: dict[int, list] = {}
    for term, coeff in terms.items():
        by_length.setdefault(len(term), []).append((term, coeff))
    total: dict[str, complex] = {}
    for length, items in sorted(by_length.items()):
        if length == 0:
            label = "I" * n
            for _, coeff in items:
                total[label] = total.get(label, 0j) + complex(coeff)
            continue
        for start in range(0, len(items), MAPPING_BATCH_TERMS):
            batch = items[start:start + MAPPING_BATCH_TERMS]
            modes = np.array([[mode for mode, _ in term] for term, _ in batch],
                             dtype=np.int64)
            daggers = np.array([[dagger for _, dagger in term]
                                for term, _ in batch], dtype=bool)
            coeffs = np.array([complex(c) for _, c in batch], dtype=complex)
            if modes.size and (modes.min() < 0 or modes.max() >= n):
                raise ValueError(f"a fermion mode outside the {n}-mode register")
            table = _map_products(modes, daggers, coeffs, tables)
            table.num_qubits = int(n)
            for label, value in table.to_pauli_sum().terms.items():
                total[label] = total.get(label, 0j) + value
    # PauliSum.simplify's threshold, applied without re-validating the labels.
    return PauliSum._from_valid_terms(
        {label: value for label, value in total.items() if abs(value) > 1e-12},
        int(n))
