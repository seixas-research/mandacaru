# -*- coding: utf-8 -*-
# file: core/mapping.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Fermionic operators and fermion-to-qubit mappings.

This module provides

* :class:`PauliSum` -- a linear combination of Pauli strings (the qubit-operator
  input and output type): a validated ``{label: coefficient}`` view whose
  products, commutators and matrices are computed by the native
  ``PauliOperator`` (``mandacaru-core::quantum::algebra``);
* :class:`Fermion` -- second-quantized fermionic operators with the builder
  algebra and the molecular Hamiltonian in physicists' notation,

  .. math::

      H = \sum_{pq} h_{pq}\, a^\dagger_p a_q
        + \tfrac12 \sum_{pqrs} \langle pq|rs\rangle\, a^\dagger_p a^\dagger_q a_s a_r ,

  where :math:`\langle pq|rs\rangle = \iint \phi_p^*(1)\phi_q^*(2)\,
  r_{12}^{-1}\,\phi_r(1)\phi_s(2)` is the two-electron integral in physicists'
  notation; the terms of the integral arrays and the mapping to qubits are
  native too;
* three fermion-to-qubit encodings -- **Jordan-Wigner** (default), **parity**
  and **Bravyi-Kitaev** -- plus the reduced parity variant
  ``"parity_reduced"``.

An encoding is an invertible binary matrix :math:`\beta` acting on the
occupation vector, :math:`q = \beta f \pmod 2`; the native ``Encoding`` builds
the Pauli image of every ladder operator from it, so each scheme is one
construction: Jordan-Wigner :math:`\beta = I`, parity the lower-triangular
matrix of ones, Bravyi-Kitaev the Fenwick-tree matrix.  The registers the
native operator holds are at most 62 qubits wide.

Qubit ``k`` of a label is character ``k`` and the most significant bit of a
basis-state index.  The results of the native algebra are simplified: strings
whose coefficient is at most :data:`SIMPLIFY_ATOL` in magnitude are dropped.
"""

from __future__ import annotations

import cmath
from functools import lru_cache

import numpy as np

from .. import _native

#: Magnitude at or below which the algebra drops a Pauli string.
SIMPLIFY_ATOL = 1e-12


def _out(terms) -> list:
    """``{key: coefficient}`` as the native ``(key, re, im)`` triples."""
    return [(key, c.real, c.imag) for key, c in terms.items()]


def _in(triples, key=lambda k: k) -> dict:
    """Native ``(key, re, im)`` triples as an insertion-ordered ``{key: coefficient}``."""
    return {key(k): complex(re, im) for k, re, im in triples}


@lru_cache(maxsize=1)
def _context():
    """The native context of the operator algebra: one thread, shared by every
    ``PauliSum`` and ``Fermion`` (they belong to no calculation)."""
    return _native.Context(threads=1)


class PauliSum:
    """A linear combination of Pauli strings on a fixed number of qubits.

    Terms are stored as a mapping from an ``N``-character string over
    ``{I, X, Y, Z}`` (character ``k`` acts on qubit ``k``) to a complex
    coefficient.
    """

    def __init__(self, terms: dict[str, complex] | None = None,
                 num_qubits: int | None = None):
        """``terms`` maps Pauli strings to coefficients.

        ``num_qubits`` records the register width explicitly.  It matters for an
        operator with no terms -- simplifying a zero operator would otherwise
        lose the register it acts on -- and it is checked against the labels.
        """
        self.terms: dict[str, complex] = {}
        self._num_qubits = None if num_qubits is None else int(num_qubits)
        if terms:
            n = len(next(iter(terms)))
            if self._num_qubits is not None and n != self._num_qubits:
                raise ValueError(
                    f"Pauli strings of length {n} on a "
                    f"{self._num_qubits}-qubit register")
            for label, coeff in terms.items():
                if len(label) != n:
                    raise ValueError("all Pauli strings must have equal length")
                if label.strip("IXYZ"):
                    raise ValueError(
                        f"invalid Pauli string {label!r}: expected characters "
                        f"from 'IXYZ'")
                value = complex(coeff)
                if not cmath.isfinite(value):
                    raise ValueError(
                        f"non-finite coefficient {coeff!r} for term {label!r}")
                self.terms[label] = value
            self._num_qubits = n

    @property
    def num_qubits(self) -> int:
        if self._num_qubits is not None:
            return self._num_qubits
        return len(next(iter(self.terms))) if self.terms else 0

    @classmethod
    def identity(cls, n: int) -> "PauliSum":
        return cls({"I" * n: 1.0}, num_qubits=n)

    @classmethod
    def _from_valid_terms(cls, terms: dict[str, complex],
                          num_qubits: int) -> "PauliSum":
        """Wrap labels that are valid by construction, skipping the checks.

        For operators that come out of the native algebra: every label has
        ``num_qubits`` letters from ``IXYZ`` and every coefficient is a finite
        complex, so re-validating them character by character would only cost
        time.
        """
        out = cls(num_qubits=num_qubits)
        out.terms = terms
        return out

    @classmethod
    def _from_native(cls, operator) -> "PauliSum":
        """The simplified :class:`PauliSum` of a native ``PauliOperator``."""
        return cls._from_valid_terms(
            _in(_native.pauli_term_drop(operator.terms, SIMPLIFY_ATOL)), operator.num_qubits)

    def _native_operator(self, width: int | None = None):
        """This operator as a native ``PauliOperator`` on ``width`` qubits."""
        width = self.num_qubits if width is None else width
        return _context().pauli_operator(
            width, [(label, c.real, c.imag) for label, c in self.terms.items()])

    def _shared_width(self, other: "PauliSum", what: str) -> int | None:
        """The common register width of two operands, or ``None`` if unknown."""
        mine, theirs = self.num_qubits, other.num_qubits
        if mine and theirs and mine != theirs:
            raise ValueError(
                f"cannot {what} a {mine}-qubit and a {theirs}-qubit operator")
        return mine or theirs or None

    def __add__(self, other: "PauliSum") -> "PauliSum":
        width = self._shared_width(other, "add")
        return PauliSum._from_valid_terms(
            _in(_native.pauli_term_sum(_out(self.terms) + _out(other.terms))), num_qubits=width)

    def __mul__(self, scalar: complex) -> "PauliSum":
        value = complex(scalar)
        if not cmath.isfinite(value):
            raise ValueError(f"non-finite scalar {scalar!r}")
        return PauliSum._from_valid_terms(
            _in(_native.pauli_term_scale(_out(self.terms), value.real, value.imag)),
            num_qubits=self._num_qubits)

    __rmul__ = __mul__

    def compose(self, other: "PauliSum") -> "PauliSum":
        """Operator product ``self * other`` (matrix multiplication order).

        Both operands must act on the same register.  The product is formed by
        the native algebra and simplified.
        """
        width = self._shared_width(other, "compose")
        if width is None:
            return PauliSum()
        return PauliSum._from_native(self._native_operator(width).compose(
            other._native_operator(width)))

    def commutator(self, other: "PauliSum") -> "PauliSum":
        """``[self, other]`` from the Pauli algebra, never from matrices.

        Only the pairs of strings that anticommute contribute (each twice
        their product), so it is as cheap on 50 qubits as on 10.  Equal to
        ``self.compose(other) - other.compose(self)``.
        """
        width = self._shared_width(other, "commute")
        if width is None:
            return PauliSum()
        return PauliSum._from_native(self._native_operator(width).commutator(
            other._native_operator(width)))

    def simplify(self, atol: float = 1e-12) -> "PauliSum":
        """Drop terms with negligible coefficient, keeping the register width."""
        cache = getattr(self, "_simplify_cache", None)
        if cache is not None and atol in cache:
            return cache[atol]
        res = PauliSum._from_valid_terms(
            _in(_native.pauli_term_drop(_out(self.terms), float(atol))),
            num_qubits=self.num_qubits or None)
        if cache is None:
            self._simplify_cache = {atol: res}
        else:
            cache[atol] = res
        return res

    def is_hermitian(self, atol: float = 1e-9) -> bool:
        return _native.pauli_term_hermitian(_out(self.terms), float(atol))

    def to_matrix(self) -> np.ndarray:
        """Dense matrix (qubit 0 is the leftmost Kronecker factor)."""
        return self._native_operator().matrix()

    def to_sparse_matrix(self):
        """Sparse CSR matrix (qubit 0 is the leftmost Kronecker factor).

        Built natively by flip mask: the strings sharing the qubits they flip
        share every image state, so the matrix is written straight into its
        compressed rows with memory of the result plus a row of scratch.
        """
        return self._native_operator().sparse_matrix()

    def hermitian_part(self) -> "PauliSum":
        r""":math:`\tfrac12(A + A^\dagger)` as a Pauli sum.

        Every Pauli string is Hermitian, so the Hermitian part of
        :math:`\sum_P c_P P` is :math:`\sum_P \mathrm{Re}(c_P)\,P`: the
        imaginary parts of the coefficients are dropped, with no matrix formed.
        """
        return PauliSum._from_native(self._native_operator().hermitian_part())

    def to_sparse_pauli_op(self):
        """Convert to a qiskit ``SparsePauliOp`` (requires qiskit)."""
        from qiskit.quantum_info import SparsePauliOp
        s = self.simplify()
        labels = list(s.terms)
        coeffs = [s.terms[l] for l in labels]
        return SparsePauliOp(labels, coeffs)

    def __repr__(self) -> str:
        s = self.simplify()
        body = ", ".join(f"{c:.4g}*{l}" for l, c in sorted(s.terms.items()))
        return f"PauliSum({body})"


# --------------------------------------------------------------------------- #
# Fermionic operators.
# --------------------------------------------------------------------------- #

# A term is a tuple of factors, each factor = (mode_index, is_creation).
FermionTerm = tuple[tuple[int, bool], ...]


def _fermion_out(terms) -> list:
    """``{ladder product: coefficient}`` as native ``(ladders, re, im)`` triples."""
    return [(list(term), c.real, c.imag) for term, c in terms.items()]


def _fermion_in(triples) -> dict:
    """Native ``(ladders, re, im)`` triples as ``{ladder product: coefficient}``."""
    return _in(triples, key=lambda ladders: tuple(tuple(f) for f in ladders))



def _integrals(values) -> np.ndarray:
    """``values`` as a C-contiguous float64 or complex128 array, the dtypes the
    native term builders take."""
    values = np.asarray(values)
    target = complex if values.dtype.kind == "c" else float
    return np.ascontiguousarray(values, dtype=target)


def _term_dict(terms) -> dict:
    """``{ladder product: coefficient}`` of native ``(ladders, re, im)`` terms,
    in their order."""
    return {tuple(ladders): complex(re, im) for ladders, re, im in terms}


class Fermion:
    """A linear combination of products of fermionic ladder operators.

    Terms map a product of factors ``((p, dagger), (q, dagger), ...)`` to a
    complex coefficient.  ``dagger`` is ``True`` for a creation operator
    :math:`a^\\dagger_p` and ``False`` for annihilation :math:`a_p`.
    """

    def __init__(self, terms: dict[FermionTerm, complex] | None = None,
                 n_modes: int | None = None):
        self.terms: dict[FermionTerm, complex] = {}
        if terms:
            for term, coeff in terms.items():
                self.terms[term] = complex(coeff)
        self._n_modes = n_modes

    # -- constructors ------------------------------------------------------ #

    @classmethod
    def creation(cls, i: int) -> "Fermion":
        """The creation operator :math:`a^\\dagger_i`."""
        return cls({(((i, True)),): 1.0})

    @classmethod
    def annihilation(cls, i: int) -> "Fermion":
        """The annihilation operator :math:`a_i`."""
        return cls({(((i, False)),): 1.0})

    @classmethod
    def zero(cls) -> "Fermion":
        return cls({})

    @classmethod
    def identity(cls) -> "Fermion":
        return cls({(): 1.0})

    # -- algebra ----------------------------------------------------------- #

    def __add__(self, other: "Fermion") -> "Fermion":
        out = Fermion(n_modes=self._merge_modes(other))
        out.terms = _fermion_in(_native.fermion_term_sum(
            _fermion_out(self.terms) + _fermion_out(other.terms)))
        return out

    def __sub__(self, other: "Fermion") -> "Fermion":
        return self + (other * -1.0)

    def __mul__(self, other) -> "Fermion":
        if isinstance(other, Fermion):
            out = Fermion(n_modes=self._merge_modes(other))
            out.terms = _fermion_in(_native.fermion_term_product(
                _fermion_out(self.terms), _fermion_out(other.terms)))
            return out
        value = complex(other)
        out = Fermion(n_modes=self._n_modes)
        out.terms = _fermion_in(_native.fermion_term_scale(
            _fermion_out(self.terms), value.real, value.imag))
        return out

    __rmul__ = __mul__

    def _merge_modes(self, other: "Fermion") -> int | None:
        if self._n_modes is None:
            return other._n_modes
        if other._n_modes is None:
            return self._n_modes
        return max(self._n_modes, other._n_modes)

    def dagger(self) -> "Fermion":
        """Hermitian conjugate (reverse each product and flip dagger flags)."""
        out = Fermion(n_modes=self._n_modes)
        out.terms = _fermion_in(_native.fermion_term_dagger(_fermion_out(self.terms)))
        return out

    def n_modes(self) -> int:
        """Number of fermionic modes (explicit, or inferred from indices)."""
        if self._n_modes is not None:
            return self._n_modes
        return _native.fermion_term_modes(_fermion_out(self.terms))

    # -- Hamiltonian from integrals --------------------------------------- #

    @classmethod
    def from_integrals(cls, h_pq: np.ndarray,
                       g_pqrs: np.ndarray | None = None,
                       tol: float = 1e-12) -> "Fermion":
        r"""Build ``H = sum h_pq a+_p a_q + 1/2 sum <pq|rs> a+_p a+_q a_s a_r``.

        ``h_pq`` is the ``(M, M)`` one-body tensor and ``g_pqrs`` the optional
        ``(M, M, M, M)`` two-electron integral in **physicists' notation**
        ``<pq|rs>`` over the *same* ``M`` spin-orbitals.  The operator ordering
        ``a+_p a+_q a_s a_r`` (note the ``s`` before ``r``) and the two-body
        factor ``1/2`` are the standard physicists'-notation convention -- see
        the class docstring.
        """
        h_pq = _integrals(h_pq)
        g = None if g_pqrs is None else _integrals(g_pqrs)
        terms = _context().fermion_integral_terms(h_pq, g, tol=tol)
        return cls(_term_dict(terms), n_modes=h_pq.shape[0])

    @classmethod
    def from_spatial_integrals(cls, h: np.ndarray, eri: np.ndarray | None = None,
                               h_spin: np.ndarray | None = None,
                               tol: float = 1e-12) -> "Fermion":
        r"""The spin-orbital Hamiltonian of spatial integrals ``(h, <pq|rs>)``.

        Term by term and in the same order, the operator
        ``from_integrals(*spin_block_integrals(h, eri, h_spin))`` returns
        (:func:`~mandacaru.core.hamiltonian.spin_block_integrals`): spin-orbital
        ``P = p + sigma M``, the one-body term diagonal in spin plus the
        spin-orbital ``h_spin``, and :math:`\langle PQ|RS\rangle =
        \langle pq|rs\rangle` when ``spin(P) == spin(R)`` and
        ``spin(Q) == spin(S)``.  The ``(2M)^4`` spin-orbital tensor is never
        formed: the nonzero entries of each spatial slab ``eri[p]`` are found
        once and written out for the four spin blocks they populate.

        Spin-resolved integrals (a spin-polarized model, one set of spatial
        functions per spin) are ``h`` of shape ``(2, M, M)`` and ``eri`` of
        shape ``(2, 2, M, M, M, M)``: ``eri[sigma, tau]`` is the block whose
        electron 1 has spin ``sigma`` and electron 2 spin ``tau``, the same
        terms in the same order as the spin-orbital tensor holding those
        blocks.
        """
        n_modes, terms = _context().fermion_spatial_terms(
            _integrals(h), None if eri is None else _integrals(eri),
            None if h_spin is None else _integrals(h_spin), tol=tol)
        return cls(_term_dict(terms), n_modes=n_modes)

    # -- native operators --------------------------------------------------- #

    def _native_terms(self) -> list:
        """The terms as native ``(ladders, re, im)`` triples."""
        return [(list(term), c.real, c.imag) for term, c in self.terms.items()]

    def _constant(self) -> complex:
        """The coefficient of the empty product."""
        return self.terms.get((), 0j)

    def _check_modes(self, n: int) -> None:
        """Refuse a term that acts on a mode outside the ``n``-mode register."""
        for term in self.terms:
            for mode, _dagger in term:
                if not 0 <= mode < n:
                    raise ValueError(
                        f"a fermion mode outside the {n}-mode register")

    def map_to_qubits(self, method: str = "jordan_wigner",
                      n_modes: int | None = None,
                      num_particles: tuple[int, int] | None = None) -> PauliSum:
        """Map to a qubit :class:`PauliSum` using the requested encoding.

        Parameters
        ----------
        method : {"jordan_wigner", "parity", "parity_reduced", "bravyi_kitaev"}
            Fermion-to-qubit mapping.  Default is Jordan-Wigner.
        n_modes : int, optional
            Number of modes/qubits; defaults to the operator's mode count.
        num_particles : (int, int)
            ``(n_alpha, n_beta)`` required by ``"parity_reduced"``, assuming
            a spin-blocked ordering (first half alpha, second half beta).
            Every term must then conserve the alpha and total particle
            parities, whose two qubits the reduction removes.
        """
        method = resolve_mapping(method)
        reduced = method == "parity_reduced"
        if reduced and num_particles is None:
            raise ValueError("parity_reduced requires num_particles")
        n = n_modes if n_modes is not None else self.n_modes()
        if n == 0:
            self._check_modes(0)
            return PauliSum({"": self._constant()}, num_qubits=0).simplify()
        particles = [int(v) for v in num_particles] if reduced else None
        return PauliSum._from_native(_context().fermion_operator(
            n, particles, self._native_terms(), mapping=method))

    # -- reference matrix (Fock space) ------------------------------------- #

    def to_matrix(self, n_modes: int | None = None) -> np.ndarray:
        """Dense matrix in the ``2^N`` occupation basis (Jordan-Wigner)."""
        n = n_modes if n_modes is not None else self.n_modes()
        if n == 0:
            self._check_modes(0)
            return np.array([[self._constant()]], dtype=complex)
        return _context().fermion_operator(
            n, None, self._native_terms(), mapping="jordan_wigner").matrix()

    def __repr__(self) -> str:
        return f"Fermion({len(self.terms)} terms, n_modes={self.n_modes()})"


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #

#: The accepted mapping names, in the order they are documented.
MAPPINGS = ("jordan_wigner", "parity", "parity_reduced", "bravyi_kitaev")


def resolve_mapping(method: str) -> str:
    """Validate and return one of the four accepted mapping names."""
    if method not in MAPPINGS:
        raise ValueError(f"unknown mapping {method!r}; use one of {MAPPINGS}")
    return method


# Convenience module-level mapping functions -------------------------------- #

def jordan_wigner(operator: Fermion, n_modes: int | None = None) -> PauliSum:
    """Jordan-Wigner map of a :class:`Fermion` operator."""
    return operator.map_to_qubits("jordan_wigner", n_modes=n_modes)


def parity(operator: Fermion, n_modes: int | None = None) -> PauliSum:
    """Parity map of a :class:`Fermion` operator."""
    return operator.map_to_qubits("parity", n_modes=n_modes)


def parity_reduced(operator: Fermion, num_particles: tuple[int, int],
                   n_modes: int | None = None) -> PauliSum:
    """Parity map with its two particle-parity qubits tapered."""
    return operator.map_to_qubits("parity_reduced", n_modes=n_modes,
                                  num_particles=num_particles)


def bravyi_kitaev(operator: Fermion, n_modes: int | None = None) -> PauliSum:
    """Bravyi-Kitaev map of a :class:`Fermion` operator."""
    return operator.map_to_qubits("bravyi_kitaev", n_modes=n_modes)
