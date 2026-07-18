# -*- coding: utf-8 -*-
# file: core/mapping.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Fermionic operators and fermion-to-qubit mappings.

This module provides

* :class:`PauliSum` -- a lightweight linear combination of Pauli strings (the
  qubit-operator output type), with the algebra needed to compose mappings;
* :class:`Fermion` -- second-quantized fermionic operators with full operator
  algebra and a builder for the molecular Hamiltonian in physicists' notation,

  .. math::

      H = \sum_{pq} h_{pq}\, a^\dagger_p a_q
        + \tfrac12 \sum_{pqrs} \langle pq|rs\rangle\, a^\dagger_p a^\dagger_q a_s a_r ,

  where :math:`\langle pq|rs\rangle = \iint \phi_p^*(1)\phi_q^*(2)\,
  r_{12}^{-1}\,\phi_r(1)\phi_s(2)` is the two-electron integral in physicists'
  notation;

* three fermion-to-qubit mappings -- **Jordan-Wigner** (default), **parity**
  (with optional two-qubit reduction) and **Bravyi-Kitaev**.

All three mappings share one construction.  A mapping is fixed by an invertible
binary *encoding matrix* :math:`\beta` acting on the occupation vector,
:math:`q = \beta f \pmod 2`:

* Jordan-Wigner: :math:`\beta = I`;
* parity: :math:`\beta` lower-triangular of ones (:math:`q_i=\sum_{k\le i} f_k`);
* Bravyi-Kitaev: the Fenwick-tree matrix (update support ``j -> j|(j+1) -> ...``).

From :math:`\beta` and :math:`\beta^{-1}` one reads the *update* set ``U(j)``
(qubits that flip with mode ``j``), the *parity* set ``P(j)`` (qubits holding the
parity of modes ``< j``) and the *flip* set ``F(j)`` (qubits, besides ``j``,
whose ``Z`` reads ``f_j``).  Each ladder operator is then

.. math::

    a_j = \tfrac12\big(X_{U(j)} X_j Z_{P(j)} + i\, X_{U(j)} Y_j Z_{P(j)\triangle F(j)}\big),

which gives the correct number operator ``n_j = (I - Z_{\{j\}\cup F(j)})/2`` for
any encoding.  Bravyi-Kitaev's Fenwick structure makes ``|U|, |P|, |F|`` all
``O(log N)``, so each ladder operator is an ``O(log N)``-weight Pauli sum.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

# --------------------------------------------------------------------------- #
# Single-qubit Pauli algebra.
# --------------------------------------------------------------------------- #

# (a, b) -> (phase, c) with a*b = phase * c.
_MUL: dict[tuple[str, str], tuple[complex, str]] = {
    ("I", "I"): (1, "I"), ("I", "X"): (1, "X"), ("I", "Y"): (1, "Y"), ("I", "Z"): (1, "Z"),
    ("X", "I"): (1, "X"), ("X", "X"): (1, "I"), ("X", "Y"): (1j, "Z"), ("X", "Z"): (-1j, "Y"),
    ("Y", "I"): (1, "Y"), ("Y", "X"): (-1j, "Z"), ("Y", "Y"): (1, "I"), ("Y", "Z"): (1j, "X"),
    ("Z", "I"): (1, "Z"), ("Z", "X"): (1j, "Y"), ("Z", "Y"): (-1j, "X"), ("Z", "Z"): (1, "I"),
}
_SINGLE = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


class PauliSum:
    """A linear combination of Pauli strings on a fixed number of qubits.

    Terms are stored as a mapping from an ``N``-character string over
    ``{I, X, Y, Z}`` (character ``k`` acts on qubit ``k``) to a complex
    coefficient.
    """

    def __init__(self, terms: dict[str, complex] | None = None):
        self.terms: dict[str, complex] = {}
        if terms:
            n = len(next(iter(terms)))
            for label, coeff in terms.items():
                if len(label) != n:
                    raise ValueError("all Pauli strings must have equal length")
                self.terms[label] = self.terms.get(label, 0j) + complex(coeff)

    @property
    def num_qubits(self) -> int:
        return len(next(iter(self.terms))) if self.terms else 0

    @classmethod
    def identity(cls, n: int) -> "PauliSum":
        return cls({"I" * n: 1.0})

    def __add__(self, other: "PauliSum") -> "PauliSum":
        out = PauliSum(dict(self.terms))
        for label, coeff in other.terms.items():
            out.terms[label] = out.terms.get(label, 0j) + coeff
        return out

    def __mul__(self, scalar: complex) -> "PauliSum":
        return PauliSum({label: coeff * scalar for label, coeff in self.terms.items()})

    __rmul__ = __mul__

    def compose(self, other: "PauliSum") -> "PauliSum":
        """Operator product ``self * other`` (matrix multiplication order)."""
        out = PauliSum()
        for l1, c1 in self.terms.items():
            for l2, c2 in other.terms.items():
                phase = 1 + 0j
                chars = []
                for a, b in zip(l1, l2):
                    p, c = _MUL[(a, b)]
                    phase *= p
                    chars.append(c)
                label = "".join(chars)
                out.terms[label] = out.terms.get(label, 0j) + c1 * c2 * phase
        return out

    def simplify(self, atol: float = 1e-12) -> "PauliSum":
        """Drop terms with negligible coefficient."""
        return PauliSum({l: c for l, c in self.terms.items() if abs(c) > atol})

    def is_hermitian(self, atol: float = 1e-9) -> bool:
        return all(abs(c.imag) < atol for c in self.simplify().terms.values())

    def to_matrix(self) -> np.ndarray:
        """Dense matrix (qubit 0 is the leftmost Kronecker factor)."""
        n = self.num_qubits
        dim = 2 ** n
        mat = np.zeros((dim, dim), dtype=complex)
        for label, coeff in self.terms.items():
            m = np.array([[1.0 + 0j]])
            for ch in label:
                m = np.kron(m, _SINGLE[ch])
            mat += coeff * m
        return mat

    def to_sparse_matrix(self):
        """Sparse CSR matrix (qubit 0 is the leftmost Kronecker factor).

        Uses :mod:`scipy.sparse` Kronecker products, so the memory scales with the
        (small) number of non-zeros of each Pauli string rather than the dense
        ``2^n * 2^n``.  Used by the memory-light path of
        :class:`~carcara.algorithms.adapt_vqe.ADAPTVQE` for larger active spaces.
        """
        import scipy.sparse as sp

        n = self.num_qubits
        dim = 2 ** n
        single = {ch: sp.csr_matrix(M) for ch, M in _SINGLE.items()}
        mat = sp.csr_matrix((dim, dim), dtype=complex)
        for label, coeff in self.terms.items():
            m = sp.identity(1, dtype=complex, format="csr")
            for ch in label:
                m = sp.kron(m, single[ch], format="csr")
            mat = mat + coeff * m
        return mat.tocsr()

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
# Encoding matrices and the update / parity / flip sets.
# --------------------------------------------------------------------------- #

def _gf2_inverse(mat: np.ndarray) -> np.ndarray:
    """Inverse of a binary matrix over GF(2) by Gauss-Jordan elimination."""
    n = mat.shape[0]
    a = (mat % 2).astype(np.int8).copy()
    inv = np.eye(n, dtype=np.int8)
    for col in range(n):
        pivot = next((r for r in range(col, n) if a[r, col]), None)
        if pivot is None:
            raise ValueError("encoding matrix is singular over GF(2)")
        if pivot != col:
            a[[col, pivot]] = a[[pivot, col]]
            inv[[col, pivot]] = inv[[pivot, col]]
        for r in range(n):
            if r != col and a[r, col]:
                a[r] ^= a[col]
                inv[r] ^= inv[col]
    return inv


def _encoding_matrix(method: str, n: int) -> np.ndarray:
    if method == "jordan_wigner":
        return np.eye(n, dtype=np.int8)
    if method == "parity":
        return np.tril(np.ones((n, n), dtype=np.int8))
    if method == "bravyi_kitaev":
        beta = np.zeros((n, n), dtype=np.int8)
        for j in range(n):
            i = j
            while i < n:
                beta[i, j] = 1
                i = i | (i + 1)         # Fenwick-tree update ancestor
        return beta
    raise ValueError(f"unknown mapping method {method!r}")


def _mapping_sets(method: str, n: int):
    """Return per-mode ``(U, P, R)`` sets (R = parity-set XOR flip-set)."""
    beta = _encoding_matrix(method, n)
    inv = _gf2_inverse(beta)
    update, parity, remainder = [], [], []
    for j in range(n):
        U = {i for i in range(n) if beta[i, j] and i != j}
        F = {i for i in range(n) if inv[j, i] and i != j}
        row = np.zeros(n, dtype=np.int8)
        for k in range(j):                     # parity of modes 0..j-1
            row ^= inv[k]
        P = {i for i in range(n) if row[i]}
        R = P ^ F                              # symmetric difference
        update.append(U)
        parity.append(P)
        remainder.append(R)
    return update, parity, remainder


def _ladder_pauli(n: int, j: int, dagger: bool,
                  U: set[int], P: set[int], R: set[int]) -> PauliSum:
    """Pauli representation of ``a_j`` (or ``a_j^dagger``) for the given sets."""
    x_term = ["I"] * n
    for i in U:
        x_term[i] = "X"
    x_term[j] = "X"
    for i in P:
        x_term[i] = "Z"

    y_term = ["I"] * n
    for i in U:
        y_term[i] = "X"
    y_term[j] = "Y"
    for i in R:
        y_term[i] = "Z"

    y_coeff = -0.5j if dagger else 0.5j
    return PauliSum({"".join(x_term): 0.5, "".join(y_term): y_coeff})


# --------------------------------------------------------------------------- #
# Fermionic operators.
# --------------------------------------------------------------------------- #

# A term is a tuple of factors, each factor = (mode_index, is_creation).
FermionTerm = tuple[tuple[int, bool], ...]


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
                self.terms[term] = self.terms.get(term, 0j) + complex(coeff)
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
        out = Fermion(dict(self.terms),
                      n_modes=self._merge_modes(other))
        for term, coeff in other.terms.items():
            out.terms[term] = out.terms.get(term, 0j) + coeff
        return out

    def __sub__(self, other: "Fermion") -> "Fermion":
        return self + (other * -1.0)

    def __mul__(self, other) -> "Fermion":
        if isinstance(other, Fermion):
            out = Fermion(n_modes=self._merge_modes(other))
            for t1, c1 in self.terms.items():
                for t2, c2 in other.terms.items():
                    term = t1 + t2
                    out.terms[term] = out.terms.get(term, 0j) + c1 * c2
            return out
        return Fermion({t: c * other for t, c in self.terms.items()},
                       n_modes=self._n_modes)

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
        for term, coeff in self.terms.items():
            rev = tuple((i, not d) for (i, d) in reversed(term))
            out.terms[rev] = out.terms.get(rev, 0j) + np.conj(coeff)
        return out

    def n_modes(self) -> int:
        """Number of fermionic modes (explicit, or inferred from indices)."""
        if self._n_modes is not None:
            return self._n_modes
        hi = -1
        for term in self.terms:
            for (i, _d) in term:
                hi = max(hi, i)
        return hi + 1

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
        h_pq = np.asarray(h_pq)
        m = h_pq.shape[0]
        terms: dict[FermionTerm, complex] = {}
        for p in range(m):
            for q in range(m):
                c = complex(h_pq[p, q])
                if abs(c) > tol:
                    terms[((p, True), (q, False))] = \
                        terms.get(((p, True), (q, False)), 0j) + c
        if g_pqrs is not None:
            g = np.asarray(g_pqrs)
            for p in range(m):
                for q in range(m):
                    for r in range(m):
                        for s in range(m):
                            c = complex(g[p, q, r, s])
                            if abs(c) > tol:
                                key = ((p, True), (q, True), (s, False), (r, False))
                                terms[key] = terms.get(key, 0j) + 0.5 * c
        return cls(terms, n_modes=m)

    # -- mapping to qubits ------------------------------------------------- #

    def map_to_qubits(self, method: str = "jordan_wigner",
                      n_modes: int | None = None,
                      two_qubit_reduction: bool = False,
                      num_particles: tuple[int, int] | None = None) -> PauliSum:
        """Map to a qubit :class:`PauliSum` using the requested encoding.

        Parameters
        ----------
        method : {"jordan_wigner", "parity", "bravyi_kitaev"}
            Fermion-to-qubit mapping (aliases ``"jw"``, ``"bk"``).  Default is
            Jordan-Wigner.
        n_modes : int, optional
            Number of modes/qubits; defaults to the operator's mode count.
        two_qubit_reduction : bool
            Parity mapping only: taper the two qubits fixed by the particle-number
            :math:`Z_2` symmetries, removing 2 qubits.  Requires ``num_particles``.
        num_particles : (int, int)
            ``(n_alpha, n_beta)`` for the reduction, assuming a spin-blocked
            ordering (first half alpha, second half beta).
        """
        method = _canonical_method(method)
        n = n_modes if n_modes is not None else self.n_modes()
        U, P, R = _mapping_sets(method, n)

        result = PauliSum()
        for term, coeff in self.terms.items():
            op = PauliSum.identity(n) * complex(coeff)
            for (mode, dagger) in term:
                op = op.compose(_ladder_pauli(n, mode, dagger,
                                              U[mode], P[mode], R[mode]))
            result = result + op
        result = result.simplify()

        if two_qubit_reduction:
            if method != "parity":
                raise ValueError("two-qubit reduction requires method='parity'")
            if num_particles is None:
                raise ValueError("two-qubit reduction requires num_particles")
            result = _parity_two_qubit_reduction(result, n, num_particles)
        return result

    # -- reference matrix (Fock space, independent JW oracle) -------------- #

    def to_matrix(self, n_modes: int | None = None) -> np.ndarray:
        """Dense matrix in the ``2^N`` occupation basis (numeric Jordan-Wigner)."""
        n = n_modes if n_modes is not None else self.n_modes()
        annih = _numeric_annihilators(n)
        dim = 2 ** n
        mat = np.zeros((dim, dim), dtype=complex)
        for term, coeff in self.terms.items():
            op = np.eye(dim, dtype=complex)
            for (mode, dagger) in term:
                a = annih[mode]
                op = op @ (a.conj().T if dagger else a)
            mat += coeff * op
        return mat

    def __repr__(self) -> str:
        return f"Fermion({len(self.terms)} terms, n_modes={self.n_modes()})"


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #

_ALIASES = {
    "jw": "jordan_wigner", "jordan_wigner": "jordan_wigner",
    "parity": "parity",
    "bk": "bravyi_kitaev", "bravyi_kitaev": "bravyi_kitaev",
}


def _canonical_method(method: str) -> str:
    try:
        return _ALIASES[method.lower()]
    except KeyError:
        raise ValueError(
            f"unknown mapping {method!r}; use 'jordan_wigner', 'parity' "
            f"or 'bravyi_kitaev'") from None


def reference_qubit_bits(method: str, n_modes: int,
                         occupied) -> np.ndarray:
    r"""Qubit bit-string of a Slater determinant under a fermion-to-qubit map.

    A determinant is the occupation vector ``x`` (``x_j = 1`` iff spin-orbital
    ``j`` is filled).  Under the encoding matrix :math:`\beta` the qubit register
    stores ``q = beta x  (mod 2)``: for Jordan-Wigner ``q = x`` (qubit ``j`` holds
    the occupation of orbital ``j``), but for parity / Bravyi-Kitaev the qubits
    hold parity sums, so the Hartree-Fock reference is a *different* computational
    basis state.  Returns ``q`` as a length-``n_modes`` array with ``q[i]`` the
    state of qubit ``i``.
    """
    x = np.zeros(int(n_modes), dtype=np.int8)
    for j in occupied:
        x[int(j)] = 1
    beta = _encoding_matrix(_canonical_method(method), int(n_modes))
    return (beta @ x) % 2


def _numeric_annihilators(n: int) -> list[np.ndarray]:
    """Fock-space annihilation matrices with Jordan-Wigner Z-strings (qubit 0 first)."""
    lower = np.array([[0, 1], [0, 0]], dtype=complex)   # |1> -> |0>
    z = _SINGLE["Z"]
    eye = _SINGLE["I"]
    ops = []
    for j in range(n):
        factors = [z] * j + [lower] + [eye] * (n - 1 - j)
        m = np.array([[1.0 + 0j]])
        for f in factors:
            m = np.kron(m, f)
        ops.append(m)
    return ops


def _parity_two_qubit_reduction(op: PauliSum, n: int,
                                num_particles: tuple[int, int]) -> PauliSum:
    """Taper the two parity qubits fixed by particle number (spin-blocked)."""
    n_alpha, n_beta = num_particles
    # Qubit n//2 - 1 holds the alpha-sector parity; qubit n-1 the total parity.
    positions = {n // 2 - 1: (-1) ** n_alpha,
                 n - 1: (-1) ** (n_alpha + n_beta)}
    out = PauliSum()
    for label, coeff in op.terms.items():
        c = coeff
        keep = []
        for k, ch in enumerate(label):
            if k in positions:
                if ch in ("X", "Y"):
                    # Term anticommutes with the symmetry: must be absent in a
                    # symmetric H; drop defensively.
                    c = 0
                    break
                if ch == "Z":
                    c *= positions[k]
            else:
                keep.append(ch)
        if c != 0:
            new_label = "".join(keep)
            out.terms[new_label] = out.terms.get(new_label, 0j) + c
    return out.simplify()


# Convenience module-level mapping functions -------------------------------- #

def jordan_wigner(operator: Fermion, n_modes: int | None = None) -> PauliSum:
    """Jordan-Wigner map of a :class:`Fermion` operator."""
    return operator.map_to_qubits("jordan_wigner", n_modes=n_modes)


def parity(operator: Fermion, n_modes: int | None = None,
           two_qubit_reduction: bool = False,
           num_particles: tuple[int, int] | None = None) -> PauliSum:
    """Parity map of a :class:`Fermion` operator (optional 2-qubit reduction)."""
    return operator.map_to_qubits("parity", n_modes=n_modes,
                                  two_qubit_reduction=two_qubit_reduction,
                                  num_particles=num_particles)


def bravyi_kitaev(operator: Fermion, n_modes: int | None = None) -> PauliSum:
    """Bravyi-Kitaev map of a :class:`Fermion` operator."""
    return operator.map_to_qubits("bravyi_kitaev", n_modes=n_modes)
