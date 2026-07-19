# -*- coding: utf-8 -*-
# file: circuits/adapt_ansatz.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Growable product-of-exponentials ansatz for adaptive VQAs.

:class:`AdaptAnsatz` is the state-vector ansatz that ADAPT-VQE grows one generator
at a time.  It lives in :mod:`carcara.circuits` (alongside :class:`~carcara.circuits.ansatz.UCCSD`)
because it is a circuit primitive, not a driver: it conforms to the same
:class:`~carcara.circuits.base.Ansatz` protocol (``num_parameters``, ``n_qubits``,
``reference_state``, ``state``, ``evolve``) so any driver can consume it.
"""

from __future__ import annotations

import numpy as np

from ..core.mapping import reference_qubit_bits
from .pools import PoolOperator


def _spm_norm(spm) -> float:
    """Largest-magnitude entry of a scipy sparse matrix (0 for the empty matrix)."""
    data = spm.tocoo().data
    return float(np.max(np.abs(data))) if data.size else 0.0


class AdaptAnsatz:
    """A product-of-exponentials ansatz that grows one generator at a time.

    ``|psi(theta)> = prod_k exp(theta_k A_k) |HF>`` applied in append order.

    Two evaluation backends, chosen by ``sparse``:

    * **dense** (default) -- each ``exp(theta_k A_k)`` is applied via the cached
      eigendecomposition of the anti-Hermitian generator ``A_k``;
    * **sparse** -- ``A_k`` is kept as a sparse matrix and, when it satisfies the
      excitation identity ``A^3 = -A`` (true for the fermionic / qubit generators),
      the exponential uses the closed form
      ``exp(theta A) = I + sin(theta) A + (1 - cos(theta)) A^2`` -- two sparse
      matrix-vector products, no ``2^n x 2^n`` dense matrix.  Generators that fail
      the identity fall back to :func:`scipy.sparse.linalg.expm_multiply`.  This is
      what keeps 12+-qubit active spaces tractable.

    Exposes the :class:`~carcara.circuits.base.Ansatz` protocol
    (``num_parameters``, ``n_qubits``, ``state``, ``reference_state``, ``evolve``).
    """

    def __init__(self, n_qubits: int, occupied: tuple[int, ...],
                 mapping: str = "jordan_wigner", sparse: bool = False):
        self.n_qubits = int(n_qubits)
        self.mapping = mapping
        self.occupied = tuple(occupied)
        self.sparse = bool(sparse)
        self._ops: list[PoolOperator] = []
        self._eig: list[tuple[np.ndarray, np.ndarray]] = []   # dense (w, V)
        self._sparse_ops: list[tuple] = []                    # (A, A2, rodrigues)
        self._hf = self._reference_vector()

    def _reference_vector(self) -> np.ndarray:
        # The Hartree-Fock determinant is a computational basis state whose bits
        # depend on the fermion-to-qubit map (occupation for JW, parity sums for
        # parity / Bravyi-Kitaev).
        bits = reference_qubit_bits(self.mapping, self.n_qubits, self.occupied)
        index = 0
        for i, bit in enumerate(bits):
            if bit:
                index |= 1 << (self.n_qubits - 1 - i)         # qubit 0 = MSB
        vec = np.zeros(2 ** self.n_qubits, dtype=complex)
        vec[index] = 1.0
        return vec

    def append(self, op: PoolOperator) -> None:
        """Add a generator to the end of the ansatz."""
        self._ops.append(op)
        if self.sparse:
            A = op.generator.to_sparse_matrix()
            A2 = (A @ A).tocsr()
            # Rodrigues closed form is valid iff A^3 = -A (excitation generators).
            rodrigues = _spm_norm(A @ A2 + A) < 1e-9 * max(1.0, _spm_norm(A))
            self._sparse_ops.append((A, A2, rodrigues))
            return
        a = op.matrix()
        # A anti-Hermitian => (-i A) is Hermitian: -i A = V diag(w) V^dag, so
        # A = i V diag(w) V^dag and exp(theta A) = V diag(exp(i theta w)) V^dag.
        w, V = np.linalg.eigh(-1j * a)
        self._eig.append((w, V))

    @property
    def num_parameters(self) -> int:
        return len(self._ops)

    @property
    def operators(self) -> list[PoolOperator]:
        return list(self._ops)

    def reference_state(self) -> np.ndarray:
        return self._hf.copy()

    def state(self, theta) -> np.ndarray:
        """Prepared state ``prod_k exp(theta_k A_k) |HF>``."""
        return self.evolve(theta, self._hf)

    def evolve(self, theta, references) -> np.ndarray:
        r"""Apply the grown unitary ``prod_k exp(theta_k A_k)`` to reference(s).

        ``references`` is a single state vector (shape ``(2**n,)``) or a stack of
        column vectors (shape ``(2**n, k)``), returned with the matching shape;
        the *same* product of exponentials is applied to every column.  This lets
        the subspace-search solver
        (:class:`~carcara.algorithms.SubspaceADAPTVQE`) send several orthogonal
        references through one shared, adaptively grown unitary.
        """
        theta = np.asarray(theta, dtype=float).ravel()
        if theta.size != self.num_parameters:
            raise ValueError(
                f"expected {self.num_parameters} parameters, got {theta.size}")
        refs = np.asarray(references, dtype=complex)
        single = refs.ndim == 1
        out = refs[:, None].copy() if single else refs.copy()
        if self.sparse:
            for angle, (A, A2, rodrigues) in zip(theta, self._sparse_ops):
                if rodrigues:
                    out = (out + np.sin(angle) * (A @ out)
                           + (1.0 - np.cos(angle)) * (A2 @ out))
                else:
                    from scipy.sparse.linalg import expm_multiply
                    out = expm_multiply(angle * A, out)
        else:
            for angle, (w, V) in zip(theta, self._eig):
                out = V @ (np.exp(1j * angle * w)[:, None] * (V.conj().T @ out))
        return out[:, 0] if single else out
