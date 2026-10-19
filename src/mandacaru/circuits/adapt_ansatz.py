# -*- coding: utf-8 -*-
# file: circuits/adapt_ansatz.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Growable product-of-exponentials ansatz for adaptive VQAs.

:class:`AdaptAnsatz` is the state-vector ansatz that ADAPT-VQE grows one generator
at a time.  It lives in :mod:`mandacaru.circuits` (alongside :class:`~mandacaru.circuits.ansatz.UCCSD`)
because it is a circuit primitive, not a driver: it conforms to the same
:class:`~mandacaru.circuits.base.Ansatz` protocol (``num_parameters``, ``n_qubits``,
``reference_state``, ``state``, ``evolve``) so any driver can consume it.
"""

from __future__ import annotations

import numpy as np

from ..core.mapping import reference_qubit_bits, resolve_mapping
from ..core.matrix_free import MATRIX_FREE_CACHE_BYTES, PauliOperator
from .pools import PoolOperator


def _spm_norm(spm) -> float:
    """Largest-magnitude entry of a scipy sparse matrix (0 for the empty matrix)."""
    data = spm.tocoo().data
    return float(np.max(np.abs(data))) if data.size else 0.0


class AdaptAnsatz:
    """A product-of-exponentials ansatz that grows one generator at a time.

    ``|psi(theta)> = prod_k exp(theta_k A_k) |HF>`` applied in append order.

    Three evaluation backends, chosen by ``sparse`` and ``matrix_free``:

    * **dense** (default) -- each ``exp(theta_k A_k)`` is applied via the cached
      eigendecomposition of the anti-Hermitian generator ``A_k``;
    * **sparse** -- ``A_k`` is kept as a sparse matrix and, when it satisfies the
      excitation identity ``A^3 = -A`` (true for the fermionic / qubit generators),
      the exponential uses the closed form
      ``exp(theta A) = I + sin(theta) A + (1 - cos(theta)) A^2`` -- two sparse
      matrix-vector products, no ``2^n x 2^n`` dense matrix.  Generators that fail
      the identity fall back to :func:`scipy.sparse.linalg.expm_multiply`.  This is
      what keeps 12+-qubit active spaces tractable.
    * **matrix-free** (``matrix_free=True``) -- the same closed form, with ``A``
      and ``A^2`` applied as Pauli strings
      (:class:`~mandacaru.core.matrix_free.PauliOperator`) and ``A^2`` formed in
      the Pauli algebra: no matrix is stored at all, only the state vectors.

    A third, **circuit-executing** backend is selected by passing a ``provider``
    (see :mod:`mandacaru.backends.providers`): the ansatz is then compiled to a real
    quantum circuit -- an ``X``-gate reference preparation followed by the exact
    Pauli-rotation decomposition of each ``exp(theta_k A_k)`` -- and run on that
    SDK's local state-vector simulator (Qiskit, Amazon Braket or Cirq).  The
    decomposition is exact for these generators, so the executed state matches the
    matrix backends to machine precision; it is simply much slower, and is what
    makes the driver's ``backend_provider`` argument meaningful.

    A **measured** ansatz (``simulate=False``) has no state backend at all: it
    is the operator list and the reference determinant a circuit is built
    from, for a shot-based run whose every expectation value is measured on
    the device.  It stores no generator matrix and no ``2^n`` vector, and asking
    it for a state is an error.

    Exposes the :class:`~mandacaru.circuits.base.Ansatz` protocol
    (``num_parameters``, ``n_qubits``, ``state``, ``reference_state``, ``evolve``).
    """

    def __init__(self, n_qubits: int, occupied: tuple[int, ...],
                 mapping: str = "jordan_wigner", sparse: bool = False,
                 provider=None, num_particles=None, sector=None,
                 matrix_free: bool = False, simulate: bool = True):
        self.n_qubits = int(n_qubits)
        #: Whether states are prepared locally; ``False`` for a measured
        #: (shot-based) ansatz, which keeps only what a circuit needs.
        self.simulate = bool(simulate)
        if not self.simulate and (sector is not None or provider is not None):
            raise ValueError("a measured ansatz (simulate=False) prepares no "
                             "state, so it takes no sector and no provider")
        #: Particle-number sector the generators act in (``None``: full register).
        self.sector = sector
        if sector is not None and provider is not None:
            raise ValueError("a circuit provider prepares full-register states; "
                             "it cannot be combined with a particle-number sector")
        self.mapping = resolve_mapping(mapping)
        self.occupied = tuple(occupied)
        # With the parity two-qubit reduction the register is two qubits
        # smaller than the spin-orbital count the occupations refer to.
        self.num_particles = num_particles
        self.n_modes = self.n_qubits + (
            2 if self.mapping == "parity_reduced" else 0)
        #: Apply the generators as Pauli strings
        #: (:class:`~mandacaru.core.matrix_free.PauliOperator`), storing no
        #: matrix; implies :attr:`sparse`.
        self.matrix_free = bool(matrix_free)
        #: What the matrix-free generators may still store between products.
        self._cache_left = MATRIX_FREE_CACHE_BYTES
        self.sparse = bool(sparse) or sector is not None or self.matrix_free
        self.provider = provider
        self._ops: list[PoolOperator] = []
        self._eig: list[tuple[np.ndarray, np.ndarray]] = []   # dense (w, V)
        self._sparse_ops: list[tuple] = []                    # (A, A2, rodrigues)
        # The Hartree-Fock determinant is a computational basis state whose bits
        # depend on the fermion-to-qubit map (occupation for JW, parity sums for
        # parity / Bravyi-Kitaev).
        bits = reference_qubit_bits(self.mapping, self.n_modes, self.occupied)
        index = 0
        for i, bit in enumerate(bits):
            if bit:
                index |= 1 << (self.n_qubits - 1 - i)         # qubit 0 = MSB
        self._reference_index = index
        self._hf = self._reference_vector() if self.simulate else None

    def _reference_vector(self) -> np.ndarray:
        if self.sector is not None:
            return self.sector.basis_vector(self._reference_index)
        vec = np.zeros(2 ** self.n_qubits, dtype=complex)
        vec[self._reference_index] = 1.0
        return vec

    def _require_states(self) -> None:
        if not self.simulate:
            raise RuntimeError(
                "this ansatz is measured on the device (a shot-based run): "
                "it prepares no state vector, and none of 2^"
                f"{self.n_qubits} amplitudes is ever allocated")

    def append(self, op: PoolOperator) -> None:
        """Add a generator to the end of the ansatz."""
        self._ops.append(op)
        if self.provider is not None or not self.simulate:
            # Circuit execution needs only the generator's Pauli terms; skip the
            # (expensive) dense eigendecomposition / sparse powers entirely.
            return
        if self.matrix_free:
            # A^2 is formed in the Pauli algebra, and A^3 = -A is checked there
            # too: no matrix at any point.
            generator = op.generator
            square = generator.compose(generator)
            residual = (generator.compose(square) + generator).simplify(1e-9)
            pair = []
            for operator in (generator, square.simplify()):
                # Every evaluation applies every appended generator, so each
                # stores its CSR block while the ansatz's shared budget lasts
                # and is recomputed on every product after that.
                bound = PauliOperator(operator, sector=self.sector,
                                      cache_bytes=0).stored_bytes_bound()
                keep = bound if bound <= self._cache_left else 0
                self._cache_left -= keep
                pair.append(PauliOperator(operator, sector=self.sector,
                                          cache_bytes=keep))
            self._sparse_ops.append((*pair, not residual.terms))
            return
        if self.sparse:
            A = (self.sector.restrict(op.generator) if self.sector is not None
                 else op.generator.to_sparse_matrix())
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

    def remove(self, index: int) -> PoolOperator:
        """Drop the operator at ``index`` and return it.

        Used by the pruning growth strategy
        (:class:`~mandacaru.algorithms.adapt_vqe.ADAPTVQE`, ``prune=True``),
        which takes an irrelevant operator back out of a grown ansatz.  The
        per-operator caches are index-aligned with ``_ops``, so each is popped
        with it rather than rebuilt -- re-appending the survivors would redo
        every eigendecomposition.
        """
        op = self._ops.pop(index)
        if self.matrix_free:
            # Its stored blocks go back to the shared budget.
            A, A2, _ = self._sparse_ops[index]
            self._cache_left += A._budget + A2._budget
        for cache in (self._sparse_ops, self._eig):
            if cache:
                cache.pop(index)
        return op

    @property
    def num_parameters(self) -> int:
        return len(self._ops)

    @property
    def operators(self) -> list[PoolOperator]:
        return list(self._ops)

    @property
    def pauli_generators(self):
        """The appended generators as qubit operators (for circuit backends)."""
        return [op.generator for op in self._ops]

    #: An adaptively grown ansatz is always an ordered product of exponentials.
    preparation = "product"

    def reference_qubits(self) -> list[int]:
        """Qubit indices set to ``|1>`` in the reference determinant."""
        from ..backends.providers import _occupied_qubits
        return _occupied_qubits(self._reference_index, self.n_qubits)

    def reference_state(self) -> np.ndarray:
        self._require_states()
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
        (:class:`~mandacaru.algorithms.SubspaceADAPTVQE`) send several orthogonal
        references through one shared, adaptively grown unitary.
        """
        self._require_states()
        theta = np.asarray(theta, dtype=float).ravel()
        if theta.size != self.num_parameters:
            raise ValueError(
                f"expected {self.num_parameters} parameters, got {theta.size}")
        refs = np.asarray(references, dtype=complex)
        single = refs.ndim == 1
        out = refs[:, None].copy() if single else refs.copy()
        if self.provider is not None:
            out = self._evolve_on_provider(theta, out)
            return out[:, 0] if single else out
        for k, angle in enumerate(theta):
            out = self._exponential(k, angle, out)
        return out[:, 0] if single else out

    def _exponential(self, k: int, angle: float, out: np.ndarray) -> np.ndarray:
        """``exp(angle A_k) out`` for a stack of column vectors ``out``."""
        if self.sparse:
            A, A2, rodrigues = self._sparse_ops[k]
            if rodrigues:
                return (out + np.sin(angle) * (A @ out)
                        + (1.0 - np.cos(angle)) * (A2 @ out))
            from scipy.sparse.linalg import expm_multiply
            if self.matrix_free:
                return expm_multiply(angle * A.as_linear_operator(), out,
                                     traceA=angle * A.trace())
            return expm_multiply(angle * A, out)
        w, V = self._eig[k]
        return V @ (np.exp(1j * angle * w)[:, None] * (V.conj().T @ out))

    def _generator(self, k: int, out: np.ndarray) -> np.ndarray:
        """``A_k out`` for a stack of column vectors ``out``."""
        if self.sparse:
            return self._sparse_ops[k][0] @ out
        # A = i V diag(w) V^dag (see `append`).
        w, V = self._eig[k]
        return 1j * (V @ (w[:, None] * (V.conj().T @ out)))

    def energy_and_gradient(self, theta, apply_hamiltonian):
        r"""``(E, dE/dtheta)`` of ``E = <psi|H|psi>``, by the adjoint method.

        ``apply_hamiltonian(v)`` returns ``H v`` for a stack of column vectors.
        One forward pass prepares :math:`\psi = U_N \cdots U_1 |HF\rangle` and
        :math:`\lambda = H\psi`; one backward sweep then undoes
        :math:`U_k = e^{\theta_k A_k}` on both, reading off

        .. math::

            \frac{\partial E}{\partial\theta_k}
            = 2\,\mathrm{Re}\,\langle\lambda_k| A_k |\psi_k\rangle,
            \qquad \psi_k = U_k\cdots U_1|HF\rangle,\;
            \lambda_k = U_{k+1}^\dagger\cdots U_N^\dagger H\psi,

        since :math:`U_k^\dagger = e^{-\theta_k A_k}` for an anti-Hermitian
        generator.  The whole gradient costs about three state preparations
        and one product with :math:`H`, against :math:`N + 1` energies for a
        finite difference.  ``None`` on a circuit provider, which returns no
        amplitudes to sweep back, and on a measured ansatz.
        """
        if self.provider is not None or not self.simulate:
            return None
        theta = np.asarray(theta, dtype=float).ravel()
        psi = self.evolve(theta, self._hf[:, None])
        lam = apply_hamiltonian(psi)
        energy = float(np.real(np.vdot(psi, lam)))
        gradient = np.empty(theta.size)
        for k in range(theta.size - 1, -1, -1):
            gradient[k] = 2.0 * float(np.real(
                np.vdot(lam, self._generator(k, psi))))
            psi = self._exponential(k, -theta[k], psi)
            lam = self._exponential(k, -theta[k], lam)
        return energy, gradient

    def _evolve_on_provider(self, theta, refs: np.ndarray) -> np.ndarray:
        """Run the ansatz circuit once per reference column on the SDK simulator.

        A circuit can only be *initialized* in a computational basis state, so
        every column must be a Slater determinant -- which is exactly what the
        Hartree-Fock reference and the SSVQE reference determinants are.
        """
        from ..backends.providers import evolve_determinants

        return evolve_determinants(self.provider, self.n_qubits,
                                   [op.generator for op in self._ops], theta,
                                   refs)
