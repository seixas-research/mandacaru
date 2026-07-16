# -*- coding: utf-8 -*-
# file: algorithms/adapt_vqe.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""ADAPT-VQE: adaptively grown variational ansatz.

:class:`AdaptVQE` implements ADAPT-VQE (Grimsley *et al.*, 2019), which builds a
compact, problem-tailored ansatz one operator at a time instead of using a fixed
template.  Each macro-iteration:

1. evaluate the energy gradient of appending each pool operator :math:`A_i` at
   zero angle,
   :math:`g_i = \partial E/\partial\theta_i = \langle\psi(\vec\theta)|[H, A_i]
   |\psi(\vec\theta)\rangle = 2\,\mathrm{Re}\,\langle H\psi|A_i\psi\rangle`;
2. stop if :math:`\max_i |g_i| < \varepsilon`;
3. append :math:`e^{\theta_k A_{\mathrm{opt}}}` for the largest-gradient operator,
   initializing :math:`\theta_k = 0`;
4. re-optimize **all** parameters with the classical optimizer (warm-started from
   the previous optimum -- ADAPT's key efficiency property).

This is an exact state-vector implementation in the same spirit as
:class:`~carcara.algorithms.vqe.VQE`: the Hamiltonian and each generator are
materialized as dense matrices and the ansatz produces the exact :math:`2^N`
state vector, so gradients and energies are noiseless.

Each grown ansatz is also **profiled** for hardware cost: the parameterized
unitary is compiled to a native ``{CNOT, U}`` gate set with Qiskit and its CNOT
count and circuit depth are logged per iteration (see :class:`CircuitMetrics`).
The four operator pools (:mod:`carcara.circuits.pools`) can then be compared on
accuracy-per-CNOT.  Profiling is optional -- if Qiskit is unavailable the run
proceeds and metrics are reported as ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..circuits.pools import PoolBase, PoolOperator, build_pool
from ..core.mapping import Fermion, PauliSum
from ..optimizers.optim import Optimizer


# --------------------------------------------------------------------------- #
# Circuit profiling.
# --------------------------------------------------------------------------- #

@dataclass
class CircuitMetrics:
    """Structural cost of a compiled ansatz circuit."""

    cnot_count: int | None            # number of CNOT (two-qubit) gates
    depth: int | None                 # circuit depth in the native gate set
    num_operators: int                # generators (parameters) in the ansatz

    def __repr__(self) -> str:
        return (f"CircuitMetrics(cnots={self.cnot_count}, depth={self.depth}, "
                f"n_ops={self.num_operators})")


def _pauli_evolution_gate(generator: PauliSum, time=1.0):
    """Qiskit ``PauliEvolutionGate`` realizing ``exp(theta * A)`` for anti-Herm ``A``.

    With ``A`` anti-Hermitian, ``G = i A`` is Hermitian with real coefficients and
    ``exp(-i * time * G) = exp(time * A)``.  Pauli strings are reversed so Qiskit's
    little-endian qubit order matches Carcará's (qubit 0 = leftmost); this does not
    affect the CNOT/depth counts, which are relabeling-invariant.
    """
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp

    labels, coeffs = [], []
    for label, coeff in generator.simplify().terms.items():
        herm = 1j * coeff                     # coefficient of G = i A (real)
        labels.append(label[::-1])            # leftmost -> qubit 0 (little-endian)
        coeffs.append(complex(herm).real)
    if not labels:
        return None
    op = SparsePauliOp(labels, np.asarray(coeffs, dtype=float))
    return PauliEvolutionGate(op, time=time)


def profile_ansatz(n_qubits: int, occupied: tuple[int, ...],
                   operators: list[PoolOperator]) -> CircuitMetrics:
    """Compile ``|HF>`` + product of ``exp(A_k)`` to ``{cx, u}`` and count cost.

    Returns a :class:`CircuitMetrics`; ``cnot_count`` / ``depth`` are ``None`` if
    Qiskit is not installed.
    """
    try:
        from qiskit import QuantumCircuit, transpile
    except Exception:
        return CircuitMetrics(None, None, len(operators))

    qc = QuantumCircuit(n_qubits)
    for q in occupied:                        # Hartree-Fock reference preparation
        qc.x(q)
    for op in operators:
        gate = _pauli_evolution_gate(op.generator)
        if gate is not None:
            qc.append(gate, range(n_qubits))

    compiled = transpile(qc, basis_gates=["cx", "u"], optimization_level=1)
    counts = compiled.count_ops()
    return CircuitMetrics(cnot_count=int(counts.get("cx", 0)),
                          depth=int(compiled.depth()),
                          num_operators=len(operators))


# --------------------------------------------------------------------------- #
# Growable state-vector ansatz.
# --------------------------------------------------------------------------- #

class AdaptAnsatz:
    """A product-of-exponentials ansatz that grows one generator at a time.

    ``|psi(theta)> = prod_k exp(theta_k A_k) |HF>`` applied in append order, with
    each ``exp(theta_k A_k)`` evaluated exactly via the eigendecomposition of the
    anti-Hermitian generator ``A_k`` (cached, so cost evaluations are cheap).

    Exposes the interface expected by :class:`~carcara.algorithms.vqe.VQE`
    (``num_parameters``, ``n_qubits``, ``state``, ``reference_state``).
    """

    def __init__(self, n_qubits: int, occupied: tuple[int, ...],
                 mapping: str = "jordan_wigner"):
        self.n_qubits = int(n_qubits)
        self.mapping = mapping
        self.occupied = tuple(occupied)
        self._ops: list[PoolOperator] = []
        self._eig: list[tuple[np.ndarray, np.ndarray]] = []   # (w, V) per generator
        self._hf = self._reference_vector()

    def _reference_vector(self) -> np.ndarray:
        index = 0
        for j in self.occupied:
            index |= 1 << (self.n_qubits - 1 - j)             # qubit 0 = MSB
        vec = np.zeros(2 ** self.n_qubits, dtype=complex)
        vec[index] = 1.0
        return vec

    def append(self, op: PoolOperator) -> None:
        """Add a generator to the end of the ansatz."""
        a = op.matrix()
        # A anti-Hermitian => (-i A) is Hermitian: -i A = V diag(w) V^dag, so
        # A = i V diag(w) V^dag and exp(theta A) = V diag(exp(i theta w)) V^dag.
        w, V = np.linalg.eigh(-1j * a)
        self._ops.append(op)
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
        theta = np.asarray(theta, dtype=float).ravel()
        if theta.size != self.num_parameters:
            raise ValueError(
                f"expected {self.num_parameters} parameters, got {theta.size}")
        psi = self._hf.copy()
        for angle, (w, V) in zip(theta, self._eig):
            psi = V @ (np.exp(1j * angle * w) * (V.conj().T @ psi))
        return psi


# --------------------------------------------------------------------------- #
# Result container.
# --------------------------------------------------------------------------- #

@dataclass
class AdaptIteration:
    """Record of one ADAPT-VQE macro-iteration."""

    operator_label: str
    operator_kind: str
    max_gradient: float
    energy: float
    cnot_count: int | None
    depth: int | None
    num_parameters: int


@dataclass
class AdaptVQEResult:
    """Result of an :class:`AdaptVQE` run."""

    optimal_energy: float
    optimal_parameters: np.ndarray
    reference_energy: float
    converged: bool
    final_max_gradient: float
    operators: list[str]                      # selected operator labels, in order
    iterations: list[AdaptIteration] = field(default_factory=list)
    num_evaluations: int = 0                  # total inner cost evaluations
    metrics: CircuitMetrics | None = None     # final compiled-circuit metrics

    @property
    def num_operators(self) -> int:
        return len(self.operators)

    @property
    def energy_history(self) -> list[float]:
        return [it.energy for it in self.iterations]

    @property
    def gradient_history(self) -> list[float]:
        return [it.max_gradient for it in self.iterations]

    def __repr__(self) -> str:
        cnots = self.metrics.cnot_count if self.metrics else None
        return (f"AdaptVQEResult(energy={self.optimal_energy:.6f}, "
                f"n_ops={self.num_operators}, cnots={cnots}, "
                f"converged={self.converged})")


# --------------------------------------------------------------------------- #
# ADAPT-VQE driver.
# --------------------------------------------------------------------------- #

class AdaptVQE:
    """Adaptive VQE on an exact state-vector backend.

    Parameters
    ----------
    hamiltonian : PauliSum or Fermion
        Qubit Hamiltonian, or a fermionic Hamiltonian mapped with ``mapping``.
    pool : PoolBase or str
        The operator pool, or a name for :func:`~carcara.circuits.pools.build_pool`
        (``"fermionic"``, ``"qubit"``, ``"qeb"``, ``"ceo"``).  When a name is given
        ``n_spatial_orbitals`` and ``num_particles`` are required.
    num_particles : (int, int), optional
        ``(n_alpha, n_beta)``; required to build a pool from a name and to set the
        Hartree-Fock reference.  Inferred from the pool object otherwise.
    n_spatial_orbitals : int, optional
        Number of spatial orbitals; required to build a pool from a name.
    optimizer : Optimizer, optional
        Classical optimizer for the inner re-optimization (default L-BFGS-B).
    mapping : str
        Fermion-to-qubit mapping used when ``hamiltonian`` is a ``Fermion`` and to
        build a named fermionic pool (default ``"jordan_wigner"``).
    profile : bool
        Compile and profile the ansatz each iteration (default ``True``).
    """

    def __init__(self, hamiltonian, pool, num_particles=None,
                 n_spatial_orbitals=None, optimizer: Optimizer | None = None,
                 mapping: str = "jordan_wigner", profile: bool = True):
        self.mapping = mapping
        self.profile = profile
        self.optimizer = optimizer or Optimizer(method="L-BFGS-B", maxiter=2000)

        # Resolve the pool.
        if isinstance(pool, PoolBase):
            self.pool = pool
        else:
            if n_spatial_orbitals is None or num_particles is None:
                raise ValueError(
                    "building a pool by name requires n_spatial_orbitals and "
                    "num_particles")
            self.pool = build_pool(pool, n_spatial_orbitals, num_particles,
                                   mapping=mapping)
        self.n_qubits = self.pool.n_qubits
        self.num_particles = (tuple(num_particles) if num_particles is not None
                              else self.pool.num_particles)

        # Materialize the qubit Hamiltonian.
        qubit_h = self._as_pauli_sum(hamiltonian)
        if qubit_h.num_qubits != self.n_qubits:
            raise ValueError(
                f"Hamiltonian acts on {qubit_h.num_qubits} qubits but the pool "
                f"has {self.n_qubits}")
        self.hamiltonian = qubit_h
        h = qubit_h.to_matrix()
        self._h_matrix = 0.5 * (h + h.conj().T)      # Hermitize away rounding

        # Precompute pool-operator matrices once.
        self._pool_ops = self.pool.operators()
        self._pool_matrices = [op.matrix() for op in self._pool_ops]

    # -- setup helpers ---------------------------------------------------- #

    def _as_pauli_sum(self, hamiltonian) -> PauliSum:
        if isinstance(hamiltonian, PauliSum):
            return hamiltonian
        if isinstance(hamiltonian, Fermion):
            return hamiltonian.map_to_qubits(self.mapping, n_modes=self.n_qubits)
        raise TypeError("hamiltonian must be a PauliSum or Fermion")

    # -- energy / gradient ------------------------------------------------ #

    def energy(self, psi: np.ndarray) -> float:
        return float(np.real(np.vdot(psi, self._h_matrix @ psi)))

    def _gradients(self, psi: np.ndarray) -> np.ndarray:
        r"""Pool gradients ``g_i = 2 Re<H psi | A_i psi>`` at the current state."""
        h_psi = self._h_matrix @ psi
        grads = np.empty(len(self._pool_matrices))
        for i, a in enumerate(self._pool_matrices):
            grads[i] = 2.0 * np.real(np.vdot(h_psi, a @ psi))
        return grads

    def reference_energy(self) -> float:
        ansatz = AdaptAnsatz(self.n_qubits, self.pool.occupied_orbitals,
                             self.mapping)
        return self.energy(ansatz.reference_state())

    # -- main loop -------------------------------------------------------- #

    def run(self, max_iterations: int = 50, gradient_tol: float = 1e-3,
            initial_parameters=None) -> AdaptVQEResult:
        """Grow and optimize the ansatz until convergence.

        Parameters
        ----------
        max_iterations : int
            Maximum number of operators to append (default ``50``).
        gradient_tol : float
            Stop when the largest pool gradient falls below this threshold
            (default ``1e-3``).
        initial_parameters : array_like, optional
            Warm-start parameters for an already-grown ansatz (rarely needed).
        """
        ansatz = AdaptAnsatz(self.n_qubits, self.pool.occupied_orbitals,
                             self.mapping)
        params = (np.asarray(initial_parameters, dtype=float).ravel()
                  if initial_parameters is not None else np.zeros(0))
        ref_energy = self.energy(ansatz.reference_state())

        iterations: list[AdaptIteration] = []
        selected: list[str] = []
        total_evals = 0
        converged = False
        max_grad = np.inf
        energy = ref_energy
        metrics: CircuitMetrics | None = None

        for _ in range(max_iterations):
            psi = ansatz.state(params) if ansatz.num_parameters else \
                ansatz.reference_state()
            grads = self._gradients(psi)
            idx = int(np.argmax(np.abs(grads)))
            max_grad = float(abs(grads[idx]))
            if max_grad < gradient_tol:
                converged = True
                break

            op = self._pool_ops[idx]
            ansatz.append(op)
            selected.append(op.label)

            # Warm start: reuse previous optimum, new parameter initialized to 0.
            x0 = np.concatenate([params, [0.0]])
            result = self.optimizer.minimize(
                lambda t: self.energy(ansatz.state(t)), x0)
            params = np.asarray(result.x, dtype=float)
            energy = float(result.fun)
            total_evals += result.nfev

            metrics = (profile_ansatz(self.n_qubits, ansatz.occupied,
                                      ansatz.operators)
                       if self.profile else
                       CircuitMetrics(None, None, ansatz.num_parameters))
            iterations.append(AdaptIteration(
                operator_label=op.label, operator_kind=op.kind,
                max_gradient=max_grad, energy=energy,
                cnot_count=metrics.cnot_count, depth=metrics.depth,
                num_parameters=ansatz.num_parameters))
        else:
            # Loop exhausted without meeting the gradient threshold; report the
            # final screening gradient so callers can see how close it got.
            psi = ansatz.state(params) if ansatz.num_parameters else \
                ansatz.reference_state()
            max_grad = float(np.max(np.abs(self._gradients(psi))))

        return AdaptVQEResult(
            optimal_energy=energy,
            optimal_parameters=params,
            reference_energy=ref_energy,
            converged=converged,
            final_max_gradient=max_grad,
            operators=selected,
            iterations=iterations,
            num_evaluations=total_evals,
            metrics=metrics)
