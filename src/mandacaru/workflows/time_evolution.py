"""Matrix-free, unitary Suzuki–Trotter propagation of Pauli Hamiltonians.

All energies are in Hartree and times in atomic units (hbar = 1).  Qubit 0
is the most significant bit, as in :class:`~mandacaru.core.mapping.PauliSum`.
"""

from __future__ import annotations
from os import PathLike
from typing import TYPE_CHECKING, TypeAlias
import numpy as np
from numpy.typing import ArrayLike, NDArray
from ..core.checkpoint import WavefunctionCheckpoint
from ..core.mapping import PauliSum

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
PreparedState: TypeAlias = ArrayLike | WavefunctionCheckpoint | str | PathLike[str]


def _positive_integer(value: int, name: str) -> int:
    """Validate a count without silently truncating floats or accepting bools."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_real(value: float, name: str) -> float:
    """Validate a finite real scalar."""
    if np.ndim(value) != 0 or not np.isrealobj(value):
        raise ValueError(f"{name} must be a finite real scalar")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    return value


def _state_vector(
    initial_state: PreparedState, n_qubits: int, *, resources=None
) -> NDArray[np.complex128]:
    """Copy and validate a normalized full-register state or checkpoint."""
    if isinstance(initial_state, (str, PathLike)):
        initial_state = WavefunctionCheckpoint.load(initial_state, resources=resources)
    if isinstance(initial_state, WavefunctionCheckpoint):
        if initial_state.n_qubits != n_qubits:
            raise ValueError("checkpoint and operator qubit counts differ")
        initial_state = initial_state.state_vector(resources=resources)
    state = np.array(initial_state, dtype=complex, copy=True)
    if state.ndim != 1 or state.size != 2**n_qubits:
        raise ValueError(f"initial_state must have shape ({2**n_qubits},)")
    if not np.all(np.isfinite(state)):
        raise ValueError("initial_state must contain only finite amplitudes")
    if not np.isclose(np.vdot(state, state).real, 1.0, rtol=0, atol=1e-10):
        raise ValueError("initial_state must be normalized")
    return state


class SuzukiTrotter:
    """Reusable product formula for ``U(t) = exp(-i H t)``.

    Parameters
    ----------
    hamiltonian : PauliSum
        Hermitian Pauli sum in Hartree. Terms retain insertion order. A copy
        is compiled, so later changes to the input do not affect propagation.
    order : int
        Default 2 (Strang splitting); 1 and all positive even orders are
        supported. Higher orders use the five-stage Suzuki recursion and cost
        roughly ``5**(order/2 - 1)`` times as much as order 2.

    Notes
    -----
    Each Pauli rotation uses ``cos(a) I - i sin(a) P`` and bit permutations,
    never a dense Hamiltonian or propagator. Working memory is O(2**n) plus
    the rotation schedule. This is still a full state-vector simulator.
    """

    def __init__(
        self, hamiltonian: PauliSum, *, order: int = 2, resources=None
    ) -> None:
        order = _positive_integer(order, "order")
        if order != 1 and order % 2:
            raise ValueError("order must be 1 or a positive even integer")
        if not isinstance(hamiltonian, PauliSum):
            raise TypeError("hamiltonian must be a PauliSum")
        self.n_qubits = hamiltonian.num_qubits
        if not 1 <= self.n_qubits <= 62:
            raise ValueError("hamiltonian must specify between 1 and 62 qubits")
        self.order = order
        self._resources = dict(resources or {})
        self._native = None
        terms: list[tuple[str, float]] = []
        for label, coefficient in hamiltonian.terms.items():
            if len(label) != self.n_qubits or label.strip("IXYZ"):
                raise ValueError("invalid Pauli label")
            if not np.isfinite(coefficient) or abs(complex(coefficient).imag) > 1e-12:
                raise ValueError("hamiltonian must be finite and Hermitian")
            if coefficient.real != 0:
                terms.append((label, float(coefficient.real)))
        self._terms = tuple(terms)
        self.norm_bound = float(sum((abs(c) for _, c in terms)))
        if not np.isfinite(self.norm_bound):
            raise ValueError("sum of absolute Pauli coefficients must be finite")
        from .._native import Context

        self._context = Context(**dict(resources or {}))
        self._native = self._context.propagator(
            self.n_qubits, [(l, c, 0.0) for l, c in terms], order=order
        )
        return

    def _apply_operator(self, state):
        return self._native.apply(np.ascontiguousarray(state, dtype=complex))

    def _evolve(self, state, time, steps, inverse=False):
        return self._native.evolve(
            np.ascontiguousarray(state, dtype=complex),
            time,
            steps=steps,
            inverse=inverse,
            normalized=False,
        )

    def evolve(
        self,
        initial_state: PreparedState,
        time: float,
        *,
        steps: int = 1,
        inverse: bool = False,
    ) -> NDArray[np.complex128]:
        """Return a new propagated state; the input is never modified.

        ``time`` is a finite signed time in atomic units and ``steps`` is the
        number of equal Trotter slices over that interval. ``inverse=True``
        applies the exact adjoint of the chosen approximation by reversing
        rotations as well as their signs, including for first order.
        Checkpoints (objects or paths) and normalized full-register vectors
        are accepted. Sector-compressed vectors must first be expanded.
        """
        time = _finite_real(time, "time")
        steps = _positive_integer(steps, "steps")
        return self._evolve(
            _state_vector(initial_state, self.n_qubits, resources=self._resources),
            time,
            steps,
            inverse,
        )

    def circuit(
        self, time: float, *, steps: int = 1, inverse: bool = False
    ) -> QuantumCircuit:
        """Export the same propagation as Qiskit Pauli rotations.

        The circuit contains evolution only, with no state preparation or
        measurement. Its state-vector ordering agrees with ``PauliSum``.
        """
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.quantum_info import SparsePauliOp

        time = _finite_real(time, "time")
        steps = _positive_integer(steps, "steps")
        circuit = QuantumCircuit(self.n_qubits)
        constant, rotations = self._native.schedule()
        circuit.global_phase = -time * constant
        for _ in range(steps):
            for label, weight in rotations:
                circuit.append(
                    PauliEvolutionGate(
                        SparsePauliOp(label), time=time * weight / steps
                    ),
                    range(self.n_qubits),
                )
        return circuit.inverse() if inverse else circuit


def time_evolve(
    initial_state: PreparedState,
    hamiltonian: PauliSum,
    time: float,
    *,
    steps: int = 1,
    order: int = 2,
    resources=None,
) -> NDArray[np.complex128]:
    """Propagate a prepared state with a Suzuki–Trotter formula (default order 2).

    Energies and time are in atomic units. Reuse :class:`SuzukiTrotter` for
    repeated calls with the same Hamiltonian to avoid recompiling its terms.
    """
    return SuzukiTrotter(hamiltonian, order=order, resources=resources).evolve(
        initial_state, time, steps=steps
    )
