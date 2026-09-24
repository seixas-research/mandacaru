"""Matrix-free, unitary Suzuki–Trotter propagation of Pauli Hamiltonians.

All energies are in Hartree and times in atomic units (hbar = 1).  Qubit 0
is the most significant bit, as in :class:`~mandacaru.core.mapping.PauliSum`.
"""

from __future__ import annotations

from collections.abc import Iterator
from os import PathLike
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.checkpoint import WavefunctionCheckpoint
from ..core.mapping import PauliSum
from ..core.sector import pauli_masks

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


def _state_vector(initial_state: PreparedState, n_qubits: int) -> NDArray[np.complex128]:
    """Copy and validate a normalized full-register state or checkpoint."""
    if isinstance(initial_state, (str, PathLike)):
        initial_state = WavefunctionCheckpoint.load(initial_state)
    if isinstance(initial_state, WavefunctionCheckpoint):
        if initial_state.n_qubits != n_qubits:
            raise ValueError("checkpoint and operator qubit counts differ")
        initial_state = initial_state.state_vector()
    state = np.array(initial_state, dtype=complex, copy=True)
    if state.ndim != 1 or state.size != 2**n_qubits:
        raise ValueError(f"initial_state must have shape ({2**n_qubits},)")
    if not np.all(np.isfinite(state)):
        raise ValueError("initial_state must contain only finite amplitudes")
    if not np.isclose(np.vdot(state, state).real, 1.0, rtol=0, atol=1e-10):
        raise ValueError("initial_state must be normalized")
    return state


def _suzuki_stages(order: int, scale: float = 1.0) -> Iterator[float]:
    """Yield second-order stage weights in Suzuki's symmetric recursion."""
    if order == 2:
        yield scale
        return
    p = 1.0 / (4.0 - 4.0 ** (1.0 / (order - 1)))
    for weight in (p, p, 1 - 4*p, p, p):
        yield from _suzuki_stages(order - 2, scale * weight)


class SuzukiTrotter:
    r"""Reusable product formula for ``U(t) = exp(-i H t)``.

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

    def __init__(self, hamiltonian: PauliSum, *, order: int = 2) -> None:
        order = _positive_integer(order, "order")
        if order != 1 and order % 2:
            raise ValueError("order must be 1 or a positive even integer")
        if not isinstance(hamiltonian, PauliSum):
            raise TypeError("hamiltonian must be a PauliSum")
        self.n_qubits = hamiltonian.num_qubits
        if not 1 <= self.n_qubits <= 62:
            raise ValueError("hamiltonian must specify between 1 and 62 qubits")
        self.order = order
        terms: list[tuple[str, float]] = []
        for label, coefficient in hamiltonian.terms.items():
            if len(label) != self.n_qubits or label.strip("IXYZ"):
                raise ValueError("invalid Pauli label")
            if not np.isfinite(coefficient) or abs(complex(coefficient).imag) > 1e-12:
                raise ValueError("hamiltonian must be finite and Hermitian")
            if coefficient.real != 0:
                terms.append((label, float(coefficient.real)))
        self._terms = tuple(terms)
        self._masks = tuple(pauli_masks(label) for label, _ in terms)
        self.norm_bound = float(sum(abs(c) for _, c in terms))
        if not np.isfinite(self.norm_bound):
            raise ValueError("sum of absolute Pauli coefficients must be finite")
        # Identity commutes with everything; evolve its phase only once.
        identity = "I" * self.n_qubits
        self._constant = sum(c for label, c in terms if label == identity)
        active = [(k, c) for k, (label, c) in enumerate(terms) if label != identity]
        rotations: list[tuple[int, float]] = []

        def append(k: int, weight: float) -> None:
            if rotations and rotations[-1][0] == k:
                rotations[-1] = (k, rotations[-1][1] + weight)
            else:
                rotations.append((k, weight))

        if order == 1 or len(active) <= 1:
            rotations.extend(active)
        else:
            for scale in _suzuki_stages(order):
                for k, coefficient in active:
                    append(k, coefficient * scale / 2)
                for k, coefficient in reversed(active):
                    append(k, coefficient * scale / 2)
        self._rotations = tuple(rotations)

    def _pauli_action(self, k: int, state: NDArray[np.complex128],
                      indices: NDArray[np.int64]) -> NDArray[np.complex128]:
        """Apply one compiled string in O(2**n) memory and time."""
        flip, phase, n_y = self._masks[k]
        source = indices ^ flip
        parity = (np.bitwise_count(source & phase) & 1).astype(np.int8)
        return (1j**n_y) * (1 - 2*parity) * state[source]

    def _apply_operator(self, state: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Apply the original Pauli sum to a vector without constructing H."""
        indices = np.arange(state.size, dtype=np.int64)
        out = np.zeros_like(state)
        for k, (_, coefficient) in enumerate(self._terms):
            out += coefficient * self._pauli_action(k, state, indices)
        return out

    def _evolve(self, state: NDArray[np.complex128], time: float,
                steps: int, inverse: bool = False) -> NDArray[np.complex128]:
        """Propagate an owned vector, including unnormalized correlation states."""
        if time == 0:
            return state
        dt = (-time if inverse else time) / steps
        sequence = self._rotations[::-1] if inverse else self._rotations
        angles = [(k, np.cos(dt*w), -1j*np.sin(dt*w)) for k, w in sequence]
        indices = np.arange(state.size, dtype=np.int64)
        for _ in range(steps):
            for k, cosine, sine in angles:
                state = cosine * state + sine * self._pauli_action(k, state, indices)
        return np.exp(-1j * (-time if inverse else time) * self._constant) * state

    def evolve(self, initial_state: PreparedState, time: float, *,
               steps: int = 1, inverse: bool = False) -> NDArray[np.complex128]:
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
        return self._evolve(_state_vector(initial_state, self.n_qubits), time,
                            steps, inverse)

    def circuit(self, time: float, *, steps: int = 1,
                inverse: bool = False) -> QuantumCircuit:
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
        circuit.global_phase = -time * self._constant
        for _ in range(steps):
            for k, weight in self._rotations:
                circuit.append(PauliEvolutionGate(
                    SparsePauliOp(self._terms[k][0]), time=time*weight/steps),
                    range(self.n_qubits))
        return circuit.inverse() if inverse else circuit


def time_evolve(initial_state: PreparedState, hamiltonian: PauliSum,
                time: float, *, steps: int = 1,
                order: int = 2) -> NDArray[np.complex128]:
    """Propagate a prepared state with a Suzuki–Trotter formula (default order 2).

    Energies and time are in atomic units. Reuse :class:`SuzukiTrotter` for
    repeated calls with the same Hamiltonian to avoid recompiling its terms.
    """
    return SuzukiTrotter(hamiltonian, order=order).evolve(initial_state, time, steps=steps)
