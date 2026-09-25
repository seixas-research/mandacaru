r"""Nested Pauli echoes implementing Eq. (1) of Nature 646, 825–830 (2025).

``C^(2k) = <[B(t) M]^(2k)>``, with ``B(t) = U† B U`` and independent
Hermitian unitary Pauli insertions B and M. DOI: 10.1038/s41586-025-09526-6.
OTOC order counts repeated echoes, independently of Suzuki–Trotter order.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..core.mapping import PauliSum
from ..units import HARTREE_TO_EV
from .time_evolution import (PreparedState, SuzukiTrotter, _finite_real,
                             _positive_integer, _state_vector)

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


@dataclass(frozen=True)
class NestedOTOCResult:
    """One nested echo on a pure state.

    ``state`` is ``U_k|psi>`` for ``U_k = B(t)[M B(t)]^(k-1)``.
    ``correlator`` is the dimensionless, generally complex ``C^(2k)``.
    ``measurement_expectation`` is the real ``<U_k psi|M|U_k psi>``.
    These coincide only for an initial +1 eigenstate of M; for eigenvalue
    -1 the correlator is minus the measured expectation. For generic states
    the simulation evaluates two branches, retaining the complex correlator.
    """

    state: NDArray[np.complex128]
    correlator: complex
    measurement_expectation: float
    initial_measurement_eigenvalue: int | None
    time: float
    otoc_order: int


@dataclass(frozen=True)
class OTOCSpectrum:
    """Signed Fourier frequencies of a dimensionless nested correlator.

    Frequencies are angular frequencies in atomic units (energy equivalents
    in Hartree with hbar=1), generally sums of energy differences, NOT a list
    of Hamiltonian eigenvalues or ground-state excitation gaps. Amplitudes
    have atomic-time units. No elastic line or sample mean is subtracted.
    """

    frequencies: NDArray[np.float64]
    amplitudes: NDArray[np.complex128]
    correlation: NDArray[np.complex128]
    times: NDArray[np.float64]
    time_step: float
    window: str
    otoc_order: int

    @property
    def frequencies_ev(self) -> NDArray[np.float64]:
        """Energy equivalents of the signed angular frequencies, in eV."""
        return self.frequencies * HARTREE_TO_EV

    @property
    def intensities(self) -> NDArray[np.float64]:
        """Fourier magnitudes in atomic-time units, not absorption strengths."""
        return np.abs(self.amplitudes)

    @property
    def resolution(self) -> float:
        """Angular-frequency bin spacing, in atomic units."""
        return 2 * np.pi / (self.times.size * self.time_step)

    @property
    def nyquist_frequency(self) -> float:
        """Nyquist angular frequency, in atomic units."""
        return np.pi / self.time_step

    def to_csv(self, path: str | PathLike[str] = "otoc_spectrum.csv") -> Path:
        """Write every signed frequency bin, overwriting an existing file.

        The parent directory must exist. Columns explicitly distinguish
        frequency equivalents from excitation energies and time-unit amplitudes
        from the Hartree amplitudes in the dipole-correlation spectrum.
        """
        destination = Path(path)
        data = np.column_stack((self.frequencies, self.frequencies_ev,
                                self.intensities, self.amplitudes.real,
                                self.amplitudes.imag))
        np.savetxt(destination, data, delimiter=",", fmt="%.12e", comments="",
                   header="angular_frequency_ha,angular_frequency_ev,"
                          "magnitude_au_time,fft_real_au_time,fft_imag_au_time")
        return destination


def _pauli_insertion(operator: PauliSum, name: str, n_qubits: int) -> SuzukiTrotter:
    """Compile a signed Pauli string and enforce the paper's involutions."""
    compiled = SuzukiTrotter(operator)
    if compiled.n_qubits != n_qubits:
        raise ValueError(f"{name} and Hamiltonian qubit counts differ")
    if len(compiled._terms) != 1 or compiled._terms[0][1] not in (-1.0, 1.0):
        raise ValueError(f"{name} must be a single Pauli string with coefficient +1 or -1")
    return compiled


class NestedOTOC:
    r"""Evaluate ``<[B(t) M]^(2k)>`` through repeated forward/backward echoes.

    Parameters
    ----------
    hamiltonian
        Hermitian Pauli Hamiltonian in Hartree.
    butterfly, measurement
        B and M, each a single signed Pauli string encoded as a PauliSum.
        Multi-qubit strings are supported. They obey B²=M²=I and use the
        Hamiltonian's register/basis/mapping. A dipole sum is not a substitute.
    trotter_order
        Product-formula accuracy (default 2; also 1 and higher even orders).
        The separate ``otoc_order`` in run/spectrum/circuit selects k.

    Notes
    -----
    Accepts the same pure prepared states and ADAPT checkpoints as time_evolve.
    This implements the nested observable, not the paper's Pauli-averaging
    experiment, hardware noise mitigation, or Hamiltonian-learning optimizer.
    Simulation is matrix-free with O(2**n) state storage. Mixed-state/trace
    expectations can be constructed by averaging pure-state correlators with
    their ensemble probabilities, not by averaging state vectors.
    """

    def __init__(self, hamiltonian: PauliSum, butterfly: PauliSum,
                 measurement: PauliSum, *, trotter_order: int = 2) -> None:
        self._evolution = SuzukiTrotter(hamiltonian, order=trotter_order)
        self._b = _pauli_insertion(butterfly, "butterfly", self._evolution.n_qubits)
        self._m = _pauli_insertion(measurement, "measurement", self._evolution.n_qubits)

    def _sequence(self, state: NDArray[np.complex128], time: float,
                  steps: int, k: int) -> NDArray[np.complex128]:
        """Apply U_k, using the exact adjoint of each discrete forward leg."""
        for index in range(k):
            if index:
                state = self._m._apply_operator(state)
            state = self._evolution._evolve(state, time, steps)
            state = self._b._apply_operator(state)
            state = self._evolution._evolve(state, time, steps, inverse=True)
        return state

    def run(self, initial_state: PreparedState, time: float, *,
            otoc_order: int = 2, steps: int = 1) -> NestedOTOCResult:
        """Compute C^(2k) and the physical sequence's final measurement.

        ``time`` is atomic time, ``steps`` subdivides EACH free-evolution leg,
        and positive integer ``otoc_order`` selects k (1: four operators,
        2: eight operators). Generic pure states require two branches:
        ``C = <U_k psi|M U_k M psi>``. An M eigenstate needs only one branch.
        The return amplitude/fidelity of the sequence is not used as an OTOC.
        """
        time = _finite_real(time, "time")
        steps = _positive_integer(steps, "steps")
        k = _positive_integer(otoc_order, "otoc_order")
        psi = _state_vector(initial_state, self._evolution.n_qubits)
        m_psi = self._m._apply_operator(psi)
        eigenvalue = None
        for sign in (1, -1):
            if np.linalg.norm(m_psi - sign * psi) <= 1e-12:
                eigenvalue = sign
                break
        state = self._sequence(psi, time, steps, k)
        m_state = self._m._apply_operator(state)
        measured = float(np.vdot(state, m_state).real)
        if eigenvalue is not None:
            correlator = complex(eigenvalue * measured)
        else:
            branch = self._sequence(m_psi, time, steps, k)
            correlator = complex(np.vdot(m_state, branch))
        return NestedOTOCResult(state, correlator, measured, eigenvalue, time, k)

    def spectrum(self, initial_state: PreparedState, *, time_step: float = 0.5,
                 num_samples: int = 256, steps_per_sample: int = 1,
                 otoc_order: int = 2, window: str = "hann") -> OTOCSpectrum:
        """Sample C^(2k)(t) and Fourier-transform its dimensionless signal.

        At time j*dt every leg uses j*steps_per_sample steps, keeping the
        Trotter step fixed as the observation time grows. Nested trajectories
        are recomputed: cost is quadratic in sample count and linear in k,
        unlike the incremental two-point dipole correlation. Only the current
        branches and scalar time series are stored. There is no zero padding
        or automatic DC subtraction. Multi-gap frequencies may require finer
        sampling than ordinary dipole spectroscopy to avoid aliasing.
        """
        dt = _finite_real(time_step, "time_step")
        n = _positive_integer(num_samples, "num_samples")
        steps = _positive_integer(steps_per_sample, "steps_per_sample")
        k = _positive_integer(otoc_order, "otoc_order")
        if dt <= 0 or n < 4:
            raise ValueError("time_step must be positive and num_samples at least four")
        if window not in ("hann", "none"):
            raise ValueError("window must be 'hann' or 'none'")
        psi = _state_vector(initial_state, self._evolution.n_qubits)
        times = np.arange(n) * dt
        correlation = np.empty(n, dtype=complex)
        for j, time in enumerate(times):
            correlation[j] = self.run(psi, float(time), otoc_order=k,
                                      steps=max(1, j * steps)).correlator
        weights = np.hanning(n) if window == "hann" else np.ones(n)
        amplitudes = dt * np.fft.fftshift(np.fft.fft(weights * correlation))
        frequencies = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(n, d=dt))
        return OTOCSpectrum(frequencies, amplitudes, correlation, times, dt, window, k)

    def circuit(self, time: float, *, otoc_order: int = 2,
                steps: int = 1) -> QuantumCircuit:
        """Export U_k as a Qiskit circuit without preparation or measurement.

        Direct M readout yields C only for an initial +1 M eigenstate; a -1
        eigenstate needs a sign correction. Generic states require a separate
        interferometric readout; this method does not add an ancilla circuit.
        """
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import PauliGate

        time = _finite_real(time, "time")
        steps = _positive_integer(steps, "steps")
        k = _positive_integer(otoc_order, "otoc_order")
        forward = self._evolution.circuit(time, steps=steps)
        backward = forward.inverse()
        circuit = QuantumCircuit(self._evolution.n_qubits)

        def insert(operator: SuzukiTrotter) -> None:
            label, sign = operator._terms[0]
            circuit.append(PauliGate(label), range(circuit.num_qubits))
            if sign == -1:
                circuit.global_phase += np.pi

        for index in range(k):
            if index:
                insert(self._m)
            circuit.compose(forward, inplace=True)
            insert(self._b)
            circuit.compose(backward, inplace=True)
        return circuit
