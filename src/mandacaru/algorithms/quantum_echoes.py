r"""Dipole-kicked forward/backward quantum echoes from a prepared state.

The protocol is ``|echo(t)> = U(t)† exp(-i tau_p V) U(t) |psi>``.  This is
a single kicked echo; its return fidelity is not itself a four-operator OTOC.
For spectroscopy the result also includes the correlation ``<V V(t)>`` and
the change in ``<V>`` after the echo. ``QuantumEchoes.spectrum`` samples the
correlation incrementally and Fourier-transforms it to excitation energies.
All quantities use atomic units, with an eV view of the spectrum available.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.mapping import PauliSum
from ..units import HARTREE_TO_EV
from .time_evolution import (PreparedState, SuzukiTrotter, _finite_real,
                             _positive_integer, _state_vector)


@dataclass(frozen=True)
class QuantumEchoesResult:
    """One echo experiment in atomic units.

    ``state`` is the final full-register vector, ``amplitude = <psi|echo>``
    and ``fidelity = |amplitude|²``. ``response`` is
    ``<echo|V|echo> - <psi|V|psi>`` (Hartree). ``correlation`` is
    ``<psi|V U† V U|psi>`` (Hartree²). For an exact ground state it is
    ``sum_n |<n|V|0>|² exp(+i (E_n-E_0) time)``; subtract ``<V>²`` to remove
    its elastic contribution. Its positive phase reflects backward evolution.
    ``perturbation_bound`` reports ``abs(tau_p) sum_P |v_P|``.
    """

    state: NDArray[np.complex128]
    amplitude: complex
    fidelity: float
    response: float
    correlation: complex
    time: float
    tau_p: float
    perturbation_bound: float


@dataclass(frozen=True)
class EchoSpectrum:
    """Fourier spectrum of the connected potential correlation.

    ``energies`` are signed excitation gaps in Hartree, not absolute total
    energies. ``amplitudes = dt * FFT(window * (C - elastic))`` uses the
    negative Fourier phase, matching ``C(t) ~ exp(+i gap*t)``. ``intensities``
    are its magnitudes (Hartree), not calibrated absorption cross sections.
    The raw correlation (Hartree squared) and its time samples are retained.
    The energy-bin spacing is ``2*pi/(N*dt)``; windowing broadens peaks further.
    There is no zero padding that could disguise the physical resolution.
    """

    energies: NDArray[np.float64]
    amplitudes: NDArray[np.complex128]
    correlation: NDArray[np.complex128]
    times: NDArray[np.float64]
    time_step: float
    window: str
    elastic: float

    @property
    def intensities(self) -> NDArray[np.float64]:
        """Magnitude of the Fourier transform in Hartree."""
        return np.abs(self.amplitudes)

    @property
    def energies_ev(self) -> NDArray[np.float64]:
        """Signed excitation gaps in eV."""
        return self.energies * HARTREE_TO_EV

    @property
    def resolution(self) -> float:
        """Fourier-bin spacing in Hartree (before window broadening)."""
        return 2*np.pi / (self.times.size * self.time_step)

    @property
    def nyquist_energy(self) -> float:
        """Nyquist energy pi/dt in Hartree; larger gaps alias."""
        return np.pi / self.time_step

    def peaks(self, relative_threshold: float = 0.05) -> NDArray[np.int64]:
        """Indices of positive-energy local maxima in increasing energy order.

        Keep maxima at least ``relative_threshold`` times the largest positive
        spectral magnitude. These are sampled bins without peak fitting.
        """
        from scipy.signal import find_peaks

        threshold = _finite_real(relative_threshold, "relative_threshold")
        if not 0 <= threshold <= 1:
            raise ValueError("relative_threshold must lie in [0, 1]")
        intensity = self.intensities
        positive = self.energies > 0
        scale = float(np.max(intensity[positive]))
        if scale == 0:
            return np.empty(0, dtype=np.int64)
        indices, _ = find_peaks(intensity, height=threshold*scale)
        return indices[self.energies[indices] > 0]

    def to_csv(self, path: str | PathLike[str] = "spectrum.csv") -> Path:
        """Write the complete signed spectrum to CSV and return its path.

        Columns are excitation energy (Hartree, eV), Fourier magnitude, and
        real/imaginary Fourier amplitudes (all three in Hartree). One plain
        header is followed by numeric rows at 12-digit scientific precision.
        Existing files are replaced; the parent directory must already exist.
        """
        destination = Path(path)
        data = np.column_stack((self.energies, self.energies_ev, self.intensities,
                                self.amplitudes.real, self.amplitudes.imag))
        np.savetxt(destination, data, delimiter=",", fmt="%.12e", comments="",
                   header="energy_ha,energy_ev,magnitude_ha,fft_real_ha,fft_imag_ha")
        return destination


def fourier_spectrum(correlation: ArrayLike, time_step: float, *,
                     elastic: float = 0.0, window: str = "hann") -> EchoSpectrum:
    """Transform uniformly sampled ``<V V(t)>`` into an excitation spectrum.

    Supply at least four finite complex samples starting at t=0, with positive
    ``time_step`` in atomic units. ``elastic`` is the known ground-state
    ``<V>**2`` contribution to remove, not the sample mean (which could erase
    unresolved low-energy transitions). ``window`` is ``'hann'`` (default) or
    ``'none'``. The FFT convention resolves positive-phase oscillations at
    positive energy and retains negative energies for nonstationary inputs.
    """
    dt = _finite_real(time_step, "time_step")
    if dt <= 0:
        raise ValueError("time_step must be positive")
    elastic = _finite_real(elastic, "elastic")
    signal = np.array(correlation, dtype=complex, copy=True)
    if signal.ndim != 1 or signal.size < 4 or not np.all(np.isfinite(signal)):
        raise ValueError("correlation must be a finite vector of at least four samples")
    if window not in ("hann", "none"):
        raise ValueError("window must be 'hann' or 'none'")
    weights = np.hanning(signal.size) if window == "hann" else np.ones(signal.size)
    transformed = dt * np.fft.fftshift(np.fft.fft(weights * (signal - elastic)))
    energies = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(signal.size, d=dt))
    return EchoSpectrum(energies=energies, amplitudes=transformed,
                        correlation=signal, times=np.arange(signal.size)*dt,
                        time_step=dt, window=window, elastic=elastic)


class QuantumEchoes:
    r"""Compile the Hamiltonian and dipole potential for repeated echoes.

    Parameters
    ----------
    hamiltonian, perturbation
        Hermitian Pauli sums in Hartree on the same register. Build the latter
        with :func:`~mandacaru.core.dipole.electric_dipole_potential` or supply
        a consistently mapped/tapered electric dipole coupling directly.
    order
        Suzuki–Trotter order for both evolutions and the kick; default 2,
        also accepts 1 or higher even orders.
    max_perturbation
        Strict upper bound on ``abs(tau_p) sum_P |v_P|``, default 0.1 and
        restricted to ``0 < max_perturbation <= 0.1``. This bounds
        ``||tau_p V||`` without diagonalization. Oversized pulses are rejected,
        never silently clipped. Lower this bound and compare pulse strengths
        when checking linear response. The Pauli bound can be conservative.

    Notes
    -----
    An ADAPT-VQE checkpoint from ``calc.solver.checkpoint`` supplies the full
    prepared state, including when the optimizer used a particle-number sector.
    Echo fidelity alone is constant in time for an exact eigenstate; use the
    correlation or signed dipole response to investigate excitation gaps.
    """

    def __init__(self, hamiltonian: PauliSum, perturbation: PauliSum, *,
                 order: int = 2, max_perturbation: float = 0.1) -> None:
        limit = _finite_real(max_perturbation, "max_perturbation")
        if not 0 < limit <= 0.1:
            raise ValueError("max_perturbation must be in (0, 0.1]")
        self._evolution = SuzukiTrotter(hamiltonian, order=order)
        self._kick = SuzukiTrotter(perturbation, order=order)
        if self._evolution.n_qubits != self._kick.n_qubits:
            raise ValueError("Hamiltonian and perturbation qubit counts differ")
        self.max_perturbation = limit

    def spectrum(self, initial_state: PreparedState, *, time_step: float = 0.5,
                 num_samples: int = 1024, steps_per_sample: int = 10,
                 window: str = "hann", remove_elastic: bool = True) -> EchoSpectrum:
        """Sample the correlation on a uniform time grid and Fourier-transform it.

        Evolve two vectors incrementally and evaluate
        ``C(t) = <U(t) V psi | V U(t) psi>``. This equals the echo correlation
        without repeating long forward/backward trajectories. Cost is linear
        in ``num_samples``; no Hamiltonian diagonalization is performed.
        ``steps_per_sample`` subdivides each interval, so Trotter accuracy
        stays controlled as the observation time grows. Check convergence in
        this parameter and time-step/Nyquist and observation-time resolution.

        This is the direct linear-response correlation; it is independent of
        ``tau_p``. The finite-pulse echo is calculated separately by ``run``.
        By default subtract ``<psi|V|psi>**2`` before the FFT. Its interpretation
        as the elastic line is exact for an eigenstate; an approximate ADAPT
        state also contributes coherences and potentially negative frequencies.
        """
        dt = _finite_real(time_step, "time_step")
        n = _positive_integer(num_samples, "num_samples")
        steps = _positive_integer(steps_per_sample, "steps_per_sample")
        if dt <= 0 or n < 4:
            raise ValueError("time_step must be positive and num_samples at least four")
        if window not in ("hann", "none"):
            raise ValueError("window must be 'hann' or 'none'")
        state = _state_vector(initial_state, self._evolution.n_qubits)
        perturbed = self._kick._apply_operator(state)
        elastic = float(np.vdot(state, perturbed).real)**2 if remove_elastic else 0.0
        correlation = np.empty(n, dtype=complex)
        for i in range(n):
            correlation[i] = np.vdot(perturbed, self._kick._apply_operator(state))
            if i + 1 < n:
                state = self._evolution._evolve(state, dt, steps)
                perturbed = self._evolution._evolve(perturbed, dt, steps)
        return fourier_spectrum(correlation, dt, elastic=elastic, window=window)

    def run(self, initial_state: PreparedState, time: float, *,
            tau_p: float = 1e-3, steps: int = 1,
            kick_steps: int = 1) -> QuantumEchoesResult:
        """Evolve forward, apply the weak dipole pulse, and reverse evolution.

        ``initial_state`` is a normalized vector or wavefunction checkpoint
        (object or path). ``time`` and signed ``tau_p`` are in atomic time
        units; ``tau_p=0`` is the unperturbed reference. ``steps`` slices each
        free evolution and ``kick_steps`` slices the pulse. Increasing the
        latter reduces splitting error, but does not weaken an oversized pulse.
        The reverse leg is exactly the adjoint of the forward approximation.
        """
        time = _finite_real(time, "time")
        tau_p = _finite_real(tau_p, "tau_p")
        steps = _positive_integer(steps, "steps")
        kick_steps = _positive_integer(kick_steps, "kick_steps")
        bound = abs(tau_p) * self._kick.norm_bound
        if bound > self.max_perturbation:
            allowed = self.max_perturbation / self._kick.norm_bound
            raise ValueError(f"pulse is not small: |tau_p| sum|v_P| = {bound:g} "
                             f"> {self.max_perturbation:g}; use |tau_p| <= {allowed:g}")
        psi = _state_vector(initial_state, self._evolution.n_qubits)
        forward = self._evolution._evolve(psi, time, steps)
        kicked = self._kick._evolve(forward, tau_p, kick_steps)
        echo = self._evolution._evolve(kicked, time, steps, inverse=True)
        v_psi = self._kick._apply_operator(psi)
        vt_psi = self._evolution._evolve(
            self._kick._apply_operator(forward), time, steps, inverse=True)
        amplitude = complex(np.vdot(psi, echo))
        response = float((np.vdot(echo, self._kick._apply_operator(echo))
                          - np.vdot(psi, v_psi)).real)
        return QuantumEchoesResult(
            state=echo, amplitude=amplitude, fidelity=float(abs(amplitude)**2),
            response=response, correlation=complex(np.vdot(v_psi, vt_psi)),
            time=time, tau_p=tau_p, perturbation_bound=bound)


def quantum_echoes(initial_state: PreparedState, hamiltonian: PauliSum,
                    perturbation: PauliSum, time: float, *, tau_p: float = 1e-3,
                    steps: int = 1, order: int = 2, kick_steps: int = 1,
                    max_perturbation: float = 0.1) -> QuantumEchoesResult:
    """Run one echo; see :class:`QuantumEchoes` for units and pulse validation."""
    return QuantumEchoes(hamiltonian, perturbation, order=order,
                         max_perturbation=max_perturbation).run(
        initial_state, time, tau_p=tau_p, steps=steps, kick_steps=kick_steps)
