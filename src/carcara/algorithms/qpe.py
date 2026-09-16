# -*- coding: utf-8 -*-
# file: algorithms/qpe.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Quantum phase estimation from a variationally prepared state.

QPE reads eigenvalues of :math:`H` off the phases of :math:`U = e^{2\pi i
(H - E_{\text{lo}})/W}`: an eigenstate :math:`|E_j\rangle` is a :math:`U`
eigenvector with phase :math:`\varphi_j = (E_j - E_{\text{lo}})/W \in [0, 1)`,
and :math:`t` evaluation qubits resolve that phase to :math:`2^{-t}`, i.e. the
energy to :math:`W/2^t`.  Fed a state :math:`|\Psi\rangle = \sum_j c_j
|E_j\rangle`, the evaluation register comes out peaked at every
:math:`\varphi_j` with weight :math:`|c_j|^2` -- so the *better* the input
state, the more probable the ground-state reading, and a variationally
optimized wavefunction (a :class:`~carcara.core.checkpoint.WavefunctionCheckpoint`
written by ADAPT-VQE or VQE) is exactly the right input.  The system register
collapses onto the eigenvector belonging to the phase read, which is also
returned.

Simulation
----------
The textbook circuit is Hadamards on the evaluation register, controlled
:math:`U^{2^k}` powers, and an inverse quantum Fourier transform.  Its state
before the transform is :math:`2^{-t/2}\sum_k |k\rangle \otimes U^k|\Psi\rangle`,
which this module builds **exactly** as a ``(2^t, 2^n)`` array -- the full
:math:`2^{n+t}` state vector, row :math:`k` being :math:`U^k|\Psi\rangle` --
and the inverse QFT on the evaluation register is then an FFT along the first
axis.  Nothing is approximated: no Trotter step, no truncated eigen-expansion.
For :math:`n \le` :data:`DENSE_LIMIT_QUBITS` the powers come from the dense
eigendecomposition of :math:`H` (which also yields the exact spectrum, reported
for comparison); beyond that :math:`U` is applied step by step with a sparse
matrix exponential.

Memory
------
A faithful simulation *is* a :math:`2^{n+t}` complex vector, and the work
needs about three of them.  :func:`qpe_memory_estimate` sizes that before
anything is allocated, :meth:`QuantumPhaseEstimation.run` compares it with the
memory actually available and, by ``memory_policy``, aborts, prompts or
proceeds with a warning.  Every run prints the estimate.

Circuit
-------
:meth:`QuantumPhaseEstimation.circuit` exports the same experiment as a Qiskit
circuit: the checkpoint's state preparation on the system wires followed by
:class:`qiskit.circuit.library.PhaseEstimation` around an *exact*
:class:`~qiskit.circuit.library.PauliEvolutionGate` (matrix-exponential
synthesis), so a state-vector simulation of the circuit reproduces the native
result to machine precision and the circuit is what a hardware run would
transpile.
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field

import numpy as np

from ..core.checkpoint import WavefunctionCheckpoint, prepare_state
from ..core.mapping import PauliSum
from ..units import energy_unit_label, from_hartree, to_hartree

#: Largest system register whose Hamiltonian is diagonalized densely (which
#: also gives the exact spectrum for comparison); above it the powers of ``U``
#: are applied with a sparse matrix exponential.
DENSE_LIMIT_QUBITS = 12
#: Bytes per complex amplitude.
COMPLEX_BYTES = 16
#: The simulation holds about this many full ``2^(n+t)`` vectors at once
#: (the evolved columns, the FFT output and scratch).
WORKING_SET_FACTOR = 3
#: The memory policies :class:`QuantumPhaseEstimation` accepts.
MEMORY_POLICIES = ("abort", "prompt", "ignore")
#: Default fraction of the available memory a run may claim.
DEFAULT_MEMORY_FRACTION = 0.5
#: Above this fraction of the available memory the estimate is printed as a
#: warning even when the policy lets the run proceed.
WARN_MEMORY_FRACTION = 0.25
#: Default number of evaluation qubits.
DEFAULT_EVALUATION_QUBITS = 8


# --------------------------------------------------------------------------- #
# Memory.
# --------------------------------------------------------------------------- #

def available_memory() -> int | None:
    """Bytes of memory available to this process, or ``None`` if unknown.

    ``psutil`` when installed (it accounts for cache and reclaimable pages);
    otherwise the total physical memory from ``sysconf``, which over-states
    what is free, so the policy check errs toward permitting.
    """
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:                                   # pragma: no cover
        pass
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):       # pragma: no cover
        return None


@dataclass(frozen=True)
class QPEMemoryEstimate:
    """What a QPE simulation of ``n_system + n_evaluation`` qubits allocates."""

    n_system: int
    n_evaluation: int
    statevector_bytes: int      #: one ``2^(n+t)`` complex vector
    working_bytes: int          #: ``WORKING_SET_FACTOR`` of them
    unitary_bytes: int          #: the dense eigenvector matrix, if used
    available_bytes: int | None

    @property
    def total_bytes(self) -> int:
        return self.working_bytes + self.unitary_bytes

    @property
    def fraction(self) -> float | None:
        """Share of the available memory the run would claim."""
        if not self.available_bytes:
            return None
        return self.total_bytes / self.available_bytes

    def summary(self) -> str:
        avail = ("unknown" if self.available_bytes is None
                 else f"{self.available_bytes / 2**30:.2f} GiB")
        share = ("" if self.fraction is None
                 else f" = {100 * self.fraction:.1f}% of the {avail} available")
        return (f"QPE statevector: {self.n_system} system + "
                f"{self.n_evaluation} evaluation qubits -> 2^"
                f"{self.n_system + self.n_evaluation} amplitudes "
                f"({_human(self.statevector_bytes)}); working set "
                f"{_human(self.working_bytes)}"
                + (f" + {_human(self.unitary_bytes)} for the dense unitary"
                   if self.unitary_bytes else "")
                + f" = {_human(self.total_bytes)}{share}")


def _human(n_bytes: int) -> str:
    value = float(n_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if value < 1024 or unit == "PiB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.2f} PiB"                           # pragma: no cover


def qpe_memory_estimate(n_system: int, n_evaluation: int,
                        dense: bool | None = None) -> QPEMemoryEstimate:
    """Size a QPE simulation **before** allocating anything.

    ``dense`` says whether the Hamiltonian will be diagonalized densely
    (``None``: the module's own rule, :data:`DENSE_LIMIT_QUBITS`).
    """
    n, t = int(n_system), int(n_evaluation)
    if n < 1 or t < 1:
        raise ValueError("need at least one system and one evaluation qubit")
    if dense is None:
        dense = n <= DENSE_LIMIT_QUBITS
    statevector = COMPLEX_BYTES * 2 ** (n + t)
    unitary = COMPLEX_BYTES * 4 ** n if dense else 0
    return QPEMemoryEstimate(n_system=n, n_evaluation=t,
                             statevector_bytes=statevector,
                             working_bytes=WORKING_SET_FACTOR * statevector,
                             unitary_bytes=unitary,
                             available_bytes=available_memory())


def check_memory(estimate: QPEMemoryEstimate, policy: str = "abort",
                 fraction: float = DEFAULT_MEMORY_FRACTION,
                 verbose: bool = True) -> bool:
    """Apply a memory policy to an estimate; ``True`` means proceed.

    ``"abort"`` raises :class:`MemoryError` when the estimate exceeds
    ``fraction`` of the available memory; ``"prompt"`` asks on the terminal
    (and aborts when there is no terminal to ask); ``"ignore"`` proceeds with a
    warning.  Below the threshold every policy proceeds, with the estimate
    printed when ``verbose`` and a warning above :data:`WARN_MEMORY_FRACTION`.
    """
    if policy not in MEMORY_POLICIES:
        raise ValueError(f"memory_policy must be one of {MEMORY_POLICIES}, "
                         f"got {policy!r}")
    text = estimate.summary()
    share = estimate.fraction
    over = share is not None and share > float(fraction)
    if not over:
        if share is not None and share > WARN_MEMORY_FRACTION:
            warnings.warn(f"large QPE simulation: {text}", RuntimeWarning,
                          stacklevel=2)
        elif verbose:
            print(text)
        return True

    message = (f"QPE would need {_human(estimate.total_bytes)}, "
               f"{100 * share:.0f}% of the {_human(estimate.available_bytes)} "
               f"available (limit {100 * float(fraction):.0f}%).  Reduce the "
               "evaluation qubits or the system register, or free memory.")
    if policy == "abort":
        raise MemoryError(f"refusing to start: {message}")
    if policy == "prompt":
        print(f"WARNING: {message}")
        if not sys.stdin or not sys.stdin.isatty():
            raise MemoryError("no terminal to confirm the memory use on; "
                              + message)
        answer = input("Proceed anyway? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            raise MemoryError(f"aborted at the prompt: {message}")
        return True
    warnings.warn(f"proceeding despite the memory estimate: {message}",
                  RuntimeWarning, stacklevel=2)
    return True


# --------------------------------------------------------------------------- #
# Result.
# --------------------------------------------------------------------------- #

@dataclass
class QPEResult:
    """The evaluation-register distribution and what it says about ``H``.

    Energies are in :attr:`energy_unit` (eV by default, Hartree with
    ``atomic_units=True``); :meth:`in_units` converts.
    """

    energies: np.ndarray            #: energy of every evaluation bin ``m``
    probabilities: np.ndarray       #: ``P(m)``
    phases: np.ndarray              #: ``m / 2^t``
    resolution: float               #: energy width of one bin, ``W / 2^t``
    energy_window: tuple[float, float]
    n_system_qubits: int
    n_evaluation_qubits: int
    peaks: list[tuple[float, float]] = field(default_factory=list)  #: (E, P) by P
    exact_energies: np.ndarray | None = None    #: the spectrum, when dense
    overlaps: np.ndarray | None = None          #: ``|<E_j|Psi>|^2``, when dense
    energy_unit: str = "eV"
    _amplitudes: np.ndarray | None = field(default=None, repr=False)

    @property
    def energy(self) -> float:
        """The most probable reading -- the ground state for a good input."""
        return float(self.peaks[0][0])

    @property
    def probability(self) -> float:
        """Probability of the most probable reading."""
        return float(self.peaks[0][1])

    @property
    def ground_state_overlap(self) -> float | None:
        """``|<E_0|Psi>|^2`` -- the success probability of reading ``E_0``."""
        if self.overlaps is None:
            return None
        return float(self.overlaps[int(np.argmin(self.exact_energies))])

    def collapsed_state(self, index: int | None = None) -> np.ndarray:
        """System-register state after reading evaluation bin ``index``.

        The register collapses onto the component of the input state with
        that phase -- the eigenvector for a resolved eigenvalue.  Defaults to
        the most probable bin.
        """
        if self._amplitudes is None:
            raise ValueError("amplitudes were not kept")
        if index is None:
            index = int(np.argmax(self.probabilities))
        row = self._amplitudes[int(index)]
        norm = np.linalg.norm(row)
        if norm == 0:
            raise ValueError(f"evaluation bin {index} has zero probability")
        return row / norm

    def in_units(self, units: str) -> "QPEResult":
        """A copy with every energy in ``units`` (``"eV"`` or ``"Ha"``)."""
        target = energy_unit_label(units)
        if target == self.energy_unit:
            return self
        convert = (lambda e: from_hartree(e, "eV")) if target == "eV" \
            else (lambda e: to_hartree(e, "eV"))
        return QPEResult(
            energies=convert(np.asarray(self.energies)),
            probabilities=self.probabilities, phases=self.phases,
            resolution=float(convert(self.resolution)),
            energy_window=tuple(float(convert(e)) for e in self.energy_window),
            n_system_qubits=self.n_system_qubits,
            n_evaluation_qubits=self.n_evaluation_qubits,
            peaks=[(float(convert(e)), p) for e, p in self.peaks],
            exact_energies=(None if self.exact_energies is None
                            else convert(np.asarray(self.exact_energies))),
            overlaps=self.overlaps, energy_unit=target,
            _amplitudes=self._amplitudes)

    def summary(self, n_peaks: int = 5) -> str:
        u = self.energy_unit
        lines = [f"QPE: {self.n_system_qubits} system + "
                 f"{self.n_evaluation_qubits} evaluation qubits, resolution "
                 f"{self.resolution:.3e} {u}, window "
                 f"[{self.energy_window[0]:+.6f}, {self.energy_window[1]:+.6f}]"
                 f" {u}",
                 "  most probable readings:"]
        for e, p in self.peaks[:n_peaks]:
            lines.append(f"    E = {e:+.8f} {u}   P = {p:.4f}")
        if self.exact_energies is not None:
            e0 = float(np.min(self.exact_energies))
            lines.append(f"  exact ground state: {e0:+.8f} {u} "
                         f"(|<E0|Psi>|^2 = {self.ground_state_overlap:.4f}; "
                         f"reading error {self.energy - e0:+.3e} {u})")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The algorithm.
# --------------------------------------------------------------------------- #

class QuantumPhaseEstimation:
    r"""Quantum phase estimation on a qubit Hamiltonian.

    Parameters
    ----------
    hamiltonian : PauliSum, optional
        The qubit Hamiltonian (Hartree).  May be omitted when the initial
        state is a checkpoint that stores one.
    n_evaluation_qubits : int
        Evaluation register size ``t``; the energy resolution is ``W / 2^t``
        for a window of width ``W``.
    energy_window : (E_lo, E_hi), optional
        Energies (Hartree) the phase ``[0, 1)`` spans.  Every eigenvalue that
        carries weight in the input must lie inside it, or its phase wraps
        around; the default is the rigorous bound
        ``c_I -/+ sum |c_P|`` from the Pauli coefficients, which is safe but
        wide -- a tighter window around the states of interest buys resolution
        at no cost in qubits.
    memory_policy : {"abort", "prompt", "ignore"}
        What to do when the estimated memory exceeds ``memory_fraction`` of
        the available memory (see :func:`check_memory`).
    memory_fraction : float
        That threshold (default 0.5).
    atomic_units : bool
        Report energies in Hartree instead of eV.
    keep_amplitudes : bool
        Keep the post-measurement system states on the result
        (:meth:`QPEResult.collapsed_state`); costs one ``2^(n+t)`` vector.
    verbose : bool
        Print the memory estimate and the result summary.
    """

    def __init__(self, hamiltonian: PauliSum | None = None,
                 n_evaluation_qubits: int = DEFAULT_EVALUATION_QUBITS,
                 energy_window=None, memory_policy: str = "abort",
                 memory_fraction: float = DEFAULT_MEMORY_FRACTION,
                 atomic_units: bool = False, keep_amplitudes: bool = True,
                 verbose: bool = True):
        if hamiltonian is not None and not isinstance(hamiltonian, PauliSum):
            raise TypeError("hamiltonian must be a PauliSum (qubit operator)")
        self.hamiltonian = hamiltonian
        self.n_evaluation_qubits = int(n_evaluation_qubits)
        if self.n_evaluation_qubits < 1:
            raise ValueError("n_evaluation_qubits must be >= 1")
        self.energy_window = (None if energy_window is None
                              else (float(energy_window[0]),
                                    float(energy_window[1])))
        if self.energy_window is not None \
                and self.energy_window[1] <= self.energy_window[0]:
            raise ValueError("energy_window must be (E_lo, E_hi) with E_hi > E_lo")
        if memory_policy not in MEMORY_POLICIES:
            raise ValueError(f"memory_policy must be one of {MEMORY_POLICIES}")
        self.memory_policy = memory_policy
        self.memory_fraction = float(memory_fraction)
        self.atomic_units = bool(atomic_units)
        self.keep_amplitudes = bool(keep_amplitudes)
        self.verbose = bool(verbose)
        #: The last :class:`QPEMemoryEstimate` checked.
        self.memory_estimate_result = None

    # -- inputs ------------------------------------------------------------ #

    def _resolve_input(self, initial_state):
        """``(psi, n_qubits, hamiltonian, problem)`` from any accepted input.

        ``initial_state`` is a :class:`~carcara.core.checkpoint.WavefunctionCheckpoint`,
        a path to one, a ``(n_qubits, reference_qubits, generators, parameters)``
        tuple, or a state vector.  ``problem`` is the preparation description
        when there is one (for :meth:`circuit`), else ``None``.
        """
        checkpoint = None
        if isinstance(initial_state, (str, os.PathLike)):
            checkpoint = WavefunctionCheckpoint.load(initial_state)
        elif isinstance(initial_state, WavefunctionCheckpoint):
            checkpoint = initial_state
        if checkpoint is not None:
            hamiltonian = (self.hamiltonian if self.hamiltonian is not None
                           else checkpoint.hamiltonian)
            if hamiltonian is None:
                raise ValueError("the checkpoint stores no Hamiltonian; pass "
                                 "one to QuantumPhaseEstimation(hamiltonian=)")
            n, ref, gens, theta, _ = checkpoint.problem()
            return None, n, hamiltonian, (n, ref, gens, theta)
        if isinstance(initial_state, tuple) and len(initial_state) == 4:
            n, ref, gens, theta = initial_state
            if self.hamiltonian is None:
                raise ValueError("pass hamiltonian= when the initial state is "
                                 "a preparation tuple")
            return None, int(n), self.hamiltonian, (int(n), list(ref),
                                                    list(gens), theta)
        psi = np.asarray(initial_state, dtype=complex).ravel()
        n = int(round(np.log2(psi.size)))
        if 2 ** n != psi.size:
            raise ValueError(f"a state vector needs 2^n amplitudes, got "
                             f"{psi.size}")
        if self.hamiltonian is None:
            raise ValueError("pass hamiltonian= when the initial state is a "
                             "state vector")
        norm = np.linalg.norm(psi)
        if norm == 0:
            raise ValueError("the initial state is zero")
        return psi / norm, n, self.hamiltonian, None

    def _window(self, hamiltonian: PauliSum) -> tuple[float, float]:
        """``(E_lo, E_hi)`` in Hartree, with a margin so no phase reaches 1."""
        if self.energy_window is not None:
            lo, hi = self.energy_window
        else:
            simplified = hamiltonian.simplify()
            n = hamiltonian.num_qubits
            identity = float(np.real(simplified.terms.get("I" * n, 0.0)))
            radius = sum(abs(c) for label, c in simplified.terms.items()
                         if label != "I" * n)
            lo, hi = identity - radius, identity + radius
            if hi <= lo:                        # a multiple of the identity
                lo, hi = identity - 0.5, identity + 0.5
        # One extra bin on top: an eigenvalue exactly at E_hi would read as
        # phase 1 = 0 and alias onto E_lo.
        width = (hi - lo) * (1.0 + 2.0 / 2 ** self.n_evaluation_qubits)
        return float(lo), float(lo + width)

    def memory_estimate(self, n_system_qubits: int) -> QPEMemoryEstimate:
        """The estimate :meth:`run` will check for this register size."""
        return qpe_memory_estimate(n_system_qubits, self.n_evaluation_qubits)

    # -- the simulation ---------------------------------------------------- #

    def run(self, initial_state) -> QPEResult:
        """Simulate QPE from ``initial_state`` and return the distribution.

        ``initial_state`` is a :class:`~carcara.core.checkpoint.WavefunctionCheckpoint`
        (or its path), a ``(n_qubits, reference_qubits, generators,
        parameters)`` preparation tuple, or a state vector.  The memory
        estimate is checked *before* the state is even prepared.
        """
        psi, n, hamiltonian, problem = self._resolve_input(initial_state)
        if hamiltonian.num_qubits != n:
            raise ValueError(f"a {hamiltonian.num_qubits}-qubit Hamiltonian for "
                             f"a {n}-qubit state")
        t = self.n_evaluation_qubits
        dense = n <= DENSE_LIMIT_QUBITS

        estimate = qpe_memory_estimate(n, t, dense=dense)
        self.memory_estimate_result = estimate
        check_memory(estimate, self.memory_policy, self.memory_fraction,
                     verbose=self.verbose)

        if psi is None:
            psi = prepare_state(*problem)
        lo, hi = self._window(hamiltonian)
        width = hi - lo
        N = 2 ** t

        exact = overlaps = None
        if dense:
            h = hamiltonian.to_matrix()
            h = 0.5 * (h + h.conj().T)
            w, V = np.linalg.eigh(h)
            c = V.conj().T @ psi
            # U^k psi = V diag(exp(2 pi i k phi)) V^dag psi, all k at once.
            phase = (w - lo) / width
            table = np.exp(2j * np.pi * np.outer(np.arange(N), phase))
            columns = (table * c) @ V.T
            exact, overlaps = w, np.abs(c) ** 2
        else:
            from scipy.sparse.linalg import expm_multiply
            generator = (2j * np.pi / width) * (
                hamiltonian.to_sparse_matrix()
                - lo * PauliSum.identity(n).to_sparse_matrix())
            columns = np.empty((N, 2 ** n), dtype=complex)
            columns[0] = psi
            for k in range(1, N):
                columns[k] = expm_multiply(generator, columns[k - 1])
        # The inverse QFT on the evaluation register is a DFT along k.
        amplitudes = np.fft.fft(columns, axis=0) / N
        del columns
        probabilities = np.sum(np.abs(amplitudes) ** 2, axis=1)
        phases = np.arange(N) / N
        energies_ha = lo + width * phases
        order = np.argsort(probabilities)[::-1]
        peaks = [(float(energies_ha[m]), float(probabilities[m]))
                 for m in order[:max(8, 2 ** min(t, 4))]
                 if probabilities[m] > 0]

        unit = "Ha" if self.atomic_units else "eV"
        result = QPEResult(
            energies=energies_ha, probabilities=probabilities, phases=phases,
            resolution=width / N, energy_window=(lo, hi),
            n_system_qubits=n, n_evaluation_qubits=t, peaks=peaks,
            exact_energies=exact, overlaps=overlaps, energy_unit="Ha",
            _amplitudes=amplitudes if self.keep_amplitudes else None)
        result = result.in_units(unit)
        if self.verbose:
            print(result.summary())
        return result

    # -- circuit export ---------------------------------------------------- #

    def circuit(self, initial_state, provider=None):
        """The QPE experiment as a Qiskit circuit.

        The checkpoint's (or tuple's) state preparation on the system wires,
        then :class:`qiskit.circuit.library.PhaseEstimation` with an exact
        :class:`~qiskit.circuit.library.PauliEvolutionGate` for ``U``.  Wires
        ``0 .. t-1`` are the evaluation register (Qiskit's little-endian order:
        the integer read off them is the bin ``m`` of :class:`QPEResult`),
        wires ``t .. t+n-1`` the system, with Carcará qubit ``k`` on wire
        ``t + n - 1 - k`` as every provider lays it out.
        """
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import PauliEvolutionGate, PhaseEstimation
        from qiskit.synthesis import MatrixExponential

        psi, n, hamiltonian, problem = self._resolve_input(initial_state)
        if problem is None:
            raise ValueError("a circuit needs a state *preparation*: pass a "
                             "checkpoint or a (n_qubits, reference_qubits, "
                             "generators, parameters) tuple, not amplitudes")
        if provider is None:
            from ..backends.providers import QiskitProvider
            provider = QiskitProvider()
        t = self.n_evaluation_qubits
        lo, hi = self._window(hamiltonian)
        shifted = hamiltonian + PauliSum.identity(n) * (-lo)
        # exp(-i op time) with time = -2 pi / W is U = exp(2 pi i (H - E_lo)/W).
        unitary = PauliEvolutionGate(shifted.to_sparse_pauli_op(),
                                     time=-2.0 * np.pi / (hi - lo),
                                     synthesis=MatrixExponential())
        preparation = provider.build(*problem)
        qc = QuantumCircuit(t + n, name="qpe")
        qc.compose(preparation, qubits=list(range(t, t + n)), inplace=True)
        qc.compose(PhaseEstimation(t, unitary), qubits=list(range(t + n)),
                   inplace=True)
        return qc

    @staticmethod
    def evaluation_distribution(circuit_or_state, n_evaluation_qubits: int
                                ) -> np.ndarray:
        """``P(m)`` of a Qiskit circuit / state, in :class:`QPEResult` order."""
        from qiskit.quantum_info import Statevector

        state = (circuit_or_state if isinstance(circuit_or_state, Statevector)
                 else Statevector(circuit_or_state))
        # PhaseEstimation writes the phase's most significant bit on wire 0,
        # so the bin index m is read with the wire order reversed.
        wires = list(range(int(n_evaluation_qubits)))[::-1]
        return np.asarray(state.probabilities(qargs=wires), dtype=float)


def phase_estimation(initial_state, hamiltonian: PauliSum | None = None,
                     **options) -> QPEResult:
    """One-call QPE: ``phase_estimation("checkpoint.json", n_evaluation_qubits=10)``."""
    return QuantumPhaseEstimation(hamiltonian, **options).run(initial_state)
