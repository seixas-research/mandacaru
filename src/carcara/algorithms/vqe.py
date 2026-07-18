# -*- coding: utf-8 -*-
# file: algorithms/vqe.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Variational Quantum Eigensolver (VQE).

:class:`VQE` minimizes the Rayleigh quotient
:math:`E(\vec\theta) = \langle\psi(\vec\theta)|H|\psi(\vec\theta)\rangle` over the
parameters of a variational ansatz, returning the (approximate) ground-state
energy and optimal parameters.

This reference implementation is an **exact state-vector simulator**: the qubit
Hamiltonian is materialized as a dense matrix and the ansatz produces the exact
:math:`2^N` state vector, so the energy is the noiseless expectation value.  It
is meant for validating the full pipeline (integrals -> Hamiltonian -> mapping ->
ansatz -> optimizer) against exact diagonalization on small molecules; a
shot-based estimator on real hardware is a later, drop-in replacement for the
energy evaluation.

The class mirrors :class:`~carcara.algorithms.adapt_vqe.ADAPTVQE`: the
``optimizer`` may be named by string, a ``verbose`` run prints the qubit
Hamiltonian as Pauli strings to standard output, and the run returns a
:class:`VQEResult` shaped like
:class:`~carcara.algorithms.adapt_vqe.ADAPTVQEResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.mapping import Fermion, PauliSum
from ..optimizers.optim import Optimizer, resolve_optimizer
from .adapt_vqe import format_pauli_sum


@dataclass
class VQEResult:
    """Result of a :class:`VQE` run.

    Shaped like :class:`~carcara.algorithms.adapt_vqe.ADAPTVQEResult`: it carries
    the optimal energy/parameters, the reference energy, the evaluation count and
    the full cost history, and exposes the same convenience views
    (:attr:`num_parameters`, :attr:`energy_history`).
    """

    optimal_energy: float                 # minimized energy <psi|H|psi>
    optimal_parameters: np.ndarray        # optimal ansatz parameters
    reference_energy: float               # energy of the ansatz reference state
    num_evaluations: int = 0              # cost-function evaluations
    history: list[float] = field(default_factory=list)   # cost per evaluation
    success: bool = True

    @property
    def num_parameters(self) -> int:
        return len(self.optimal_parameters)

    @property
    def energy_history(self) -> list[float]:
        return list(self.history)

    @property
    def correlation_energy(self) -> float:
        """Energy lowered relative to the reference (``E - E_ref``)."""
        return self.optimal_energy - self.reference_energy

    def __repr__(self) -> str:
        return (f"VQEResult(energy={self.optimal_energy:.6f}, "
                f"n_params={self.num_parameters}, "
                f"nfev={self.num_evaluations}, success={self.success})")


class VQE:
    """Variational Quantum Eigensolver on an exact state-vector backend.

    Parameters
    ----------
    hamiltonian : PauliSum or Fermion
        The qubit Hamiltonian, or a fermionic Hamiltonian which is mapped with
        ``ansatz.mapping`` to match the ansatz's encoding.
    ansatz : object
        A parameterized ansatz exposing ``num_parameters``, ``n_qubits``,
        ``state(theta) -> np.ndarray`` and ``reference_state()`` (e.g.
        :class:`~carcara.circuits.ansatz.UCCSD`).
    optimizer : str or Optimizer
        Classical optimizer.  Either a method name -- one of ``"COBYLA"``
        (default), ``"Nelder-Mead"``, ``"BFGS"`` -- or a pre-built
        :class:`~carcara.optimizers.optim.Optimizer` instance.
    verbose : bool
        Print the quantum-simulation trace to standard output (default ``True``):
        the qubit Hamiltonian as Pauli strings before the optimization, and the
        optimized energy after it.
    """

    _OPTIMIZERS = ("COBYLA", "Nelder-Mead", "BFGS")

    def __init__(self, hamiltonian, ansatz,
                 optimizer: str | Optimizer = "COBYLA",
                 verbose: bool = True):
        self.ansatz = ansatz
        self.verbose = bool(verbose)
        self.optimizer = resolve_optimizer(optimizer, allowed=self._OPTIMIZERS)

        qubit_h = self._as_pauli_sum(hamiltonian, ansatz)
        if qubit_h.num_qubits != ansatz.n_qubits:
            raise ValueError(
                f"Hamiltonian acts on {qubit_h.num_qubits} qubits but the ansatz "
                f"has {ansatz.n_qubits}")
        self.hamiltonian = qubit_h
        self.n_qubits = ansatz.n_qubits
        self.mapping = getattr(ansatz, "mapping", "jordan_wigner")
        # Hermitize away rounding noise; the expectation value is then real.
        h = qubit_h.to_matrix()
        self._h_matrix = 0.5 * (h + h.conj().T)

    @staticmethod
    def _as_pauli_sum(hamiltonian, ansatz) -> PauliSum:
        if isinstance(hamiltonian, PauliSum):
            return hamiltonian
        if isinstance(hamiltonian, Fermion):
            mapping = getattr(ansatz, "mapping", "jordan_wigner")
            return hamiltonian.map_to_qubits(mapping, n_modes=ansatz.n_qubits)
        raise TypeError("hamiltonian must be a PauliSum or Fermion")

    def energy(self, theta) -> float:
        """Expectation value ``<psi(theta)| H |psi(theta)>`` (real)."""
        psi = self.ansatz.state(theta)
        return float(np.real(np.vdot(psi, self._h_matrix @ psi)))

    def reference_energy(self) -> float:
        """Energy of the ansatz reference state (all parameters zero)."""
        psi = self.ansatz.reference_state()
        return float(np.real(np.vdot(psi, self._h_matrix @ psi)))

    def run(self, initial_parameters=None) -> VQEResult:
        """Optimize the parameters and return the ground-state estimate.

        The optimizer and the ``verbose`` flag come from the constructor; the
        only argument is ``initial_parameters`` (which the constructor does not
        carry), defaulting to all-zero (the reference state).
        """
        n = self.ansatz.num_parameters
        x0 = (np.zeros(n) if initial_parameters is None
              else np.asarray(initial_parameters, dtype=float).ravel())
        if x0.size != n:
            raise ValueError(f"expected {n} initial parameters, got {x0.size}")

        ref_energy = self.reference_energy()
        if self.verbose:
            self._print_header(ref_energy)

        result = self.optimizer.minimize(self.energy, x0)
        vqe_result = VQEResult(
            optimal_energy=result.fun,
            optimal_parameters=result.x,
            reference_energy=ref_energy,
            num_evaluations=result.nfev,
            history=result.history,
            success=result.success)

        if self.verbose:
            self._print_summary(vqe_result)
        return vqe_result

    # -- standard-output trace ------------------------------------------- #

    def _print_header(self, ref_energy: float) -> None:
        """Print the run banner and the qubit Hamiltonian as Pauli strings."""
        rule = "=" * 70
        print(rule)
        print(f"VQE  |  mapping: {self.mapping}  |  {self.n_qubits} qubits  |  "
              f"optimizer: {self.optimizer.method}")
        print(f"ansatz: {type(self.ansatz).__name__}  |  "
              f"parameters: {self.ansatz.num_parameters}")
        print(rule)
        n_terms = len(self.hamiltonian.simplify().terms)
        print(f"Qubit Hamiltonian ({n_terms} Pauli terms):")
        print(format_pauli_sum(self.hamiltonian))
        print(f"Reference (all-zero) energy = {ref_energy:+.8f} Ha")
        print(rule)

    def _print_summary(self, result: VQEResult) -> None:
        """Print the closing summary line."""
        rule = "=" * 70
        print(rule)
        status = "converged" if result.success else "did not converge"
        print(f"VQE finished ({status}): "
              f"E = {result.optimal_energy:+.8f} Ha, "
              f"{result.num_parameters} parameters, {result.num_evaluations} "
              f"evaluations")
        print(rule)
