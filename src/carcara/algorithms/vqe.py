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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.mapping import Fermion, PauliSum
from ..optimizers.optim import Optimizer


@dataclass
class VQEResult:
    """Result of a :class:`VQE` run."""

    optimal_energy: float                 # minimized energy <psi|H|psi>
    optimal_parameters: np.ndarray        # optimal ansatz parameters
    reference_energy: float               # energy of the ansatz reference state
    num_evaluations: int                  # cost-function evaluations
    history: list[float]                  # energy per evaluation
    success: bool = True

    def __repr__(self) -> str:
        return (f"VQEResult(optimal_energy={self.optimal_energy:.6f}, "
                f"n_params={len(self.optimal_parameters)}, "
                f"nfev={self.num_evaluations})")


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
    optimizer : Optimizer, optional
        Classical optimizer (default :class:`~carcara.optimizers.optim.Optimizer`,
        i.e. COBYLA).
    """

    def __init__(self, hamiltonian, ansatz, optimizer: Optimizer | None = None):
        self.ansatz = ansatz
        self.optimizer = optimizer or Optimizer()

        qubit_h = self._as_pauli_sum(hamiltonian, ansatz)
        if qubit_h.num_qubits != ansatz.n_qubits:
            raise ValueError(
                f"Hamiltonian acts on {qubit_h.num_qubits} qubits but the ansatz "
                f"has {ansatz.n_qubits}")
        self.hamiltonian = qubit_h
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

        ``initial_parameters`` defaults to all-zero (the reference state).
        """
        n = self.ansatz.num_parameters
        x0 = (np.zeros(n) if initial_parameters is None
              else np.asarray(initial_parameters, dtype=float).ravel())
        if x0.size != n:
            raise ValueError(f"expected {n} initial parameters, got {x0.size}")

        result = self.optimizer.minimize(self.energy, x0)
        return VQEResult(
            optimal_energy=result.fun,
            optimal_parameters=result.x,
            reference_energy=self.reference_energy(),
            num_evaluations=result.nfev,
            history=result.history,
            success=result.success)
