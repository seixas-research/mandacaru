"""Python descriptions of native solver states for reporting and SDK adapters."""

from __future__ import annotations

import json


class AnsatzView:
    """A parameterized native ansatz exposed as full-register state vectors."""

    def __init__(self, problem, indices, product_form):
        self.problem = problem
        self.indices = tuple(indices)
        self.product_form = bool(product_form)
        self.n_qubits = problem.num_qubits
        self.num_particles = tuple(problem.num_particles)
        self.sector = None  # the returned vectors use the complete register

    @property
    def num_parameters(self):
        return len(self.indices)

    @property
    def circuit_serializable(self):
        return self.product_form and not self.problem.exact_hva

    def state(self, theta):
        return self.problem.state(
            list(theta), indices=self.indices, trotter=self.product_form
        )

    def reference_state(self):
        return self.problem.state([], indices=[], trotter=True)

    def reference_qubits(self):
        return self.problem.reference_qubits

    @property
    def pauli_generators(self):
        from ..core.mapping import PauliSum

        terms = self.problem.pool_terms
        return [
            PauliSum(
                {label: complex(re, im) for label, re, im in terms[i]},
                num_qubits=self.n_qubits,
            )
            for i in self.indices
        ]


class SolverView:
    """Read-only descriptions of an optimization performed by the Rust solver."""

    def __init__(self, problem, result, checkpoint, product_form):
        self.problem = problem
        self.n_qubits = problem.num_qubits
        self.num_particles = tuple(problem.num_particles)
        self.ansatz = AnsatzView(problem, result["indices"], product_form)
        self._checkpoint = checkpoint

    @property
    def checkpoint(self):
        from ..core.checkpoint import WavefunctionCheckpoint

        if self._checkpoint is None:
            return None
        return WavefunctionCheckpoint.from_payload(json.loads(self._checkpoint.json))

    @property
    def metrics(self):
        from types import SimpleNamespace
        from ..circuits.profiling import CircuitMetrics, profile_ansatz

        if not self.ansatz.circuit_serializable:
            return CircuitMetrics(None, None, self.ansatz.num_parameters)
        generators = (
            self.checkpoint.generators
            if self.problem.exact_hva is False and self._checkpoint is not None
            else self.ansatz.pauli_generators
        )
        metrics = profile_ansatz(
            self.n_qubits,
            tuple(self.ansatz.reference_qubits()),
            [SimpleNamespace(generator=g) for g in generators],
        )
        metrics.num_operators = self.ansatz.num_parameters
        return metrics
