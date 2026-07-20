# -*- coding: utf-8 -*-
# file: circuits/profiling.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Hardware-cost profiling of a prepared ansatz circuit.

An ansatz built as ``|HF>`` followed by a product of excitation exponentials
``prod_k exp(A_k)`` is compiled to a native ``{CNOT, U}`` gate set with Qiskit and
its CNOT count and depth are measured (:class:`CircuitMetrics`).  Profiling is
**optional**: when Qiskit is unavailable the counts come back ``None`` so the rest
of the pipeline still runs.  This lives in :mod:`carcara.circuits` (independent of
any driver) so any algorithm growing such an ansatz can profile it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.mapping import PauliSum
from .pools import PoolOperator


@dataclass
class CircuitMetrics:
    """Structural cost of a compiled ansatz circuit."""

    cnot_count: int | None            # number of CNOT (two-qubit) gates
    depth: int | None                 # circuit depth in the native gate set
    num_operators: int                # generators (parameters) in the ansatz
    num_1q_gates: int | None = None   # single-qubit (``u``) gates
    total_gates: int | None = None    # all gates in the native gate set

    def __repr__(self) -> str:
        return (f"CircuitMetrics(cnots={self.cnot_count}, depth={self.depth}, "
                f"gates={self.total_gates}, n_ops={self.num_operators})")


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
                   operators: list[PoolOperator],
                   provider=None) -> CircuitMetrics:
    """Compile ``|HF>`` + product of ``exp(A_k)`` to ``{cx, u}`` and count cost.

    Returns a :class:`CircuitMetrics`; ``cnot_count`` / ``depth`` are ``None`` if
    the SDK is not installed.

    With ``provider`` given (a :class:`~carcara.backends.providers.CircuitProvider`)
    the ansatz is compiled and counted with **that** SDK -- Qiskit, Amazon Braket
    or Cirq -- from the explicit Pauli-rotation decomposition.  Counts differ
    between providers because only Qiskit re-optimizes the circuit during
    transpilation; the *unitary* is identical either way.
    """
    if provider is not None:
        try:
            counts = provider.profile(n_qubits, list(occupied),
                                      [op.generator for op in operators])
        except Exception:
            return CircuitMetrics(None, None, len(operators))
        return CircuitMetrics(cnot_count=counts["cnot_count"],
                              depth=counts["depth"],
                              num_operators=len(operators),
                              num_1q_gates=counts["num_1q_gates"],
                              total_gates=counts["total_gates"])
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
                          num_operators=len(operators),
                          num_1q_gates=int(counts.get("u", 0)),
                          total_gates=int(sum(counts.values())))
