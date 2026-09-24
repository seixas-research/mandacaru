# -*- coding: utf-8 -*-
# file: test/backends/test_fractional_gates.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Fractional gates, and counting two-qubit gates without missing any.

These two belong together.  Asking Runtime for a processor's fractional
instruction set changes its entangler from a fixed ``CZ`` to a parameterized
``RZZ``, and the gate counter used to work from a list of names that did not
contain ``rzz`` -- so enabling the option would have made a deep circuit report
*zero* two-qubit gates and a perfect expected fidelity.  Counting by arity
cannot miss an entangler whatever it is called.
"""

from __future__ import annotations

import pytest

from mandacaru.backends.measurement import (NOMINAL_2Q_ERROR,
                                            two_qubit_gate_count)
from mandacaru.backends.providers import QiskitProvider


@pytest.fixture(scope="module")
def mixed_circuit():
    """A circuit carrying one of each IBM entangler, plus non-gate instructions."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(4)
    qc.cx(0, 1)
    qc.cz(1, 2)
    qc.rzz(0.3, 0, 2)        # the fractional entangler
    qc.ecr(2, 3)
    qc.h(0)
    qc.rz(0.1, 1)
    qc.barrier()
    qc.measure_all()
    return qc


class TestCountingByArity:
    def test_it_counts_every_entangler(self, mixed_circuit):
        assert two_qubit_gate_count(mixed_circuit) == 4

    def test_the_old_name_list_would_have_missed_the_fractional_one(
            self, mixed_circuit):
        # The regression this exists to prevent, stated as the measurement.
        old = sum(v for k, v in mixed_circuit.count_ops().items()
                  if k in ("cz", "cx", "ecr", "cnot"))
        assert old == 3
        assert two_qubit_gate_count(mixed_circuit) == 4

    def test_barriers_and_measurements_are_not_gates(self):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.barrier()
        qc.measure_all()
        assert two_qubit_gate_count(qc) == 0

    def test_single_qubit_gates_are_not_counted(self):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.x(1)
        qc.rz(0.2, 0)
        assert two_qubit_gate_count(qc) == 0

    def test_an_empty_circuit_counts_zero(self):
        from qiskit import QuantumCircuit

        assert two_qubit_gate_count(QuantumCircuit(3)) == 0

    def test_a_missed_entangler_would_claim_a_perfect_fidelity(self,
                                                              mixed_circuit):
        # Why the counter matters rather than merely being tidy: the fidelity
        # estimate is (1 - e)^n2q, so undercounting reads as a better circuit.
        honest = (1.0 - NOMINAL_2Q_ERROR) ** two_qubit_gate_count(mixed_circuit)
        undercounted = (1.0 - NOMINAL_2Q_ERROR) ** 3
        assert undercounted > honest


class TestTheFractionalOption:
    def test_it_is_off_by_default(self):
        assert QiskitProvider(device="statevector").enable_fractional_gates \
            is False

    def test_it_is_recorded_when_asked_for(self):
        provider = QiskitProvider(device="ibm_kingston", shots=1024,
                                  enable_fractional_gates=True)
        assert provider.enable_fractional_gates is True

    def test_it_reaches_the_service_call(self, monkeypatch):
        # No account is touched: a stub service records what it was asked for.
        seen = {}

        class Stub:
            def backend(self, name, **kwargs):
                seen["name"] = name
                seen.update(kwargs)
                return object()

        provider = QiskitProvider(device="ibm_kingston",
                                  enable_fractional_gates=True)
        provider._ibm_backend(Stub())
        assert seen["name"] == "ibm_kingston"
        assert seen["use_fractional_gates"] is True

    def test_the_least_busy_path_forwards_it_too(self):
        seen = {}

        class Stub:
            def least_busy(self, **kwargs):
                seen.update(kwargs)
                return object()

        provider = QiskitProvider(device="ibm-quantum",
                                  enable_fractional_gates=True)
        provider._ibm_backend(Stub())
        assert seen["use_fractional_gates"] is True

    def test_it_is_not_forwarded_when_off(self):
        seen = {}

        class Stub:
            def backend(self, name, **kwargs):
                seen.update(kwargs)
                return object()

        QiskitProvider(device="ibm_fez")._ibm_backend(Stub())
        assert seen["use_fractional_gates"] is False


class TestProfileUsesTheDeviceISA:
    def test_a_local_provider_labels_its_proxy_basis(self):
        from mandacaru.core.mapping import PauliSum

        provider = QiskitProvider(device="statevector")
        generator = PauliSum({"XYII": 0 + 0.5j, "YXII": 0 - 0.5j})
        counts = provider.profile(4, [0, 1], [generator])
        assert "proxy" in counts["isa"]
        assert counts["two_qubit_gates"] == counts["cnot_count"]

    def test_a_fake_device_reports_its_own_name(self):
        # No ordering is asserted between the two counts, and that is the point.
        # They are *different numbers for different instruction sets*: the device
        # has a coupling map that costs routing swaps, and the provider
        # transpiles it at its own optimization_level (3) while the proxy uses 1,
        # so the ISA count can come out either side of the proxy one.  Measured
        # on a 4-qubit generator: 4 against the proxy basis, 2 on fake_manila.
        # What must be true is that each count says which basis it belongs to,
        # because a gate count without that label is not comparable to anything.
        from mandacaru.core.mapping import PauliSum

        generator = PauliSum({"XYII": 0 + 0.5j, "YXII": 0 - 0.5j})
        proxy = QiskitProvider(device="statevector").profile(4, [0, 1],
                                                            [generator])
        isa = QiskitProvider(device="fake_manila").profile(4, [0, 1],
                                                          [generator])
        assert "proxy" in proxy["isa"]
        assert "proxy" not in isa["isa"]
        assert isa["isa"] != proxy["isa"]
        assert isa["two_qubit_gates"] > 0

    def test_the_fractional_basis_needs_fewer_entanglers(self):
        """The measured reason ``enable_fractional_gates`` is worth having.

        Transpiling the same circuit to a Heron coupling map with ``rzz`` in the
        basis instead of ``cz``.  The saving is real and **modest** -- 11 to 18 %
        on the circuits measured, shrinking with size, because routing swaps on a
        heavy-hex lattice dominate the local ZZ decomposition.  Worth recording
        as a test because it is easy to assume the parameterized entangler halves
        the count, and it does not.
        """
        import numpy as np
        from qiskit import QuantumCircuit, transpile

        from mandacaru.circuits.profiling import _pauli_evolution_gate
        from mandacaru.core.mapping import PauliSum

        backend = QiskitProvider._fake_backend("fake_kingston")
        rng = np.random.default_rng(0)
        n = 12
        qc = QuantumCircuit(n)
        for q in range(0, n, 2):
            qc.x(q)
        for _ in range(6):
            picks = sorted(rng.choice(n, size=4, replace=False))
            terms = {}
            for k, letters in enumerate(("XXXY", "XXYX", "XYXX", "YXXX",
                                         "YYYX", "YYXY", "YXYY", "XYYY")):
                label = ["I"] * n
                for q, letter in zip(picks, letters):
                    label[q] = letter
                terms["".join(label)] = complex(0, (1 if k < 4 else -1) * 0.125)
            gate = _pauli_evolution_gate(PauliSum(terms, num_qubits=n))
            if gate is not None:
                qc.append(gate, range(n))

        counts = {}
        for name, basis in (("cz", ["cz", "rz", "sx", "x"]),
                            ("rzz", ["rzz", "rz", "sx", "x"])):
            compiled = transpile(qc, basis_gates=basis,
                                 coupling_map=backend.coupling_map,
                                 optimization_level=3, seed_transpiler=7)
            counts[name] = two_qubit_gate_count(compiled)
        assert counts["rzz"] < counts["cz"], counts
        # Modest, not halved: assert the size of the effect, not just its sign.
        assert counts["rzz"] > 0.7 * counts["cz"], counts
