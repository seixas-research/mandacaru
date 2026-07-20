# -*- coding: utf-8 -*-
# file: test/test_backend_providers.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Multi-backend circuit providers: Qiskit, Amazon Braket and Cirq.

The three SDKs build the *same* unitary from the same Pauli-rotation
decomposition, so every test here is ultimately an **equivalence** check: the
executed circuit must reproduce the internal state-vector backend (and therefore
the other providers) to machine precision, both at the ansatz level and driven
end-to-end through ADAPT-VQE.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from carcara.algorithms import ADAPTVQE, VQE
from carcara.backends.providers import (BACKEND_PROVIDERS, basis_state_index,
                                        build_provider, normalize_provider,
                                        pauli_rotations, provider_available)
from carcara.circuits import UCCSD, AdaptAnsatz
from carcara.circuits.pools import build_pool
from carcara.core import PauliSum

PROVIDERS = [
    pytest.param(name,
                 marks=pytest.mark.skipif(not provider_available(name),
                                          reason=f"{name} SDK not installed"))
    for name in BACKEND_PROVIDERS
]


@pytest.fixture(scope="module")
def h2_cache(tmp_path_factory):
    """H2 qubit Hamiltonian on disk, so each provider run skips the integrals."""
    path = str(tmp_path_factory.mktemp("ham") / "h2.parquet")
    atoms = Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
                  cell=[[6, 0, 0], [0, 6, 0], [0, 0, 6]], pbc=True)
    atoms.calc = ADAPTVQE(pool="fermionic", basis="FAO", h=0.4, verbose=False,
                          max_iterations=1, save_hamiltonian=path)
    atoms.get_total_energy()
    return path


# --------------------------------------------------------------------------- #
# Registry and the provider-independent decomposition.
# --------------------------------------------------------------------------- #

class TestProviderRegistry:
    @pytest.mark.parametrize("spelling,expected", [
        ("qiskit", "qiskit"), ("IBM", "qiskit"),
        ("braket", "braket"), ("amazon-braket", "braket"), ("AWS", "braket"),
        ("cirq", "cirq"), ("Google", "cirq"),
    ])
    def test_normalize_provider(self, spelling, expected):
        assert normalize_provider(spelling) == expected

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="unknown backend_provider"):
            normalize_provider("rigetti")

    @pytest.mark.parametrize("name", PROVIDERS)
    def test_build_provider_is_cached(self, name):
        assert build_provider(name) is build_provider(name)
        assert build_provider(name).name == name

    def test_driver_rejects_unknown_provider(self):
        with pytest.raises(ValueError, match="unknown backend_provider"):
            ADAPTVQE(backend_provider="rigetti")

    def test_execute_circuits_defaults_per_provider(self):
        assert ADAPTVQE(backend_provider="qiskit").execute_circuits is False
        assert ADAPTVQE(backend_provider="braket").execute_circuits is True
        assert ADAPTVQE(backend_provider="cirq").execute_circuits is True
        # ... and is overridable in both directions.
        assert ADAPTVQE(backend_provider="qiskit",
                        execute_circuits=True).execute_circuits is True
        assert ADAPTVQE(backend_provider="cirq",
                        execute_circuits=False).execute_circuits is False


class TestPauliRotationDecomposition:
    def test_extracts_imaginary_coefficients(self):
        generator = PauliSum({"XY": 0.5j, "YX": -0.5j, "II": 0.0})
        assert pauli_rotations(generator) == [("XY", 0.5), ("YX", -0.5)]

    def test_drops_global_phase_identity(self):
        assert pauli_rotations(PauliSum({"II": 3.0j})) == []

    def test_rejects_non_anti_hermitian(self):
        with pytest.raises(ValueError, match="anti-Hermitian"):
            pauli_rotations(PauliSum({"XY": 0.5}))

    def test_basis_state_index(self):
        vec = np.zeros(8, dtype=complex)
        vec[5] = 1.0
        assert basis_state_index(vec) == 5
        vec[2] = 1.0                                  # a superposition
        assert basis_state_index(vec) is None


# --------------------------------------------------------------------------- #
# Equivalence: circuit execution vs the internal state-vector backend.
# --------------------------------------------------------------------------- #

class TestAnsatzEquivalence:
    @staticmethod
    def _ansatz(pool_name, provider=None, n_ops=3):
        pool = build_pool(pool_name, 2, (1, 1))
        ansatz = AdaptAnsatz(4, pool.occupied_orbitals, "jordan_wigner",
                             provider=provider)
        for op in pool.operators()[:n_ops]:
            ansatz.append(op)
        return ansatz

    @pytest.mark.parametrize("name", PROVIDERS)
    @pytest.mark.parametrize("pool_name", ["fermionic", "qubit", "qeb", "ceo"])
    def test_adapt_ansatz_state_matches_reference(self, name, pool_name):
        theta = np.array([0.31, -0.72, 0.45])
        reference = self._ansatz(pool_name).state(theta)
        executed = self._ansatz(pool_name, build_provider(name)).state(theta)
        # Exact, not just up to a global phase: the decomposition adds none.
        assert np.allclose(executed, reference, atol=1e-10)

    @pytest.mark.parametrize("name", PROVIDERS)
    def test_providers_agree_with_each_other(self, name):
        theta = np.array([0.2, 0.9, -1.3])
        qiskit_state = self._ansatz("qeb", build_provider("qiskit")).state(theta)
        other = self._ansatz("qeb", build_provider(name)).state(theta)
        assert np.allclose(other, qiskit_state, atol=1e-10)

    @pytest.mark.parametrize("name", PROVIDERS)
    def test_state_is_normalized(self, name):
        state = self._ansatz("fermionic", build_provider(name)).state(
            np.array([0.4, 0.4, 0.4]))
        assert np.vdot(state, state).real == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("name", PROVIDERS)
    def test_evolve_handles_a_stack_of_determinants(self, name):
        ansatz = self._ansatz("qeb", build_provider(name))
        plain = self._ansatz("qeb")
        refs = np.zeros((16, 2), dtype=complex)
        refs[int(np.argmax(np.abs(plain.reference_state()))), 0] = 1.0
        refs[6, 1] = 1.0
        theta = np.array([0.3, -0.4, 0.8])
        assert np.allclose(ansatz.evolve(theta, refs),
                           plain.evolve(theta, refs), atol=1e-10)

    @pytest.mark.parametrize("name", PROVIDERS)
    def test_superposition_reference_is_rejected(self, name):
        ansatz = self._ansatz("qeb", build_provider(name))
        refs = np.zeros(16, dtype=complex)
        refs[3] = refs[5] = 1.0 / np.sqrt(2)
        with pytest.raises(ValueError, match="computational-basis"):
            ansatz.evolve(np.array([0.1, 0.1, 0.1]), refs)

    @pytest.mark.parametrize("name", PROVIDERS)
    def test_trotter_uccsd_matches_the_matrix_backend(self, name):
        theta = np.array([0.21, -0.34, 0.55])
        plain = UCCSD(2, (1, 1), trotter=True)
        assert plain.num_parameters == 3
        executed = UCCSD(2, (1, 1), trotter=True, provider=build_provider(name))
        assert np.allclose(executed.state(theta), plain.state(theta), atol=1e-10)

    @pytest.mark.parametrize("name", PROVIDERS)
    def test_uccsd_requires_trotter_for_circuit_execution(self, name):
        with pytest.raises(ValueError, match="Trotter"):
            UCCSD(2, (1, 1), trotter=False, provider=build_provider(name))


class TestProviderProfiling:
    @pytest.mark.parametrize("name", PROVIDERS)
    def test_profile_reports_positive_counts(self, name):
        pool = build_pool("qeb", 2, (1, 1))
        generators = [op.generator for op in pool.operators()[:2]]
        counts = build_provider(name).profile(4, [0, 2], generators)
        assert counts["cnot_count"] > 0
        assert counts["depth"] > 0
        assert counts["total_gates"] >= counts["cnot_count"]


# --------------------------------------------------------------------------- #
# End-to-end: the same ADAPT-VQE run on all three SDKs.
# --------------------------------------------------------------------------- #

class TestDriverAcrossProviders:
    @pytest.mark.parametrize("name", PROVIDERS)
    def test_adapt_vqe_energy_is_provider_independent(self, name, h2_cache):
        reference = ADAPTVQE(pool="qeb", load_hamiltonian=h2_cache,
                             verbose=False, max_iterations=4).run()
        executed = ADAPTVQE(pool="qeb", load_hamiltonian=h2_cache, verbose=False,
                            max_iterations=4, backend_provider=name,
                            execute_circuits=True).run()
        assert executed.optimal_energy == pytest.approx(
            reference.optimal_energy, abs=1e-8)
        # The selected *set* must match; the order of symmetry-degenerate
        # operators can be decided by optimizer noise below 1e-9.
        assert set(executed.operators) == set(reference.operators)

    @pytest.mark.parametrize("name", PROVIDERS)
    def test_vqe_energy_is_provider_independent(self, name, h2_cache):
        reference = VQE(load_hamiltonian=h2_cache, verbose=False).run()
        executed = VQE(load_hamiltonian=h2_cache, verbose=False,
                       backend_provider=name, execute_circuits=True).run()
        # The circuit path is the Trotter product form, exact for H2's single
        # (double) excitation but only close for the general cluster operator.
        assert executed.optimal_energy == pytest.approx(
            reference.optimal_energy, abs=1e-6)

    @pytest.mark.parametrize("name", PROVIDERS)
    def test_verbose_header_reports_the_provider(self, name, h2_cache, capsys):
        ADAPTVQE(pool="qeb", load_hamiltonian=h2_cache, verbose=True,
                 max_iterations=1, backend_provider=name,
                 execute_circuits=True).run()
        out = capsys.readouterr().out
        assert f"backend provider: {name}" in out
        assert "circuit execution: True" in out
