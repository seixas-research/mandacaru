# -*- coding: utf-8 -*-
# file: test/test_backend_providers.py

# This code is part of Mandacaru.
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

import re

import numpy as np
from mandacaru.units import HARTREE_TO_EV
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.backends.providers import (BACKEND_PROVIDERS, basis_state_index,
                                          build_provider, normalize_provider,
                                          pauli_rotations, provider_available)
from mandacaru.circuits.pools import build_pool
from mandacaru.core import PauliSum

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
                  cell=[[6, 0, 0], [0, 6, 0], [0, 0, 6]], pbc=False)
    atoms.calc = Mandacaru(method="adapt-vqe", pool="fermionic", basis="HAO",
                           grid_spacing=0.4, trace=False, max_iterations=1,
                           save_hamiltonian=path)
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
    def test_build_provider_names_its_family(self, name):
        assert build_provider(name).name == name

    def test_driver_rejects_unknown_provider(self):
        with pytest.raises(ValueError, match="unknown backend_provider"):
            Mandacaru(method="adapt-vqe", backend_provider="rigetti")

    def test_execute_circuits_defaults_per_provider(self):
        assert Mandacaru(method="adapt-vqe", backend_provider="qiskit")._execute_circuits is False
        assert Mandacaru(method="adapt-vqe", backend_provider="braket")._execute_circuits is True
        assert Mandacaru(method="adapt-vqe", backend_provider="cirq")._execute_circuits is True
        # ... and is overridable in both directions.
        assert Mandacaru(method="adapt-vqe", backend_provider="qiskit",
                         execute_circuits=True)._execute_circuits is True
        assert Mandacaru(method="adapt-vqe", backend_provider="cirq",
                         execute_circuits=False)._execute_circuits is False


class TestAnglesMatchGenerators:
    """A circuit whose angles and generators differ in number is refused: the
    gate stream used to ``zip`` them, silently dropping the unmatched ones."""

    @pytest.mark.parametrize("n_angles", [1, 3])
    def test_a_count_mismatch_is_refused(self, n_angles):
        generators = [PauliSum({"XY": 0.5j, "YX": -0.5j}, num_qubits=2)] * 2
        provider = build_provider("qiskit")
        with pytest.raises(ValueError, match="must match the register and generators"):
            provider.build(2, [0], generators, np.full(n_angles, 0.3))


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
        reference = Mandacaru(method="adapt-vqe", pool="qeb",
                              load_hamiltonian=h2_cache, trace=False,
                              max_iterations=4).run()
        executed = Mandacaru(method="adapt-vqe", pool="qeb",
                             load_hamiltonian=h2_cache, trace=False,
                             max_iterations=4, backend_provider=name,
                             execute_circuits=True).run()
        assert executed.optimal_energy == pytest.approx(
            reference.optimal_energy, abs=1e-8 * HARTREE_TO_EV)
        # The selected *set* must match; the order of symmetry-degenerate
        # operators can be decided by optimizer noise below 1e-9.
        assert set(executed.operators) == set(reference.operators)

    @pytest.mark.parametrize("name", PROVIDERS)
    def test_vqe_energy_is_provider_independent(self, name, h2_cache):
        reference = Mandacaru(method="vqe", load_hamiltonian=h2_cache,
                              trace=False).run()
        executed = Mandacaru(method="vqe", load_hamiltonian=h2_cache,
                             trace=False, backend_provider=name,
                             execute_circuits=True).run()
        # The circuit path is the Trotter product form, exact for H2's single
        # (double) excitation but only close for the general cluster operator.
        assert executed.optimal_energy == pytest.approx(
            reference.optimal_energy, abs=1e-6 * HARTREE_TO_EV)

    @pytest.mark.parametrize("name", PROVIDERS)
    def test_verbose_header_reports_the_provider(self, name, h2_cache, capsys):
        Mandacaru(method="adapt-vqe", pool="qeb", load_hamiltonian=h2_cache,
                  trace=True, max_iterations=1, backend_provider=name,
                  execute_circuits=True).run()
        out = capsys.readouterr().out
        # `[OPTIMIZATION SETUP]` is one option per line, `key: value`, and it
        # is the same block whether the run prints it or writes it to `txt=`.
        assert re.search(rf"^\s+backend_provider: {name}$", out, re.M)
        assert re.search(r"^\s+circuit_execution: True$", out, re.M)
