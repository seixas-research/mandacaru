# -*- coding: utf-8 -*-
# file: test_vqe.py

"""VQE with the UCCSD ansatz, validated end-to-end on H2 (Jordan-Wigner)."""

import numpy as np
import pytest

from carcara.algorithms import VQE, VQEResult
from carcara.circuits import UCCSD, double_excitation, single_excitation
from carcara.core import HydrogenicIntegrals, minimal_hydrogenic_basis
from carcara.core.mapping import PauliSum
from carcara.integrals import Grid
from carcara.optimizers import Optimizer

CHEMICAL_ACCURACY = 1.6e-3  # Ha


def _n_electrons_expectation(psi: np.ndarray) -> float:
    """<psi| N |psi> via occupation bit-counts of the computational basis."""
    probs = np.abs(psi) ** 2
    occ = np.array([bin(i).count("1") for i in range(psi.size)])
    return float(probs @ occ)


@pytest.fixture(scope="module")
def h2_hamiltonian():
    R = 0.74
    nuclei = [(1.0, np.array([0.0, 0.0, -R / 2])),
              (1.0, np.array([0.0, 0.0, +R / 2]))]
    grid = Grid(center=[0.0, 0.0, 0.0], box_size=5.0, h=0.25)
    return HydrogenicIntegrals(nuclei, minimal_hydrogenic_basis(nuclei),
                               grid).molecular_hamiltonian()


@pytest.fixture(scope="module")
def h2_exact(h2_hamiltonian):
    m = h2_hamiltonian.map_to_qubits("jordan_wigner").to_matrix()
    return float(np.linalg.eigvalsh(0.5 * (m + m.conj().T)).min())


# --- excitation generators ---

class TestExcitationGenerators:
    def test_single_is_anti_hermitian(self):
        m = single_excitation(0, 1).map_to_qubits("jordan_wigner", n_modes=4).to_matrix()
        assert np.allclose(m, -m.conj().T)

    def test_double_is_anti_hermitian(self):
        g = double_excitation(0, 2, 1, 3)
        m = g.map_to_qubits("jordan_wigner", n_modes=4).to_matrix()
        assert np.allclose(m, -m.conj().T)

    def test_single_excitation_moves_a_particle(self):
        # a+_1 a_0 - a+_0 a_1 connects |1000> and |0100> (qubit0 = MSB).
        g = single_excitation(0, 1).map_to_qubits("jordan_wigner", n_modes=2).to_matrix()
        assert abs(g[0b01, 0b10]) > 0.5   # |..> couples the two occupations


# --- UCCSD ansatz ---

class TestUCCSD:
    def test_h2_parameter_count(self):
        # 2 singles (a/b) + 1 double for H2.
        assert UCCSD(2, (1, 1)).num_parameters == 3

    def test_reference_is_hf_determinant(self):
        ansatz = UCCSD(2, (1, 1))
        hf = ansatz.reference_state()
        assert np.isclose(np.linalg.norm(hf), 1.0)
        # occupied spin-orbitals {0, 2} -> bits at positions 0 and 2 (MSB-first).
        idx = int(np.argmax(np.abs(hf)))
        assert idx == (1 << (4 - 1 - 0)) | (1 << (4 - 1 - 2))
        assert np.isclose(_n_electrons_expectation(hf), 2.0)

    def test_state_conserves_particle_number_and_norm(self):
        ansatz = UCCSD(2, (1, 1))
        rng = np.random.default_rng(1)
        for _ in range(5):
            psi = ansatz.state(rng.uniform(-np.pi, np.pi, ansatz.num_parameters))
            assert np.isclose(np.linalg.norm(psi), 1.0)
            assert np.isclose(_n_electrons_expectation(psi), 2.0, atol=1e-9)

    def test_zero_parameters_gives_reference(self):
        ansatz = UCCSD(2, (1, 1))
        assert np.allclose(ansatz.state(np.zeros(3)), ansatz.reference_state())

    def test_wrong_parameter_count_raises(self):
        with pytest.raises(ValueError):
            UCCSD(2, (1, 1)).state([0.1, 0.2])

    def test_trotter_mode_is_valid_state(self):
        ansatz = UCCSD(2, (1, 1), trotter=True)
        psi = ansatz.state([0.3, -0.2, 0.5])
        assert np.isclose(np.linalg.norm(psi), 1.0)
        assert np.isclose(_n_electrons_expectation(psi), 2.0, atol=1e-9)


# --- optimizer ---

class TestOptimizer:
    def test_minimizes_quadratic(self):
        res = Optimizer(method="COBYLA").minimize(lambda x: (x[0] - 3.0) ** 2, [0.0])
        assert np.isclose(res.x[0], 3.0, atol=1e-3)
        assert res.history and res.nfev == len(res.history)

    def test_slsqp_backend(self):
        res = Optimizer(method="SLSQP").minimize(
            lambda x: (x[0] + 1) ** 2 + (x[1] - 2) ** 2, [0.0, 0.0])
        assert np.allclose(res.x, [-1.0, 2.0], atol=1e-4)


# --- VQE end-to-end ---

class TestVQE:
    def test_h2_reaches_exact_ground_state(self, h2_hamiltonian, h2_exact):
        vqe = VQE(h2_hamiltonian, UCCSD(2, (1, 1)),
                  Optimizer("COBYLA", maxiter=2000))
        result = vqe.run()
        assert isinstance(result, VQEResult)
        assert abs(result.optimal_energy - h2_exact) < CHEMICAL_ACCURACY

    def test_vqe_lowers_the_reference_energy(self, h2_hamiltonian):
        vqe = VQE(h2_hamiltonian, UCCSD(2, (1, 1)))
        result = vqe.run()
        assert result.optimal_energy < result.reference_energy

    def test_accepts_pauli_sum_hamiltonian(self, h2_hamiltonian, h2_exact):
        qubit_h = h2_hamiltonian.map_to_qubits("jordan_wigner")
        assert isinstance(qubit_h, PauliSum)
        vqe = VQE(qubit_h, UCCSD(2, (1, 1)), Optimizer("COBYLA", maxiter=2000))
        assert abs(vqe.run().optimal_energy - h2_exact) < CHEMICAL_ACCURACY

    def test_reference_energy_matches_hf_expectation(self, h2_hamiltonian):
        vqe = VQE(h2_hamiltonian, UCCSD(2, (1, 1)))
        assert np.isclose(vqe.reference_energy(), vqe.energy(np.zeros(3)))

    def test_qubit_count_mismatch_raises(self, h2_hamiltonian):
        # A 4-qubit qubit Hamiltonian against a 6-qubit ansatz must be rejected.
        qubit_h = h2_hamiltonian.map_to_qubits("jordan_wigner")   # 4 qubits
        with pytest.raises(ValueError):
            VQE(qubit_h, UCCSD(3, (1, 1)))                        # 6 qubits
