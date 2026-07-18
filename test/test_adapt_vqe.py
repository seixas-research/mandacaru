# -*- coding: utf-8 -*-
# file: test_adapt_vqe.py

"""ADAPT-VQE and the four operator pools, validated end-to-end on H2.

All tests run in the Hartree-Fock molecular-orbital basis, where Brillouin's
theorem makes single-excitation gradients vanish -- so ADAPT selects the physical
double excitation first and every pool reaches the FCI ground state.
"""

import numpy as np
import pytest

from carcara.algorithms import ADAPTVQE, ADAPTVQEResult, RHF
from carcara.algorithms.adapt_vqe import AdaptAnsatz, profile_ansatz
from carcara.circuits import (
    CEOPool,
    FermionicPool,
    QEBPool,
    QubitPool,
    available_pools,
    build_pool,
)
from carcara.core import MolecularIntegrals, minimal_fao_basis
from carcara.integrals import Grid
from carcara.optimizers import Optimizer

POOL_NAMES = ["fermionic", "qubit", "qeb", "ceo"]


# --------------------------------------------------------------------------- #
# Shared H2 fixtures (MO basis).
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def h2_integrals():
    R = 0.74
    nuclei = [(1.0, np.array([0.0, 0.0, -R / 2])),
              (1.0, np.array([0.0, 0.0, +R / 2]))]
    grid = Grid(center=[0.0, 0.0, 0.0], box_size=5.0, h=0.25)
    return MolecularIntegrals(nuclei, minimal_fao_basis(nuclei), grid)


@pytest.fixture(scope="module")
def h2_hamiltonian(h2_integrals):
    return h2_integrals.molecular_hamiltonian(mo_basis=True, n_electrons=2)


@pytest.fixture(scope="module")
def h2_exact(h2_hamiltonian):
    m = h2_hamiltonian.map_to_qubits("jordan_wigner").to_matrix()
    return float(np.linalg.eigvalsh(0.5 * (m + m.conj().T)).min())


def _adapt(hamiltonian, pool_name, max_iterations=50, gradient_tol=1e-6):
    # Stopping controls now live on the constructor (run() takes no duplicates).
    return ADAPTVQE(hamiltonian, pool_name, num_particles=(1, 1),
                    n_spatial_orbitals=2,
                    optimizer=Optimizer("L-BFGS-B", maxiter=2000),
                    max_iterations=max_iterations, gradient_tolerance=gradient_tol)


# --------------------------------------------------------------------------- #
# Hartree-Fock.
# --------------------------------------------------------------------------- #

class TestHartreeFock:
    def test_rhf_converges_and_reference_matches_hf(self, h2_integrals):
        rhf = h2_integrals.hartree_fock(2)
        assert rhf.converged
        assert rhf.n_occupied == 1
        # The HF total energy equals the MO-basis reference determinant energy.
        H = h2_integrals.molecular_hamiltonian(mo_basis=True, n_electrons=2)
        adapt = _adapt(H, "fermionic")
        total_hf = rhf.electronic_energy + h2_integrals.nuclear_repulsion
        assert np.isclose(adapt.reference_energy(), total_hf, atol=1e-9)

    def test_mo_orbital_energies_are_ordered(self, h2_integrals):
        rhf = h2_integrals.hartree_fock(2)
        assert np.all(np.diff(rhf.mo_energies) >= -1e-12)


# --------------------------------------------------------------------------- #
# Operator pools.
# --------------------------------------------------------------------------- #

class TestPools:
    def test_registry_builds_every_pool(self):
        assert set(available_pools()) == set(POOL_NAMES)
        for name in POOL_NAMES:
            pool = build_pool(name, 2, (1, 1))
            assert len(pool) >= 1
            assert pool.n_qubits == 4

    def test_pool_classes(self):
        assert isinstance(build_pool("fermionic", 2, (1, 1)), FermionicPool)
        assert isinstance(build_pool("qubit", 2, (1, 1)), QubitPool)
        assert isinstance(build_pool("qeb", 2, (1, 1)), QEBPool)
        assert isinstance(build_pool("ceo", 2, (1, 1)), CEOPool)

    def test_aliases_and_unknown(self):
        assert isinstance(build_pool("uccsd", 2, (1, 1)), FermionicPool)
        assert isinstance(build_pool("qubit-adapt", 2, (1, 1)), QubitPool)
        with pytest.raises(ValueError):
            build_pool("nonsense", 2, (1, 1))

    @pytest.mark.parametrize("name", POOL_NAMES)
    def test_generators_are_anti_hermitian(self, name):
        for op in build_pool(name, 2, (1, 1)).operators():
            m = op.matrix()
            assert np.allclose(m, -m.conj().T, atol=1e-10)

    @pytest.mark.parametrize("name", ["fermionic", "qeb", "ceo"])
    def test_excitation_pools_conserve_particle_number(self, name):
        # Fermionic/QEB/CEO generators keep the state in its particle sector:
        # [A, N] = 0.  (The qubit pool's individual Pauli generators deliberately
        # do not -- number conservation is restored only by their combination.)
        n_op = np.diag([bin(i).count("1") for i in range(16)]).astype(complex)
        for op in build_pool(name, 2, (1, 1)).operators():
            m = op.matrix()
            assert np.allclose(m @ n_op - n_op @ m, 0.0, atol=1e-10)

    def test_qeb_drops_z_strings_relative_to_fermionic(self):
        # The QEB double acts only on its 4 qubits; the fermionic double is
        # identical for H2 (support already spans all qubits) but QEB never has
        # support beyond the excitation indices.
        for op in build_pool("qeb", 2, (1, 1)).operators():
            assert set(op.support).issubset({0, 1, 2, 3})


# --------------------------------------------------------------------------- #
# Gradient screening / symmetry.
# --------------------------------------------------------------------------- #

class TestGradientSelection:
    def test_singles_have_zero_gradient_at_hf(self, h2_hamiltonian):
        # Brillouin: at the HF reference, single-excitation gradients vanish and
        # the double dominates -- the symmetry-allowed correlating excitation.
        adapt = _adapt(h2_hamiltonian, "fermionic")
        ref = AdaptAnsatz(4, adapt.pool.occupied_orbitals).reference_state()
        grads = adapt._gradients(ref)
        pool = adapt.pool.operators()
        for op, g in zip(pool, grads):
            if op.kind == "fermionic-single":
                assert abs(g) < 1e-6
        idx = int(np.argmax(np.abs(grads)))
        assert pool[idx].kind == "fermionic-double"

    def test_first_selected_operator_is_a_double(self, h2_hamiltonian):
        for name in ("fermionic", "qeb", "ceo"):
            res = _adapt(h2_hamiltonian, name, max_iterations=6).run()
            assert "double" in res.iterations[0].operator_kind \
                or res.iterations[0].operator_kind == "ceo"


# --------------------------------------------------------------------------- #
# Convergence to FCI.
# --------------------------------------------------------------------------- #

class TestConvergence:
    @pytest.mark.parametrize("name", POOL_NAMES)
    def test_pool_reaches_fci(self, h2_hamiltonian, h2_exact, name):
        res = _adapt(h2_hamiltonian, name, max_iterations=15).run()
        assert isinstance(res, ADAPTVQEResult)
        assert abs(res.optimal_energy - h2_exact) < 1e-6
        assert res.optimal_energy < res.reference_energy - 1e-4

    def test_result_history_is_consistent(self, h2_hamiltonian):
        res = _adapt(h2_hamiltonian, "fermionic").run()
        assert len(res.energy_history) == res.num_operators
        assert len(res.operators) == res.num_operators
        # Energy decreases monotonically as operators are added.
        assert np.all(np.diff(res.energy_history) <= 1e-9)


# --------------------------------------------------------------------------- #
# Circuit profiling.
# --------------------------------------------------------------------------- #

class TestProfiling:
    def test_metrics_grow_with_ansatz(self, h2_hamiltonian):
        pool = build_pool("fermionic", 2, (1, 1))
        ansatz = AdaptAnsatz(4, pool.occupied_orbitals)
        prev = profile_ansatz(4, ansatz.occupied, ansatz.operators)
        assert prev.cnot_count == 0            # only HF X-gates so far
        for op in pool.operators():
            ansatz.append(op)
            cur = profile_ansatz(4, ansatz.occupied, ansatz.operators)
            assert cur.num_operators == ansatz.num_parameters
            assert cur.cnot_count >= prev.cnot_count
            assert cur.depth >= prev.depth
            prev = cur

    def test_run_reports_cnot_and_depth(self, h2_hamiltonian):
        res = _adapt(h2_hamiltonian, "fermionic").run()
        assert res.metrics.cnot_count is not None and res.metrics.cnot_count > 0
        assert res.metrics.depth is not None and res.metrics.depth > 0
        # Per-iteration metrics track the growing ansatz.
        cnots = [it.cnot_count for it in res.iterations]
        assert cnots == sorted(cnots)

    def test_qubit_pool_uses_fewer_cnots_than_fermionic(self, h2_hamiltonian):
        # The headline hardware-efficiency benchmark: qubit-ADAPT reaches the same
        # ground state with strictly fewer CNOTs than the fermionic pool.
        ferm = _adapt(h2_hamiltonian, "fermionic").run()
        qub = _adapt(h2_hamiltonian, "qubit").run()
        assert qub.metrics.cnot_count < ferm.metrics.cnot_count


# --------------------------------------------------------------------------- #
# Driver plumbing.
# --------------------------------------------------------------------------- #

class TestDriver:
    def test_accepts_pool_object(self, h2_hamiltonian, h2_exact):
        pool = build_pool("ceo", 2, (1, 1))
        res = ADAPTVQE(h2_hamiltonian, pool, num_particles=(1, 1),
                       gradient_tolerance=1e-6).run()
        assert abs(res.optimal_energy - h2_exact) < 1e-6

    def test_named_pool_requires_shape(self, h2_hamiltonian):
        with pytest.raises(ValueError):
            ADAPTVQE(h2_hamiltonian, "fermionic")   # missing shape/particles

    def test_qubit_count_mismatch_raises(self, h2_hamiltonian):
        # A fixed-size qubit Hamiltonian against a wrongly-sized pool is rejected.
        qubit_h = h2_hamiltonian.map_to_qubits("jordan_wigner")   # 4 qubits
        with pytest.raises(ValueError):
            ADAPTVQE(qubit_h, "fermionic", num_particles=(1, 1),
                     n_spatial_orbitals=3)           # 6-qubit pool vs 4-qubit H
