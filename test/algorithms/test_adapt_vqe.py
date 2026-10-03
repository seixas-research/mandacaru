# -*- coding: utf-8 -*-
# file: test_adapt_vqe.py

"""ADAPT-VQE and the four operator pools, validated end-to-end on H2.

All tests run in the Hartree-Fock molecular-orbital basis, where Brillouin's
theorem makes single-excitation gradients vanish -- so ADAPT selects the physical
double excitation first and every pool reaches the FCI ground state.
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru.algorithms import ADAPTVQEResult, Mandacaru
from mandacaru.circuits import AdaptAnsatz, profile_ansatz
from mandacaru.circuits import (
    CEOPool,
    FermionicPool,
    QEBPool,
    QubitPool,
    available_pools,
    build_pool,
)
from mandacaru.core import MolecularIntegrals, minimal_hao_basis
from mandacaru.integrals import Grid
from mandacaru.optimizers import Optimizer
from mandacaru.units import HARTREE_TO_EV
from mandacaru.optimizers import DEFAULT_OPTIMIZER

# The classical optimizers used below, with the iteration budget and
# the convergence tolerance written out rather than left to the
# library default: a test that pins an energy should say what it was
# optimized with.
LBFGS = Optimizer(method="L-BFGS", maxiter=2000, tol=1e-12)

POOL_NAMES = ["fermionic", "qubit", "qeb", "ceo", "ceo-ovp", "spin-orbit"]


# --------------------------------------------------------------------------- #
# Shared H2 fixtures (MO basis).
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def h2_integrals():
    R = 0.74
    nuclei = [(1.0, np.array([0.0, 0.0, -R / 2])),
              (1.0, np.array([0.0, 0.0, +R / 2]))]
    grid = Grid(center=[0.0, 0.0, 0.0], box_size=5.0, h=0.25)
    return MolecularIntegrals(nuclei, minimal_hao_basis(nuclei), grid)


@pytest.fixture(scope="module")
def h2_hamiltonian(h2_integrals):
    return h2_integrals.molecular_hamiltonian(mo_basis=True, n_electrons=2)


@pytest.fixture(scope="module")
def h2_exact(h2_hamiltonian):
    """FCI ground state in Hartree (the qubit Hamiltonian's own unit)."""
    m = h2_hamiltonian.map_to_qubits("jordan_wigner").to_matrix()
    return float(np.linalg.eigvalsh(0.5 * (m + m.conj().T)).min())


@pytest.fixture(scope="module")
def h2_exact_ev(h2_exact):
    """The same FCI energy in eV -- the unit of every driver result."""
    return h2_exact * HARTREE_TO_EV


def _adapt(hamiltonian, pool_name, max_iterations=50, gradient=1e-6):
    # Stopping controls now live on the constructor (run() takes no duplicates).
    return Mandacaru(method="adapt-vqe", hamiltonian=hamiltonian,
                     pool=pool_name, num_particles=(1, 1),
                     n_spatial_orbitals=2,
                     optimizer=LBFGS,
                     max_iterations=max_iterations,
                     convergence={"gradient": gradient})


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

    def test_hartree_fock_hamiltonian_ground_state_is_hf(self, h2_integrals,
                                                         h2_exact):
        # The mean-field HF Hamiltonian's ground state (in the 2e sector) equals
        # the RHF total energy, and lies above the full FCI (correlation energy).
        rhf = h2_integrals.hartree_fock(2)
        e_hf = rhf.electronic_energy + h2_integrals.nuclear_repulsion
        H_hf = h2_integrals.hartree_fock_hamiltonian(2)
        m = H_hf.map_to_qubits("jordan_wigner").to_matrix()
        # Lowest eigenvalue of the diagonal mean-field Hamiltonian.
        hf_ground = float(np.linalg.eigvalsh(0.5 * (m + m.conj().T)).min())
        assert hf_ground == pytest.approx(e_hf, abs=1e-8)
        assert h2_exact < e_hf - 1e-4          # FCI below HF: correlation captured


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
    def test_pool_reaches_fci(self, h2_hamiltonian, h2_exact, h2_exact_ev, name):
        res = _adapt(h2_hamiltonian, name, max_iterations=15).run()
        assert isinstance(res, ADAPTVQEResult)
        # Results are in eV (1e-6 Ha = 2.7e-5 eV); in_units("Ha") converts back.
        assert res.energy_unit == "eV"
        assert abs(res.optimal_energy - h2_exact_ev) < 1e-6 * HARTREE_TO_EV
        assert abs(res.in_units("Ha") - h2_exact) < 1e-6
        assert res.optimal_energy < res.reference_energy - 1e-4 * HARTREE_TO_EV

    def test_result_history_is_consistent(self, h2_hamiltonian):
        res = _adapt(h2_hamiltonian, "fermionic").run()
        assert len(res.energy_history) == res.num_operators
        assert len(res.operators) == res.num_operators
        # One screening gradient per grown operator, each above the threshold
        # that would have stopped the loop.
        assert len(res.gradient_history) == res.num_operators
        assert all(g > 0 for g in res.gradient_history)
        # Energy decreases monotonically as operators are added.
        assert np.all(np.diff(res.energy_history) <= 1e-9 * HARTREE_TO_EV)


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
    def test_accepts_pool_object(self, h2_hamiltonian, h2_exact_ev):
        pool = build_pool("ceo", 2, (1, 1))
        res = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                        pool=pool, num_particles=(1, 1),
                        convergence={"gradient": 1e-6}).run()
        assert abs(res.optimal_energy - h2_exact_ev) < 1e-6 * HARTREE_TO_EV

    def test_named_pool_requires_shape(self, h2_hamiltonian):
        calc = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                         pool="fermionic")   # missing shape/particles
        # Refused where the problem is built (first use), not by the
        # option-checking constructor.
        with pytest.raises(ValueError):
            calc.solver

    def test_qubit_count_mismatch_raises(self, h2_hamiltonian):
        # A fixed-size qubit Hamiltonian against a wrongly-sized pool is rejected.
        qubit_h = h2_hamiltonian.map_to_qubits("jordan_wigner")   # 4 qubits
        calc = Mandacaru(method="adapt-vqe", hamiltonian=qubit_h,
                         pool="fermionic", num_particles=(1, 1),
                         n_spatial_orbitals=3)        # 6-qubit pool vs 4-qubit H
        with pytest.raises(ValueError):
            calc.solver


class TestADAPTVQECalculator:
    def test_class_named_all_caps(self):
        # The driver class is ADAPTVQE (all caps); the old AdaptVQE alias is gone.
        import mandacaru.algorithms as algs
        assert not hasattr(algs, "AdaptVQE")
        assert not hasattr(algs, "AdaptVQEResult")

    def test_ase_calculator_get_total_energy(self, tmp_path):
        # Requirement 7: attach ADAPTVQE as an ASE calculator; get_total_energy
        # runs the simulation and returns eV.
        def builder(atoms):
            nuclei = [(float(Z), np.asarray(R)) for Z, R in
                      zip(atoms.get_atomic_numbers(), atoms.get_positions())]
            grid = Grid(center=[0, 0, 0], box_size=5.0, h=0.35)
            H = MolecularIntegrals(
                nuclei, minimal_hao_basis(nuclei), grid
            ).molecular_hamiltonian(mo_basis=True, n_electrons=2)
            return H, (1, 1), 2

        atoms = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]])
        atoms.calc = Mandacaru(method="adapt-vqe", pool="ceo",
                               hamiltonian_builder=builder, max_iterations=6,
                               convergence={"gradient": 1e-4})
        energy_ev = atoms.get_total_energy()
        result = atoms.calc.result

        # ASE returns eV, and so does the result object itself.
        assert result.energy_unit == "eV"
        assert energy_ev == pytest.approx(result.optimal_energy, rel=1e-9)
        # And match the exact FCI of the built Hamiltonian (Hartree -> eV).
        h = atoms.calc.hamiltonian.to_matrix()
        exact = float(np.linalg.eigvalsh(0.5 * (h + h.conj().T)).min())
        assert result.optimal_energy == pytest.approx(
            exact * 27.211386245988, abs=1e-4 * 27.211386245988)
        assert result.in_units("Ha") == pytest.approx(exact, abs=1e-4)

    def test_calculator_builds_from_default_basis(self):
        # With the default basis="HAO", no explicit builder is needed: the
        # calculator builds the Hamiltonian from the geometry itself.
        atoms = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]])
        atoms.calc = Mandacaru(method="adapt-vqe", pool="ceo", basis="HAO",
                               grid=Grid(center=[0, 0, 0], box_size=6.0, h=0.30),
                               max_iterations=6, convergence={"gradient": 1e-3})
        energy = atoms.get_total_energy()
        assert np.isfinite(energy)
        assert atoms.calc.n_qubits == 4        # H2 in HAO -> 2 orbitals

    def test_ibm_quantum_device_requires_shots(self):
        # Real hardware never returns a state vector: refused up front.
        with pytest.raises(ValueError, match="shots > 0"):
            Mandacaru(method="adapt-vqe", pool="ceo", basis="HAO",
                      device="ibm-quantum")

    def test_grid_auto_generated_from_cell(self):
        # No explicit grid: the calculator builds one from atoms.cell at
        # resolution h, and the run still reaches a finite energy.
        atoms = Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
                      cell=[[6, 0, 0], [0, 6, 0], [0, 0, 6]], pbc=True)
        atoms.calc = Mandacaru(method="adapt-vqe", pool="ceo", basis="HAO",
                               h=0.30, max_iterations=6,
                               convergence={"gradient": 1e-3})
        assert np.isfinite(atoms.get_total_energy())
        assert atoms.calc.n_qubits == 4

    def test_grid_requires_cell_when_not_given(self):
        # Without an explicit grid AND without a unit cell, grid auto-generation
        # is impossible -> a clear error.
        atoms = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]])  # no cell
        atoms.calc = Mandacaru(method="adapt-vqe", pool="ceo", basis="HAO",
                               max_iterations=4, convergence={"gradient": 1e-3})
        with pytest.raises(ValueError, match="no unit cell"):
            atoms.get_total_energy()


class TestArgumentSurface:
    @pytest.mark.parametrize("pool", ["ceo", "fermionic", "qubit", "qeb"])
    def test_pool_options(self, pool):
        atoms = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]])
        atoms.calc = Mandacaru(method="adapt-vqe", pool=pool, basis="HAO",
                               grid=Grid(center=[0, 0, 0], box_size=6.0, h=0.3),
                               max_iterations=8, convergence={"gradient": 1e-3})
        assert np.isfinite(atoms.get_total_energy())

    @pytest.mark.parametrize("mapping",
                             ["jordan_wigner", "parity", "bravyi_kitaev"])
    def test_mapping_options_reach_fci(self, mapping):
        atoms = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]])
        atoms.calc = Mandacaru(method="adapt-vqe", pool="fermionic",
                               basis="HAO", mapping=mapping,
                               grid=Grid(center=[0, 0, 0], box_size=6.0, h=0.25),
                               max_iterations=8, convergence={"gradient": 1e-3})
        energy_ev = atoms.get_total_energy()
        h = atoms.calc.hamiltonian.to_matrix()
        exact = float(np.linalg.eigvalsh(0.5 * (h + h.conj().T)).min())
        exact_ev = exact * 27.211386245988
        assert abs(energy_ev - exact_ev) < 1e-3, mapping

    def test_run_defaults_come_from_constructor(self, h2_hamiltonian):
        # max_iterations / convergence / output are constructor args and
        # supply the defaults for run().
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool="ceo", num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=False,
                          max_iterations=3, convergence={"gradient": 1e-2})
        assert adapt.max_iterations == 3
        assert adapt.convergence.gradient == 1e-2
        assert adapt.convergence.energy is None
        res = adapt.run()                       # no args -> uses the defaults
        assert res.num_operators <= 3

    def test_output_constructor_arg_writes_file(self, h2_hamiltonian, tmp_path):
        from mandacaru.utils import parse_output
        out = str(tmp_path / "output.txt")
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool="fermionic", num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=False,
                          max_iterations=4, convergence={"gradient": 1e-3},
                          txt=out)
        adapt.run()                             # output taken from constructor
        parsed = parse_output(out)
        assert parsed["setup"]["classical_optimizer"] == DEFAULT_OPTIMIZER
        assert len(parsed["iterations"]) >= 1

    def test_basis_option_sets_qubit_count(self):
        # HAO on LiH -> Li{1s,2s} + H{1s} = 3 orbitals -> 6 qubits.
        atoms = Atoms("LiH", positions=[[0, 0, -0.8], [0, 0, 0.8]])
        atoms.calc = Mandacaru(method="adapt-vqe", pool="ceo", basis="HAO",
                               grid=Grid(center=[0, 0, 0], box_size=7.0, h=0.3),
                               max_iterations=4, convergence={"gradient": 1e-2})
        atoms.get_total_energy()
        assert atoms.calc.n_qubits == 6
        assert atoms.calc.num_particles == (2, 2)


class TestOptimizerOption:
    def test_default_is_cobyla(self, h2_hamiltonian):
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool="ceo", num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=False)
        assert adapt.optimizer.method == DEFAULT_OPTIMIZER

    @pytest.mark.parametrize(
        "name",
        ["SPSA", "COBYLA", "Nelder-Mead", "SLSQP", "L-BFGS", "BFGS"])
    def test_named_optimizers_build(self, h2_hamiltonian, name):
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool="ceo", num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=False, optimizer=name)
        assert adapt.optimizer.method == name

    def test_optimizer_instance_passthrough(self, h2_hamiltonian):
        opt = LBFGS
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool="ceo", num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=False, optimizer=opt)
        assert adapt.optimizer is opt

    def test_unknown_optimizer_rejected(self, h2_hamiltonian):
        with pytest.raises(ValueError):
            Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                      pool="ceo", num_particles=(1, 1), n_spatial_orbitals=2,
                      optimizer="nope")


class TestSpinOrbitCoupling:
    """A Hamiltonian that conserves N but not S_z (spin-orbit coupling)."""

    M = 3

    @pytest.fixture(scope="class")
    def problem(self):
        from mandacaru.core.hamiltonian import spin_block_integrals
        from mandacaru.core.mapping import Fermion

        M = self.M
        rng = np.random.default_rng(11)
        a, b = rng.normal(size=(M, M)), rng.normal(size=(M, M))
        eri = 0.2 * np.einsum("pr,qs->pqrs", b + b.T, b + b.T)
        coupling = rng.normal(size=(2 * M, 2 * M)) \
            + 1j * rng.normal(size=(2 * M, 2 * M))
        coupling = 0.2 * (coupling + coupling.conj().T)
        coupling[:M, :M] = coupling[M:, M:] = 0.0
        fermion = Fermion.from_integrals(*spin_block_integrals(a + a.T, eri,
                                                               coupling))
        # Exact ground state among all two-electron states, any S_z.
        matrix = fermion.map_to_qubits("jordan_wigner",
                                       n_modes=2 * M).to_matrix()
        states = [i for i in range(4 ** M) if bin(i).count("1") == 2]
        exact = np.linalg.eigvalsh(matrix[np.ix_(states, states)])[0]
        return fermion, exact

    def _run(self, fermion, **options):
        return Mandacaru(method="adapt-vqe", hamiltonian=fermion,
                         num_particles=(1, 1), n_spatial_orbitals=self.M,
                         atomic_units=True, trace=False, profile=False,
                         **options).run()

    @pytest.mark.parametrize("options", [
        {}, {"sector": True}, {"sector": True, "sparse": "matrix-free"}])
    def test_the_spin_orbit_pool_reaches_the_exact_ground_state(
            self, problem, options):
        fermion, exact = problem
        result = self._run(fermion, pool="spin-orbit", max_iterations=40,
                           convergence={"gradient": 1e-6, "energy": None},
                           **options)
        assert result.in_units("Ha") == pytest.approx(exact, abs=1e-9)

    def test_a_spin_conserving_pool_is_refused(self, problem):
        fermion, _ = problem
        with pytest.raises(ValueError, match="pool='spin-orbit'"):
            self._run(fermion, pool="fermionic", max_iterations=1)

    # Their ansatz and references conserve S_z: they would settle in a sector
    # the Hamiltonian does not keep.

    def test_vqe_refuses_it(self, problem):
        from mandacaru.circuits import UCCSD

        fermion, _ = problem
        with pytest.raises(NotImplementedError, match="spin-orbit"):
            Mandacaru(method="vqe", hamiltonian=fermion,
                      ansatz=UCCSD(self.M, (1, 1)), atomic_units=True,
                      trace=False).run()

    def test_subspace_adapt_refuses_it(self, problem):
        fermion, _ = problem
        with pytest.raises(NotImplementedError, match="spin-orbit"):
            Mandacaru(method="subspace-adapt-vqe", hamiltonian=fermion,
                      pool="spin-orbit", num_particles=(1, 1),
                      n_spatial_orbitals=self.M, atomic_units=True,
                      trace=False, profile=False, max_iterations=1).run()


class TestTheLogOpensBeforeTheBuild:
    """``[SYSTEM]`` is written before the basis and integrals are built."""

    def test_the_geometry_is_logged_even_when_the_build_fails(
            self, tmp_path, monkeypatch):
        import mandacaru.algorithms._hamiltonian_from_atoms as builder
        from mandacaru.utils.logging import parse_output

        def fail(*args, **kwargs):
            raise RuntimeError("stop before the integrals")

        monkeypatch.setattr(builder, "build_basis_hamiltonian", fail)
        path = tmp_path / "output.txt"
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[6] * 3)
        atoms.calc = Mandacaru(method="adapt-vqe", h=0.35, pool="qeb",
                               txt=str(path))
        with pytest.raises(RuntimeError, match="stop before"):
            atoms.get_potential_energy()
        text = path.read_text()
        assert "ADAPT-VQE (QEBPool)" in text and "[SYSTEM]" in text
        assert "[ELECTRONS]" not in text
        assert parse_output(str(path))["system"]["n_atoms"] == "2"


# --------------------------------------------------------------------------- #
# Recording growth steps (the training data of VALQA's sequence model).
# --------------------------------------------------------------------------- #

class TestRecording:
    def _run(self, h2_hamiltonian, **options):
        return Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                         pool="fermionic", num_particles=(1, 1),
                         n_spatial_orbitals=2, profile=False, trace=False,
                         convergence={"gradient": 1e-6}, **options).run()

    def test_one_row_per_growth_step(self, h2_hamiltonian, tmp_path):
        from mandacaru.algorithms.proposal_data import load_edits
        result = self._run(h2_hamiltonian, record=tmp_path)
        rows, problems = load_edits(tmp_path)
        assert len(rows) == len(result.operators) >= 1
        prefix = []
        for row in rows:
            problem = problems[row["problem"]]
            assert row["method"] == "ADAPT-VQE" and row["move"] == "insert"
            assert row["source"] == prefix
            assert row["position"] == len(prefix)
            gradients = np.abs(row["source_gradients"])
            assert len(gradients) == len(problem.pool_labels)
            # ADAPT appends the operator of the largest gradient.
            assert row["operator"] == int(np.argmax(gradients))
            prefix = prefix + [row["operator"]]
        assert [problems[rows[0]["problem"]].pool_labels[i]
                for i in prefix] == result.operators
        energies = np.cumsum([r["delta_energy"] for r in rows])
        assert energies[-1] == pytest.approx(
            (result.optimal_energy - result.reference_energy)
            / HARTREE_TO_EV, abs=1e-9)

    def test_nothing_is_recorded_by_default(self, h2_hamiltonian, tmp_path,
                                            monkeypatch):
        monkeypatch.setenv("MANDACARU_PROPOSAL_DATA", str(tmp_path))
        self._run(h2_hamiltonian)
        assert not any(tmp_path.iterdir())

    def test_tetris_cannot_record(self, tmp_path):
        with pytest.raises(ValueError, match="tetris or prune"):
            Mandacaru(method="adapt-vqe", tetris=True, record=tmp_path)


# --------------------------------------------------------------------------- #
# Carrying the ansatz from one geometry to the next.
# --------------------------------------------------------------------------- #

def _lih(distance):
    atoms = Atoms("LiH", positions=[[0, 0, 0], [0, 0, distance]])
    atoms.center(vacuum=3.5)
    return atoms


def _lih_calculator(**options):
    return Mandacaru(method="adapt-vqe",
                     basis={"name": "PAW-LCAO", "size": "DZP"}, h=0.3,
                     active_space={"orbitals": 4, "method": "mp2"},
                     pool="qeb", max_iterations=20,
                     optimizer={"method": "BFGS", "maxiter": 500,
                                "tol": 1e-10},
                     trace=False, **options)


class TestTransfer:
    """``transfer=True``: each geometry starts from the previous one's
    ansatz, re-optimized here, and grows only if the gradient asks for it."""

    @pytest.fixture(scope="class")
    def runs(self):
        carried = _lih_calculator(transfer=True)
        atoms = _lih(1.595)
        atoms.calc = carried
        atoms.get_potential_energy()
        moved = _lih(1.580)
        moved.calc = carried
        moved.get_potential_energy()
        fresh = _lih(1.580)
        fresh.calc = _lih_calculator()
        fresh.get_potential_energy()
        return carried.solver.result, fresh.calc.solver.result

    def test_the_previous_ansatz_is_carried_and_relaxed(self, runs):
        carried, _fresh = runs
        assert carried.start.startswith("previous geometry's ansatz")
        assert "orbitals tracked" in carried.start

    def test_it_reaches_the_fresh_energy_for_less(self, runs):
        carried, fresh = runs
        # LiH / 4 orbitals: the carried 5-operator ansatz already satisfies
        # the gradient criterion at the new geometry, so nothing is appended.
        assert carried.optimal_energy == pytest.approx(fresh.optimal_energy,
                                                       abs=1e-6)
        assert carried.num_evaluations < fresh.num_evaluations
        assert len(set(carried.operators)) == len(carried.operators)

    def test_a_different_system_is_not_carried(self):
        calc = Mandacaru(method="adapt-vqe", basis="HAO", h=0.35,
                         pool="fermionic", max_iterations=4, transfer=True,
                         trace=False)
        h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        h2.center(vacuum=3.0)
        h2.calc = calc
        h2.get_potential_energy()
        lih = Atoms("LiH", positions=[[0, 0, 0], [0, 0, 1.6]])
        lih.center(vacuum=3.0)
        lih.calc = calc
        lih.get_potential_energy()
        assert "another system or pool" in calc.solver.result.start


class TestResumeIntoAnotherGeometry:
    def test_the_angles_are_relaxed_before_the_gradient_is_read(
            self, tmp_path):
        path = tmp_path / "ansatz.json"
        first = _lih(1.595)
        first.calc = _lih_calculator(checkpoint=str(path))
        first.get_potential_energy()
        moved = _lih(1.580)
        moved.calc = _lih_calculator(resume=str(path))
        with pytest.warns(RuntimeWarning, match="different Hamiltonian"):
            moved.get_potential_energy()
        result = moved.calc.solver.result
        assert "re-optimized" in result.start
        # Relaxed first, the existing operators are not re-selected.
        assert len(set(result.operators)) == len(result.operators)
        fresh = _lih(1.580)
        fresh.calc = _lih_calculator()
        fresh.get_potential_energy()
        assert result.optimal_energy == pytest.approx(
            fresh.calc.solver.result.optimal_energy, abs=1e-6)
