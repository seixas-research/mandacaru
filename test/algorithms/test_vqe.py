# -*- coding: utf-8 -*-
# file: test_vqe.py

"""VQE with the UCCSD ansatz, validated end-to-end on H2 (Jordan-Wigner)."""

import numpy as np
import pytest
from ase import Atoms

from mandacaru.algorithms import Mandacaru, VQEResult
from mandacaru.circuits import UCCSD, double_excitation, single_excitation
from mandacaru.core import MolecularIntegrals, minimal_hao_basis
from mandacaru.core.mapping import PauliSum
from mandacaru.integrals import Grid
from mandacaru.optimizers import DEFAULT_OPTIMIZER, Optimizer
from mandacaru.units import HARTREE_TO_EV, from_hartree

# The classical optimizers used below, with the iteration budget and
# the convergence tolerance written out rather than left to the
# library default: a test that pins an energy should say what it was
# optimized with.
COBYLA_OPT = Optimizer(method="COBYLA", maxiter=2000, tol=1e-12)
LBFGS = Optimizer(method="L-BFGS", maxiter=500, tol=1e-12)

CHEMICAL_ACCURACY = 1.6e-3 * HARTREE_TO_EV  # 0.043 eV (results are in eV)


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
    grid = Grid(center=[0.0, 0.0, 0.0], box_size=5.0, grid_spacing=0.25)
    return MolecularIntegrals(nuclei, minimal_hao_basis(nuclei),
                               grid).molecular_hamiltonian()


@pytest.fixture(scope="module")
def h2_exact(h2_hamiltonian):
    """FCI ground state in eV (the Hamiltonian is in Hartree; results in eV)."""
    m = h2_hamiltonian.map_to_qubits("jordan_wigner").to_matrix()
    return float(np.linalg.eigvalsh(0.5 * (m + m.conj().T)).min()) * HARTREE_TO_EV


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
        res = Optimizer(method="COBYLA").minimize(
            lambda x: (x[0] - 3.0) ** 2, [0.0])
        assert np.isclose(res.x[0], 3.0, atol=1e-3)
        assert res.history and res.nfev == len(res.history)

    def test_slsqp_backend(self):
        res = Optimizer(method="SLSQP").minimize(
            lambda x: (x[0] + 1) ** 2 + (x[1] - 2) ** 2, [0.0, 0.0])
        assert np.allclose(res.x, [-1.0, 2.0], atol=1e-4)

    def test_spsa_minimizes_quadratic(self):
        res = Optimizer(method="SPSA", maxiter=600).minimize(
            lambda x: (x[0] + 1) ** 2 + (x[1] - 2) ** 2, [0.0, 0.0])
        assert np.allclose(res.x, [-1.0, 2.0], atol=1e-1)
        assert res.fun < 0.1

    def test_zero_parameter_ansatz(self):
        res = Optimizer(method="SPSA").minimize(lambda x: 7.0, [])
        assert res.fun == 7.0 and res.nfev == 1


# --- VQE end-to-end ---

class TestVQE:
    def test_h2_reaches_exact_ground_state(self, h2_hamiltonian, h2_exact):
        vqe = Mandacaru(method="vqe", hamiltonian=h2_hamiltonian,
                        ansatz=UCCSD(2, (1, 1)),
                        optimizer=COBYLA_OPT)
        result = vqe.run()
        assert isinstance(result, VQEResult)
        assert abs(result.optimal_energy - h2_exact) < CHEMICAL_ACCURACY

    def test_vqe_lowers_the_reference_energy(self, h2_hamiltonian):
        vqe = Mandacaru(method="vqe", hamiltonian=h2_hamiltonian,
                        ansatz=UCCSD(2, (1, 1)))
        result = vqe.run()
        assert result.optimal_energy < result.reference_energy

    def test_accepts_pauli_sum_hamiltonian(self, h2_hamiltonian, h2_exact):
        qubit_h = h2_hamiltonian.map_to_qubits("jordan_wigner")
        assert isinstance(qubit_h, PauliSum)
        vqe = Mandacaru(method="vqe", hamiltonian=qubit_h,
                        ansatz=UCCSD(2, (1, 1)),
                        optimizer=COBYLA_OPT)
        assert abs(vqe.run().optimal_energy - h2_exact) < CHEMICAL_ACCURACY

    def test_reference_energy_matches_hf_expectation(self, h2_hamiltonian):
        vqe = Mandacaru(method="vqe", hamiltonian=h2_hamiltonian,
                        ansatz=UCCSD(2, (1, 1)))
        assert np.isclose(vqe.solver.reference_energy(), vqe.solver.energy_at(np.zeros(3)))

    def test_qubit_count_mismatch_raises(self, h2_hamiltonian):
        # A 4-qubit qubit Hamiltonian against a 6-qubit ansatz must be rejected.
        qubit_h = h2_hamiltonian.map_to_qubits("jordan_wigner")   # 4 qubits
        calc = Mandacaru(method="vqe", hamiltonian=qubit_h,
                         ansatz=UCCSD(3, (1, 1)))                     # 6 qubits
        # A property of the *problem*, so it is refused where the problem is
        # built -- on first use -- not by the (option-checking) constructor.
        with pytest.raises(ValueError):
            calc.solver


# --- Parity with ADAPTVQE / ADAPTVQEResult (requirement: mirror the API) ---

class TestVQEMirrorsADAPT:
    def test_default_optimizer_is_cobyla(self, h2_hamiltonian):
        vqe = Mandacaru(method="vqe", hamiltonian=h2_hamiltonian,
                        ansatz=UCCSD(2, (1, 1)))
        assert vqe.solver.optimizer.method == DEFAULT_OPTIMIZER

    @pytest.mark.parametrize(
        "name",
        ["SPSA", "COBYLA", "Nelder-Mead", "SLSQP", "L-BFGS", "BFGS"])
    def test_named_optimizers_build(self, h2_hamiltonian, name):
        vqe = Mandacaru(method="vqe", hamiltonian=h2_hamiltonian,
                        ansatz=UCCSD(2, (1, 1)), optimizer=name)
        assert vqe.solver.optimizer.method == name

    def test_optimizer_instance_passthrough(self, h2_hamiltonian):
        opt = LBFGS
        vqe = Mandacaru(method="vqe", hamiltonian=h2_hamiltonian,
                        ansatz=UCCSD(2, (1, 1)), optimizer=opt)
        assert vqe.solver.optimizer is opt

    def test_unknown_optimizer_rejected(self, h2_hamiltonian):
        with pytest.raises(ValueError):
            Mandacaru(method="vqe", hamiltonian=h2_hamiltonian,
                      ansatz=UCCSD(2, (1, 1)), optimizer="nope")

    def test_result_mirrors_adapt_result_shape(self, h2_hamiltonian):
        res = Mandacaru(method="vqe", hamiltonian=h2_hamiltonian,
                        ansatz=UCCSD(2, (1, 1)), trace=False).run()
        # Same convenience views as ADAPTVQEResult.
        assert res.num_parameters == len(res.optimal_parameters)
        assert res.energy_history == list(res.history)
        assert res.correlation_energy == pytest.approx(
            res.optimal_energy - res.reference_energy)

    def test_verbose_summarizes_hamiltonian_without_dumping_paulis(
            self, h2_hamiltonian, capsys):
        vqe = Mandacaru(method="vqe", hamiltonian=h2_hamiltonian,
                        ansatz=UCCSD(2, (1, 1)), trace=True)
        vqe.run()
        out = capsys.readouterr().out
        # Only the term count is printed, not the Pauli-string expansion.
        n_terms = len(vqe.hamiltonian.simplify().terms)
        assert f"Hamiltonian: {n_terms} Pauli terms" in out
        assert "* ZIII" not in out
        assert "[VARIATIONAL QUANTUM SUMMARY]" in out

    def test_verbose_false_is_silent(self, h2_hamiltonian, capsys):
        Mandacaru(method="vqe", hamiltonian=h2_hamiltonian,
                  ansatz=UCCSD(2, (1, 1)), trace=False).run()
        assert capsys.readouterr().out == ""


# --- VQE as an ASE calculator (requirement: ASE for all quantum simulations) ---

class TestVQEAsASECalculator:
    def test_get_total_energy_matches_exact(self):
        atoms = Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
                      cell=[[6, 0, 0], [0, 6, 0], [0, 0, 6]], pbc=True)
        atoms.calc = Mandacaru(method="vqe", basis="HAO", optimizer=COBYLA_OPT,
                               grid_spacing=0.30, trace=False)
        energy_ev = atoms.get_total_energy()
        result = atoms.calc.result
        # ASE returns eV, and so does the result (its unit is recorded).
        assert result.energy_unit == "eV"
        assert energy_ev == pytest.approx(result.optimal_energy, rel=1e-9)
        # And match the exact FCI of the built Hamiltonian (Hartree -> eV).
        h = atoms.calc.hamiltonian.to_matrix()
        exact = float(np.linalg.eigvalsh(0.5 * (h + h.conj().T)).min())
        assert result.in_units("Ha") == pytest.approx(exact, abs=1e-3)
        assert result.optimal_energy == pytest.approx(exact * HARTREE_TO_EV,
                                                      abs=1e-3 * HARTREE_TO_EV)
        assert atoms.calc.n_qubits == 4

    def test_builder_hamiltonian_and_default_ansatz(self):
        # An explicit hamiltonian_builder + the default UCCSD ansatz factory.
        def builder(atoms):
            nuclei = [(float(Z), np.asarray(R)) for Z, R in
                      zip(atoms.get_atomic_numbers(), atoms.get_positions())]
            grid = Grid(center=[0, 0, 0], box_size=5.0, grid_spacing=0.35)
            H = MolecularIntegrals(
                nuclei, minimal_hao_basis(nuclei), grid
            ).molecular_hamiltonian(mo_basis=True, n_electrons=2)
            return H, (1, 1), 2

        atoms = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]])
        atoms.calc = Mandacaru(method="vqe", hamiltonian_builder=builder,
                               trace=False)
        assert np.isfinite(atoms.get_total_energy())

    def test_uccsd_is_the_default_ansatz_and_reaches_fci_on_lih(self):
        """What ``examples/vqe/01_vqe_lih.py`` relies on, in one assertion.

        A calculator-mode ``method="vqe"`` needs nothing but a geometry: the
        ansatz is built from the problem's own shape as every single and double
        excitation out of the Hartree-Fock determinant, and on a 6-qubit active
        space that is enough to be exact.
        """
        atoms = Atoms("LiH", positions=[(0, 0, 0), (0, 0, 1.6)],
                      cell=[10.0] * 3)
        atoms.center()
        atoms.calc = Mandacaru(method="vqe", basis="HAO", grid_spacing=0.30, trace=False)
        energy = atoms.get_total_energy()

        assert isinstance(atoms.calc.solver.ansatz, UCCSD)
        matrix = atoms.calc.hamiltonian.to_matrix()
        exact = float(from_hartree(
            np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T)).min(), "eV"))
        assert energy == pytest.approx(exact, abs=1e-8)
        # Both currencies are reported, as they are for ADAPT-VQE.
        result = atoms.calc.result
        assert result.num_evaluations > 0
        assert result.optimizer_steps is not None
        assert result.optimizer_steps <= result.num_evaluations

    def test_missing_cell_raises(self):
        atoms = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]])  # no cell
        atoms.calc = Mandacaru(method="vqe", basis="HAO", trace=False)
        with pytest.raises(ValueError, match="no unit cell"):
            atoms.get_total_energy()

    def test_ibm_quantum_device_requires_shots(self):
        with pytest.raises(ValueError, match="shots > 0"):
            Mandacaru(method="vqe", basis="HAO", device="ibm-quantum",
                      trace=False)


# --- Timing / memory / cores in the result and the summary (requirements 2-3) ---

class TestVQEProfiling:
    def test_result_carries_timings(self, h2_hamiltonian):
        res = Mandacaru(method="vqe", hamiltonian=h2_hamiltonian,
                        ansatz=UCCSD(2, (1, 1)), trace=False).run()
        assert res.timings is not None
        assert "parameter optimization" in res.timings["stages_s"]
        assert res.timings["peak_memory_mb"] > 0.0

    def test_summary_shows_timings_and_resources(self, h2_hamiltonian, capsys):
        Mandacaru(method="vqe", hamiltonian=h2_hamiltonian,
                  ansatz=UCCSD(2, (1, 1)), trace=True).run()
        out = capsys.readouterr().out
        assert "[PERFORMANCE]" in out
        assert "parameter optimization" in out
        assert "openmp_threads" in out
        assert "peak_memory_MiB" in out

    def test_calculator_summary_includes_integration(self, capsys):
        atoms = Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
                      cell=[[6, 0, 0], [0, 6, 0], [0, 0, 6]], pbc=True)
        atoms.calc = Mandacaru(method="vqe", basis="HAO", optimizer=COBYLA_OPT,
                               grid_spacing=0.4, trace=True)
        atoms.get_total_energy()
        out = capsys.readouterr().out
        assert "integration:" in out              # integration stage is timed
        assert atoms.calc.result.integration_profile is not None


class TestVQERunLog:
    """``txt=`` writes the shared run log; a fixed ansatz has no table."""

    @pytest.fixture(scope="class")
    def log(self, tmp_path_factory):
        from mandacaru.utils.logging import parse_output, reset_log

        path = str(tmp_path_factory.mktemp("vqe") / "out.txt")
        atoms = Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
                      cell=[6.0] * 3)
        atoms.calc = Mandacaru(method="vqe", basis="HAO", grid_spacing=0.40,
                               optimizer=COBYLA_OPT, txt=path)
        atoms.get_total_energy()
        reset_log(path)
        with open(path, encoding="utf-8") as fh:
            return atoms.calc.result, fh.read(), parse_output(path)

    def test_every_shared_block_is_written(self, log):
        _result, text, parsed = log
        for block in ("[SYSTEM]", "[BASIS]", "[ELECTRONS]",
                      "[OPTIMIZATION SETUP]", "[VARIATIONAL QUANTUM SUMMARY]",
                      "[PERFORMANCE]"):
            assert block in text
        assert "VQE (UCCSD)" in text
        assert parsed["electrons"]["qubits"] == "4"

    def test_a_fixed_ansatz_has_no_progress_table(self, log):
        _result, text, parsed = log
        assert "[ITERATIONS]" not in text and "[MARKOV CHAIN]" not in text
        assert parsed["iterations"] == []

    def test_the_setup_names_the_ansatz(self, log):
        _result, _text, parsed = log
        setup = parsed["setup"]
        assert setup["classical_optimizer"] == "COBYLA"
        assert setup["ansatz"] == "UCCSD"
        assert setup["initial_parameters"] == "all zero (the reference state)"
        assert setup["parameter_gradient"] == \
            "none (COBYLA uses no gradient)"

    def test_the_summary_is_the_result(self, log):
        result, _text, parsed = log
        summary = parsed["summary"]
        assert float(summary["optimal_energy_eV"]) == pytest.approx(
            result.optimal_energy, abs=1e-9)
        assert int(summary["num_parameters"]) == result.num_parameters
        assert int(summary["cost_evaluations"]) == result.num_evaluations
        # A fixed ansatz grows no operator sequence.
        assert "num_operators" not in summary

    def test_the_file_is_the_whole_report(self, log, tmp_path, capsys):
        """With ``txt=`` standard output stays quiet, as for ADAPT-VQE."""
        atoms = Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
                      cell=[6.0] * 3)
        atoms.calc = Mandacaru(method="vqe", basis="HAO", grid_spacing=0.45,
                               optimizer=COBYLA_OPT,
                               txt=str(tmp_path / "quiet.txt"))
        capsys.readouterr()
        atoms.get_total_energy()
        assert "[SYSTEM]" not in capsys.readouterr().out


# --- Monkhorst-Pack k-points (requirement: kpts via ASE) ---

class TestVQEKPoints:
    def test_default_is_gamma(self, h2_hamiltonian):
        vqe = Mandacaru(method="vqe", hamiltonian=h2_hamiltonian,
                        ansatz=UCCSD(2, (1, 1)))
        assert vqe.solver.kpts == (1, 1, 1) and len(vqe.kpoints) == 1

    def test_mesh_from_ase(self):
        from ase.dft.kpoints import monkhorst_pack
        vqe = Mandacaru(method="vqe", basis="HAO", kpts=(3, 1, 1))
        np.testing.assert_allclose(vqe.kpoints, monkhorst_pack((3, 1, 1)))

    def test_non_gamma_rejected_at_run(self, h2_hamiltonian):
        vqe = Mandacaru(method="vqe", hamiltonian=h2_hamiltonian,
                        ansatz=UCCSD(2, (1, 1)), kpts=(2, 2, 2), trace=False)
        with pytest.raises(NotImplementedError, match="Monkhorst-Pack"):
            vqe.run()

    def test_dict_gamma_centering(self):
        vqe = Mandacaru(method="vqe", basis="HAO",
                        kpts={"size": (2, 2, 2), "gamma": True})
        assert vqe.solver.kpts == (2, 2, 2) and vqe.solver.kpts_gamma is True
        assert any(np.allclose(k, [0, 0, 0]) for k in vqe.kpoints)


# --- Spin and initial-state arguments ---

class TestVQESpinAndInitialState:
    def test_spin_default_false(self, h2_hamiltonian):
        assert Mandacaru(method="vqe", hamiltonian=h2_hamiltonian,
                         ansatz=UCCSD(2, (1, 1))).solver.spin is False

    def test_spin_flag_stored(self, h2_hamiltonian):
        assert Mandacaru(method="vqe", hamiltonian=h2_hamiltonian,
                         ansatz=UCCSD(2, (1, 1)), spin=True).solver.spin is True

    def test_initial_state_default_and_validation(self, h2_hamiltonian):
        assert Mandacaru(method="vqe", hamiltonian=h2_hamiltonian,
                         ansatz=UCCSD(2, (1, 1))).solver.initial_state == "hartree-fock"
        with pytest.raises(ValueError):
            Mandacaru(method="vqe", hamiltonian=h2_hamiltonian,
                      ansatz=UCCSD(2, (1, 1)), initial_state="excited")


class TestTheShotObjectiveIsTheDeclaredCircuit:
    """With ``shots > 0`` every objective evaluation is a circuit, and it goes
    through the same validated export as the final measurement.  It used to
    hand the provider the logical angles and zip them against the generators:
    a one-angle Trotter HVA with several compiled rotations submitted only
    the first, and a prebuilt exact-UCC ansatz was run as the ordered
    product it does not declare."""

    class ExactCircuit:
        """A provider whose "measurement" is the executed circuit's exact
        expectation value, so any difference is the circuit, not noise."""

        def energy(self, n, occupied, generators, theta, hamiltonian):
            from mandacaru.backends.providers import QiskitProvider
            psi = QiskitProvider().statevector(n, occupied, generators, theta)
            return float(np.vdot(psi, hamiltonian.to_matrix() @ psi).real)

    def test_a_prebuilt_trotter_hva_submits_its_whole_circuit(self):
        from mandacaru.circuits.hva import HamiltonianVariationalAnsatz
        from mandacaru.core.mapping import Fermion

        n0 = Fermion.creation(0) * Fermion.annihilation(0)
        hop = Fermion.creation(0) * Fermion.annihilation(1)
        fermion = n0 + hop + Fermion.creation(1) * Fermion.annihilation(0)
        ansatz = HamiltonianVariationalAnsatz(
            fermion, (1, 0), mapping="jordan_wigner", layers=1,
            evolution="trotter", order=2, steps=2)
        assert len(ansatz.pauli_generators) > ansatz.num_parameters
        qubit_h = fermion.map_to_qubits("jordan_wigner", n_modes=2)
        calc = Mandacaru(method="vqe", hamiltonian=qubit_h, ansatz=ansatz,
                         shots=1000, trace=False)
        solver = calc.solver
        solver.circuit_provider = lambda: self.ExactCircuit()
        theta = np.array([0.71])
        assert solver.ansatz_energy(ansatz, theta) == pytest.approx(
            solver.energy(ansatz.state(theta)), abs=1e-10)

    def test_a_prebuilt_exact_ucc_is_refused_with_shots(self):
        hamiltonian = PauliSum({"ZIII": 1.0, "XXII": 0.3}, num_qubits=4)
        with pytest.raises(ValueError, match="ordered product"):
            Mandacaru(method="vqe", hamiltonian=hamiltonian,
                      ansatz=UCCSD(2, (1, 1), trotter=False), shots=1000,
                      trace=False).solver


class TestAShotRunBuildsNoStateVector:
    """With shots every energy is a measured circuit, yet the driver used to
    build the Hamiltonian matrix (and meet its state-memory guard) and the
    UCCSD every dense generator -- 256 MiB each at 12 qubits -- before
    submitting anything.  Here every dense constructor is forbidden."""

    def test_vqe_with_shots_never_materializes(self, monkeypatch):
        from mandacaru.core import mapping

        def forbidden(*_args, **_kwargs):
            raise AssertionError("a shot-based run built a 2^N matrix")

        hamiltonian = PauliSum({"ZIII": 1.0, "IZII": -0.5, "XXYY": 0.2,
                                "IIII": -0.3}, num_qubits=4)
        calc = Mandacaru(method="vqe", hamiltonian=hamiltonian,
                         ansatz=UCCSD(2, (1, 1), trotter=True), shots=1000,
                         optimizer={"method": "SPSA", "maxiter": 3},
                         backend_options={"seed": 7}, trace=False)
        monkeypatch.setattr(mapping.PauliSum, "to_matrix", forbidden)
        monkeypatch.setattr(mapping.PauliSum, "to_sparse_matrix", forbidden)
        result = calc.solver.run()
        assert np.isfinite(result.optimal_energy)
        # The reference energy came from the determinant: |1010> here.
        assert calc.solver.reference_energy() == pytest.approx(
            -1.0 - 0.5 - 0.3)

    def test_fifty_qubits(self, monkeypatch):
        """A 50-qubit UCCSD on a device double: every energy measured, the
        reference from the determinant, no sector, and every exponential
        constructor forbidden; the classical memory stays under a gigabyte."""
        import tracemalloc

        from .test_adapt_vqe import (PolynomialDevice, _nothing_exponential,
                                     _sparse_molecular_fermion)

        fermion = _sparse_molecular_fermion(25)
        device = PolynomialDevice()
        with _nothing_exponential(monkeypatch):
            tracemalloc.start()
            try:
                calc = Mandacaru(method="vqe", hamiltonian=fermion,
                                 num_particles=(1, 1), n_spatial_orbitals=25,
                                 ansatz={"name": "uccsd", "trotter": True},
                                 shots=1000, trace=False,
                                 optimizer={"method": "SPSA", "maxiter": 3})
                solver = calc.solver
                solver.circuit_provider = lambda: device
                result = solver.run()
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
        assert solver.n_qubits == 50 and solver._sector is None
        assert np.isfinite(result.optimal_energy)
        assert device.energies >= 6
        assert peak < 2 ** 30, f"peak {peak / 2 ** 20:.0f} MiB"


# --- the particle-number sector (the dense 2^N matrices of a 12-qubit run) ---

def _random_molecular_fermion(n_orbitals, seed=3):
    """A random Hermitian two-body Hamiltonian with real-orbital symmetry."""
    from mandacaru.core.mapping import Fermion

    rng = np.random.default_rng(seed)
    h = rng.standard_normal((n_orbitals,) * 2)
    h = np.diag(np.linspace(-3, 3, n_orbitals)) + 0.1 * (h + h.T)
    chemist = np.zeros((n_orbitals,) * 4)
    for _ in range(3):
        L = rng.standard_normal((n_orbitals,) * 2)
        L = (L + L.T) / 2
        chemist += np.einsum("pr,qs->prqs", L, L)
    return Fermion.from_spatial_integrals(h, 0.05 * chemist.transpose(0, 2, 1, 3))


class TestVQEInTheSector:
    """``method="vqe"`` simulates the ``(n_alpha, n_beta)`` sector, as ADAPT does."""

    def _calc(self, fermion, n_orbitals, **options):
        return Mandacaru(method="vqe", hamiltonian=fermion,
                         num_particles=(2, 2), n_spatial_orbitals=n_orbitals,
                         trace=False, **options)

    @pytest.mark.parametrize("trotter", [False, True])
    def test_energies_equal_the_register_simulation(self, trotter):
        """8 qubits, the same random angles: sector against the 256-state
        register, in the exact and in the ordered-product ansatz."""
        fermion = _random_molecular_fermion(4)
        ansatz = {"name": "uccsd", "trotter": trotter}
        full = self._calc(fermion, 4, sector=False, ansatz=ansatz).solver
        sector = self._calc(fermion, 4, sector=True, ansatz=ansatz).solver
        assert full._sector is None and sector._sector.dim == 36
        rng = np.random.default_rng(1)
        for _ in range(3):
            theta = rng.uniform(-1, 1, full.ansatz.num_parameters)
            assert sector.energy_at(theta) == pytest.approx(
                full.energy_at(theta), abs=1e-11)
        assert sector.reference_energy() == pytest.approx(
            full.reference_energy(), abs=1e-12)

    def test_an_optimized_run_matches_the_register_run(self):
        fermion = _random_molecular_fermion(4)
        optimizer = {"method": "L-BFGS", "maxiter": 200, "tol": 1e-10}
        energies = [self._calc(fermion, 4, sector=flag,
                               optimizer=optimizer).run().optimal_energy
                    for flag in (False, True)]
        assert energies[1] == pytest.approx(energies[0], abs=1e-9)

    def test_the_sector_is_automatic_from_ten_qubits(self):
        small = self._calc(_random_molecular_fermion(4), 4).solver
        large = self._calc(_random_molecular_fermion(5), 5).solver
        assert small._sector is None and large._sector.dim == 100
        assert "particle-number sector, 100 states" in \
            large._backend_description()

    def test_a_shot_run_builds_no_sector(self):
        """With shots every energy is measured on the device, so the automatic
        choice builds no sector (whose enumeration and leak check would cost
        the sector dimension before anything is submitted), and an explicit
        one is refused."""
        fermion = _random_molecular_fermion(5)
        calc = self._calc(fermion, 5, shots=1000)
        assert calc.solver._sector is None
        with pytest.raises(ValueError, match="sector=True"):
            self._calc(fermion, 5, shots=1000, sector=True).solver

    def test_a_sector_too_large_to_enumerate_is_not_a_setup_error(
            self, monkeypatch):
        """The automatic choice leaves an oversized sector to the register
        path and says so; only an explicit ``sector=True`` raises."""
        from mandacaru.core import sector as sector_module

        monkeypatch.setattr(sector_module, "MAX_SECTOR_DIMENSION", 50)
        fermion = _random_molecular_fermion(5)
        with pytest.warns(RuntimeWarning, match="full register"):
            solver = self._calc(fermion, 5).solver
        assert solver._sector is None
        with pytest.raises(ValueError, match="limit 50"):
            self._calc(fermion, 5, sector=True).solver

    def test_a_twelve_qubit_run_allocates_no_register_matrix(self):
        """Dense: 92 generators of 268 MB, a 268 MB Hamiltonian.  Sector: a
        few megabytes (the ceiling is a hundredth of one dense matrix)."""
        import tracemalloc

        fermion = _random_molecular_fermion(6)
        tracemalloc.start()
        try:
            solver = self._calc(fermion, 6).solver
            theta = np.full(solver.ansatz.num_parameters, 0.02)
            energy = solver.energy_at(theta)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert solver._sector.dim == 225
        assert solver.energy_at(np.zeros_like(theta)) == pytest.approx(
            solver.reference_energy(), abs=1e-12)
        assert np.isfinite(energy)
        assert peak < 16 * 2 ** 20

    def test_the_ordered_product_uses_the_analytic_gradient(self):
        fermion = _random_molecular_fermion(5)
        calc = self._calc(fermion, 5, ansatz={"name": "uccsd", "trotter": True},
                          optimizer={"method": "L-BFGS", "maxiter": 60,
                                     "tol": 1e-9})
        assert calc.solver._energy_gradient(calc.solver.ansatz) is not None
        exact = self._calc(fermion, 5)
        assert exact.solver._energy_gradient(exact.solver.ansatz) is None

    def test_a_hamiltonian_that_leaves_the_sector_keeps_the_register(
            self, monkeypatch):
        from mandacaru.core.mapping import PauliSum
        from mandacaru.circuits import UCCSD

        leaking = (_random_molecular_fermion(4).map_to_qubits("jordan_wigner")
                   + PauliSum({"X" + "I" * 7: 0.2}, num_qubits=8))
        calc = Mandacaru(method="vqe", hamiltonian=leaking,
                         ansatz=UCCSD(4, (2, 2)), sector="auto", trace=False)
        # Lower the automatic threshold so that a cheap 8-qubit run meets it.
        monkeypatch.setattr(calc._solver_class, "SECTOR_AUTO_QUBITS", 8)
        with pytest.warns(RuntimeWarning, match="full register"):
            solver = calc.solver
        assert solver._sector is None
        with pytest.raises(ValueError, match="sector=True"):
            Mandacaru(method="vqe", hamiltonian=leaking,
                      ansatz=UCCSD(4, (2, 2)), sector=True,
                      trace=False).solver

    def test_only_the_uccsd_ansatz_runs_in_a_sector(self):
        fermion = _random_molecular_fermion(4)
        with pytest.raises(NotImplementedError, match="sector"):
            self._calc(fermion, 4, sector=True, ansatz="hva").solver
        with pytest.raises(ValueError, match="sector"):
            self._calc(fermion, 4, sector="sometimes")

    def test_the_subspace_solver_keeps_the_register(self, monkeypatch):
        """Its reference states are built on the full register, so it neither
        chooses a sector by itself nor accepts one."""
        fermion = _random_molecular_fermion(2)
        options = dict(method="subspace-vqe", hamiltonian=fermion,
                       num_particles=(1, 1), n_spatial_orbitals=2, trace=False)
        calc = Mandacaru(**options)
        monkeypatch.setattr(calc._solver_class, "SECTOR_AUTO_QUBITS", 4)
        assert calc.solver._sector is None
        with pytest.raises(NotImplementedError, match="sector"):
            Mandacaru(sector=True, **options).solver

    def test_the_ordered_product_optimizes_with_the_adjoint_gradient(
            self, monkeypatch):
        """Same minimum as the differenced run, in far fewer evaluations, and
        the setup block says which gradient ran."""
        fermion = _random_molecular_fermion(4)
        ansatz = {"name": "uccsd", "trotter": True}
        optimizer = {"method": "L-BFGS", "maxiter": 300, "tol": 1e-10}

        def run(differenced):
            calc = self._calc(fermion, 4, sector=True, ansatz=ansatz,
                              optimizer=optimizer)
            if differenced:
                monkeypatch.setattr(calc.solver, "_energy_gradient",
                                    lambda ansatz: None)
            return calc, calc.run()

        analytic_calc, analytic = run(False)
        differenced_calc, differenced = run(True)
        assert analytic.optimal_energy == pytest.approx(
            differenced.optimal_energy, abs=1e-7)
        assert analytic.num_evaluations < differenced.num_evaluations / 5
        for calc, kind in ((analytic_calc, "analytic"),
                           (differenced_calc, "finite difference")):
            assert calc.solver._parameter_gradient_text(
                calc.solver.ansatz).startswith(kind)
