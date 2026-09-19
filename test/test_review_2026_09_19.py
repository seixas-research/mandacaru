# -*- coding: utf-8 -*-
# file: test_review_2026_09_19.py

"""Regressions for the review of 2026-09-19 (findings R01-R08).

One class per finding, each reproducing the defect the way the review did and
asserting the repaired behavior.  Every solver is reached through
:class:`~mandacaru.Mandacaru`, the single entry point.
"""

import json
import os

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.backends.providers import (QiskitProvider, pauli_rotations,
                                          pauli_strings_commute)
from mandacaru.circuits import UCCSD, Ansatz, SerializableAnsatz
from mandacaru.circuits.pools import build_pool
from mandacaru.cli import build_parser, main, solver_options
from mandacaru.core import (PauliSum, WavefunctionCheckpoint, load_hamiltonian,
                            read_hamiltonian_header, save_hamiltonian)
from mandacaru.optimizers.optim import OptimizeResult, Optimizer

HAMILTONIAN = PauliSum({"ZIII": 1.0, "IXII": 0.4})
ANGLES = np.array([0.4, -0.7, 0.6])


def h2():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[6.0] * 3)
    atoms.center()
    return atoms


class KeepInitial(Optimizer):
    """A zero-update optimizer: the result is the state at the given angles."""

    def minimize(self, cost, x0, callback=None):
        x = np.asarray(x0, dtype=float)
        energy = cost(x)
        return OptimizeResult(x, energy, 1, [energy], False, "zero budget")


# --------------------------------------------------------------------------- #
# R01 -- exp(sum) and prod(exp) are different states; the checkpoint says which.
# --------------------------------------------------------------------------- #

class TestR01CheckpointRecordsThePreparation:
    @pytest.mark.parametrize("trotter, form", [(False, "sum"), (True, "product")])
    def test_the_checkpoint_is_the_state_that_was_optimized(self, trotter, form):
        ansatz = UCCSD(2, (1, 1), trotter=trotter)
        calc = Mandacaru(method="vqe", hamiltonian=HAMILTONIAN, ansatz=ansatz,
                         optimizer=KeepInitial(), trace=False,
                         atomic_units=True)
        result = calc.run(initial_parameters=ANGLES)
        record = calc.checkpoint
        assert record.preparation == form
        fidelity = abs(np.vdot(ansatz.state(ANGLES), record.state_vector())) ** 2
        assert fidelity == pytest.approx(1.0, abs=1e-12)    # it was 0.9422
        assert record.expectation() == pytest.approx(result.optimal_energy,
                                                     abs=1e-12)

    def test_the_two_forms_really_differ_here(self):
        """Several non-zero angles, or the test could not see the defect."""
        exact = UCCSD(2, (1, 1)).state(ANGLES)
        product = UCCSD(2, (1, 1), trotter=True).state(ANGLES)
        assert abs(np.vdot(exact, product)) ** 2 < 0.99

    def test_the_form_survives_the_file(self, tmp_path):
        path = str(tmp_path / "state.json")
        Mandacaru(method="vqe", hamiltonian=HAMILTONIAN, ansatz=UCCSD(2, (1, 1)),
                  optimizer=KeepInitial(), trace=False,
                  checkpoint=path).run(initial_parameters=ANGLES)
        assert json.loads(open(path).read())["preparation"] == "sum"
        loaded = WavefunctionCheckpoint.load(path)
        assert loaded.preparation == "sum"
        assert loaded.expectation() == pytest.approx(loaded.energy, abs=1e-12)

    def test_a_version_1_file_is_a_product(self):
        payload = WavefunctionCheckpoint(1, [], [PauliSum({"Y": 1j})],
                                         [0.3]).to_payload()
        del payload["preparation"]
        assert WavefunctionCheckpoint.from_payload(payload).preparation == "product"

    def test_an_exact_ucc_state_has_no_circuit(self):
        record = WavefunctionCheckpoint(
            4, [0, 2], UCCSD(2, (1, 1)).pauli_generators, ANGLES,
            preparation="sum")
        with pytest.raises(ValueError, match="product of exponentials"):
            record.circuit()
        with pytest.raises(ValueError, match="product of exponentials"):
            record.problem()

    def test_one_generator_is_both_forms(self):
        record = WavefunctionCheckpoint(1, [], [PauliSum({"Y": 1j})], [0.3],
                                        preparation="sum")
        assert record.is_product and record.problem()[0] == 1

    def test_resuming_into_the_other_form_is_refused(self, tmp_path):
        path = str(tmp_path / "exact.json")
        Mandacaru(method="vqe", hamiltonian=HAMILTONIAN, ansatz=UCCSD(2, (1, 1)),
                  optimizer=KeepInitial(), trace=False,
                  checkpoint=path).run(initial_parameters=ANGLES)
        trotter = Mandacaru(method="vqe", hamiltonian=HAMILTONIAN,
                            ansatz=UCCSD(2, (1, 1), trotter=True),
                            trace=False, resume=path)
        with pytest.raises(ValueError, match="exact-UCC state"):
            trotter.run()

    def test_measuring_an_exact_ucc_state_is_refused(self):
        calc = Mandacaru(method="vqe", hamiltonian=HAMILTONIAN,
                         ansatz=UCCSD(2, (1, 1)), optimizer=KeepInitial(),
                         trace=False)
        calc.run(initial_parameters=ANGLES)
        with pytest.raises(ValueError, match="trotter=True"):
            calc.measured_energy(QiskitProvider())

    def test_qpe_takes_an_exact_ucc_checkpoint_as_a_state(self):
        from mandacaru.algorithms import QuantumPhaseEstimation
        record = WavefunctionCheckpoint(
            4, [0, 2], UCCSD(2, (1, 1)).pauli_generators, ANGLES,
            hamiltonian=HAMILTONIAN, preparation="sum")
        qpe = QuantumPhaseEstimation(n_evaluation_qubits=4, verbose=False)
        psi, n, _h, problem = qpe._resolve_input(record)
        assert n == 4 and problem is None
        assert np.allclose(psi, UCCSD(2, (1, 1)).state(ANGLES))


# --------------------------------------------------------------------------- #
# R02 -- the collision check sees every path, for every method.
# --------------------------------------------------------------------------- #

class TestR02OutputCollisions:
    @pytest.mark.parametrize("method, extra", [
        ("vqe", {"ansatz": UCCSD(2, (1, 1))}),
        ("adapt-vqe", {"num_particles": (1, 1), "n_spatial_orbitals": 2}),
    ])
    def test_checkpoint_may_not_share_the_cache_path(self, tmp_path, method,
                                                     extra):
        path = str(tmp_path / "same.json")
        with pytest.raises(ValueError, match="both resolve to"):
            Mandacaru(method=method, hamiltonian=HAMILTONIAN, trace=False,
                      save_hamiltonian=path, checkpoint=path,
                      hamiltonian_format="json", **extra)
        assert not os.path.exists(path)              # refused before any write

    def test_a_symlinked_directory_is_still_the_same_file(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        alias = tmp_path / "alias"
        alias.symlink_to(real, target_is_directory=True)
        with pytest.raises(ValueError, match="both resolve to"):
            Mandacaru(method="vqe", hamiltonian=HAMILTONIAN,
                      ansatz=UCCSD(2, (1, 1)), trace=False,
                      save_hamiltonian=str(real / "h.json"),
                      checkpoint=str(alias / "h.json"))

    def test_distinct_paths_are_fine(self, tmp_path):
        calc = Mandacaru(method="vqe", hamiltonian=HAMILTONIAN,
                         ansatz=UCCSD(2, (1, 1)), trace=False,
                         optimizer=KeepInitial(),
                         save_hamiltonian=str(tmp_path / "h.json"),
                         checkpoint=str(tmp_path / "state.json"))
        calc.run(initial_parameters=ANGLES)
        assert load_hamiltonian(str(tmp_path / "h.json")).num_qubits == 4
        assert WavefunctionCheckpoint.load(str(tmp_path / "state.json"))


# --------------------------------------------------------------------------- #
# R03 / R04 -- the subspace solvers: forces, and exporting the right branch.
# --------------------------------------------------------------------------- #

class TestR03SubspaceForces:
    @pytest.mark.parametrize("method, options", [
        ("subspace-vqe", {}),
        ("subspace-adapt-vqe", {"max_iterations": 0, "profile": False}),
    ])
    def test_forces_use_the_ground_state_energy(self, method, options):
        atoms = h2()
        atoms.calc = Mandacaru(method=method, h=0.4, trace=False, **options)
        forces = atoms.get_forces()                   # it raised ValueError
        assert forces.shape == (2, 3) and np.all(np.isfinite(forces))
        assert forces[0, 2] == pytest.approx(-forces[1, 2], abs=1e-6)


class TestR04SubspaceExportFollowsTheSortedLevels:
    @pytest.fixture
    def swapped(self):
        """H = -Z_2 puts the *second* reference below Hartree-Fock."""
        calc = Mandacaru(method="subspace-vqe", hamiltonian=PauliSum({"IIZI": -1.0}),
                         ansatz=UCCSD(2, (1, 1), trotter=True), num_states=2,
                         optimizer=KeepInitial(), trace=False, atomic_units=True)
        calc.run()
        return calc

    def test_the_ground_state_is_not_the_hartree_fock_branch(self, swapped):
        assert swapped.result.energies.tolist() == [-1.0, 1.0]
        hf = UCCSD(2, (1, 1)).reference_qubits()
        assert swapped.ansatz_problem()[1] != hf
        assert swapped.ansatz_problem(state=1)[1] == hf

    def test_the_measured_energy_is_the_reported_one(self, swapped):
        provider = QiskitProvider()
        assert swapped.measured_energy(provider) == pytest.approx(-1.0)   # was +1
        assert swapped.measured_energy(provider, state=1) == pytest.approx(1.0)

    def test_an_unknown_level_is_refused(self, swapped):
        with pytest.raises(IndexError):
            swapped.ansatz_problem(state=2)

    def test_the_grown_ansatz_is_kept_for_export(self):
        atoms = h2()
        atoms.calc = Mandacaru(method="subspace-adapt-vqe", h=0.4, trace=False,
                               profile=False, max_iterations=2)
        atoms.get_potential_energy()
        n, occupied, generators, theta, _h = atoms.calc.ansatz_problem()
        assert n == 4 and len(generators) == len(theta)   # it raised AttributeError


# --------------------------------------------------------------------------- #
# R05 -- an ansatz that only evaluates states still runs.
# --------------------------------------------------------------------------- #

class Rotation:
    """The documented structural protocol and nothing more."""

    n_qubits = 1
    num_parameters = 1

    def reference_state(self):
        return np.array([1.0, 0.0])

    def state(self, theta):
        return np.array([np.cos(theta[0]), np.sin(theta[0])])

    def evolve(self, theta, references):
        c, s = np.cos(theta[0]), np.sin(theta[0])
        return np.array([[c, -s], [s, c]]) @ references


class TestR05CustomAnsatz:
    def test_the_protocols(self):
        assert isinstance(Rotation(), Ansatz)
        assert not isinstance(Rotation(), SerializableAnsatz)
        assert isinstance(UCCSD(2, (1, 1)), SerializableAnsatz)

    def test_it_runs_without_a_checkpoint(self):
        calc = Mandacaru(method="vqe", hamiltonian=PauliSum({"Z": 1.0}),
                         ansatz=Rotation(), trace=False, atomic_units=True)
        result = calc.run()
        assert result.optimal_energy == pytest.approx(-1.0, abs=1e-4)
        assert calc.checkpoint is None

    def test_asking_for_a_checkpoint_is_refused_before_the_optimization(
            self, tmp_path):
        evaluations = []

        class Counting(Rotation):
            def state(self, theta):
                evaluations.append(1)
                return super().state(theta)

        calc = Mandacaru(method="vqe", hamiltonian=PauliSum({"Z": 1.0}),
                         ansatz=Counting(), trace=False,
                         checkpoint=str(tmp_path / "state.json"))
        with pytest.raises(TypeError, match="SerializableAnsatz"):
            calc.run()
        assert not evaluations


# --------------------------------------------------------------------------- #
# R06 -- estimating never materializes what it estimates.
# --------------------------------------------------------------------------- #

class TestR06DryRunIsDry:
    @pytest.fixture
    def no_configure(self, monkeypatch):
        from mandacaru.algorithms.base import VariationalDriver

        def refuse(self, *_a, **_k):
            raise AssertionError("the problem was materialized")
        monkeypatch.setattr(VariationalDriver, "_materialize_hamiltonian", refuse)

    def test_constructing_the_calculator_materializes_nothing(self, no_configure):
        Mandacaru(method="vqe", hamiltonian=HAMILTONIAN, ansatz=UCCSD(2, (1, 1)),
                  trace=False)

    def test_the_one_off_estimate_materializes_nothing(self, no_configure):
        calc = Mandacaru(method="vqe", hamiltonian=HAMILTONIAN,
                         ansatz=UCCSD(2, (1, 1)), trace=False)
        estimate = calc.dry_run()
        assert estimate.n_qubits == 4 and calc.dry_run_result is estimate
        assert not calc.solver._configured

    def test_the_calculator_still_runs_afterwards(self):
        calc = Mandacaru(method="vqe", hamiltonian=HAMILTONIAN,
                         ansatz=UCCSD(2, (1, 1)), trace=False,
                         optimizer=KeepInitial())
        calc.dry_run()
        assert np.isfinite(calc.run(initial_parameters=ANGLES).optimal_energy)

    def test_options_are_still_refused_by_the_constructor(self):
        with pytest.raises(ValueError, match="gradient"):
            Mandacaru(method="adapt-vqe", gradient="nope")
        with pytest.raises(ValueError, match="shots > 0"):
            Mandacaru(method="vqe", device="braket-ionq-aria")
        Mandacaru(method="vqe", device="braket-ionq-aria", dry_run=True)

    @pytest.mark.parametrize("name", ["h.parquet", "h.json"])
    def test_a_cache_is_estimated_from_its_header(self, tmp_path, monkeypatch,
                                                  name):
        import mandacaru.core.serialization as serialization
        path = str(tmp_path / name)
        save_hamiltonian(path, HAMILTONIAN, num_particles=(1, 1),
                         n_spatial_orbitals=2)
        header = read_hamiltonian_header(path)
        assert (header.num_qubits, header.num_particles, header.n_terms) == \
            (4, (1, 1), 2)

        def refuse(*_a, **_k):
            raise AssertionError("the Pauli table was loaded")
        monkeypatch.setattr(serialization, "load_hamiltonian", refuse)
        if name.endswith(".parquet"):              # the footer is all it needs
            monkeypatch.setattr(serialization, "_read_fastparquet", refuse)
        calc = Mandacaru(method="adapt-vqe", load_hamiltonian=path, trace=False)
        assert calc.dry_run().n_qubits == 4

    def test_a_missing_cache_is_refused_by_the_constructor(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Mandacaru(method="adapt-vqe",
                      load_hamiltonian=str(tmp_path / "absent.parquet"))


# --------------------------------------------------------------------------- #
# R07 -- a product of rotations is exact only for commuting terms.
# --------------------------------------------------------------------------- #

class TestR07CommutingTermsOnly:
    def test_the_pairwise_rule(self):
        assert pauli_strings_commute("XX", "YY")           # two clashes
        assert not pauli_strings_commute("XI", "ZI")       # one clash
        assert pauli_strings_commute("XI", "IZ")           # disjoint

    def test_an_anticommuting_generator_is_refused(self):
        generator = PauliSum({"X": 0.3j, "Z": 0.2j})
        with pytest.raises(ValueError, match="anticommute"):
            pauli_rotations(generator)
        record = WavefunctionCheckpoint(1, [], [generator], [1.7])
        with pytest.raises(ValueError, match="anticommute"):   # fidelity 0.983
            QiskitProvider().statevector(*record.problem()[:4])
        assert np.linalg.norm(record.state_vector()) == pytest.approx(1.0)

    @pytest.mark.parametrize("mapping", ["jordan_wigner", "parity",
                                         "bravyi_kitaev"])
    @pytest.mark.parametrize("pool", ["fermionic", "qubit", "qeb"])
    def test_these_pools_are_exportable_in_every_mapping(self, pool, mapping):
        for operator in build_pool(pool, 3, (2, 1), mapping=mapping).operators():
            assert pauli_rotations(operator.generator)

    def test_ceo_is_exportable_only_where_it_is_qeb(self):
        """Under parity / Bravyi-Kitaev a CEO generator sums several qubit
        excitations of one support, and those anticommute: the state-vector
        backend exponentiates it exactly, a circuit cannot (no CEO synthesis is
        implemented), so export is refused rather than silently approximated."""
        for operator in build_pool("ceo", 3, (2, 1),
                                   mapping="jordan_wigner").operators():
            assert pauli_rotations(operator.generator)
        refused = 0
        for mapping in ("parity", "bravyi_kitaev"):
            for operator in build_pool("ceo", 3, (2, 1),
                                       mapping=mapping).operators():
                try:
                    pauli_rotations(operator.generator)
                except ValueError:
                    refused += 1
        assert refused > 0


# --------------------------------------------------------------------------- #
# R08 -- the command line forwards what was typed.
# --------------------------------------------------------------------------- #

class TestR08CommandLineOptions:
    def _options(self, *argv):
        return solver_options(build_parser().parse_args(
            ["H2", "--cell", "6", *argv]))

    def test_typed_options_are_forwarded_for_every_method(self):
        options = self._options("--method", "vqe", "--output", "out.txt",
                                "--max-iterations", "0")
        assert options["output"] == "out.txt" and options["max_iterations"] == 0

    def test_an_adaptive_method_gets_the_default_pool(self):
        assert self._options("--method", "adapt-vqe")["pool"] == "fermionic"
        assert "pool" not in self._options("--method", "vqe")

    @pytest.mark.parametrize("flag", [["--output", "o.txt"],
                                      ["--max-iterations", "3"],
                                      ["--pool", "qeb"]])
    def test_an_option_the_method_ignores_is_a_usage_error(self, flag, capsys):
        with pytest.raises(SystemExit) as raised:
            main(["H2", "--cell", "6", "--method", "vqe", "--dry-run", *flag])
        assert raised.value.code == 2
        assert "does not take" in capsys.readouterr().err

    def test_a_supported_combination_still_runs(self, capsys):
        assert main(["H2", "--cell", "6", "--method", "adapt-vqe", "--dry-run",
                     "--max-iterations", "3"]) == 0
