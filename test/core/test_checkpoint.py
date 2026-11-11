# -*- coding: utf-8 -*-
# file: test/test_checkpoint.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Wavefunction checkpoints: written during a run, resumed, and handed on.

A checkpoint is the reference determinant, the ordered generators, the angles
and the solver's progress.  Pinned here: the record reproduces the driver's
state exactly (as amplitudes and as a circuit); ADAPT-VQE writes it when the run
ends and resumes to the same operators and energy as an uninterrupted run; VQE
checkpoints its best point and warm-starts from it; the calculator forwards the
options; and a file from another register, method or Hamiltonian is refused
rather than misapplied.
"""

import json

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.core import PauliSum, WavefunctionCheckpoint, load_checkpoint
from mandacaru.core.checkpoint import prepare_state
import warnings
from mandacaru.core.checkpoint import fingerprint


#: A direct-mode (no geometry) problem: the smallest register that carries a
#: real ansatz, in Hartree so the numbers read as the solver stores them.
DIRECT = dict(num_particles=(1, 1), n_spatial_orbitals=2, atomic_units=True,
              trace=False, profile=False)

# The classical optimizers used below, with the iteration budget and
# the convergence tolerance written out rather than left to the
# library default: a test that pins an energy should say what it was
# optimized with.
COBYLA_OPT = {"method": "COBYLA", "maxiter": 2000, "tol": 1e-12}
LBFGS = {"method": "L-BFGS", "maxiter": 2000, "tol": 1e-12}
LBFGS_CERTIFIED = LBFGS


def h2(distance=0.74):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, distance]], cell=[6.0] * 3)
    atoms.center()
    return atoms


def lih(distance=1.6):
    """The growth system.

    H2 in a minimal basis is exactly one double excitation away from its
    ground state, so ADAPT converges after a *single* operator and there is no
    growth to interrupt, checkpoint and resume.  LiH needs six operators at
    this tolerance, which is what these tests are about; ``h2`` stays for the
    register- and serialization-level checks above.
    """
    atoms = Atoms("LiH", positions=[[0, 0, 0], [0, 0, distance]],
                  cell=[8.0] * 3)
    atoms.center()
    return atoms


def adapt(**options):
    return Mandacaru(method="adapt-vqe", pool="qeb", basis="HAO", grid_spacing=0.4,
                     trace=False, profile=False, convergence={"gradient": 1e-6},
                     **options)


@pytest.fixture(scope="module")
def straight_run():
    """A six-operator LiH run: the growth the resume tests replay."""
    atoms = lih()
    atoms.calc = adapt(max_iterations=6)
    energy = atoms.get_total_energy()
    return atoms.calc, energy


@pytest.fixture(scope="module")
def h2_run():
    """A one-operator H2 run: the 4-qubit register the record tests name."""
    atoms = h2()
    atoms.calc = adapt(max_iterations=6)
    energy = atoms.get_total_energy()
    return atoms.calc, energy


# --------------------------------------------------------------------------- #
# The record.
# --------------------------------------------------------------------------- #

class TestRecord:
    def test_reproduces_the_driver_state_and_round_trips(self, tmp_path,
                                                         h2_run):
        driver, _ = h2_run
        record = driver.solver.checkpoint                  # written at the end of run()
        assert isinstance(record, WavefunctionCheckpoint)
        assert record.labels == driver.result.operators
        assert record.method == "adapt-vqe"
        assert record.status["complete"] is True

        psi = driver.solver.ansatz.state(driver.result.optimal_parameters)
        assert abs(abs(np.vdot(psi, record.state_vector())) - 1) < 1e-12
        assert record.expectation() == pytest.approx(
            driver.result.in_units("Ha"), abs=1e-10)

        path = record.save(tmp_path / "ck.json")
        back = load_checkpoint(path)
        assert back.n_qubits == 4 and back.reference_qubits == [0, 2]
        assert np.allclose(back.parameters, record.parameters)
        assert back.labels == record.labels and back.kinds == record.kinds
        assert back.hamiltonian.simplify().terms == \
            record.hamiltonian.simplify().terms
        assert back.status == record.status
        assert back.metadata == record.metadata
        assert json.load(open(path))["energy_unit"] == "Ha"

    def test_the_circuit_prepares_the_same_state(self, h2_run):
        from qiskit.quantum_info import Statevector
        record = h2_run[0].solver.checkpoint
        qc = record.circuit()
        assert qc.num_qubits == 4
        amplitudes = np.asarray(Statevector(qc).data)
        assert abs(abs(np.vdot(amplitudes, record.state_vector())) - 1) < 1e-10

    def test_prepare_state_handles_both_exponential_forms(self):
        """A generator with A^3 = -A takes the closed form; another does not."""
        ref = np.zeros(4, dtype=complex)
        ref[0b10] = 1.0                       # qubit 0 is the leftmost bit
        excitation = PauliSum({"XY": 0.5j, "YX": -0.5j})    # A^3 = -A
        generic = PauliSum({"XX": 0.3j, "ZI": 0.2j})        # A^3 != -A
        from scipy.linalg import expm
        for gen in (excitation, generic):
            psi = prepare_state(2, [0], [gen], [0.7])
            exact = expm(0.7 * gen.to_matrix()) @ ref
            assert np.allclose(psi, exact, atol=1e-12)

    def test_float32_prepares_a_complex64_state(self):
        gen = PauliSum({"XX": 0.3j, "ZI": 0.2j})
        wide = prepare_state(2, [0], [gen], [0.7])
        narrow = prepare_state(2, [0], [gen], [0.7], precision="float32")
        assert wide.dtype == np.complex128 and narrow.dtype == np.complex64
        assert np.allclose(narrow, wide, atol=1e-6)
        with pytest.raises(ValueError, match="precision"):
            prepare_state(2, [0], [gen], [0.7], precision="single")

    def test_inconsistent_records_are_refused(self):
        with pytest.raises(ValueError, match="parameters"):
            WavefunctionCheckpoint(2, [0], [PauliSum({"XY": 1j})], [])
        with pytest.raises(ValueError, match="qubit generator"):
            WavefunctionCheckpoint(3, [0], [PauliSum({"XY": 1j})], [0.1])
        with pytest.raises(ValueError, match="qubit Hamiltonian"):
            WavefunctionCheckpoint(2, [0], [], [], hamiltonian=PauliSum({"Z": 1}))

    def test_writes_are_atomic(self, tmp_path):
        record = WavefunctionCheckpoint(2, [0], [PauliSum({"XY": 1j})], [0.1])
        path = tmp_path / "nested" / "ck.json"
        record.save(path)
        assert path.is_file()
        assert not any(p.name.startswith(".checkpoint-")
                       for p in path.parent.iterdir())


# --------------------------------------------------------------------------- #
# ADAPT-VQE: periodic checkpoints and resume.
# --------------------------------------------------------------------------- #

class TestAdaptResume:
    def test_resume_equals_an_uninterrupted_run(self, tmp_path, straight_run):
        driver, energy = straight_run
        path = str(tmp_path / "adapt.json")

        atoms = lih()
        atoms.calc = adapt(max_iterations=2, checkpoint=path)
        atoms.get_total_energy()
        partial = load_checkpoint(path)
        assert partial.num_parameters == 2
        assert partial.status["complete"] is True      # the run *ended*
        assert partial.status["converged"] is False    # ... unconverged

        atoms = lih()
        atoms.calc = adapt(max_iterations=6, resume=path, checkpoint=path)
        resumed_energy = atoms.get_total_energy()
        result = atoms.calc.result
        assert result.operators == driver.result.operators
        assert resumed_energy == pytest.approx(energy, abs=1e-8)
        # The iteration history is not stored in the file: only the operators
        # grown after the resume carry rows.
        assert [it.operator_label for it in result.iterations] == \
            driver.result.operators[2:]
        assert result.num_evaluations > 0
        final = load_checkpoint(path)
        assert final.num_parameters == 6 and final.status["complete"]

    def test_the_hessian_is_stored_and_an_older_file_resumes_cold_with_a_warning(
            self, tmp_path, straight_run):
        """SLSQP's Hessian approximation of the angles goes into the file (schema version 3) and a
        resumed run starts from it; a version-2 file records none: it resumes from the identity,
        says so, and still reaches the uninterrupted run's operators and energy."""
        import json
        import warnings

        driver, energy = straight_run
        path = str(tmp_path / "adapt.json")
        atoms = lih()
        atoms.calc = adapt(max_iterations=2, checkpoint=path)
        atoms.get_total_energy()
        hessian = np.asarray(atoms.calc.result._raw["hessian"])
        assert hessian.shape == (2, 2) and np.allclose(hessian, hessian.T)
        stored = load_checkpoint(path)
        assert np.array_equal(stored.status["hessian"]["values"], hessian.ravel())

        atoms = lih()
        atoms.calc = adapt(max_iterations=6, resume=path)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert atoms.get_total_energy() == pytest.approx(energy, abs=1e-8)
        # Continued, not relaxed: the stored point is the uninterrupted run's, so the growth after
        # it is too, bit for bit.
        assert "seed_relaxation" not in atoms.calc.result._raw
        assert atoms.calc.result.operators == driver.result.operators
        assert atoms.get_total_energy() == energy

        payload = json.load(open(path))
        payload["version"] = 2
        del payload["status"]["hessian"]
        older = str(tmp_path / "older.json")
        json.dump(payload, open(older, "w"))
        atoms = lih()
        atoms.calc = adapt(max_iterations=6, resume=older)
        with pytest.warns(RuntimeWarning, match="Hessian"):
            assert atoms.get_total_energy() == pytest.approx(energy, abs=1e-8)
        assert atoms.calc.result._raw["seed_relaxation"]["warm_hessian"] is False
        # From the identity the growth may take operators of equal gradient in the other order.
        assert sorted(atoms.calc.result.operators) == sorted(driver.result.operators)

        # L-BFGS keeps no dense estimate: nothing is stored and nothing is warned about.
        other = str(tmp_path / "lbfgs.json")
        atoms = lih()
        atoms.calc = adapt(max_iterations=2, checkpoint=other, optimizer={"method": "L-BFGS"})
        atoms.get_total_energy()
        assert atoms.calc.result._raw["hessian"] is None
        assert "hessian" not in load_checkpoint(other).status

    def test_bfgs_stores_its_inverse_hessian_and_resumes_bit_identically(self, tmp_path):
        """BFGS carries its inverse Hessian approximation like SLSQP its Hessian, the file names
        the optimizer that kept it, and the other optimizer does not use it (nor convert it):
        it starts from the identity and warns, naming both."""
        import warnings

        bfgs = {"method": "BFGS"}
        path = str(tmp_path / "bfgs.json")
        straight = lih()
        straight.calc = adapt(max_iterations=6, optimizer=bfgs)
        energy = straight.get_total_energy()
        atoms = lih()
        atoms.calc = adapt(max_iterations=2, checkpoint=path, optimizer=bfgs)
        atoms.get_total_energy()
        stored = load_checkpoint(path).status["hessian"]
        assert stored["optimizer"] == "BFGS" and stored["dimension"] == 2

        atoms = lih()
        atoms.calc = adapt(max_iterations=6, resume=path, optimizer=bfgs)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert atoms.get_total_energy() == energy
        assert atoms.calc.result.operators == straight.calc.result.operators
        assert "seed_relaxation" not in atoms.calc.result._raw

        atoms = lih()
        atoms.calc = adapt(max_iterations=6, resume=path)          # SLSQP, the default
        with pytest.warns(RuntimeWarning, match="BFGS.*SLSQP|SLSQP.*BFGS"):
            atoms.get_total_energy()
        assert atoms.calc.result._raw["seed_relaxation"]["warm_hessian"] is False

    def test_a_mid_run_checkpoint_resumes_to_the_uninterrupted_run(self, tmp_path,
                                                                    straight_run):
        """``checkpoint_every=2`` writes after the second growth step what that
        step reached, marked incomplete; a run resumed from that file grows the
        same six operators to the same energy as the uninterrupted one."""
        import shutil

        driver, energy = straight_run
        path, copy = str(tmp_path / "adapt.json"), str(tmp_path / "step2.json")
        atoms = lih()
        atoms.calc = adapt(max_iterations=6, checkpoint=path, checkpoint_every=2)

        def keep_step_two(info):
            # The file is written before the callback sees the step.
            if info["num_operators"] == 2:       # one operator per qeb step
                shutil.copy(path, copy)

        atoms.calc.run(atoms, callback=keep_step_two)
        mid = load_checkpoint(copy)
        assert mid.num_parameters == 2
        # SLSQP growth stores its Hessian approximation of the two angles (schema version 3).
        hessian = mid.status.pop("hessian")
        assert mid.status == {"complete": False, "converged": False, "iteration": 2}
        assert hessian["dimension"] == 2 and len(hessian["values"]) == 4
        assert hessian["values"][1] == hessian["values"][2]
        assert load_checkpoint(path).status["complete"] is True

        atoms = lih()
        atoms.calc = adapt(max_iterations=6, resume=copy)
        assert atoms.get_total_energy() == pytest.approx(energy, abs=1e-8)
        assert atoms.calc.result.operators == driver.result.operators

    def test_the_cadence_needs_a_path_and_a_positive_integer(self, tmp_path):
        for every, checkpoint in ((0, "c.json"), (1.5, "c.json"), (2, None)):
            with pytest.raises(ValueError, match="checkpoint_every"):
                Mandacaru(method="adapt-vqe", checkpoint=checkpoint, checkpoint_every=every)
        with pytest.raises(ValueError, match="checkpoint_every"):
            Mandacaru(method="vasqa", checkpoint="c.json", checkpoint_every=2)

    def test_a_pool_generator_is_adopted_and_a_foreign_one_refused(self,
                                                                  tmp_path):
        """Resuming matches generators to the pool by content, not by name.

        The fermionic double D(0,2->1,3) *is* qeb's QD(0,2->1,3) on H2, so a
        qeb run adopts its own operator for it; a single Pauli string from the
        qubit pool is no qeb generator and the native pool refuses it.
        """
        from mandacaru.core import load_checkpoint as load

        fermionic = str(tmp_path / "fermionic.json")
        atoms = h2()
        atoms.calc = Mandacaru(method="adapt-vqe", pool="fermionic",
                               basis="HAO", grid_spacing=0.4, trace=False, profile=False,
                               max_iterations=1, checkpoint=fermionic)
        atoms.get_total_energy()
        atoms = h2()
        atoms.calc = adapt(max_iterations=3, resume=fermionic)
        atoms.get_total_energy()
        assert atoms.calc.result.operators[0] == "QD(0,2->1,3)"   # the pool's own
        assert atoms.calc.solver.ansatz.pauli_generators[0].simplify().terms == \
            load(fermionic).generators[0].simplify().terms

        pauli = str(tmp_path / "pauli.json")
        atoms = h2()
        atoms.calc = Mandacaru(method="adapt-vqe", pool="qubit", basis="HAO",
                               grid_spacing=0.4, trace=False, profile=False,
                               max_iterations=1, checkpoint=pauli)
        atoms.get_total_energy()
        atoms = h2()
        atoms.calc = adapt(max_iterations=4, resume=pauli)
        with pytest.raises(ValueError, match="absent from native pool"):
            atoms.get_total_energy()

    def test_mismatched_register_is_refused(self, tmp_path):
        path = str(tmp_path / "parity.json")
        atoms = lih()
        atoms.calc = adapt(max_iterations=1, mapping="parity", checkpoint=path)
        atoms.get_total_energy()
        atoms = lih()
        atoms.calc = adapt(max_iterations=2, resume=path)       # Jordan-Wigner
        with pytest.raises(ValueError, match="mapping"):
            atoms.get_total_energy()

    def test_resume_and_initial_parameters_are_exclusive(self, tmp_path):
        path = str(tmp_path / "x.json")
        atoms = lih()
        atoms.calc = adapt(max_iterations=1, checkpoint=path)
        atoms.get_total_energy()
        atoms = lih()
        atoms.calc = adapt(max_iterations=2, resume=path)
        with pytest.raises(ValueError, match="choose initial parameters"):
            atoms.calc.run(atoms, initial_parameters=[0.1])


# --------------------------------------------------------------------------- #
# VQE.
# --------------------------------------------------------------------------- #

class TestSubspaceResume:
    """A ``subspace-*`` checkpoint stores the shared ansatz with the state weights
    and energies; a resumed subspace-ADAPT run grows on as if never stopped."""

    @staticmethod
    def subspace(**options):
        return Mandacaru(method="subspace-adapt-vqe", pool="qeb", basis="HAO",
                         grid_spacing=0.4, trace=False, profile=False,
                         convergence={"gradient": 1e-6}, **options)

    def test_resume_equals_an_uninterrupted_run(self, tmp_path):
        import json as _json
        path = str(tmp_path / "subspace.json")
        straight = lih()
        straight.calc = self.subspace(max_iterations=5)
        straight.get_total_energy()
        stopped = lih()
        stopped.calc = self.subspace(max_iterations=2, checkpoint=path)
        stopped.get_total_energy()
        stored = _json.loads(open(path, encoding="utf-8").read())
        assert stored["method"] == "subspace-adapt-vqe"
        assert stored["subspace"]["weights"] == [2.0, 1.0]
        assert len(stored["operators"]) == 2
        resumed = lih()
        resumed.calc = self.subspace(max_iterations=5, resume=path)
        resumed.get_total_energy()
        assert np.allclose(resumed.calc.result.energies, straight.calc.result.energies,
                           atol=1e-8)
        assert resumed.calc.result.num_operators == straight.calc.result.num_operators
        # The file stores the SLSQP Hessian approximation, and the resumed growth is the
        # uninterrupted run's, bit for bit.
        assert stored["status"]["hessian"]["dimension"] == 2
        assert np.array_equal(resumed.calc.result.energies, straight.calc.result.energies)
        assert resumed.calc.result.operators == straight.calc.result.operators

    def test_a_version_2_file_resumes_from_the_identity_with_a_warning(self, tmp_path):
        import json as _json
        path = str(tmp_path / "subspace.json")
        stopped = lih()
        stopped.calc = self.subspace(max_iterations=2, checkpoint=path)
        stopped.get_total_energy()
        stored = _json.load(open(path, encoding="utf-8"))
        stored["version"] = 2
        del stored["status"]["hessian"]
        _json.dump(stored, open(path, "w", encoding="utf-8"))
        resumed = lih()
        resumed.calc = self.subspace(max_iterations=5, resume=path)
        with pytest.warns(RuntimeWarning, match="Hessian"):
            resumed.get_total_energy()

    def test_other_weights_are_refused(self, tmp_path):
        path = str(tmp_path / "subspace.json")
        atoms = lih()
        atoms.calc = self.subspace(max_iterations=1, checkpoint=path)
        atoms.get_total_energy()
        atoms = lih()
        atoms.calc = self.subspace(max_iterations=2, resume=path, weights=[3.0, 1.0])
        with pytest.raises(ValueError, match="weights"):
            atoms.get_total_energy()


class TestVQEResume:
    def test_checkpoint_and_warm_start(self, tmp_path):
        path = str(tmp_path / "vqe.json")
        atoms = h2()
        atoms.calc = Mandacaru(method="vqe", basis="HAO", grid_spacing=0.4,
                               optimizer=LBFGS, trace=False,
                               checkpoint=path)
        energy = atoms.get_total_energy()
        record = load_checkpoint(path)
        assert record.method == "vqe" and record.status["complete"]
        assert record.num_parameters == 3
        assert record.energy == pytest.approx(atoms.calc.result.in_units("Ha"),
                                              abs=1e-12)

        cold = h2()
        cold.calc = Mandacaru(method="vqe", basis="HAO", grid_spacing=0.4,
                              optimizer=LBFGS_CERTIFIED, trace=False)
        cold.get_total_energy()

        atoms = h2()
        atoms.calc = Mandacaru(method="vqe", basis="HAO", grid_spacing=0.4,
                               optimizer=LBFGS_CERTIFIED, trace=False,
                               resume=path)
        assert atoms.get_total_energy() == pytest.approx(energy, abs=1e-8)
        # Starting at the optimum costs less than starting from zero.  How
        # much less is the optimizer's tolerance to decide -- a tight one
        # still spends a gradient and a line search confirming it is there --
        # so this compares the two runs rather than pinning a number.
        assert atoms.calc.result.num_evaluations < \
            cold.calc.result.num_evaluations

    def test_the_best_point_is_what_gets_checkpointed(self, tmp_path):
        path = str(tmp_path / "best.json")
        atoms = h2()
        atoms.calc = Mandacaru(method="vqe", basis="HAO", grid_spacing=0.4,
                               optimizer=COBYLA_OPT, trace=False,
                               checkpoint=path)
        atoms.get_total_energy()
        assert load_checkpoint(path).energy == pytest.approx(
            atoms.calc.result.in_units("Ha"), abs=1e-12)

    def test_an_adapt_file_does_not_resume_a_fixed_ansatz(self, tmp_path):
        path = str(tmp_path / "adapt.json")
        atoms = h2()
        atoms.calc = adapt(max_iterations=1, checkpoint=path)
        atoms.get_total_energy()
        atoms = h2()
        atoms.calc = Mandacaru(method="vqe", basis="HAO", grid_spacing=0.4, trace=False,
                               resume=path)
        with pytest.raises(ValueError, match="method differs"):
            atoms.get_total_energy()


# --------------------------------------------------------------------------- #
# Through the calculator.
# --------------------------------------------------------------------------- #

class TestCalculator:
    def test_options_pass_through_and_resume_at_the_same_geometry(self, tmp_path):
        path = str(tmp_path / "relax.json")
        atoms = h2(0.74)
        atoms.calc = Mandacaru(method="adapt-vqe", pool="qeb", basis="HAO",
                               grid_spacing=0.4, profile=False,
                               max_iterations=6, checkpoint=path, resume=None)
        atoms.get_potential_energy()
        first = load_checkpoint(path)

        # The same geometry resumes from the first state (same file).
        atoms.calc = Mandacaru(method="adapt-vqe", pool="qeb", basis="HAO",
                               grid_spacing=0.4, profile=False,
                               max_iterations=8, checkpoint=path, resume=path)
        atoms.get_potential_energy()
        second = load_checkpoint(path)
        assert second.num_parameters >= first.num_parameters
        assert second.labels[:first.num_parameters] == first.labels

        # A displaced geometry has another Hamiltonian: the file is refused.
        atoms.calc = Mandacaru(method="adapt-vqe", pool="qeb", basis="HAO",
                               grid_spacing=0.4, profile=False,
                               max_iterations=8, resume=path)
        atoms.positions[1, 2] += 0.05
        with pytest.raises(ValueError, match="Hamiltonian differs"):
            atoms.get_potential_energy()


class TestResumeEnergyIdentity:
    """A resumed run reports *its* energy, not the file's."""

    SHIFT = 2.0

    @staticmethod
    def _state(path, hamiltonian):
        Mandacaru(method="adapt-vqe", hamiltonian=hamiltonian,
                  pool="fermionic", checkpoint=path, max_iterations=0,
                  **DIRECT).run()

    def test_a_shifted_hamiltonian_is_refused(self, tmp_path):
        """A state stored for one Hamiltonian is not resumed under another."""
        path = str(tmp_path / "state.json")
        self._state(path, PauliSum({"ZIII": 1.0}))
        shifted = PauliSum({"ZIII": 1.0, "IIII": self.SHIFT})
        with pytest.raises(ValueError, match="Hamiltonian differs"):
            Mandacaru(method="adapt-vqe", hamiltonian=shifted,
                      pool="fermionic", resume=path, max_iterations=0,
                      **DIRECT).run()

    def test_the_same_hamiltonian_resumes_without_a_warning(self, tmp_path):
        path = str(tmp_path / "state.json")
        hamiltonian = PauliSum({"ZIII": 1.0, "XXII": 0.25})
        self._state(path, hamiltonian)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = Mandacaru(method="adapt-vqe", hamiltonian=hamiltonian,
                               pool="fermionic", resume=path, max_iterations=0,
                               **DIRECT).run()
        assert np.isfinite(result.optimal_energy)

    def test_the_fingerprint_ignores_order_and_noise(self):
        one = PauliSum({"ZI": 1.0, "IX": 0.5})
        other = PauliSum({"IX": 0.5 + 1e-14, "ZI": 1.0})
        assert fingerprint(one) == fingerprint(other)
        assert fingerprint(PauliSum({"ZI": 1.0})) != fingerprint(one)
        assert fingerprint(None) is None


class TestCheckpointRecordsThePreparation:



    def test_a_version_1_file_is_a_product(self):
        payload = WavefunctionCheckpoint(1, [], [PauliSum({"Y": 1j})],
                                         [0.3]).to_payload()
        del payload["preparation"]
        assert WavefunctionCheckpoint.from_payload(payload).preparation == "product"


    def test_one_generator_is_both_forms(self):
        record = WavefunctionCheckpoint(1, [], [PauliSum({"Y": 1j})], [0.3],
                                        preparation="sum")
        assert record.is_product and record.problem()[0] == 1





def test_a_wide_checkpoint_refuses_its_state_vector_by_name():
    """A shot run's register is too wide for a state vector: preparing one is refused with the
    cause, not with a bare memory-budget error, and nothing is allocated."""
    record = WavefunctionCheckpoint(50, [0, 1], [], [])
    with pytest.raises(ValueError, match=r"2\^50 amplitudes.*use its circuit"):
        record.state_vector()
