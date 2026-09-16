# -*- coding: utf-8 -*-
# file: test/test_checkpoint.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Wavefunction checkpoints: written during a run, resumed, and handed on.

A checkpoint is the reference determinant, the ordered generators, the angles
and the solver's progress.  Pinned here: the record reproduces the driver's
state exactly (as amplitudes and as a circuit); ADAPT-VQE writes it after every
accepted operator and resumes to the same operators, energy and history as an
uninterrupted run -- including after a crash mid-run; VQE checkpoints its best
point and warm-starts from it; the calculator forwards the options; and a file
from another register is refused rather than misapplied.
"""

import json

import numpy as np
import pytest
from ase import Atoms

from carcara import Carcara
from carcara.algorithms import ADAPTVQE, VQE
from carcara.core import PauliSum, WavefunctionCheckpoint, load_checkpoint
from carcara.core.checkpoint import prepare_state, reference_vector


def h2(distance=0.74):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, distance]], cell=[6.0] * 3)
    atoms.center()
    return atoms


def adapt(**options):
    return ADAPTVQE(pool="qeb", basis="FAO", h=0.4, verbose=False,
                    profile=False, gradient_tolerance=1e-6, **options)


@pytest.fixture(scope="module")
def straight_run():
    atoms = h2()
    atoms.calc = adapt(max_iterations=6)
    energy = atoms.get_total_energy()
    return atoms.calc, energy


# --------------------------------------------------------------------------- #
# The record.
# --------------------------------------------------------------------------- #

class TestRecord:
    def test_reproduces_the_driver_state_and_round_trips(self, tmp_path,
                                                         straight_run):
        driver, _ = straight_run
        record = driver.checkpoint                  # written at the end of run()
        assert isinstance(record, WavefunctionCheckpoint)
        assert record.labels == driver.result.operators
        assert record.method == "adapt-vqe"
        assert record.status["complete"] is True

        psi = driver.ansatz.state(driver.result.optimal_parameters)
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
        assert back.metadata["basis"] == "FAO"
        assert back.metadata["geometry"]["symbols"] == ["H", "H"]
        assert json.load(open(path))["energy_unit"] == "Ha"

    def test_the_circuit_prepares_the_same_state(self, straight_run):
        from qiskit.quantum_info import Statevector
        record = straight_run[0].checkpoint
        qc = record.circuit()
        assert qc.num_qubits == 4
        amplitudes = np.asarray(Statevector(qc).data)
        assert abs(abs(np.vdot(amplitudes, record.state_vector())) - 1) < 1e-10

    def test_prepare_state_handles_both_exponential_forms(self):
        """A generator with A^3 = -A takes the closed form; another does not."""
        ref = reference_vector(2, [0])
        excitation = PauliSum({"XY": 0.5j, "YX": -0.5j})    # A^3 = -A
        generic = PauliSum({"XX": 0.3j, "ZI": 0.2j})        # A^3 != -A
        from scipy.linalg import expm
        for gen in (excitation, generic):
            psi = prepare_state(2, [0], [gen], [0.7])
            exact = expm(0.7 * gen.to_matrix()) @ ref
            assert np.allclose(psi, exact, atol=1e-12)

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

        atoms = h2()
        atoms.calc = adapt(max_iterations=2, checkpoint=path)
        atoms.get_total_energy()
        partial = load_checkpoint(path)
        assert partial.num_parameters == 2
        assert partial.status["iteration"] == 2
        assert partial.status["complete"] is True      # the run *ended*
        assert partial.status["converged"] is False    # ... unconverged

        atoms = h2()
        atoms.calc = adapt(max_iterations=6, resume=path, checkpoint=path)
        resumed_energy = atoms.get_total_energy()
        result = atoms.calc.result
        assert result.operators == driver.result.operators
        assert resumed_energy == pytest.approx(energy, abs=1e-8)
        assert len(result.iterations) == 6
        assert [it.operator_label for it in result.iterations] == \
            driver.result.operators
        assert result.num_evaluations > 0
        final = load_checkpoint(path)
        assert final.num_parameters == 6 and final.status["complete"]

    def test_a_crash_mid_run_leaves_a_resumable_file(self, tmp_path,
                                                     straight_run):
        driver, energy = straight_run
        path = str(tmp_path / "crash.json")

        def crash(info):
            if info["iteration"] == 3:
                raise RuntimeError("simulated interruption")

        atoms = h2()
        atoms.calc = adapt(max_iterations=6, checkpoint=path)
        with pytest.raises(RuntimeError, match="simulated"):
            _configured(atoms).run(callback=crash)
        saved = load_checkpoint(path)
        assert saved.num_parameters == 3
        assert saved.status["complete"] is False
        assert len(saved.status["iterations"]) == 3

        atoms = h2()
        atoms.calc = adapt(max_iterations=6, resume=path)
        assert atoms.get_total_energy() == pytest.approx(energy, abs=1e-8)
        assert atoms.calc.result.operators == driver.result.operators

    def test_checkpoint_every(self, tmp_path):
        path = str(tmp_path / "every.json")
        seen = []

        def watch(info):
            seen.append(load_checkpoint(path).num_parameters
                        if info["iteration"] % 2 == 0 else None)

        atoms = h2()
        atoms.calc = adapt(max_iterations=4, checkpoint=path, checkpoint_every=2)
        _configured(atoms).run(callback=watch)
        assert seen == [None, 2, None, 4]

    def test_a_pool_generator_is_adopted_and_a_foreign_one_kept(self,
                                                               tmp_path):
        """Resuming matches generators to the pool by content, not by name.

        The fermionic double D(0,2->1,3) *is* qeb's QD(0,2->1,3) on H2, so a
        qeb run adopts its own operator for it; a single Pauli string from the
        qubit pool is no qeb generator and is applied exactly as stored.
        """
        from carcara.core import load_checkpoint as load

        fermionic = str(tmp_path / "fermionic.json")
        atoms = h2()
        atoms.calc = ADAPTVQE(pool="fermionic", basis="FAO", h=0.4,
                              verbose=False, profile=False, max_iterations=1,
                              checkpoint=fermionic)
        atoms.get_total_energy()
        atoms = h2()
        atoms.calc = adapt(max_iterations=3, resume=fermionic)
        atoms.get_total_energy()
        first = atoms.calc.ansatz.operators[0]
        assert first.label == "QD(0,2->1,3)"                # the pool's own
        assert first.generator.simplify().terms == \
            load(fermionic).generators[0].simplify().terms

        pauli = str(tmp_path / "pauli.json")
        atoms = h2()
        atoms.calc = ADAPTVQE(pool="qubit", basis="FAO", h=0.4, verbose=False,
                              profile=False, max_iterations=1, checkpoint=pauli)
        atoms.get_total_energy()
        stored = load(pauli)
        atoms = h2()
        atoms.calc = adapt(max_iterations=4, resume=pauli)
        energy = atoms.get_total_energy()
        first = atoms.calc.ansatz.operators[0]
        assert first.label == stored.labels[0] and first.kind == stored.kinds[0]
        assert first.generator.simplify().terms == \
            stored.generators[0].simplify().terms
        assert energy < stored.energy * 27.2 + 1e-6        # kept growing (eV)

    def test_mismatched_register_is_refused(self, tmp_path):
        path = str(tmp_path / "parity.json")
        atoms = h2()
        atoms.calc = adapt(max_iterations=1, mapping="parity", checkpoint=path)
        atoms.get_total_energy()
        atoms = h2()
        atoms.calc = adapt(max_iterations=2, resume=path)       # Jordan-Wigner
        with pytest.raises(ValueError, match="mapping"):
            atoms.get_total_energy()

    def test_resume_and_initial_parameters_are_exclusive(self, tmp_path):
        path = str(tmp_path / "x.json")
        atoms = h2()
        atoms.calc = adapt(max_iterations=1, checkpoint=path)
        atoms.get_total_energy()
        atoms = h2()
        atoms.calc = adapt(max_iterations=2, resume=path)
        with pytest.raises(ValueError, match="not both"):
            _configured(atoms).run(initial_parameters=[0.1])


def _configured(atoms):
    """The driver configured for ``atoms`` without running (direct ``run``)."""
    driver = atoms.calc
    driver.atoms = atoms.copy()
    hamiltonian, particles, n_orb = driver._build_hamiltonian(atoms)
    driver._configure(hamiltonian, particles, n_orb)
    return driver


# --------------------------------------------------------------------------- #
# VQE.
# --------------------------------------------------------------------------- #

class TestVQEResume:
    def test_checkpoint_and_warm_start(self, tmp_path):
        path = str(tmp_path / "vqe.json")
        atoms = h2()
        atoms.calc = VQE(basis="FAO", h=0.4, optimizer="L-BFGS-B",
                         verbose=False, checkpoint=path, checkpoint_every=5)
        energy = atoms.get_total_energy()
        record = load_checkpoint(path)
        assert record.method == "vqe" and record.status["complete"]
        assert record.num_parameters == 3
        assert record.energy == pytest.approx(atoms.calc.result.in_units("Ha"),
                                              abs=1e-12)

        atoms = h2()
        atoms.calc = VQE(basis="FAO", h=0.4, optimizer="L-BFGS-B",
                         verbose=False, resume=path)
        assert atoms.get_total_energy() == pytest.approx(energy, abs=1e-8)
        # Starting at the optimum, the warm start needs only a few evaluations.
        assert atoms.calc.result.num_evaluations < 10

    def test_the_best_point_is_what_gets_checkpointed(self, tmp_path):
        path = str(tmp_path / "best.json")
        atoms = h2()
        atoms.calc = VQE(basis="FAO", h=0.4, optimizer="COBYLA", verbose=False,
                         checkpoint=path, checkpoint_every=1)
        atoms.get_total_energy()
        assert load_checkpoint(path).energy == pytest.approx(
            atoms.calc.result.in_units("Ha"), abs=1e-12)

    def test_an_adapt_file_does_not_resume_a_fixed_ansatz(self, tmp_path):
        path = str(tmp_path / "adapt.json")
        atoms = h2()
        atoms.calc = adapt(max_iterations=1, checkpoint=path)
        atoms.get_total_energy()
        atoms = h2()
        atoms.calc = VQE(basis="FAO", h=0.4, verbose=False, resume=path)
        with pytest.raises(ValueError, match="not this ansatz"):
            atoms.get_total_energy()


# --------------------------------------------------------------------------- #
# Through the calculator.
# --------------------------------------------------------------------------- #

class TestCalculator:
    def test_options_pass_through_and_warm_start_a_relaxation(self, tmp_path):
        path = str(tmp_path / "relax.json")
        atoms = h2(0.74)
        atoms.calc = Carcara(method="adapt-vqe", pool="qeb", basis="FAO",
                             h=0.4, verbose=False, profile=False,
                             max_iterations=6, checkpoint=path, resume=None)
        atoms.get_potential_energy()
        first = load_checkpoint(path)
        assert first.metadata["geometry"]["positions_angstrom"][1][2] == \
            pytest.approx(atoms.positions[1, 2])

        # A second geometry resumes from the first one's state (same file).
        atoms.calc = Carcara(method="adapt-vqe", pool="qeb", basis="FAO",
                             h=0.4, verbose=False, profile=False,
                             max_iterations=8, checkpoint=path, resume=path)
        atoms.positions[1, 2] += 0.05
        atoms.get_potential_energy()
        second = load_checkpoint(path)
        assert second.num_parameters >= first.num_parameters
        assert second.labels[:first.num_parameters] == first.labels
        assert second.metadata["geometry"]["positions_angstrom"][1][2] == \
            pytest.approx(atoms.positions[1, 2])
