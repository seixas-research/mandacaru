# -*- coding: utf-8 -*-
# file: test/test_qpe.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Quantum phase estimation on H2 in a minimal basis, from a checkpoint.

The variationally optimized H2 state (ADAPT-VQE, HAO, 4 qubits) is written as
a wavefunction checkpoint and handed to QPE with nothing else.  Pinned: the
most probable reading is the exact ground-state energy to within one
resolution bin, the register collapses onto the exact eigenvector, a tight
energy window sharpens the reading, a poorer input state spreads the readings
by its overlaps, the exported Qiskit circuit reproduces the native
distribution, and the memory check runs *before* anything is allocated with
each of its three policies.
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.workflows.qpe import phase_estimation
from mandacaru.workflows.qpe import qpe_memory_estimate
from mandacaru.workflows.qpe import QPEResult
from mandacaru.workflows.qpe import QuantumPhaseEstimation
from mandacaru.workflows import qpe as qpe_module
from mandacaru.core import PauliSum, load_checkpoint, matrix_free
from mandacaru.core.checkpoint import reference_vector
from mandacaru.units import HARTREE_TO_EV


@pytest.fixture(scope="module")
def h2_checkpoint(tmp_path_factory):
    """ADAPT-VQE ground state of H2/HAO, checkpointed with its Hamiltonian."""
    path = str(tmp_path_factory.mktemp("qpe") / "h2.json")
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[6.0] * 3)
    atoms.center()
    atoms.calc = Mandacaru(method="adapt-vqe", pool="fermionic", basis="HAO",
                           grid_spacing=0.4, trace=False, max_iterations=6,
                           convergence={"gradient": 1e-8}, checkpoint=path)
    atoms.get_total_energy()
    return load_checkpoint(path)


@pytest.fixture(scope="module")
def exact(h2_checkpoint):
    h = h2_checkpoint.hamiltonian.to_matrix()
    w, v = np.linalg.eigh(0.5 * (h + h.conj().T))
    return w, v


class TestGroundStateFromCheckpoint:
    def test_reads_the_exact_energy_within_one_bin(self, h2_checkpoint, exact):
        w, _ = exact
        qpe = QuantumPhaseEstimation(n_evaluation_qubits=10, atomic_units=True,
                                     verbose=False)
        result = qpe.run(h2_checkpoint)
        assert isinstance(result, QPEResult) and result.energy_unit == "Ha"
        assert result.n_system_qubits == 4 and result.n_evaluation_qubits == 10
        assert abs(result.energy - w[0]) <= result.resolution
        assert result.ground_state_overlap == pytest.approx(1.0, abs=1e-6)
        # Everything read lies inside the window and the bins sum to one.
        assert result.probabilities.sum() == pytest.approx(1.0, abs=1e-10)
        assert np.allclose(result.exact_energies, w)
        # The energy the checkpoint itself carries is the one QPE refines.
        assert abs(h2_checkpoint.energy - w[0]) < 1e-6

    def test_the_register_collapses_onto_the_eigenvector(self, h2_checkpoint,
                                                         exact):
        _, v = exact
        result = QuantumPhaseEstimation(n_evaluation_qubits=8, verbose=False,
                                        atomic_units=True).run(h2_checkpoint)
        psi = result.collapsed_state()
        assert abs(np.vdot(v[:, 0], psi)) ** 2 == pytest.approx(1.0, abs=1e-6)

    def test_a_tight_window_sharpens_the_reading(self, h2_checkpoint, exact):
        w, _ = exact
        wide = QuantumPhaseEstimation(n_evaluation_qubits=8, verbose=False,
                                      atomic_units=True).run(h2_checkpoint)
        tight = QuantumPhaseEstimation(
            n_evaluation_qubits=8, energy_window=(w[0] - 0.02, w[0] + 0.02),
            verbose=False, atomic_units=True).run(h2_checkpoint)
        assert tight.resolution < wide.resolution / 50
        assert abs(tight.energy - w[0]) <= tight.resolution
        assert abs(tight.energy - w[0]) < abs(wide.energy - w[0])

    def test_units_default_to_ev(self, h2_checkpoint, exact):
        w, _ = exact
        result = QuantumPhaseEstimation(n_evaluation_qubits=8,
                                        verbose=False).run(h2_checkpoint)
        assert result.energy_unit == "eV"
        assert abs(result.energy - w[0] * HARTREE_TO_EV) <= result.resolution
        assert result.in_units("Ha").energy == pytest.approx(
            result.energy / HARTREE_TO_EV)

    def test_the_one_call_form_takes_a_path(self, h2_checkpoint, tmp_path,
                                            exact):
        path = h2_checkpoint.save(tmp_path / "again.json")
        result = phase_estimation(path, n_evaluation_qubits=8, verbose=False,
                                  atomic_units=True)
        assert abs(result.energy - exact[0][0]) <= result.resolution


class TestOtherInputs:
    def test_a_poorer_state_spreads_by_its_overlaps(self, h2_checkpoint, exact):
        """The Hartree-Fock determinant: |<E0|HF>|^2 < 1, and the excited
        state it also overlaps shows up with the complementary weight."""
        w, v = exact
        hf = reference_vector(h2_checkpoint.n_qubits,
                              h2_checkpoint.reference_qubits)
        result = QuantumPhaseEstimation(
            h2_checkpoint.hamiltonian, n_evaluation_qubits=10, verbose=False,
            atomic_units=True).run(hf)
        overlap = abs(np.vdot(v[:, 0], hf)) ** 2
        assert 0.9 < overlap < 1.0
        assert result.ground_state_overlap == pytest.approx(overlap, abs=1e-9)
        # The standard QPE bound: at least 1 - 1/(2(k-1)) of a component's
        # weight lands within k bins of its phase.  Both eigenstates the
        # determinant overlaps must show up with their own weights.
        k = 4
        for j, weight in ((0, overlap), (int(np.argmax(np.abs(v.conj().T @ hf)
                                                       ** 2 * (np.arange(16)
                                                               > 0))),
                           1 - overlap)):
            near = np.abs(result.energies - w[j]) <= k * result.resolution
            captured = result.probabilities[near].sum()
            assert captured >= weight * (1 - 1 / (2 * (k - 1))) - 1e-9
            assert captured <= weight + 0.02

    def test_a_preparation_tuple_is_accepted(self, h2_checkpoint, exact):
        n, ref, gens, theta, _ = h2_checkpoint.problem()
        result = QuantumPhaseEstimation(
            h2_checkpoint.hamiltonian, n_evaluation_qubits=8, verbose=False,
            atomic_units=True).run((n, ref, gens, theta))
        assert abs(result.energy - exact[0][0]) <= result.resolution

    def test_native_matrix_free_distribution_matches_the_dense_problem(self, h2_checkpoint):
        dense = QuantumPhaseEstimation(n_evaluation_qubits=6, verbose=False,
                                       atomic_units=True).run(h2_checkpoint)
        # Put the same four-qubit operator in a 13-qubit register. Spectator
        # qubits stay in |0>, while Rust selects its matrix-free propagation.
        extra = 13 - h2_checkpoint.n_qubits
        h = PauliSum({label + "I" * extra: coefficient
                      for label, coefficient in h2_checkpoint.hamiltonian.terms.items()})
        spectator = np.zeros(2**extra, dtype=complex)
        spectator[0] = 1.
        psi = np.kron(h2_checkpoint.state_vector(), spectator)
        sparse = QuantumPhaseEstimation(h, n_evaluation_qubits=6, verbose=False,
                                        atomic_units=True).run(psi)
        assert sparse.exact_energies is None
        assert np.allclose(sparse.probabilities, dense.probabilities, atol=1e-10)

    def test_missing_hamiltonian_is_refused(self, h2_checkpoint):
        bare = type(h2_checkpoint)(
            h2_checkpoint.n_qubits, h2_checkpoint.reference_qubits,
            h2_checkpoint.generators, h2_checkpoint.parameters)
        with pytest.raises(ValueError, match="stores no Hamiltonian"):
            QuantumPhaseEstimation(n_evaluation_qubits=4,
                                   verbose=False).run(bare)
        with pytest.raises(ValueError, match="qubit Hamiltonian"):
            QuantumPhaseEstimation(PauliSum({"ZZ": 1.0}), n_evaluation_qubits=4,
                                   verbose=False).run(h2_checkpoint)


class TestCircuit:
    def test_the_exported_circuit_reproduces_the_native_distribution(
            self, h2_checkpoint):
        t = 5
        qpe = QuantumPhaseEstimation(n_evaluation_qubits=t, verbose=False,
                                     atomic_units=True)
        native = qpe.run(h2_checkpoint)
        qc = qpe.circuit(h2_checkpoint)
        assert qc.num_qubits == t + 4
        measured = QuantumPhaseEstimation.evaluation_distribution(qc, t)
        assert np.allclose(measured, native.probabilities, atol=1e-9)

    def test_amplitudes_are_not_a_circuit(self, h2_checkpoint):
        qpe = QuantumPhaseEstimation(h2_checkpoint.hamiltonian,
                                     n_evaluation_qubits=3, verbose=False)
        with pytest.raises(ValueError, match="state \\*preparation\\*"):
            qpe.circuit(h2_checkpoint.state_vector())


class TestMemoryCheck:
    def test_estimate_counts_the_full_register(self):
        est = qpe_memory_estimate(4, 10)
        assert est.statevector_bytes == 16 * 2 ** 14
        assert est.working_bytes == qpe_module.WORKING_SET_FACTOR * est.statevector_bytes
        assert est.unitary_bytes == 16 * 4 ** 4
        assert "2^14 amplitudes" in est.summary()
        assert qpe_memory_estimate(30, 20).total_bytes > 2 ** 50
        with pytest.raises(ValueError):
            qpe_memory_estimate(0, 4)

    def test_the_check_runs_before_any_allocation(self, h2_checkpoint,
                                                  monkeypatch):
        """With 1 KiB "available" the run must abort without preparing a state."""
        monkeypatch.setattr(matrix_free, "available_memory_bytes",
                            lambda: 1024)
        monkeypatch.setattr(qpe_module, "prepare_state",
                            lambda *a: pytest.fail("a state was prepared"))
        qpe = QuantumPhaseEstimation(n_evaluation_qubits=8, verbose=False)
        with pytest.raises(MemoryError, match="refusing to start"):
            qpe.run(h2_checkpoint)
        assert qpe.memory_estimate_result.fraction > 1

    def test_ignore_proceeds_with_a_warning(self, h2_checkpoint, monkeypatch):
        monkeypatch.setattr(matrix_free, "available_memory_bytes",
                            lambda: 1024)
        qpe = QuantumPhaseEstimation(n_evaluation_qubits=4, verbose=False,
                                     memory_policy="ignore", atomic_units=True)
        with pytest.warns(RuntimeWarning, match="despite the memory estimate"):
            result = qpe.run(h2_checkpoint)
        assert result.probabilities.sum() == pytest.approx(1.0)

    def test_prompt_asks_and_honors_the_answer(self, h2_checkpoint,
                                                monkeypatch):
        monkeypatch.setattr(matrix_free, "available_memory_bytes",
                            lambda: 1024)
        monkeypatch.setattr(qpe_module.sys.stdin, "isatty", lambda: True,
                            raising=False)
        qpe = QuantumPhaseEstimation(n_evaluation_qubits=4, verbose=False,
                                     memory_policy="prompt", atomic_units=True)
        monkeypatch.setattr("builtins.input", lambda prompt="": "n")
        with pytest.raises(MemoryError, match="aborted at the prompt"):
            qpe.run(h2_checkpoint)
        monkeypatch.setattr("builtins.input", lambda prompt="": "y")
        assert qpe.run(h2_checkpoint).probabilities.sum() == pytest.approx(1.0)

    def test_prompt_without_a_terminal_aborts(self, h2_checkpoint, monkeypatch):
        monkeypatch.setattr(matrix_free, "available_memory_bytes",
                            lambda: 1024)
        monkeypatch.setattr(qpe_module.sys.stdin, "isatty", lambda: False,
                            raising=False)
        qpe = QuantumPhaseEstimation(n_evaluation_qubits=4, verbose=False,
                                     memory_policy="prompt")
        with pytest.raises(MemoryError, match="no terminal"):
            qpe.run(h2_checkpoint)

    def test_a_large_but_allowed_run_warns(self, monkeypatch):
        est = qpe_memory_estimate(4, 10)
        monkeypatch.setattr(matrix_free, "available_memory_bytes",
                            lambda: int(est.total_bytes / 0.3))
        with pytest.warns(RuntimeWarning, match="large QPE simulation"):
            assert qpe_module.check_memory(qpe_memory_estimate(4, 10), "abort",
                                           0.5, verbose=False)

    def test_unknown_policy(self):
        with pytest.raises(ValueError, match="memory_policy"):
            QuantumPhaseEstimation(memory_policy="hope")


class TestPrecision:
    """``precision="float32"`` simulates the ``2^(n+t)`` register in complex64."""

    def test_float32_reads_the_same_energy_from_complex64_amplitudes(
            self, h2_checkpoint):
        wide = QuantumPhaseEstimation(n_evaluation_qubits=8, atomic_units=True,
                                      verbose=False).run(h2_checkpoint)
        narrow = QuantumPhaseEstimation(n_evaluation_qubits=8, atomic_units=True,
                                        verbose=False,
                                        precision="float32").run(h2_checkpoint)
        assert wide._amplitudes.dtype == np.complex128
        assert narrow._amplitudes.dtype == np.complex64
        assert narrow.collapsed_state().dtype == np.complex64
        assert narrow.energy == wide.energy
        np.testing.assert_allclose(narrow.probabilities, wide.probabilities,
                                   atol=1e-6)
        assert narrow.probabilities.sum() == pytest.approx(1.0, abs=1e-6)

    def test_float32_from_a_state_vector_and_a_preparation_tuple(self, h2_checkpoint):
        h = h2_checkpoint.hamiltonian
        qpe = QuantumPhaseEstimation(h, n_evaluation_qubits=6, verbose=False,
                                     precision="float32")
        from_vector = qpe.run(np.asarray(h2_checkpoint.state_vector(),
                                         dtype=np.complex128))
        from_tuple = qpe.run((h2_checkpoint.n_qubits,
                              h2_checkpoint.reference_qubits,
                              h2_checkpoint.generators, h2_checkpoint.parameters))
        assert from_vector._amplitudes.dtype == np.complex64
        assert from_tuple._amplitudes.dtype == np.complex64
        np.testing.assert_allclose(from_tuple.probabilities,
                                   from_vector.probabilities, atol=1e-6)

    def test_the_estimate_counts_eight_bytes_per_stored_amplitude(self):
        wide = qpe_memory_estimate(4, 10)
        narrow = qpe_memory_estimate(4, 10, precision="float32")
        assert narrow.statevector_bytes * 2 == wide.statevector_bytes
        assert narrow.unitary_bytes == wide.unitary_bytes
        assert QuantumPhaseEstimation(
            precision="float32").memory_estimate(4).statevector_bytes == 8 * 2 ** 12

    @pytest.mark.parametrize("precision", ["single", "double", "FLOAT32", None])
    def test_any_other_precision_raises(self, precision):
        with pytest.raises(ValueError, match="precision"):
            QuantumPhaseEstimation(precision=precision)
        with pytest.raises(ValueError, match="precision"):
            qpe_memory_estimate(4, 10, precision=precision)
