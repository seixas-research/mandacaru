# -*- coding: utf-8 -*-
# file: test/test_review_2026_09_18.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Regressions for the source-and-output review of 2026-09-18.

Each class pins one finding of ``codex/source_and_output_review_2026-09-18.md``,
named by its identifier, and each was verified to fail before the fix.  The
common thread is **result identity**: a number in a file must be the number the
run produced, for the problem the run actually solved.
"""

import os
import warnings

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.core.checkpoint import WavefunctionCheckpoint, fingerprint
from mandacaru.core.mapping import PauliSum
from mandacaru.core.serialization import load_hamiltonian, save_hamiltonian
from mandacaru.utils import parse_output

DIRECT = dict(num_particles=(1, 1), n_spatial_orbitals=2, atomic_units=True,
              trace=False, profile=False)


def h2(distance=0.74, cell=6.0):
    middle = cell / 2
    return Atoms("H2", positions=[[middle, middle, middle - distance / 2],
                                  [middle, middle, middle + distance / 2]],
                 cell=[cell] * 3)


class TestF01ResumeEnergyIdentity:
    """A resumed run reports *its* energy, not the file's."""

    SHIFT = 2.0

    @staticmethod
    def _state(path, hamiltonian):
        Mandacaru(method="adapt-vqe", hamiltonian=hamiltonian,
                  pool="fermionic", checkpoint=path, max_iterations=0,
                  **DIRECT).run()

    def test_a_shifted_hamiltonian_is_reported_at_its_own_energy(self, tmp_path):
        path = str(tmp_path / "state.json")
        original = PauliSum({"ZIII": 1.0})
        self._state(path, original)
        shifted = PauliSum({"ZIII": 1.0, "IIII": self.SHIFT})

        with pytest.warns(RuntimeWarning, match="different Hamiltonian"):
            result = Mandacaru(method="adapt-vqe", hamiltonian=shifted,
                               pool="fermionic", resume=path, max_iterations=0,
                               **DIRECT).run()
        psi = WavefunctionCheckpoint.load(path).state_vector()
        exact = float(np.real(psi.conj() @ (shifted.to_sparse_matrix() @ psi)))
        # Before the fix this returned the stored -1 Ha for a +1 Ha state.
        assert result.optimal_energy == pytest.approx(exact, abs=1e-10)

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

    def test_a_warm_start_drops_the_foreign_history(self, tmp_path):
        path = str(tmp_path / "state.json")
        Mandacaru(method="adapt-vqe",
                  hamiltonian=PauliSum({"ZIII": 1.0, "XXII": 0.3}),
                  pool="fermionic", checkpoint=path, max_iterations=2,
                  **DIRECT).run()
        with pytest.warns(RuntimeWarning, match="history discarded"):
            result = Mandacaru(method="adapt-vqe",
                               hamiltonian=PauliSum({"ZIII": 1.0, "XXII": 0.3, "IIII": 1.0}),
                               pool="fermionic", resume=path, max_iterations=2,
                               **DIRECT).run()
        # The restored operators stay; their energies, measured against another
        # operator, do not -- so there are fewer iteration rows than operators.
        assert result.num_operators >= 1
        assert len(result.iterations) < result.num_operators


class TestF02ReportingOptionsAreHonest:
    """An option that writes a file either writes it or is refused."""

    def test_vqe_refuses_a_log_it_would_ignore(self):
        # `txt=` is a driver-wide option now (every driver has to know where it
        # reports), so the refusal is about the *capability*, not the keyword.
        with pytest.raises(NotImplementedError, match="does not write 'txt'"):
            Mandacaru(method="vqe", basis="FAO", txt="x.txt")

    def test_a_subspace_method_refuses_checkpoints(self):
        with pytest.raises(NotImplementedError, match="does not write"):
            Mandacaru(method="subspace-adapt-vqe", basis="FAO",
                      checkpoint="x.json")

    def test_an_unknown_option_is_not_kept_as_an_ase_parameter(self):
        with pytest.raises(TypeError, match="does not take 'nonsense'"):
            Mandacaru(method="adapt-vqe", basis="FAO", nonsense=1)

    def test_vqe_keeps_a_checkpoint_it_does_support(self, tmp_path):
        Mandacaru(method="vqe", basis="FAO", checkpoint=str(tmp_path / "c.json"))

    def test_a_method_without_a_log_keeps_its_trace(self, capsys, tmp_path):
        """The trap: no file *and* no trace is a run that reports nothing."""
        atoms = h2()
        atoms.calc = Mandacaru(method="vqe", basis="FAO", h=0.45)
        atoms.get_potential_energy()
        assert "VQE" in capsys.readouterr().out


class TestF03SubspaceConvergence:
    def test_a_stationary_state_with_no_budget_is_converged(self):
        solver = Mandacaru(method="subspace-adapt-vqe",
                           hamiltonian=PauliSum({"ZIII": 1.0, "IZII": 0.5}),
                           pool="fermionic", num_states=2, max_iterations=0,
                           gradient_tolerance=1e3, **DIRECT)
        result = solver.run()
        # The verdict follows the final screening, as ordinary ADAPT's does.
        assert result.final_max_gradient < 1e3
        assert result.converged is True


class TestF04MeasuredEnergyIsRecorded:
    def test_the_log_carries_the_energy_ase_returned(self, tmp_path):
        from mandacaru.backends.providers import QiskitProvider

        out = str(tmp_path / "output.txt")
        atoms = h2()
        atoms.calc = Mandacaru(
            method="adapt-vqe", basis="FAO", h=0.45, pool="fermionic",
            max_iterations=2, gradient_tolerance=1e-3, txt=out, profile=False,
            measurement_provider=QiskitProvider(device="statevector", shots=0))
        energy = atoms.get_potential_energy()
        block = parse_output(out)["measurement"]
        assert float(block["energy_eV"]) == pytest.approx(energy, abs=1e-9)
        assert block["reported_by"].startswith("ASE")
        # The optimization's own value is kept, under its own name.
        assert "variational_energy_eV" in block


class TestF05CompletionIsNotAssumed:
    def test_the_exit_hook_does_not_claim_completion(self, tmp_path):
        out = str(tmp_path / "output.txt")
        atoms = h2()
        atoms.calc = Mandacaru(method="adapt-vqe", basis="FAO", h=0.45,
                               pool="fermionic", max_iterations=2, profile=False,
                               gradient_tolerance=1e-3, txt=out)
        atoms.get_forces()
        atoms.positions[1, 2] += 0.03
        atoms.get_forces()
        # Reaching interpreter exit says the process ended, not that the
        # optimization finished: exit handlers run after an exception too.
        atoms.calc._write_summary_at_exit()
        status = parse_output(out)["completion"]["status"]
        assert "not signaled" in status
        # It must not *claim* a verdict.  The word appears only in the clause
        # explaining that reaching exit does not establish one.
        assert not status.startswith(("converged", "NOT converged", "finished"))

    def test_an_empty_trajectory_is_not_forced(self, tmp_path):
        calc = Mandacaru(method="adapt-vqe", basis="FAO",
                         txt=str(tmp_path / "output.txt"))
        # Used to raise IndexError.
        assert calc.write_optimization_summary(force=True) is False

    def test_a_failed_write_can_be_retried(self, tmp_path, monkeypatch):
        out = str(tmp_path / "output.txt")
        atoms = h2()
        atoms.calc = Mandacaru(method="adapt-vqe", basis="FAO", h=0.45,
                               pool="fermionic", max_iterations=2, profile=False,
                               gradient_tolerance=1e-3, txt=out)
        atoms.get_forces()
        atoms.positions[1, 2] += 0.03
        atoms.get_forces()

        import mandacaru.utils.logging as logging_module
        def explode(*args, **kwargs):
            raise OSError("disk full")
        monkeypatch.setattr(logging_module, "append_optimization_summary",
                            explode)
        with pytest.raises(OSError):
            atoms.calc.write_optimization_summary()
        # The state must not say "written" after a write that failed.
        monkeypatch.undo()
        assert atoms.calc.write_optimization_summary() is True


class TestF06LogLifetime:
    def test_a_missing_parent_is_created_without_consuming_a_step(self, tmp_path):
        from mandacaru.utils.logging import AdaptOutputLogger, log_steps

        path = str(tmp_path / "deep" / "nested" / "output.txt")
        with AdaptOutputLogger(path, n_qubits=4) as logger:
            logger.write_system()
            assert logger.step == 1
        assert log_steps(path) == 1
        assert os.path.exists(path)


class TestF07ParserPreservesData:
    @pytest.fixture
    def written(self, tmp_path):
        from types import SimpleNamespace

        from mandacaru.utils.logging import AdaptOutputLogger

        path = str(tmp_path / "output.txt")
        cell = np.array([[4.0, 0.0, 0.0], [0.5, 5.0, 0.0], [0.0, 0.0, 6.0]])
        pool = [SimpleNamespace(label="custom operator with spaces",
                                kind="double",
                                generator=PauliSum({"XXXX": 0.5j}))]
        with AdaptOutputLogger(path, n_qubits=4) as logger:
            logger.write_system(symbols=["H", "He"],
                                positions=[[1.0, 2.0, 3.0], [0.0, 0.0, 0.74]],
                                cell=cell)
            logger.write_iteration(1, pool, [0.3], 0, None, -1.5, 7)
        return path, cell

    def test_the_geometry_survives_as_numbers(self, written):
        path, _cell = written
        geometry = parse_output(path)["system"]["geometry"]
        assert [row[0] for row in geometry] == ["H", "He"]
        assert np.allclose([row[1:] for row in geometry],
                           [[1.0, 2.0, 3.0], [0.0, 0.0, 0.74]])

    def test_the_cell_vectors_survive(self, written):
        path, cell = written
        # Lengths and angles cannot recover the orientation; the vectors can.
        assert np.allclose(parse_output(path)["system"]["cell_vectors"], cell)

    def test_an_operator_label_may_contain_spaces(self, written):
        path, _cell = written
        entry = parse_output(path)["iterations"][0]
        assert entry["selected_operator"] == "custom operator with spaces"
        assert entry["operator_kind"] == "double"
        assert entry["energy"] == pytest.approx(-1.5)

    def test_an_unrecorded_field_is_not_invented(self, written):
        path, _cell = written
        # The writer took num_parameters=7 and does not serialize it: reported
        # as unknown rather than as the iteration index.
        assert parse_output(path)["iterations"][0]["num_parameters"] is None


class TestF08OutputPathsAndSnapshots:
    def test_the_default_dump_and_cache_names_differ(self):
        from mandacaru.core.serialization import resolve_save_path
        from mandacaru.utils.dumps import HAMILTONIAN_FILE, resolve_dump_path

        assert resolve_dump_path(True, HAMILTONIAN_FILE) != \
            resolve_save_path(True, "json")

    def test_two_outputs_on_one_path_are_refused(self, tmp_path):
        path = str(tmp_path / "same.json")
        with pytest.raises(ValueError, match="both resolve to"):
            Mandacaru(method="adapt-vqe", hamiltonian=PauliSum({"ZI": 1.0}),
                      num_particles=(1, 1), n_spatial_orbitals=1,
                      save_hamiltonian=path, verbose_hamiltonian=path)

    def test_an_inspection_dump_is_not_loadable_as_a_cache(self, tmp_path):
        from mandacaru.utils.dumps import dump_hamiltonian

        path = str(tmp_path / "h.json")
        dump_hamiltonian(path, PauliSum({"ZI": 1.0}), n_qubits=2)
        # Used to fail with KeyError: 0.
        with pytest.raises(ValueError, match="inspection"):
            load_hamiltonian(path)

    def test_a_failed_write_keeps_the_previous_snapshot(self, tmp_path,
                                                       monkeypatch):
        path = str(tmp_path / "h.json")
        save_hamiltonian(path, PauliSum({"ZI": 1.0}), num_particles=(1, 1),
                         n_spatial_orbitals=1)
        good = open(path, encoding="utf-8").read()

        import mandacaru.core.serialization as serialization
        def explode(*args, **kwargs):
            raise OSError("disk full")
        monkeypatch.setattr(serialization, "_write_json", explode)
        with pytest.raises(OSError):
            save_hamiltonian(path, PauliSum({"ZI": 2.0}), num_particles=(1, 1),
                             n_spatial_orbitals=1)
        # The snapshot that was there is still there, and still loadable.
        assert open(path, encoding="utf-8").read() == good
        assert load_hamiltonian(path).hamiltonian.terms == {"ZI": 1.0 + 0j}
        assert not [name for name in os.listdir(tmp_path)
                    if name.endswith(".tmp")]


class TestF09ZeroHamiltonianWidth:
    @pytest.mark.parametrize("fmt", ["json", "parquet"])
    def test_a_zero_operator_keeps_its_register(self, tmp_path, fmt):
        path = str(tmp_path / f"zero.{fmt}")
        save_hamiltonian(path, PauliSum({}, num_qubits=4), format=fmt,
                         num_particles=(1, 1), n_spatial_orbitals=2)
        record = load_hamiltonian(path)
        assert record.hamiltonian.num_qubits == 4

    def test_terms_that_cancel_keep_their_register(self, tmp_path):
        path = str(tmp_path / "cancel.json")
        # A term whose coefficient canceled to zero is dropped on simplify();
        # the register width must survive it.  (A dict literal cannot repeat a
        # key, so the canceled term is written as the zero it sums to.)
        save_hamiltonian(path, PauliSum({"ZIII": 0.0}),
                         num_particles=(1, 1), n_spatial_orbitals=2)
        assert load_hamiltonian(path).hamiltonian.num_qubits == 4


class TestF10ProjectedForcesAreQualified:
    @pytest.fixture(scope="class")
    def logged(self, tmp_path_factory):
        out = str(tmp_path_factory.mktemp("f10") / "output.txt")
        # Asymmetric, so the projection removes something.
        atoms = Atoms("H3", positions=[[3, 3, 2.6], [3, 3, 3.4], [3, 3.7, 3.1]],
                      cell=[6.0] * 3)
        # profile=False: this file reads Parquet above, and building Qiskit
        # circuits in the same process then crashes in Qiskit's allocator (the
        # documented pandas/pyarrow interaction).  The forces need no circuits.
        atoms.calc = Mandacaru(method="adapt-vqe", basis="FAO", h=0.45, charge=1,
                               pool="fermionic", max_iterations=3, profile=False,
                               gradient_tolerance=1e-3, txt=out)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            atoms.get_forces()
        return parse_output(out)["forces"], atoms.calc.force_result

    def test_both_arrays_are_written(self, logged):
        block, result = logged
        assert np.allclose(block["forces"], result.forces, atol=1e-7)
        assert np.allclose(block["forces_unprojected"], result.unprojected,
                           atol=1e-7)

    def test_the_relation_between_them_is_recorded(self, logged):
        block, _result = logged
        raw = np.asarray(block["forces_unprojected"])
        assert np.allclose(block["forces"], raw - raw.mean(axis=0), atol=1e-7)
        assert "forces_unprojected - mean" in block["convention"]

    def test_the_unqualified_convention_is_not_claimed(self, logged):
        block, _result = logged
        # The projected force is not -(HF + Pulay); the block must not say it is.
        assert block["convention"] != \
            "forces = -dE/dR (ASE sign); hellmann_feynman and pulay are +dE/dR"
        assert "translation_removed" in block


class TestF11Provenance:
    def test_the_gradient_units_are_stated(self, tmp_path):
        out = str(tmp_path / "output.txt")
        Mandacaru(method="adapt-vqe",
                  hamiltonian=PauliSum({"ZIII": 1.0, "XXII": 0.2}),
                  pool="fermionic", txt=out, max_iterations=1, **DIRECT).run()
        assert parse_output(out)["setup"]["gradient_units"] == "Hartree"

    def test_a_resumed_run_records_its_lineage(self, tmp_path):
        state = str(tmp_path / "state.json")
        out = str(tmp_path / "output.txt")
        hamiltonian = PauliSum({"ZIII": 1.0, "XXII": 0.2})
        Mandacaru(method="adapt-vqe", hamiltonian=hamiltonian,
                  pool="fermionic", checkpoint=state, max_iterations=1,
                  **DIRECT).run()
        Mandacaru(method="adapt-vqe", hamiltonian=hamiltonian,
                  pool="fermionic", resume=state, txt=out, max_iterations=2,
                  **DIRECT).run()
        setup = parse_output(out)["setup"]
        assert setup["resumed_from"] == state
        assert int(setup["restored_operators"]) == 1
        assert setup["resume_same_hamiltonian"] == "True"
        # And the block no longer claims the run started from |HF>.
        assert setup["initial_ansatz"].startswith("resumed")
