# -*- coding: utf-8 -*-
# file: test_proposal_data.py

"""The edit store a Markov-chain ansatz search writes with ``record=DIR``.

Every proposal becomes a row -- rejected ones included, as they are the
negative examples -- and the problem it searched is stored once.  Reached
through ``Mandacaru(method="mcas-vqe" | "valqa", record=...)``.
"""

import json
import math

import numpy as np
import pytest

from mandacaru import Mandacaru
from mandacaru.algorithms.proposal_data import (EDITS_FILE, PROBLEMS_DIR,
                                                EditRecorder, ProblemRecord,
                                                load_edits)
from mandacaru.core import MolecularIntegrals, minimal_hao_basis
from mandacaru.integrals import Grid
from mandacaru.units import HARTREE_TO_EV


def _h2(R):
    nuclei = [(1.0, np.array([0.0, 0.0, -R / 2])),
              (1.0, np.array([0.0, 0.0, +R / 2]))]
    grid = Grid(center=[0.0, 0.0, 0.0], box_size=5.0, h=0.25)
    mints = MolecularIntegrals(nuclei, minimal_hao_basis(nuclei), grid)
    return mints.molecular_hamiltonian(mo_basis=True, n_electrons=2)


@pytest.fixture(scope="module")
def h2_hamiltonian():
    return _h2(0.74)


def _chain(h2_hamiltonian, method="mcas-vqe", **options):
    options = {"pool": "qubit", "max_steps": 12, "max_length": 4, "seed": 7,
               "profile": False, "trace": False, **options}
    return Mandacaru(method=method, hamiltonian=h2_hamiltonian,
                     num_particles=(1, 1), n_spatial_orbitals=2, **options)


def _rows(directory):
    return [json.loads(line) for line in
            (directory / EDITS_FILE).read_text().splitlines()]


class TestRecording:
    def test_one_row_per_proposal_rejections_included(self, h2_hamiltonian,
                                                      tmp_path):
        result = _chain(h2_hamiltonian, record=tmp_path).run()
        rows = _rows(tmp_path)
        assert len(rows) == len(result.steps)
        assert [r["accepted"] for r in rows] == \
            [s.accepted for s in result.steps]
        assert not all(r["accepted"] for r in rows)
        assert [tuple(r["proposed"]) for r in rows] == \
            [s.proposed for s in result.steps]

    def test_a_row_holds_the_state_before_and_the_change_in_hartree(
            self, h2_hamiltonian, tmp_path):
        result = _chain(h2_hamiltonian, record=tmp_path).run()
        rows = _rows(tmp_path)
        current = rows[0]["reference_energy"]
        for row, step in zip(rows, result.steps):
            assert row["source_energy"] == pytest.approx(current, abs=1e-12)
            assert (row["source_energy"] + row["delta_energy"]) \
                * HARTREE_TO_EV == pytest.approx(step.proposed_energy,
                                                 abs=1e-9)
            assert len(row["source_gradients"]) == 12
            current = step.current_energy / HARTREE_TO_EV

    def test_the_drawn_probability_and_ratio_are_kept(self, h2_hamiltonian,
                                                      tmp_path):
        result = _chain(h2_hamiltonian, record=tmp_path).run()
        for row, step in zip(_rows(tmp_path), result.steps):
            assert row["log_q_forward"] == step.log_q_forward
            if row["move"] in ("insert", "replace"):
                assert 0.0 < row["operator_probability"] <= 1.0
            else:
                assert row["operator_probability"] is None
            assert row["model"] is None and row["method"] == "MCAS-VQE"

    def test_a_uniform_chain_that_records_screens_the_pool(
            self, h2_hamiltonian, tmp_path):
        """The gradients are features of every row, whatever the proposal."""
        plain = _chain(h2_hamiltonian, proposal="uniform").run()
        recorded = _chain(h2_hamiltonian, proposal="uniform",
                          record=tmp_path).run()
        assert plain.num_screenings == 0 and recorded.num_screenings > 0
        assert [s.action for s in plain.steps] == \
            [s.action for s in recorded.steps]
        assert all(r["operator_probability"] in (None, 1.0 / 12)
                   for r in _rows(tmp_path))

    def test_runs_on_one_problem_share_its_file(self, h2_hamiltonian,
                                                tmp_path):
        _chain(h2_hamiltonian, record=tmp_path).run()
        _chain(h2_hamiltonian, record=tmp_path, seed=3).run()
        files = list((tmp_path / PROBLEMS_DIR).iterdir())
        assert len(files) == 1
        rows, problems = load_edits(tmp_path)
        assert len({r["run"] for r in rows}) == 2
        assert list(problems) == [files[0].stem]

    def test_the_problem_round_trips(self, h2_hamiltonian, tmp_path):
        calc = _chain(h2_hamiltonian, record=tmp_path)
        calc.run()
        _, problems = load_edits(tmp_path)
        (problem,) = problems.values()
        solver = calc.solver
        again = ProblemRecord.from_problem(
            solver.hamiltonian, solver._new_ansatz().reference_qubits(),
            solver._pool_ops)
        assert again.key == problem.key
        assert problem.pool_labels == tuple(op.label
                                            for op in solver._pool_ops)
        np.testing.assert_allclose(problem.coefficients, again.coefficients)
        assert problem.reference.sum() == 2
        assert problem.terms.shape[1] == 4 and not (problem.terms == 0).all(
            axis=1).any()


class TestAppending:
    """The store only grows: runs append, interrupted rows are tolerated."""

    def test_a_second_run_keeps_the_first_runs_rows(self, h2_hamiltonian,
                                                    tmp_path):
        _chain(h2_hamiltonian, record=tmp_path).run()
        before = (tmp_path / EDITS_FILE).read_bytes()
        second = _chain(h2_hamiltonian, record=tmp_path, seed=3).run()
        after = (tmp_path / EDITS_FILE).read_bytes()
        assert after.startswith(before)
        assert len(after.splitlines()) == \
            len(before.splitlines()) + len(second.steps)

    def test_an_existing_problem_file_is_not_rewritten(self, h2_hamiltonian,
                                                       tmp_path):
        _chain(h2_hamiltonian, record=tmp_path).run()
        (path,) = (tmp_path / PROBLEMS_DIR).iterdir()
        stamp = path.stat().st_mtime_ns
        _chain(h2_hamiltonian, record=tmp_path, seed=3).run()
        assert path.stat().st_mtime_ns == stamp
        assert [p.name for p in (tmp_path / PROBLEMS_DIR).iterdir()] == \
            [path.name]

    def test_two_problems_share_one_directory(self, h2_hamiltonian,
                                              tmp_path):
        _chain(h2_hamiltonian, record=tmp_path).run()
        _chain(_h2(0.9), record=tmp_path).run()
        rows, problems = load_edits(tmp_path)
        assert len(problems) == 2
        assert len(list((tmp_path / PROBLEMS_DIR).iterdir())) == 2
        assert {r["problem"] for r in rows} == set(problems)

    def test_a_row_cut_short_is_skipped_with_its_line(self, h2_hamiltonian,
                                                      tmp_path):
        result = _chain(h2_hamiltonian, record=tmp_path).run()
        with open(tmp_path / EDITS_FILE, "a") as handle:
            handle.write('{"schema":1,"prob')
        with pytest.warns(RuntimeWarning, match=f"line\\(s\\) "
                          f"{len(result.steps) + 1}$"):
            rows, _ = load_edits(tmp_path)
        assert len(rows) == len(result.steps)

    def test_the_next_run_starts_a_new_line_after_a_cut_row(
            self, h2_hamiltonian, tmp_path):
        first = _chain(h2_hamiltonian, record=tmp_path).run()
        with open(tmp_path / EDITS_FILE, "a") as handle:
            handle.write('{"schema":1,"prob')
        second = _chain(h2_hamiltonian, record=tmp_path, seed=3).run()
        with pytest.warns(RuntimeWarning, match="skipped 1 row"):
            rows, _ = load_edits(tmp_path)
        assert len(rows) == len(first.steps) + len(second.steps)

    def test_close_is_idempotent_and_ends_writing(self, h2_hamiltonian,
                                                  tmp_path):
        calc = _chain(h2_hamiltonian)
        problem = ProblemRecord.from_problem(
            calc.solver.hamiltonian,
            calc.solver._new_ansatz().reference_qubits(),
            calc.solver._pool_ops)
        recorder = EditRecorder(tmp_path, problem, run={"run": "r"})
        recorder.write({"step": 1})
        recorder.close()
        recorder.close()
        with pytest.raises(ValueError, match="closed"):
            recorder.write({"step": 2})
        assert len(_rows(tmp_path)) == 1

    @pytest.mark.parametrize("where", ["banner", "step"])
    def test_a_failing_run_closes_the_file_and_keeps_its_rows(
            self, h2_hamiltonian, tmp_path, monkeypatch, where):
        calc = _chain(h2_hamiltonian, record=tmp_path)
        solver = calc.solver
        opened = []
        open_recorder = solver._open_recorder

        def spy(*args):
            opened.append(open_recorder(*args))
            return opened[-1]

        monkeypatch.setattr(solver, "_open_recorder", spy)
        if where == "banner":
            def fail(*args, **kwargs):
                raise RuntimeError("boom")
            monkeypatch.setattr(solver, "_show_banner", fail)
        else:
            evaluate, calls = solver._evaluate, []

            def fail(*args, **kwargs):
                calls.append(1)
                if len(calls) == 5:
                    raise RuntimeError("boom")
                return evaluate(*args, **kwargs)
            monkeypatch.setattr(solver, "_evaluate", fail)
        with pytest.raises(RuntimeError, match="boom"):
            calc.run()
        assert solver._recorder is None and opened[0]._fd is None
        if where == "step":
            assert len(_rows(tmp_path)) == 3


class TestRefusals:
    def test_a_file_is_not_a_record_directory(self, tmp_path):
        path = tmp_path / "file"
        path.write_text("")
        with pytest.raises(ValueError, match="is a file"):
            Mandacaru(method="mcas-vqe", record=str(path))

    def test_the_parent_must_exist(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            Mandacaru(method="mcas-vqe", record=str(tmp_path / "a" / "b"))

    def test_rows_without_their_problem_are_refused(self, h2_hamiltonian,
                                                    tmp_path):
        _chain(h2_hamiltonian, record=tmp_path).run()
        for path in (tmp_path / PROBLEMS_DIR).iterdir():
            path.unlink()
        with pytest.raises(FileNotFoundError, match="is missing"):
            load_edits(tmp_path)

    def test_an_empty_directory_has_no_edits(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="record a chain"):
            load_edits(tmp_path)

    def test_problems_without_edits_are_refused(self, h2_hamiltonian,
                                                tmp_path):
        _chain(h2_hamiltonian, record=tmp_path).run()
        (tmp_path / EDITS_FILE).unlink()
        with pytest.raises(FileNotFoundError, match="record a chain"):
            load_edits(tmp_path)

    def test_rows_of_another_schema_are_refused(self, h2_hamiltonian,
                                                tmp_path):
        _chain(h2_hamiltonian, record=tmp_path).run()
        with open(tmp_path / EDITS_FILE, "a") as handle:
            handle.write(json.dumps({"schema": 2, "problem": "x"}) + "\n")
        with pytest.raises(ValueError, match="row schema 2"):
            load_edits(tmp_path)


def test_rejected_reverse_moves_serialize(h2_hamiltonian, tmp_path):
    """``-inf`` log-probabilities survive the JSON round trip."""
    _chain(h2_hamiltonian, record=tmp_path, max_steps=20).run()
    rows, _ = load_edits(tmp_path)
    assert all(isinstance(r["log_acceptance"], float) for r in rows)
    assert all(not math.isnan(r["log_acceptance"]) for r in rows)


class TestTheSharedStore:
    """``MANDACARU_PROPOSAL_DATA``: where chains record by default."""

    @pytest.fixture
    def store(self, tmp_path, monkeypatch):
        path = tmp_path / "store"
        monkeypatch.setenv("MANDACARU_PROPOSAL_DATA", str(path))
        return path

    def test_a_chain_records_there_by_default(self, h2_hamiltonian, store):
        result = _chain(h2_hamiltonian).run()
        assert len(_rows(store)) == len(result.steps)
        rows, _ = load_edits()
        assert len(rows) == len(result.steps)

    def test_record_false_records_nothing(self, h2_hamiltonian, store):
        _chain(h2_hamiltonian, record=False).run()
        assert not store.exists()

    def test_a_named_directory_takes_precedence(self, h2_hamiltonian, store,
                                                tmp_path):
        _chain(h2_hamiltonian, record=tmp_path / "elsewhere").run()
        assert not store.exists()
        assert _rows(tmp_path / "elsewhere")

    def test_record_true_needs_the_store(self, monkeypatch):
        monkeypatch.delenv("MANDACARU_PROPOSAL_DATA", raising=False)
        with pytest.raises(ValueError, match="MANDACARU_PROPOSAL_DATA"):
            Mandacaru(method="mcas-vqe", record=True)

    def test_without_the_store_nothing_is_recorded(self, h2_hamiltonian,
                                                   monkeypatch, tmp_path):
        monkeypatch.delenv("MANDACARU_PROPOSAL_DATA", raising=False)
        monkeypatch.chdir(tmp_path)
        _chain(h2_hamiltonian).run()
        assert list(tmp_path.iterdir()) == []
        with pytest.raises(ValueError, match="--set-proposal-data"):
            load_edits()
