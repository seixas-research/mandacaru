"""Search energies, reversible edits, recorded observations and owned results."""
import json

import numpy as np
import pytest

from mandacaru import Mandacaru, _native
from mandacaru.utils.logging import parse_output


def calculator(**options):
    context = _native.Context(threads=1)
    operator = context.pauli_operator(4, [("ZIII", .3, 0.), ("XXII", .2, 0.), ("YYII", .2, 0.)])
    settings = dict(method="mcas-vqe", hamiltonian=operator,
                    n_spatial_orbitals=2, num_particles=(1, 0),
                    max_steps=40, max_length=4, seed=7, atomic_units=True,
                    profile=False, resources={"threads": 1})
    settings.update(options)
    return Mandacaru(**settings)


def test_native_search_preserves_state_energy_checkpoint_and_records(tmp_path, monkeypatch):
    import scipy.optimize

    def forbidden(*args, **kwargs):
        pytest.fail("the search must execute in Rust")

    monkeypatch.setattr(scipy.optimize, "minimize", forbidden)
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    with calculator(record=tmp_path / "store", screen_insertions=1,
                    warm_start=False, txt=tmp_path / "run.txt") as calc:
        result = calc.run()
        assert result.completed and not result.converged
        assert result.energy_hartree == pytest.approx(-.5, abs=1e-9)
        assert result.best_energy == pytest.approx(-.5, abs=1e-9)
        assert result.optimal_cost == pytest.approx(result.optimal_energy)
        assert len(result.steps) == 40
        assert sum(tried for taken, tried in result.acceptance_by_move.values()) == 40
        assert result.num_evaluations > 0
        state = result.state()
        h = result.qubit_hamiltonian().matrix()
        assert np.vdot(state, h @ state).real == pytest.approx(result.energy_hartree, abs=1e-12)
        assert np.allclose(state, calc.checkpoint.state(), atol=1e-12)
        assert result.steps[-1].current_energy == pytest.approx(result.final_energy)
        snapshot = parse_output(tmp_path / "run.txt")
        assert snapshot
    paths = list((tmp_path / "store" / "chains").glob("*.jsonl"))
    assert len(paths) == 1
    records = [json.loads(row) for row in paths[0].read_text().splitlines()]
    assert len([row for row in records if not row["screened"]]) == 40
    assert any(row["screened"] for row in records)
    for row in records:
        assert (tmp_path / "store" / "problems" / f"{row['problem_id']}.json").exists()
        assert sum(row["source_probabilities"]) == pytest.approx(1.)
        assert row["delta_energy"] == pytest.approx(row["energy"] - row["source_energy"])


def test_architecture_search_rejects_invalid_controls():
    for options in ({"min_length": 1}, {"screen_insertions": 1},
                    {"temperature": {"initial": .1}}, {"seed": -1},
                    {"replace_start": "alias"}):
        with pytest.raises((TypeError, ValueError)):
            with calculator(**options) as calc:
                calc.run()
