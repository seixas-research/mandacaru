"""Search energies, reversible edits, recorded observations and owned results."""
import json

import numpy as np
import pytest

from mandacaru import Mandacaru, _native
from mandacaru.utils.logging import parse_output


def calculator(**options):
    context = _native.Context(threads=1)
    operator = context.pauli_operator(4, [("ZIII", .3, 0.), ("XXII", .2, 0.), ("YYII", .2, 0.)])
    settings = dict(method="vasqa", hamiltonian=operator,
                    n_spatial_orbitals=2, num_particles=(1, 0),
                    max_steps=40, max_length=4, seed=7, atomic_units=True,
                    profile=False, parallelization={"threads": 1})
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
        assert set(result.acceptance_by_move) == {"insert", "delete", "replace"}
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
        # Each operator of the source carries its importance and its removal weight.
        assert len(row["source_importances"]) == len(row["source"]) == len(row["source_removal"])
        assert all(value >= 0 for value in row["source_importances"])
        if row["source"]:
            assert sum(row["source_removal"]) == pytest.approx(1.)


def test_architecture_search_rejects_invalid_controls():
    for options in ({"min_length": 1}, {"screen_insertions": 1},
                    {"temperature": {"initial": .1}}, {"seed": -1},
                    {"replace_start": "alias"}, {"move_weights": {"swap": 1.0}},
                    {"replicas": 0}, {"replicas": 1.5}, {"replicas": 2, "replica_spacing": 0.5},
                    {"replicas": 2, "replica_spacing": float("inf")},
                    {"replica_spacing": 3.0},
                    {"replicas": 2, "temperature": 0.0}):
        with pytest.raises((TypeError, ValueError)):
            with calculator(**options) as calc:
                calc.run()


def test_replica_exchange_reports_every_replica_and_its_swaps(tmp_path):
    with calculator(replicas=3, replica_spacing=2.0, txt=tmp_path / "run.txt") as calc:
        result = calc.run()
        raw = result._raw
        assert result.energy_hartree == pytest.approx(-.5, abs=1e-9)
        # Each step relaxes one proposal per replica and attempts one adjacent swap.
        assert len(raw["chain_steps"]) == 3 * 40
        assert {row["replica"] for row in raw["chain_steps"]} == {0, 1, 2}
        taken, tried = result.replica_swaps
        assert tried == len(raw["swaps"]) == 40 and 0 <= taken <= tried
        cold = [row["temperature"] for row in raw["chain_steps"] if row["replica"] == 0]
        hot = [row["temperature"] for row in raw["chain_steps"] if row["replica"] == 2]
        assert hot == pytest.approx([4 * t for t in cold])
    text = (tmp_path / "run.txt").read_text()
    assert "replicas" in text and f"{taken}/{tried}" in text


def test_one_replica_is_the_single_chain():
    with calculator() as single, calculator(replicas=1, replica_spacing=2.0) as named:
        first, second = single.run(), named.run()
        assert first._raw["chain_steps"] == second._raw["chain_steps"]
        assert second.replica_swaps is None
        assert "replica" not in second._raw["chain_steps"][0]

