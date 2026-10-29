"""Native learned proposal execution, persistence, and honest readiness."""
import json

import numpy as np
import pytest

from mandacaru import Mandacaru, _native
from mandacaru.analysis.proposals import ProposalModel, fit


def calculator(**options):
    ctx = _native.Context(threads=1)
    h = ctx.pauli_operator(4, [("ZIII", .3, 0.), ("XXII", .2, 0.), ("YYII", .2, 0.)])
    settings = dict(method="valqa", hamiltonian=h, n_spatial_orbitals=2,
                    num_particles=(1, 0), max_steps=20, max_length=3,
                    seed=7, atomic_units=True, resources={"threads": 1})
    settings.update(options)
    return Mandacaru(**settings)


def test_empty_training_saves_a_valid_unready_model_and_native_gradient_chain(tmp_path, monkeypatch):
    import scipy.optimize

    def forbidden(*args, **kwargs):
        pytest.fail("proposal learning and variational optimization must execute in Rust")

    monkeypatch.setattr(scipy.optimize, "minimize", forbidden)
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    store = tmp_path / "empty"
    model = fit(store, fractions=(1.,), resources={"threads": 1})
    assert not model.ready
    assert "0 groups" in model.report["reason"]
    restored = ProposalModel.load(store / "proposal_model.npz")
    assert restored.native_json == model.native_json
    with calculator(proposal_model=restored) as calc:
        result = calc.run()
        assert result.energy_hartree == pytest.approx(-.5, abs=1e-9)
        assert not result.proposal_ready
        assert result.proposal_reason == model.report["reason"]
        assert not result.converged and result.completed
        assert np.allclose(calc.checkpoint.state(), result.state(), atol=1e-12)


def test_frozen_ready_model_actually_changes_the_native_proposal_probabilities(tmp_path):
    # A deliberately supplied test model can be marked ready; production fit
    # decides readiness only by held-out validation. Compare the actual score
    # mixture in recorded chain transitions, rather than just a model flag.
    model = fit(tmp_path / "empty", fractions=(1.,), resources={"threads": 1})
    value = json.loads(model.native_json)
    value["ready"] = True
    value["mixing"] = .3
    for row in value["weights"].values():
        row["values"] = [0.] * len(row["values"])
    uniform = ProposalModel(json.dumps(value))
    store = tmp_path / "records"
    with calculator(proposal_model=uniform, record=store, max_steps=1,
                    pool="qeb") as calc:
        result = calc.run()
        assert result.proposal_ready
        pool = len(calc.problem.pool_labels)
    records = [json.loads(line) for source in (store / "chains").glob("*.jsonl")
               for line in source.read_text().splitlines()]
    first = records[0]
    gradients = np.abs(first["source_gradients"])
    gradient = np.exp((gradients / gradients.max() - 1.) / .2)
    gradient /= gradient.sum()
    assert np.allclose(first["source_probabilities"], .3 * gradient + .7 / pool,
                       atol=1e-14)


def test_malformed_recorded_training_data_is_rejected(tmp_path):
    store = tmp_path / "broken"
    chains = store / "chains"
    chains.mkdir(parents=True)
    (chains / "bad.jsonl").write_text(json.dumps(dict(format="legacy", version=0)) + "\n")
    with pytest.raises(ValueError, match="observation format"):
        fit(store, fractions=(1.,))
    assert not (store / "proposal_model.npz").exists()
