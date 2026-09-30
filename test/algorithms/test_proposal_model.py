# -*- coding: utf-8 -*-
# file: test_proposal_model.py

"""VALQA's learned proposal: the Hamiltonian factor graph, the network, the
Gaussian-process head, the mixture with the gradient softmax and the
readiness check.

Nothing here trains a real model: the fits are a few Adam steps on a handful
of rows, enough to exercise the code paths.
"""

import math

import numpy as np
import pytest

from mandacaru.algorithms.mcas import gradient_softmax
from mandacaru.algorithms.proposal_data import ProblemRecord
from mandacaru.algorithms.proposal_model import (
    DESCRIPTORS, PROJECTION, HamiltonianGraph,
    ProposalModel, assess_data_volume, candidate_rows, compress, embed,
    features, init_parameters, train_proposal_model)
from mandacaru.circuits.pools import PoolOperator
from mandacaru.core.mapping import PauliSum


def _problem(order=None):
    """A 3-qubit toy: ``H = 0.5 + 0.3 Z0 - 0.2 X0 Z1 X2 + 0.1 Y1 Y2``."""
    terms = {"III": 0.5, "ZII": 0.3, "XZX": -0.2, "IYY": 0.1}
    labels = list(terms) if order is None else [list(terms)[i] for i in order]
    hamiltonian = PauliSum({label: terms[label] for label in labels})
    pool = [PoolOperator("a", PauliSum({"XYI": 0.5j, "YXI": -0.5j}),
                         (0, 1), "pauli"),
            PoolOperator("b", PauliSum({"IXY": 1j}), (1, 2), "pauli"),
            PoolOperator("c", PauliSum({"YIZ": 0.25j}), (0, 2), "pauli")]
    return ProblemRecord.from_problem(hamiltonian, [0], pool)


def _synthetic_model(ready=True, use_graph=True, seed=3):
    """A model assembled without training: random weights, five fake
    training points (NumPy only)."""
    rng = np.random.default_rng(seed)
    params = init_parameters(seed, use_graph)
    phi = rng.normal(size=(5, PROJECTION))
    k = np.exp(-0.5 * ((phi[:, None] - phi[None]) ** 2).sum(-1)) \
        + 0.01 * np.eye(5)
    chol = np.linalg.cholesky(k)
    alpha = np.linalg.solve(k, rng.normal(size=5))
    return ProposalModel(params=params, use_graph=use_graph,
                         descriptor_mean=np.zeros(len(DESCRIPTORS)),
                         descriptor_std=np.ones(len(DESCRIPTORS)),
                         target_mean=0.0, target_std=1.0, train_phi=phi,
                         alpha=alpha, cholesky=chol, ready=ready,
                         version="synthetic")


class TestTheFactorGraph:
    def test_one_factor_per_non_identity_term_with_labeled_edges(self):
        graph = HamiltonianGraph.from_problem(_problem())
        assert graph.term_features.shape == (3, 7)
        assert graph.incidence.shape == (3, 3, 3)
        # The X0 Z1 X2 term is one factor with three labeled edges.
        problem = _problem()
        row = [i for i, t in enumerate(problem.terms)
               if list(t) == [1, 3, 1]][0]
        assert graph.incidence[0, row].tolist() == [1, 0, 1]      # X
        assert graph.incidence[2, row].tolist() == [0, 1, 0]      # Z
        assert graph.incidence[1, row].sum() == 0                 # no Y

    def test_the_scale_is_the_largest_coefficient(self):
        graph = HamiltonianGraph.from_problem(_problem())
        assert graph.scale == pytest.approx(0.3)
        assert np.abs(graph.term_features[:, 0]).max() == pytest.approx(1.0)
        # The Z0 coefficient is qubit 0's orbital-energy proxy.
        assert graph.qubit_features[:, 2].tolist() == pytest.approx(
            [1.0, 0.0, 0.0])
        assert graph.qubit_features[:, 0].tolist() == [1, 0, 0]

    def test_a_sum_generator_keeps_every_string(self):
        """``a`` is X0Y1 - Y0X1: both letters on both qubits, half each."""
        graph = HamiltonianGraph.from_problem(_problem())
        np.testing.assert_allclose(graph.pool_incidence[:, 0, 0],
                                   [0.25, 0.25, 0.0])
        np.testing.assert_allclose(graph.pool_incidence[:, 0].sum(), 1.0)
        np.testing.assert_allclose(graph.pool_descriptors[0],
                                   [2 / 3, 0.5, 0.5, 0.0])

    def test_the_embedding_does_not_depend_on_the_term_order(self):
        params = init_parameters(1)
        a = embed(np, params, HamiltonianGraph.from_problem(
            _problem()).arrays())
        b = embed(np, params, HamiltonianGraph.from_problem(
            _problem(order=[3, 1, 0, 2])).arrays())
        np.testing.assert_allclose(a[0], b[0], atol=1e-12)
        np.testing.assert_allclose(a[1], b[1], atol=1e-12)


class TestCandidates:
    def test_only_what_is_known_before_the_evaluation(self):
        graph = HamiltonianGraph.from_problem(_problem())
        rows = candidate_rows(graph, [(1, 0, 1)], [1], [[0.1, -0.4, 0.0]],
                              [-0.002])
        d = dict(zip(DESCRIPTORS, rows.descriptors[0]))
        assert d["relative_gradient"] == pytest.approx(1.0)
        assert d["gradient"] == pytest.approx(float(compress(0.4)))
        assert d["source_energy"] == pytest.approx(float(compress(-0.002)))
        assert d["multiplicity"] == 2 and d["length"] == 3
        np.testing.assert_allclose(rows.occupancy[0], [1 / 3, 2 / 3, 0])
        # Later occurrences weigh more: positions 1, 2, 3 of 6.
        np.testing.assert_allclose(rows.positional[0], [2 / 6, 4 / 6, 0])

    def test_the_compression_is_odd_and_monotonic(self):
        x = np.array([-1e-2, -1e-5, 0.0, 1e-5, 1e-2])
        t = compress(x)
        np.testing.assert_allclose(t, -t[::-1])
        assert np.all(np.diff(t) > 0) and t[3] == pytest.approx(math.log(2))


class TestTheMixture:
    def test_an_unready_model_returns_the_gradient_proposal_itself(self):
        bound = _synthetic_model(ready=False).bind(_problem())
        baseline = gradient_softmax([0.3, 0.1, 0.0], 0.2)
        assert bound.probabilities((), [0.3, 0.1, 0.0], 0.0,
                                   baseline) is baseline

    def test_every_operator_keeps_a_positive_probability(self):
        """A learned distribution that underflows to zero is rescued by the
        mixture, so no move loses its reverse."""
        model = _synthetic_model()
        model.temperature = 1e-6
        bound = model.bind(_problem())
        learned = bound.learned((0,), [0.3, 0.1, 0.0], -1e-3)
        assert (learned == 0.0).any()
        baseline = gradient_softmax([0.3, 0.1, 0.0], 0.2)
        mixed = bound.probabilities((0,), [0.3, 0.1, 0.0], -1e-3, baseline)
        assert np.all(mixed > 0.0)
        assert mixed.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.all(mixed >= model.mixing * baseline - 1e-15)

    def test_the_score_rewards_predicted_improvement(self):
        model = _synthetic_model()
        bound = model.bind(_problem())
        graph = bound.graph
        rows = candidate_rows(graph, [()] * 3, range(3), [[0.3, 0.1, 0.0]] * 3,
                              [0.0] * 3)
        mean, std = model.predict(graph.arrays(), rows)
        expected = np.exp(-mean + model.kappa * std)
        np.testing.assert_allclose(bound.learned((), [0.3, 0.1, 0.0], 0.0),
                                   expected / expected.sum())

    def test_a_model_round_trips_through_its_file(self, tmp_path):
        model = _synthetic_model()
        model.report = {"ready": True, "reason": "synthetic"}
        model.save(tmp_path / "model.npz")
        again = ProposalModel.load(tmp_path / "model.npz")
        assert again.version == "synthetic" and again.ready
        assert again.report["reason"] == "synthetic"
        graph = HamiltonianGraph.from_problem(_problem())
        rows = candidate_rows(graph, [(2,)] * 3, range(3),
                              [[0.3, 0.1, 0.0]] * 3, [-1e-3] * 3)
        for a, b in zip(model.predict(graph.arrays(), rows),
                        again.predict(graph.arrays(), rows)):
            np.testing.assert_allclose(a, b)

    def test_a_missing_model_file_is_named(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no proposal model"):
            ProposalModel.load(tmp_path / "none.npz")


class TestTraining:
    def test_numpy_and_jax_compute_the_same_features(self):
        """The network is written once; inference must not drift from what
        was trained."""
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        params = init_parameters(2)
        graph = HamiltonianGraph.from_problem(_problem())
        rows = candidate_rows(graph, [(0, 2)] * 3, range(3),
                              [[0.3, 0.1, 0.0]] * 3, [-1e-3] * 3)
        a = features(np, params, graph.arrays(), rows, rows.descriptors, True)
        b = features(jnp, {k: jnp.asarray(v) for k, v in params.items()},
                     {k: jnp.asarray(v) for k, v in graph.arrays().items()},
                     rows, jnp.asarray(rows.descriptors), True)
        np.testing.assert_allclose(a, np.asarray(b), atol=1e-12)

    def test_fewer_than_three_groups_are_never_ready(self, tmp_path):
        _record_toy(tmp_path, formulas=("H2", "H2", "LiH"))
        report = assess_data_volume(tmp_path)
        assert report["ready"] is False
        assert "2 molecule group(s)" in report["reason"]

    def test_a_trained_model_is_saved_with_its_report(self, tmp_path):
        pytest.importorskip("jax")
        _record_toy(tmp_path, formulas=("H2", "LiH", "BeH2"))
        model = train_proposal_model(tmp_path, tmp_path / "m.npz",
                                     fractions=(1.0,), steps=5)
        again = ProposalModel.load(tmp_path / "m.npz")
        assert again.version == model.version and len(model.version) == 12
        assert again.report["groups"] == 3
        assert again.ready == again.report["ready"]
        assert set(again.report["curve"][0]) >= {"graph", "descriptors",
                                                 "gradient", "train"}

    def test_mixing_zero_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="reverse move"):
            train_proposal_model(tmp_path, tmp_path / "m.npz", mixing=0.0)


def _record_toy(directory, formulas):
    """Write a synthetic edit store on the toy problem: insert rows whose
    energy change follows the gradient, one group per formula."""
    from mandacaru.algorithms.proposal_data import EditRecorder
    rng = np.random.default_rng(0)
    problem = _problem()
    for formula in formulas:
        recorder = EditRecorder(directory, problem, run={
            "run": formula, "method": "VASQA", "formula": formula})
        for step in range(8):
            g = rng.normal(size=3)
            mu = int(rng.integers(3))
            recorder.write({
                "step": step, "move": "insert", "position": 0,
                "operator": mu, "source": [int(rng.integers(3))],
                "source_energy": -1.0, "reference_energy": -1.0,
                "delta_energy": -abs(g[mu]) * 1e-3 + rng.normal() * 1e-5,
                "source_gradients": g.tolist()})
        recorder.close()


def test_training_defaults_to_the_shared_store(tmp_path, monkeypatch):
    pytest.importorskip("jax")
    from mandacaru.algorithms.proposal_data import MODEL_FILE
    monkeypatch.setenv("MANDACARU_PROPOSAL_DATA", str(tmp_path))
    _record_toy(tmp_path, formulas=("H2", "LiH", "BeH2"))
    model = train_proposal_model(fractions=(1.0,), steps=3)
    assert ProposalModel.load(tmp_path / MODEL_FILE).version == model.version
