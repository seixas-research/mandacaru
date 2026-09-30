# -*- coding: utf-8 -*-
# file: test_valqa.py

"""VALQA: VASQA's chain with a learned operator proposal.

The model starts where VASQA is -- without one, or with one that has not
passed its readiness check, VALQA draws what VASQA's gradient proposal draws,
step for step -- and a ready model changes the draws without breaking the
Metropolis-Hastings ratio.  Reached only through ``Mandacaru(method="valqa")``.
"""

import numpy as np
import pytest

from mandacaru import Mandacaru
from mandacaru.algorithms import VALQAResult
from mandacaru.algorithms.proposal_model import (DESCRIPTORS, PROJECTION,
                                                 ProposalModel,
                                                 init_parameters)
from mandacaru.core import MolecularIntegrals, minimal_hao_basis
from mandacaru.integrals import Grid
from mandacaru.units import HARTREE_TO_EV


@pytest.fixture(scope="module")
def h2_hamiltonian():
    R = 0.74
    nuclei = [(1.0, np.array([0.0, 0.0, -R / 2])),
              (1.0, np.array([0.0, 0.0, +R / 2]))]
    grid = Grid(center=[0.0, 0.0, 0.0], box_size=5.0, h=0.25)
    mints = MolecularIntegrals(nuclei, minimal_hao_basis(nuclei), grid)
    return mints.molecular_hamiltonian(mo_basis=True, n_electrons=2)


@pytest.fixture(scope="module")
def sector_ground_ev(h2_hamiltonian):
    m = h2_hamiltonian.map_to_qubits("jordan_wigner").to_matrix()
    m = 0.5 * (m + m.conj().T)
    two = [i for i in range(16) if bin(i).count("1") == 2]
    return float(np.linalg.eigvalsh(m[np.ix_(two, two)])[0]) * HARTREE_TO_EV


def _model(path, ready=True, reason="synthetic"):
    """A model assembled without training (NumPy only), saved to ``path``."""
    rng = np.random.default_rng(5)
    phi = rng.normal(size=(6, PROJECTION))
    k = np.exp(-0.5 * ((phi[:, None] - phi[None]) ** 2).sum(-1)) \
        + 0.01 * np.eye(6)
    ProposalModel(params=init_parameters(4), use_graph=True,
                  descriptor_mean=np.zeros(len(DESCRIPTORS)),
                  descriptor_std=np.ones(len(DESCRIPTORS)), target_mean=0.0,
                  target_std=1.0, train_phi=phi,
                  alpha=np.linalg.solve(k, rng.normal(size=6)),
                  cholesky=np.linalg.cholesky(k), ready=ready,
                  report={"reason": reason}, version="synthetic",
                  temperature=0.3).save(path)
    return str(path)


def _chain(h2_hamiltonian, method="valqa", **options):
    options = {"pool": "qubit", "max_steps": 25, "max_length": 5, "seed": 7,
               "profile": False, "trace": False, **options}
    return Mandacaru(method=method, hamiltonian=h2_hamiltonian,
                     num_particles=(1, 1), n_spatial_orbitals=2, **options)


def _trajectory(result):
    return [(s.action, s.accepted, s.log_q_forward, s.log_q_reverse)
            for s in result.steps]


class TestTheModelStartsWhereVASQAIs:
    def test_without_a_model_the_chain_is_vasqas(self, h2_hamiltonian):
        vasqa = _chain(h2_hamiltonian, method="vasqa").run()
        valqa = _chain(h2_hamiltonian).run()
        assert isinstance(valqa, VALQAResult)
        assert _trajectory(valqa) == _trajectory(vasqa)
        assert valqa.optimal_energy == vasqa.optimal_energy
        assert valqa.proposal_model is None and not valqa.model_ready

    def test_a_model_that_is_not_ready_changes_nothing(self, h2_hamiltonian,
                                                       tmp_path):
        path = _model(tmp_path / "m.npz", ready=False)
        vasqa = _chain(h2_hamiltonian, method="vasqa").run()
        valqa = _chain(h2_hamiltonian, proposal_model=path).run()
        assert _trajectory(valqa) == _trajectory(vasqa)
        assert valqa.proposal_model == "synthetic" and not valqa.model_ready

    def test_the_setup_block_says_why(self, h2_hamiltonian, tmp_path):
        from mandacaru.utils.logging import parse_output, reset_log
        path = tmp_path / "output.txt"
        reset_log(str(path))
        model = _model(tmp_path / "m.npz", ready=False,
                       reason="too few groups")
        _chain(h2_hamiltonian, max_steps=3, txt=str(path),
               proposal_model=model).run()
        setup = parse_output(str(path))["setup"]
        assert "model synthetic not ready: too few groups" in \
            setup["proposal"]
        assert setup["proposal_model"] == model


class TestAReadyModel:
    def test_it_changes_the_draws_and_keeps_every_ratio_finite(
            self, h2_hamiltonian, tmp_path):
        path = _model(tmp_path / "m.npz")
        vasqa = _chain(h2_hamiltonian, method="vasqa").run()
        valqa = _chain(h2_hamiltonian, proposal_model=path).run()
        assert valqa.model_ready and valqa.proposal_model == "synthetic"
        assert _trajectory(valqa) != _trajectory(vasqa)
        # Every operator keeps a positive probability, so an insertion
        # always has its reverse deletion and a replacement its reverse.
        for step in valqa.steps:
            assert np.isfinite(step.log_q_forward)
            if step.move in ("insert", "replace"):
                assert np.isfinite(step.log_q_reverse)

    def test_it_reaches_the_sector_ground_state(self, h2_hamiltonian,
                                                sector_ground_ev, tmp_path):
        result = _chain(h2_hamiltonian, max_steps=40,
                        proposal_model=_model(tmp_path / "m.npz")).run()
        assert result.optimal_energy == pytest.approx(sector_ground_ev,
                                                      abs=1e-6)

    def test_its_version_is_recorded_with_each_edit(self, h2_hamiltonian,
                                                    tmp_path):
        from mandacaru.algorithms.proposal_data import load_edits
        _chain(h2_hamiltonian, max_steps=5, record=tmp_path / "edits",
               proposal_model=_model(tmp_path / "m.npz")).run()
        rows, _ = load_edits(tmp_path / "edits")
        assert {r["model"] for r in rows} == {"synthetic"}
        assert {r["method"] for r in rows} == {"VALQA"}

    def test_the_same_seed_gives_the_same_chain(self, h2_hamiltonian,
                                                tmp_path):
        path = _model(tmp_path / "m.npz")
        a = _chain(h2_hamiltonian, proposal_model=path).run()
        b = _chain(h2_hamiltonian, proposal_model=path).run()
        assert _trajectory(a) == _trajectory(b)


class TestOptions:
    def test_a_missing_model_is_refused_at_construction(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no proposal model"):
            Mandacaru(method="valqa", proposal_model=str(tmp_path / "x.npz"))

    def test_valqa_does_not_take_vasqas_proposal(self):
        with pytest.raises(TypeError, match="does not take"):
            Mandacaru(method="valqa", proposal="uniform")

    def test_vasqa_does_not_take_a_model(self, tmp_path):
        with pytest.raises(TypeError, match="does not take"):
            Mandacaru(method="vasqa",
                      proposal_model=_model(tmp_path / "m.npz"))


class TestTheSharedStore:
    def test_the_stores_model_is_used_by_default(self, h2_hamiltonian,
                                                 tmp_path, monkeypatch):
        from mandacaru.algorithms.proposal_data import MODEL_FILE
        store = tmp_path / "store"
        store.mkdir()
        _model(store / MODEL_FILE)
        monkeypatch.setenv("MANDACARU_PROPOSAL_DATA", str(store))
        result = _chain(h2_hamiltonian, max_steps=5).run()
        assert result.proposal_model == "synthetic" and result.model_ready

    def test_false_uses_no_model(self, h2_hamiltonian, tmp_path,
                                 monkeypatch):
        from mandacaru.algorithms.proposal_data import MODEL_FILE
        store = tmp_path / "store"
        store.mkdir()
        _model(store / MODEL_FILE)
        monkeypatch.setenv("MANDACARU_PROPOSAL_DATA", str(store))
        vasqa = _chain(h2_hamiltonian, method="vasqa", record=False).run()
        valqa = _chain(h2_hamiltonian, proposal_model=False,
                       record=False).run()
        assert valqa.proposal_model is None
        assert _trajectory(valqa) == _trajectory(vasqa)
