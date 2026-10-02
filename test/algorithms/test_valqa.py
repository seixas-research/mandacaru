# -*- coding: utf-8 -*-
# file: test_valqa.py

"""VALQA: MCAS-VQE's chain with a learned operator proposal.

The model starts where MCAS-VQE is -- without one, or with one that has not
passed its readiness check, VALQA draws what MCAS-VQE's gradient proposal draws,
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
                  report={"reason": reason}, version="synthetic", kind="gp",
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


class TestTheModelStartsWhereMCASVQEIs:
    def test_without_a_model_the_chain_is_mcas_vqes(self, h2_hamiltonian):
        mcas_vqe = _chain(h2_hamiltonian, method="mcas-vqe").run()
        valqa = _chain(h2_hamiltonian).run()
        assert isinstance(valqa, VALQAResult)
        assert _trajectory(valqa) == _trajectory(mcas_vqe)
        assert valqa.optimal_energy == mcas_vqe.optimal_energy
        assert valqa.proposal_model is None and not valqa.model_ready

    def test_a_model_that_is_not_ready_changes_nothing(self, h2_hamiltonian,
                                                       tmp_path):
        path = _model(tmp_path / "m.npz", ready=False)
        mcas_vqe = _chain(h2_hamiltonian, method="mcas-vqe").run()
        valqa = _chain(h2_hamiltonian, proposal_model=path).run()
        assert _trajectory(valqa) == _trajectory(mcas_vqe)
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
        mcas_vqe = _chain(h2_hamiltonian, method="mcas-vqe").run()
        valqa = _chain(h2_hamiltonian, proposal_model=path).run()
        assert valqa.model_ready and valqa.proposal_model == "synthetic"
        assert _trajectory(valqa) != _trajectory(mcas_vqe)
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


class TestTheModelIsFrozenForTheRun:
    def test_the_file_is_read_when_the_solver_is_built(self, h2_hamiltonian,
                                                       tmp_path):
        """Retraining into the same file while a run is set up does not
        change the model that run draws from."""
        path = tmp_path / "m.npz"
        _model(path)
        calc = _chain(h2_hamiltonian, max_steps=5, proposal_model=str(path),
                      record=tmp_path / "edits")
        calc.solver                          # the model is loaded here
        path.unlink()
        _model(path, ready=False, reason="replaced")
        result = calc.run()
        assert result.model_ready and result.proposal_model == "synthetic"
        from mandacaru.algorithms.proposal_data import load_edits
        rows, _ = load_edits(tmp_path / "edits")
        assert {r["model"] for r in rows} == {"synthetic"}

    def test_a_model_of_another_schema_fails_at_construction(self,
                                                             tmp_path):
        import json
        path = tmp_path / "m.npz"
        _model(path)
        with np.load(path) as data:
            arrays = {name: data[name] for name in data.files}
        meta = json.loads(str(arrays["meta"]))
        meta["schema"] += 1
        arrays["meta"] = np.array(json.dumps(meta))
        np.savez_compressed(path, **arrays)
        with pytest.raises(ValueError, match="model schema"):
            Mandacaru(method="valqa", proposal_model=str(path))


class TestAGraphNeuralNetworkModel:
    def test_the_chain_draws_from_the_gnn(self, h2_hamiltonian, tmp_path):
        from mandacaru.algorithms.proposal_model import GNNModel, init_gnn
        path = tmp_path / "n.npz"
        model = ProposalModel.load(_model(path))
        model.kind = "graph"
        model.gnn = GNNModel(params=init_gnn(2),
                             descriptor_mean=np.zeros(len(DESCRIPTORS)),
                             descriptor_std=np.ones(len(DESCRIPTORS)))
        model.save(path)
        valqa = _chain(h2_hamiltonian, proposal_model=str(path)).run()
        vasqa = _chain(h2_hamiltonian, method="mcas-vqe").run()
        assert valqa.model_ready and valqa.proposal_model == "synthetic"
        assert _trajectory(valqa) != _trajectory(vasqa)
        for step in valqa.steps:
            assert np.isfinite(step.log_q_forward)
            if step.move in ("insert", "replace"):
                assert np.isfinite(step.log_q_reverse)


class TestOptions:
    def test_a_missing_model_is_refused_at_construction(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no proposal model"):
            Mandacaru(method="valqa", proposal_model=str(tmp_path / "x.npz"))

    def test_valqa_does_not_take_mcas_vqes_proposal(self):
        with pytest.raises(TypeError, match="does not take"):
            Mandacaru(method="valqa", proposal="uniform")

    def test_mcas_vqe_does_not_take_a_model(self, tmp_path):
        with pytest.raises(TypeError, match="does not take"):
            Mandacaru(method="mcas-vqe",
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
        mcas_vqe = _chain(h2_hamiltonian, method="mcas-vqe", record=False).run()
        valqa = _chain(h2_hamiltonian, proposal_model=False,
                       record=False).run()
        assert valqa.proposal_model is None
        assert _trajectory(valqa) == _trajectory(mcas_vqe)


def _h2_at(distance):
    from ase import Atoms
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, distance]])
    atoms.center(vacuum=2.5)
    return atoms


class TestAlongATrajectory:
    """Two geometries through one ASE calculator: the second chain is
    conditioned on the first one's Hamiltonian, and with
    ``update_between_geometries`` on its insertions too."""

    def _run(self, tmp_path, distances=(0.74, 0.78), **options):
        options = {"basis": "HAO", "h": 0.4, "pool": "qeb", "max_steps": 12,
                   "max_length": 4, "seed": 3, "profile": False,
                   "trace": False, "record": str(tmp_path / "edits"),
                   "proposal_model": _model(tmp_path / "m.npz"), **options}
        calc = Mandacaru(method="valqa", **options)
        results = []
        for distance in distances:
            atoms = _h2_at(distance)
            atoms.calc = calc
            atoms.get_potential_energy()
            results.append(calc.result)
        return results

    def test_rows_carry_the_trajectory_and_the_previous_hamiltonian(
            self, tmp_path):
        from mandacaru.algorithms.proposal_data import load_edits
        self._run(tmp_path)
        rows, problems = load_edits(tmp_path / "edits")
        steps = {r["geometry_step"] for r in rows}
        assert steps == {0, 1}
        assert len({r["trajectory"] for r in rows}) == 1
        first = {r["problem"] for r in rows if r["geometry_step"] == 0}
        second = [r for r in rows if r["geometry_step"] == 1]
        assert {r["previous_problem"] for r in second} == first
        assert all(r["previous_problem"] is None for r in rows
                   if r["geometry_step"] == 0)
        assert first <= set(problems)

    def test_without_the_update_the_model_is_the_same_at_every_geometry(
            self, tmp_path):
        first, second = self._run(tmp_path)
        assert first.proposal_model == second.proposal_model == "synthetic"
        assert second.model_update is None

    def test_the_update_scores_then_conditions(self, tmp_path):
        first, second = self._run(tmp_path, update_between_geometries=True)
        assert first.proposal_model == "synthetic"
        assert first.model_update.startswith("none (no previous geometry")
        assert "scored before the update" in second.model_update
        assert second.proposal_model != "synthetic"
        assert second.model_ready

    def test_another_molecule_starts_a_new_trajectory(self, tmp_path):
        from ase import Atoms
        from mandacaru.algorithms.proposal_data import load_edits
        calc = Mandacaru(method="valqa", basis="HAO", h=0.4, pool="qeb",
                         max_steps=4, max_length=3, seed=3, profile=False,
                         trace=False, record=str(tmp_path / "edits"),
                         update_between_geometries=True,
                         proposal_model=_model(tmp_path / "m.npz"))
        for atoms in (_h2_at(0.74), Atoms("H3", positions=[
                [0, 0, 0], [0, 0, 0.9], [0, 0, 1.8]])):
            atoms.center(vacuum=2.5)
            atoms.calc = calc
            atoms.get_potential_energy()
        rows, _ = load_edits(tmp_path / "edits")
        assert len({r["trajectory"] for r in rows}) == 2
        assert {r["geometry_step"] for r in rows} == {0}
        assert calc.result.model_update.startswith("none")

    def test_without_a_ready_model_the_update_says_why(self, tmp_path):
        _, second = self._run(tmp_path, update_between_geometries=True,
                              proposal_model=_model(tmp_path / "u.npz",
                                                    ready=False))
        assert second.model_update.startswith("none (model synthetic is "
                                              "not ready)")
        _, second = self._run(tmp_path, update_between_geometries=True,
                              proposal_model=False)
        assert second.model_update == "none (no proposal model)"

    def test_the_online_ranker_takes_the_offline_ones_place(self, tmp_path):
        """The update replaces the offline predictor, not the gradient: the
        ranker is updated with the previous geometry's pairs, and the
        gradient stays in the mixture."""
        path = tmp_path / "r.npz"
        model = ProposalModel.load(_model(path))
        model.kind = "ranker"
        from mandacaru.algorithms.proposal_model import PairwiseRanker
        model.ranker = PairwiseRanker(weights=np.ones(len(DESCRIPTORS)),
                                      mean=np.zeros(len(DESCRIPTORS)),
                                      std=np.ones(len(DESCRIPTORS)))
        model.save(path)
        first, second = self._run(tmp_path, update_between_geometries=True,
                                  proposal_model=str(path),
                                  screen_insertions=2)
        assert "scored before the update" in second.model_update
        assert "within-state Spearman" in second.model_update
        assert second.proposal_model != first.proposal_model
        assert second.model_ready

    def test_the_option_is_validated_at_construction(self):
        with pytest.raises(ValueError, match="update_between_geometries"):
            Mandacaru(method="valqa", update_between_geometries="yes")
