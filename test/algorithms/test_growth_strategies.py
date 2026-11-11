# -*- coding: utf-8 -*-
# file: test/algorithms/test_growth_strategies.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""TETRIS growth and ansatz pruning, two literature refinements of ADAPT-VQE.

``tetris=True`` (Anastasiou *et al.*, Phys. Rev. Research **6**, 013254, 2024)
appends, after the largest gradient, every operator whose qubit support is
disjoint from those already taken this step.  ``prune=True`` (J. Chem. Theory
Comput. **21**, 8720, 2025) removes after each step the operator ranked first by
``exp(-10 x) / theta^2`` when its amplitude is under a tenth of the recent mean.
The selection and the pruning rule are pinned in Rust
(``crates/mandacaru-core/tests/solver_growth.rs``); these tests check the
calculator's options, result view and run log.
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.utils import parse_output


@pytest.fixture(scope="module")
def lih_cache(tmp_path_factory):
    """LiH qubit Hamiltonian on disk: 6 qubits, enough operators to matter."""
    path = str(tmp_path_factory.mktemp("ham") / "lih.parquet")
    atoms = Atoms("LiH", positions=[[7.5, 7.5, 6.7], [7.5, 7.5, 8.3]],
                  cell=[[15, 0, 0], [0, 15, 0], [0, 0, 15]], pbc=False)
    atoms.calc = Mandacaru(method="adapt-vqe", pool="qeb", basis="HAO", grid_spacing=0.35,
                           trace=False, max_iterations=1, save_hamiltonian=path)
    atoms.get_total_energy()
    return path


def _run(lih_cache, tmp_path, **options):
    seen = []
    out = str(tmp_path / "output.txt")
    driver = Mandacaru(method="adapt-vqe", pool="qeb", load_hamiltonian=lih_cache,
                       trace=False, txt=out, max_iterations=8, **options)
    result = driver.run(callback=lambda info: seen.append(len(info["parameters"])))
    return result, seen, parse_output(out)


class TestTheOptions:
    @pytest.mark.parametrize("name", ["tetris", "prune"])
    def test_they_are_booleans(self, name):
        with pytest.raises(TypeError, match=f"{name} must be True or False"):
            Mandacaru(method="adapt-vqe", **{name: 1})

    @pytest.mark.parametrize("method", ["vqe", "subspace-adapt-vqe", "vasqa"])
    @pytest.mark.parametrize("name", ["tetris", "prune"])
    def test_only_adapt_vqe_takes_them(self, method, name):
        with pytest.raises(ValueError, match=name):
            Mandacaru(method=method, **{name: True})


class TestTetris:
    def test_a_step_can_append_a_layer_and_the_log_says_so(self, lih_cache, tmp_path):
        result, seen, log = _run(lih_cache, tmp_path, tetris=True)
        growth = np.diff([0] + seen)
        assert (growth >= 1).all()
        assert (growth > 1).any(), "some step appended several disjoint operators"
        assert log["setup"]["growth"].startswith("tetris")
        assert result.optimal_energy < result.reference_energy


class TestPruning:
    def test_removals_are_reported_where_the_operators_went(self, lih_cache, tmp_path):
        loose = {"method": "L-BFGS", "maxiter": 2, "tol": 1e-5}
        result, seen, log = _run(lih_cache, tmp_path, prune=True, optimizer=loose)
        pruned = result.pruned_operators
        assert all(isinstance(step, int) and isinstance(label, str) for step, label in pruned)
        # The qeb pool appends one operator per step: kept plus removed is the step count.
        assert result.num_operators + len(pruned) == len(seen)
        assert log["setup"]["pruning"] == "on"
        assert int(log["summary"]["pruned_operators"]) == len(pruned)

    def test_off_by_default_with_nothing_removed(self, lih_cache, tmp_path):
        result, _seen, log = _run(lih_cache, tmp_path)
        assert result.pruned_operators == []
        assert log["setup"]["pruning"] == "off"
        assert "pruned_operators" not in log["summary"]
