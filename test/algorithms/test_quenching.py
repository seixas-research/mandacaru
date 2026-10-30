# -*- coding: utf-8 -*-
# file: test/test_quenching.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Joint re-optimization across VQE and ADAPT-VQE.

The native engine optimizes every variational parameter at each iteration --
textbook ADAPT-VQE.  The former ``quenching=False`` policy (freeze the earlier
parameters, vary only the newest; sweep a fixed ansatz one angle at a time) has
no native counterpart, so ``Mandacaru`` rejects the keyword.  The invariants
tested here are structural (which parameters move) and variational.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru


@pytest.fixture(scope="module")
def lih_cache(tmp_path_factory):
    """LiH qubit Hamiltonian on disk: 6 qubits, enough operators to matter."""
    path = str(tmp_path_factory.mktemp("ham") / "lih.parquet")
    atoms = Atoms("LiH", positions=[[7.5, 7.5, 6.7], [7.5, 7.5, 8.3]],
                  cell=[[15, 0, 0], [0, 15, 0], [0, 0, 15]], pbc=False)
    atoms.calc = Mandacaru(method="adapt-vqe", pool="qeb", basis="HAO", grid_spacing=0.35,
                           trace=False, max_iterations=1,
                           save_hamiltonian=path)
    atoms.get_total_energy()
    return path


# --------------------------------------------------------------------------- #
# The removed flag.
# --------------------------------------------------------------------------- #

class TestQuenchingArgument:
    @pytest.mark.parametrize("method", ["vqe", "adapt-vqe"])
    @pytest.mark.parametrize("value", [True, False])
    def test_the_keyword_is_rejected(self, method, value):
        """No policy selector: an unsupported option raises, never selects."""
        with pytest.raises(TypeError, match="quenching"):
            Mandacaru(method=method, quenching=value)


# --------------------------------------------------------------------------- #
# ADAPT-VQE: growth with every parameter re-optimized.
# --------------------------------------------------------------------------- #

class TestADAPTGrowth:
    @staticmethod
    def _growth(lih_cache):
        seen: list[np.ndarray] = []
        driver = Mandacaru(method="adapt-vqe", pool="qeb",
                           load_hamiltonian=lih_cache, trace=False,
                           max_iterations=6)
        driver.run(callback=lambda info: seen.append(
            np.array(info["parameters"], dtype=float)))
        return seen

    def test_each_step_appends_exactly_one_parameter(self, lih_cache):
        seen = self._growth(lih_cache)
        assert len(seen) >= 3
        for earlier, later in zip(seen, seen[1:]):
            assert later.size == earlier.size + 1

    def test_growth_reoptimizes_earlier_parameters(self, lih_cache):
        seen = self._growth(lih_cache)
        assert len(seen) >= 3
        # At least one growth step moved a previously optimized parameter.
        moved = any(not np.array_equal(later[:earlier.size], earlier)
                    for earlier, later in zip(seen, seen[1:]))
        assert moved


# --------------------------------------------------------------------------- #
# VQE: joint minimization of the fixed ansatz.
# --------------------------------------------------------------------------- #

class TestVQEJointOptimization:
    def test_joint_optimization_lowers_the_energy(self, lih_cache):
        vqe = Mandacaru(method="vqe", load_hamiltonian=lih_cache, trace=False)
        result = vqe.run()
        assert result.optimal_energy < result.reference_energy
        assert result.num_parameters == vqe.solver.ansatz.num_parameters
        assert result.num_evaluations > 0


def test_one_hva_layer_is_refused_loudly_because_it_cannot_beat_hartree_fock():
    """One layer builds an imaginary relative amplitude, so the energy has no
    first-order term and the result is exactly Hartree-Fock (H2 and LiH, any
    seed angle). Two layers recover the correlation of H2."""
    with pytest.warns(RuntimeWarning, match="cannot lower the Hartree-Fock energy"):
        Mandacaru(method="vqe", ansatz={"name": "hva", "layers": 1}, trace=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Mandacaru(method="vqe", ansatz={"name": "hva", "layers": 2}, trace=False)
        Mandacaru(method="vqe", ansatz="hva", trace=False)    # default is two layers

