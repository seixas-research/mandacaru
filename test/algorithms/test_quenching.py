# -*- coding: utf-8 -*-
# file: test/test_quenching.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""``quenching``: joint re-optimization, or only what changed, in VQE and ADAPT-VQE.

``quenching=True`` (default) optimizes every variational parameter at each
iteration -- textbook ADAPT-VQE.  ``quenching=False`` freezes the earlier
parameters and varies only the newly appended ones (ADAPT), or sweeps a fixed
ansatz once, one angle at a time (VQE).  The invariants tested here are
structural (which parameters move) and variational.
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
    def test_it_is_a_boolean(self):
        with pytest.raises(TypeError, match="quenching must be True or False"):
            Mandacaru(method="adapt-vqe", quenching=0)

    @pytest.mark.parametrize("method", ["subspace-vqe", "vasqa", "rhf"])
    def test_a_method_that_always_reoptimizes_refuses_false(self, method):
        with pytest.raises(ValueError, match="quenching"):
            Mandacaru(method=method, quenching=False)


# --------------------------------------------------------------------------- #
# ADAPT-VQE: growth with every parameter re-optimized.
# --------------------------------------------------------------------------- #

class TestADAPTGrowth:
    @staticmethod
    def _growth(lih_cache, quenching=True):
        seen: list[np.ndarray] = []
        driver = Mandacaru(method="adapt-vqe", pool="qeb",
                           load_hamiltonian=lih_cache, trace=False,
                           max_iterations=6, quenching=quenching)
        result = driver.run(callback=lambda info: seen.append(
            np.array(info["parameters"], dtype=float)))
        return seen, result

    def test_each_step_appends_exactly_one_parameter(self, lih_cache):
        seen, _ = self._growth(lih_cache)
        assert len(seen) >= 3
        for earlier, later in zip(seen, seen[1:]):
            assert later.size == earlier.size + 1

    def test_growth_reoptimizes_earlier_parameters(self, lih_cache):
        seen, _ = self._growth(lih_cache)
        assert len(seen) >= 3
        # At least one growth step moved a previously optimized parameter.
        moved = any(not np.array_equal(later[:earlier.size], earlier)
                    for earlier, later in zip(seen, seen[1:]))
        assert moved

    def test_frozen_growth_keeps_every_earlier_parameter(self, lih_cache):
        seen, frozen = self._growth(lih_cache, quenching=False)
        assert len(seen) >= 3
        for earlier, later in zip(seen, seen[1:]):
            assert np.array_equal(later[:earlier.size], earlier)
        # The first step optimizes one parameter either way; past it, the joint
        # optimization of the same first operator can only be lower.
        _, joint = self._growth(lih_cache)
        assert frozen.iterations[0].energy == pytest.approx(joint.iterations[0].energy, abs=1e-9)
        assert frozen.optimal_energy < frozen.reference_energy


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

    def test_one_sweep_is_an_upper_bound_to_the_joint_minimum(self, lih_cache):
        joint = Mandacaru(method="vqe", load_hamiltonian=lih_cache, trace=False).run()
        swept = Mandacaru(method="vqe", load_hamiltonian=lih_cache, trace=False,
                          quenching=False).run()
        assert joint.optimal_energy - 1e-10 <= swept.optimal_energy < swept.reference_energy
        assert swept.num_parameters == joint.num_parameters


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

