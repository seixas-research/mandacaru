# -*- coding: utf-8 -*-
# file: test/test_quenching.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Dynamic parametrization (``quenching``) across VQE, ADAPT-VQE and VASQE.

``quenching=True`` (the default) hands every variational parameter to the
classical optimizer at each iteration -- textbook ADAPT-VQE.  ``quenching=False``
freezes the previously optimized parameters and varies only the newest one, so
the ansatz is *quenched* into place one angle at a time.  The invariants tested
here are structural (which parameters move) and variational (the quenched energy
can never beat the fully re-optimized one).
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from carcara.algorithms import ADAPTVQE, VASQE, VQE


@pytest.fixture(scope="module")
def lih_cache(tmp_path_factory):
    """LiH qubit Hamiltonian on disk: 6 qubits, enough operators to matter."""
    path = str(tmp_path_factory.mktemp("ham") / "lih.parquet")
    atoms = Atoms("LiH", positions=[[7.5, 7.5, 6.7], [7.5, 7.5, 8.3]],
                  cell=[[15, 0, 0], [0, 15, 0], [0, 0, 15]], pbc=True)
    atoms.calc = ADAPTVQE(pool="qeb", basis="FAO", h=0.35, verbose=False,
                          max_iterations=1, save_hamiltonian=path)
    atoms.get_total_energy()
    return path


# --------------------------------------------------------------------------- #
# The flag itself.
# --------------------------------------------------------------------------- #

class TestQuenchingArgument:
    def test_defaults_to_true_on_every_driver(self):
        assert ADAPTVQE().quenching is True
        assert VQE().quenching is True
        assert VASQE().quenching is True

    @pytest.mark.parametrize("driver", [ADAPTVQE, VQE, VASQE])
    def test_flag_is_stored(self, driver):
        assert driver(quenching=False).quenching is False


# --------------------------------------------------------------------------- #
# ADAPT-VQE: growth with frozen vs re-optimized parameters.
# --------------------------------------------------------------------------- #

class TestADAPTQuenching:
    def test_quenched_run_freezes_earlier_parameters(self, lih_cache):
        """With ``quenching=False`` a parameter never changes after its own step."""
        seen: list[np.ndarray] = []

        driver = ADAPTVQE(pool="qeb", load_hamiltonian=lih_cache, verbose=False,
                          max_iterations=6, quenching=False)
        driver.run(callback=lambda info: seen.append(
            np.array(info["parameters"], dtype=float)))

        assert len(seen) >= 3
        for earlier, later in zip(seen, seen[1:]):
            # The new step appended exactly one parameter ...
            assert later.size == earlier.size + 1
            # ... and left every earlier one untouched.
            assert np.array_equal(later[:earlier.size], earlier)

    def test_unquenched_run_reoptimizes_earlier_parameters(self, lih_cache):
        seen: list[np.ndarray] = []

        driver = ADAPTVQE(pool="qeb", load_hamiltonian=lih_cache, verbose=False,
                          max_iterations=6, quenching=True)
        driver.run(callback=lambda info: seen.append(
            np.array(info["parameters"], dtype=float)))

        assert len(seen) >= 3
        # At least one growth step moved a previously optimized parameter.
        moved = any(not np.array_equal(later[:earlier.size], earlier)
                    for earlier, later in zip(seen, seen[1:]))
        assert moved

    def test_quenched_energy_is_an_upper_bound(self, lih_cache):
        common = dict(pool="qeb", load_hamiltonian=lih_cache, verbose=False,
                      max_iterations=8)
        full = ADAPTVQE(**common, quenching=True).run()
        quenched = ADAPTVQE(**common, quenching=False).run()

        # Both lower the energy below Hartree-Fock ...
        assert quenched.optimal_energy < quenched.reference_energy
        # ... and the frozen-parameter variant cannot beat the free one.
        assert quenched.optimal_energy >= full.optimal_energy - 1e-8

    def test_quenched_step_is_cheaper_per_operator(self, lih_cache):
        common = dict(pool="qeb", load_hamiltonian=lih_cache, verbose=False,
                      max_iterations=6)
        full = ADAPTVQE(**common, quenching=True).run()
        quenched = ADAPTVQE(**common, quenching=False).run()
        # A 1-D line search per growth step costs far fewer evaluations than a
        # k-dimensional re-optimization.
        assert (quenched.num_evaluations / max(quenched.num_operators, 1)
                < full.num_evaluations / max(full.num_operators, 1))

    def test_both_settings_agree_on_the_first_operator(self, lih_cache):
        """Step 1 has a single parameter, so quenching cannot change it."""
        common = dict(pool="qeb", load_hamiltonian=lih_cache, verbose=False,
                      max_iterations=1)
        full = ADAPTVQE(**common, quenching=True).run()
        quenched = ADAPTVQE(**common, quenching=False).run()
        assert quenched.operators == full.operators
        assert quenched.optimal_energy == pytest.approx(full.optimal_energy,
                                                        abs=1e-6)


# --------------------------------------------------------------------------- #
# VQE: joint minimization vs a one-parameter-at-a-time sweep.
# --------------------------------------------------------------------------- #

class TestVQEQuenching:
    def test_sequential_sweep_still_lowers_the_energy(self, lih_cache):
        vqe = VQE(load_hamiltonian=lih_cache, verbose=False, quenching=False)
        result = vqe.run()
        assert result.optimal_energy < result.reference_energy
        assert result.num_parameters == vqe.ansatz.num_parameters
        assert len(result.history) == result.num_evaluations

    def test_joint_optimization_is_at_least_as_good(self, lih_cache):
        joint = VQE(load_hamiltonian=lih_cache, verbose=False,
                    quenching=True).run()
        swept = VQE(load_hamiltonian=lih_cache, verbose=False,
                    quenching=False).run()
        assert swept.optimal_energy >= joint.optimal_energy - 1e-6

    def test_sweep_moves_every_parameter_in_order(self, lih_cache):
        """Each parameter is optimized once, alone, in index order."""
        vqe = VQE(load_hamiltonian=lih_cache, verbose=False, quenching=False)
        evaluated: list[np.ndarray] = []

        def cost(theta):
            theta = np.asarray(theta, dtype=float).copy()
            evaluated.append(theta)
            return float(np.sum((theta - 0.5) ** 2))

        result = vqe._optimize_all(cost, np.zeros(4))
        assert result.x.size == 4

        # Between consecutive evaluations exactly one coordinate ever moves, and
        # the coordinate being varied advances 0 -> 1 -> 2 -> 3.
        varied: list[int] = []
        for before, after in zip(evaluated, evaluated[1:]):
            moving = np.flatnonzero(np.abs(after - before) > 1e-12)
            assert moving.size <= 1
            if moving.size == 1 and (not varied or varied[-1] != moving[0]):
                varied.append(int(moving[0]))
        assert varied == [0, 1, 2, 3]
        # Every parameter reaches its own minimum, so the sweep is exact here.
        assert np.allclose(result.x, 0.5, atol=1e-3)

    def test_single_parameter_is_unaffected_by_the_policy(self, lih_cache):
        """With one parameter the two policies are the same optimization."""
        vqe = VQE(load_hamiltonian=lih_cache, verbose=False, quenching=False)

        def cost(theta):
            return float((np.asarray(theta, dtype=float)[0] - 0.3) ** 2)

        swept = vqe._optimize_all(cost, np.zeros(1))
        joint = vqe.optimizer.minimize(cost, np.zeros(1))
        assert swept.x[0] == pytest.approx(joint.x[0], abs=1e-9)
        assert swept.nfev == joint.nfev


# --------------------------------------------------------------------------- #
# VASQE inherits the policy through ADAPT-VQE's growth loop.
# --------------------------------------------------------------------------- #

class TestVASQEQuenching:
    def test_quenched_vasqe_freezes_earlier_parameters(self, lih_cache):
        seen: list[np.ndarray] = []
        VASQE(pool="qeb", load_hamiltonian=lih_cache, verbose=False,
              max_iterations=5, temperature=1e-3, seed=2,
              quenching=False).run(callback=lambda info: seen.append(
                  np.array(info["parameters"], dtype=float)))

        assert len(seen) >= 3
        for earlier, later in zip(seen, seen[1:]):
            assert np.array_equal(later[:earlier.size], earlier)

    def test_low_temperature_quenched_vasqe_tracks_quenched_adapt(self,
                                                                 lih_cache):
        """tau -> 0 reduces VASQE to ADAPT-VQE, quenching policy included."""
        common = dict(pool="qeb", load_hamiltonian=lih_cache, verbose=False,
                      max_iterations=5, quenching=False)
        adapt = ADAPTVQE(**common).run()
        vasqe = VASQE(**common, temperature=1e-6, seed=0).run()
        assert vasqe.operators == adapt.operators
        assert vasqe.optimal_energy == pytest.approx(adapt.optimal_energy,
                                                     abs=1e-6)
