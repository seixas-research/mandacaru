# -*- coding: utf-8 -*-
# file: test/algorithms/test_convergence.py

"""The ``convergence`` option of the adaptive solvers
(:mod:`mandacaru.algorithms.convergence`).

End to end on LiH (HAO, qubit pool, 6 qubits), where the two criteria stop the
growth at different places: the gradient at 1e-3 after nine operators, the
energy change at 1e-4 Ha after five; and on H2, which one operator makes
exact.
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.algorithms.convergence import (DEFAULT_CONVERGENCE,
                                              DEFAULT_ENERGY_CONVERGENCE,
                                              DEFAULT_GRADIENT_CONVERGENCE,
                                              VANISHED_GRADIENT, Convergence)
from mandacaru.units import HARTREE_TO_EV


class TestResolve:
    def test_none_is_both_at_1e_3(self):
        assert Convergence.resolve(None) is DEFAULT_CONVERGENCE
        assert DEFAULT_CONVERGENCE.as_dict() == {"gradient": 1e-3,
                                                 "energy": 1e-3}
        assert (DEFAULT_GRADIENT_CONVERGENCE, DEFAULT_ENERGY_CONVERGENCE) == \
            (1e-3, 1e-3)

    def test_a_criterion_left_out_is_not_used(self):
        assert Convergence.resolve({"energy": 1e-6}).as_dict() == {
            "gradient": None, "energy": 1e-6}
        assert Convergence.resolve({"gradient": 1e-4}).as_dict() == {
            "gradient": 1e-4, "energy": None}

    def test_none_switches_a_criterion_off(self):
        assert Convergence.resolve({"gradient": None, "energy": 1e-6}) == \
            Convergence.resolve({"energy": 1e-6})

    @pytest.mark.parametrize("spec, error", [
        ({"gradient": None, "energy": None}, "at least one criterion"),
        ({}, "at least one criterion"),
        ({"grad": 1e-3}, "unknown convergence criteria"),
        ({"gradient": 0.0}, "positive number"),
        ({"energy": -1e-6}, "positive number"),
        ({"energy": float("nan")}, "positive number"),
        ({"gradient": True}, "positive number"),
        ({"gradient": "1e-3"}, "positive number"),
    ])
    def test_refused(self, spec, error):
        with pytest.raises(ValueError, match=error):
            Convergence.resolve(spec)

    def test_a_bare_number_is_not_a_dictionary(self):
        with pytest.raises(TypeError, match="must be a dict"):
            Convergence.resolve(1e-3)


class TestReached:
    BOTH = Convergence(gradient=1e-3, energy=1e-6)

    @pytest.mark.parametrize("gradient, delta, expected", [
        (1e-4, 1e-7, True),
        (1e-4, -1e-7, True),          # |dE|: a step may raise the energy
        (1e-2, 1e-7, False),          # the energy alone is not enough
        (1e-4, 1e-5, False),          # nor is the gradient
        (1e-4, None, False),          # no step taken yet
    ])
    def test_both_must_hold(self, gradient, delta, expected):
        assert self.BOTH.reached(gradient, delta) is expected

    def test_an_unused_criterion_is_ignored(self):
        assert Convergence(gradient=None, energy=1e-6).reached(np.inf, 1e-7)
        assert Convergence(gradient=1e-3, energy=None).reached(1e-4, None)

    def test_the_threshold_is_strict(self):
        assert not Convergence(gradient=1e-3, energy=None).reached(1e-3, None)

    @pytest.mark.parametrize("spec", [{"gradient": 1e-3, "energy": 1e-6},
                                      {"energy": 1e-9}])
    def test_a_vanished_gradient_stops_whatever_the_criteria(self, spec):
        """No operator can change the energy then, so a large last dE (or
        none yet) does not keep the growth going."""
        criteria = Convergence.resolve(spec)
        assert criteria.reached(0.5 * VANISHED_GRADIENT, 1.0)
        assert criteria.reached(0.5 * VANISHED_GRADIENT, None)
        assert not criteria.reached(2.0 * VANISHED_GRADIENT, 1.0)

    def test_described_in_the_unit_of_the_de_column(self):
        text = self.BOTH.describe(HARTREE_TO_EV, "eV")
        assert text == f"max|grad| < 0.001 Ha and |dE| < {1e-6 * HARTREE_TO_EV:g} eV"


def _lih(**options):
    atoms = Atoms("LiH", positions=[[0, 0, 0], [0, 0, 1.6]], cell=[8.0] * 3)
    atoms.center()
    options = {"max_iterations": 30, **options}
    atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", h=0.30,
                           pool="qubit", trace=False, **options)
    atoms.get_potential_energy()
    return atoms.calc.result


@pytest.fixture(scope="module")
def default_run():
    return _lih()


class TestGrowth:
    def test_the_default_is_both_at_1e_3(self, default_run):
        explicit = _lih(convergence={"gradient": 1e-3, "energy": 1e-3})
        assert explicit.operators == default_run.operators
        assert default_run.converged
        assert default_run.final_max_gradient < 1e-3
        energies = [step.energy for step in default_run.iterations]
        assert abs(energies[-1] - energies[-2]) / HARTREE_TO_EV < 1e-3

    def test_an_exact_first_step_is_not_followed_by_a_redundant_one(self):
        """H2 is exact after one operator; its large first dE must not make the
        default grow a second one."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]],
                      cell=[8.0] * 3)
        atoms.center()
        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", h=0.30,
                               pool="fermionic", trace=False)
        atoms.get_potential_energy()
        result = atoms.calc.result
        assert result.converged and len(result.operators) == 1
        assert result.final_max_gradient < VANISHED_GRADIENT

    def test_the_energy_alone_stops_on_the_last_step(self, default_run):
        threshold = 1e-4
        result = _lih(convergence={"energy": threshold})
        assert result.converged
        energies = [step.energy for step in result.iterations]   # eV
        last = abs(energies[-1] - energies[-2]) / HARTREE_TO_EV
        before = abs(energies[-2] - energies[-3]) / HARTREE_TO_EV
        assert last < threshold <= before
        # It stops earlier than the gradient does here, with the gradient
        # still well above the default threshold -- the criterion really was
        # the energy.
        assert len(result.operators) < len(default_run.operators)
        assert result.final_max_gradient > 1e-3

    def test_both_run_until_the_later_one_holds(self, default_run):
        result = _lih(convergence={"gradient": 1e-3, "energy": 1e-6})
        assert result.converged
        assert result.final_max_gradient < 1e-3
        energies = [step.energy for step in result.iterations]
        assert abs(energies[-1] - energies[-2]) / HARTREE_TO_EV < 1e-6
        assert len(result.operators) >= len(default_run.operators)

    def test_an_unmet_energy_criterion_is_not_converged(self):
        """A budget spent before |dE| fell is reported as such."""
        result = _lih(convergence={"energy": 1e-12}, max_iterations=3)
        assert not result.converged
