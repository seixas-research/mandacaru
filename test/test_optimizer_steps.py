# -*- coding: utf-8 -*-
# file: test/test_optimizer_steps.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Counting the classical optimizer's steps, and reporting them.

A *step* is a parameter update; an *evaluation* is a call to the cost function.
The two differ by an order of magnitude between methods -- a finite-difference
gradient costs ``2N`` evaluations for one step, a direct search costs one --
so the run log reports both: ``steps`` per growth step in the ``[ITERATIONS]``
table, ``optimizer_steps`` and ``cost_evaluations`` for the whole run in the
summary block.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.optimizers import NAMED_OPTIMIZERS, Optimizer
from mandacaru.utils.logging import AdaptOutputLogger, parse_output

# The classical optimizers used below, with the iteration budget and
# the convergence tolerance written out rather than left to the
# library default: a test that pins an energy should say what it was
# optimized with.
COBYLA_OPT = Optimizer(method="COBYLA", maxiter=2000, tol=1e-12)
LBFGSB = Optimizer(method="L-BFGS-B", maxiter=2000, tol=1e-12)


def quadratic(x):
    return float(np.sum((np.asarray(x, dtype=float) - 0.3) ** 2))


def lih():
    atoms = Atoms("LiH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]],
                  cell=[10.0, 10.0, 10.0], pbc=True)
    atoms.center()
    return atoms


def adapt(**kwargs):
    options = dict(method="adapt-vqe", basis="FAO", h=0.35, pool="qubit",
                   max_iterations=3, trace=False)
    options.update(kwargs)
    return Mandacaru(**options)


# --------------------------------------------------------------------------- #

class TestEveryOptimizerReportsItsSteps:
    @pytest.mark.parametrize("method", NAMED_OPTIMIZERS)
    def test_a_step_count_is_reported(self, method):
        result = Optimizer(method=method, maxiter=200,
                           tol=1e-12).minimize(quadratic, np.zeros(3))
        assert result.nit is not None, method
        assert result.nit >= 1, method

    @pytest.mark.parametrize("method", NAMED_OPTIMIZERS)
    def test_a_step_never_costs_less_than_an_evaluation(self, method):
        # Whatever the method, it cannot move more often than it looked.
        result = Optimizer(method=method, maxiter=200,
                           tol=1e-12).minimize(quadratic, np.zeros(3))
        assert result.nit <= result.nfev, method

    def test_cobyla_has_no_scipy_nit_and_is_counted_here(self):
        """The default optimizer is exactly the one SciPy does not count."""
        from scipy.optimize import minimize
        raw = minimize(quadratic, np.zeros(3), method="COBYLA",
                       options={"maxiter": 200})
        assert getattr(raw, "nit", None) is None
        assert Optimizer(method="COBYLA", maxiter=200).minimize(
            quadratic, np.zeros(3)).nit > 0

    def test_nothing_to_optimize_is_zero_steps(self):
        result = Optimizer().minimize(lambda x: 1.0, [])
        assert result.nit == 0 and result.nfev == 1

    def test_spsa_steps_are_the_loop_count(self):
        # Two evaluations for the gradient plus one to keep the best iterate,
        # and one more before the loop: nfev == 3 * steps + 1.
        result = Optimizer(method="SPSA", maxiter=17).minimize(
            quadratic, np.zeros(2))
        assert result.nit == 17
        assert result.nfev == 3 * result.nit + 1

    def test_adam_steps_are_the_loop_count(self):
        # A central-difference gradient (2N) plus one evaluation at the new
        # point, and one before the loop.
        n = 2
        result = Optimizer(method="Adam", maxiter=11).minimize(
            quadratic, np.zeros(n))
        assert result.nit == 11
        assert result.nfev == (2 * n + 1) * result.nit + 1

    def test_a_gradient_method_takes_fewer_steps_than_a_direct_search(self):
        gradient = Optimizer(method="L-BFGS-B", maxiter=500,
                             tol=1e-10).minimize(quadratic, np.zeros(4))
        search = Optimizer(method="Nelder-Mead", maxiter=500,
                           tol=1e-10).minimize(quadratic, np.zeros(4))
        assert gradient.nit < search.nit


class TestTheDriverAccumulatesThem:
    def test_each_iteration_records_its_own_effort(self):
        atoms = lih()
        atoms.calc = adapt(optimizer=LBFGSB)
        atoms.get_total_energy()
        result = atoms.calc.result
        assert result.iterations
        for iteration in result.iterations:
            assert iteration.optimizer_steps is not None
            assert iteration.num_evaluations is not None
            assert 1 <= iteration.optimizer_steps <= iteration.num_evaluations

    def test_the_total_is_the_sum_of_the_iterations(self):
        atoms = lih()
        atoms.calc = adapt(optimizer=COBYLA_OPT)
        atoms.get_total_energy()
        result = atoms.calc.result
        assert result.optimizer_steps == sum(
            it.optimizer_steps for it in result.iterations)
        assert result.num_evaluations == sum(
            it.num_evaluations for it in result.iterations)

    def test_steps_and_evaluations_are_different_numbers(self):
        """The point of the column: a gradient method's two costs diverge."""
        atoms = lih()
        atoms.calc = adapt(optimizer=LBFGSB)
        atoms.get_total_energy()
        result = atoms.calc.result
        assert result.optimizer_steps < result.num_evaluations

    def test_a_sequential_sweep_sums_its_per_parameter_searches(self):
        # quenching=False optimizes one parameter at a time; the step count is
        # the total effort, not the number of sweeps.
        atoms = lih()
        atoms.calc = adapt(optimizer=LBFGSB, quenching=False)
        atoms.get_total_energy()
        result = atoms.calc.result
        assert result.optimizer_steps >= len(result.iterations)


class TestTheIterationsTable:
    def test_the_table_has_a_steps_column(self, tmp_path):
        atoms = lih()
        atoms.calc = adapt(optimizer=LBFGSB,
                           output=str(tmp_path / "output.txt"))
        atoms.get_total_energy()
        parsed = parse_output(str(tmp_path / "output.txt"))
        assert parsed["iterations"]
        for row, iteration in zip(parsed["iterations"],
                                  atoms.calc.result.iterations):
            assert "steps" in row["columns"]
            assert int(row["optimizer_steps"]) == iteration.optimizer_steps

    def test_the_summary_reports_the_run_total(self, tmp_path):
        atoms = lih()
        atoms.calc = adapt(optimizer=COBYLA_OPT,
                           output=str(tmp_path / "output.txt"))
        atoms.get_total_energy()
        parsed = parse_output(str(tmp_path / "output.txt"))
        summary = parsed["summary"]
        assert int(summary["optimizer_steps"]) == \
            atoms.calc.result.optimizer_steps
        # Both currencies, side by side, and they are not the same number.
        assert int(summary["cost_evaluations"]) == \
            atoms.calc.result.num_evaluations

    def test_an_unreported_step_count_is_a_dash_not_a_zero(self, tmp_path):
        """A method that counts nothing must not look like one that did
        nothing."""
        path = tmp_path / "output.txt"
        logger = AdaptOutputLogger(str(path), banner=False)

        class Op:
            kind, label = "pauli", "iP[XYZ]"

        logger.write_iteration(iteration=1, pool_operators=[Op()],
                               gradients=[0.5], selected_index=0,
                               expressivity=None, energy=-1.0,
                               num_parameters=1, optimizer_steps=None)
        logger.close()
        row = parse_output(str(path))["iterations"][0]
        assert row["optimizer_steps"] == "-"

    def test_an_older_log_without_the_column_still_parses(self, tmp_path):
        path = tmp_path / "old.txt"
        path.write_text(
            "[ITERATIONS]\n"
            "    iter        energy (eV)     type    |grad| operator\n"
            "    ------------------------------------------------------\n"
            "       1       -1.2345678901    pauli  1.0e-01 iP[XYZ]\n")
        row = parse_output(str(path))["iterations"][0]
        assert row["optimizer_steps"] == "-"
        assert row["selected_operator"] == "iP[XYZ]"


class TestCheckpoints:
    def test_a_resumed_run_keeps_the_effort_it_already_spent(self, tmp_path):
        path = str(tmp_path / "state.json")
        atoms = lih()
        atoms.calc = adapt(optimizer=LBFGSB, max_iterations=2,
                           checkpoint=path)
        atoms.get_total_energy()
        first = atoms.calc.result

        resumed = lih()
        resumed.calc = adapt(optimizer=LBFGSB, max_iterations=4,
                             resume=path)
        resumed.get_total_energy()
        second = resumed.calc.result

        # The restored iterations carry their own counts...
        restored = second.iterations[:len(first.iterations)]
        assert [it.optimizer_steps for it in restored] == \
            [it.optimizer_steps for it in first.iterations]
        # ...and the run total continues from where the file stopped.
        assert second.optimizer_steps > first.optimizer_steps
