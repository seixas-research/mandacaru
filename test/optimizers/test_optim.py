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
from mandacaru.optimizers import (DEFAULT_OPTIMIZER, DEFAULT_TOL,
                                  NAMED_OPTIMIZERS, Optimizer)
from mandacaru.optimizers.optim import OPTIMIZER_KEYS, resolve_optimizer
from mandacaru.utils.logging import Logger, parse_output

# The classical optimizers used below, with the iteration budget and
# the convergence tolerance written out rather than left to the
# library default: a test that pins an energy should say what it was
# optimized with.
COBYLA_OPT = Optimizer(method="COBYLA", maxiter=2000, tol=1e-12)
LBFGS = Optimizer(method="L-BFGS", maxiter=2000, tol=1e-12)


def quadratic(x):
    return float(np.sum((np.asarray(x, dtype=float) - 0.3) ** 2))


def lih():
    atoms = Atoms("LiH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]],
                  cell=[10.0, 10.0, 10.0], pbc=True)
    atoms.center()
    return atoms


def adapt(**kwargs):
    options = dict(method="adapt-vqe", basis="HAO", grid_spacing=0.35, pool="qubit",
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

    def test_a_gradient_method_takes_fewer_steps_than_a_direct_search(self):
        gradient = Optimizer(method="L-BFGS", maxiter=500,
                             tol=1e-10).minimize(quadratic, np.zeros(4))
        search = Optimizer(method="Nelder-Mead", maxiter=500,
                           tol=1e-10).minimize(quadratic, np.zeros(4))
        assert gradient.nit < search.nit


class TestTheDictForm:
    """``optimizer={"method": ..., "maxiter": ..., "tol": ...}``.

    The point is that setting the budget and the tolerance should not require
    importing :class:`~mandacaru.optimizers.Optimizer` into a script that
    otherwise only imports :class:`~mandacaru.Mandacaru`.
    """

    def test_it_builds_the_same_optimizer_as_the_object(self):
        spec = {"method": "COBYLA", "maxiter": 500, "tol": 1e-9}
        built = resolve_optimizer(spec)
        assert (built.method, built.maxiter, built.tol) == \
            ("COBYLA", 500, 1e-9)

    def test_keys_left_out_keep_their_defaults(self):
        built = resolve_optimizer({"maxiter": 500})
        assert built.method == DEFAULT_OPTIMIZER
        assert (built.maxiter, built.tol) == (500, DEFAULT_TOL)
        assert resolve_optimizer({}).method == DEFAULT_OPTIMIZER

    def test_the_other_constructor_keys_work_too(self):
        built = resolve_optimizer({"method": "SPSA", "options": {"a": 0.1},
                                   "seed": 7})
        assert built.seed == 7 and built.options == {"a": 0.1}
        assert set(OPTIMIZER_KEYS) == {"method", "maxiter", "tol", "options",
                                       "seed"}

    def test_an_unknown_key_is_refused_as_the_typo_it_is(self):
        with pytest.raises(ValueError, match="unknown optimizer option"):
            resolve_optimizer({"maxiterations": 5})

    def test_an_unknown_method_is_refused_like_the_string_form(self):
        with pytest.raises(ValueError, match="unknown optimizer"):
            resolve_optimizer({"method": "nope"})

    def test_something_else_entirely_is_a_type_error(self):
        with pytest.raises(TypeError, match="method name, a dict"):
            resolve_optimizer(3)

    def test_the_calculator_takes_it(self):
        atoms = lih()
        atoms.calc = adapt(optimizer={"method": "COBYLA", "maxiter": 500,
                                      "tol": 1e-9})
        atoms.get_total_energy()
        built = atoms.calc.solver.optimizer
        assert (built.method, built.maxiter, built.tol) == \
            ("COBYLA", 500, 1e-9)
        assert atoms.calc.result.optimizer_steps > 0

    def test_a_bad_dict_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="unknown optimizer option"):
            adapt(optimizer={"method": "COBYLA", "iterations": 5})


class TestTheDriverAccumulatesThem:
    def test_each_iteration_records_its_own_effort(self):
        atoms = lih()
        atoms.calc = adapt(optimizer=LBFGS)
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
        atoms.calc = adapt(optimizer=LBFGS)
        atoms.get_total_energy()
        result = atoms.calc.result
        assert result.optimizer_steps < result.num_evaluations

    def test_a_sequential_sweep_sums_its_per_parameter_searches(self):
        # quenching=False optimizes one parameter at a time; the step count is
        # the total effort, not the number of sweeps.
        atoms = lih()
        atoms.calc = adapt(optimizer=LBFGS, quenching=False)
        atoms.get_total_energy()
        result = atoms.calc.result
        assert result.optimizer_steps >= len(result.iterations)


class TestTheIterationsTable:
    def test_the_table_has_a_steps_column(self, tmp_path):
        atoms = lih()
        atoms.calc = adapt(optimizer=LBFGS,
                           txt=str(tmp_path / "output.txt"))
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
                           txt=str(tmp_path / "output.txt"))
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
        logger = Logger(str(path), banner=False)

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
        atoms.calc = adapt(optimizer=LBFGS, max_iterations=2,
                           checkpoint=path)
        atoms.get_total_energy()
        first = atoms.calc.result

        resumed = lih()
        resumed.calc = adapt(optimizer=LBFGS, max_iterations=4,
                             resume=path)
        resumed.get_total_energy()
        second = resumed.calc.result

        # The restored iterations carry their own counts...
        restored = second.iterations[:len(first.iterations)]
        assert [it.optimizer_steps for it in restored] == \
            [it.optimizer_steps for it in first.iterations]
        # ...and the run total continues from where the file stopped.
        assert second.optimizer_steps > first.optimizer_steps


class TestAShotObjectiveGetsANoiseTolerantOptimizer:
    """SLSQP at ``tol=1e-12`` was the default on shot-based objectives too:
    it differences the energy at ~1.5e-8 rad, turning a 0.01 Ha shot noise
    into ~10^6 Ha/rad of gradient noise."""

    def test_the_default_is_spsa_with_shots(self):
        spsa = resolve_optimizer(None, shots=1000)
        assert spsa.method == "SPSA" and spsa.tol is None
        assert spsa.options["track_best"] is False
        assert resolve_optimizer(None).method == DEFAULT_OPTIMIZER

    def test_a_dict_without_a_method_keeps_the_shot_default(self):
        spsa = resolve_optimizer({"maxiter": 20}, shots=1000)
        assert spsa.method == "SPSA" and spsa.maxiter == 20

    @pytest.mark.parametrize("method", ["SLSQP", "BFGS", "L-BFGS", "CG"])
    def test_a_finite_difference_method_is_refused(self, method):
        with pytest.raises(ValueError, match="shot noise"):
            resolve_optimizer(method, shots=1000)
        with pytest.raises(ValueError, match="shot noise"):
            resolve_optimizer({"method": method}, shots=1000)
        assert resolve_optimizer(method).method == method

    @pytest.mark.parametrize("method", ["COBYLA", "SPSA", "Nelder-Mead"])
    def test_the_others_are_accepted(self, method):
        assert resolve_optimizer(method, shots=1000).method == method

    def test_the_driver_follows_its_shots(self):
        calc = Mandacaru(method="vqe", basis="HAO", grid_spacing=0.4, shots=1000,
                         trace=False)
        assert calc.solver.optimizer.method == "SPSA"
        with pytest.raises(ValueError, match="shot noise"):
            Mandacaru(method="vqe", basis="HAO", grid_spacing=0.4, shots=1000,
                      optimizer="SLSQP", trace=False).solver


class TestAnExactGradient:
    """``minimize(..., gradient=)``: the gradient methods take the caller's
    exact gradient instead of differencing the cost; the others ignore it."""

    @staticmethod
    def value_and_gradient(x):
        x = np.asarray(x, dtype=float)
        return quadratic(x), 2.0 * (x - 0.3)

    @pytest.mark.parametrize("method", ["SLSQP", "BFGS", "L-BFGS", "CG"])
    def test_a_gradient_method_uses_it(self, method):
        x0 = np.zeros(8)
        optimizer = Optimizer(method=method, maxiter=200, tol=1e-12)
        differenced = optimizer.minimize(quadratic, x0)
        exact = optimizer.minimize(quadratic, x0,
                                   gradient=self.value_and_gradient)
        assert np.allclose(exact.x, 0.3, atol=1e-5)
        assert exact.fun == pytest.approx(differenced.fun, abs=1e-9)
        assert exact.nfev < differenced.nfev / 3
        assert exact.nfev == len(exact.history)

    @pytest.mark.parametrize("method", ["COBYLA", "Nelder-Mead", "SPSA"])
    def test_a_gradient_free_method_never_calls_it(self, method):
        def gradient(_x):
            raise AssertionError("a gradient-free method asked for a gradient")

        result = Optimizer(method=method, maxiter=50, tol=1e-8).minimize(
            quadratic, np.zeros(2), gradient=gradient)
        assert result.nfev > 0

    def test_a_nonfinite_gradient_is_refused(self):
        with pytest.raises(ValueError, match="gradient is not finite"):
            Optimizer(method="L-BFGS", maxiter=10, tol=1e-12).minimize(
                quadratic, np.zeros(2),
                gradient=lambda x: (quadratic(x), np.array([np.nan, 0.0])))


class TestTheDriverUsesTheAdjointGradient:
    """ADAPT-VQE's inner optimizations take the adjoint gradient by default:
    the same energy as the optimizer's own finite difference, with a fraction
    of the evaluations, and the log says which one ran."""

    def test_same_energy_far_fewer_evaluations(self, tmp_path):
        runs = {}
        for gradient in ("analytic", "finite_difference"):
            atoms = lih()
            out = str(tmp_path / f"{gradient}.txt")
            atoms.calc = adapt(pool="qeb", max_iterations=5,
                               optimizer=LBFGS, gradient=gradient, txt=out)
            energy = atoms.get_potential_energy()
            runs[gradient] = (energy, atoms.calc.result.num_evaluations,
                              parse_output(out)["setup"]["parameter_gradient"])
        assert runs["analytic"][0] == pytest.approx(runs["finite_difference"][0],
                                                    abs=1e-7)
        assert runs["analytic"][1] < runs["finite_difference"][1] / 2
        assert runs["analytic"][2].startswith("analytic (adjoint")
        assert runs["finite_difference"][2].startswith("finite difference")

    def test_a_gradient_free_optimizer_says_so(self, tmp_path):
        out = str(tmp_path / "cobyla.txt")
        atoms = lih()
        atoms.calc = adapt(optimizer=COBYLA_OPT, max_iterations=2, txt=out)
        atoms.get_potential_energy()
        assert parse_output(out)["setup"]["parameter_gradient"].startswith(
            "none (COBYLA")


class TestACustomOptimizer:
    """An :class:`Optimizer` subclass whose ``minimize`` takes no
    ``gradient`` keeps working: the driver offers the adjoint gradient only
    to a ``minimize`` that accepts one."""

    class Plain(Optimizer):
        def minimize(self, cost, x0, callback=None):
            return Optimizer(method="COBYLA", maxiter=500,
                             tol=1e-10).minimize(cost, x0, callback=callback)

    def test_adapt_runs_with_it_and_differences(self, tmp_path):
        out = str(tmp_path / "plain.txt")
        atoms = lih()
        atoms.calc = adapt(optimizer=self.Plain(method="L-BFGS"),
                           max_iterations=2, txt=out)
        assert np.isfinite(atoms.get_potential_energy())
        assert not parse_output(out)["setup"]["parameter_gradient"].startswith(
            "analytic")
