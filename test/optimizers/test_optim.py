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

import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.optimizers import Optimizer
from mandacaru.utils.logging import Logger, parse_output

# The classical optimizers used below, with the iteration budget and
# the convergence tolerance written out rather than left to the
# library default: a test that pins an energy should say what it was
# optimized with.
COBYLA_OPT = Optimizer(method="COBYLA", maxiter=2000, tol=1e-12)
LBFGS = Optimizer(method="L-BFGS", maxiter=2000, tol=1e-12)


def lih():
    # A molecule in a 10 A box: with pbc=True it would be a crystal of LiH, solved as a supercell.
    atoms = Atoms("LiH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]],
                  cell=[10.0, 10.0, 10.0], pbc=False)
    atoms.center()
    return atoms


def adapt(**kwargs):
    options = dict(method="adapt-vqe", basis="HAO", grid_spacing=0.35, pool="qubit",
                   max_iterations=3, trace=False)
    options.update(kwargs)
    return Mandacaru(**options)


# --------------------------------------------------------------------------- #

class TestTheDictForm:
    """``optimizer={"method": ..., "maxiter": ..., "tol": ...}``.

    The point is that setting the budget and the tolerance should not require
    importing :class:`~mandacaru.optimizers.Optimizer` into a script that
    otherwise only imports :class:`~mandacaru.Mandacaru`.
    """

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
    def test_a_resumed_run_counts_the_effort_of_its_own_growth_steps(
            self, tmp_path):
        """``resume=`` restores the ansatz as a warm start (run_log.md,
        "lineage"): the iteration rows are the growth steps taken after it,
        each with its own count, and the run total covers them."""
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

        assert len(second.iterations) == 4 - len(first.iterations)
        for iteration in second.iterations:
            assert 1 <= iteration.optimizer_steps <= iteration.num_evaluations
        # The total adds the re-optimization of the restored ansatz to the
        # steps of the new rows.
        assert second.optimizer_steps >= sum(
            it.optimizer_steps for it in second.iterations)
        assert second.num_evaluations >= sum(
            it.num_evaluations for it in second.iterations)


class TestAShotObjectiveGetsANoiseTolerantOptimizer:
    """SLSQP at ``tol=1e-12`` was the default on shot-based objectives too:
    it differences the energy at ~1.5e-8 rad, turning a 0.01 Ha shot noise
    into ~10^6 Ha/rad of gradient noise."""

    def test_the_driver_follows_its_shots(self):
        calc = Mandacaru(method="vqe", basis="HAO", grid_spacing=0.4, shots=1000,
                         trace=False)
        assert calc.solver.optimizer.method == "SPSA"
        with pytest.raises(ValueError, match="shot noise"):
            Mandacaru(method="vqe", basis="HAO", grid_spacing=0.4, shots=1000,
                      optimizer="SLSQP", trace=False).solver


class TestTheGradientIsAnalytic:
    """The native solver has one parameter gradient, the analytic one, and
    the run log reads ``gradient_method: analytic`` in every run
    (``run_log.md``): the finite-difference spelling is refused at
    construction."""

    def test_the_log_says_analytic(self, tmp_path):
        out = str(tmp_path / "lbfgs.txt")
        atoms = lih()
        atoms.calc = adapt(pool="qeb", max_iterations=2, optimizer=LBFGS,
                           txt=out)
        atoms.get_potential_energy()
        assert parse_output(out)["setup"]["gradient_method"] == "analytic"

    def test_a_gradient_free_optimizer_reads_the_same_line(self, tmp_path):
        out = str(tmp_path / "cobyla.txt")
        atoms = lih()
        atoms.calc = adapt(optimizer=COBYLA_OPT, max_iterations=2, txt=out)
        atoms.get_potential_energy()
        setup = parse_output(out)["setup"]
        assert setup["classical_optimizer"] == "COBYLA"
        assert setup["gradient_method"] == "analytic"

    @pytest.mark.parametrize("spelling", ["finite_difference",
                                          "parameter_shift"])
    def test_other_spellings_are_refused(self, spelling):
        with pytest.raises(ValueError, match="analytic"):
            adapt(gradient=spelling)
