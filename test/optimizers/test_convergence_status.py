# -*- coding: utf-8 -*-
# file: test/test_convergence_status.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Convergence is reported for the state that is actually returned.

ADAPT screens once more after its last growth step, so a run that meets the
gradient threshold on that step says ``converged``; the classical optimizers
report success only when a tolerance was actually met, and a non-finite cost
stops the run instead of steering it.
"""

import numpy as np
import pytest

from mandacaru.optimizers.optim import Optimizer

# The classical optimizers used below, with the iteration budget and
# the convergence tolerance written out rather than left to the
# library default: a test that pins an energy should say what it was
# optimized with.
SPSA_OPT = Optimizer(method="SPSA", maxiter=2000, tol=1e-12)
# Without a tolerance a native method has no criterion to certify against,
# so it always reports failure -- which is what the reporting test needs.
UNCERTIFIABLE = Optimizer(method="SPSA", maxiter=20, tol=None)










class TestOptimizerStatus:
    def test_non_finite_cost_is_rejected(self):
        with pytest.raises(ValueError, match="cost function returned"):
            Optimizer(method="COBYLA", maxiter=5).minimize(
                lambda x: float("nan"), [0.1])

    def test_the_native_optimizer_only_claims_success_with_a_tolerance(self):
        quadratic = lambda x: float(np.sum(np.asarray(x) ** 2))
        # tol=None is the only way to get there now that DEFAULT_TOL is set.
        without = Optimizer(method="SPSA", maxiter=20, tol=None).minimize(
            quadratic, [0.3])
        assert not without.success and "no tol set" in without.message
        with_tol = Optimizer(method="SPSA", maxiter=400,
                             tol=1e-3).minimize(quadratic, [0.3])
        assert with_tol.success and "met tol" in with_tol.message

    def test_a_zero_parameter_problem_still_succeeds(self):
        result = Optimizer(method="COBYLA").minimize(lambda x: 1.0, [])
        assert result.success and result.nfev == 1




class TestSPSAEvaluationBudget:
    """What an SPSA step costs, and when it may call itself converged."""

    @staticmethod
    def _quadratic(x):
        return float(np.sum(np.asarray(x) ** 2))

    def test_strict_form_costs_two_evaluations_per_step(self):
        result = Optimizer(method="SPSA", maxiter=10,
                           options={"track_best": False}).minimize(
                               self._quadratic, [0.3, -0.2])
        assert result.nfev == 2 * 10 + 1          # + the one final evaluation

    def test_default_keeps_the_best_iterate_with_a_third_evaluation(self):
        result = Optimizer(method="SPSA", maxiter=10).minimize(
            self._quadratic, [0.3, -0.2])
        assert result.nfev == 3 * 10 + 1          # + the starting point
        assert result.fun <= self._quadratic([0.3, -0.2])

    def test_one_small_step_is_not_convergence(self):
        """``patience`` consecutive calm steps are required, not one."""
        one = Optimizer(method="SPSA", maxiter=50, tol=1e-2,
                        options={"patience": 1}).minimize(self._quadratic, [0.3])
        five = Optimizer(method="SPSA", maxiter=50, tol=1e-2,
                         options={"patience": 5}).minimize(self._quadratic, [0.3])
        assert one.success and five.success
        assert five.nfev >= one.nfev + 4 * 3       # at least four more steps
