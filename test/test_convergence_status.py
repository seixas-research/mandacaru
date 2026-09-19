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
from ase import Atoms

from mandacaru.algorithms import Mandacaru, resolve_method
from mandacaru.core.hamiltonian import spin_block_integrals
from mandacaru.core.mapping import Fermion
from mandacaru.optimizers.optim import Optimizer


def random_hamiltonian(orbitals=3, seed=5):
    """A Hermitian, number-conserving Hamiltonian on ``2 * orbitals`` modes."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(orbitals, orbitals))
    h = a + a.T
    eri = np.zeros((orbitals,) * 4)
    for _ in range(2):
        b = rng.normal(size=(orbitals, orbitals))
        L = b + b.T
        eri += np.einsum("pr,qs->pqrs", L, L)
    fermion = Fermion.from_integrals(*spin_block_integrals(h, eri))
    return fermion.map_to_qubits("jordan_wigner", n_modes=2 * orbitals)


def driver(**options):
    return Mandacaru(method="adapt-vqe", hamiltonian=random_hamiltonian(),
                     num_particles=(1, 1), n_spatial_orbitals=3, trace=False,
                     profile=False, **options)


class TestAdaptConvergenceFlag:
    def test_reports_convergence_after_the_final_screening(self):
        """A loose threshold is met at the last step, not before it."""
        result = driver(pool="qeb", max_iterations=0,
                        gradient_tolerance=1e3).run()
        assert result.converged
        assert np.isfinite(result.final_max_gradient)

    def test_an_unconverged_run_reports_a_finite_final_gradient(self):
        result = driver(pool="qeb", max_iterations=1,
                        gradient_tolerance=1e-12).run()
        assert not result.converged
        assert np.isfinite(result.final_max_gradient)

    def test_optimizer_failures_are_recorded(self):
        """Inner optimizations that never certify convergence are reported."""
        with pytest.warns(RuntimeWarning, match="no convergence"):
            result = driver(pool="qeb", max_iterations=2,
                            gradient_tolerance=1e-12,
                            optimizer="SPSA").run()
        assert result.optimizer_failures
        assert all(isinstance(step, int) for step, _ in result.optimizer_failures)


class TestSectorGuard:
    def test_requested_sector_refuses_a_leaking_pool(self):
        with pytest.raises(ValueError, match="conserve"):
            driver(pool="qubit", sector=True, max_iterations=1).run()

    def test_automatic_sector_falls_back_to_the_full_register(self, monkeypatch):
        monkeypatch.setattr(resolve_method("adapt-vqe")[1],
                            "SECTOR_AUTO_QUBITS", 4)
        # In direct mode the problem is configured in the constructor.
        with pytest.warns(RuntimeWarning, match="full register"):
            solver = driver(pool="qubit", sector="auto", max_iterations=1,
                            gradient_tolerance=1e-6)
        assert solver._sector is None
        assert np.isfinite(solver.run().optimal_energy)

    def test_automatic_sector_is_used_for_a_conserving_pool(self, monkeypatch):
        monkeypatch.setattr(resolve_method("adapt-vqe")[1],
                            "SECTOR_AUTO_QUBITS", 4)
        sector_solver = driver(pool="qeb", sector="auto", max_iterations=3)
        full_solver = driver(pool="qeb", sector=False, max_iterations=3)
        sector_result, full_result = sector_solver.run(), full_solver.run()
        assert sector_solver._sector is not None and full_solver._sector is None
        # Same ansatz, same minimum -- only the state representation differs
        # (9 sector amplitudes against 64 full-register ones).
        assert sector_result.operators == full_result.operators
        assert sector_result.optimal_energy == pytest.approx(
            full_result.optimal_energy, abs=1e-4)


class TestOptimizerStatus:
    def test_non_finite_cost_is_rejected(self):
        with pytest.raises(ValueError, match="cost function returned"):
            Optimizer(method="COBYLA", maxiter=5).minimize(
                lambda x: float("nan"), [0.1])

    def test_native_optimizers_only_claim_success_with_a_tolerance(self):
        quadratic = lambda x: float(np.sum(np.asarray(x) ** 2))
        without = Optimizer(method="SPSA", maxiter=20).minimize(quadratic, [0.3])
        assert not without.success and "no tol set" in without.message
        with_tol = Optimizer(method="Adam", maxiter=200,
                             tol=1e-3).minimize(quadratic, [0.3])
        assert with_tol.success and "met tol" in with_tol.message

    def test_a_zero_parameter_problem_still_succeeds(self):
        result = Optimizer(method="COBYLA").minimize(lambda x: 1.0, [])
        assert result.success and result.nfev == 1


class TestAdaptEndToEndStillWorks:
    def test_h2_reaches_its_fci_energy(self):
        atoms = Atoms("H2", positions=[[4, 4, 3.63], [4, 4, 4.37]],
                      cell=[8.0] * 3)
        atoms.calc = Mandacaru(method="adapt-vqe", basis="FAO", h=0.4,
                               pool="fermionic", trace=False, profile=False,
                               gradient_tolerance=1e-6)
        energy = atoms.get_potential_energy()
        exact = np.linalg.eigvalsh(atoms.calc._h_matrix if isinstance(
            atoms.calc._h_matrix, np.ndarray)
            else atoms.calc._h_matrix.toarray())[0]
        from mandacaru.units import HARTREE_TO_EV
        assert atoms.calc.result.converged
        assert energy == pytest.approx(exact * HARTREE_TO_EV, abs=1e-6)


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
