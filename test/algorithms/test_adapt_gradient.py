# -*- coding: utf-8 -*-
# file: test/algorithms/test_adapt_gradient.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""How ADAPT screens its pool: the gradient, by name and by formula.

``gradient=`` picks between the analytic commutator, a finite difference and
the parameter-shift rule (:data:`~mandacaru.algorithms.GRADIENT_METHODS`).
All three must agree on the same state, the name must be canonicalized at
construction, and the run report must say which one *ran* -- a sparse or
sector run screens analytically whatever was asked for.
"""

import numpy as np
import pytest

from mandacaru.algorithms import GRADIENT_METHODS, Mandacaru
from mandacaru.core import MolecularIntegrals, minimal_hao_basis
from mandacaru.integrals import Grid
from mandacaru.optimizers import Optimizer
from mandacaru.algorithms import resolve_gradient_method
from mandacaru.circuits import AdaptAnsatz


LBFGS = Optimizer(method="L-BFGS", maxiter=500, tol=1e-12)


@pytest.fixture(scope="module")
def h2_hamiltonian():
    """A 4-qubit H2 Hamiltonian, built once for the whole module."""
    R = 0.74
    nuclei = [(1.0, np.array([0.0, 0.0, -R / 2])),
              (1.0, np.array([0.0, 0.0, +R / 2]))]
    grid = Grid(center=[0, 0, 0], box_size=5.0, h=0.35)
    integrals = MolecularIntegrals(nuclei, minimal_hao_basis(nuclei), grid)
    return integrals.molecular_hamiltonian(mo_basis=True, n_electrons=2)


class TestGradientStrategies:
    @pytest.mark.parametrize("pool", ["fermionic", "qubit", "qeb", "ceo"])
    def test_finite_difference_matches_analytic(self, h2_hamiltonian, pool):
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool=pool, num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=False,
                          gradient="finite_difference")
        psi = AdaptAnsatz(adapt.n_qubits, adapt.pool.occupied_orbitals).state(
            np.zeros(0))
        g_an = adapt._analytic_gradients(psi)
        g_fd = adapt._finite_difference_gradients(psi)
        np.testing.assert_allclose(g_fd, g_an, atol=1e-6)

    @pytest.mark.parametrize("pool", ["fermionic", "qubit", "qeb", "ceo"])
    def test_parameter_shift_is_exact(self, h2_hamiltonian, pool):
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool=pool, num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=False,
                          gradient="parameter_shift")
        psi = AdaptAnsatz(adapt.n_qubits, adapt.pool.occupied_orbitals).state(
            np.zeros(0))
        g_an = adapt._analytic_gradients(psi)
        g_ps = adapt._parameter_shift_gradients(psi)
        np.testing.assert_allclose(g_ps, g_an, atol=1e-9)

    def test_analytic_is_the_default_and_screens_analytically(
            self, h2_hamiltonian):
        """The exact derivative, not an estimate of it, out of the box."""
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool="ceo", num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=False)
        assert adapt.gradient == "analytic"
        psi = AdaptAnsatz(adapt.n_qubits, adapt.pool.occupied_orbitals).state(
            np.zeros(0))
        np.testing.assert_allclose(adapt._gradients(psi),
                                   adapt._analytic_gradients(psi), atol=0)

    def test_the_eigendecomposition_is_built_only_when_asked_for(
            self, h2_hamiltonian):
        """|pool| dense diagonalizations are the shift estimators' cost alone."""
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool="fermionic", num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=False)
        psi = AdaptAnsatz(adapt.n_qubits, adapt.pool.occupied_orbitals).state(
            np.zeros(0))
        adapt._gradients(psi)
        assert adapt._pool_eig is None
        adapt._parameter_shift_gradients(psi)
        assert len(adapt._pool_eig) == len(adapt._pool_matrices)

    def test_every_gradient_reaches_fci(self, h2_hamiltonian):
        m = h2_hamiltonian.map_to_qubits("jordan_wigner").to_matrix()
        exact = float(np.linalg.eigvalsh(0.5 * (m + m.conj().T)).min())
        for grad in GRADIENT_METHODS:
            adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                              pool="ceo", num_particles=(1, 1),
                              n_spatial_orbitals=2, profile=False,
                              gradient=grad, max_iterations=10,
                              gradient_tolerance=1e-4)
            res = adapt.run()
            assert abs(res.in_units("Ha") - exact) < 1e-4, grad

    def test_invalid_gradient_rejected(self, h2_hamiltonian):
        with pytest.raises(ValueError):
            Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                      pool="ceo", num_particles=(1, 1), n_spatial_orbitals=2,
                      gradient="nope")


class TestGradientNames:
    """The three estimators have one canonical spelling each.

    The names used to disagree with each other -- ``finite_difference`` with an
    underscore but ``parameter-shift`` with a hyphen -- so a user could not
    guess either from the other, and the run log wrote back whichever the user
    typed.  Input is now spelling-insensitive (the rule ``method=`` uses) and
    everything written out is the canonical underscore form.
    """

    def test_the_canonical_names_are_underscored(self):
        assert GRADIENT_METHODS == ("analytic", "finite_difference",
                                    "parameter_shift")

    @pytest.mark.parametrize("spelling, canonical", [
        ("parameter-shift", "parameter_shift"),
        ("parameter_shift", "parameter_shift"),
        ("Parameter Shift", "parameter_shift"),
        ("PARAMETER-SHIFT", "parameter_shift"),
        ("finite-difference", "finite_difference"),
        ("finite_difference", "finite_difference"),
        ("Finite Difference", "finite_difference"),
        ("  Analytic  ", "analytic"),
    ])
    def test_every_spelling_resolves(self, spelling, canonical):
        assert resolve_gradient_method(spelling) == canonical

    @pytest.mark.parametrize("spelling", ["parameter-shift", "Parameter Shift",
                                          "finite-difference"])
    def test_the_driver_stores_the_canonical_name(self, h2_hamiltonian,
                                                  spelling):
        """However it was typed, `.gradient` is what the log and tests read."""
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool="ceo", num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=False,
                          gradient=spelling)
        assert adapt.gradient == resolve_gradient_method(spelling)
        assert adapt.gradient in GRADIENT_METHODS

    def test_an_unknown_name_names_the_canonical_ones(self, h2_hamiltonian):
        """The refusal has to be actionable: it lists what is accepted."""
        with pytest.raises(ValueError) as excinfo:
            Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                      pool="ceo", num_particles=(1, 1), n_spatial_orbitals=2,
                      gradient="parametershifted")
        message = str(excinfo.value)
        for name in GRADIENT_METHODS:
            assert name in message

    @pytest.mark.parametrize("gradient", GRADIENT_METHODS)
    def test_the_formula_line_has_no_colon(self, gradient):
        """`parse_output` splits `key: value` on the first colon."""
        from mandacaru.algorithms.adapt_vqe import GRADIENT_FORMULAS
        assert ":" not in GRADIENT_FORMULAS[gradient]

    def test_the_finite_difference_step_is_the_one_used(self, h2_hamiltonian):
        """The step reported in the log is the default the estimator runs at."""
        import inspect

        from mandacaru.algorithms.adapt_vqe import (FINITE_DIFFERENCE_STEP,
                                                    GRADIENT_FORMULAS)
        adapt = Mandacaru(method="adapt-vqe", hamiltonian=h2_hamiltonian,
                          pool="ceo", num_particles=(1, 1),
                          n_spatial_orbitals=2, profile=False,
                          gradient="finite_difference")
        signature = inspect.signature(adapt.solver._finite_difference_gradients)
        assert signature.parameters["eps"].default == FINITE_DIFFERENCE_STEP
        assert f"{FINITE_DIFFERENCE_STEP:g}" in \
            GRADIENT_FORMULAS["finite_difference"]
