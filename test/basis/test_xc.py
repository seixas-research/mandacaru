# -*- coding: utf-8 -*-
# file: test/basis/test_xc.py

"""Exchange-correlation functionals of the radial atom.

The load-bearing test here is
``TestThePotentialIsTheDerivativeOfTheEnergy``: a GGA potential carries a
divergence term, and getting it wrong produces an SCF that converges to the
wrong answer rather than one that crashes.  The check is the definition --
``E[rho + d] - E[rho]`` against ``int v_xc d`` -- and it is stated as a
**convergence rate**, because the identity is exact only in the continuum and
both sides are discretized.
"""

import numpy as np
import pytest

from mandacaru.basis.atomic_solver import lda_correlation, lda_xc
from mandacaru.basis.xc import (DENSITY_FLOOR, FUNCTIONALS, PBE_KAPPA,
                                pbe_correlation, pbe_exchange,
                                pw92_correlation, xc_energy_density,
                                xc_potential)


def radial_grid(points=4000, r_max=25.0):
    return np.arange(1, points + 1) * (r_max / (points + 1))


def two_shell_density(r):
    """A realistic atomic density: a tight core and a diffuse valence."""
    return (8 * np.exp(-16 * r) * (8 ** 3 / np.pi)
            + 2 * np.exp(-2.4 * r) * (1.2 ** 3 / np.pi))


class TestPerdewWang92:
    def test_the_potential_is_the_derivative_of_the_energy(self):
        rho = np.array([1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0])
        step = rho * 1e-7

        def f(x):
            return x * pw92_correlation(x)[0]

        numeric = (f(rho + step) - f(rho - step)) / (2 * step)
        assert np.allclose(pw92_correlation(rho)[1], numeric, rtol=1e-6)

    def test_it_agrees_with_perdew_zunger_to_about_a_millihartree(self):
        """Same Ceperley-Alder data, different parameterization."""
        rho = np.array([1e-3, 1e-2, 0.1, 1.0])
        spread = np.abs(pw92_correlation(rho)[0] - lda_correlation(rho)[0])
        assert spread.max() < 1e-3

    def test_correlation_is_negative(self):
        assert np.all(pw92_correlation(np.array([1e-3, 0.1, 10.0]))[0] < 0)


class TestPBEReducesToItsUniformLimit:
    def test_exchange_at_zero_gradient_is_slater(self):
        rho = np.array([1e-3, 1e-2, 0.1, 1.0, 10.0])
        zero = np.zeros_like(rho)
        slater = lda_xc(rho)[0] - lda_correlation(rho)[0]
        assert np.allclose(pbe_exchange(rho, zero), slater, rtol=1e-14)

    def test_correlation_at_zero_gradient_is_pw92(self):
        rho = np.array([1e-3, 1e-2, 0.1, 1.0, 10.0])
        zero = np.zeros_like(rho)
        assert np.allclose(pbe_correlation(rho, zero),
                           pw92_correlation(rho)[0], rtol=1e-14)

    def test_exchange_saturates_at_the_lieb_oxford_bound(self):
        """F_x -> 1 + kappa as s -> infinity, and never exceeds it."""
        rho = np.array([1.0])
        slater = lda_xc(rho)[0] - lda_correlation(rho)[0]
        huge = pbe_exchange(rho, np.array([1e8])) / slater
        assert huge == pytest.approx(1.0 + PBE_KAPPA, rel=1e-9)
        for gradient in (0.0, 1.0, 10.0, 1e3, 1e6):
            ratio = pbe_exchange(rho, np.array([gradient])) / slater
            assert 1.0 <= ratio <= 1.0 + PBE_KAPPA + 1e-12

    def test_a_gradient_lowers_the_exchange_energy(self):
        rho = np.array([0.5])
        assert pbe_exchange(rho, np.array([0.3])) < pbe_exchange(
            rho, np.array([0.0]))

    def test_the_tail_falls_back_to_the_uniform_gas(self):
        """Below the floor s and t would diverge on numerical noise."""
        tiny = np.array([DENSITY_FLOOR / 10.0])
        assert np.isfinite(pbe_exchange(tiny, np.array([1e-30])))
        assert np.isfinite(pbe_correlation(tiny, np.array([1e-30])))


class TestThePotentialIsTheDerivativeOfTheEnergy:
    """``dE = int v_xc drho`` over ``4 pi r^2 dr`` -- the definition."""

    def _residual(self, points, functional="pbe", epsilon=1e-4):
        r = radial_grid(points)
        rho = two_shell_density(r)
        shell = 4.0 * np.pi * r * r

        def energy(density):
            gradient = np.gradient(density, r, edge_order=2)
            return np.trapezoid(
                xc_energy_density(density, gradient, functional) * shell, r)

        _e, v = xc_potential(r, rho, functional)
        delta = epsilon * np.exp(-((r - 0.9) / 0.45) ** 2) * rho
        left = energy(rho + delta) - energy(rho - delta)
        right = 2.0 * np.trapezoid(v * delta * shell, r)
        return abs(left - right) / abs(right)

    def test_pbe_converges_at_second_order(self):
        """Not a tolerance: a *rate*.  The identity is exact only as h -> 0."""
        coarse = self._residual(4000)
        fine = self._residual(8000)
        assert np.log2(coarse / fine) == pytest.approx(2.0, abs=0.2)
        assert fine < 2e-5

    def test_the_residual_does_not_depend_on_the_perturbation_size(self):
        """A wrong derivative would scale with epsilon; discretization does not."""
        big = self._residual(4000, epsilon=1e-3)
        small = self._residual(4000, epsilon=1e-5)
        assert big == pytest.approx(small, rel=1e-3)

    def test_lda_satisfies_it_too(self):
        assert self._residual(4000, "lda") < 1e-9


class TestTheDispatcher:
    def test_lda_delegates_unchanged(self):
        """Switching to 'pbe' and back must reproduce the old numbers exactly."""
        r = radial_grid(2000)
        rho = two_shell_density(r)
        e_xc, v_xc = xc_potential(r, rho, "lda")
        e_ref, v_ref = lda_xc(rho)
        assert np.array_equal(e_xc, e_ref)
        assert np.array_equal(v_xc, v_ref)

    @pytest.mark.parametrize("name", ["lda", "LDA", "pz", "pbe", "PBE", "gga"])
    def test_the_accepted_spellings(self, name):
        r = radial_grid(1000)
        assert xc_potential(r, two_shell_density(r), name)[1].shape == r.shape

    def test_an_unknown_functional_is_named_with_the_alternatives(self):
        r = radial_grid(100)
        with pytest.raises(ValueError, match="unknown exchange-correlation"):
            xc_potential(r, two_shell_density(r), "b3lyp")

    def test_the_two_functionals_disagree_where_the_gradient_is_large(self):
        r = radial_grid(4000)
        rho = two_shell_density(r)
        lda = xc_potential(r, rho, "lda")[1]
        pbe = xc_potential(r, rho, "pbe")[1]
        assert np.max(np.abs(pbe - lda)) > 0.01

    def test_the_catalogue_is_what_is_implemented(self):
        assert set(FUNCTIONALS) == {"lda", "pbe"}


class TestAgainstTheSelfConsistentAtom:
    def test_a_pbe_atom_differs_from_an_lda_one_and_stays_bound(self):
        from mandacaru.basis.atomic_solver import solve_atom

        lda = solve_atom(8, points=2500, r_max=18.0, tolerance=1e-5)
        pbe = solve_atom(8, points=2500, r_max=18.0, tolerance=1e-5, xc="pbe")
        assert pbe.xc == "pbe" and lda.xc == "lda"
        assert pbe.total_energy != lda.total_energy
        # PBE binds oxygen more tightly than LDA does, but not by a lot.
        assert abs(pbe.total_energy - lda.total_energy) < 2.0
        assert pbe.eigenvalues[(2, 1)] < 0.0
