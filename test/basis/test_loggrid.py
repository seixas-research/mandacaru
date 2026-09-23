# -*- coding: utf-8 -*-
# file: test/basis/test_loggrid.py

"""The logarithmic-grid radial solver.

Every test here compares against something the grid cannot know: a closed-form
spectrum, an exact degeneracy, or a convergence rate.  A radial solver that is
only checked against itself will happily converge to the wrong equation, which
is precisely what happened to the first version of this module.
"""

from __future__ import annotations

import numpy as np
import pytest

from mandacaru.basis.loggrid import (log_grid, radial_g, solve_radial_log,
                                     _fourth_order_derivatives)
from mandacaru.basis.relativity import hydrogenic_dirac_energy


def coulomb(Z, points=1200, r_max=None):
    r, x, dx = log_grid(160.0 / Z if r_max is None else r_max, Z, points=points)
    return r, -Z / r, dx


class TestTheGrid:

    def test_it_is_uniform_in_log_r(self):
        r, x, dx = log_grid(20.0, 8.0, points=500)
        assert np.allclose(np.diff(np.log(r)), dx)

    def test_the_inner_edge_scales_with_the_nuclear_charge(self):
        """r_min tracks the 1s shell, not the box: a heavier atom starts
        closer in."""
        _, x1, _ = log_grid(20.0, 1.0)
        _, x8, _ = log_grid(20.0, 8.0)
        assert x8[0] == pytest.approx(x1[0] - np.log(8.0))


class TestTheNonRelativisticSpectrum:
    """eps = -Z^2/2n^2, exactly."""

    @pytest.mark.parametrize("Z", [1, 8, 20])
    @pytest.mark.parametrize("n,l", [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)])
    def test_it_reproduces_the_hydrogenic_level(self, Z, n, l):
        r, V, _ = coulomb(Z)
        _, eps = solve_radial_log(r, V, l, n - l - 1, atomic_number=Z)
        assert eps == pytest.approx(-Z * Z / (2.0 * n * n), rel=1e-6)

    @pytest.mark.parametrize("n,l", [(1, 0), (2, 0), (2, 1), (3, 1), (3, 2)])
    def test_the_wave_carries_the_requested_number_of_nodes(self, n, l):
        """Node count is how the state is selected, so it is not a detail."""
        r, V, _ = coulomb(8)
        P, _ = solve_radial_log(r, V, l, n - l - 1, atomic_number=8)
        meaningful = np.abs(P) > 1e-10 * np.max(np.abs(P))
        nodes = int(np.count_nonzero(np.diff(np.signbit(P[meaningful]))))
        assert nodes == n - l - 1

    def test_the_wave_is_normalized_with_the_logarithmic_measure(self):
        r, V, _ = coulomb(8)
        P, _ = solve_radial_log(r, V, 0, 0, atomic_number=8)
        assert np.trapezoid(P * P * r, np.log(r)) == pytest.approx(1.0, rel=1e-8)


class TestTheDiracSpectrum:
    """The point of the module: what the uniform grid could not do."""

    @pytest.mark.parametrize("Z", [1, 4, 8, 20])
    def test_the_1s_matches_the_closed_form(self, Z):
        r, V, _ = coulomb(Z)
        _, eps = solve_radial_log(r, V, 0, 0, kappa=-1, treatment="dirac",
                                  atomic_number=Z)
        assert eps == pytest.approx(hydrogenic_dirac_energy(Z, 1, -1), rel=1e-5)

    @pytest.mark.parametrize("n,l,kappa", [(2, 0, -1), (2, 1, 1), (2, 1, -2),
                                           (3, 2, 2), (3, 2, -3)])
    def test_the_j_resolved_levels_match_the_closed_form(self, n, l, kappa):
        r, V, _ = coulomb(20)
        _, eps = solve_radial_log(r, V, l, n - l - 1, kappa=kappa,
                                  treatment="dirac", atomic_number=20)
        assert eps == pytest.approx(hydrogenic_dirac_energy(20, n, kappa),
                                    rel=1e-5)

    def test_2s_and_2p_one_half_stay_degenerate(self):
        """A Dirac accident the equation must reproduce, not a fitted number:
        levels with the same n and j coincide however different their l."""
        r, V, _ = coulomb(20)
        _, s = solve_radial_log(r, V, 0, 1, kappa=-1, treatment="dirac",
                                atomic_number=20)
        _, p = solve_radial_log(r, V, 1, 0, kappa=1, treatment="dirac",
                                atomic_number=20)
        assert s == pytest.approx(p, rel=1e-5)

    def test_the_spin_orbit_splitting_matches_the_closed_form(self):
        r, V, _ = coulomb(20)
        _, lo = solve_radial_log(r, V, 1, 0, kappa=1, treatment="dirac",
                                 atomic_number=20)
        _, hi = solve_radial_log(r, V, 1, 0, kappa=-2, treatment="dirac",
                                 atomic_number=20)
        exact = (hydrogenic_dirac_energy(20, 2, -2)
                 - hydrogenic_dirac_energy(20, 2, 1))
        assert hi - lo == pytest.approx(exact, rel=1e-3)


class TestTheEquationIsTheRightOne:
    """Guards on ``g`` itself.  The first version of this module differed from
    the correct equation by a constant 1/4 -- an error no convergence study
    can see, because it does not shrink with the grid."""

    def test_the_relativistic_branch_reduces_to_the_non_relativistic_one(self):
        """c -> infinity is the check that caught the 1/4: the relativistic
        1/4 comes from (1+w)^2/4, so carrying it in the centrifugal term as
        well double-counts it."""
        r, V, dx = coulomb(1)
        g_nr, _ = radial_g(r, V, 0, -0.5, relativistic=False, dx=dx)
        g_r, _ = radial_g(r, V, 0, -0.5, kappa=-1, relativistic=True,
                          c=1e8, dx=dx)
        assert np.max(np.abs(g_r - g_nr)) < 1e-8

    def test_the_eigenvalue_reduces_to_the_non_relativistic_one(self):
        r, V, _ = coulomb(1)
        _, eps = solve_radial_log(r, V, 0, 0, kappa=-1, treatment="dirac",
                                  atomic_number=1, c=1e8)
        assert eps == pytest.approx(-0.5, rel=1e-8)

    def test_the_two_written_forms_of_g_agree(self):
        """-w_x/2 and w^2/2 - M_xx/2M are the same term; the docstring gives
        both."""
        from mandacaru.basis.relativity import SPEED_OF_LIGHT, MASS_FLOOR
        r, V, dx = coulomb(8)
        eps, c, k, l = -32.0, SPEED_OF_LIGHT, -1, 0
        M = np.maximum(1.0 + (eps - V) / (2 * c * c), MASS_FLOOR)
        Vx, Vxx = _fourth_order_derivatives(V, dx)
        Mx, Mxx = -Vx / (2 * c * c), -Vxx / (2 * c * c)
        w = Mx / M
        as_wx = l * (l + 1) + k * w + (1 + w) ** 2 / 4 - (Mxx / M - w * w) / 2
        as_mxx = l * (l + 1) + k * w + (1 + w) ** 2 / 4 + w * w / 2 - Mxx / (2 * M)
        assert np.allclose(as_wx, as_mxx)


class TestConvergence:

    def test_the_dirac_1s_converges_at_fourth_order(self):
        """The uniform grid's rate never approaches 2 for this state; that is
        why the module exists."""
        Z = 8
        exact = hydrogenic_dirac_energy(Z, 1, -1)
        errors = []
        for points in (300, 600, 1200):
            r, V, _ = coulomb(Z, points=points)
            _, eps = solve_radial_log(r, V, 0, 0, kappa=-1, treatment="dirac",
                                      atomic_number=Z)
            errors.append(abs(eps - exact))
        rates = [np.log2(a / b) for a, b in zip(errors, errors[1:])]
        assert all(r > 3.5 for r in rates), rates

    def test_analytic_derivatives_are_at_least_as_good(self):
        """V_x, V_xx are exact for a Coulomb tail on x = ln r: V_x = -V,
        V_xx = V.  Supplying them may not hurt."""
        Z = 8
        r, V, dx = coulomb(Z, points=600)
        exact = hydrogenic_dirac_energy(Z, 1, -1)
        _, fd = solve_radial_log(r, V, 0, 0, kappa=-1, treatment="dirac",
                                 atomic_number=Z)
        _, an = solve_radial_log(r, V, 0, 0, kappa=-1, treatment="dirac",
                                 atomic_number=Z, derivatives=(-V, V))
        assert abs(an - exact) <= abs(fd - exact) * 1.05


class TestTheDerivativeStencils:

    def test_they_are_fourth_order_on_a_known_function(self):
        x = np.linspace(0.0, 2.0, 400)
        dx = float(x[1] - x[0])
        f = np.exp(-1.3 * x)
        fx, fxx = _fourth_order_derivatives(f, dx)
        assert np.allclose(fx, -1.3 * f, rtol=1e-8)
        assert np.allclose(fxx, 1.69 * f, rtol=1e-7)

    def test_the_edges_are_not_left_second_order(self):
        """np.gradient's edges were what capped the whole solver at rate 2."""
        x = np.linspace(0.0, 2.0, 400)
        dx = float(x[1] - x[0])
        f = np.exp(-1.3 * x)
        fx, fxx = _fourth_order_derivatives(f, dx)
        for i in (0, 1, -2, -1):
            assert fx[i] == pytest.approx(-1.3 * f[i], rel=1e-8)
            assert fxx[i] == pytest.approx(1.69 * f[i], rel=1e-6)


class TestRefusals:

    def test_a_potential_with_no_bound_state_is_refused_by_name(self):
        r, x, dx = log_grid(20.0, 1.0, points=400)
        with pytest.raises(ValueError, match="no bound state"):
            solve_radial_log(r, np.zeros_like(r), 0, 0, atomic_number=1)


class TestItIsWiredIn:
    """``grid=`` on the relativistic solver and on the atom.

    These pin the *reason* the log grid is not the default as much as the
    option itself: the accurate eigenvalue and the generator-consumable wave
    are different requirements, and the tests that follow are what keeps
    someone from "simplifying" the option away.
    """

    def test_the_relativistic_solver_takes_the_grid(self):
        from mandacaru.basis.relativity import solve_radial_relativistic
        Z, points = 8, 6000
        r = np.linspace(20.0 / Z / points, 20.0 / Z, points)
        V = -Z / r
        exact = hydrogenic_dirac_energy(Z, 1, -1)
        _, uniform = solve_radial_relativistic(
            r, V, 0, 0, kappa=-1, treatment="dirac", atomic_number=Z,
            grid="uniform")
        _, log = solve_radial_relativistic(
            r, V, 0, 0, kappa=-1, treatment="dirac", atomic_number=Z,
            grid="log")
        assert abs(log - exact) < 1e-6 * abs(exact)
        assert abs(log - exact) < 0.01 * abs(uniform - exact)

    def test_the_default_is_still_the_uniform_grid(self):
        """Changing this default silently would change every shipped dataset
        and break the atom-generator agreement; it is a decision, not a knob."""
        import inspect
        from mandacaru.basis.relativity import solve_radial_relativistic
        assert (inspect.signature(solve_radial_relativistic)
                .parameters["grid"].default == "uniform")

    def test_an_unknown_grid_is_refused_by_name(self):
        from mandacaru.basis.relativity import solve_radial_relativistic
        r = np.linspace(1e-3, 20.0, 500)
        with pytest.raises(ValueError, match="grid must be"):
            solve_radial_relativistic(r, -1.0 / r, 0, 0, treatment="scalar",
                                      atomic_number=1.0, grid="spline")

    @pytest.mark.slow
    def test_the_atom_lowers_its_energy_on_either_grid(self):
        """Scalar relativity binds an atom *more*.  Both grids must agree on
        that sign, and each has to be compared against its own
        non-relativistic self -- a shift measured across two discretizations
        is a grid difference wearing a physical name, and came out with the
        wrong sign when this was first wired."""
        from mandacaru.basis.atomic_solver import solve_atom
        for grid in ("uniform", "log"):
            nr = solve_atom(8, relativity="none", grid=grid).total_energy
            sr = solve_atom(8, relativity="scalar", grid=grid).total_energy
            assert sr < nr, (grid, nr, sr)

    @pytest.mark.slow
    def test_the_uniform_grid_overstates_the_relativistic_shift(self):
        """The measurement that says which grid to believe: the uniform grid
        exaggerates oxygen's shift by more than half again."""
        from mandacaru.basis.atomic_solver import solve_atom
        shifts = {}
        for grid in ("uniform", "log"):
            nr = solve_atom(8, relativity="none", grid=grid).total_energy
            sr = solve_atom(8, relativity="scalar", grid=grid).total_energy
            shifts[grid] = sr - nr
        assert shifts["log"] == pytest.approx(-0.0556, abs=2e-3)
        assert shifts["uniform"] < shifts["log"] * 1.4
