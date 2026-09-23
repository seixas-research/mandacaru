# -*- coding: utf-8 -*-
# file: test/basis/test_relativity.py

"""Scalar-relativistic and Dirac radial atoms.

The oracle is the **closed-form** Dirac spectrum of a one-electron ion, so
most of this file needs no self-consistent field at all and costs
milliseconds.  The two facts worth stating up front, because they are what the
implementation rests on:

* for ``l = 0`` there is one ``j``, so ``kappa = -1`` is its *physical* value
  and the scalar-relativistic equation **is** the Dirac equation there;
* a point-nucleus Dirac s or p(1/2) state behaves like ``r**gamma`` with
  ``gamma = sqrt(kappa^2 - (Z alpha)^2) < 1``, which a uniform grid cannot
  converge at second order.  That is a property of the problem, not a defect
  of the solver, and the test that pins it contrasts the rate against a state
  with no cusp.
"""

import numpy as np
import pytest

from mandacaru.basis.atomic_solver import solve_atom, solve_radial
from mandacaru.basis.relativity import (SPEED_OF_LIGHT, degeneracy,
                                        hydrogenic_dirac_energy, j_average,
                                        j_of, kappa_of, kappa_values, l_of,
                                        solve_radial_relativistic,
                                        spin_orbit_difference)

HARTREE_EV = 27.211386245988


def grid(points=40000, r_max=40.0):
    return np.arange(1, points + 1) * (r_max / (points + 1))


class TestAngularBookkeeping:
    @pytest.mark.parametrize("l", [0, 1, 2, 3])
    def test_kappa_round_trips_through_l_and_j(self, l):
        for kappa in kappa_values(l):
            assert l_of(kappa) == l
            assert kappa_of(l, j_of(kappa)) == kappa

    @pytest.mark.parametrize("l", [1, 2, 3])
    def test_the_two_j_hold_2l_plus_2_and_2l_electrons(self, l):
        down, up = kappa_values(l)
        assert degeneracy(up) == 2 * l + 2
        assert degeneracy(down) == 2 * l
        assert degeneracy(up) + degeneracy(down) == 2 * (2 * l + 1)

    def test_an_s_shell_has_one_j(self):
        assert kappa_values(0) == [-1]
        assert degeneracy(-1) == 2

    @pytest.mark.parametrize("l", [1, 2, 3])
    def test_the_average_and_difference_invert_the_two_j(self, l):
        """``avg + <L.S> so`` must return each j exactly -- it is a 2x2 solve."""
        values = {kappa_values(l)[0]: -0.40, kappa_values(l)[1]: -0.37}
        avg = j_average(values, l)
        so = spin_orbit_difference(values, l)
        assert avg + 0.5 * l * so == pytest.approx(values[-(l + 1)], abs=1e-14)
        assert avg - 0.5 * (l + 1) * so == pytest.approx(values[l], abs=1e-14)

    def test_an_s_shell_has_no_spin_orbit_term(self):
        assert spin_orbit_difference({-1: -0.5}, 0) == 0.0


class TestAgainstTheClosedFormDiracSpectrum:
    """The one place the implementation meets an answer it did not produce."""

    @pytest.mark.parametrize("Z, n, l, kappa", [
        (1, 2, 1, 1), (1, 2, 1, -2),
        (8, 2, 1, 1), (8, 2, 1, -2), (8, 3, 2, 2), (8, 3, 2, -3),
        (26, 3, 2, 2), (26, 3, 2, -3),
    ])
    def test_smooth_states_reproduce_it(self, Z, n, l, kappa):
        """Away from the r**gamma cusp the solver is good to 1e-5 relative.

        Every s state is excluded on purpose: ``kappa = -1`` gives
        ``gamma < 1`` for any Z, so an s state always carries the cusp --
        even hydrogen's, which lands at 5e-5 rather than 1e-5.  Those are
        the business of ``TestTheUniformGridCusp``.
        """
        r = grid()
        _P, eps = solve_radial_relativistic(r, -Z / r, l, n - l - 1,
                                            kappa=kappa, treatment="dirac",
                                            atomic_number=Z)
        exact = hydrogenic_dirac_energy(Z, n, kappa)
        assert eps == pytest.approx(exact, rel=1e-5)

    def test_the_dirac_degeneracy_holds(self):
        """2s(1/2) and 2p(1/2) share an energy: Dirac depends on n and j only."""
        assert hydrogenic_dirac_energy(20, 2, -1) == pytest.approx(
            hydrogenic_dirac_energy(20, 2, 1), rel=1e-14)

    def test_it_reduces_to_the_schrodinger_spectrum_as_c_grows(self):
        for n, kappa in ((1, -1), (2, -2)):
            far = hydrogenic_dirac_energy(10, n, kappa, c=1e7)
            assert far == pytest.approx(-100.0 / (2 * n * n), rel=1e-10)

    def test_a_point_nucleus_has_no_bound_state_past_z_equals_c(self):
        with pytest.raises(ValueError, match="no bound state"):
            hydrogenic_dirac_energy(140, 1, -1)


class TestTheUniformGridCusp:
    """Why the s states are the inaccurate ones, stated as a convergence rate."""

    def _error(self, Z, n, l, kappa, points):
        r = grid(points)
        _P, eps = solve_radial_relativistic(r, -Z / r, l, n - l - 1,
                                            kappa=kappa, treatment="dirac",
                                            atomic_number=Z)
        exact = hydrogenic_dirac_energy(Z, n, kappa)
        return abs(eps - exact) / abs(exact)

    def test_a_state_without_a_cusp_converges_at_second_order(self):
        """Au 2p(3/2): gamma = 1.92, so the solution is smooth and the rate is 2."""
        coarse = self._error(79, 2, 1, -2, 15000)
        fine = self._error(79, 2, 1, -2, 30000)
        assert np.log2(coarse / fine) == pytest.approx(2.0, abs=0.15)

    def test_a_cusped_state_does_not(self):
        """Au 1s: gamma = 0.82, and no uniform grid recovers second order."""
        coarse = self._error(79, 1, 0, -1, 15000)
        fine = self._error(79, 1, 0, -1, 30000)
        assert np.log2(coarse / fine) < 1.0
        # ... and it is still the right ballpark, not nonsense.
        assert fine < 0.05


class TestTheThreeTheoriesAgree:
    def test_no_relativity_reproduces_the_plain_solver(self):
        r = grid(4000, 25.0)
        V = -8.0 / r
        for l, nodes in ((0, 0), (0, 1), (1, 0)):
            u_ref, e_ref = solve_radial(r, V, l, nodes)
            u, e = solve_radial_relativistic(r, V, l, nodes, treatment="none")
            assert e == pytest.approx(e_ref, rel=1e-14)
            assert np.allclose(np.abs(u), np.abs(u_ref), atol=1e-12)

    def test_scalar_is_exactly_dirac_for_an_s_state(self):
        """l = 0 has one j, and kappa = -1 is it."""
        r = grid(20000)
        V = -20.0 / r
        _p, scalar = solve_radial_relativistic(r, V, 0, 0, treatment="scalar",
                                               atomic_number=20)
        _q, dirac = solve_radial_relativistic(r, V, 0, 0, kappa=-1,
                                              treatment="dirac",
                                              atomic_number=20)
        assert scalar == pytest.approx(dirac, rel=1e-13)

    def test_scalar_sits_between_the_two_j(self):
        r = grid(20000)
        V = -30.0 / r
        levels = {k: solve_radial_relativistic(r, V, 1, 0, kappa=k,
                                               treatment="dirac",
                                               atomic_number=30)[1]
                  for k in kappa_values(1)}
        _p, scalar = solve_radial_relativistic(r, V, 1, 0, treatment="scalar",
                                               atomic_number=30)
        assert min(levels.values()) < scalar < max(levels.values())
        # The scalar equation reproduces the j average only to O(c^-4): the
        # averaging of kappa is exact, but M itself carries the energy of the
        # level being solved for, and the two j have different energies.
        assert scalar == pytest.approx(j_average(levels, 1), rel=1e-3)

    def test_relativity_lowers_a_core_level(self):
        r = grid(20000)
        V = -40.0 / r
        _a, plain = solve_radial_relativistic(r, V, 0, 0, treatment="none")
        _b, rel = solve_radial_relativistic(r, V, 0, 0, treatment="scalar",
                                            atomic_number=40)
        assert rel < plain
        # -Z^2/2 up to the grid: a Z = 40 1s orbital has scale 0.025 Bohr and
        # this grid's spacing is 0.002, which is worth about 0.2 %.
        assert plain == pytest.approx(-800.0, rel=1e-2)


class TestSpinOrbitSplittingsOfRealAtoms:
    """The only SCF in this file: two atoms, coarse grids, against experiment."""

    @pytest.mark.parametrize("Z, n, l, measured_ev, tolerance", [
        (18, 3, 1, 0.178, 0.35),      # argon 3p
    ])
    def test_it_lands_near_the_measured_fine_structure(self, Z, n, l,
                                                       measured_ev, tolerance):
        atom = solve_atom(Z, points=4000, r_max=20.0, relativity="dirac",
                          tolerance=1e-5, mixing=0.3)
        splitting = atom.spin_orbit_splitting(n, l) * HARTREE_EV
        assert splitting > 0                       # j = l+1/2 lies above
        assert splitting == pytest.approx(measured_ev, rel=tolerance)

    def test_an_s_shell_does_not_split(self):
        atom = solve_atom(10, points=3000, r_max=18.0, relativity="dirac",
                          tolerance=1e-5, mixing=0.3)
        assert atom.spin_orbit_splitting(1, 0) == 0.0
        assert atom.spin_orbit_splitting(2, 0) == 0.0

    def test_a_scalar_atom_reports_no_splitting_because_it_has_no_term(self):
        atom = solve_atom(10, points=3000, r_max=18.0, relativity="scalar",
                          tolerance=1e-5, mixing=0.3)
        assert not atom.eigenvalues_j
        assert atom.spin_orbit_splitting(2, 1) == 0.0


class TestTheSelfConsistentAtom:
    def test_the_defaults_are_the_pre_relativistic_atom_bit_for_bit(self):
        """Nothing that did not ask for relativity may move."""
        plain = solve_atom(8, points=2500, r_max=18.0, tolerance=1e-5)
        explicit = solve_atom(8, points=2500, r_max=18.0, tolerance=1e-5,
                              xc="lda", relativity="none")
        assert plain.total_energy == explicit.total_energy
        for key in plain.eigenvalues:
            assert plain.eigenvalues[key] == explicit.eigenvalues[key]

    def test_a_dirac_atom_fills_both_the_j_and_the_averaged_tables(self):
        atom = solve_atom(10, points=3000, r_max=18.0, relativity="dirac",
                          tolerance=1e-5, mixing=0.3)
        assert atom.eigenvalues_j and atom.occupations_j
        # The (n, l) entry is the (2j+1) average of the j-resolved ones.
        levels = {k: atom.eigenvalues_j[(2, 1, k)] for k in kappa_values(1)}
        assert atom.eigenvalues[(2, 1)] == pytest.approx(j_average(levels, 1),
                                                         rel=1e-12)

    def test_the_j_occupations_add_up_to_the_subshell(self):
        atom = solve_atom(10, points=3000, r_max=18.0, relativity="dirac",
                          tolerance=1e-5, mixing=0.3)
        for (n, l), total in atom.occupations.items():
            shared = sum(atom.occupations_j[(n, l, k)]
                         for k in kappa_values(l))
            assert shared == pytest.approx(total)

    def test_the_core_valence_partition_conserves_charge(self):
        atom = solve_atom(8, points=2500, r_max=18.0, tolerance=1e-5)
        core, valence = atom.partition_density([(2, 0), (2, 1)])
        shell = 4.0 * np.pi * atom.r * atom.r
        assert np.trapezoid(core * shell, atom.r) == pytest.approx(2.0,
                                                                   abs=1e-6)
        assert np.trapezoid(valence * shell, atom.r) == pytest.approx(6.0,
                                                                      abs=1e-6)


class TestRejections:
    def test_dirac_needs_a_kappa(self):
        r = grid(1000, 10.0)
        with pytest.raises(ValueError, match="needs kappa"):
            solve_radial_relativistic(r, -1.0 / r, 1, 0, treatment="dirac")

    def test_an_unknown_treatment_is_named(self):
        r = grid(1000, 10.0)
        with pytest.raises(ValueError, match="unknown relativistic treatment"):
            solve_radial_relativistic(r, -1.0 / r, 0, 0, treatment="zora")

    def test_kappa_zero_is_not_a_quantum_number(self):
        with pytest.raises(ValueError, match="not a Dirac quantum number"):
            l_of(0)

    def test_j_must_be_l_plus_or_minus_a_half(self):
        with pytest.raises(ValueError, match="j must be"):
            kappa_of(1, 2.5)

    def test_the_speed_of_light_is_the_inverse_fine_structure_constant(self):
        assert SPEED_OF_LIGHT == pytest.approx(137.035999177, abs=1e-6)
