# -*- coding: utf-8 -*-
# file: test_nao.py

import numpy as np
import pytest

from carcara.basis import NumericalAtomicOrbital, energy_shift_to_rc
from carcara.basis.nao import solve_confined_radial
from carcara.integrals import Grid
from carcara.units import EV_TO_HARTREE


class TestEnergyShift:
    def test_matches_particle_in_sphere_formula(self):
        # r_c = pi / sqrt(2 dE) in atomic units.
        rc = energy_shift_to_rc(0.03)
        dE = 0.03 * EV_TO_HARTREE
        assert np.isclose(rc, np.pi / np.sqrt(2.0 * dE))

    def test_smaller_shift_gives_larger_radius(self):
        assert energy_shift_to_rc(0.01) > energy_shift_to_rc(0.10)

    def test_nonpositive_shift_raises(self):
        with pytest.raises(ValueError):
            energy_shift_to_rc(0.0)


class TestConfinedRadial:
    def test_large_box_recovers_hydrogen_1s(self):
        # A weakly confined 1s approaches the free H 1s: E -> -0.5 Ha.
        r, R, E = solve_confined_radial(1, 0, Z=1.0, r_c=40.0, n_grid=4000)
        assert abs(E - (-0.5)) < 0.01
        assert np.isclose(np.trapezoid(R * R * r * r, r), 1.0, atol=1e-3)

    def test_hydrogen_2s_energy_and_nodes(self):
        r, R, E = solve_confined_radial(2, 0, Z=1.0, r_c=40.0, n_grid=4000)
        assert abs(E - (-0.125)) < 0.01           # H 2s = -1/8 Ha
        interior = R[1:-1]
        nodes = np.sum(np.diff(np.sign(interior)) != 0)
        assert nodes == 1                          # n - l - 1 = 1 radial node

    def test_boundary_is_exactly_zero(self):
        r, R, E = solve_confined_radial(1, 0, Z=1.0, r_c=8.0, n_grid=2000)
        assert R[-1] == 0.0 and r[-1] == 8.0

    def test_confinement_raises_energy(self):
        _, _, e_tight = solve_confined_radial(1, 0, Z=1.0, r_c=2.0, n_grid=2000)
        _, _, e_loose = solve_confined_radial(1, 0, Z=1.0, r_c=20.0, n_grid=2000)
        assert e_tight > e_loose


class TestNumericalAtomicOrbital:
    def test_zero_outside_cutoff(self):
        orb = NumericalAtomicOrbital(1, 0, 0, Z=1.0, r_c=6.0, units="bohr")
        assert abs(complex(orb.evaluate(3.0, 0.0, 0.0))) > 0.0
        assert complex(orb.evaluate(6.5, 0.0, 0.0)) == 0.0
        assert complex(orb.evaluate(10.0, 0.0, 0.0)) == 0.0

    def test_normalized_on_a_3d_grid(self):
        orb = NumericalAtomicOrbital(1, 0, 0, Z=1.0, r_c=6.0, units="bohr")
        # Even node count keeps the -1/r-free sampling smooth; box > r_c.
        grid = Grid(center=[0, 0, 0], box_size=8.0, h=0.2, units="bohr")
        psi = orb.evaluate(grid.X, grid.Y, grid.Z)
        norm = float(np.sum(np.abs(psi) ** 2) * grid.dV)
        assert abs(norm - 1.0) < 0.02

    def test_energy_shift_sets_cutoff(self):
        orb = NumericalAtomicOrbital(1, 0, 0, Z=1.0, energy_shift=0.5)
        assert np.isclose(orb.r_c, energy_shift_to_rc(0.5))

    def test_is_basis_function(self):
        from carcara.basis import BasisFunction
        orb = NumericalAtomicOrbital(1, 0, 0, Z=1.0, r_c=6.0, units="bohr")
        assert isinstance(orb, BasisFunction)
        assert orb.state == (1, 0, 0)
