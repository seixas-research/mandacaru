# -*- coding: utf-8 -*-
# file: test_wavefunction.py

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from carcara.wavefunction import Wavefunction


# --- Fixtures ---

@pytest.fixture
def h_atom_file(tmp_path):
    """Single hydrogen atom at the Cartesian origin."""
    atoms = Atoms('H', positions=[[0.0, 0.0, 0.0]])
    filepath = str(tmp_path / "h_atom.xyz")
    write(filepath, atoms)
    return filepath


@pytest.fixture
def h2_file(tmp_path):
    """H2 molecule along the x-axis with bond length 0.74 Angstrom."""
    atoms = Atoms('H2', positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    filepath = str(tmp_path / "h2.xyz")
    write(filepath, atoms)
    return filepath


@pytest.fixture
def wf_h(h_atom_file):
    return Wavefunction(h_atom_file, atom_index=0)


@pytest.fixture
def wf_h2(h2_file):
    return Wavefunction(h2_file, atom_index=0)


# --- Constructor ---

class TestConstructor:
    def test_single_atom_count(self, wf_h):
        assert wf_h.n_atoms == 1

    def test_single_atom_Z(self, wf_h):
        assert wf_h.Z == 1

    def test_single_atom_symbol(self, wf_h):
        assert wf_h.all_symbols[0] == 'H'

    def test_single_atom_positions_shape(self, wf_h):
        assert wf_h.all_positions_bohr.shape == (1, 3)

    def test_single_atom_origin_at_zero(self, wf_h):
        np.testing.assert_allclose(wf_h.origin_cart, [0.0, 0.0, 0.0], atol=1e-10)

    def test_h2_atom_count(self, wf_h2):
        assert wf_h2.n_atoms == 2

    def test_h2_all_numbers(self, wf_h2):
        assert list(wf_h2.all_numbers) == [1, 1]

    def test_h2_positions_shape(self, wf_h2):
        assert wf_h2.all_positions_bohr.shape == (2, 3)

    def test_h2_bond_length_in_bohr(self, wf_h2):
        # 0.74 Angstrom in Bohr
        expected_bond = 0.74 * 1.8897259886
        R = wf_h2.all_positions_bohr
        bond = np.linalg.norm(R[1] - R[0])
        assert abs(bond - expected_bond) < 1e-6

    def test_atom_index_stored(self, wf_h2):
        assert wf_h2.atom_index == 0

    def test_repr_contains_symbol(self, wf_h):
        assert 'H' in repr(wf_h)


# --- Coordinate conversion ---

class TestCoordinateConversion:
    def test_sph_to_cart_along_z(self):
        sph = np.array([3.0, 0.0, 0.0])  # r=3, theta=0 -> (0, 0, 3)
        cart = Wavefunction.spherical_to_cartesian(sph)
        np.testing.assert_allclose(cart, [0.0, 0.0, 3.0], atol=1e-10)

    def test_sph_to_cart_along_x(self):
        sph = np.array([2.0, np.pi / 2, 0.0])  # r=2, equator, phi=0 -> (2, 0, 0)
        cart = Wavefunction.spherical_to_cartesian(sph)
        np.testing.assert_allclose(cart, [2.0, 0.0, 0.0], atol=1e-10)

    def test_roundtrip_cartesian(self):
        cart = np.array([1.0, 2.0, 3.0])
        back = Wavefunction.spherical_to_cartesian(Wavefunction.cartesian_to_spherical(cart))
        np.testing.assert_allclose(back, cart, atol=1e-10)

    def test_roundtrip_spherical(self):
        sph = np.array([4.0, 1.1, 0.7])
        back = Wavefunction.cartesian_to_spherical(Wavefunction.spherical_to_cartesian(sph))
        np.testing.assert_allclose(back, sph, atol=1e-10)


# --- Wavefunction evaluation ---

class TestWavefunctionEvaluation:
    def test_ground_state_at_origin(self, wf_h):
        # psi_1s(r=0) = 1 / sqrt(pi) for Z=1
        X = np.array([0.0])
        Y = np.array([0.0])
        Z = np.array([0.0])
        psi = wf_h._psi_on_cart_grid([1, 0, 0], wf_h.origin_cart, X, Y, Z)
        expected = 1.0 / np.sqrt(np.pi)
        assert abs(np.real(psi[0]) - expected) < 1e-4

    def test_wavefunction_decays_with_r(self, wf_h):
        X = np.array([1.0, 3.0, 6.0])
        Y = np.zeros(3)
        Z = np.zeros(3)
        psi = wf_h._psi_on_cart_grid([1, 0, 0], wf_h.origin_cart, X, Y, Z)
        magnitudes = np.abs(psi)
        assert magnitudes[0] > magnitudes[1] > magnitudes[2]

    def test_wavefunction_real_for_m0(self, wf_h):
        X = np.array([1.0])
        Y = np.array([0.5])
        Z = np.array([0.5])
        psi = wf_h._psi_on_cart_grid([1, 0, 0], wf_h.origin_cart, X, Y, Z)
        # m=0 spherical harmonic Y_00 is real -> psi is real
        assert abs(np.imag(psi[0])) < 1e-10


# --- Coulomb potential ---

class TestCoulombPotential:
    def test_potential_is_negative(self, wf_h):
        pos = np.array([1.0, np.pi / 2, 0.0])  # [r, theta, phi]
        V = wf_h.coulomb_potential(pos)
        assert V < 0

    def test_potential_at_unit_radius(self, wf_h):
        # V = -Z/r = -1 at r=1 for Z=1
        pos = np.array([1.0, np.pi / 2, 0.0])
        V = wf_h.coulomb_potential(pos)
        assert abs(V - (-1.0)) < 1e-8

    def test_potential_scales_with_Z(self, h2_file):
        # Use H atom; if Z were 2, V = -2/r
        wf = Wavefunction(h2_file, atom_index=0)
        pos = np.array([2.0, np.pi / 2, 0.0])
        V = wf.coulomb_potential(pos)
        assert abs(V - (-1.0 / 2.0)) < 1e-8

    def test_potential_decreases_with_r(self, wf_h):
        pos1 = np.array([1.0, np.pi / 2, 0.0])
        pos2 = np.array([2.0, np.pi / 2, 0.0])
        assert wf_h.coulomb_potential(pos1) < wf_h.coulomb_potential(pos2)


# --- One-body integrals ---

class TestOneBodyIntegral:
    def test_returns_required_keys(self, wf_h):
        result = wf_h.one_body_integral([1, 0, 0], [1, 0, 0])
        assert {'kinetic', 'potential', 'total'} == set(result.keys())

    def test_kinetic_is_positive(self, wf_h):
        result = wf_h.one_body_integral([1, 0, 0], [1, 0, 0])
        assert result['kinetic'] > 0

    def test_potential_is_negative(self, wf_h):
        result = wf_h.one_body_integral([1, 0, 0], [1, 0, 0])
        assert result['potential'] < 0

    def test_total_is_negative(self, wf_h):
        result = wf_h.one_body_integral([1, 0, 0], [1, 0, 0])
        assert result['total'] < 0

    def test_total_equals_sum(self, wf_h):
        result = wf_h.one_body_integral([1, 0, 0], [1, 0, 0])
        assert abs(result['total'] - (result['kinetic'] + result['potential'])) < 1e-12

    def test_kinetic_approximate_value(self, wf_h):
        # Exact <T>_1s = Z^2/2 = 0.5 a.u. (virial theorem)
        result = wf_h.one_body_integral([1, 0, 0], [1, 0, 0], box_size=10, points=60)
        assert abs(result['kinetic'] - 0.5) < 0.15

    def test_potential_approximate_value(self, wf_h):
        # Exact <V>_1s = -Z^2 = -1.0 a.u. (virial theorem)
        result = wf_h.one_body_integral([1, 0, 0], [1, 0, 0], box_size=10, points=60)
        assert abs(result['potential'] - (-1.0)) < 0.1

    def test_total_approximate_value(self, wf_h):
        # Exact E_1s = -Z^2/2 = -0.5 a.u.
        result = wf_h.one_body_integral([1, 0, 0], [1, 0, 0], box_size=10, points=60)
        assert abs(result['total'] - (-0.5)) < 0.2

    def test_results_are_floats(self, wf_h):
        result = wf_h.one_body_integral([1, 0, 0], [1, 0, 0])
        assert isinstance(result['kinetic'], float)
        assert isinstance(result['potential'], float)
        assert isinstance(result['total'], float)

    def test_h2_potential_more_negative_than_h(self, wf_h, wf_h2):
        # Two nuclei pull the electron more strongly than one
        r_h = wf_h.one_body_integral([1, 0, 0], [1, 0, 0], box_size=10, points=40)
        r_h2 = wf_h2.one_body_integral([1, 0, 0], [1, 0, 0], box_size=10, points=40)
        assert r_h2['potential'] < r_h['potential']
