# -*- coding: utf-8 -*-
# file: test/algorithms/test_berry_phase.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The Berry-phase polarization of an insulating crystal and the Born
effective charges built on it."""

from types import SimpleNamespace

import numpy as np
import pytest
from ase.build import bulk

from mandacaru import Mandacaru
from mandacaru.algorithms.berry_phase import (_average_phases,
                                              occupied_bands,
                                              polarization_quantum)


class TestTheBookkeeping:
    def test_an_insulator_has_one_count_of_occupied_bands(self):
        result = SimpleNamespace(n_spins=1, occupations=[
            np.array([2.0, 2.0, 0.0]), np.array([2.0, 2.0, 1e-7])])
        assert occupied_bands(result) == [2]
        spin = SimpleNamespace(n_spins=2, occupations=[
            np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])])
        assert occupied_bands(spin) == [2, 1]

    def test_a_metal_and_a_semimetal_are_refused(self):
        metal = SimpleNamespace(n_spins=1, occupations=[
            np.array([2.0, 0.7, 0.0])])
        with pytest.raises(NotImplementedError, match="insulator"):
            occupied_bands(metal)
        semimetal = SimpleNamespace(n_spins=1, occupations=[
            np.array([2.0, 2.0, 0.0]), np.array([2.0, 0.0, 0.0])])
        with pytest.raises(NotImplementedError, match="semimetal"):
            occupied_bands(semimetal)

    def test_the_quantum_is_the_gcd_of_electrons_and_ions(self):
        """Doubly occupied bands move charge 2e at a time, but one ion of
        charge 1 moves e: the polarization is only defined modulo e."""
        assert polarization_quantum(2.0, [1.0, 1.0]) == 1      # LiH
        assert polarization_quantum(2.0, [4.0, 4.0]) == 2      # Si
        assert polarization_quantum(2.0, [3.0, 5.0]) == 1      # AlN
        assert polarization_quantum(1.0, [8.0]) == 1           # spin

    def test_string_phases_are_averaged_on_one_branch(self):
        near_pi = [np.pi - 0.01, -np.pi + 0.03]
        assert _average_phases(near_pi) == pytest.approx(np.pi + 0.01)
        assert _average_phases([0.2, 0.4]) == pytest.approx(0.3)


@pytest.fixture(scope="module")
def lih():
    """Rocksalt LiH: an ionic insulator of two singly charged ions (each
    geometry computed once)."""
    done = {}

    def run(shift=(0.0, 0.0, 0.0), move=None):
        key = (tuple(shift), move)
        if key in done:
            return done[key]
        atoms = bulk("LiH", "rocksalt", a=4.0)
        atoms.positions += np.asarray(shift)
        if move is not None:
            atoms.positions[move[0], move[1]] += move[2]
        atoms.calc = Mandacaru(method="dft", xc="lda", h=0.3, trace=False,
                               basis={"name": "PAW-LCAO", "size": "SZ"},
                               kpts={"size": (3, 3, 3), "gamma": True},
                               smearing={"method": "fermi-dirac",
                                         "width": 0.001})
        atoms.get_potential_energy()
        atoms.calc.get_polarization()
        done[key] = atoms.calc.polarization_result
        return done[key]
    return run


class TestTheCrystal:
    def test_a_centrosymmetric_crystal_sits_on_zero_or_half(self, lih):
        result = lih()
        assert np.all(np.isclose(np.abs(result.fractional), 0.5, atol=1e-6)
                      | np.isclose(result.fractional, 0.0, atol=1e-6))

    def test_a_rigid_shift_changes_nothing_but_the_branch(self, lih):
        """Every atom and the electrons move together: the raw coordinates
        change by whole quanta."""
        change = lih(shift=(0.13, 0.07, 0.21)).raw - lih().raw
        assert np.allclose(change, np.round(change), atol=1e-6)

    @pytest.mark.slow
    def test_the_born_charge_of_an_ion(self, lih):
        """Moving H- by +-0.02 Angstrom along x: Z*_xx near -1 (an ionic
        crystal of charges +-1), the off-diagonal zero by symmetry."""
        from mandacaru.algorithms.berry_phase import E_PER_BOHR2_IN_C_PER_M2
        from mandacaru.units import ANGSTROM_TO_BOHR
        plus = lih(move=(1, 0, 0.02))
        minus = lih(move=(1, 0, -0.02))
        change = plus.raw - minus.raw
        change -= np.round(change)
        # A quantum's row is g e a_j / Omega: times Omega / e it is g a_j.
        volume = bulk("LiH", "rocksalt", a=4.0).get_volume() \
            * ANGSTROM_TO_BOHR ** 3
        lattice_steps = plus.quanta / E_PER_BOHR2_IN_C_PER_M2 * volume
        Z = (change @ lattice_steps) / (0.04 * ANGSTROM_TO_BOHR)
        assert -1.3 < Z[0] < -0.8
        assert abs(Z[1]) < 0.02 and abs(Z[2]) < 0.02


class TestThePiezoelectricTensor:
    @pytest.mark.slow
    def test_a_centrosymmetric_crystal_is_not_piezoelectric(self):
        """Inversion maps P to -P and a strain to itself: rocksalt LiH's
        proper tensor vanishes (one shear column, two runs)."""
        from mandacaru.algorithms import piezoelectric_tensor
        atoms = bulk("LiH", "rocksalt", a=4.0)
        result = piezoelectric_tensor(
            atoms, components=(3,), method="dft", xc="lda", h=0.3,
            basis={"name": "PAW-LCAO", "size": "SZ"}, trace=False,
            kpts={"size": (3, 3, 3), "gamma": True},
            smearing={"method": "fermi-dirac", "width": 0.001})
        assert np.abs(result.tensor).max() < 1e-3
        assert not result.relaxed

    def test_the_relaxation_keeps_the_centroid(self):
        """The relaxed-ion strains relax under a constraint that removes the
        mean force (a crystal's exact forces sum to zero; the grid's egg-box
        does not) and keeps the centroid, also on the calculator's copy."""
        from mandacaru.algorithms.berry_phase import _FixedCentroid
        atoms = bulk("AlN", "wurtzite", a=3.11, c=4.98, u=0.382)
        atoms.set_constraint(_FixedCentroid())
        forces = np.array([[0.0, 0.0, -0.12], [0.0, 0.0, 0.06],
                           [0.0, 0.0, -0.12], [0.0, 0.0, 0.06]])
        adjusted = forces.copy()
        atoms.constraints[0].adjust_forces(atoms, adjusted)
        assert np.allclose(adjusted.sum(axis=0), 0.0)
        assert np.allclose(adjusted - forces, -forces.mean(axis=0))
        centroid = atoms.positions.mean(axis=0)
        moved = atoms.positions + [0.1, -0.2, 0.3]
        moved[1, 2] -= 0.02
        atoms.set_positions(moved)
        assert np.allclose(atoms.positions.mean(axis=0), centroid)
        assert np.isclose(atoms.positions[1, 2] - atoms.positions[0, 2],
                          moved[1, 2] - moved[0, 2])
        assert isinstance(atoms.copy().constraints[0], _FixedCentroid)

    def test_bad_requests_are_refused(self):
        from mandacaru.algorithms import piezoelectric_tensor
        atoms = bulk("LiH", "rocksalt", a=4.0)
        with pytest.raises(ValueError, match="Voigt"):
            piezoelectric_tensor(atoms, components=(7,))
        with pytest.raises(ValueError, match="positive"):
            piezoelectric_tensor(atoms, strain=0.0)
        atoms.pbc = (True, True, False)
        with pytest.raises(NotImplementedError, match="three directions"):
            piezoelectric_tensor(atoms)


class TestTheFiniteField:
    """A finite electric field coupled through the Berry phase (the full
    mesh, the field term rebuilt from each iteration's states)."""

    OPTIONS = dict(method="dft", xc="lda", h=0.35, trace=False,
                   basis={"name": "PAW-LCAO", "size": "SZ"},
                   kpts={"size": (2, 2, 2), "gamma": True},
                   smearing={"method": "fermi-dirac", "width": 0.001})

    def test_no_field_on_the_full_mesh_is_the_reduced_crystal(self):
        energies = []
        for field in (None, (0.0, 0.0, 0.0)):
            atoms = bulk("LiH", "rocksalt", a=4.0)
            atoms.calc = Mandacaru(electric_field=field, **self.OPTIONS)
            energies.append(atoms.get_potential_energy())
        assert energies[1] == pytest.approx(energies[0], abs=1e-6)

    @pytest.mark.slow
    def test_the_response_is_the_energy_curvature(self):
        """The field operator is the functional's gradient: chi from the
        polarization equals -d2E/dF2 / Omega.  A cubic crystal's tensor is
        diagonal and isotropic, to the anisotropy of a 2^3 Gamma-centered
        mesh in the fcc primitive basis (0.7 % off-diagonal; 0.07 % at
        3^3)."""
        from mandacaru.algorithms import dielectric_tensor
        result = dielectric_tensor(bulk("LiH", "rocksalt", a=4.0),
                                   **self.OPTIONS)
        chi = result.susceptibility
        assert np.allclose(result.energy_diagonal, np.diag(chi), rtol=1e-3)
        assert np.abs(chi - np.diag(np.diag(chi))).max() < 2e-2 * chi.max()
        assert np.allclose(np.diag(chi), chi[0, 0], rtol=1e-3)
        assert result.tensor[0, 0] > 1.0

    def test_bad_requests_are_refused(self):
        from ase.build import molecule
        from mandacaru.algorithms import dielectric_tensor
        with pytest.raises(NotImplementedError, match="polarizability"):
            dielectric_tensor(molecule("H2O"), **self.OPTIONS)
        with pytest.raises(ValueError, match="applies the fields itself"):
            dielectric_tensor(bulk("LiH", "rocksalt", a=4.0),
                              electric_field=(0, 0, 1e-3), **self.OPTIONS)
        with pytest.raises(ValueError, match="positive"):
            dielectric_tensor(bulk("LiH", "rocksalt", a=4.0), field=0.0,
                              **self.OPTIONS)

    @pytest.mark.slow
    def test_what_a_field_cannot_do_is_refused(self):
        atoms = bulk("LiH", "rocksalt", a=4.0)
        atoms.set_initial_magnetic_moments([0.5, 0.0])
        atoms.calc = Mandacaru(electric_field=(0, 0, 1e-4), **self.OPTIONS)
        with pytest.raises(NotImplementedError, match="spin-restricted"):
            atoms.get_potential_energy()
        atoms = bulk("LiH", "rocksalt", a=4.0)
        atoms.calc = Mandacaru(electric_field=(0, 0, 0.5), **self.OPTIONS)
        # Past the Zener limit the occupied manifolds stop overlapping (or
        # the SCF runs away first): either way there is no state to report.
        with pytest.raises(RuntimeError, match="Zener|did not converge"):
            atoms.get_potential_energy()
