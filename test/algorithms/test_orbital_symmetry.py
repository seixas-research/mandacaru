# -*- coding: utf-8 -*-
# file: test/algorithms/test_orbital_symmetry.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Point groups of molecules and the symmetry labels of their orbitals."""

import numpy as np
import pytest
from ase.build import molecule

from mandacaru.algorithms.orbital_symmetry import point_group
from mandacaru.units import BOHR_TO_ANGSTROM


def _group(name):
    atoms = molecule(name)
    return point_group(atoms.numbers, atoms.positions / BOHR_TO_ANGSTROM)


class TestPointGroups:
    @pytest.mark.parametrize("name, group, order", [
        ("H2O", "C2v", 4), ("NH3", "C3v", 6), ("CH4", "Td", 24),
        ("C6H6", "D6h", 24), ("C2H4", "D2h", 8), ("C2H6", "D3d", 12),
        ("BF3", "D3h", 12), ("H2O2", "C2", 2), ("cyclobutane", "D2d", 8),
    ])
    def test_the_textbook_groups(self, name, group, order):
        found = _group(name)
        assert (found.name, found.order) == (group, order)

    def test_a_linear_molecule_uses_its_finite_stand_in(self):
        assert (_group("N2").name, _group("N2").order) == ("Dinfh", 32)
        assert (_group("CO").name, _group("CO").order) == ("Cinfv", 16)

    def test_octahedral_and_atomic_groups(self):
        d = 3.0
        positions = [(0, 0, 0)] + [tuple(d * v) for v in np.vstack(
            [np.eye(3), -np.eye(3)])]
        assert point_group([16] + [9] * 6, positions).name == "Oh"
        atom = point_group([10], [(0, 0, 0)])
        assert (atom.name, atom.order) == ("Kh", 120)

    def test_a_different_basis_breaks_the_equivalence(self):
        positions = [(0, 0, 0), (0, 0, 2.0)]
        assert point_group([1, 1], positions).name == "Dinfh"
        split = point_group([1, 1], positions, signatures=["dz", "tz"])
        assert split.name == "Cinfv"

    def test_the_orientation_does_not_matter(self):
        atoms = molecule("NH3")
        atoms.rotate(37, (1, 2, 3))
        found = point_group(atoms.numbers, atoms.positions / BOHR_TO_ANGSTROM)
        assert found.name == "C3v"


# --------------------------------------------------------------------------- #
# Orbitals, through the calculator (PAW-LCAO: the grid model the labels are
# meant for -- a bare all-electron Gaussian on the grid breaks the symmetry
# far more than any production basis).
# --------------------------------------------------------------------------- #

def _space(name, orbitals, *, rotate=None, method="energy", **options):
    from mandacaru import Mandacaru

    atoms = molecule(name)
    atoms.center(vacuum=3.0)
    if rotate is not None:
        atoms.rotate(*rotate, center="COM")
    atoms.calc = Mandacaru(
        method=options.pop("solver", "adapt-vqe"),
        basis={"name": "PAW-LCAO", "size": "SZ"}, h=0.25,
        active_space={"orbitals": orbitals, "method": method,
                      "symmetry": True},
        max_iterations=1, trace=False, **options)
    atoms.get_potential_energy()
    return atoms.calc.solver._gradient_context["active_space"]


@pytest.fixture(scope="module")
def water():
    return _space("H2O", 6)


@pytest.fixture(scope="module")
def ethylene():
    return _space("C2H4", 7)


class TestOrbitalLabels:
    def test_water_follows_mulliken(self, water):
        # Valence orbitals in canonical order; the molecule lies in the yz
        # plane, so the out-of-plane lone pair (the HOMO) is b1.
        assert water.point_group == "C2v"
        assert water.irreps[:4] == ("1a1", "1b2", "2a1", "1b1")

    def test_ethylene_pi_and_pi_star(self, ethylene):
        assert ethylene.point_group == "D2h"
        assert ethylene.irreps[5] == "1b3u"         # pi, the HOMO
        assert ethylene.irreps[6] == "1b2g"         # pi*, the LUMO

    def test_an_axis_off_the_grid_reads_the_same(self):
        aligned = _space("NH3", 6)
        tilted = _space("NH3", 6, rotate=(41, (1, 2, 3)))
        assert aligned.point_group == tilted.point_group == "C3v"
        assert aligned.irreps == tilted.irreps
        assert "?" not in tilted.irreps


class TestTheSymmetryConstraint:
    """``symmetry=True`` keeps degenerate sets whole."""

    def test_a_split_pair_is_dropped_and_the_slot_left_empty(self):
        # NH3 / SZ: four occupied, then 3a1 and the 2e pair.  Six orbitals
        # would keep one 2e component; with symmetry no whole set fits the
        # last slot, so it stays empty.
        plain = _space_plain("NH3", 6)
        space = _space("NH3", 6)
        assert plain.n_active == 6
        assert space.n_active == 5
        kept = {space.irreps[p] for p in space.active}
        assert "2e" not in kept
        note = dict(space.notes)["symmetry"]
        assert "displaced" in note and "left empty" in note

    def test_a_broken_symmetry_reference_is_refused(self):
        # N2 / SZ at this grid converges to a determinant that occupies one
        # pi* component: no active space can be chosen by its symmetry.
        with pytest.raises(ValueError, match="breaks the molecule's Dinfh"):
            _space("N2", 6)


def _space_plain(name, orbitals):
    from mandacaru import Mandacaru

    atoms = molecule(name)
    atoms.center(vacuum=3.0)
    atoms.calc = Mandacaru(method="adapt-vqe",
                           basis={"name": "PAW-LCAO", "size": "SZ"}, h=0.25,
                           active_space={"orbitals": orbitals},
                           max_iterations=1, trace=False)
    atoms.get_potential_energy()
    return atoms.calc.solver._gradient_context["active_space"]


@pytest.fixture(scope="module")
def water_orbitals():
    """Water / PAW-LCAO-SZ integrals, orbitals and their symmetry provider:
    four occupied (1a1 1b2 2a1 1b1) and two virtual (3a1, 2b2) orbitals."""
    from mandacaru import Mandacaru
    from mandacaru.core.hamiltonian import molecular_orbital_integrals

    atoms = molecule("H2O")
    atoms.center(vacuum=3.0)
    atoms.calc = Mandacaru(method="adapt-vqe",
                           basis={"name": "PAW-LCAO", "size": "SZ"}, h=0.25,
                           active_space={"orbitals": 6, "symmetry": True},
                           max_iterations=1, trace=False)
    atoms.get_potential_energy()
    integrals = atoms.calc.solver._gradient_context["integrals"]
    h_mo, eri_mo, orbitals = molecular_orbital_integrals(integrals, 8,
                                                         (4, 4), None)
    integrals.mo_coefficients = orbitals
    return (np.real(h_mo), np.real(eri_mo), orbitals.shape[1],
            integrals._orbital_symmetry_provider())


def _targeted(water_orbitals, states, **options):
    from dataclasses import replace

    from mandacaru.algorithms.active_space import (resolve_active_space,
                                                   resolve_active_space_spec)

    h_mo, eri_mo, M, provider = water_orbitals
    spec = replace(resolve_active_space_spec({"symmetry": True, **options}),
                   states=states)
    return resolve_active_space(h_mo, eri_mo, n_orbitals=M,
                                num_particles=(4, 4), spec=spec,
                                orbital_symmetry=provider)


class TestTargetStates:
    """``symmetry`` with ``num_states > 1`` keeps orbitals that reach the
    symmetry of the lowest excitations."""

    def test_the_lowest_excitation_is_named_and_reached(self, water_orbitals):
        space = _targeted(water_orbitals, 2, orbitals=5)
        assert "targets 1b1->3a1" in dict(space.notes)["symmetry"]
        assert "3a1" in {space.irreps[p] for p in space.active}

    def test_a_target_brings_in_the_orbital_that_reaches_it(
            self, water_orbitals):
        # MP2 ranks 2b2 first, and b1 x b2 is not B1: the B1 target swaps
        # 3a1 in for it.
        plain = _targeted(water_orbitals, 1, orbitals=5, method="mp2")
        space = _targeted(water_orbitals, 2, orbitals=5, method="mp2")
        assert {plain.irreps[p] for p in plain.active if p >= 4} == {"2b2"}
        assert {space.irreps[p] for p in space.active if p >= 4} == {"3a1"}
        assert "brought in" in dict(space.notes)["symmetry"]

    def test_a_count_too_small_for_every_target_is_refused(
            self, water_orbitals):
        # B1 needs 3a1 and A2 needs 2b2; one virtual slot cannot hold both.
        with pytest.raises(ValueError, match="raise 'orbitals' by 1"):
            _targeted(water_orbitals, 4, orbitals=5)
        wide = _targeted(water_orbitals, 4, orbitals=6)
        assert wide.n_active == 6

    def test_deflation_warns_beyond_the_targeted_states(self):
        from mandacaru import Mandacaru

        atoms = molecule("H2")
        atoms.center(vacuum=3.0)
        atoms.calc = Mandacaru(method="adapt-vqe",
                               basis={"name": "PAW-LCAO", "size": "DZP"},
                               h=0.3, active_space={"orbitals": 2,
                                                    "symmetry": True},
                               max_iterations=2, trace=False)
        atoms.get_potential_energy()
        with pytest.warns(RuntimeWarning, match="ground state only"):
            atoms.calc.energy_levels(2, max_iterations=2)
