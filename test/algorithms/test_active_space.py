# -*- coding: utf-8 -*-
# file: test/algorithms/test_active_space.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Virtual-orbital truncation: the partition, the selectors, and what it costs.

The partition is bookkeeping and is tested as such.  The claim that makes the
feature worth having is physical -- that ranking the virtuals by the second-order
density recovers far more correlation, at a fixed register width, than ranking
them by orbital energy -- and it is measured here against exact diagonalization
in the particle-number sector.

A note on the numbers below.  These fixtures put a Gaussian basis on a coarse
real-space grid (``h = 0.3``), which under-resolves a compact core badly -- for
LiH the Li 1s resolution ratio is ~0.63, so absolute Hartree-Fock and full-CI
energies here are Hartree away from published 6-31G values.  That is harmless
for what is being tested, because every comparison is made between two
calculations sharing the same integrals, and it is a pre-existing property of
building a Gaussian basis on a grid rather than anything to do with the active
space.  It does mean **no absolute energy in this file is literature-comparable**
and none should be quoted as one.
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru.algorithms.active_space import (ACTIVE_SPACE_METHODS,
                                               correlating_partners,
                                               DEFAULT_OCCUPATION_THRESHOLD,
                                               ActiveSpace,
                                               normalize_orbitals,
                                               resolve_active_space,
                                               resolve_active_space_spec,
                                               resolve_method,
                                               resolve_threshold)
from mandacaru.core import MolecularIntegrals
from mandacaru.core.hamiltonian import molecular_orbital_integrals
from mandacaru.core.sector import ParticleSector
from mandacaru.integrals import Grid


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #


def spec(**options):
    """The resolved ``active_space`` option, as the driver hands it on."""
    return resolve_active_space_spec(options)

def _integrals(formula, positions, basis, box=8.0, h=0.3):
    from mandacaru.basis import BasisSet

    atoms = Atoms(formula, positions=positions)
    coordinates = atoms.get_positions()
    bset = BasisSet.build(basis)
    functions, nuclei = [], []
    for Z, symbol, position in zip(atoms.get_atomic_numbers(),
                                   atoms.get_chemical_symbols(), coordinates):
        functions += bset.atom(symbol, center=position, units="angstrom")
        nuclei.append((float(Z), position))
    grid = Grid(center=coordinates.mean(axis=0), box_size=box, h=h,
                units="angstrom")
    return MolecularIntegrals(nuclei, functions, grid,
                              softening=0.5 * min(grid.dx, grid.dy, grid.dz))


def _ground_state(hamiltonian, num_particles):
    """Lowest eigenvalue in the ``num_particles`` sector -- exact for these sizes."""
    n_modes = hamiltonian.n_modes()
    sector = ParticleSector(n_modes, num_particles, "jordan_wigner")
    restricted = sector.restrict(
        hamiltonian.map_to_qubits("jordan_wigner", n_modes=n_modes))
    return float(np.min(np.linalg.eigvalsh(restricted.toarray())))


@pytest.fixture(scope="module")
def h2():
    return _integrals("H2", [(0, 0, 0), (0, 0, 0.74)], "6-31G")


@pytest.fixture(scope="module")
def h2_mo(h2):
    h_mo, eri_mo, _orbitals = molecular_orbital_integrals(h2, 2, (1, 1), None)
    return np.real(h_mo), np.real(eri_mo)


# --------------------------------------------------------------------------- #
# The specification.
# --------------------------------------------------------------------------- #

class TestNormalizingTheSpec:
    def test_none_stays_none(self):
        assert normalize_orbitals(None) is None

    def test_an_integer_stays_an_integer(self):
        assert normalize_orbitals(np.int64(6)) == 6

    def test_a_sequence_becomes_a_sorted_tuple(self):
        assert normalize_orbitals([3, 0, 1, 0]) == (0, 1, 3)
        assert normalize_orbitals(np.array([2, 1])) == (1, 2)

    def test_a_dict_is_validated(self):
        assert normalize_orbitals({"occupied": 2, "virtual": 3}) == \
            {"occupied": 2, "virtual": 3}
        with pytest.raises(ValueError, match="unknown key.*'orbitals'"):
            normalize_orbitals({"occupied": 2, "virtuals": 3})

    def test_a_boolean_is_refused_by_name(self):
        # `"orbitals": True` would otherwise be read as the integer 1 and
        # silently produce a one-orbital register.
        with pytest.raises(TypeError, match="not a flag"):
            normalize_orbitals(True)

    def test_an_unknown_type_is_refused(self):
        with pytest.raises(TypeError, match="unknown active_space 'orbitals'"):
            normalize_orbitals("all")


class TestResolvingTheSelection:
    def test_the_known_names_round_trip(self):
        for name in ACTIVE_SPACE_METHODS:
            assert resolve_method(name) == name

    def test_none_is_the_energy_ordering(self):
        assert resolve_method(None) == "energy"

    def test_underscores_and_case_are_accepted(self):
        assert resolve_method("MP2") == "mp2"

    def test_a_typo_is_refused(self):
        with pytest.raises(ValueError, match="unknown active_space method"):
            resolve_method("natual")


# --------------------------------------------------------------------------- #
# The partition.
# --------------------------------------------------------------------------- #

class TestThePartition:
    def test_no_truncation_keeps_the_complement_of_the_frozen_set(self):
        space = resolve_active_space(n_orbitals=6, num_particles=(2, 2),
                                     frozen=(0,))
        assert space.frozen == (0,)
        assert space.active == (1, 2, 3, 4, 5)
        assert space.deleted == ()
        assert not space.truncated

    def test_an_integer_is_the_register_width(self):
        space = resolve_active_space(n_orbitals=10, num_particles=(2, 2),
                                     spec=spec(orbitals=4))
        assert space.active == (0, 1, 2, 3)
        assert space.deleted == (4, 5, 6, 7, 8, 9)
        assert space.n_active == 4
        assert space.truncated

    def test_an_integer_never_removes_an_occupied_orbital(self):
        with pytest.raises(ValueError, match="never removes an occupied"):
            resolve_active_space(n_orbitals=10, num_particles=(3, 3),
                                 spec=spec(orbitals=2))

    def test_an_integer_larger_than_the_basis_is_refused(self):
        with pytest.raises(ValueError, match="exceeds the"):
            resolve_active_space(n_orbitals=6, num_particles=(1, 1),
                                 spec=spec(orbitals=7))

    def test_a_non_positive_integer_is_refused(self):
        with pytest.raises(ValueError, match="positive number of spatial"):
            resolve_active_space(n_orbitals=6, num_particles=(1, 1),
                                 spec=spec(orbitals=0))

    def test_the_dict_form_splits_occupied_from_virtual(self):
        space = resolve_active_space(n_orbitals=10, num_particles=(3, 3),
                                     spec=spec(orbitals={"occupied": 2,
                                                         "virtual": 3}))
        # The lowest occupied orbital is folded into the mean field; the two
        # highest stay active.
        assert space.frozen == (0,)
        assert space.active == (1, 2, 3, 4, 5)
        assert space.deleted == (6, 7, 8, 9)

    def test_the_dict_form_composes_with_an_explicit_frozen_core(self):
        space = resolve_active_space(n_orbitals=10, num_particles=(4, 4),
                                     spec=spec(orbitals={"occupied": 2,
                                                         "virtual": 2}),
                                    frozen=(0,))
        assert space.frozen == (0, 1)
        assert space.active == (2, 3, 4, 5)
        assert space.deleted == (6, 7, 8, 9)

    def test_the_dict_form_refuses_more_occupied_than_there_are(self):
        with pytest.raises(ValueError, match="active doubly occupied"):
            resolve_active_space(n_orbitals=10, num_particles=(2, 2),
                                 spec=spec(orbitals={"occupied": 5}))

    def test_the_dict_form_refuses_more_virtual_than_there_are(self):
        with pytest.raises(ValueError, match="active virtual"):
            resolve_active_space(n_orbitals=6, num_particles=(2, 2),
                                 spec=spec(orbitals={"virtual": 9}))

    def test_an_open_shell_keeps_its_singly_occupied_orbitals(self):
        # Orbital 2 holds one electron: it can be neither frozen (not doubly
        # occupied) nor deleted (not empty), so it has to stay active.
        space = resolve_active_space(n_orbitals=8, num_particles=(3, 2),
                                     spec=spec(orbitals=4))
        assert space.active == (0, 1, 2, 3)
        assert space.frozen == ()
        assert space.deleted == (4, 5, 6, 7)

    def test_an_open_shell_integer_width_counts_the_singly_occupied_one(self):
        with pytest.raises(ValueError, match="singly occupied"):
            resolve_active_space(n_orbitals=8, num_particles=(3, 2),
                                 spec=spec(orbitals=2))

    def test_a_frozen_orbital_that_is_not_doubly_occupied_is_refused(self):
        with pytest.raises(ValueError, match="not doubly occupied"):
            resolve_active_space(n_orbitals=8, num_particles=(2, 2),
                                 spec=spec(orbitals=4), frozen=(3,))


class TestAnExplicitOrbitalList:
    def test_it_names_the_active_set(self):
        space = resolve_active_space(n_orbitals=8, num_particles=(2, 2),
                                     spec=spec(orbitals=[0, 1, 5, 6]))
        assert space.active == (0, 1, 5, 6)
        assert space.deleted == (2, 3, 4, 7)
        assert space.frozen == ()

    def test_dropping_an_occupied_orbital_is_refused(self):
        # It would take its electrons with it, which is a different molecule and
        # not a smaller correlation treatment.
        with pytest.raises(ValueError, match="neither active nor frozen"):
            resolve_active_space(n_orbitals=8, num_particles=(2, 2),
                                 spec=spec(orbitals=[1, 4, 5]))

    def test_freezing_the_occupied_orbital_instead_is_accepted(self):
        space = resolve_active_space(n_orbitals=8, num_particles=(2, 2),
                                     spec=spec(orbitals=[1, 4, 5]),
                                     frozen=(0,))
        assert space.frozen == (0,)
        assert space.active == (1, 4, 5)
        assert space.deleted == (2, 3, 6, 7)

    def test_an_index_out_of_range_is_refused(self):
        with pytest.raises(ValueError, match="out of range"):
            resolve_active_space(n_orbitals=4, num_particles=(1, 1),
                                 spec=spec(orbitals=[0, 9]))

    def test_an_orbital_that_is_both_frozen_and_active_is_refused(self):
        with pytest.raises(ValueError, match="both frozen and active"):
            resolve_active_space(n_orbitals=8, num_particles=(2, 2),
                                 spec=spec(orbitals=[0, 1, 4]), frozen=(0,))

    def test_an_empty_list_is_refused(self):
        with pytest.raises(ValueError, match="is an empty list"):
            resolve_active_space(n_orbitals=4, num_particles=(1, 1),
                                 spec=spec(orbitals=[]))

    def test_it_cannot_be_combined_with_a_rotating_selector(self):
        # The indices would refer to a basis the selector chose, which the user
        # never saw.
        with pytest.raises(ValueError, match="cannot be combined"):
            resolve_active_space(n_orbitals=8, num_particles=(2, 2),
                                 spec=spec(orbitals=[0, 1, 4], method="mp2"))


# --------------------------------------------------------------------------- #
# The selectors.
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def h2_pair_mo():
    """Two H2 molecules side by side: two doubly occupied orbitals."""
    pair = _integrals("H4", [(0, 0, 0), (0, 0, 0.74), (0, 1.6, 0),
                             (0, 1.6, 0.74)], "6-31G")
    h_mo, eri_mo, _orbitals = molecular_orbital_integrals(pair, 4, (2, 2),
                                                          None)
    return np.real(h_mo), np.real(eri_mo)


def _determinant_energy(h_mo, eri_mo, n_doubly):
    """Closed-shell determinant energy (electronic) in the given basis."""
    o = slice(0, n_doubly)
    J = np.einsum("iijj->ij", eri_mo[o, o, o, o])
    K = np.einsum("ijji->ij", eri_mo[o, o, o, o])
    return float(2.0 * np.trace(h_mo[o, o]) + np.sum(2.0 * J - K))


class TestTheActiveOccupiedNaturalOrbitals:
    """``"mp2"`` diagonalizes the density over the active doubly occupied
    orbitals too, so every active row is an eigenvalue."""

    def test_the_active_occupations_are_eigenvalues(self, h2_pair_mo):
        from mandacaru.algorithms.mp2 import mp2_natural_orbitals
        from mandacaru.algorithms.mp2 import rotate_integrals

        h_mo, eri_mo = h2_pair_mo
        space = resolve_active_space(h_mo, eri_mo, n_orbitals=8,
                                     num_particles=(2, 2),
                                     spec=spec(orbitals=4, method="mp2"))
        block = mp2_natural_orbitals(h_mo, eri_mo, 2).occupied_density
        expected = np.sort(np.linalg.eigvalsh(block))[::-1]
        assert np.allclose(space.occupations[:2], expected, atol=1e-12)
        # Natural orbitals: the density is diagonal in the rotated basis.
        R = space.rotation
        assert np.allclose(R.T @ R, np.eye(8), atol=1e-12)
        assert np.allclose(R[:2, 2:], 0.0, atol=1e-12)
        # The determinant, and so the Hartree-Fock energy, is unchanged.
        h_new, eri_new = rotate_integrals(h_mo, eri_mo, R)
        assert _determinant_energy(np.real(h_new), np.real(eri_new), 2) == \
            pytest.approx(_determinant_energy(h_mo, eri_mo, 2), abs=1e-10)

    def test_a_frozen_orbital_is_not_rotated(self, h2_pair_mo):
        h_mo, eri_mo = h2_pair_mo
        space = resolve_active_space(h_mo, eri_mo, n_orbitals=8,
                                     num_particles=(2, 2),
                                     spec=spec(orbitals=3, method="mp2"),
                                     frozen=(0,))
        assert space.frozen == (0,)
        assert np.allclose(space.rotation[:, 0], np.eye(8)[:, 0], atol=1e-12)


class TestCorrelatingPairs:
    """``correlating_pairs`` keeps each active occupied orbital's partner --
    the virtual its pair excitation goes to -- and bars the partner of an
    occupied orbital the count froze."""

    def test_the_h2_bonding_orbital_pairs_with_the_leading_natural_orbital(
            self, h2_mo):
        h_mo, eri_mo = h2_mo
        space = resolve_active_space(
            h_mo, eri_mo, n_orbitals=4, num_particles=(1, 1),
            spec=spec(orbitals=2, method="mp2", correlating_pairs=True))
        assert space.partners == ((0, 1),)
        assert space.active == (0, 1)
        assert space.partner(1) == 0

    def test_two_occupied_orbitals_get_two_distinct_partners(self,
                                                            h2_pair_mo):
        h_mo, eri_mo = h2_pair_mo
        space = resolve_active_space(
            h_mo, eri_mo, n_orbitals=8, num_particles=(2, 2),
            spec=spec(orbitals=4, method="mp2", correlating_pairs=True))
        partners = dict(space.partners)
        assert sorted(partners) == [0, 1]
        assert len(set(partners.values())) == 2
        assert set(partners.values()) <= set(space.active)

    def test_a_count_frozen_orbital_bars_its_partner(self, h2_pair_mo):
        h_mo, eri_mo = h2_pair_mo
        orbitals = {"occupied": 1, "virtual": 2}
        plain = resolve_active_space(
            h_mo, eri_mo, n_orbitals=8, num_particles=(2, 2),
            spec=spec(orbitals=orbitals, method="mp2"))
        paired = resolve_active_space(
            h_mo, eri_mo, n_orbitals=8, num_particles=(2, 2),
            spec=spec(orbitals=orbitals, method="mp2",
                      correlating_pairs=True))
        assert paired.frozen == plain.frozen == (0,)
        barred = dict(paired.partners)[0]
        assert barred in plain.active and barred not in paired.active
        # The register keeps the size that was asked for.
        assert paired.n_active == plain.n_active == 3
        assert dict(paired.partners)[1] in paired.active
        assert paired.notes[0][0] == "correlating_pairs"
        assert "displaced" in paired.notes[0][1]

    def test_a_count_too_small_for_the_partners_is_refused(self, h2_pair_mo):
        h_mo, eri_mo = h2_pair_mo
        with pytest.raises(ValueError, match="raise 'orbitals' by 1"):
            resolve_active_space(
                h_mo, eri_mo, n_orbitals=8, num_particles=(2, 2),
                spec=spec(orbitals=3, method="mp2", correlating_pairs=True))

    def test_an_explicit_list_is_checked_not_changed(self, h2_mo):
        h_mo, eri_mo = h2_mo
        from mandacaru.algorithms.orbital_integrals import \
            TensorOrbitalIntegrals

        partner = correlating_partners(TensorOrbitalIntegrals(h_mo, eri_mo),
                                       [0], [1, 2, 3], [2, 0, 0, 0])[0][1]
        other = next(a for a in (1, 2, 3) if a != partner)
        with pytest.raises(ValueError, match="correlating partner"):
            resolve_active_space(
                h_mo, eri_mo, n_orbitals=4, num_particles=(1, 1),
                spec=spec(orbitals=[0, other], correlating_pairs=True))

    def test_symmetry_needs_a_geometry(self, h2_mo):
        h_mo, eri_mo = h2_mo
        with pytest.raises(ValueError, match="needs the molecule's geometry"):
            resolve_active_space(
                h_mo, eri_mo, n_orbitals=4, num_particles=(1, 1),
                spec=spec(orbitals=2, method="mp2", symmetry=True))

    def test_the_solver_sets_the_target_state_count(self):
        from mandacaru import Mandacaru

        calc = Mandacaru(method="subspace-vqe", num_states=3,
                         active_space={"orbitals": 4, "symmetry": True})
        assert calc.solver._active_space_request().states == 3
        ground = Mandacaru(method="adapt-vqe",
                           active_space={"orbitals": 4, "symmetry": True})
        assert ground.solver._active_space_request().states == 1

    def test_the_flags_must_be_bools(self):
        with pytest.raises(TypeError, match="is a bool"):
            spec(orbitals=4, correlating_pairs="yes")


class TestTheSelectors:
    def test_the_energy_selector_needs_no_integrals(self):
        space = resolve_active_space(n_orbitals=8, num_particles=(2, 2),
                                     spec=spec(orbitals=4, method="energy"))
        assert space.rotation is None
        assert space.occupations is None
        assert space.correlation_energy is None

    def test_the_mp2_selector_never_mixes_occupied_and_virtual(self, h2_mo):
        h_mo, eri_mo = h2_mo
        space = resolve_active_space(h_mo, eri_mo, n_orbitals=4,
                                    num_particles=(1, 1),
                                    spec=spec(orbitals=2, method="mp2"))
        assert space.rotation is not None
        assert np.allclose(space.rotation[:1, :1], np.eye(1), atol=1e-12)
        assert np.allclose(space.rotation[:1, 1:], 0.0, atol=1e-12)
        assert np.allclose(space.rotation.T @ space.rotation, np.eye(4),
                           atol=1e-12)

    def test_the_mp2_selector_reports_the_full_space_correlation_energy(self, h2_mo):
        h_mo, eri_mo = h2_mo
        space = resolve_active_space(h_mo, eri_mo, n_orbitals=4,
                                    num_particles=(1, 1),
                                    spec=spec(orbitals=2, method="mp2"))
        # It is the quantity the truncation is measured against, so it has to be
        # the *untruncated* one -- negative, and the same whatever is kept.
        assert space.correlation_energy < 0.0
        wider = resolve_active_space(h_mo, eri_mo, n_orbitals=4,
                                     num_particles=(1, 1),
                                     spec=spec(orbitals=3, method="mp2"))
        assert wider.correlation_energy == pytest.approx(
            space.correlation_energy, rel=1e-12)

    def test_the_natural_selector_is_refused_for_a_closed_shell(self):
        # The RHF density is idempotent: its natural occupations are exactly 2
        # and 0, so this would be the energy ordering wearing another name.
        with pytest.raises(ValueError, match="carries no information"):
            resolve_active_space(n_orbitals=8, num_particles=(2, 2),
                                 spec=spec(orbitals=4, method="natural"),
                                 open_shell=False)

    def test_the_mp2_selector_ranks_an_open_shell_by_a_threshold(self):
        # H3 doublet in its UHF natural orbitals: one doubly and one singly
        # occupied orbital, four true virtuals.  The threshold alone sets the
        # width, and the occupied orbitals -- including the singly occupied
        # one -- are left exactly where they are.
        from mandacaru.algorithms.mp2 import open_shell_mp2_natural_orbitals

        integrals = _integrals("H3", [(0, 0, 0), (0, 0, 0.9), (0, 0, 1.8)],
                               "6-31G")
        h_mo, eri_mo, _orbitals = molecular_orbital_integrals(
            integrals, 3, (2, 1), True)
        h_mo, eri_mo = np.real(h_mo), np.real(eri_mo)
        virtual = open_shell_mp2_natural_orbitals(
            h_mo, eri_mo, 2, 1).virtual_occupations
        threshold = 0.5 * (virtual[1] + virtual[2])        # keeps exactly two
        space = resolve_active_space(h_mo, eri_mo, n_orbitals=6,
                                    num_particles=(2, 1),
                                    spec=spec(method="mp2",
                                              threshold=threshold),
                                    open_shell=True)
        assert space.active == (0, 1, 2, 3)
        assert space.deleted == (4, 5)
        assert np.allclose(space.rotation[:2, :], np.eye(6)[:2, :], atol=1e-14)
        assert space.correlation_energy < 0.0

    def test_the_natural_selector_ranks_an_open_shell_by_occupation(self):
        occupations = np.array([2.0, 1.0, 0.01, 0.40, 0.05])
        space = resolve_active_space(n_orbitals=5, num_particles=(2, 1),
                                     spec=spec(orbitals=3, method="natural"),
                                    reference_occupations=occupations,
                                    open_shell=True)
        # Virtuals start at index 2; the most populated of them is index 3, so
        # the permutation has to bring it first.
        assert space.rotation is not None
        assert space.rotation[3, 2] == pytest.approx(1.0)
        assert space.occupations[2] == pytest.approx(0.40)

    def test_the_natural_selector_checks_the_occupation_count(self):
        with pytest.raises(ValueError, match="one occupation per spatial"):
            resolve_active_space(n_orbitals=5, num_particles=(2, 1),
                                 spec=spec(orbitals=3, method="natural"),
                                 reference_occupations=[1.0, 0.5],
                                 open_shell=True)


class TestTheSummary:
    def test_it_names_every_part_of_the_partition(self):
        space = ActiveSpace(n_orbitals=10, frozen=(0,), active=(1, 2, 3),
                            deleted=(4, 5, 6, 7, 8, 9), method="mp2")
        line = space.summary()
        assert "3 active" in line
        assert "1 frozen" in line
        assert "6 deleted" in line
        assert "mp2" in line

    def test_an_untruncated_space_does_not_claim_a_ranking(self):
        space = ActiveSpace(n_orbitals=4, active=(0, 1, 2, 3))
        assert "ranked by" not in space.summary()

    def test_every_orbital_has_an_occupancy_and_a_role(self):
        space = resolve_active_space(n_orbitals=6, num_particles=(3, 2),
                                     spec=spec(orbitals=4), frozen=(0,))
        assert [space.occupancy(p) for p in range(6)] == [
            "occupied", "occupied", "singly", "virtual", "virtual", "virtual"]
        assert [space.role(p) for p in range(6)] == [
            "frozen", "active", "active", "active", "active", "deleted"]


# --------------------------------------------------------------------------- #
# What the truncation costs -- the reason the feature exists.
# --------------------------------------------------------------------------- #

class TestWhatItCosts:
    """Exact energies in each truncated space, against the untruncated one."""

    @staticmethod
    def _energies(integrals, n_electrons, num_particles, width, selection):
        hamiltonian = integrals.molecular_hamiltonian(
            mo_basis=True, n_electrons=n_electrons,
            num_particles=num_particles,
            active_space={"orbitals": width, "method": selection})
        return _ground_state(hamiltonian, num_particles)

    def test_keeping_everything_reproduces_the_untruncated_energy(self, h2):
        full = _ground_state(
            h2.molecular_hamiltonian(mo_basis=True, n_electrons=2,
                                     num_particles=(1, 1)), (1, 1))
        for selection in ("energy", "mp2"):
            # At full width the selector still rotates the virtual block, so
            # this also pins that the rotation changes no physics.
            assert self._energies(h2, 2, (1, 1), 4, selection) == \
                pytest.approx(full, abs=1e-9)

    def test_a_wider_active_space_is_never_worse(self, h2):
        for selection in ("energy", "mp2"):
            energies = [self._energies(h2, 2, (1, 1), width, selection)
                        for width in (2, 3, 4)]
            assert energies[1] <= energies[0] + 1e-10
            assert energies[2] <= energies[1] + 1e-10

    def test_the_mp2_ranking_recovers_far_more_correlation(self, h2):
        # The claim the selector is for.  At two of four orbitals -- half the
        # register -- the second-order ranking keeps a virtual that carries most
        # of the pair correlation, and the canonical ranking keeps one that
        # carries almost none.  Measured on this basis: ~92 % against ~6 %.
        reference = h2.hartree_fock(2)
        e_hf = (reference.electronic_energy + h2.constant_energy
                + h2.nuclear_repulsion)
        e_full = _ground_state(
            h2.molecular_hamiltonian(mo_basis=True, n_electrons=2,
                                     num_particles=(1, 1)), (1, 1))
        correlation = e_full - e_hf
        assert correlation < 0.0

        recovered = {}
        for selection in ("energy", "mp2"):
            energy = self._energies(h2, 2, (1, 1), 2, selection)
            recovered[selection] = (energy - e_hf) / correlation

        assert recovered["mp2"] > 0.85, recovered
        assert recovered["energy"] < 0.20, recovered
        assert recovered["mp2"] > 4.0 * recovered["energy"], recovered

    def test_the_truncated_energy_is_above_the_untruncated_one(self, h2):
        # Deleting a virtual orbital removes variational freedom, so it can only
        # raise the energy.  A truncation that lowered it would mean the
        # deletion had changed the Hamiltonian rather than restricted it.
        full = _ground_state(
            h2.molecular_hamiltonian(mo_basis=True, n_electrons=2,
                                     num_particles=(1, 1)), (1, 1))
        for selection in ("energy", "mp2"):
            assert self._energies(h2, 2, (1, 1), 2, selection) >= full - 1e-10


# --------------------------------------------------------------------------- #
# What it refuses to combine with.
# --------------------------------------------------------------------------- #

class TestTheRefusals:
    def test_the_atomic_orbital_hamiltonian_has_no_orbitals_to_choose(self, h2):
        with pytest.raises(ValueError, match="requires mo_basis=True"):
            h2.molecular_hamiltonian(mo_basis=False,
                                     active_space={"orbitals": 2})

    def test_the_plane_wave_basis_is_refused_by_name(self):
        from mandacaru.algorithms._hamiltonian_from_atoms import \
            build_basis_hamiltonian

        atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.74)],
                      cell=[6, 6, 6])
        with pytest.raises(NotImplementedError, match="plane-wave"):
            build_basis_hamiltonian(
                atoms, {"name": "PW", "energy_cutoff": 100}, None, 0.3, 0, None,
                active_space={"orbitals": 2})


# --------------------------------------------------------------------------- #
# Through the calculator, on the basis this was built for.
# --------------------------------------------------------------------------- #

def _lih(cell=(9.0, 9.0, 11.0), distance=1.60):
    atoms = Atoms("LiH", positions=[(0, 0, 0), (0, 0, distance)], cell=cell)
    atoms.center()
    return atoms


@pytest.fixture(scope="module")
def truncated_paw_run():
    """A converged PAW-LCAO-TZP LiH run on four of its twelve orbitals."""
    from mandacaru import Mandacaru

    atoms = _lih()
    atoms.calc = Mandacaru(method="adapt-vqe",
                           basis={"name": "PAW-LCAO", "size": "TZP"}, h=0.3,
                           active_space={"orbitals": 4, "method": "mp2"},
                           optimizer={"method": "SLSQP", "maxiter": 200},
                           trace=False)
    energy = atoms.get_potential_energy()
    return atoms, energy


class TestThroughTheCalculator:
    def test_the_pseudopotential_basis_accepts_it(self, truncated_paw_run):
        # The point of the exercise: a valence-only basis has no core to freeze
        # (a `"frozen"` core is refused for it) but plenty of virtual orbitals to
        # drop, and dropping them is what fits the register.
        atoms, _energy = truncated_paw_run
        assert atoms.calc.solver.n_qubits == 8

    def test_a_frozen_core_is_still_refused_for_a_pseudopotential_basis(self):
        from mandacaru import Mandacaru

        with pytest.raises(ValueError, match="redundant"):
            Mandacaru(basis={"name": "PAW-LCAO", "size": "SZ"},
                      active_space={"frozen": "auto"})

    def test_the_partition_reaches_the_gradient_context(self, truncated_paw_run):
        atoms, _energy = truncated_paw_run
        context = atoms.calc.solver._gradient_context
        space = context["active_space"]
        assert space.n_active == 4
        assert len(space.deleted) == 8
        assert context["active"] == space.active
        assert context["deleted"] == space.deleted

    def test_the_run_log_block_lists_every_orbital(self, truncated_paw_run,
                                                   tmp_path):
        from mandacaru.utils.logging import Logger, parse_output

        atoms, _energy = truncated_paw_run
        out = str(tmp_path / "output.txt")
        with Logger(out, n_qubits=8) as logger:
            logger.write_system()
            logger.write_active_space(
                *atoms.calc.solver._active_space_report())
        block = parse_output(out)["active_space"]
        assert block["method"] == "mp2"
        assert block["orbitals_requested"] == "4"
        assert block["spatial_orbitals"] == \
            "12 (0 frozen, 4 active, 8 deleted)"
        assert float(block["mp2_correlation_energy_Ha"]) < 0.0
        assert block["occupation"].startswith("eigenvalues of the MP2")
        rows = block["orbitals"]
        assert [row["index"] for row in rows] == list(range(12))
        assert [row["occupancy"] for row in rows] == \
            ["occupied"] + ["virtual"] * 11
        assert [row["role"] for row in rows] == \
            ["active"] * 4 + ["deleted"] * 8
        # The virtual natural occupations, most populated first, and the
        # occupied one just below 2.
        virtual = [row["occupation"] for row in rows[1:]]
        assert virtual == sorted(virtual, reverse=True)
        assert 1.9 < rows[0]["occupation"] < 2.0

    def test_the_log_line_names_the_truncation(self, truncated_paw_run):
        atoms, _energy = truncated_paw_run
        line = atoms.calc.solver._active_space_label()
        assert "4 active" in line
        assert "8 deleted" in line
        assert "mp2" in line
        assert "MP2 E_corr" in line

    def test_the_density_still_holds_every_valence_electron(self,
                                                            truncated_paw_run):
        # The integration test for the `active=` threading: a deleted virtual is
        # neither frozen nor active, so a density path that assumed the active
        # set was the complement of the frozen set would lay the RDM out against
        # the wrong orbitals and lose (or double-count) charge.
        atoms, _energy = truncated_paw_run
        field = atoms.calc.volumetric_field("density")
        assert field.n_electrons == pytest.approx(2.0, abs=1e-6)

    def test_the_atomic_charges_still_sum_to_the_total(self, truncated_paw_run):
        atoms, _energy = truncated_paw_run
        charges = atoms.calc.get_charges()
        assert float(np.sum(charges)) == pytest.approx(0.0, abs=1e-8)

    def test_frozen_core_and_deleted_virtual_forces(self):
        """Core response and active valence RDMs follow the reduced energy."""
        from mandacaru import Mandacaru
        from mandacaru.algorithms._hamiltonian_from_atoms import grid_from_cell

        atoms = _lih()
        grid = grid_from_cell(atoms, 0.35)
        atoms.calc = Mandacaru(
            method="adapt-vqe", basis="6-31G", grid=grid, h=0.35,
            active_space={"frozen": 1, "orbitals": 2},
            optimizer={"method": "SLSQP", "maxiter": 200},
            project_translation=False, trace=False)
        force = atoms.get_forces()[1, 2]
        context = atoms.calc.solver._gradient_context
        assert context["frozen"] == (0,)
        assert context["deleted"]
        step = 0.005
        energies = []
        for sign in (+1, -1):
            displaced = atoms.copy()
            displaced.positions[1, 2] += sign * step
            displaced.calc = atoms.calc
            energies.append(displaced.get_potential_energy())
        assert force == pytest.approx(-(energies[0] - energies[1]) / (2 * step),
                                      abs=0.12)

    def test_the_energy_is_above_the_untruncated_one(self):
        from mandacaru import Mandacaru

        energies = {}
        for width in (None, 4):
            atoms = _lih()
            atoms.calc = Mandacaru(
                method="adapt-vqe",
                basis={"name": "PAW-LCAO", "size": "DZP"}, h=0.3,
                active_space=(None if width is None
                              else {"orbitals": width, "method": "mp2"}),
                optimizer={"method": "SLSQP", "maxiter": 200}, trace=False)
            energies[width] = atoms.get_potential_energy()
        assert energies[4] >= energies[None] - 1e-6

    def test_the_dry_run_predicts_the_register_it_gets(self, truncated_paw_run):
        # A dry run computes no integrals, so it cannot know *which* orbitals the
        # selector keeps -- but the width does not depend on that, and a dry run
        # that disagreed with the run would be worse than no dry run.
        from mandacaru import Mandacaru

        atoms, _energy = truncated_paw_run
        estimate = Mandacaru(
            method="adapt-vqe", basis={"name": "PAW-LCAO", "size": "TZP"},
            h=0.3, active_space={"orbitals": 4, "method": "mp2"},
            dry_run=True, trace=False).estimate_qubits(atoms)
        assert estimate.n_qubits == atoms.calc.solver.n_qubits
        assert estimate.n_spatial_orbitals == 4
        assert estimate.n_deleted_orbitals == 8
        assert estimate.active_method == "mp2"

    def test_the_saved_hamiltonian_metadata_survives_json(self, tmp_path):
        import json

        from mandacaru import Mandacaru

        path = tmp_path / "h.json"
        atoms = _lih()
        atoms.calc = Mandacaru(
            method="adapt-vqe", basis={"name": "PAW-LCAO", "size": "DZP"},
            h=0.3, active_space={"orbitals": {"occupied": 1, "virtual": 3}},
            save_hamiltonian=str(path),
            optimizer={"method": "SLSQP", "maxiter": 50}, trace=False)
        atoms.get_potential_energy()
        record = json.loads(path.read_text())
        metadata = record["metadata"]
        assert metadata["active_space"] == {
            "method": "energy", "orbitals": {"occupied": 1, "virtual": 3},
            "threshold": None, "frozen": None, "correlating_pairs": False,
            "symmetry": False}


# --------------------------------------------------------------------------- #
# Paths that must refuse a truncated virtual space rather than use it.
# --------------------------------------------------------------------------- #

class TestThePeriodicAndFiniteSizePaths:
    """A truncated virtual space reaches these, and they must not use it.

    Both were found by review rather than by a failing test: the periodic
    Hamiltonian builder forwards ``active_space`` (``PeriodicIntegrals``
    subclasses ``MolecularIntegrals``, so nothing stopped it), and the
    finite-size corrections checked only for a frozen core.  A pseudopotential
    basis cannot reach either -- it has no periodic lattice sum -- so the run
    here is an all-electron periodic chain.
    """

    @pytest.fixture(scope="class")
    def truncated_chain(self):
        from mandacaru import Mandacaru

        # H2 per cell, so there are two occupied and two virtual orbitals and a
        # truncation has something to remove.
        atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                      cell=np.diag([4.0, 8.0, 8.0]), pbc=[True, False, False])
        atoms.calc = Mandacaru(
            method="bloch-adapt-vqe", kpts={"size": (2, 1, 1), "gamma": True},
            basis="HAO", h=0.4, active_space={"orbitals": 3}, trace=False,
            optimizer={"method": "SLSQP", "maxiter": 120})
        atoms.get_potential_energy()
        return atoms

    def test_the_truncation_reaches_the_periodic_path_at_all(self,
                                                            truncated_chain):
        # If this ever stops holding, the two refusals below stop being reachable
        # and should be revisited rather than left as dead code.
        context = truncated_chain.calc.solver._gradient_context
        assert context["deleted"]

    def test_the_spectral_function_refuses_it(self, truncated_chain):
        # Its addition branch puts an extra electron into the virtual orbitals,
        # and the deleted ones are exactly where it would go -- the peaks would
        # be the truncated model's, presented as the material's.
        with pytest.raises(NotImplementedError,
                           match="truncated virtual space"):
            truncated_chain.calc.get_spectral_function()

    @pytest.mark.parametrize("scheme", ["mpc", "ccmh"])
    def test_the_pair_density_corrections_refuse_it(self, truncated_chain,
                                                    scheme):
        # They contract the pair density over the active orbitals, which a
        # virtual-only truncation shrinks just as a frozen core does.  Before
        # this check the run reached a shape mismatch reported as a frozen-core
        # problem, which named the wrong cause.
        with pytest.raises(NotImplementedError,
                           match="truncated virtual space"):
            truncated_chain.calc.finite_size_correction(scheme)

    def test_the_kzk_correction_reads_the_density_instead(self, truncated_chain):
        # kzk goes through the same density path as the cube writer, so it is the
        # one scheme that composes with a truncated space -- and it exercises the
        # orbital-count fix on that path (it used to pass the active count where
        # the total was needed).
        result = truncated_chain.calc.finite_size_correction("kzk")
        assert np.isfinite(float(result.correction))
        # `n_electrons` is the integral of the density it actually read, so it is
        # the assertion that catches the orbital-count bug rather than merely
        # surviving it.  Four, not two: the Bloch methods solve the Born-von
        # Karman supercell, which at a 2x1x1 mesh is two primitive cells of two
        # electrons each.
        assert result.n_electrons == pytest.approx(4.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# The occupation criterion.
# --------------------------------------------------------------------------- #

class TestResolvingTheThreshold:
    def test_none_and_false_mean_no_threshold(self):
        assert resolve_threshold(None) is None
        assert resolve_threshold(False) is None

    def test_true_takes_the_default(self):
        assert resolve_threshold(True) == DEFAULT_OCCUPATION_THRESHOLD

    def test_a_number_is_used_as_given(self):
        assert resolve_threshold(1e-4) == pytest.approx(1e-4)

    def test_zero_is_refused_because_it_truncates_nothing(self):
        with pytest.raises(ValueError, match="positive occupation number"):
            resolve_threshold(0.0)

    def test_a_negative_threshold_is_refused(self):
        with pytest.raises(ValueError, match="positive occupation number"):
            resolve_threshold(-1e-3)

    def test_a_threshold_past_a_full_orbital_is_refused_by_name(self):
        # 2 is a *doubly occupied* orbital; a virtual natural occupation is the
        # small charge correlation promotes, never anything like it.
        with pytest.raises(ValueError, match="not an occupation number"):
            resolve_threshold(2.0)


class TestTheOccupationCriterion:
    def test_it_keeps_the_orbitals_that_clear_it(self, h2_mo):
        h_mo, eri_mo = h2_mo
        # Measured spectrum for this fixture: 1.06e-2, 1.63e-3, 7.0e-5.
        space = resolve_active_space(h_mo, eri_mo, n_orbitals=4,
                                    num_particles=(1, 1),
                                    spec=spec(method="mp2", threshold=1e-3))
        assert len(space.active) == 3        # 1 occupied + the two above 1e-3
        assert len(space.deleted) == 1

    def test_a_lower_threshold_keeps_more(self, h2_mo):
        h_mo, eri_mo = h2_mo
        widths = []
        for threshold in (3e-3, 1e-3, 1e-5):
            space = resolve_active_space(h_mo, eri_mo, n_orbitals=4,
                                        num_particles=(1, 1),
                                        spec=spec(method="mp2",
                                                  threshold=threshold))
            widths.append(space.n_active)
        assert widths == sorted(widths)
        assert widths[0] < widths[-1]

    def test_it_needs_no_count(self, h2_mo):
        # A threshold is a complete criterion on its own -- that is the point of
        # it, since the right register width is what one is trying to find out.
        h_mo, eri_mo = h2_mo
        space = resolve_active_space(h_mo, eri_mo, n_orbitals=4,
                                    num_particles=(1, 1),
                                    spec=spec(method="mp2", threshold=1e-3))
        assert space.truncated

    def test_a_count_given_with_it_caps_the_register(self, h2_mo):
        h_mo, eri_mo = h2_mo
        # The threshold alone would keep two virtuals; the count leaves room for
        # one, and the count is what can make a run impossible, so it binds.
        space = resolve_active_space(h_mo, eri_mo, n_orbitals=4,
                                    num_particles=(1, 1),
                                    spec=spec(method="mp2", orbitals=2,
                                              threshold=1e-5))
        assert space.n_active == 2

    def test_a_threshold_looser_than_the_count_leaves_the_count_alone(self, h2_mo):
        h_mo, eri_mo = h2_mo
        space = resolve_active_space(h_mo, eri_mo, n_orbitals=4,
                                    num_particles=(1, 1),
                                    spec=spec(method="mp2", orbitals=4,
                                              threshold=1e-3))
        # Room for three virtuals, but only two earn a place.
        assert space.n_active == 3

    def test_a_threshold_above_the_whole_spectrum_is_refused_with_the_number(self):
        # This is the case a user reaches for by guessing 0.02: it is above every
        # virtual occupation there is, so it would silently leave an active space
        # with no virtual orbital, which returns Hartree-Fock through a
        # variational solver.  The refusal quotes the largest occupation found so
        # the threshold can be chosen rather than guessed again.
        integrals = _integrals("H2", [(0, 0, 0), (0, 0, 0.74)], "6-31G")
        with pytest.raises(ValueError, match="keeps no virtual orbital"):
            integrals.molecular_hamiltonian(
                mo_basis=True, n_electrons=2, num_particles=(1, 1),
                active_space={"method": "mp2", "threshold": 0.02})

    def test_the_energy_ranking_has_no_occupations_to_compare_against(self):
        with pytest.raises(ValueError, match="needs occupation numbers"):
            resolve_active_space(n_orbitals=8, num_particles=(2, 2),
                                 spec=spec(method="energy", threshold=1e-3))

    def test_the_calculator_refuses_the_same_combination_at_construction(self):
        from mandacaru import Mandacaru

        with pytest.raises(ValueError, match="needs occupation numbers"):
            Mandacaru(basis="HAO", active_space={"threshold": 1e-3})

    def test_the_bound_is_kept_and_tightens_as_the_threshold_falls(self, h2):
        # Exact diagonalization, not a variational run: at these widths the
        # truncation error is smaller than an ADAPT optimizer's residual, so
        # comparing two solver runs can invert the ordering while the underlying
        # spaces are perfectly nested.
        full = _ground_state(
            h2.molecular_hamiltonian(mo_basis=True, n_electrons=2,
                                     num_particles=(1, 1)), (1, 1))
        errors = []
        for threshold in (3e-3, 1e-3, 1e-5):
            energy = _ground_state(
                h2.molecular_hamiltonian(
                    mo_basis=True, n_electrons=2, num_particles=(1, 1),
                    active_space={"method": "mp2", "threshold": threshold}),
                (1, 1))
            assert energy >= full - 1e-10
            errors.append(energy - full)
        assert errors == sorted(errors, reverse=True), errors
