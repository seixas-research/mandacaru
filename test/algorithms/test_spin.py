# -*- coding: utf-8 -*-
# file: test/algorithms/test_spin.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The spin state a run actually solves for.

The initial magnetic moments select it (`resolve_num_unpaired`), the `spin`
flag is only the fallback, and `_num_particles` checks parity rather than
refusing odd electron counts.  `initial_state=` names the reference the
ansatz starts from.
"""

import pytest
from ase import Atoms

from mandacaru.algorithms import Mandacaru
from mandacaru.algorithms._hamiltonian_from_atoms import resolve_num_unpaired
from mandacaru.algorithms._hamiltonian_from_atoms import (_num_particles, build_basis_hamiltonian)


class TestResolveNumUnpaired:
    def test_no_magmoms_closed_shell(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        assert resolve_num_unpaired(atoms, spin=False, n_el=2) == 0

    def test_magmoms_triplet(self):
        atoms = Atoms("O2", positions=[[0, 0, 0], [0, 0, 1.2]], magmoms=[1.0, 1.0])
        assert resolve_num_unpaired(atoms, spin=False, n_el=16) == 2

    def test_magmoms_take_priority_over_spin_flag(self):
        atoms = Atoms("O2", positions=[[0, 0, 0], [0, 0, 1.2]], magmoms=[1.0, 1.0])
        assert resolve_num_unpaired(atoms, spin=True, n_el=16) == 2

    def test_odd_count_is_a_doublet_with_or_without_the_flag(self):
        atoms = Atoms("H", positions=[[0, 0, 0]])
        assert resolve_num_unpaired(atoms, spin=True, n_el=1) == 1
        assert resolve_num_unpaired(atoms, spin=False, n_el=1) == 1

    def test_spin_flag_noop_for_even(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        assert resolve_num_unpaired(atoms, spin=True, n_el=2) == 0


class TestNumParticles:
    def test_singlet(self):
        assert _num_particles(8, 0, "HAO") == (4, 4)

    def test_triplet(self):
        assert _num_particles(16, 2, "HAO") == (9, 7)

    def test_odd_electron_doublet_and_quartet(self):
        assert _num_particles(7, 1, "HAO") == (4, 3)
        assert _num_particles(7, 3, "HAO") == (5, 2)
        with pytest.raises(ValueError):
            _num_particles(7, 0, "HAO")     # even n_unpaired with odd electrons

    def test_incompatible_spin_parity(self):
        with pytest.raises(ValueError):
            _num_particles(8, 1, "HAO")     # odd n_unpaired with even electrons


class TestTripletReference:
    def test_o2_magmoms_build_triplet(self):
        atoms = Atoms("O2", positions=[[4, 4, 4 - 0.6], [4, 4, 4 + 0.6]],
                      cell=[8, 8, 8], pbc=True, magmoms=[1.0, 1.0])
        # A compact active space keeps this cheap; the spin state is what matters.
        _, num_particles, _, _, _ctx = build_basis_hamiltonian(
            atoms, "HAO", None, 0.5, 0, None, frozen_orbitals=[0, 1, 2, 3, 4])
        na, nb = num_particles
        assert na - nb == 2                 # two unpaired electrons (triplet)


class TestSpinAndInitialState:
    def test_spin_defaults_false(self):
        assert Mandacaru(method="adapt-vqe", pool="ceo", basis="HAO").spin is False

    def test_spin_flag_stored(self):
        assert Mandacaru(method="adapt-vqe", pool="ceo", basis="HAO",
                         spin=True).spin is True

    def test_even_electron_spin_polarized_matches_closed_shell(self):
        # For a singlet (even electrons) spin-polarized == closed-shell.
        atoms = Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
                      cell=[[6, 0, 0], [0, 6, 0], [0, 0, 6]], pbc=True)

        def energy(spin):
            atoms.calc = Mandacaru(method="adapt-vqe", pool="ceo", basis="HAO",
                                   h=0.4, spin=spin, trace=False,
                                   max_iterations=6, gradient_tolerance=1e-3)
            return atoms.get_total_energy()

        assert energy(False) == pytest.approx(energy(True), abs=1e-6)

    def test_initial_state_default_is_hartree_fock(self):
        assert Mandacaru(method="adapt-vqe", pool="ceo", basis="HAO").initial_state == "hartree-fock"

    def test_initial_state_none_is_hartree_fock(self):
        assert Mandacaru(method="adapt-vqe", pool="ceo", basis="HAO",
                         initial_state=None).initial_state == "hartree-fock"

    def test_unknown_initial_state_rejected(self):
        with pytest.raises(ValueError):
            Mandacaru(method="adapt-vqe", pool="ceo", basis="HAO",
                      initial_state="random")
