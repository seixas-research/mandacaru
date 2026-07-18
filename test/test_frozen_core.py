# -*- coding: utf-8 -*-
# file: test/test_frozen_core.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Frozen-core approximation and the FAO actual-atomic-number change."""

import numpy as np
import pytest
from ase import Atoms

from carcara.basis import BasisSet, FullAtomicOrbital
from carcara.core.hamiltonian import freeze_core_integrals
from carcara.algorithms.hartree_fock import RHF
from carcara.algorithms._hamiltonian_from_atoms import (
    build_basis_hamiltonian, core_electrons, resolve_frozen_core)


# --------------------------------------------------------------------------- #
# Part 1: FAO uses the actual atomic number (no Slater screening).
# --------------------------------------------------------------------------- #

class TestFAOActualAtomicNumber:
    def test_hydrogen_unchanged(self):
        # H 1s Slater charge is already 1.0 == Z, so H is unaffected.
        (orb,) = BasisSet.build("FAO").atom("H")
        assert orb.Z == 1.0

    def test_lithium_uses_bare_Z(self):
        # Every Li subshell now carries the bare nuclear charge Z = 3, not the
        # Slater effective charge (2.7 for 1s, 1.3 for 2s).
        orbs = BasisSet.build("FAO").atom("Li")
        assert all(o.Z == 3.0 for o in orbs)
        # sanity: the Slater method still exists (used by NAO) and differs.
        assert FullAtomicOrbital.slater_effective_charge(3, 1, 0) != 3.0

    def test_carbon_uses_bare_Z(self):
        orbs = BasisSet.build("FAO").atom("C")
        assert all(o.Z == 6.0 for o in orbs)


# --------------------------------------------------------------------------- #
# Part 2: frozen-core integral reduction is exact for a frozen determinant.
# --------------------------------------------------------------------------- #

def _random_mo_integrals(M, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, M))
    h = A + A.T
    chem = rng.standard_normal((M, M, M, M)) * 0.2
    chem = 0.25 * (chem + chem.transpose(1, 0, 2, 3) + chem.transpose(0, 1, 3, 2)
                   + chem.transpose(1, 0, 3, 2))
    chem = 0.5 * (chem + chem.transpose(2, 3, 0, 1))
    eri = chem.transpose(0, 2, 1, 3)          # physicists' <pq|rs> = (pr|qs)
    return h, eri


def _hf_det_energy(h, eri, n_occ):
    e = 0.0
    for i in range(n_occ):
        e += 2.0 * h[i, i]
    for i in range(n_occ):
        for j in range(n_occ):
            e += 2.0 * eri[i, j, i, j] - eri[i, j, j, i]
    return float(np.real(e))


class TestFreezeCoreIntegrals:
    def test_freeze_all_occupied_gives_hf_energy(self):
        h, eri = _random_mo_integrals(4, seed=1)
        rhf = RHF(h, eri, n_electrons=4).run()
        n_occ = 2
        _, _, ecore = freeze_core_integrals(
            rhf.h_mo, rhf.eri_mo, list(range(n_occ)), [])
        assert np.isclose(ecore, rhf.electronic_energy)

    def test_partial_freeze_reproduces_full_determinant(self):
        h, eri = _random_mo_integrals(4, seed=2)
        rhf = RHF(h, eri, n_electrons=4).run()
        n_occ = 2
        E_full = _hf_det_energy(rhf.h_mo, rhf.eri_mo, n_occ)

        h_a, eri_a, ecore = freeze_core_integrals(
            rhf.h_mo, rhf.eri_mo, frozen=[0], active=[1, 2, 3])
        E_active = _hf_det_energy(h_a, eri_a, n_occ - 1)
        assert np.isclose(ecore + E_active, E_full)
        assert h_a.shape == (3, 3)
        assert eri_a.shape == (3, 3, 3, 3)


# --------------------------------------------------------------------------- #
# Part 3: driver-facing resolvers and the active-space reduction.
# --------------------------------------------------------------------------- #

class TestCoreElectrons:
    @pytest.mark.parametrize("Z, expected", [
        (1, 0), (2, 0), (3, 2), (8, 2), (10, 2),
        (11, 10), (18, 10), (19, 18), (36, 18)])
    def test_noble_gas_core(self, Z, expected):
        assert core_electrons(Z) == expected


class TestResolveFrozenCore:
    def test_false_freezes_nothing(self):
        assert resolve_frozen_core(False, None, [3, 1], n_el=4, n_orbitals=3) == []

    def test_auto_freezes_chemical_core(self):
        # LiH: Li He-core (2 e-) -> 1 spatial orbital.
        assert resolve_frozen_core(True, None, [3, 1], n_el=4, n_orbitals=3) == [0]
        assert resolve_frozen_core("auto", None, [3, 1], 4, 3) == [0]

    def test_integer_freezes_lowest(self):
        assert resolve_frozen_core(2, None, [8, 1, 1], n_el=10, n_orbitals=6) \
            == [0, 1]

    def test_explicit_list_overrides(self):
        assert resolve_frozen_core(True, [0, 1], [8, 1, 1], 10, 6) == [0, 1]

    def test_rejects_freezing_virtual(self):
        # only doubly occupied orbitals (index < n_occ) may be frozen.
        with pytest.raises(ValueError):
            resolve_frozen_core(None, [2], [3, 1], n_el=4, n_orbitals=3)

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            resolve_frozen_core(None, [5], [3, 1], n_el=4, n_orbitals=3)


class TestActiveSpaceReduction:
    def test_lih_frozen_core_reduces_qubits(self):
        atoms = Atoms("LiH", positions=[[0, 0, 0], [0, 0, 1.6]],
                      cell=[6, 6, 6], pbc=True)
        _, np_full, norb_full, _ = build_basis_hamiltonian(
            atoms, "FAO", None, 0.5, 0, None, frozen_core=False)
        _, np_fc, norb_fc, _ = build_basis_hamiltonian(
            atoms, "FAO", None, 0.5, 0, None, frozen_core=True)

        assert (norb_full, np_full) == (3, (2, 2))       # 6 qubits
        assert (norb_fc, np_fc) == (2, (1, 1))           # 4 qubits (Li 1s frozen)

    def test_plane_wave_frozen_core_not_supported(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]],
                      cell=[4, 4, 4], pbc=True)
        with pytest.raises(NotImplementedError):
            build_basis_hamiltonian(
                atoms, {"name": "PW", "energy_cutoff": 100}, None, 0.5, 0, None,
                frozen_core=True)
