# -*- coding: utf-8 -*-
# file: test/test_spin_and_sparse.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Initial spin state (ASE magmoms) and the sparse / closed-form ADAPT pool."""

import numpy as np
import pytest
from ase import Atoms

from carcara.algorithms import ADAPTVQE
from carcara.algorithms._hamiltonian_from_atoms import (
    _num_particles, build_basis_hamiltonian, resolve_num_unpaired)


# --------------------------------------------------------------------------- #
# Initial spin state from the Atoms object.
# --------------------------------------------------------------------------- #

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

    def test_spin_flag_high_spin_doublet_for_odd(self):
        atoms = Atoms("H", positions=[[0, 0, 0]])
        assert resolve_num_unpaired(atoms, spin=True, n_el=1) == 1

    def test_spin_flag_noop_for_even(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        assert resolve_num_unpaired(atoms, spin=True, n_el=2) == 0


class TestNumParticles:
    def test_singlet(self):
        assert _num_particles(8, 0, "FAO") == (4, 4)

    def test_triplet(self):
        assert _num_particles(16, 2, "FAO") == (9, 7)

    def test_odd_electron_not_supported(self):
        with pytest.raises(NotImplementedError):
            _num_particles(7, 1, "FAO")

    def test_incompatible_spin_parity(self):
        with pytest.raises(ValueError):
            _num_particles(8, 1, "FAO")     # odd n_unpaired with even electrons


class TestTripletReference:
    def test_o2_magmoms_build_triplet(self):
        atoms = Atoms("O2", positions=[[4, 4, 4 - 0.6], [4, 4, 4 + 0.6]],
                      cell=[8, 8, 8], pbc=True, magmoms=[1.0, 1.0])
        # A compact active space keeps this cheap; the spin state is what matters.
        _, num_particles, _, _ = build_basis_hamiltonian(
            atoms, "FAO", None, 0.5, 0, None, frozen_orbitals=[0, 1, 2, 3, 4])
        na, nb = num_particles
        assert na - nb == 2                 # two unpaired electrons (triplet)


# --------------------------------------------------------------------------- #
# Sparse / closed-form ADAPT pool.
# --------------------------------------------------------------------------- #

class TestSparsePool:
    @pytest.fixture(scope="class")
    def lih(self):
        return Atoms("LiH", positions=[[3, 3, 3 - 0.8], [3, 3, 3 + 0.8]],
                     cell=[6, 6, 6], pbc=True)

    def _energy(self, atoms, sparse):
        atoms.calc = ADAPTVQE(pool="fermionic", basis="FAO", h=0.5, sparse=sparse,
                              verbose=False, max_iterations=6,
                              gradient_tolerance=1e-4)
        return atoms.get_total_energy()

    def test_sparse_matches_dense(self, lih):
        dense = self._energy(lih, sparse=False)
        sparse = self._energy(lih, sparse=True)
        assert dense == pytest.approx(sparse, abs=1e-5)

    def test_auto_enables_sparse_beyond_12_qubits(self):
        assert ADAPTVQE._resolve_sparse("auto", 12) is True
        assert ADAPTVQE._resolve_sparse("auto", 10) is False
        assert ADAPTVQE._resolve_sparse(True, 4) is True
        assert ADAPTVQE._resolve_sparse(False, 20) is False

    def test_sparse_flag_stored(self, lih):
        lih.calc = ADAPTVQE(pool="fermionic", basis="FAO", h=0.5, sparse=True,
                            verbose=False, max_iterations=2)
        lih.get_total_energy()
        assert lih.calc._sparse is True


class TestClosedFormAnsatz:
    def test_sparse_ansatz_matches_dense_state(self):
        from carcara.algorithms.adapt_vqe import AdaptAnsatz
        from carcara.circuits.pools import build_pool

        pool = build_pool("fermionic", 3, (2, 1), mapping="jordan_wigner")
        ops = pool.operators()[:3]
        occ = pool.occupied_orbitals
        dense = AdaptAnsatz(pool.n_qubits, occ, "jordan_wigner", sparse=False)
        sparse = AdaptAnsatz(pool.n_qubits, occ, "jordan_wigner", sparse=True)
        for op in ops:
            dense.append(op)
            sparse.append(op)
        theta = np.array([0.31, -0.72, 0.15])
        assert np.allclose(dense.state(theta), sparse.state(theta), atol=1e-10)
