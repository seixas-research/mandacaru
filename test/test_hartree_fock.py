# -*- coding: utf-8 -*-
# file: test_hartree_fock.py

"""Restricted and unrestricted Hartree-Fock on the real-space integral engine."""

import numpy as np
import pytest

from carcara.algorithms import RHF, UHF, transform_integrals
from carcara.core import HydrogenicIntegrals, minimal_hydrogenic_basis
from carcara.integrals import Grid


@pytest.fixture(scope="module")
def h2_integrals():
    R = 0.74
    nuclei = [(1.0, np.array([0.0, 0.0, -R / 2])),
              (1.0, np.array([0.0, 0.0, +R / 2]))]
    grid = Grid(center=[0.0, 0.0, 0.0], box_size=6.0, h=0.20)
    return HydrogenicIntegrals(nuclei, minimal_hydrogenic_basis(nuclei), grid)


class TestRHF:
    def test_converges(self, h2_integrals):
        rhf = h2_integrals.hartree_fock(2)
        assert rhf.converged
        assert rhf.n_occupied == 1
        assert rhf.mo_coefficients.shape == (2, 2)

    def test_is_variational_above_fci(self, h2_integrals):
        rhf = h2_integrals.hartree_fock(2)
        total = rhf.electronic_energy + h2_integrals.nuclear_repulsion
        H = h2_integrals.molecular_hamiltonian(mo_basis=True, n_electrons=2)
        m = H.map_to_qubits("jordan_wigner").to_matrix()
        fci = float(np.linalg.eigvalsh(0.5 * (m + m.conj().T)).min())
        # The mean-field energy lies above the correlated ground state.
        assert total >= fci - 1e-9

    def test_odd_electron_count_rejected(self, h2_integrals):
        with pytest.raises(ValueError):
            RHF(h2_integrals.one_body(), h2_integrals.two_body(), 3)


class TestUHF:
    def test_hydrogen_atom_energy(self):
        # A single electron: UHF is self-interaction-free, E -> the exact H 1s.
        c = np.array([0.03, 0.017, 0.023])          # offset from any grid node
        grid = Grid(center=[0.0, 0.0, 0.0], box_size=8.0, h=0.15)
        ig = HydrogenicIntegrals([(1.0, c)],
                                 minimal_hydrogenic_basis([(1.0, c)]), grid)
        e = UHF(ig.one_body(), ig.two_body(), 1, 0).run()
        assert e == pytest.approx(-0.5, abs=0.05)

    def test_matches_rhf_for_closed_shell(self, h2_integrals):
        rhf = RHF(h2_integrals.one_body(), h2_integrals.two_body(), 2).run()
        uhf = UHF(h2_integrals.one_body(), h2_integrals.two_body(), 1, 1).run()
        assert uhf == pytest.approx(rhf.electronic_energy, abs=1e-7)


class TestTransform:
    def test_identity_transform_is_noop(self, h2_integrals):
        h, eri = h2_integrals.one_body(), h2_integrals.two_body()
        C = np.eye(h.shape[0])
        h_new, eri_new = transform_integrals(h, eri, C)
        assert np.allclose(h_new, h)
        assert np.allclose(eri_new, eri)

    def test_mo_hamiltonian_reference_is_hf(self, h2_integrals):
        rhf = h2_integrals.hartree_fock(2)
        H = h2_integrals.molecular_hamiltonian(mo_basis=True, n_electrons=2)
        m = H.map_to_qubits("jordan_wigner").to_matrix()
        # HF determinant |0101>-type reference (occ spin-orbitals {0, 2}).
        hf_index = (1 << (4 - 1 - 0)) | (1 << (4 - 1 - 2))
        e_ref = float(np.real(m[hf_index, hf_index]))
        assert e_ref == pytest.approx(
            rhf.electronic_energy + h2_integrals.nuclear_repulsion, abs=1e-9)
