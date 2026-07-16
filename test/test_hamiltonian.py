# -*- coding: utf-8 -*-
# file: test_hamiltonian.py

import numpy as np
import pytest

from carcara.core.hamiltonian import HydrogenicIntegrals, minimal_hydrogenic_basis
from carcara.core.mapping import Fermion
from carcara.integrals import Grid


R = 0.74  # H2 bond length, Angstrom


@pytest.fixture(scope="module")
def h2_integrals():
    nuclei = [(1.0, np.array([0.0, 0.0, -R / 2])),
              (1.0, np.array([0.0, 0.0, +R / 2]))]
    grid = Grid(center=[0.0, 0.0, 0.0], box_size=6.0, h=0.18)
    basis = minimal_hydrogenic_basis(nuclei)
    return HydrogenicIntegrals(nuclei, basis, grid, units="angstrom")


class TestHydrogenicIntegrals:
    def test_orbital_count(self, h2_integrals):
        assert h2_integrals.n_orbitals == 2

    def test_nuclear_repulsion(self, h2_integrals):
        # Z_A Z_B / |R_A - R_B| with R = 0.74 A = 1.39838 bohr.
        bohr = R / 0.52917721067
        assert np.isclose(h2_integrals.nuclear_repulsion, 1.0 / bohr, rtol=1e-3)

    def test_one_body_hermitian_and_symmetric(self, h2_integrals):
        h = h2_integrals.one_body()
        assert np.allclose(h, h.conj().T, atol=1e-8)
        # homonuclear: the two diagonal (on-site) energies are equal
        assert np.isclose(h[0, 0], h[1, 1], atol=1e-3)

    def test_orthonormalized_basis_is_identity_overlap(self, h2_integrals):
        # after Lowdin orthogonalization the effective overlap is I; check the
        # raw overlap is a sensible symmetric matrix with unit-ish diagonal
        S = h2_integrals.overlap()
        assert np.allclose(S, S.conj().T, atol=1e-8)
        assert np.allclose(np.diag(S).real, 1.0, atol=5e-3)

    def test_two_body_symmetries(self, h2_integrals):
        eri = h2_integrals.two_body()
        # Real, physicists'-notation <ab|cd>: the full 8-fold permutation symmetry.
        assert np.abs(eri.imag).max() < 1e-9
        assert np.allclose(eri, eri.transpose(2, 1, 0, 3), atol=1e-8)  # e-1 bra swap
        assert np.allclose(eri, eri.transpose(0, 3, 2, 1), atol=1e-8)  # e-2 bra swap
        assert np.allclose(eri, eri.transpose(2, 3, 0, 1), atol=1e-8)  # <ab|cd>=<cd|ab>

    def test_coulomb_exchange_not_swapped(self, h2_integrals):
        # Regression guard for the physicist/chemist mix-up: the opposite-spin
        # two-electron determinant |0a 1b> must feel the Coulomb J=<01|01>, not
        # the exchange K=<00|11>.
        eri = h2_integrals.two_body().real
        h = h2_integrals.one_body().real
        J01, K01 = eri[0, 1, 0, 1], eri[0, 0, 1, 1]
        H = h2_integrals.molecular_hamiltonian(include_nuclear_repulsion=False)
        n = H.n_modes()
        mat = H.to_matrix().real

        def index(occ):
            b = 0
            for j in occ:
                b |= 1 << (n - 1 - j)
            return b

        # spin-orbital P = p + sigma*M, M=2 -> alpha {0,1}, beta {2,3}
        two_body_ab = mat[index([0, 3]), index([0, 3])] - (h[0, 0] + h[1, 1])
        two_body_aa = mat[index([0, 1]), index([0, 1])] - (h[0, 0] + h[1, 1])
        assert np.isclose(two_body_ab, J01, atol=1e-8)
        assert np.isclose(two_body_aa, J01 - K01, atol=1e-8)


class TestMolecularHamiltonian:
    def test_hamiltonian_is_hermitian(self, h2_integrals):
        H = h2_integrals.molecular_hamiltonian()
        assert H.n_modes() == 4
        assert H.map_to_qubits("jordan_wigner").is_hermitian(atol=1e-7)

    def test_all_mappings_share_ground_state(self, h2_integrals):
        H = h2_integrals.molecular_hamiltonian()
        ref = None
        for method in ("jordan_wigner", "parity", "bravyi_kitaev"):
            mat = H.map_to_qubits(method).to_matrix()
            mat = 0.5 * (mat + mat.conj().T)
            e0 = np.linalg.eigvalsh(mat).min()
            if ref is None:
                ref = e0
            assert np.isclose(e0, ref, atol=1e-7), method

    def test_two_qubit_reduction_matches_sector(self, h2_integrals):
        H = h2_integrals.molecular_hamiltonian()
        n = H.n_modes()
        # Ground state of the 2-electron sector by occupation-block restriction.
        full = _herm(H.to_matrix())
        idx = [i for i in range(2 ** n) if bin(i).count("1") == 2]
        sector_min = float(np.linalg.eigvalsh(full[np.ix_(idx, idx)]).min())

        red = H.map_to_qubits("parity", two_qubit_reduction=True,
                              num_particles=(1, 1))
        assert red.num_qubits == 2
        rmat = _herm(red.to_matrix())
        assert np.isclose(np.linalg.eigvalsh(rmat).min(), sector_min, atol=1e-7)

    def test_nuclear_repulsion_shifts_spectrum(self, h2_integrals):
        H_with = h2_integrals.molecular_hamiltonian(include_nuclear_repulsion=True)
        H_without = h2_integrals.molecular_hamiltonian(include_nuclear_repulsion=False)
        e_with = np.linalg.eigvalsh(
            _herm(H_with.map_to_qubits("jordan_wigner").to_matrix())).min()
        e_without = np.linalg.eigvalsh(
            _herm(H_without.map_to_qubits("jordan_wigner").to_matrix())).min()
        assert np.isclose(e_with - e_without, h2_integrals.nuclear_repulsion,
                          atol=1e-7)


def _herm(m):
    return 0.5 * (m + m.conj().T)
