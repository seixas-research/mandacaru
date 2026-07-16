# -*- coding: utf-8 -*-
# file: test_mapping.py

import numpy as np
import pytest

from carcara.core.mapping import (
    Fermion,
    PauliSum,
    bravyi_kitaev,
    jordan_wigner,
    parity,
)

METHODS = ["jordan_wigner", "parity", "bravyi_kitaev"]


# --- PauliSum algebra ---

class TestPauliSum:
    def test_identity_matrix(self):
        assert np.allclose(PauliSum.identity(2).to_matrix(), np.eye(4))

    def test_add_and_scale(self):
        p = PauliSum({"X": 1.0}) + PauliSum({"Z": 2.0})
        p = p * 0.5
        assert p.terms == {"X": 0.5, "Z": 1.0}

    def test_compose_paulis(self):
        # X * Y = i Z on a single qubit.
        out = PauliSum({"X": 1.0}).compose(PauliSum({"Y": 1.0})).simplify()
        assert list(out.terms) == ["Z"]
        assert np.isclose(out.terms["Z"], 1j)

    def test_compose_matches_matmul(self):
        a = PauliSum({"XY": 0.5, "ZI": 1.0 - 0.3j})
        b = PauliSum({"YX": 2.0, "IZ": 0.7j})
        assert np.allclose(a.compose(b).to_matrix(), a.to_matrix() @ b.to_matrix())

    def test_hermiticity_check(self):
        assert PauliSum({"XZ": 1.0, "YY": -2.0}).is_hermitian()
        assert not PauliSum({"XZ": 1j}).is_hermitian()

    def test_unequal_lengths_raise(self):
        with pytest.raises(ValueError):
            PauliSum({"X": 1.0, "XY": 1.0})


# --- Fermion operator algebra ---

class TestFermionAlgebra:
    def test_creation_annihilation_adjoint(self):
        a = Fermion.annihilation(0)
        assert a.dagger().terms == Fermion.creation(0).terms

    def test_annihilation_squared_is_zero(self):
        a = Fermion.annihilation(1)
        prod = (a * a).map_to_qubits("jordan_wigner", n_modes=3).simplify()
        assert prod.terms == {} or all(abs(c) < 1e-12 for c in prod.terms.values())

    @pytest.mark.parametrize("i,j", [(0, 0), (0, 1), (1, 0), (2, 1)])
    def test_canonical_anticommutation(self, i, j):
        # {a_i, a_j^dagger} = delta_ij * I  (checked on the 2^n Fock space).
        n = 3
        ai = Fermion.annihilation(i)
        ajd = Fermion.creation(j)
        anti = (ai * ajd + ajd * ai).to_matrix(n_modes=n)
        expected = (np.eye(2 ** n) if i == j else np.zeros((2 ** n, 2 ** n)))
        assert np.allclose(anti, expected)

    def test_number_operator_spectrum(self):
        # a_0^dagger a_0 is a projector: eigenvalues in {0, 1}.
        num = (Fermion.creation(0) * Fermion.annihilation(0)).to_matrix(n_modes=2)
        assert np.allclose(np.sort(np.unique(np.round(np.linalg.eigvalsh(num), 9))),
                           [0.0, 1.0])


# --- from_integrals and the mappings ---

def _random_hermitian_fermion(m, seed=0):
    """A Hermitian one- plus two-body Fermion over ``m`` spin-orbitals."""
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
    h = h + h.conj().T
    eri = rng.standard_normal((m, m, m, m))
    # symmetrize a real (pq|rs)-style array, then relabel to the physicists'
    # tensor g[p,q,r,s] = <pq|rs> = (pr|qs) so the resulting H is Hermitian.
    eri = 0.25 * (eri + eri.transpose(1, 0, 2, 3)
                  + eri.transpose(0, 1, 3, 2) + eri.transpose(1, 0, 3, 2))
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    g = np.einsum("prqs->pqrs", eri)  # <pq|rs> = (pr|qs)
    return Fermion.from_integrals(h, g)


class TestMappings:
    def test_from_integrals_one_body_matches_matrix(self):
        h = np.array([[1.0, 0.5], [0.5, -0.3]])
        H = Fermion.from_integrals(h)
        ref = np.zeros((4, 4), dtype=complex)
        # occupation basis |n0 n1>: h_pq a+_p a_q
        # easiest check: JW matrix equals to_matrix reference
        assert np.allclose(H.map_to_qubits("jordan_wigner").to_matrix(),
                           H.to_matrix())

    def test_jordan_wigner_matches_fock_reference(self):
        H = _random_hermitian_fermion(3)
        assert np.allclose(H.map_to_qubits("jordan_wigner").to_matrix(),
                           H.to_matrix(), atol=1e-9)

    @pytest.mark.parametrize("m", [2, 3])
    def test_all_mappings_share_spectrum(self, m):
        H = _random_hermitian_fermion(m, seed=m)
        ref = np.linalg.eigvalsh(_herm(H.to_matrix()))
        for method in METHODS:
            mat = _herm(H.map_to_qubits(method).to_matrix())
            assert np.allclose(np.linalg.eigvalsh(mat), ref, atol=1e-8), method

    @pytest.mark.parametrize("method", METHODS)
    def test_mapping_is_hermitian(self, method):
        H = _random_hermitian_fermion(3, seed=7)
        assert H.map_to_qubits(method).is_hermitian(atol=1e-8)

    @pytest.mark.parametrize("method", METHODS)
    def test_number_operator_diagonal(self, method):
        # n_j = a_j^dag a_j maps to a diagonal (all-Z/I) operator with 0/1 spectrum.
        n = 4
        for j in range(n):
            nj = Fermion.creation(j) * Fermion.annihilation(j)
            ps = nj.map_to_qubits(method, n_modes=n).simplify()
            assert all(set(label) <= {"I", "Z"} for label in ps.terms), method
            ev = np.linalg.eigvalsh(_herm(ps.to_matrix()))
            assert np.allclose(np.unique(np.round(ev, 9)), [0.0, 1.0])

    def test_aliases_and_module_functions(self):
        H = _random_hermitian_fermion(2, seed=3)
        assert np.allclose(H.map_to_qubits("jw").to_matrix(),
                           jordan_wigner(H).to_matrix())
        assert np.allclose(H.map_to_qubits("bk").to_matrix(),
                           bravyi_kitaev(H).to_matrix())
        assert np.allclose(H.map_to_qubits("parity").to_matrix(),
                           parity(H).to_matrix())

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            _random_hermitian_fermion(2).map_to_qubits("nonsense")


# --- parity two-qubit reduction ---

class TestTwoQubitReduction:
    def _spin_conserving_h(self, n_alpha=1, n_beta=1):
        # 2 alpha + 2 beta modes (spin-blocked), a small spin/number-conserving H.
        m = 4  # spin-orbitals; block 0,1 alpha, 2,3 beta
        h = np.zeros((m, m))
        # hopping within each spin block
        h[0, 1] = h[1, 0] = -0.5
        h[2, 3] = h[3, 2] = -0.5
        h[0, 0] = h[2, 2] = -1.2
        h[1, 1] = h[3, 3] = -0.9
        eri = np.zeros((m, m, m, m))
        for p in range(m):
            eri[p, p, p, p] = 0.8  # on-site density-density
        g = np.einsum("prqs->pqrs", eri)
        return Fermion.from_integrals(h, g), (n_alpha, n_beta)

    def test_reduction_removes_two_qubits(self):
        H, npart = self._spin_conserving_h()
        red = H.map_to_qubits("parity", two_qubit_reduction=True,
                              num_particles=npart)
        assert red.num_qubits == H.n_modes() - 2

    def test_reduction_preserves_sector_ground_state(self):
        H, npart = self._spin_conserving_h()
        # Ground state of the full N=(n_alpha+n_beta) sector, computed by
        # restricting the particle-conserving Fock matrix to that occupation
        # block (robust to cross-sector degeneracies).
        sector_min = _sector_ground_state(H, sum(npart))
        red = H.map_to_qubits("parity", two_qubit_reduction=True,
                              num_particles=npart)
        rmin = np.linalg.eigvalsh(_herm(red.to_matrix())).min()
        assert np.isclose(rmin, sector_min, atol=1e-8)

    def test_reduction_requires_parity(self):
        H, npart = self._spin_conserving_h()
        with pytest.raises(ValueError):
            H.map_to_qubits("bravyi_kitaev", two_qubit_reduction=True,
                            num_particles=npart)

    def test_reduction_requires_num_particles(self):
        H, _ = self._spin_conserving_h()
        with pytest.raises(ValueError):
            H.map_to_qubits("parity", two_qubit_reduction=True)


def _herm(m):
    return 0.5 * (m + m.conj().T)


def _sector_ground_state(H, n_particles):
    """Lowest eigenvalue of ``H`` within the fixed particle-number block.

    ``H`` conserves particle number, so its Fock-space (Jordan-Wigner) matrix is
    block-diagonal in the occupation number.  Restricting to the rows/columns
    whose bit-count equals ``n_particles`` isolates that sector exactly.
    """
    n = H.n_modes()
    mat = _herm(H.to_matrix())
    idx = [i for i in range(2 ** n) if bin(i).count("1") == n_particles]
    block = mat[np.ix_(idx, idx)]
    return float(np.linalg.eigvalsh(block).min())
