# -*- coding: utf-8 -*-
# file: test_mapping.py

import numpy as np
import pytest

from mandacaru.core.mapping import (
    Fermion,
    MAPPINGS,
    PauliSum,
    bravyi_kitaev,
    jordan_wigner,
    parity,
    parity_reduced,
    reencode_pauli,
)


@pytest.mark.parametrize("mapping", MAPPINGS)
def test_reencode_pauli_matches_direct_fermion_mapping(mapping):
    """A canonical JW string decomposition survives the encoding Clifford."""
    hopping = (Fermion.creation(0) * Fermion.annihilation(1)
               + Fermion.creation(1) * Fermion.annihilation(0))
    density = Fermion.creation(0) * Fermion.annihilation(0)
    hamiltonian = hopping + density
    jw = hamiltonian.map_to_qubits("jordan_wigner", n_modes=4)
    expected = hamiltonian.map_to_qubits(
        mapping, n_modes=4,
        num_particles=(1, 1) if mapping == "parity_reduced" else None)
    actual = reencode_pauli(
        jw, mapping,
        num_particles=(1, 1) if mapping == "parity_reduced" else None)
    labels = set(expected.terms) | set(actual.terms)
    assert all(abs(expected.terms.get(label, 0j)
                   - actual.terms.get(label, 0j)) < 1e-10
               for label in labels)

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

class TestMatrixForms:
    """The sparse matrix is built by flip mask; the dense one is the oracle."""

    @staticmethod
    def _random(n_qubits, n_terms, seed):
        rng = np.random.default_rng(seed)
        terms = {"".join(rng.choice(list("IXYZ"), n_qubits)):
                 complex(rng.normal(), rng.normal()) for _ in range(n_terms)}
        return PauliSum(terms, num_qubits=n_qubits)

    @pytest.mark.parametrize("n_qubits", [1, 3, 6])
    def test_the_sparse_matrix_is_the_dense_one(self, n_qubits):
        operator = self._random(n_qubits, 40, seed=n_qubits)
        sparse = operator.to_sparse_matrix()
        assert sparse.format == "csr"
        assert np.allclose(sparse.toarray(), operator.to_matrix(), atol=1e-13)

    def test_the_empty_operator_is_the_zero_matrix(self):
        sparse = PauliSum({}, num_qubits=3).to_sparse_matrix()
        assert sparse.shape == (8, 8) and sparse.nnz == 0

    def test_the_hermitian_part_is_half_the_sum_with_the_adjoint(self):
        operator = self._random(5, 60, seed=4)
        dense = operator.to_matrix()
        part = operator.hermitian_part()
        assert all(isinstance(c, complex) and c.imag == 0.0
                   for c in part.terms.values())
        assert np.allclose(part.to_matrix(), 0.5 * (dense + dense.conj().T),
                           atol=1e-13)


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
        # The JW matrix must equal the occupation-basis reference.
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

    def test_module_functions(self):
        H = _random_hermitian_fermion(2, seed=3)
        assert np.allclose(H.map_to_qubits("jordan_wigner").to_matrix(),
                           jordan_wigner(H).to_matrix())
        assert np.allclose(H.map_to_qubits("bravyi_kitaev").to_matrix(),
                           bravyi_kitaev(H).to_matrix())
        assert np.allclose(H.map_to_qubits("parity").to_matrix(),
                           parity(H).to_matrix())

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            _random_hermitian_fermion(2).map_to_qubits("nonsense")

    def test_exactly_four_mapping_names_are_accepted(self):
        assert MAPPINGS == ("jordan_wigner", "parity", "parity_reduced",
                            "bravyi_kitaev")
        for alias in ("jw", "bk"):
            with pytest.raises(ValueError):
                _random_hermitian_fermion(2).map_to_qubits(alias)


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
        red = H.map_to_qubits("parity_reduced", num_particles=npart)
        assert red.num_qubits == H.n_modes() - 2

    def test_reduced_mapping_function_matches_the_variant(self):
        H, npart = self._spin_conserving_h()
        variant = H.map_to_qubits("parity_reduced", num_particles=npart)
        helper = parity_reduced(H, npart)
        assert helper.terms == pytest.approx(variant.terms)

    def test_reduction_preserves_sector_ground_state(self):
        H, npart = self._spin_conserving_h()
        # Ground state of the full N=(n_alpha+n_beta) sector, computed by
        # restricting the particle-conserving Fock matrix to that occupation
        # block (robust to cross-sector degeneracies).
        sector_min = _sector_ground_state(H, sum(npart))
        red = H.map_to_qubits("parity_reduced", num_particles=npart)
        rmin = np.linalg.eigvalsh(_herm(red.to_matrix())).min()
        assert np.isclose(rmin, sector_min, atol=1e-8)

    @pytest.mark.parametrize(
        "method", ["jordan_wigner", "parity", "bravyi_kitaev"])
    def test_unreduced_mappings_keep_every_qubit(self, method):
        H, _ = self._spin_conserving_h()
        assert H.map_to_qubits(method).num_qubits == H.n_modes()

    def test_reduction_requires_num_particles(self):
        H, _ = self._spin_conserving_h()
        with pytest.raises(ValueError):
            H.map_to_qubits("parity_reduced")

    def test_reduced_mapping_rejects_a_sector_changing_operator(self):
        with pytest.raises(ValueError, match="preserve"):
            Fermion.creation(0).map_to_qubits(
                "parity_reduced", n_modes=4, num_particles=(1, 1))


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


# --- spin-orbital terms without the spin-orbital tensor ---

def _whole_array_terms(h, g, tol=1e-12):
    """``from_integrals`` as one whole-array ``nonzero`` scan wrote it."""
    terms = {}
    for p, q in zip(*np.nonzero(np.abs(h) > tol)):
        key = ((int(p), True), (int(q), False))
        terms[key] = terms.get(key, 0j) + complex(h[p, q])
    for p, q, r, s in zip(*np.nonzero(np.abs(g) > tol)):
        key = ((int(p), True), (int(q), True), (int(s), False), (int(r), False))
        terms[key] = terms.get(key, 0j) + 0.5 * complex(g[p, q, r, s])
    return terms


def _spatial(m, seed, complex_eri):
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((m, m))
    h = h + h.T
    eri = rng.standard_normal((m,) * 4)
    if complex_eri:
        eri = eri + 1j * rng.standard_normal((m,) * 4)
    # Zeros and sub-threshold entries, so the scan has something to skip.
    eri[rng.random((m,) * 4) < 0.3] = 0.0
    eri[0, 1, 1, 0] = 1e-14
    return h, eri


class TestSpatialIntegrals:
    @pytest.mark.parametrize("complex_eri", [False, True])
    def test_terms_and_their_order_match_the_spin_blocked_tensor(
            self, complex_eri):
        from mandacaru.core.hamiltonian import spin_block_integrals

        h, eri = _spatial(4, seed=5, complex_eri=complex_eri)
        reference = Fermion.from_integrals(*spin_block_integrals(h, eri))
        built = Fermion.from_spatial_integrals(h, eri)
        # The same keys, values and insertion order: every later sum over
        # the terms (the mapping included) runs in the same order.
        assert list(built.terms.items()) == list(reference.terms.items())
        assert built.n_modes() == reference.n_modes() == 8
        assert (built.map_to_qubits("jordan_wigner").terms
                == reference.map_to_qubits("jordan_wigner").terms)

    def test_a_spin_orbit_term_enters_the_one_body_block(self):
        from mandacaru.core.hamiltonian import spin_block_integrals

        h, eri = _spatial(3, seed=6, complex_eri=False)
        rng = np.random.default_rng(7)
        h_spin = rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))
        h_spin = h_spin + h_spin.conj().T
        reference = Fermion.from_integrals(*spin_block_integrals(h, eri, h_spin))
        built = Fermion.from_spatial_integrals(h, eri, h_spin)
        assert list(built.terms.items()) == list(reference.terms.items())
        with pytest.raises(ValueError, match="h_spin"):
            Fermion.from_spatial_integrals(h, eri, h_spin[:4, :4])

    def test_spin_resolved_integrals_match_their_spin_orbital_tensor(self):
        """A spin-polarized model's integrals (one block per spin pair) give
        the operator of the spin-orbital tensor holding those blocks, term
        for term and in order."""
        m = 3
        h = np.stack([_spatial(m, seed=10 + s, complex_eri=True)[0]
                      for s in range(2)])
        eri = np.stack([np.stack([_spatial(m, seed=20 + 2 * s + t,
                                           complex_eri=True)[1]
                                  for t in range(2)]) for s in range(2)])
        h_so = np.zeros((2 * m, 2 * m), dtype=complex)
        g_so = np.zeros((2 * m,) * 4, dtype=complex)
        for s in range(2):
            a = slice(s * m, (s + 1) * m)
            h_so[a, a] = h[s]
            for t in range(2):
                b = slice(t * m, (t + 1) * m)
                g_so[a, b, a, b] = eri[s, t]
        reference = Fermion.from_integrals(h_so, g_so)
        built = Fermion.from_spatial_integrals(h, eri)
        assert list(built.terms.items()) == list(reference.terms.items())
        # Equal blocks are the spin-independent tensor.
        same = Fermion.from_spatial_integrals(
            np.stack([h[0], h[0]]), np.stack([np.stack([eri[0, 0]] * 2)] * 2))
        assert (list(same.terms.items())
                == list(Fermion.from_spatial_integrals(h[0],
                                                       eri[0, 0]).terms.items()))
        with pytest.raises(ValueError, match="spin-resolved two-body"):
            Fermion.from_spatial_integrals(h, eri[0])

    def test_a_packed_tensor_gives_the_same_operator_term_for_term(self):
        """The eightfold-packed tensor of real orbitals is read slab by slab:
        the operator is the dense tensor's, key for key and in order."""
        from mandacaru.integrals.packed import PackedERI

        rng = np.random.default_rng(12)
        m = 5
        factors = [(lambda a: (a + a.T) / 2)(rng.standard_normal((m, m)))
                   for _ in range(3)]
        chemist = sum(np.einsum("pr,qs->prqs", L, L) for L in factors)
        dense = chemist.transpose(0, 2, 1, 3)                    # <pq|rs>
        h = rng.standard_normal((m, m)); h = h + h.T
        expected = Fermion.from_spatial_integrals(h, dense)
        packed = Fermion.from_spatial_integrals(h, PackedERI.from_dense(dense))
        assert list(packed.terms) == list(expected.terms)
        assert np.allclose(list(packed.terms.values()),
                           list(expected.terms.values()), atol=1e-13, rtol=0)

    def test_the_slab_scan_is_the_whole_array_scan(self):
        rng = np.random.default_rng(8)
        h = rng.standard_normal((5, 5))
        g = rng.standard_normal((5,) * 4) + 1j * rng.standard_normal((5,) * 4)
        g[rng.random((5,) * 4) < 0.5] = 0.0
        assert (list(Fermion.from_integrals(h, g).terms.items())
                == list(_whole_array_terms(h, g).items()))

    def test_no_spin_orbital_tensor_is_allocated(self):
        import tracemalloc

        # A sparse tensor keeps the terms dictionary small, so the peak is
        # what the build allocates around it.
        M = 12
        h = np.eye(M)
        eri = np.zeros((M,) * 4)
        eri[np.arange(M), np.arange(M), np.arange(M), np.arange(M)] = 1.0
        tracemalloc.start()
        try:
            built = Fermion.from_spatial_integrals(h, eri)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert len(built.terms) == 2 * M + 4 * M
        # The complex (2M)^4 tensor alone would be 5.3 MB.
        assert peak < 16 * (2 * M) ** 4 / 10
