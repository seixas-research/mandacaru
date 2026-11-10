# -*- coding: utf-8 -*-
# file: test_mapping.py

import numpy as np
import pytest

from mandacaru import _native
from mandacaru.core.mapping import (
    Fermion,
    MAPPINGS,
    PauliSum,
    bravyi_kitaev,
    jordan_wigner,
    parity,
    parity_reduced,
)

METHODS = ["jordan_wigner", "parity", "bravyi_kitaev"]


def fock_matrix(operator, n):
    """The matrix of a ``Fermion`` in the ``2^n`` occupation basis.

    An oracle independent of the native mapping: the annihilation operators
    are Kronecker products ``Z ... Z (|0><1|) I ... I`` (qubit 0 first).
    """
    lower = np.array([[0, 1], [0, 0]], dtype=complex)
    z, eye = np.diag([1.0, -1.0]).astype(complex), np.eye(2, dtype=complex)

    def annihilator(j):
        out = np.array([[1.0 + 0j]])
        for factor in [z] * j + [lower] + [eye] * (n - 1 - j):
            out = np.kron(out, factor)
        return out

    annihilators = [annihilator(j) for j in range(n)]
    total = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for term, coeff in operator.terms.items():
        product = np.eye(2 ** n, dtype=complex)
        for mode, dagger in term:
            a = annihilators[mode]
            product = product @ (a.conj().T if dagger else a)
        total += coeff * product
    return total


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
        anti = fock_matrix(ai * ajd + ajd * ai, n)
        expected = (np.eye(2 ** n) if i == j else np.zeros((2 ** n, 2 ** n)))
        assert np.allclose(anti, expected)

    def test_number_operator_spectrum(self):
        # a_0^dagger a_0 is a projector: eigenvalues in {0, 1}.
        num = fock_matrix(Fermion.creation(0) * Fermion.annihilation(0), 2)
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
                           fock_matrix(H, 2))
        assert np.allclose(H.to_matrix(), fock_matrix(H, 2))

    def test_jordan_wigner_matches_fock_reference(self):
        H = _random_hermitian_fermion(3)
        assert np.allclose(H.map_to_qubits("jordan_wigner").to_matrix(),
                           fock_matrix(H, 3), atol=1e-9)
        assert np.allclose(H.to_matrix(), fock_matrix(H, 3), atol=1e-12)

    @pytest.mark.parametrize("m", [2, 3])
    def test_all_mappings_share_spectrum(self, m):
        H = _random_hermitian_fermion(m, seed=m)
        ref = np.linalg.eigvalsh(_herm(fock_matrix(H, m)))
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
        with pytest.raises(ValueError, match="tapered"):
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
    mat = _herm(fock_matrix(H, H.n_modes()))
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


def spin_block_integrals(h, eri, h_spin=None):
    h = np.asarray(h)
    eri = np.asarray(eri)
    M = h.shape[0]
    n_so = 2 * M
    dtype = np.result_type(h, eri, float)
    h_so = np.zeros((n_so, n_so), dtype=np.result_type(dtype, *(
        () if h_spin is None else (np.asarray(h_spin),))))
    g_so = np.zeros((n_so,) * 4, dtype=dtype)
    for sigma in range(2):
        a = slice(sigma * M, (sigma + 1) * M)
        h_so[a, a] = h
        for tau in range(2):
            b = slice(tau * M, (tau + 1) * M)
            g_so[a, b, a, b] = eri
    if h_spin is not None:
        h_spin = np.asarray(h_spin)
        if h_spin.shape != (n_so, n_so):
            raise ValueError(f"h_spin must be ({n_so}, {n_so})")
        h_so = h_so + h_spin
    return h_so, g_so


class TestSpatialIntegrals:
    @pytest.mark.parametrize("complex_eri", [False, True])
    def test_terms_and_their_order_match_the_spin_blocked_tensor(
            self, complex_eri):
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


# --- the native algebra behind PauliSum and Fermion ---

def random_operator(n, n_terms, seed):
    rng = np.random.default_rng(seed)
    labels = set()
    while len(labels) < n_terms:
        labels.add("".join(rng.choice(list("IXYZ"), n)))
    return PauliSum({label: complex(rng.normal(), rng.normal())
                     for label in sorted(labels)}, num_qubits=n)


def double_excitation(i, j, a, b):
    """The anti-Hermitian generator ``a+_a a+_b a_j a_i - h.c.``."""
    t = (Fermion.creation(a) * Fermion.creation(b)
         * Fermion.annihilation(j) * Fermion.annihilation(i))
    return t - t.dagger()


def hubbard_chain(sites, hopping=1.0, repulsion=4.0):
    """Open Hubbard chain as a spin-blocked Fermion on ``2 * sites`` modes."""
    terms = {}
    for spin in (0, sites):
        for i in range(sites - 1):
            p, q = i + spin, i + 1 + spin
            terms[((p, True), (q, False))] = -hopping
            terms[((q, True), (p, False))] = -hopping
    for i in range(sites):
        up, down = i, i + sites
        terms[((up, True), (up, False), (down, True), (down, False))] = repulsion
    return Fermion(terms, n_modes=2 * sites)


class TestCommutator:
    @pytest.mark.parametrize("n", [3, 6])
    def test_matches_the_matrices(self, n):
        a, b = random_operator(n, 30, seed=1), random_operator(n, 9, seed=2)
        ma, mb = a.to_matrix(), b.to_matrix()
        assert np.allclose(a.commutator(b).to_matrix(), ma @ mb - mb @ ma,
                           atol=1e-12)

    def test_matches_compose_on_a_wide_register(self):
        a, b = random_operator(60, 40, seed=3), random_operator(60, 8, seed=4)
        reference = (a.compose(b) + b.compose(a) * -1).simplify()
        result = a.commutator(b)
        assert set(result.terms) == set(reference.terms)
        for label, value in reference.terms.items():
            assert result.terms[label] == pytest.approx(value, abs=1e-12)

    def test_antisymmetric(self):
        a, b = random_operator(10, 20, seed=5), random_operator(10, 6, seed=6)
        ab, ba = a.commutator(b), b.commutator(a)
        assert set(ab.terms) == set(ba.terms)
        for label, value in ab.terms.items():
            assert value == pytest.approx(-ba.terms[label], abs=1e-12)

    def test_the_gradient_observable_is_the_analytic_gradient(self):
        # <psi|[H, A]|psi> = 2 Re <H psi|A psi> for an anti-Hermitian A: the
        # commutator is the measurable form of the ADAPT screening gradient.
        n = 8
        hamiltonian = hubbard_chain(n // 2).map_to_qubits("jordan_wigner",
                                                          n_modes=n)
        generator = double_excitation(0, 4, 1, 5).map_to_qubits(
            "jordan_wigner", n_modes=n)
        rng = np.random.default_rng(7)
        psi = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
        psi /= np.linalg.norm(psi)
        h, a = hamiltonian.to_sparse_matrix(), generator.to_sparse_matrix()
        observable = hamiltonian.commutator(generator)
        measured = np.vdot(psi, observable.to_sparse_matrix() @ psi)
        assert measured.imag == pytest.approx(0.0, abs=1e-12)
        assert measured.real == pytest.approx(
            2.0 * np.real(np.vdot(h @ psi, a @ psi)), abs=1e-12)

    def test_the_native_pool_observables_are_the_operator_gradients(self):
        """The native pool's measured commutators are the energy gradients."""
        n = 8
        hamiltonian = hubbard_chain(n // 2).map_to_qubits("jordan_wigner",
                                                          n_modes=n)
        context = _native.Context()
        native = hamiltonian._native_operator()
        problem = context.quantum_problem(native, n // 2, [2, 2], pool="qeb",
                                          sector=False)

        def operator(terms):
            return PauliSum({label: complex(re, im) for label, re, im in terms},
                            num_qubits=n)

        rng = np.random.default_rng(9)
        psi = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
        psi /= np.linalg.norm(psi)
        h_psi = hamiltonian.to_sparse_matrix() @ psi
        assert len(problem.pool_terms) == len(problem.pool_observables) > 0
        for generator, observable in zip(problem.pool_terms,
                                         problem.pool_observables):
            analytic = 2.0 * np.real(np.vdot(
                h_psi, operator(generator).to_sparse_matrix() @ psi))
            measured = np.vdot(psi, operator(observable).to_sparse_matrix() @ psi)
            assert measured == pytest.approx(analytic, abs=1e-12)


class TestNativeRoute:
    def test_the_product_drops_the_strings_that_cancel(self):
        # (X + Z)^2 = 2 I + (XZ + ZX) and XZ + ZX = 0: no Y string survives.
        a = PauliSum({"X": 1.0, "Z": 1.0})
        product = a.compose(a)
        assert product.terms == {"I": pytest.approx(2.0)}
        assert product.num_qubits == 1

    def test_a_vanishing_commutator_keeps_its_register(self):
        zero = PauliSum({"ZZ": 1.0}).commutator(PauliSum({"IZ": 2.0}))
        assert zero.terms == {} and zero.num_qubits == 2

    @pytest.mark.parametrize("name", ["compose", "commutator"])
    def test_operands_on_different_registers_raise(self, name):
        with pytest.raises(ValueError, match="2-qubit and a 3-qubit"):
            getattr(PauliSum({"XX": 1.0}), name)(PauliSum({"XXX": 1.0}))

    def test_an_empty_operand_takes_the_width_of_the_other(self):
        out = PauliSum({"XY": 1.0}).compose(PauliSum({}, num_qubits=2))
        assert out.terms == {} and out.num_qubits == 2

    def test_registers_wider_than_the_native_index_are_refused(self):
        wide = PauliSum({"Z" + "I" * 62: 1.0})
        for call in (wide.to_sparse_matrix, wide.to_matrix,
                     lambda: wide.compose(wide), lambda: wide.commutator(wide)):
            with pytest.raises(ValueError, match="62"):
                call()
        with pytest.raises(ValueError, match="62"):
            Fermion.creation(0).map_to_qubits("jordan_wigner", n_modes=64)

    def test_a_term_outside_the_register_raises(self):
        with pytest.raises(ValueError, match="mode"):
            Fermion.creation(5).map_to_qubits("jordan_wigner", n_modes=4)

    def test_the_constant_term_maps_to_the_identity(self):
        out = (Fermion.identity() * 0.75).map_to_qubits("parity", n_modes=3)
        assert out.terms == {"III": 0.75}
        assert (Fermion.identity() * 0.75).to_matrix(n_modes=2) == \
            pytest.approx(0.75 * np.eye(4))

    def test_the_two_integral_builders_refuse_wrong_shapes(self):
        with pytest.raises(ValueError, match="square"):
            Fermion.from_integrals(np.zeros((2, 3)))
        with pytest.raises(ValueError, match="two-body"):
            Fermion.from_integrals(np.zeros((2, 2)), np.zeros((2, 2, 2)))
        with pytest.raises(TypeError, match="float64 or complex128"):
            _native.Context(threads=1).fermion_integral_terms(
                np.zeros((2, 2), dtype=np.float32))

    def test_integer_and_single_precision_integrals_are_converted(self):
        exact = Fermion.from_integrals(np.array([[1.0, 2.0], [0.0, 3.0]]))
        assert (Fermion.from_integrals(np.array([[1, 2], [0, 3]])).terms
                == exact.terms)
        single = Fermion.from_integrals(
            np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float32))
        assert single.terms == exact.terms


class TestEncodingsAgainstTheFockMatrix:
    @pytest.mark.parametrize("method", METHODS)
    def test_every_encoding_matches_the_fock_matrix(self, method):
        rng = np.random.default_rng(8)
        n = 5
        terms = {(): 0.3}
        for _ in range(40):
            k = int(rng.choice([1, 2, 3, 4]))
            term = tuple((int(rng.integers(0, n)), bool(rng.integers(0, 2)))
                         for _ in range(k))
            terms[term] = complex(rng.normal(), rng.normal())
        fermion = Fermion(terms, n_modes=n)
        mapped = fermion.map_to_qubits(method, n_modes=n)
        reference = fock_matrix(fermion, n)
        assert np.allclose(fermion.to_matrix(n), reference, atol=1e-12)
        if method == "jordan_wigner":
            assert np.allclose(mapped.to_matrix(), reference, atol=1e-12)
        else:
            assert np.allclose(np.linalg.eigvalsh(_herm(mapped.to_matrix())),
                               np.linalg.eigvalsh(_herm(reference)),
                               atol=1e-10)
