"""The operator algebra after it moved to Rust (MG-Q5).

``reference/operator_algebra.json`` was written by ``freeze_operator_algebra.py``
on the Python route (``core/pauli_algebra.py``, ``core/sector.py``,
``core/matrix_free.py`` and the Python mappings of ``core/mapping.py``) before it
was deleted. ``PauliSum`` and ``Fermion`` keep their public constructors and
route their algebra, mappings and matrices to the native ``PauliOperator``; every
frozen case is reproduced through the public API (coefficients to 1e-12, labels
exact, matrices to 1e-12). The operations that left the public surface because a
native twin covers them (``qubit_excitation``, ``reencode_pauli``,
``reference_qubit_bits``, ``ParticleSector``, ``apply_pauli_sum``) are reproduced
through that twin.
"""

import numpy as np
import pytest

from mandacaru import _native
from mandacaru.core.mapping import Fermion, PauliSum

from .operator_reference import (REFERENCE, complex_of, encoded_index, frozen_sector, matrix_of,
                                 pauli_terms, vector_of)

TOL = 1e-12
MAPPINGS = ("jordan_wigner", "parity", "bravyi_kitaev", "parity_reduced")
CONTEXT = _native.Context(threads=1)


def fermion_of(frozen_terms, n_modes=None):
    return Fermion({tuple((int(m), bool(d)) for m, d in term): complex_of(c)
                    for term, c in frozen_terms}, n_modes=n_modes)


def assert_pauli(actual, frozen, num_qubits=None):
    """Labels exact and coefficients to 1e-12 against a frozen Pauli sum."""
    expected = pauli_terms(frozen)
    assert set(actual.terms) == set(expected)
    for label, value in expected.items():
        assert abs(actual.terms[label] - value) < TOL, label
    if num_qubits is not None:
        assert actual.num_qubits == num_qubits


def pauli_sum(frozen, num_qubits):
    return PauliSum(pauli_terms(frozen), num_qubits=num_qubits)


def native(operator):
    return operator._native_operator()


def from_native(operator):
    return PauliSum({label: complex(re, im) for label, re, im in operator.terms},
                    num_qubits=operator.num_qubits).simplify()


# --- mappings -------------------------------------------------------------------------------

@pytest.mark.parametrize("case", REFERENCE["mappings"],
                         ids=lambda c: f"{c['operator']}-{c['mapping']}-{c['n_modes']}")
def test_every_frozen_mapping_is_reproduced(case):
    operator = fermion_of(case["terms"])
    kwargs = {} if case["n_modes"] is None else {"n_modes": case["n_modes"]}
    if "num_particles" in case:
        kwargs["num_particles"] = tuple(case["num_particles"])
    if "raises" in case:
        with pytest.raises(ValueError, match="tapered"):
            operator.map_to_qubits(case["mapping"], **kwargs)
        return
    mapped = operator.map_to_qubits(case["mapping"], **kwargs)
    assert_pauli(mapped, case["result"], case["num_qubits"])


def test_the_frozen_mappings_cover_the_requested_variety():
    kinds = {(c["operator"], c["mapping"], "raises" in c) for c in REFERENCE["mappings"]}
    # Number-conserving and number-changing operators, with and without the parity reduction.
    assert ("hopping", "parity_reduced", False) in kinds
    assert ("number_changing", "parity_reduced", True) in kinds
    assert ("number_changing", "jordan_wigner", False) in kinds
    assert {c["mapping"] for c in REFERENCE["mappings"]} == set(MAPPINGS)


# --- products, commutators, Hermitian part -------------------------------------------------

@pytest.mark.parametrize("case", REFERENCE["algebra"], ids=lambda c: c["name"])
def test_compose_commutator_and_hermitian_part(case):
    n = case["num_qubits"]
    a, b = pauli_sum(case["a"], n), pauli_sum(case["b"], n)
    assert_pauli(a.compose(b), case["compose"], n)
    assert_pauli(b.compose(a), case["compose_reversed"], n)
    assert_pauli(a.commutator(b), case["commutator"], n)
    assert_pauli(a.hermitian_part(), case["hermitian_part"], n)
    assert_pauli(b.hermitian_part(), case["hermitian_part_b"], n)
    # The same products from the matrices.
    ma, mb = a.to_matrix(), b.to_matrix()
    assert np.allclose(a.compose(b).to_matrix(), ma @ mb, atol=TOL)
    assert np.allclose(a.commutator(b).to_matrix(), ma @ mb - mb @ ma, atol=TOL)
    assert np.allclose(a.hermitian_part().to_matrix(), 0.5 * (ma + ma.conj().T), atol=TOL)


def test_the_cancelling_product_leaves_no_string():
    case = next(c for c in REFERENCE["algebra"] if c["name"] == "cancelling_product")
    a, b = pauli_sum(case["a"], 2), pauli_sum(case["b"], 2)
    product = a.compose(b)
    # XI * ZI and ZI * XI cancel in the product: the Y string is gone, not left as zero.
    assert "YI" not in product.terms and "YI" not in case["compose"]
    assert all(abs(c) > TOL for c in product.terms.values())
    assert a.commutator(a).terms == {} and PauliSum({"ZZ": 1.0, "IZ": 2.0}).commutator(
        PauliSum({"ZI": 1.0})).terms == {}


# --- matrices -------------------------------------------------------------------------------

@pytest.mark.parametrize("case", REFERENCE["matrices"],
                         ids=lambda c: f"{c['num_qubits']}-qubits-{len(c['operator'])}-terms")
def test_dense_and_sparse_matrices(case):
    operator = pauli_sum(case["operator"], case["num_qubits"])
    expected = matrix_of(case["dense"])
    assert np.allclose(operator.to_matrix(), expected, atol=TOL, rtol=0)
    sparse = operator.to_sparse_matrix()
    assert sparse.format == "csr" and sparse.shape == expected.shape
    assert sparse.nnz == case["sparse_nnz"]
    assert np.allclose(sparse.toarray(), matrix_of(case["sparse"]), atol=TOL, rtol=0)
    state = np.random.default_rng(case["num_qubits"]).normal(size=len(expected)) + 0j
    assert np.allclose(sparse @ state, expected @ state, atol=TOL)


@pytest.mark.parametrize("case", REFERENCE["fermion_matrices"], ids=lambda c: c["name"])
def test_the_fermion_matrix_is_the_frozen_fock_matrix(case):
    operator = fermion_of(case["terms"])
    assert np.allclose(operator.to_matrix(n_modes=case["n_modes"]), matrix_of(case["matrix"]),
                       atol=TOL, rtol=0)


# --- the integral builders ------------------------------------------------------------------

def array_of(flat, shape):
    if flat is None:
        return None
    values = [complex_of(v) if isinstance(v, list) else float(v) for v in flat]
    return np.array(values).reshape(shape)


@pytest.mark.parametrize("index", range(len(REFERENCE["integrals"])))
def test_the_integral_builders_reproduce_the_terms_and_their_order(index):
    case = REFERENCE["integrals"][index]
    inputs = case["inputs"]
    one = array_of(inputs["one"], inputs["one_shape"])
    two = array_of(inputs["two"], inputs["two_shape"])
    if inputs["kind"] == "spin_orbital":
        built = Fermion.from_integrals(one, two, tol=inputs["tol"])
    else:
        built = Fermion.from_spatial_integrals(
            one, two, array_of(inputs.get("spin"), [2 * one.shape[-1]] * 2), tol=inputs["tol"])
    expected = [(tuple((m, d) for m, d in term), complex_of(c)) for term, c in case["terms"]]
    assert list(built.terms.items()) == expected
    assert built.n_modes() == case["n_modes"]


# --- the qubit excitations (native twin: composition of qubit ladders, then the encoding) -----

def jw_ladder(n, j, dagger):
    """``(X_j -+ i Y_j) / 2`` of the Jordan-Wigner qubit ladder, without the parity string."""
    x = "I" * j + "X" + "I" * (n - 1 - j)
    y = "I" * j + "Y" + "I" * (n - 1 - j)
    return CONTEXT.pauli_operator(n, [(x, .5, 0.), (y, 0., -.5 if dagger else .5)])


def qubit_excitation(out, into, n, mapping, num_particles=(1, 1)):
    term = CONTEXT.pauli_operator(n, [("I" * n, 1., 0.)])
    for j in out:
        term = term.compose(jw_ladder(n, j, True))
    for j in into:
        term = term.compose(jw_ladder(n, j, False))
    generator = term.add(term.dagger(), -1.0)
    return generator.map_to(mapping, list(num_particles))


@pytest.mark.parametrize("index", range(len(REFERENCE["excitations"])))
def test_the_qubit_excitation_generators_are_the_native_composition(index):
    case = REFERENCE["excitations"][index]
    if "raises" in case:
        with pytest.raises(ValueError, match="tapered"):
            qubit_excitation(case["out"], case["in"], 6, case["mapping"])
        return
    generator = from_native(qubit_excitation(case["out"], case["in"], 6, case["mapping"]))
    assert_pauli(generator, case["result"], case["num_qubits"])


def hopping_fermion():
    return Fermion({((j, True), (j, False)): .1 * j for j in range(6)}
                   | {((0, True), (1, False)): .2, ((1, True), (0, False)): .2,
                      ((3, True), (4, False)): .2, ((4, True), (3, False)): .2}, n_modes=6)


def hopping_hamiltonian():
    return native(hopping_fermion().map_to_qubits("jordan_wigner"))


# --- re-encoding a Jordan-Wigner operator (native twin: ``PauliOperator.map_to``) --------------

@pytest.mark.parametrize("index", range(len(REFERENCE["reencoded"])))
def test_a_jordan_wigner_operator_re_encodes_like_the_frozen_clifford_map(index):
    case = REFERENCE["reencoded"][index]
    source = CONTEXT.pauli_operator(6, [(label, complex_of(v).real, complex_of(v).imag)
                                        for label, v in case["terms"].items()])
    particles = case["num_particles"] or [1, 1]
    if "raises" in case:
        with pytest.raises(ValueError, match="tapered"):
            source.map_to(case["mapping"], particles)
        return
    assert_pauli(from_native(source.map_to(case["mapping"], particles)), case["result"],
                 case["num_qubits"])


# --- determinants (native twin: ladder operators applied to the vacuum) ------------------------

@pytest.mark.parametrize("mapping", MAPPINGS[:3])
def test_every_determinant_is_the_frozen_basis_state(mapping):
    """Creation operators applied to the vacuum (``|0...0>`` in every encoding) land on the
    frozen basis state of each of the 64 determinants of 6 spin-orbitals."""
    creators = [CONTEXT.fermion_operator(6, None, [([(j, True)], 1., 0.)], mapping=mapping)
                for j in range(6)]
    for mask in range(64):
        state = np.zeros(64, dtype=complex)
        state[0] = 1.0
        occupied = [j for j in range(6) if mask >> j & 1]
        for j in reversed(occupied):
            state = creators[j].apply(state)
        assert np.count_nonzero(np.abs(state) > 1e-12) == 1
        assert int(np.argmax(np.abs(state))) == encoded_index(mapping, occupied), (mapping, mask)


def test_the_reduced_determinants_drop_the_two_parity_qubits_of_the_parity_encoding():
    for mask in range(64):
        parity_bits = REFERENCE["reference_bits"]["parity"][str(mask)]
        reduced = "".join(b for k, b in enumerate(parity_bits) if k not in (2, 5))
        assert REFERENCE["reference_bits"]["parity_reduced"][str(mask)] == reduced


@pytest.mark.parametrize("mapping", MAPPINGS)
def test_the_native_reference_state_is_the_frozen_hartree_fock_determinant(mapping):
    hamiltonian = hopping_hamiltonian()
    if mapping != "jordan_wigner":
        fermion = Fermion({((j, True), (j, False)): .1 * j for j in range(6)}, n_modes=6)
        hamiltonian = native(fermion.map_to_qubits(
            mapping, num_particles=(1, 1) if mapping == "parity_reduced" else None))
    problem = CONTEXT.quantum_problem(hamiltonian, 3, [1, 1], mapping=mapping)
    bits = REFERENCE["reference_bits"][mapping][str(1 << 0 | 1 << 3)]
    assert problem.reference_qubits == [k for k, b in enumerate(bits) if b == "1"]


# --- particle sectors (native twin: the sector basis of a problem) ----------------------------

def sector_hamiltonian(mapping):
    case = next(c for c in REFERENCE["sectors"] if c["mapping"] == mapping
                and "hamiltonian" in c)
    return pauli_sum(case["hamiltonian"], case["num_qubits"]), case


@pytest.mark.parametrize("mapping", MAPPINGS)
def test_the_sector_bases_and_their_restrictions(mapping):
    hamiltonian, frozen = sector_hamiltonian(mapping)
    full = hamiltonian.to_matrix()
    for case in (c for c in REFERENCE["sectors"]
                 if c["mapping"] == mapping and c.get("spin_conserving", True)):
        # The reduced encoding evaluates its two parity qubits in the requested sector.
        kwargs = {"num_particles": tuple(case["num_particles"])} if mapping == "parity_reduced" else {}
        conserving = hopping_fermion().map_to_qubits(mapping, n_modes=6, **kwargs)
        problem = CONTEXT.quantum_problem(native(conserving), 3, case["num_particles"],
                                          mapping=mapping, sector=True)
        indices = np.asarray(problem.basis_indices, dtype=np.int64)
        assert np.array_equal(indices, np.array(case["indices"])), case["num_particles"]
        assert problem.dimension == len(case["indices"])
        if "restricted" in case:
            restricted = full[np.ix_(indices, indices)]
            assert np.allclose(restricted, matrix_of(case["restricted"]), atol=TOL, rtol=0)
    assert np.array_equal(frozen_sector(mapping), np.array(frozen["indices"]))


@pytest.mark.parametrize("mapping", MAPPINGS[:3])
def test_conservation_is_the_vanishing_commutator_with_the_number_operators(mapping):
    hamiltonian, case = sector_hamiltonian(mapping)
    alpha = CONTEXT.fermion_operator(
        6, None, [([(j, True), (j, False)], 1., 0.) for j in range(3)], mapping=mapping)
    beta = CONTEXT.fermion_operator(
        6, None, [([(j, True), (j, False)], 1., 0.) for j in range(3, 6)], mapping=mapping)
    conserving = native(hamiltonian)
    leaking = CONTEXT.pauli_operator(6, [(label, complex_of(v).real, complex_of(v).imag)
                                         for label, v in case["leaking"].items()])
    assert case["conserves"] is True and case["leaking_conserves"] is False
    for number in (alpha, beta):
        assert not conserving.commutator(number).terms
    assert any(leaking.commutator(number).terms for number in (alpha, beta))
    # The sector problem refuses what leaks.
    with pytest.raises(ValueError):
        CONTEXT.quantum_problem(leaking, 3, [1, 1], mapping=mapping, sector=True)


def test_the_frozen_total_number_sectors_are_the_states_of_that_popcount():
    """The total-number sector (spin-orbit Hamiltonians) had no caller and has no native twin;
    its frozen indices are the Jordan-Wigner basis states with ``N`` electrons."""
    cases = [c for c in REFERENCE["sectors"] if c.get("spin_conserving") is False]
    assert cases
    for case in cases:
        total = sum(case["num_particles"])
        expected = [i for i in range(64) if bin(i).count("1") == total]
        assert case["indices"] == expected and case["dim"] == len(expected)


# --- applying an operator to a sparse state ---------------------------------------------------

@pytest.mark.parametrize("index", range(len(REFERENCE["applied"])))
def test_a_sparse_state_is_mapped_to_the_frozen_image(index):
    case = REFERENCE["applied"][index]
    n = case["num_qubits"]
    operator = pauli_sum(case["operator"], n)
    state = np.zeros(2 ** n, dtype=complex)
    state[case["indices"]] = vector_of(case["amplitudes"])
    image = np.zeros(2 ** n, dtype=complex)
    image[case["image_indices"]] = vector_of(case["image_amplitudes"])
    assert np.allclose(native(operator).apply(state), image, atol=TOL, rtol=0)
    assert np.allclose(operator.to_sparse_matrix() @ state, image, atol=TOL, rtol=0)


@pytest.mark.parametrize("index", range(len(REFERENCE["matvec"])))
def test_the_matrix_free_product_is_the_frozen_one(index):
    case = REFERENCE["matvec"][index]
    operator = pauli_sum(case["operator"], case["num_qubits"])
    state = vector_of(case["state"])
    assert np.allclose(native(operator).apply(state), vector_of(case["product"]), atol=TOL, rtol=0)
