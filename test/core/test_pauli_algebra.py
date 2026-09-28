# -*- coding: utf-8 -*-
# file: test/core/test_pauli_algebra.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Pauli-string algebra on bit tables: commutators and the fermion map.

Everything here must agree with the matrices on small registers and must keep
working, without a matrix, on a 100-qubit one.
"""

import time

import numpy as np
import pytest

from mandacaru.circuits.gates import double_excitation
from mandacaru.core.mapping import Fermion, PauliSum
from mandacaru.core.pauli_algebra import PauliTable, commutator


def random_operator(n, n_terms, seed):
    rng = np.random.default_rng(seed)
    labels = set()
    while len(labels) < n_terms:
        labels.add("".join(rng.choice(list("IXYZ"), n)))
    return PauliSum({label: complex(rng.normal(), rng.normal())
                     for label in sorted(labels)}, num_qubits=n)


def hubbard_chain(sites, hopping=1.0, repulsion=4.0):
    """Open Hubbard chain as a spin-blocked Fermion on ``2 * sites`` modes.

    Written term by term: its integral tensor on 100 spin-orbitals would be
    10^8 entries, nearly all zero.
    """
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


class TestTable:
    @pytest.mark.parametrize("n", [3, 64, 100])
    def test_round_trip(self, n):
        operator = random_operator(n, 25, seed=n)
        back = PauliTable.from_pauli_sum(operator).to_pauli_sum()
        assert back.terms == pytest.approx(operator.terms)
        assert back.num_qubits == n


class TestCommutator:
    @pytest.mark.parametrize("n", [3, 6])
    def test_matches_the_matrices(self, n):
        a, b = random_operator(n, 30, seed=1), random_operator(n, 9, seed=2)
        ma, mb = a.to_matrix(), b.to_matrix()
        assert np.allclose(a.commutator(b).to_matrix(), ma @ mb - mb @ ma,
                           atol=1e-12)

    def test_matches_compose_on_a_wide_register(self):
        a, b = random_operator(70, 40, seed=3), random_operator(70, 8, seed=4)
        reference = (a.compose(b) + b.compose(a) * -1).simplify()
        result = commutator(a, b)
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


class TestPoolObservables:
    def test_every_observable_is_its_operator_gradient(self):
        from mandacaru.circuits.pools import build_pool

        n = 8
        hamiltonian = hubbard_chain(n // 2).map_to_qubits("jordan_wigner",
                                                          n_modes=n)
        pool = build_pool("qeb", n // 2, (2, 2))
        rng = np.random.default_rng(9)
        psi = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
        psi /= np.linalg.norm(psi)
        h_psi = hamiltonian.to_sparse_matrix() @ psi
        for op, observable in zip(pool.operators(),
                                  pool.gradient_observables(hamiltonian)):
            analytic = 2.0 * np.real(np.vdot(
                h_psi, op.generator.to_sparse_matrix() @ psi))
            measured = np.vdot(psi, observable.to_sparse_matrix() @ psi)
            assert measured == pytest.approx(analytic, abs=1e-12)


class TestHundredQubits:
    """A 100-qubit problem is built, mapped and differentiated symbolically."""

    @pytest.fixture(scope="class")
    def hamiltonian(self):
        return hubbard_chain(50).map_to_qubits("jordan_wigner", n_modes=100)

    def test_the_hamiltonian_maps(self, hamiltonian):
        assert hamiltonian.num_qubits == 100
        assert hamiltonian.is_hermitian()
        # Each of the 2 * 49 bonds hops in both directions: XZ..ZX and YZ..ZY
        # strings; the on-site terms give 100 single Z's, 50 ZZ's and the
        # identity.
        assert len(hamiltonian.terms) == 2 * 2 * 49 + 100 + 50 + 1

    def test_the_gradient_observable_is_hermitian_and_matrix_free(
            self, hamiltonian):
        generator = double_excitation(10, 60, 11, 61).map_to_qubits(
            "jordan_wigner", n_modes=100)
        start = time.perf_counter()
        observable = hamiltonian.commutator(generator)
        assert time.perf_counter() - start < 5.0
        assert observable.num_qubits == 100 and observable.terms
        assert observable.is_hermitian(atol=1e-12)
        # Only strings overlapping the excitation's support survive.
        support = {10, 11, 60, 61}
        for label in observable.terms:
            assert any(label[q] != "I" for q in support)


class TestMapping:
    @pytest.mark.parametrize("method",
                             ["jordan_wigner", "parity", "bravyi_kitaev"])
    def test_every_encoding_matches_the_fock_matrix(self, method):
        # The Fock-space matrix is an independent oracle of the encoding.
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
        if method == "jordan_wigner":
            assert np.allclose(mapped.to_matrix(), fermion.to_matrix(n),
                               atol=1e-12)
        else:
            reference = fermion.map_to_qubits("jordan_wigner", n_modes=n)
            assert np.allclose(np.linalg.eigvalsh(
                                   _hermitian(mapped.to_matrix())),
                               np.linalg.eigvalsh(
                                   _hermitian(reference.to_matrix())),
                               atol=1e-10)


def _hermitian(matrix):
    return 0.5 * (matrix + matrix.conj().T)
