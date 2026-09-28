# -*- coding: utf-8 -*-
# file: test/core/test_matrix_free.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Qubit operators applied as Pauli strings, never stored as matrices.

:class:`~mandacaru.core.matrix_free.PauliOperator` has to reproduce the stored
forms it replaces -- the full-register CSR matrix and the sector restriction
-- whatever part of its weights it caches, and the memory guard has to refuse
a register no representation can hold before anything is allocated.
"""

import numpy as np
import pytest

from mandacaru.circuits.pools import build_pool
from mandacaru.core.mapping import PauliSum
from mandacaru.core.matrix_free import (PauliOperator, refuse_unaffordable_state,
                                        stored_operator_bytes,
                                        walsh_hadamard_signs)
from mandacaru.core.sector import ParticleSector


def random_operator(n, n_terms, seed, diagonal=0):
    """``n_terms`` random strings plus ``diagonal`` I/Z-only ones."""
    rng = np.random.default_rng(seed)
    labels = set()
    while len(labels) < n_terms:
        labels.add("".join(rng.choice(list("IXYZ"), n)))
    while len(labels) < n_terms + diagonal:
        labels.add("".join(rng.choice(list("IZ"), n)))
    return PauliSum({label: complex(rng.normal(), rng.normal())
                     for label in sorted(labels)}, num_qubits=n)


def random_states(dim, columns, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(dim, columns)) + 1j * rng.normal(size=(dim, columns))


class TestFullRegister:
    @pytest.mark.parametrize("cache_bytes", [None, 0])
    def test_matches_the_stored_matrix(self, cache_bytes):
        # Thirty Z-only strings put more terms in the f = 0 group than there
        # are qubits, so that group goes through the Walsh-Hadamard transform.
        operator = random_operator(8, 60, seed=1, diagonal=30)
        matrix = operator.to_sparse_matrix()
        product = PauliOperator(operator, cache_bytes=cache_bytes)
        states = random_states(2 ** 8, 3, seed=2)
        assert np.allclose(product @ states[:, 0], matrix @ states[:, 0],
                           atol=1e-12)
        assert np.allclose(product @ states, matrix @ states, atol=1e-12)

    def test_hermitian_part_is_the_symmetrized_matrix(self):
        operator = random_operator(6, 40, seed=3)
        matrix = operator.to_sparse_matrix()
        product = PauliOperator(operator, hermitian=True)
        state = random_states(2 ** 6, 1, seed=4)[:, 0]
        assert np.allclose(product @ state,
                           0.5 * (matrix + matrix.conj().T) @ state, atol=1e-12)

    def test_trace_and_adjoint(self):
        operator = random_operator(5, 30, seed=5, diagonal=5)
        matrix = operator.to_matrix()
        product = PauliOperator(operator)
        assert product.trace() == pytest.approx(np.trace(matrix), abs=1e-12)
        state = random_states(2 ** 5, 1, seed=6)[:, 0]
        assert np.allclose(product.adjoint() @ state,
                           matrix.conj().T @ state, atol=1e-12)

    def test_walsh_hadamard_is_the_signed_sum(self):
        phases = np.array([0, 3, 5, 6])
        factors = np.array([0.5, -1.0, 2.0, 0.25j])
        signs = walsh_hadamard_signs(3, phases, factors)
        for x in range(8):
            expected = sum(c * (-1) ** bin(x & z).count("1")
                           for z, c in zip(phases, factors))
            assert signs[x] == pytest.approx(expected, abs=1e-14)

    def test_the_cache_is_bounded(self):
        operator = random_operator(8, 60, seed=7)
        budget = 5 * 20 * 2 ** 8
        product = PauliOperator(operator, cache_bytes=budget)
        state = random_states(2 ** 8, 1, seed=8)[:, 0]
        assert np.allclose(product @ state, operator.to_sparse_matrix() @ state,
                           atol=1e-12)
        assert 0 < product.cached_bytes <= budget
        assert product.cached_bytes < product.num_groups * 20 * 2 ** 8


class TestSector:
    @pytest.mark.parametrize("cache_bytes", [None, 0])
    def test_matches_the_restriction(self, cache_bytes):
        # A generic operator leaves the sector; both forms drop what leaves.
        sector = ParticleSector(8, (2, 1))
        operator = random_operator(8, 80, seed=9)
        restricted = sector.restrict(operator)
        product = PauliOperator(operator, sector=sector, cache_bytes=cache_bytes)
        states = random_states(sector.dim, 2, seed=10)
        assert np.allclose(product @ states, restricted @ states, atol=1e-12)

    def test_pool_generators_match_the_restriction(self):
        pool = build_pool("fermionic", 4, (2, 1))
        sector = ParticleSector(8, (2, 1))
        state = random_states(sector.dim, 1, seed=11)[:, 0]
        for op in pool.operators():
            product = PauliOperator(op.generator, sector=sector, cache_bytes=0)
            assert np.allclose(product @ state,
                               sector.restrict(op.generator) @ state,
                               atol=1e-13)


class TestMemory:
    def test_the_stored_bound_is_never_below_the_matrix(self):
        sector = ParticleSector(8, (2, 2))
        operator = random_operator(8, 120, seed=12)
        restricted = sector.restrict(operator)
        stored = restricted.data.nbytes + restricted.indices.nbytes
        assert stored_operator_bytes(operator, sector.dim) >= stored

    def test_a_100_qubit_state_vector_is_refused_with_the_arithmetic(self):
        with pytest.raises(MemoryError, match=r"2\^100 amplitudes"):
            refuse_unaffordable_state(100, 1 << 100)

    def test_a_small_register_passes(self):
        refuse_unaffordable_state(12, 1 << 12)

    def test_the_full_register_product_stops_at_the_index_width(self):
        with pytest.raises(ValueError, match="2\\^100"):
            PauliOperator(PauliSum({"Z" + "I" * 99: 1.0}))
