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
                                        register_indices,
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

    def test_a_pool_holds_one_index_array(self):
        # The audit's defect: every full-register operator allocated its own
        # arange(2^q), so a pool of K generators held K index arrays.  Check
        # ownership, not only the products.
        import tracemalloc

        pool = build_pool("fermionic", 5, (2, 2))
        generators = [op.generator for op in pool.operators()]
        tracemalloc.start()
        try:
            operators = [PauliOperator(g, cache_bytes=0) for g in generators]
            retained, _ = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        shared = register_indices(10)
        assert len(operators) > 50
        assert all(op._states is shared for op in operators)
        assert not shared.flags.writeable
        assert operators[0].index_bytes == 8 * 2 ** 10
        # One 8 KiB array for the whole pool, not one per generator.
        assert retained < 8 * 2 ** 10 * len(operators) / 4
        state = random_states(2 ** 10, 1, seed=13)[:, 0]
        for op, generator in zip(operators[:5], generators):
            assert np.allclose(op @ state, generator.to_sparse_matrix() @ state,
                               atol=1e-13)

    def test_a_sector_operator_reads_the_sector_indices(self):
        sector = ParticleSector(8, (2, 1))
        product = PauliOperator(random_operator(8, 10, seed=14), sector=sector)
        assert product._states is sector.indices
        assert product.index_bytes == sector.indices.nbytes

    def test_the_full_register_product_stops_at_the_index_width(self):
        with pytest.raises(ValueError, match="2\\^100"):
            PauliOperator(PauliSum({"Z" + "I" * 99: 1.0}))


class TestAvailableMemory:
    """The guards read what can be allocated now, not the installed memory."""

    @staticmethod
    def _cgroup(tmp_path, v2=True, limit="max", usage="0"):
        membership = tmp_path / "cgroup"
        root = tmp_path / "fs"
        if v2:
            membership.write_text("0::/job/step\n")
            group = root / "job" / "step"
            group.mkdir(parents=True)
            (group / "memory.max").write_text("max\n")
            (group / "memory.current").write_text(f"{usage}\n")
            (root / "job" / "memory.max").write_text(f"{limit}\n")
        else:
            membership.write_text("4:memory:/job\n3:cpu:/job\n")
            group = root / "memory" / "job"
            group.mkdir(parents=True)
            (group / "memory.limit_in_bytes").write_text(f"{limit}\n")
            (group / "memory.usage_in_bytes").write_text(f"{usage}\n")
        return str(root), str(membership)

    def test_a_v2_limit_is_the_tightest_ancestor(self, tmp_path):
        from mandacaru.core.matrix_free import cgroup_memory

        root, membership = self._cgroup(tmp_path, limit=str(4 * 2 ** 30),
                                        usage=str(2 ** 30))
        assert cgroup_memory(root, membership) == (4 * 2 ** 30, 2 ** 30)

    def test_v1_and_unlimited_groups(self, tmp_path):
        from mandacaru.core.matrix_free import cgroup_memory

        root, membership = self._cgroup(tmp_path, v2=False,
                                        limit=str(2 ** 33), usage="5")
        assert cgroup_memory(root, membership) == (2 ** 33, 5)
        unlimited = tmp_path / "unlimited"
        unlimited.mkdir()
        root, membership = self._cgroup(unlimited, v2=False,
                                        limit=str(2 ** 63 - 4096), usage="5")
        assert cgroup_memory(root, membership) == (None, 5)
        assert cgroup_memory(str(tmp_path), str(tmp_path / "absent")) == (
            None, None)

    def test_available_is_capped_by_the_group_and_meminfo(self, monkeypatch):
        from mandacaru.core import matrix_free

        monkeypatch.setattr(matrix_free, "cgroup_memory",
                            lambda: (3 * 2 ** 30, 2 ** 30))
        available = matrix_free.available_memory_bytes()
        assert 0 < available <= 2 * 2 ** 30
        assert matrix_free.memory_limit_bytes() <= 3 * 2 ** 30
        monkeypatch.setattr(matrix_free, "cgroup_memory", lambda: (None, None))
        # MemAvailable never exceeds the installed memory.
        assert (matrix_free.available_memory_bytes()
                <= matrix_free.memory_limit_bytes())

    def test_the_state_guard_reads_available_memory(self, monkeypatch):
        from mandacaru.core import matrix_free

        # 2^20 amplitudes: 8 vectors and the indices need 136 MiB.
        monkeypatch.setattr(matrix_free, "available_memory_bytes",
                            lambda: 100 * 2 ** 20)
        with pytest.raises(MemoryError, match="available memory"):
            refuse_unaffordable_state(20, 1 << 20)
        monkeypatch.setattr(matrix_free, "available_memory_bytes",
                            lambda: 200 * 2 ** 20)
        refuse_unaffordable_state(20, 1 << 20)
