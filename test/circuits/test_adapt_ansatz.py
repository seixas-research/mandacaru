# -*- coding: utf-8 -*-
# file: test/circuits/test_adapt_ansatz.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The growable ansatz: a sparse pool and the closed-form exponential.

``sparse`` keeps the Hamiltonian and the pool generators as scipy-sparse
matrices, and :class:`~mandacaru.circuits.AdaptAnsatz` then applies
``exp(theta A)`` through ``I + sin(theta) A + (1 - cos(theta)) A**2`` -- exact
because the excitation generators satisfy ``A**3 = -A``.  ``sparse=
"matrix-free"`` applies the same operators as Pauli strings and stores no
matrix at all (:mod:`mandacaru.core.matrix_free`).
"""

import numpy as np
import pytest

from mandacaru.algorithms import Mandacaru
from ase import Atoms


class TestSparsePool:
    @pytest.fixture(scope="class")
    def lih(self):
        return Atoms("LiH", positions=[[3, 3, 3 - 0.8], [3, 3, 3 + 0.8]],
                     cell=[6, 6, 6], pbc=True)

    def _energy(self, atoms, sparse):
        atoms.calc = Mandacaru(method="adapt-vqe", pool="fermionic",
                               basis="HAO", h=0.5, sparse=sparse, trace=False,
                               max_iterations=6, convergence={"gradient": 1e-4})
        return atoms.get_total_energy()

    def test_sparse_matches_dense(self, lih):
        dense = self._energy(lih, sparse=False)
        sparse = self._energy(lih, sparse=True)
        assert dense == pytest.approx(sparse, abs=1e-5)

    def test_auto_enables_sparse_beyond_10_qubits(self):
        adapt = Mandacaru(method="adapt-vqe")
        assert adapt._resolve_sparse("auto", 10) is True
        assert adapt._resolve_sparse("auto", 8) is False
        assert adapt._resolve_sparse(True, 4) is True
        assert adapt._resolve_sparse(False, 20) is False
        assert adapt._resolve_sparse("matrix-free", 4) is True
        with pytest.raises(ValueError, match="matrix-free"):
            adapt._resolve_sparse("lazy", 4)

    def test_sparse_flag_stored(self, lih):
        lih.calc = Mandacaru(method="adapt-vqe", pool="fermionic", basis="HAO",
                             h=0.5, sparse=True, trace=False, max_iterations=2)
        lih.get_total_energy()
        assert lih.calc._sparse is True


class TestMatrixFree:
    @pytest.fixture(scope="class")
    def lih(self):
        return Atoms("LiH", positions=[[3, 3, 3 - 0.8], [3, 3, 3 + 0.8]],
                     cell=[6, 6, 6], pbc=True)

    def _run(self, atoms, **options):
        atoms.calc = Mandacaru(method="adapt-vqe", pool="fermionic",
                               basis="HAO", h=0.5, trace=False,
                               max_iterations=6,
                               convergence={"gradient": 1e-4}, **options)
        return atoms.get_total_energy()

    @pytest.mark.parametrize("sector", [False, True])
    def test_matches_the_stored_operators(self, lih, sector):
        stored = self._run(lih, sparse=True, sector=sector)
        free = self._run(lih, sparse="matrix-free", sector=sector)
        assert lih.calc.solver._matrix_free is True
        assert free == pytest.approx(stored, abs=1e-8)

    def test_auto_goes_matrix_free_when_the_matrices_cannot_fit(
            self, lih, monkeypatch):
        import mandacaru.algorithms.base as base
        # 64 KiB of "memory": a quarter of it is below the 6-qubit problem's
        # 30 KiB CSR bound (Hamiltonian plus pool).  The state-vector guard
        # reads the real machine, so only the choice of form is affected.
        monkeypatch.setattr(base, "physical_memory_bytes", lambda: 2 ** 16)
        lih.calc = Mandacaru(method="adapt-vqe", pool="fermionic", basis="HAO",
                             h=0.5, trace=False, max_iterations=1)
        lih.get_total_energy()
        assert lih.calc.solver._matrix_free is True

    def test_auto_keeps_the_stored_matrices_when_they_fit(self, lih):
        lih.calc = Mandacaru(method="adapt-vqe", pool="fermionic", basis="HAO",
                             h=0.5, trace=False, max_iterations=1)
        lih.get_total_energy()
        assert lih.calc.solver._matrix_free is False


class TestClosedFormAnsatz:
    def test_sparse_ansatz_matches_dense_state(self):
        from mandacaru.circuits import AdaptAnsatz
        from mandacaru.circuits.pools import build_pool

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

    def test_matrix_free_ansatz_matches_dense_state(self):
        from mandacaru.circuits import AdaptAnsatz
        from mandacaru.circuits.pools import build_pool

        pool = build_pool("fermionic", 3, (2, 1), mapping="jordan_wigner")
        ops = pool.operators()[:3]
        occ = pool.occupied_orbitals
        dense = AdaptAnsatz(pool.n_qubits, occ, "jordan_wigner", sparse=False)
        free = AdaptAnsatz(pool.n_qubits, occ, "jordan_wigner",
                           matrix_free=True)
        for op in ops:
            dense.append(op)
            free.append(op)
        theta = np.array([0.31, -0.72, 0.15])
        assert np.allclose(dense.state(theta), free.state(theta), atol=1e-10)
