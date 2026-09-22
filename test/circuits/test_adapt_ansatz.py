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
because the excitation generators satisfy ``A**3 = -A``.
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
                               max_iterations=6, gradient_tolerance=1e-4)
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

    def test_sparse_flag_stored(self, lih):
        lih.calc = Mandacaru(method="adapt-vqe", pool="fermionic", basis="HAO",
                             h=0.5, sparse=True, trace=False, max_iterations=2)
        lih.get_total_energy()
        assert lih.calc._sparse is True


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
