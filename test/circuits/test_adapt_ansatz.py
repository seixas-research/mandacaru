# -*- coding: utf-8 -*-
# file: test/circuits/test_adapt_ansatz.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The growable ansatz: the ``sparse`` operator form.

``sparse=True`` holds the Hamiltonian and the pool generators in compressed
sparse form in the native solver; the energy is the dense one. ``sparse`` is a
boolean (``docs/source/guide/scalability.md``): the string spellings of the
retired Python solver (``"auto"``, ``"matrix-free"``) are refused.
"""

import pytest

from mandacaru import Mandacaru
from ase import Atoms


class TestSparsePool:
    @pytest.fixture(scope="class")
    def lih(self):
        return Atoms("LiH", positions=[[3, 3, 3 - 0.8], [3, 3, 3 + 0.8]],
                     cell=[6, 6, 6], pbc=True)

    def _energy(self, atoms, sparse):
        atoms.calc = Mandacaru(method="adapt-vqe", pool="fermionic",
                               crystal_orbitals="real-space", basis="HAO", grid_spacing=0.5, sparse=sparse, trace=False,
                               max_iterations=6, convergence={"gradient": 1e-4})
        return atoms.get_total_energy()

    def test_sparse_matches_dense(self, lih):
        dense = self._energy(lih, sparse=False)
        sparse = self._energy(lih, sparse=True)
        assert dense == pytest.approx(sparse, abs=1e-5)

    @pytest.mark.parametrize("value", ["auto", "matrix-free", "lazy", 1, None])
    def test_a_non_boolean_is_refused(self, value):
        with pytest.raises(TypeError, match="booleans"):
            Mandacaru(method="adapt-vqe", sparse=value)
