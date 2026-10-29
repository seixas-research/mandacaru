# -*- coding: utf-8 -*-
# file: test/test_sector.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The particle-number sector agrees with the full qubit register."""

import numpy as np
import pytest

from mandacaru.core.mapping import Fermion, PauliSum
from mandacaru.core.sector import ParticleSector, apply_pauli_sum

M = 3








# --------------------------------------------------------------------------- #
# Restricting a large operator without holding every term at once.
# --------------------------------------------------------------------------- #

def test_restrict_is_the_term_by_term_sum():
    """The group-wise fill is exactly the sum of every term's own action.

    Each term is applied to every sector state with
    :func:`~mandacaru.core.sector.apply_pauli_sum` (images and phases by
    bit operations on one state at a time) and the entries inside the sector
    are summed, which shares no code with the flip-mask grouping and the
    in-place CSR fill of ``restrict``.
    """
    rng = np.random.default_rng(11)
    n_qubits = 8
    labels = {"".join(rng.choice(list("IXYZ"), n_qubits)):
              complex(rng.normal(), rng.normal()) for _ in range(150)}
    operator = PauliSum(labels, num_qubits=n_qubits)
    sector = ParticleSector(n_qubits, (2, 2), "jordan_wigner")

    expected = np.zeros((sector.dim, sector.dim), dtype=complex)
    for column, state in enumerate(sector.indices):
        images, amplitudes = apply_pauli_sum(operator, [state], [1.0])
        rows = sector.positions(images)
        keep = rows >= 0
        expected[rows[keep], column] += amplitudes[keep]
    restricted = sector.restrict(operator)
    assert np.allclose(restricted.toarray(), expected, atol=1e-12)
    # Sorted rows, no stored zero, no duplicate.
    assert restricted.has_sorted_indices
    assert np.all(restricted.data != 0)


def test_the_position_table_agrees_with_the_search():
    """The ``int32`` lookup table and ``searchsorted`` give the same positions,
    inside and outside the sector, and out-of-range indices fall back."""
    sector = ParticleSector(8, (2, 2), "jordan_wigner")
    probe = np.arange(1 << 8, dtype=np.int64)
    where = np.searchsorted(sector.indices, probe)
    where = np.clip(where, 0, sector.dim - 1)
    reference = np.where(sector.indices[where] == probe, where, -1)
    assert np.array_equal(sector.positions(probe), reference)
    assert sector.positions(np.array([-1, 1 << 8, 1 << 40])).tolist() == [-1] * 3


def test_restrict_holds_little_more_than_its_result():
    """The peak of a restriction is the matrix plus a few vectors.

    16 qubits, (4, 4), a molecular-type Hamiltonian (5,793 Pauli terms, 1.77M
    entries, 34 MiB).  The coordinate-format fold this replaced peaked at
    3.8 times the result; the ceiling is 1.25.
    """
    import tracemalloc

    rng = np.random.default_rng(3)
    M = 8
    h = rng.standard_normal((M, M))
    h = np.diag(np.linspace(-3, 3, M)) + 0.1 * (h + h.T)
    chemist = sum(np.einsum("pr,qs->prqs", L, L) for L in
                  ((lambda a: (a + a.T) / 2)(rng.standard_normal((M, M)))
                   for _ in range(4)))
    operator = Fermion.from_spatial_integrals(
        h, 0.05 * chemist.transpose(0, 2, 1, 3)).map_to_qubits(
        "jordan_wigner", n_modes=2 * M)
    sector = ParticleSector(2 * M, (4, 4), "jordan_wigner")
    tracemalloc.start()
    try:
        matrix = sector.restrict(operator)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    stored = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    assert matrix.nnz > 1_000_000
    assert peak < 1.25 * stored


class TestFlipMaskGrouping:
    """``restrict`` pays its search once per flip mask, not once per term.

    A Pauli string sends ``|x>`` to a phase times ``|x ^ f>``, so terms sharing
    ``f`` share every image.  The grouping is what makes a 20-qubit sector
    Hamiltonian minutes instead of an hour, and it is structural rather than
    lucky: ``X`` and ``Y`` both set a flip bit, so all eight strings of a
    Jordan-Wigner double excitation carry the same mask.
    """

    def molecular(self, M, seed=0):
        from mandacaru.core.mapping import Fermion

        rng = np.random.default_rng(seed)
        n = 2 * M
        h = rng.normal(size=(n, n))
        h = 0.5 * (h + h.T)
        g = rng.normal(size=(n,) * 4) * 0.1
        g = 0.25 * (g + g.transpose(1, 0, 3, 2)
                    + g.transpose(2, 3, 0, 1) + g.transpose(3, 2, 1, 0))
        return Fermion.from_integrals(h, g).map_to_qubits("jordan_wigner",
                                                          n_modes=n)

    def test_a_molecular_hamiltonian_has_far_fewer_masks_than_terms(self):
        from mandacaru.core.sector import flip_groups

        for M in (4, 6, 8):
            H = self.molecular(M)
            ratio = len(H.terms) / len(flip_groups(H))
            assert ratio > 5.0, (M, ratio)

    def test_the_groups_account_for_every_nonzero_term(self):
        from mandacaru.core.sector import flip_groups

        H = self.molecular(4)
        H.terms["I" * H.num_qubits] = 0.0            # dropped, not counted
        grouped = sum(len(v) for v in flip_groups(H).values())
        assert grouped == sum(1 for c in H.terms.values() if c != 0)

    def test_every_string_of_a_double_excitation_shares_one_mask(self):
        """Why the ratio is ~8 and not a coincidence."""
        from mandacaru.core.mapping import Fermion
        from mandacaru.core.sector import flip_groups

        double = Fermion({((0, True), (1, True), (3, False), (2, False)): 1.0},
                         n_modes=8)
        excitation = double.map_to_qubits("jordan_wigner", n_modes=8)
        assert len(excitation.terms) > 1
        assert len(flip_groups(excitation)) == 1

    def test_it_agrees_with_the_dense_restriction(self):
        """The grouping is an optimization, so the answer may not move."""
        H = self.molecular(5, seed=3)
        sector = ParticleSector(10, (2, 2), "jordan_wigner")
        dense = H.to_sparse_matrix()[sector.indices][:, sector.indices]
        assert np.allclose(sector.restrict(H).toarray(), dense.toarray(),
                           atol=1e-11)

    def test_a_leaking_operator_still_drops_what_leaves(self):
        """The ``inside.all()`` fast path must not change a leaking case."""
        rng = np.random.default_rng(7)
        labels = {"".join(rng.choice(list("IXYZ"), 8)): complex(rng.normal())
                  for _ in range(120)}
        operator = PauliSum(labels, num_qubits=8)
        sector = ParticleSector(8, (2, 2), "jordan_wigner")
        dense = operator.to_sparse_matrix()[sector.indices][:, sector.indices]
        assert np.allclose(sector.restrict(operator).toarray(),
                           dense.toarray(), atol=1e-12)


class TestConserves:
    """``conserves`` tests each flip-mask group on every state at once; it
    must answer what the state-by-state image test answers."""

    @staticmethod
    def _state_by_state(sector, operator, atol=1e-12):
        from mandacaru.core.sector import apply_pauli_sum

        for index in sector.indices:
            images, amplitudes = apply_pauli_sum(operator, [index], [1.0])
            outside = sector.positions(images) < 0
            if np.any(np.abs(amplitudes[outside]) > atol):
                return False
        return True


    def test_strings_that_leak_alone_can_conserve_together(self):
        """``a+_1 a_0 - h.c.`` maps to ``X0 Y1`` and ``Y0 X1``, which share a
        flip mask.  From an alpha electron in orbital 2 each string alone
        fills orbitals 0 and 1; only their sum cancels that."""
        excitation = Fermion({((1, True), (0, False)): 1.0,
                              ((0, True), (1, False)): -1.0}, n_modes=2 * M)
        generator = excitation.map_to_qubits("jordan_wigner")
        sector = ParticleSector(2 * M, (1, 1), "jordan_wigner")
        assert sector.conserves(generator)
        assert self._state_by_state(sector, generator)
        assert not all(sector.conserves(PauliSum({label: coeff}))
                       for label, coeff in generator.terms.items())

    def test_a_string_lost_to_truncation_is_not_a_leak(self):
        """The mappings drop each string at or below 1e-12 on its own, so a
        conserving term near that scale can keep half of its flip group.
        That remainder must not cost the run its sector (H2 in a last-zeta
        DZP basis fell back to the full 20-qubit register and took over an
        hour)."""
        excitation = Fermion({((1, True), (0, False)): 1.0,
                              ((0, True), (1, False)): -1.0}, n_modes=2 * M)
        generator = excitation.map_to_qubits("jordan_wigner")
        sector = ParticleSector(2 * M, (1, 1), "jordan_wigner")
        label, coeff = next(iter(generator.terms.items()))
        assert sector.conserves(PauliSum({label: 2e-12 * coeff}))
        assert not sector.conserves(PauliSum({label: 1e-9 * coeff}))


def test_restrict_of_nothing_is_empty():
    sector = ParticleSector(4, (1, 1), "jordan_wigner")
    empty = sector.restrict(PauliSum({}, num_qubits=4))
    assert empty.shape == (sector.dim, sector.dim) and empty.nnz == 0




class TestTotalNumberSector:
    """``spin_conserving=False``: every state with N electrons, any S_z."""

    def test_the_dimension_counts_every_spin_distribution(self):
        from math import comb
        sector = ParticleSector(2 * M, (2, 1), spin_conserving=False)
        assert sector.dim == comb(2 * M, 3)
        assert np.all(sector.occupations.sum(axis=1) == 3)


    def test_the_parity_reduction_is_refused(self):
        with pytest.raises(ValueError, match="S_z"):
            ParticleSector(2 * M - 2, (1, 1), "parity_reduced",
                           spin_conserving=False)
