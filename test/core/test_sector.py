# -*- coding: utf-8 -*-
# file: test/test_sector.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The particle-number sector agrees with the full qubit register."""

import numpy as np
import pytest

from mandacaru.algorithms.rdm import one_rdm, two_rdm
from mandacaru.core.hamiltonian import spin_block_integrals
from mandacaru.core.mapping import Fermion, PauliSum
from mandacaru.core.sector import ParticleSector

M = 3


def random_hamiltonian(seed=7):
    """A Hermitian, complex, number-conserving Hamiltonian on 2M spin-orbitals."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(M, M)) + 1j * rng.normal(size=(M, M))
    h = a + a.conj().T
    eri = np.zeros((M, M, M, M), dtype=complex)
    for _ in range(3):
        b = rng.normal(size=(M, M)) + 1j * rng.normal(size=(M, M))
        L = b + b.conj().T
        eri += np.einsum("pr,qs->pqrs", L, L)
    return Fermion.from_integrals(*spin_block_integrals(h, eri))


@pytest.mark.parametrize("mapping, reduction", [("jordan_wigner", False),
                                                ("parity", False),
                                                ("bravyi_kitaev", False),
                                                ("parity_reduced", True)])
@pytest.mark.parametrize("num_particles", [(1, 1), (2, 1)])
def test_restriction_is_the_full_matrix_on_the_sector(mapping, reduction,
                                                      num_particles):
    H = random_hamiltonian()
    P = H.map_to_qubits(mapping, n_modes=2 * M,
                        num_particles=num_particles if reduction else None)
    sector = ParticleSector(P.num_qubits, num_particles, mapping)
    full = P.to_sparse_matrix()[sector.indices][:, sector.indices].toarray()
    assert np.allclose(sector.restrict(P).toarray(), full, atol=1e-12)


@pytest.mark.parametrize("mapping", ["jordan_wigner", "bravyi_kitaev"])
def test_sector_rdms_match_the_full_register(mapping):
    sector = ParticleSector(2 * M, (2, 1), mapping)
    rng = np.random.default_rng(3)
    psi = rng.normal(size=sector.dim) + 1j * rng.normal(size=sector.dim)
    psi /= np.linalg.norm(psi)
    full = sector.embed(psi)
    assert np.allclose(one_rdm(psi, 2 * M, mapping, sector=sector),
                       one_rdm(full, 2 * M, mapping), atol=1e-12)
    assert np.allclose(two_rdm(psi, 2 * M, mapping, sector=sector),
                       two_rdm(full, 2 * M, mapping), atol=1e-12)


# --------------------------------------------------------------------------- #
# Restricting a large operator without holding every term at once.
# --------------------------------------------------------------------------- #

def test_restrict_batches_without_changing_the_result():
    """The batched fold is exact, whatever the batch size.

    Every Pauli term contributes one entry per sector state, so keeping all of
    them before de-duplicating costs ``len(terms) * dim`` entries -- 3.7e8 of
    them (~12 GB) for OH in PAW-LCAO-DZ, against a summed result of 16.5M nonzeros.
    ``restrict`` therefore folds the terms into the running matrix in batches,
    which is only legitimate because the sum is linear in them.
    """
    from mandacaru.core.sector import RESTRICT_BATCH_ENTRIES

    rng = np.random.default_rng(11)
    n_qubits = 8
    labels = {"".join(rng.choice(list("IXYZ"), n_qubits)):
              complex(rng.normal(), rng.normal()) for _ in range(150)}
    operator = PauliSum(labels, num_qubits=n_qubits)
    sector = ParticleSector(n_qubits, (2, 2), "jordan_wigner")

    reference = sector.restrict(operator, max_entries=10 ** 9).toarray()
    for batch in (1, 3, 97, RESTRICT_BATCH_ENTRIES):
        restricted = sector.restrict(operator, max_entries=batch).toarray()
        assert np.allclose(restricted, reference, atol=1e-12), batch


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


def test_restrict_of_nothing_is_empty():
    sector = ParticleSector(4, (1, 1), "jordan_wigner")
    empty = sector.restrict(PauliSum({}, num_qubits=4))
    assert empty.shape == (sector.dim, sector.dim) and empty.nnz == 0
