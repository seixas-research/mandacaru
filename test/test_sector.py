# -*- coding: utf-8 -*-
# file: test/test_sector.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The particle-number sector agrees with the full qubit register."""

import numpy as np
import pytest

from carcara.algorithms.rdm import one_rdm, two_rdm
from carcara.core.hamiltonian import spin_block_integrals
from carcara.core.mapping import Fermion, PauliSum
from carcara.core.sector import ParticleSector

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
                                                ("parity", True)])
@pytest.mark.parametrize("num_particles", [(1, 1), (2, 1)])
def test_restriction_is_the_full_matrix_on_the_sector(mapping, reduction,
                                                      num_particles):
    H = random_hamiltonian()
    P = H.map_to_qubits(mapping, n_modes=2 * M, two_qubit_reduction=reduction,
                        num_particles=num_particles if reduction else None)
    sector = ParticleSector(P.num_qubits, num_particles, mapping,
                            two_qubit_reduction=reduction)
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
    them (~12 GB) for OH in PAW-DZ, against a summed result of 16.5M nonzeros.
    ``restrict`` therefore folds the terms into the running matrix in batches,
    which is only legitimate because the sum is linear in them.
    """
    from carcara.core.sector import RESTRICT_BATCH_ENTRIES

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


def test_restrict_of_nothing_is_empty():
    sector = ParticleSector(4, (1, 1), "jordan_wigner")
    empty = sector.restrict(PauliSum({}, num_qubits=4))
    assert empty.shape == (sector.dim, sector.dim) and empty.nnz == 0
