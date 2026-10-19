# -*- coding: utf-8 -*-
# file: test/integrals/test_packed.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The eightfold-packed real two-electron tensor against the full array."""

import numpy as np
import pytest

from mandacaru.basis import HydrogenicAtomicOrbital
from mandacaru.integrals import Grid, IntegralEngine
from mandacaru.integrals.packed import PackedERI, dense_tensor, is_packed


def _symmetric_tensor(n, seed=0):
    """A real tensor with the eightfold symmetry of the integrals: a sum of
    products of symmetric pair matrices, ``<ab|cd> = sum_k F_ac F_bd``."""
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(2 * n, n, n))
    F = F + F.transpose(0, 2, 1)
    weights = rng.normal(size=2 * n)
    return np.einsum("k,kac,kbd->abcd", weights, F, F)


@pytest.fixture(scope="module")
def tensor():
    eri = _symmetric_tensor(7)
    return eri, PackedERI.from_dense(eri)


@pytest.fixture(scope="module")
def engine():
    c = (0.6, 0.2, -0.3)
    fns = [HydrogenicAtomicOrbital(1, 0, 0),
           HydrogenicAtomicOrbital(2, 0, 0, center=c),
           HydrogenicAtomicOrbital(2, 1, 1, center=c),
           HydrogenicAtomicOrbital(2, 1, 0, center=c),
           HydrogenicAtomicOrbital(2, 1, -1, center=c),
           HydrogenicAtomicOrbital(3, 2, 0)]
    return IntegralEngine(fns, Grid(box_size=6.0, grid_spacing=0.5,
                                    center=(0, 0, 0)))


class TestTheStore:
    def test_the_engine_writes_the_tensor_packed(self, engine):
        """Both contraction routes, row by row, the full tensor's entries."""
        dense = engine.two_body(energy_units="Ha")
        for budget in (None, 0.05):
            packed = engine.two_body(energy_units="Ha", packed=True,
                                     max_memory_mb=budget)
            assert is_packed(packed) and engine.two_body_profile["packed"]
            assert np.abs(packed.dense() - dense).max() < 1e-14 * np.abs(
                dense).max()

    def test_it_holds_one_eighth_of_the_entries(self, tensor):
        eri, packed = tensor
        n = eri.shape[0]
        P = n * (n + 1) // 2
        assert packed.data.size == P * (P + 1) // 2
        # (P(P+1)/2) / n^4 -> 1/8 for many orbitals; 0.17 at n = 7.
        assert packed.nbytes == 8 * packed.data.size < 0.17 * eri.nbytes
        assert packed.shape == eri.shape and packed.n_pairs == P

    def test_the_round_trip_is_exact(self, tensor):
        eri, packed = tensor
        assert np.array_equal(PackedERI.from_dense(packed.dense()).data,
                              packed.data)
        assert np.abs(packed.dense() - eri).max() < 1e-12
        assert dense_tensor(packed).shape == eri.shape
        assert dense_tensor(eri) is eri or np.shares_memory(
            dense_tensor(eri), eri)
        W = packed.pair_matrix()
        assert np.array_equal(W, W.T)

    def test_a_complex_tensor_does_not_pack(self):
        with pytest.raises(ValueError, match="real"):
            PackedERI.from_dense(np.zeros((2,) * 4, dtype=complex))
        with pytest.raises(ValueError, match="holds"):
            PackedERI(np.zeros(5), 3)


class TestOrbitalChanges:
    def test_a_transform_is_the_dense_one(self, tensor):
        eri, packed = tensor
        n = eri.shape[0]
        C = np.linalg.qr(np.random.default_rng(1).normal(size=(n, n)))[0]
        reference = np.einsum("ap,bq,cr,ds,abcd->pqrs", C, C, C, C, eri,
                              optimize=True)
        assert np.abs(packed.transform(C).dense() - reference).max() < 1e-12
        # A rectangular block of orbitals.
        assert np.abs(packed.transform(C[:, :3]).dense()
                      - reference[:3, :3, :3, :3]).max() < 1e-12

    def test_a_mixed_block_is_the_dense_one(self, tensor):
        eri, packed = tensor
        n = eri.shape[0]
        C = np.random.default_rng(2).normal(size=(n, n))
        C1, C2, C3, C4 = C[:, :2], C[:, 1:4], C[:, 2:7], C[:, 5:]
        reference = np.einsum("ai,bj,ck,dl,abcd->ijkl", C1, C2, C3, C4, eri,
                              optimize=True)
        assert np.abs(packed.block(C1, C2, C3, C4) - reference).max() < 1e-11

    def test_a_subset_is_a_gather(self, tensor):
        eri, packed = tensor
        index = [0, 2, 3, 6]
        assert np.abs(packed.subset(index).dense()
                      - eri[np.ix_(index, index, index, index)]).max() < 1e-12

    def test_complex_orbitals_are_refused(self, tensor):
        _eri, packed = tensor
        with pytest.raises(ValueError, match="real orbitals"):
            packed.transform(1j * np.eye(packed.n))


class TestContractions:
    @pytest.mark.parametrize("complex_density", [False, True])
    def test_coulomb_and_exchange(self, tensor, complex_density):
        eri, packed = tensor
        n = eri.shape[0]
        rng = np.random.default_rng(3)
        D = rng.normal(size=(n, n))
        if complex_density:
            D = D + 1j * rng.normal(size=(n, n))
        D = D + D.conj().T
        J = np.einsum("sr,prqs->pq", D, eri)
        K = np.einsum("sr,prsq->pq", D, eri)
        assert np.abs(packed.coulomb(D) - J).max() < 1e-11
        assert np.abs(packed.exchange(D) - K).max() < 1e-11

    def test_slabs_and_a_density_matrix(self, tensor):
        eri, packed = tensor
        n = eri.shape[0]
        for p in range(n):
            assert np.abs(packed.slab(p) - eri[p]).max() < 1e-12
        rng = np.random.default_rng(4)
        G = rng.normal(size=(n,) * 4) + 1j * rng.normal(size=(n,) * 4)
        assert abs(packed.contract(G) - np.sum(G * eri)) < 1e-10

    def test_the_augmentation_is_added_in_place(self, tensor):
        from mandacaru.core.augmentation import PairAugmentation

        eri, packed = tensor
        n = eri.shape[0]
        rng = np.random.default_rng(5)
        F = rng.normal(size=(3, n, n))
        F = F + F.transpose(0, 2, 1)
        K = rng.normal(size=(3, 3))
        augmentation = PairAugmentation(F, K + K.T)
        expected = augmentation.add_to(eri.copy())
        target = PackedERI(packed.data.copy(), n)
        assert target.add_augmentation(augmentation) is target
        assert np.abs(target.dense() - expected).max() < 1e-11
        with pytest.raises(ValueError, match="symmetric"):
            target.add_augmentation(PairAugmentation(
                rng.normal(size=(1, n, n)), np.eye(1)))
