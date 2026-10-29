# -*- coding: utf-8 -*-
# file: test/core/test_augmentation.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The PAW-LCAO augmentation kept as factors, never as a dense ``M^4`` array.

:class:`~mandacaru.core.augmentation.PairAugmentation` has to reproduce the
dense tensor the compensation charges used to be written out as -- added in
place, transformed factor by factor -- and the Hamiltonian built from it has
to equal the one built from the dense tensor to round-off.
"""


import numpy as np
import pytest

from mandacaru.core.augmentation import PairAugmentation


def outer_sum(factors, coupling):
    """``sum_ij K_ij F_i[p, r] F_j[q, s]`` written out."""
    return np.einsum("ij,ipr,jqs->pqrs", coupling, factors, factors)


def random_augmentation(n=5, k=4, seed=0):
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(k, n, n)) + 1j * rng.normal(size=(k, n, n))
    coupling = rng.normal(size=(k, k)) + 1j * rng.normal(size=(k, k))
    return PairAugmentation(factors, coupling)


def dense_compensation(integrals):
    """The compensation terms as the dense loop wrote them before."""
    Q = integrals.compensation_moments()
    W = integrals.compensation_potentials()
    U = integrals.compensation_coulomb()
    channels = integrals.multipole_channels()
    M = integrals.n_orbitals
    aug = np.zeros((M, M, M, M), dtype=complex)
    for i, ca in enumerate(channels):
        aug += np.einsum("pr,qs->pqrs", Q[ca], W[ca])
        aug += np.einsum("pr,qs->pqrs", W[ca], Q[ca])
        for j, cb in enumerate(channels):
            aug += U[i, j] * np.einsum("pr,qs->pqrs", Q[ca], Q[cb])
    return aug


def dense_spheres(integrals):
    """The exact one-center Coulomb tensor, sum_k lambda_k G^k (x) G^k."""
    G, values = integrals.one_center_channels()
    return np.einsum("k,kpr,kqs->pqrs", values, G, G)


def dense_long_range(integrals, omega):
    """The long-range compensation terms as the dense loop wrote them."""
    Q = integrals.compensation_moments()
    W, U = integrals.long_range_compensation(omega)
    channels = integrals.multipole_channels()
    M = integrals.n_orbitals
    out = np.zeros((M, M, M, M), dtype=complex)
    for i, channel in enumerate(channels):
        T = W[channel] + 0.5 * sum(U[i, j] * Q[other]
                                   for j, other in enumerate(channels))
        out += np.einsum("pr,qs->pqrs", Q[channel], T)
        out += np.einsum("pr,qs->pqrs", T, Q[channel])
    return out


def lowdin(tensor, X):
    return np.einsum("ap,bq,cr,ds,abcd->pqrs", X.conj(), X.conj(), X, X,
                     tensor, optimize=True)


class TestPairAugmentation:
    def test_add_to_is_the_outer_sum(self):
        aug = random_augmentation()
        base = np.random.default_rng(1).normal(size=(5,) * 4).astype(complex)
        expected = base + 0.5 * outer_sum(aug.factors, aug.coupling)
        assert np.abs(aug.add_to(base.copy(), scale=0.5) - expected).max() < 1e-12
        assert np.abs(aug.dense() - outer_sum(aug.factors, aug.coupling)
                      ).max() < 1e-12

    def test_a_transform_acts_on_each_factor(self):
        aug = random_augmentation(seed=2)
        rng = np.random.default_rng(3)
        # Rectangular, orthonormal columns: an active block of three orbitals.
        B = np.linalg.qr(rng.normal(size=(5, 3))
                         + 1j * rng.normal(size=(5, 3)))[0]
        assert np.abs(aug.transformed(B).dense() - lowdin(aug.dense(), B)
                      ).max() < 1e-12

    def test_shapes_are_checked(self):
        with pytest.raises(ValueError, match="coupling"):
            PairAugmentation(np.zeros((2, 3, 3)), np.zeros((3, 3)))
        with pytest.raises(ValueError, match="eri"):
            random_augmentation().add_to(np.zeros((4,) * 4, dtype=complex))




