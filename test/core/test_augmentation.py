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

import warnings

import numpy as np
import pytest
from ase import Atoms

from mandacaru.algorithms._hamiltonian_from_atoms import build_basis_hamiltonian
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


@pytest.fixture(scope="module")
def water_paw():
    atoms = Atoms("H2O", positions=[[0.0, 0.757, 0.587],
                                    [0.0, -0.757, 0.587],
                                    [0.0, 0.0, 0.0]], cell=[6.0] * 3)
    atoms.center()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        context = build_basis_hamiltonian(
            atoms, {"name": "PAW-LCAO", "size": "SZ"}, None, 0.3, 0, None)[4]
    return context["integrals"]


class TestPAWAugmentation:
    def test_the_structured_terms_are_the_dense_ones(self, water_paw):
        aug = water_paw.two_body_augmentation()
        assert isinstance(aug, PairAugmentation)
        M = water_paw.n_orbitals
        # k M^2 numbers: the compensation channels' Q and W, and the exact
        # spheres' factors (rank 33 for O, 3 per H; HISTORY.md 2026-10-12).
        # At M = 6 that is more than the M^4 tensor; it grows as M^2.
        k = (2 * len(water_paw.multipole_channels())
             + water_paw.one_center_factors().rank)
        assert aug.factors.shape == (k, M, M)
        assert aug.nbytes == 8 * (k * M * M + k * k)
        assert np.abs(aug.dense() - dense_compensation(water_paw)
                      - dense_spheres(water_paw)).max() < 1e-12
        omega = 0.11
        assert np.abs(water_paw.long_range_augmentation(omega).dense()
                      - dense_long_range(water_paw, omega)).max() < 1e-12

    def test_the_tensors_equal_the_dense_route(self, water_paw):
        grid = water_paw._engine.two_body(energy_units="Ha")
        dense_ao = grid + dense_compensation(water_paw) + dense_spheres(water_paw)
        assert np.abs(water_paw.ao_two_body() - dense_ao).max() < 1e-12
        X = water_paw._lowdin_x()
        reference = lowdin(dense_ao, X)
        assert np.abs(water_paw.two_body() - reference).max() < 1e-12
        # The molecular-orbital tensor, and the factors transformed straight
        # to it instead of through the dense augmentation.
        V = X @ water_paw.hartree_fock(8).mo_coefficients
        aug = water_paw.two_body_augmentation().transformed(V)
        mo_structured = aug.add_to(lowdin(grid, V))
        assert np.abs(mo_structured - lowdin(dense_ao, V)).max() < 1e-12
