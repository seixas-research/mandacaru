"""Tests for native quantum chemistry tensor factorizations."""
import numpy as np
import pytest

from mandacaru import _native
from mandacaru.backends.factorization import (
    DoubleFactorization,
    FactorizationLeaf,
    double_factorization,
    spatial_double_factorization,
)


def test_native_double_factorization_basic():
    rng = np.random.default_rng(42)
    m = 3
    h = rng.normal(size=(m, m))
    h = 0.5 * (h + h.T)
    g = rng.normal(size=(m, m, m, m))
    # Symmetrize g in chemists/physicists pairs
    g = 0.5 * (g + g.transpose(1, 0, 3, 2))

    n_modes, const, ob, ob_rot, ob_e, leaves, trunc = _native.double_factorization(
        np.ascontiguousarray(h, dtype=np.float64),
        np.ascontiguousarray(g, dtype=np.float64),
        1e-8,
        1e-10,
        None,
        1.23,
    )
    assert n_modes == m
    assert const == pytest.approx(1.23)
    assert ob.shape == (m, m)
    assert ob_rot.shape == (m, m)
    assert ob_e.shape == (m,)
    assert len(leaves) > 0
    for w, orb, coeff in leaves:
        assert isinstance(w, float)
        assert orb.shape == (m, m)
        assert coeff.shape == (m,)


def test_native_spatial_factorization_consistency():
    rng = np.random.default_rng(123)
    m, k = 4, 6
    F = rng.normal(size=(k, m, m))
    F = F + F.transpose(0, 2, 1)
    K = np.diag([1.0, 1.0, 1.0, 1.0, 0.5, 0.5])
    K[4, 5] = K[5, 4] = 0.2
    chemist = np.einsum("ij,ipr,jqs->prqs", K, F, F)
    eri = chemist.transpose(0, 2, 1, 3)
    h = rng.normal(size=(m, m))
    h = 0.5 * (h + h.T)

    df_dense = spatial_double_factorization(h, eri=eri)
    df_factored = spatial_double_factorization(h, factors=F, coupling=K)

    assert df_dense.n_modes == 2 * m
    assert df_factored.n_modes == 2 * m
    assert df_dense.measurement_bases == df_factored.measurement_bases
    assert np.allclose(df_dense.one_body, df_factored.one_body, atol=1e-12)
    assert np.allclose(
        sorted(leaf.weight for leaf in df_dense.leaves),
        sorted(leaf.weight for leaf in df_factored.leaves),
        atol=1e-10,
    )


def test_native_factorized_energy_rdm():
    rng = np.random.default_rng(99)
    m = 2
    h = rng.normal(size=(m, m))
    h = 0.5 * (h + h.T)
    g = rng.normal(size=(m, m, m, m))
    g = 0.5 * (g + g.transpose(1, 0, 3, 2))

    df = double_factorization(h, g, constant=0.5)

    one_rdm = rng.normal(size=(m, m))
    one_rdm = 0.5 * (one_rdm + one_rdm.T)
    two_rdm = rng.normal(size=(m, m, m, m))
    two_rdm = 0.5 * (two_rdm + two_rdm.transpose(1, 0, 3, 2))

    energy = df.energy(one_rdm, two_rdm)
    assert isinstance(energy, float)
    assert np.isfinite(energy)
