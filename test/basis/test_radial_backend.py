"""The C radial kernels against their reference implementations
(:mod:`mandacaru.basis.radial_backend`)."""
import numpy as np
import pytest
from scipy.linalg import eigh_tridiagonal

from mandacaru.basis import radial_backend
from mandacaru.basis.radial_backend import (numerov_inward_kernel,
                                            numerov_outward_kernel,
                                            radial_backend_status,
                                            tridiagonal_eigenpair)

pytestmark = pytest.mark.skipif(not radial_backend_status()[0],
                                reason="the C radial backend could not be built")


def _radial_matrix(n=6000, l=0, r_max=30.0):
    h = r_max / (n + 1)
    r = np.arange(1, n + 1) * h
    v = -26.0 / r * np.exp(-r) - 1.0 / (r + 1.0)
    diag = 1.0 / h ** 2 + v + l * (l + 1) / (2.0 * r * r)
    return diag, -0.5 / h ** 2 * np.ones(n - 1)


class TestTridiagonalEigenpair:
    @pytest.mark.parametrize("l, k", [(0, 0), (0, 3), (1, 1), (2, 0), (3, 0)])
    def test_it_matches_scipy(self, l, k):
        diag, off = _radial_matrix(l=l)
        values, vectors = eigh_tridiagonal(diag, off, select="i",
                                           select_range=(k, k))
        value, vector = tridiagonal_eigenpair(diag, off, k)
        reference = vectors[:, 0] * np.sign(vectors[:, 0] @ vector)
        assert value == pytest.approx(values[0], abs=1e-9)
        assert np.max(np.abs(vector - reference)) < 1e-12
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-14)

    def test_the_guess_only_says_where_to_look(self):
        diag, off = _radial_matrix()
        exact, _v = tridiagonal_eigenpair(diag, off, 2)
        for guess in (exact, exact + 1e-3, exact - 0.5, 10.0 * exact):
            value, _v = tridiagonal_eigenpair(diag, off, 2, guess=guess)
            assert value == pytest.approx(exact, abs=1e-9)

    def test_a_small_dense_check(self):
        rng = np.random.default_rng(3)
        diag, off = rng.normal(size=12), rng.normal(size=11)
        dense = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
        exact = np.linalg.eigvalsh(dense)
        for k in range(12):
            value, vector = tridiagonal_eigenpair(diag, off, k)
            assert value == pytest.approx(exact[k], abs=1e-12)
            assert np.linalg.norm(dense @ vector - value * vector) < 1e-11

    def test_a_near_degenerate_pair_gives_the_kth_level(self):
        # Two uncoupled copies of the radial matrix, one shifted by 1e-9 Ha:
        # the 1e-6 bisection bracket holds both levels, so only the Sturm
        # confirmation keeps k = 0 and k = 1 apart.
        diag, off = _radial_matrix(n=3000)
        both = np.concatenate([diag, diag + 1e-9])
        link = np.concatenate([off, [0.0], off])
        exact = eigh_tridiagonal(both, link, select="i",
                                 select_range=(0, 1))[0]
        for k in (0, 1):
            value, vector = tridiagonal_eigenpair(both, link, k)
            assert value == pytest.approx(exact[k], abs=1e-10)
            assert np.all(np.isfinite(vector))

    def test_a_short_off_diagonal_is_refused(self):
        diag, off = _radial_matrix(n=100)
        with pytest.raises(ValueError, match="off-diagonal"):
            tridiagonal_eigenpair(diag, off[:-1], 0)


class TestNumerov:
    def _reference_outward(self, f, s, h2, start, u):
        a = 1.0 - h2 * f / 12.0
        b = 2.0 * (1.0 + 5.0 * h2 * f / 12.0)
        c = h2 / 12.0
        for i in range(start + 1, u.size - 1):
            u[i + 1] = (b[i] * u[i] - a[i - 1] * u[i - 1]
                        + c * (s[i + 1] + 10.0 * s[i] + s[i - 1])) / a[i + 1]
        return u

    def test_outward_is_the_reference_recursion(self):
        rng = np.random.default_rng(5)
        f, s = rng.normal(size=500), rng.normal(size=500)
        seed = np.zeros(500)
        seed[1], seed[2] = 0.1, 0.2
        c = numerov_outward_kernel(f, s, 1e-4, 1, seed.copy())
        ref = self._reference_outward(f, s, 1e-4, 1, seed.copy())
        assert np.max(np.abs(c - ref)) <= 1e-12 * np.max(np.abs(ref))

    def test_inward_is_the_reference_recursion(self):
        rng = np.random.default_rng(6)
        f = np.abs(rng.normal(size=400))
        seed = np.zeros(400)
        seed[-1], seed[-2] = 1e-8, 2e-8
        c = numerov_inward_kernel(f, 1e-4, 10, seed.copy())
        a = 1.0 - 1e-4 * f / 12.0
        b = 2.0 * (1.0 + 5.0 * 1e-4 * f / 12.0)
        ref = seed.copy()
        for i in range(398, 10, -1):
            ref[i - 1] = (b[i] * ref[i] - a[i + 1] * ref[i + 1]) / a[i - 1]
        assert np.max(np.abs(c - ref)) <= 1e-12 * np.max(np.abs(ref))


def test_the_python_policy_uses_the_reference_kernels(monkeypatch):
    monkeypatch.setenv("MANDACARU_BACKEND", "numpy")
    uses_c, message = radial_backend.radial_backend_status()
    assert not uses_c and "numpy" in message
    diag, off = _radial_matrix(n=800)
    value, _v = tridiagonal_eigenpair(diag, off, 0)
    values, _vs = eigh_tridiagonal(diag, off, select="i", select_range=(0, 0))
    assert value == values[0]
