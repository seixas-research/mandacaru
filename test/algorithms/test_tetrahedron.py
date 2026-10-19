# -*- coding: utf-8 -*-
# file: test/algorithms/test_tetrahedron.py

# This code is part of Mandacaru.
# MIT License

"""The linear tetrahedron method: one tetrahedron, a whole mesh, and the
model bands whose densities of states are known."""

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import erfc

from mandacaru.algorithms import tetrahedron as tt

#: Sorted, distinct corner energies used throughout.
CORNERS = np.array([0.0, 1.0, 2.5, 4.0])

#: Corner energies with every kind of coincidence.
DEGENERATE = [(0, 0, 1, 2), (0, 1, 1, 2), (0, 1, 2, 2), (0, 0, 0, 1),
              (0, 1, 1, 1), (0, 0, 1, 1), (1, 1, 1, 1)]


def analytic_dos(E, e):
    """The tetrahedron DOS of sorted corners ``e``: piecewise quadratic."""
    e1, e2, e3, e4 = e
    E = np.asarray(E, dtype=float)
    out = np.zeros_like(E)
    one = (E > e1) & (E < e2)
    out[one] = 3 * (E[one] - e1) ** 2 / ((e2 - e1) * (e3 - e1) * (e4 - e1))
    two = (E >= e2) & (E < e3)
    b = E[two] - e2
    out[two] = (3 * (e2 - e1) + 6 * b - 3 * ((e3 - e1) + (e4 - e2)) * b ** 2
                / ((e3 - e2) * (e4 - e2))) / ((e3 - e1) * (e4 - e1))
    three = (E >= e3) & (E < e4)
    out[three] = 3 * (e4 - E[three]) ** 2 / ((e4 - e1) * (e4 - e2)
                                             * (e4 - e3))
    return out


def piecewise_integral(f, breaks):
    """``int f`` exactly for a piecewise quadratic ``f`` (vectorized, any
    trailing shape) with the given breakpoints: 3-point Gauss-Legendre."""
    x, w = np.polynomial.legendre.leggauss(3)
    breaks = np.unique(breaks)
    lo, hi = breaks[:-1], breaks[1:]
    nodes = (0.5 * (hi - lo)[:, None] * x[None] + 0.5 * (hi + lo)[:, None])
    values = f(nodes.ravel())
    values = values.reshape(nodes.shape + values.shape[1:])
    scale = (0.5 * (hi - lo)[:, None] * w[None])
    return np.tensordot(scale, values, axes=([0, 1], [0, 1]))


def single(E, corners, correction):
    w, dw = tt.corner_weights(E, np.asarray(corners, float)[None],
                              curvature_correction=correction)
    return w[:, 0], dw[:, 0]


# --------------------------------------------------------------------------- #

class TestOneTetrahedron:
    def test_the_dos_is_the_analytic_piecewise_quadratic(self):
        E = np.linspace(-1.0, 5.0, 3001)
        _w, dw = single(E, CORNERS, False)
        assert np.abs(dw.sum(axis=1) - analytic_dos(E, CORNERS)).max() \
            < 1e-13

    @pytest.mark.parametrize("correction", [False, True])
    def test_the_dos_integrates_to_one(self, correction):
        """Exactly, by a quadrature exact on every quadratic piece; each
        corner's DOS weight integrates to a quarter, corrected or not."""
        breaks = np.concatenate([[-1.0], CORNERS, [5.0]])
        corners = piecewise_integral(
            lambda E: single(E, CORNERS, correction)[1], breaks)
        assert np.abs(corners - 0.25).max() < 1e-13

    def test_the_count_rises_from_zero_to_one_continuously(self):
        E = np.linspace(-1.0, 5.0, 6001)
        w, _dw = single(E, CORNERS, False)
        n = w.sum(axis=1)
        assert np.all(n[E <= CORNERS[0]] == 0.0)
        assert np.all(n[E >= CORNERS[-1]] == 1.0)
        assert np.all(np.diff(n) >= -1e-15)
        # Continuous: no step larger than the steepest DOS allows.
        step = E[1] - E[0]
        assert np.abs(np.diff(n)).max() < 1.01 * analytic_dos(
            E, CORNERS).max() * step

    @pytest.mark.parametrize("correction", [False, True])
    def test_corner_weights_sum_to_the_tetrahedron(self, correction):
        """The four weights sum to the count and the DOS -- the correction,
        which sums to zero over the corners, changes neither."""
        E = np.linspace(-1.0, 5.0, 601)
        w, dw = single(E, CORNERS, correction)
        reference_w, reference_dw = single(E, CORNERS, False)
        assert np.abs(w.sum(axis=1) - reference_w.sum(axis=1)).max() < 1e-14
        assert np.abs(dw.sum(axis=1) - analytic_dos(E, CORNERS)).max() \
            < 1e-13

    @pytest.mark.parametrize("correction", [False, True])
    def test_the_dos_weights_are_the_derivatives_of_the_weights(self,
                                                                 correction):
        E = np.linspace(-0.9, 4.9, 2901)
        away = np.min(np.abs(E[:, None] - CORNERS[None]), axis=1) > 1e-3
        h = 1e-6
        w_plus, _ = single(E + h, CORNERS, correction)
        w_minus, _ = single(E - h, CORNERS, correction)
        _w, dw = single(E, CORNERS, correction)
        derivative = (w_plus - w_minus) / (2 * h)
        assert np.abs(derivative - dw)[away].max() < 1e-7

    def test_the_corner_weights_are_the_barycentric_integrals(self):
        r"""``w_i(E)`` is the integral of the barycentric coordinate
        :math:`\lambda_i` over the part below ``E``: its first moment is the
        band energy below ``E`` exactly, and a sampled tetrahedron agrees
        with each corner."""
        breaks = np.concatenate([[-1.0], CORNERS])
        for E in (0.5, 1.7, 3.2):
            w, _dw = single([E], CORNERS, False)
            moment = piecewise_integral(
                lambda x: x * analytic_dos(x, CORNERS),
                np.concatenate([breaks[breaks < E], [E]]))
            assert w[0] @ CORNERS == pytest.approx(moment, abs=1e-13)
        rng = np.random.default_rng(7)
        points = rng.dirichlet(np.ones(4), 400_000)
        below = points @ CORNERS < 1.7
        sampled = points[below].sum(axis=0) / len(points)
        w, _dw = single([1.7], CORNERS, False)
        assert np.abs(w[0] - sampled).max() < 2e-3

    def test_the_weights_follow_the_corners_in_any_order(self):
        E = np.linspace(-1.0, 5.0, 121)
        order = [2, 0, 3, 1]
        w, dw = single(E, CORNERS, True)
        w2, dw2 = single(E, CORNERS[order], True)
        assert np.allclose(w2, w[:, order], atol=1e-15)
        assert np.allclose(dw2, dw[:, order], atol=1e-15)

    @pytest.mark.parametrize("corners", DEGENERATE)
    def test_degenerate_corners_stay_finite_and_continuous(self, corners):
        corners = np.asarray(corners, float)
        E = np.linspace(-0.5, 2.5, 3001)
        for correction in (False, True):
            w, dw = single(E, corners, correction)
            assert np.all(np.isfinite(w)) and np.all(np.isfinite(dw))
        w, _dw = single(E, corners, False)
        n = w.sum(axis=1)
        assert n[0] == 0.0 and n[-1] == 1.0
        if np.ptp(corners) > 0:
            # A finite DOS: no jump in the count, and each corner's DOS
            # weight integrates to a quarter.
            assert np.abs(np.diff(n)).max() < 0.01
            breaks = np.concatenate([[-0.5], corners, [2.5]])
            total = piecewise_integral(
                lambda x: single(x, corners, False)[1], breaks)
            assert np.abs(total - 0.25).max() < 1e-12

    @pytest.mark.parametrize("corners", DEGENERATE)
    @pytest.mark.parametrize("correction", [False, True])
    def test_the_cell_average_keeps_what_has_no_width(self, corners,
                                                      correction):
        """Four equal corners are a delta of the whole tetrahedron; three
        make the DOS, and so the corrected weights, step.  The pointwise DOS
        weights miss both (the corrected corners of (0, 0, 0, 1) integrate
        to 0.175 and 0.475); the cell average holds a quarter per corner."""
        corners = np.asarray(corners, float)
        E = np.linspace(-0.5, 2.5, 301)
        width = E[1] - E[0]
        average = tt.tetrahedron_dos(
            E, corners[:, None], [[0, 1, 2, 3]], np.eye(4)[:, :, None],
            curvature_correction=correction, cell_average=True)
        assert np.abs(average.sum(axis=0) * width - 0.25).max() < 1e-12

    @pytest.mark.parametrize("corners", DEGENERATE[:6])
    def test_degenerate_corners_are_the_limit_of_split_ones(self, corners):
        corners = np.asarray(corners, float)
        split = corners + 1e-9 * np.arange(4)
        E = np.linspace(-0.5, 2.5, 301)
        away = np.min(np.abs(E[:, None] - corners[None]), axis=1) > 1e-6
        for correction in (False, True):
            w, dw = single(E, corners, correction)
            w2, dw2 = single(E, split, correction)
            assert np.abs(w - w2)[away].max() < 1e-7
            assert np.abs(dw - dw2)[away].max() < 1e-6


# --------------------------------------------------------------------------- #

class TestTheDecomposition:
    def test_six_tetrahedra_of_equal_volume_fill_each_subcell(self):
        tetrahedra = tt.mesh_tetrahedra((3, 3, 3))
        assert tetrahedra.shape == (6 * 27, 4)
        # The subcell at the origin: corners in {0, 1}^3, no wrap.
        cell = np.stack(np.unravel_index(tetrahedra[:6], (3, 3, 3)), axis=-1)
        edges = cell[:, 1:] - cell[:, :1]
        assert np.allclose(np.abs(np.linalg.det(edges)), 1.0)
        rng = np.random.default_rng(1)
        x = rng.random((20000, 3))
        inside = np.zeros(len(x), dtype=int)
        for corner in cell:
            A = (corner[1:] - corner[0]).T.astype(float)
            lam = np.linalg.solve(A, (x - corner[0]).T).T
            inside += np.all(lam >= 0, axis=1) & (lam.sum(axis=1) <= 1)
        assert np.all(inside == 1)

    def test_corners_wrap_periodically(self):
        tetrahedra = tt.mesh_tetrahedra((2, 3, 4))
        assert tetrahedra.min() == 0 and tetrahedra.max() == 23
        # Every mesh point is a corner of 24 tetrahedra.
        assert np.all(np.bincount(tetrahedra.ravel()) == 24)

    def test_the_shortest_main_diagonal_is_shared(self):
        """A bcc lattice's reciprocal is fcc: its subcell's shortest diagonal
        is (1,0,0)-(0,1,1), and all six tetrahedra share it; for the cube any
        diagonal will do and (0,0,0)-(1,1,1) is kept."""
        from ase.build import bulk
        reciprocal = bulk("Fe", "bcc", a=2.87).cell.reciprocal()
        size = (3, 3, 3)
        cell = tt.mesh_tetrahedra(size, reciprocal)[:6]
        ends = (np.ravel_multi_index((1, 0, 0), size),
                np.ravel_multi_index((0, 1, 1), size))
        assert all(set(ends) <= set(t) for t in cell)
        cube = tt.mesh_tetrahedra(size, np.eye(3))[:6]
        assert all({0, np.ravel_multi_index((1, 1, 1), size)} <= set(t)
                   for t in cube)


# --------------------------------------------------------------------------- #

def random_bands(size, n_bands, seed=0):
    """Smooth random bands on a mesh: a few Fourier terms each."""
    rng = np.random.default_rng(seed)
    k = (np.indices(size).reshape(3, -1).T + 0.5) / np.asarray(size)
    bands = []
    for _ in range(n_bands):
        G = rng.integers(-1, 2, (4, 3))
        c = rng.normal(size=4)
        bands.append(rng.normal() + np.cos(2 * np.pi * k @ G.T) @ c)
    return np.sort(np.array(bands).T, axis=1)


class TestTheMesh:
    SIZE = (5, 4, 3)

    def test_the_count_runs_from_zero_to_the_number_of_bands(self):
        bands = random_bands(self.SIZE, 3)
        tetrahedra = tt.mesh_tetrahedra(self.SIZE)
        n = tt.tetrahedron_dos([bands.min() - 1e-9, bands.max() + 1e-9],
                               bands, tetrahedra, integrated=True)
        assert n[0] == 0.0
        assert n[1] == pytest.approx(3.0, abs=1e-12)

    def test_the_dos_integrates_to_the_count(self):
        """Between consecutive mesh energies the DOS is one quadratic, so
        the quadrature is exact: its running integral is the count to 1e-12
        at every breakpoint."""
        bands = random_bands(self.SIZE, 2, seed=3)
        tetrahedra = tt.mesh_tetrahedra(self.SIZE)
        breaks = np.unique(bands)
        x, w = np.polynomial.legendre.leggauss(3)
        lo, hi = breaks[:-1], breaks[1:]
        nodes = 0.5 * (hi - lo)[:, None] * x + 0.5 * (hi + lo)[:, None]
        g = tt.tetrahedron_dos(nodes.ravel(), bands, tetrahedra
                               ).reshape(nodes.shape)
        running = np.concatenate([[0.0], np.cumsum(
            (g * w).sum(axis=1) * 0.5 * (hi - lo))])
        count = tt.tetrahedron_dos(breaks, bands, tetrahedra, integrated=True)
        assert np.abs(running - count).max() < 1e-12
        assert running[-1] == pytest.approx(2.0, abs=1e-12)

    @pytest.mark.parametrize("correction", [False, True])
    def test_complete_projections_sum_to_the_total(self, correction):
        bands = random_bands(self.SIZE, 3, seed=5)
        rng = np.random.default_rng(2)
        weights = rng.random((len(bands), 4, 3))
        weights /= weights.sum(axis=1, keepdims=True)
        tetrahedra = tt.mesh_tetrahedra(self.SIZE)
        E = np.linspace(bands.min() - 0.5, bands.max() + 0.5, 401)
        total = tt.tetrahedron_dos(E, bands, tetrahedra)
        projected = tt.tetrahedron_dos(E, bands, tetrahedra, weights,
                                       curvature_correction=correction)
        assert projected.shape == (401, 4)
        assert np.abs(projected.sum(axis=1) - total).max() < 1e-12
        counted = tt.tetrahedron_dos(E, bands, tetrahedra, weights,
                                     curvature_correction=correction,
                                     integrated=True)
        assert np.abs(counted.sum(axis=1) - tt.tetrahedron_dos(
            E, bands, tetrahedra, integrated=True)).max() < 1e-12

    def test_chunking_and_energy_order_change_nothing(self, monkeypatch):
        bands = random_bands(self.SIZE, 3, seed=4)
        weights = np.random.default_rng(0).random((len(bands), 2, 3))
        tetrahedra = tt.mesh_tetrahedra(self.SIZE)
        E = np.linspace(bands.min() - 0.2, bands.max() + 0.2, 300)
        shuffle = np.random.default_rng(1).permutation(len(E))
        reference = [tt.tetrahedron_dos(E, bands, tetrahedra, weights,
                                        integrated=flag)
                     for flag in (False, True)]
        monkeypatch.setattr(tt, "CHUNK_ELEMENTS", 300)
        for flag, expected in zip((False, True), reference):
            small = tt.tetrahedron_dos(E[shuffle], bands, tetrahedra, weights,
                                       integrated=flag)
            assert np.abs(small - expected[shuffle]).max() < 1e-13

    def test_the_cell_average_conserves_the_count(self):
        """On an uneven axis the cell averages times the cell widths add up
        to the count between the outer edges, projections included."""
        bands = random_bands(self.SIZE, 3, seed=6)
        weights = np.random.default_rng(3).random((len(bands), 2, 3))
        tetrahedra = tt.mesh_tetrahedra(self.SIZE)
        E = np.sort(np.random.default_rng(4).uniform(
            bands.min() - 0.3, bands.max() + 0.3, 400))
        edges = np.concatenate([[1.5 * E[0] - 0.5 * E[1]],
                                0.5 * (E[1:] + E[:-1]),
                                [1.5 * E[-1] - 0.5 * E[-2]]])
        average = tt.tetrahedron_dos(E, bands, tetrahedra, weights,
                                     cell_average=True)
        count = tt.tetrahedron_dos(edges[[0, -1]], bands, tetrahedra,
                                   weights, integrated=True)
        assert np.abs(average.T @ np.diff(edges) - (count[1] - count[0])
                      ).max() < 1e-12
        with pytest.raises(ValueError, match="distinct"):
            tt.tetrahedron_dos([0.0, 0.0], bands, tetrahedra,
                               cell_average=True)
        with pytest.raises(ValueError, match="already exact"):
            tt.tetrahedron_dos(E, bands, tetrahedra, integrated=True,
                               cell_average=True)

    def test_a_degenerate_set_shares_its_projections(self):
        """Two states 1e-5 apart split one way or another between two
        orbitals; averaged, any split gives the same projections, and a
        level apart keeps its own."""
        levels = np.array([[0.0, 1e-5, 2.0]])
        one = np.array([[[1.0, 0.0, 0.3], [0.0, 1.0, 0.7]]])
        other = np.array([[[0.5, 0.5, 0.3], [0.5, 0.5, 0.7]]])
        a = tt.degenerate_average(levels, one, 1e-3)
        b = tt.degenerate_average(levels, other, 1e-3)
        assert np.allclose(a, b, atol=1e-15)
        assert np.allclose(a[0, :, 0], [0.5, 0.5])
        assert np.allclose(a[0, :, 2], [0.3, 0.7])
        assert np.array_equal(tt.degenerate_average(levels, one, 1e-6), one)

    def test_malformed_input_is_refused(self):
        tetrahedra = tt.mesh_tetrahedra((2, 2, 2))
        with pytest.raises(ValueError, match="N_k, n_bands"):
            tt.tetrahedron_dos([0.0], np.zeros(8), tetrahedra)
        with pytest.raises(ValueError, match="outside the mesh"):
            tt.tetrahedron_dos([0.0], np.zeros((4, 1)), tetrahedra)
        with pytest.raises(ValueError, match="projections"):
            tt.tetrahedron_dos([0.0], np.zeros((8, 1)), tetrahedra,
                               np.zeros((8, 1)))
        with pytest.raises(ValueError, match="positive"):
            tt.mesh_tetrahedra((0, 2, 2))


# --------------------------------------------------------------------------- #

def free_electrons(n):
    """``E = |k|^2/2`` on a Gamma-centered mesh of the unit cube zone
    (``k`` folded into it), and the projection ``cos^2(pi k_x)``."""
    k = np.indices((n,) * 3).reshape(3, -1).T / n
    k -= np.round(k)
    return (0.5 * (k ** 2).sum(axis=1)[:, None],
            np.cos(np.pi * k[:, 0])[:, None, None] ** 2,
            tt.mesh_tetrahedra((n,) * 3, np.eye(3)))


class TestFreeElectrons:
    r"""Below the zone-boundary energy 1/8 the isoenergy surfaces are spheres:
    :math:`g = 4\pi\sqrt{2E}`, :math:`N = (4\pi/3)(2E)^{3/2}`, the band
    energy :math:`(8\sqrt2\pi/5) E^{5/2}`, and the projection
    :math:`\cos^2\pi k_x` averages :math:`(1 + \operatorname{sinc} 2\pi k)/2`
    over a sphere."""

    EF = 0.08
    COUNT = 4 * np.pi / 3 * (2 * EF) ** 1.5
    BAND_ENERGY = 8 * np.sqrt(2) * np.pi / 5 * EF ** 2.5

    @staticmethod
    def projected(E):
        k = np.sqrt(2 * E)
        return 2 * np.pi * k * (1 + np.sin(2 * np.pi * k) / (2 * np.pi * k))

    def fermi_level(self, bands, tetrahedra):
        """The level holding the exact count, from the uncorrected count."""
        return brentq(lambda mu: tt.tetrahedron_dos(
            [mu], bands, tetrahedra, integrated=True)[0] - self.COUNT,
            0.06, 0.1, xtol=1e-14)

    def test_the_dos_approaches_the_square_root_law(self):
        """Relative L1 error on [0.02, 0.1]: 3.7e-2, 8.9e-3, 2.2e-3 at
        n = 8, 16, 32 (second order)."""
        E = np.linspace(0.02, 0.1, 401)
        exact = 4 * np.pi * np.sqrt(2 * E)
        errors = []
        for n in (8, 16, 32):
            bands, _p, tetrahedra = free_electrons(n)
            g = tt.tetrahedron_dos(E, bands, tetrahedra)
            errors.append(np.trapezoid(np.abs(g - exact), E)
                          / np.trapezoid(exact, E))
        assert errors[0] > errors[1] > errors[2]
        assert errors[2] < 5e-3
        assert errors[0] / errors[2] > 8.0

    def test_the_count_converges_at_second_order(self):
        errors = []
        for n in (12, 24):
            bands, _p, tetrahedra = free_electrons(n)
            errors.append(tt.tetrahedron_dos(
                [self.EF], bands, tetrahedra, integrated=True)[0]
                - self.COUNT)
        # -8.9e-3 and -2.2e-3: linear interpolation overestimates a convex
        # band, so it undercounts.
        assert errors[0] < errors[1] < 0
        assert 3.5 < errors[0] / errors[1] < 4.5

    def test_the_correction_raises_the_order_of_the_band_energy(self):
        """The Fermi level fixed by the electron count (from the uncorrected
        count, which the correction does not change): the band energy is
        second order uncorrected (3.6e-2, 9.0e-3 relative at n = 12, 24) and
        about fourth order corrected (9.2e-4, 4.8e-5)."""
        errors = {False: [], True: []}
        for n in (12, 24):
            bands, _p, tetrahedra = free_electrons(n)
            mu = self.fermi_level(bands, tetrahedra)
            for correction in errors:
                energy = tt.tetrahedron_dos(
                    [mu], bands, tetrahedra, bands[:, None, :],
                    curvature_correction=correction, integrated=True)[0, 0]
                errors[correction].append(
                    abs(energy - self.BAND_ENERGY) / self.BAND_ENERGY)
        plain, corrected = errors[False], errors[True]
        assert 3.5 < plain[0] / plain[1] < 4.5
        assert corrected[0] / corrected[1] > 10.0
        assert corrected[1] < plain[1] / 50

    def test_the_correction_improves_a_projected_dos(self):
        """``cos^2(pi k_x)`` at n = 16: the relative L1 error of the PDOS on
        [0.02, 0.1] falls from 8.0e-3 to 5.8e-3 with the correction, and the
        projection's Fermi-sea integral at the fixed count from 4.2e-3 to
        4e-5."""
        bands, p, tetrahedra = free_electrons(16)
        E = np.linspace(0.02, 0.1, 401)
        exact = self.projected(E)
        error = {}
        for correction in (False, True):
            g = tt.tetrahedron_dos(E, bands, tetrahedra, p,
                                   curvature_correction=correction)[:, 0]
            error[correction] = (np.trapezoid(np.abs(g - exact), E)
                                 / np.trapezoid(exact, E))
        assert error[True] < 0.8 * error[False]
        mu = self.fermi_level(bands, tetrahedra)
        sea = piecewise_integral(self.projected, np.linspace(1e-12,
                                                             self.EF, 200))
        filled = {c: tt.tetrahedron_dos([mu], bands, tetrahedra, p,
                                        curvature_correction=c,
                                        integrated=True)[0, 0]
                  for c in (False, True)}
        assert abs(filled[True] - sea) < abs(filled[False] - sea) / 20


# --------------------------------------------------------------------------- #

def simple_cubic(n):
    """``E = -2(cos kx + cos ky + cos kz)`` (t = 1) on a Gamma-centered mesh."""
    k = np.indices((n,) * 3).reshape(3, -1).T / n
    return (-2 * np.cos(2 * np.pi * k).sum(axis=1))[:, None]


def gaussian_dos(E, levels, width, weights=None):
    """The Gaussian-broadened DOS (per k-point, one spin) of ``levels``
    ``(N_k, n)``, projected on ``weights`` ``(N_k, M, n)`` if given."""
    flat = levels.ravel()
    columns = (np.ones((flat.size, 1)) if weights is None
               else np.moveaxis(weights, 1, 2).reshape(flat.size, -1))
    out = np.zeros((len(E), columns.shape[1]))
    for start in range(0, flat.size, 20000):
        x = (E[:, None] - flat[None, start:start + 20000]) / width
        out += np.exp(-0.5 * x * x) @ columns[start:start + 20000]
    out /= len(levels) * width * np.sqrt(2 * np.pi)
    return out[:, 0] if weights is None else out


def smoothed(g, E, width):
    """``g`` on the uniform axis ``E`` convolved with a Gaussian."""
    step = E[1] - E[0]
    x = np.arange(-int(6 * width / step), int(6 * width / step) + 1) * step
    kernel = np.exp(-0.5 * (x / width) ** 2)
    kernel /= kernel.sum()
    if g.ndim == 1:
        return np.convolve(g, kernel, mode="same")
    return np.stack([np.convolve(c, kernel, mode="same") for c in g.T], 1)


@pytest.fixture(scope="module")
def cubic_reference():
    """The tetrahedron DOS of the 32^3 mesh: converged to an L1 distance of
    2.5e-3 from the 48^3 one."""
    E = np.linspace(-8.0, 8.0, 801)
    return E, tt.tetrahedron_dos(E, simple_cubic(32),
                                 tt.mesh_tetrahedra((32,) * 3, np.eye(3)))


class TestSimpleCubicTightBinding:
    def test_band_edges_symmetry_and_half_filling(self):
        """Exactly zero outside +-6t; on an even Gamma-centered mesh the
        shift by (1/2, 1/2, 1/2) maps E to -E point by point and tetrahedron
        by tetrahedron, so the DOS is symmetric and the count at E = 0 is
        one half, to round-off."""
        tetrahedra = tt.mesh_tetrahedra((8,) * 3, np.eye(3))
        bands = simple_cubic(8)
        E = np.linspace(-7.0, 7.0, 1401)
        g = tt.tetrahedron_dos(E, bands, tetrahedra)
        assert np.all(g[np.abs(E) >= 6.0] == 0.0)
        assert np.all(g[np.abs(E) < 5.5] > 0.0)
        assert np.abs(g - g[::-1]).max() < 1e-13
        assert tt.tetrahedron_dos([0.0], bands, tetrahedra, integrated=True
                                  )[0] == pytest.approx(0.5, abs=1e-13)

    def test_the_converged_dos_matches_a_dense_gaussian_baseline(
            self, cubic_reference):
        """The 48^3 mesh broadened by 0.2 t against the 32^3 tetrahedron DOS
        convolved with the same Gaussian: 0.4 % of the peak at most."""
        E, reference = cubic_reference
        baseline = gaussian_dos(E, simple_cubic(48), 0.2)
        error = np.abs(smoothed(reference, E, 0.2) - baseline).max()
        assert error < 6e-3 * baseline.max()

    def test_a_coarse_mesh_resolves_better_by_tetrahedra(self,
                                                          cubic_reference):
        """At one resolution -- a Gaussian of 0.2 t -- from the 6^3 mesh:
        broadening its tetrahedron DOS is 6x closer to the converged
        spectrum at that resolution (L1 distance 0.091) than broadening its
        levels (0.59), and only the tetrahedra keep the band edges: the
        broadened levels put 0.5 % of the states beyond +-6t."""
        E, reference = cubic_reference
        target = smoothed(reference, E, 0.2)
        bands = simple_cubic(6)
        tetra = tt.tetrahedron_dos(E, bands, tt.mesh_tetrahedra((6,) * 3,
                                                                np.eye(3)))
        levels = gaussian_dos(E, bands, 0.2)
        distance_tetra = np.trapezoid(np.abs(smoothed(tetra, E, 0.2)
                                             - target), E)
        distance_levels = np.trapezoid(np.abs(levels - target), E)
        assert distance_tetra < distance_levels / 4
        outside = np.abs(E) > 6.0
        assert np.all(tetra[outside] == 0.0)
        leak = 0.5 * erfc((6.0 - bands.ravel()) / (0.2 * np.sqrt(2))).mean()
        assert 2 * leak > 0.004


class TestTwoBandProjections:
    """Two orbitals, ``-2 sum cos k`` and ``1 + (sum cos k)/2``, coupled by
    ``0.6 + 0.5 cos kx``: each band's orbital character varies across the
    zone, so the corner-resolved weights are what carries it."""

    @staticmethod
    def model(n):
        c = np.cos(2 * np.pi * np.indices((n,) * 3).reshape(3, -1).T / n)
        H = np.zeros((len(c), 2, 2))
        H[:, 0, 0] = -2 * c.sum(axis=1)
        H[:, 1, 1] = 1.0 + 0.5 * c.sum(axis=1)
        H[:, 0, 1] = H[:, 1, 0] = 0.6 + 0.5 * c[:, 0]
        levels, vectors = np.linalg.eigh(H)
        return levels, np.abs(vectors) ** 2       # (N_k, 2), (N_k, M, n)

    def test_the_projections_against_a_dense_gaussian_sum(self):
        """Broadened by 0.2, the corrected corner-resolved PDOS of the 12^3
        and 24^3 meshes is within an L1 distance of (0.053, 0.060) and
        (0.015, 0.016) -- of one state per orbital -- of the brute-force
        Gaussian sum over the 40^3 mesh: second order onto the brute-force
        answer (uncorrected: (0.036, 0.071) and (0.010, 0.020)).  The
        anticrossing curves the bands sharply, so here the correction trades
        error between the orbitals rather than removing it."""
        E = np.linspace(-8.0, 6.0, 701)
        levels, weights = self.model(40)
        baseline = gaussian_dos(E, levels, 0.2, weights)
        distance = {}
        for n in (12, 24):
            levels, weights = self.model(n)
            tetrahedra = tt.mesh_tetrahedra((n,) * 3, np.eye(3))
            pdos = tt.tetrahedron_dos(E, levels, tetrahedra, weights)
            assert np.abs(pdos.sum(axis=1) - tt.tetrahedron_dos(
                E, levels, tetrahedra)).max() < 1e-12
            distance[n] = np.trapezoid(np.abs(smoothed(pdos, E, 0.2)
                                              - baseline), E, axis=0)
        assert np.all(distance[24] < 0.02)
        assert np.all(distance[12] / distance[24] > 3.0)
