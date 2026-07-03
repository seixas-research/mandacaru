# -*- coding: utf-8 -*-
# file: test_integrals.py

import numpy as np
import pytest

from carcara.basis import HydrogenicOrbital
from carcara.integrals import Grid, IntegralEngine, PoissonFFTSolver


@pytest.fixture
def h1s_engine():
    grid = Grid(center=[0.0, 0.0, 0.0], box_size=12.0, points=48)
    basis = [HydrogenicOrbital(1, 0, 0, Z=1.0)]
    return IntegralEngine(basis, grid)


class TestPoissonFFT:
    def test_1s_self_repulsion_matches_exact(self, h1s_engine):
        # (1s 1s | 1s 1s) = 5/8 Ha exactly for the hydrogen 1s orbital.
        J = h1s_engine.two_body(method="fft")[0, 0, 0, 0].real
        assert abs(J - 0.625) < 0.02

    def test_converges_toward_exact_with_resolution(self):
        errs = []
        for pts in (24, 48):
            grid = Grid(center=[0, 0, 0], box_size=12.0, points=pts)
            eng = IntegralEngine([HydrogenicOrbital(1, 0, 0, Z=1.0)], grid)
            J = eng.two_body(method="fft")[0, 0, 0, 0].real
            errs.append(abs(J - 0.625))
        assert errs[1] < errs[0]  # finer grid is closer to the exact value

    def test_eri_is_real_and_positive_diagonal(self, h1s_engine):
        J = h1s_engine.two_body(method="fft")[0, 0, 0, 0]
        assert abs(J.imag) < 1e-10
        assert J.real > 0

    def test_eight_fold_symmetry(self):
        grid = Grid(center=[0, 0, 0], box_size=8.0, points=20)
        basis = [HydrogenicOrbital(1, 0, 0, Z=1.0),
                 HydrogenicOrbital(2, 1, 0, Z=1.0)]
        eri = IntegralEngine(basis, grid).two_body(method="fft")
        # (ab|cd) == (cd|ab) and (ab|cd) == (ba|dc)* for a real basis set.
        assert np.allclose(eri, np.transpose(eri, (2, 3, 0, 1)))
        assert np.allclose(eri, np.conj(np.transpose(eri, (1, 0, 3, 2))))


class TestPoissonSolverDirect:
    def test_solver_matches_direct_convolution_on_tiny_grid(self):
        # On a tiny grid, compare the FFT solver to the explicit O(N^2) sum
        # with the SAME self-energy on the diagonal cell -> must agree.
        grid = Grid(center=[0, 0, 0], box_size=3.0, points=6)
        rng = np.random.default_rng(0)
        rho = (rng.standard_normal(grid.size)
               + 1j * rng.standard_normal(grid.size))

        solver = PoissonFFTSolver(grid.points, grid.dx)
        phi_fft = solver.solve(rho)

        xg, yg, zg = grid.flat_coords()
        n = grid.size
        phi_ref = np.zeros(n, dtype=complex)
        G0 = 2.3800756 / grid.dx
        for i in range(n):
            d = np.sqrt((xg[i] - xg) ** 2 + (yg[i] - yg) ** 2
                        + (zg[i] - zg) ** 2)
            kern = np.where(d > 0, np.divide(1.0, d, where=d > 0, out=None), G0)
            phi_ref[i] = np.sum(rho * kern) * grid.dV

        assert np.allclose(phi_fft, phi_ref, atol=1e-10)
