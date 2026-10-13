# -*- coding: utf-8 -*-
# file: test/test_eri_blocking.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The FFT two-body step runs in memory-bounded blocks over the Hermitian
pair densities and reproduces the dense contraction exactly.

The dense formula (all ``M^2`` pair densities and potentials resident, one
GEMM) is re-implemented here as the reference; the engine must match it for
every block size -- including one pair per block -- on a basis with complex
``p`` functions, where the conjugation bookkeeping of the flipped pairs
actually matters.
"""

import numpy as np
import pytest

from mandacaru.basis import HydrogenicAtomicOrbital
from mandacaru.integrals import Grid, IntegralEngine, PoissonFFTSolver
from mandacaru.integrals import engine as engine_mod


def _engine(real=False):
    """Complex p functions by default (a crystal's harmonics), where the
    conjugation bookkeeping of the flipped pairs matters; ``real=True`` is a
    molecule's basis, whose tensor is real."""
    c = (0.6, 0.2, -0.3)
    fns = [HydrogenicAtomicOrbital(1, 0, 0, real=real),
           HydrogenicAtomicOrbital(2, 0, 0, center=c, real=real),
           HydrogenicAtomicOrbital(2, 1, 1, center=c, real=real),
           HydrogenicAtomicOrbital(2, 1, -1, center=c, real=real),
           HydrogenicAtomicOrbital(2, 1, 0, real=real)]
    return IntegralEngine(fns, Grid(box_size=6.0, grid_spacing=0.5, center=(0, 0, 0)))


def _dense_reference(eng):
    psi, M, ng = eng._psi, len(eng.basis), eng.grid.size
    pairs = (np.conj(psi)[:, None, :] * psi[None, :, :]).reshape(M * M, ng)
    phi = PoissonFFTSolver(eng.grid.shape, eng.grid.dx).solve_stack(pairs)
    R = (pairs @ phi.T) * eng.grid.dV
    return R.reshape(M, M, M, M).transpose(0, 2, 1, 3)


class TestBlockedHermitianERI:
    @pytest.mark.parametrize("budget_mb, route", [(None, "cached"),
                                                  (1e-3, "streamed"),
                                                  (0.3, "streamed"),
                                                  (1e6, "cached")])
    def test_matches_the_dense_contraction(self, budget_mb, route):
        """Both contractions -- the streamed products and the kept pair
        densities -- give the dense contraction in every entry."""
        eng = _engine()
        ref = _dense_reference(eng)
        eri = eng.two_body(energy_units="Ha", max_memory_mb=budget_mb)
        assert eng.two_body_profile["route"] == route
        assert np.abs(eri - ref).max() < 1e-12

    @pytest.mark.parametrize("budget_mb", [None, 1e-3])
    def test_a_real_basis_gives_the_real_tensor(self, budget_mb):
        """A molecule's real harmonics: real pair densities through the real
        transforms, a float64 tensor equal to the dense contraction and to
        the complex basis's tensor rotated shell by shell."""
        from mandacaru.basis._angular import real_harmonic_transform

        real = _engine(real=True)
        eri = real.two_body(energy_units="Ha", max_memory_mb=budget_mb)
        assert real._psi.dtype == np.float64 and eri.dtype == np.float64
        assert real.two_body_profile["real"]
        assert np.abs(eri - _dense_reference(real)).max() < 1e-12
        complex_ = _engine(real=False).two_body(energy_units="Ha")
        U = np.eye(5, dtype=complex)
        U[2:4, 2:4] = real_harmonic_transform(1)[[2, 0]][:, [2, 0]]
        # The p shell on center c is (m = +1, -1); its m = 0 partner sits on
        # the origin, a different shell, so the 2x2 block of (x, y) is it.
        # S_m = sum U_mk Y_k, so the real functions are the complex ones
        # times B = U^T.
        B = U.T
        rotated = np.einsum("ap,bq,cr,ds,abcd->pqrs", B.conj(), B.conj(), B,
                            B, complex_, optimize=True)
        assert np.abs(rotated - eri).max() < 1e-12

    def test_a_periodic_cell_matches_the_dense_contraction(self):
        """The circular kernel of a crystal: the same slab-by-slab tensor."""
        from mandacaru.integrals.poisson import PeriodicPoissonSolver

        fns = [HydrogenicAtomicOrbital(1, 0, 0),
               HydrogenicAtomicOrbital(2, 1, 1, center=(0.5, 0.0, 0.2)),
               HydrogenicAtomicOrbital(2, 1, -1, center=(0.5, 0.0, 0.2))]
        grid = Grid(box_size=5.0, grid_spacing=0.5, center=(0, 0, 0))
        eng = IntegralEngine(fns, grid)
        solver = PeriodicPoissonSolver(grid.shape, step=grid.step)
        psi, M = eng._psi, len(fns)
        pairs = (np.conj(psi)[:, None, :] * psi[None, :, :]).reshape(M * M, -1)
        R = (pairs @ solver.solve_stack(pairs).T) * grid.dV
        ref = R.reshape(M, M, M, M).transpose(0, 2, 1, 3)
        for budget in (None, 1e-3):
            eri = eng.two_body(energy_units="Ha", max_memory_mb=budget,
                               solver=solver)
            assert np.abs(eri - ref).max() < 1e-12

    def test_block_size_follows_the_budget(self):
        eng = _engine()
        U, ng = 15, eng.grid.size
        solver = PoissonFFTSolver(eng.grid.shape, eng.grid.dx)
        assert eng._eri_block_size(U, ng, solver, 1e-3) == 1     # too small
        assert eng._eri_block_size(U, ng, solver, 1e6) == U      # all at once
        mid = eng._eri_block_size(U, ng, solver, 5.0)
        assert 1 <= mid <= U

    def test_tensor_symmetries_hold_for_every_block_size(self):
        eng = _engine()
        for budget in (1e-3, None):
            eri = eng.two_body(energy_units="Ha", max_memory_mb=budget)
            assert np.abs(eri - eri.transpose(1, 0, 3, 2)).max() < 1e-12
            assert np.abs(eri - np.conj(eri.transpose(2, 3, 0, 1))).max() < 1e-12

    def test_environment_budget(self, monkeypatch):
        monkeypatch.delenv("MANDACARU_ERI_MEMORY_MB", raising=False)
        assert engine_mod.eri_memory_budget_mb() == engine_mod.ERI_MEMORY_MB
        monkeypatch.setenv("MANDACARU_ERI_MEMORY_MB", "12.5")
        assert engine_mod.eri_memory_budget_mb() == 12.5
        monkeypatch.setenv("MANDACARU_ERI_MEMORY_MB", "0")
        with pytest.raises(ValueError):
            engine_mod.eri_memory_budget_mb()


class TestTheThreadedContractions:
    """The contractions split over threads, each with one-core BLAS: the
    same slabs as the single-threaded and the kept-density products."""

    @pytest.mark.parametrize("complex_", [False, True])
    def test_the_streamed_pair_products_match_the_kept_ones(self, monkeypatch,
                                                           complex_):
        monkeypatch.setattr(engine_mod, "PARALLEL_MIN_VALUES", 0)
        rng = np.random.default_rng(3)
        M, G, nv = 7, 230, 5
        psi = rng.normal(size=(M, G))
        phi = rng.normal(size=(nv, G))
        if complex_:
            psi = psi + 1j * rng.normal(size=(M, G))
            phi = phi + 1j * rng.normal(size=(nv, G))
        iu, ju = np.triu_indices(M)
        ref = IntegralEngine._contract_cached(np.conj(psi[iu]) * psi[ju],
                                              phi.copy(), M, iu, ju)
        dense = np.einsum("ag,vg,cg->vac", np.conj(psi), phi, psi)
        assert np.abs(ref - dense).max() < 1e-12
        for width, workers in ((G, 1), (17, 3), (1, 4)):
            X = IntegralEngine._contract_streamed(psi, phi, iu, ju, width,
                                                  workers)
            assert np.abs(X - ref).max() < 1e-12

    def test_the_potential_products_match_on_threads(self, monkeypatch):
        monkeypatch.setattr(engine_mod, "PARALLEL_MIN_VALUES", 0)
        monkeypatch.setattr(engine_mod, "CONTRACTION_TILE_BYTES", 8 * 40)
        monkeypatch.setattr(engine_mod, "CONTRACTION_TILE_ROWS", 4)
        rng = np.random.default_rng(4)
        psi = rng.normal(size=(6, 150)) + 1j * rng.normal(size=(6, 150))
        phi = rng.normal(size=(4, 150))
        dense = np.einsum("ag,vg,cg->vac", np.conj(psi), phi, psi)
        for workers in (1, 3):
            X = engine_mod.contract_pair_potentials(np.conj(psi), psi, phi,
                                                    workers)
            assert np.abs(X - dense).max() < 1e-12

    def test_the_tile_width_follows_the_budget(self):
        width = IntegralEngine._pair_tile_width
        U, G = 5886, 27930
        w = width(U, G, 256 * 2 ** 20, 8)
        assert 8 * U * w <= engine_mod.PAIR_TILE_SHARE * 256 * 2 ** 20
        assert width(U, G, 512 * 2 ** 20, 8) > w
        assert width(U, G, 1.0, 8) == 1 and width(3, G, 1e12, 8) == G


class TestTheTensorItselfIsBudgeted:
    """``max_memory_mb`` bounds the Poisson blocks; the ``M^4`` result and
    its Gram blocks are checked against memory before any solve runs."""

    def test_the_estimate_counts_the_result_without_gram_blocks(self):
        """The result is written in place from each block's contraction: no
        ``U x U`` Gram block and no assembly temporaries remain."""
        from mandacaru.integrals.engine import eri_peak_bytes

        M = 100
        U = M * (M + 1) // 2
        assert eri_peak_bytes(M) >= 16 * M ** 4
        assert eri_peak_bytes(M) < 16 * (M ** 4 + 2 * U * U)
        assert eri_peak_bytes(M) / 2 ** 30 > 1.49          # the report's figure

    def test_an_unaffordable_tensor_is_refused_up_front(self, monkeypatch):
        from mandacaru.core import matrix_free
        from mandacaru.integrals.engine import refuse_unaffordable_eri

        monkeypatch.setattr(matrix_free, "available_memory_bytes",
                            lambda: 10 ** 6)
        with pytest.raises(MemoryError, match="two-electron tensor"):
            refuse_unaffordable_eri(40)
