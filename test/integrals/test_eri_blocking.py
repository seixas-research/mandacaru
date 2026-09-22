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


def _engine():
    c = (0.6, 0.2, -0.3)
    fns = [HydrogenicAtomicOrbital(1, 0, 0), HydrogenicAtomicOrbital(2, 0, 0, center=c),
           HydrogenicAtomicOrbital(2, 1, 1, center=c),
           HydrogenicAtomicOrbital(2, 1, -1, center=c), HydrogenicAtomicOrbital(2, 1, 0)]
    return IntegralEngine(fns, Grid(box_size=6.0, h=0.5, center=(0, 0, 0)))


def _dense_reference(eng):
    psi, M, ng = eng._psi, len(eng.basis), eng.grid.size
    pairs = (np.conj(psi)[:, None, :] * psi[None, :, :]).reshape(M * M, ng)
    phi = PoissonFFTSolver(eng.grid.shape, eng.grid.dx).solve_stack(pairs)
    R = (pairs @ phi.T) * eng.grid.dV
    return R.reshape(M, M, M, M).transpose(0, 2, 1, 3)


class TestBlockedHermitianERI:
    @pytest.mark.parametrize("budget_mb", [None, 1e-3, 0.3, 1e6])
    def test_matches_the_dense_contraction(self, budget_mb):
        eng = _engine()
        ref = _dense_reference(eng)
        eri = eng.two_body("fft", energy_units="Ha", max_memory_mb=budget_mb)
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
            eri = eng.two_body("fft", energy_units="Ha", max_memory_mb=budget)
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
