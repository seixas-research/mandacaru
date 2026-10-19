# -*- coding: utf-8 -*-
# file: test/integrals/test_fft.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The shared FFT workspace of :mod:`mandacaru.integrals.fft`.

Its pruned, in-place, batched transforms must give the plain zero-padded
convolution to round-off, for complex and real densities, padded and
circular grids, and must not allocate a padded volume per density.
"""

import tracemalloc

import numpy as np
import pytest
from scipy import fft as sfft

from mandacaru.integrals import fft as fft_mod
from mandacaru.integrals.fft import FFTWorkspace, workspace
from mandacaru.integrals.poisson import PoissonFFTSolver

SHAPE = (7, 9, 6)
PADDED = (14, 18, 12)


def _reference(rho, kernel, shape, padded):
    """The plain convolution: zero-pad, full transforms, keep the box."""
    nx, ny, nz = shape
    out = np.empty(rho.shape, dtype=complex)
    for p, row in enumerate(rho):
        pad = np.zeros(padded, dtype=complex)
        pad[:nx, :ny, :nz] = row.reshape(shape)
        value = np.fft.ifftn(np.fft.fftn(pad) * kernel)
        out[p] = value[:nx, :ny, :nz].reshape(-1)
    return out


def _even_kernel(padded, rng):
    """A random real kernel whose transform is real (an even array)."""
    k = rng.normal(size=padded)
    k = 0.5 * (k + np.roll(k[::-1, ::-1, ::-1], 1, axis=(0, 1, 2)))
    return np.fft.fftn(k).real


class TestPrunedConvolution:
    @pytest.mark.parametrize("padded", [PADDED, SHAPE])
    def test_complex_densities_match_the_plain_convolution(self, padded):
        rng = np.random.default_rng(1)
        G = int(np.prod(SHAPE))
        rho = rng.normal(size=(11, G)) + 1j * rng.normal(size=(11, G))
        kernel = (rng.normal(size=padded)
                  + 1j * rng.normal(size=padded))
        space = FFTWorkspace(SHAPE, padded, batch_bytes=3 * 16
                             * int(np.prod(padded)))
        out = np.empty_like(rho)
        space.convolve(rho, kernel, out, scale=0.5)
        reference = 0.5 * _reference(rho, kernel, SHAPE, padded)
        assert np.abs(out - reference).max() < 1e-12 * np.abs(reference).max()

    @pytest.mark.parametrize("padded", [PADDED, SHAPE])
    def test_real_densities_through_rfft_match(self, padded):
        rng = np.random.default_rng(2)
        G = int(np.prod(SHAPE))
        rho = rng.normal(size=(9, G))
        kernel = _even_kernel(padded, rng)
        half = kernel[..., :padded[2] // 2 + 1]
        space = FFTWorkspace(SHAPE, padded, batch_bytes=2 * 8
                             * int(np.prod(padded)))
        out = np.empty_like(rho)
        space.convolve_real(rho, half, out)
        reference = _reference(rho, kernel, SHAPE, padded)
        assert np.abs(reference.imag).max() < 1e-12
        assert np.abs(out - reference.real).max() < 1e-12 * np.abs(
            reference).max()

    def test_the_power_is_the_self_energy_by_parseval(self):
        """``sum_k |rho(k)|^2 K(k) = H sum_x rho (K * rho)`` with forward
        transforms only."""
        rng = np.random.default_rng(3)
        G = int(np.prod(SHAPE))
        rho = rng.normal(size=(5, G))
        kernel = _even_kernel(PADDED, rng)
        space = FFTWorkspace(SHAPE, PADDED)
        power = space.power(rho, kernel[..., :PADDED[2] // 2 + 1])
        potential = _reference(rho, kernel, SHAPE, PADDED).real
        direct = np.sum(rho * potential, axis=1) * np.prod(PADDED)
        np.testing.assert_allclose(power, direct, rtol=1e-12)

    @pytest.mark.parametrize("padded", [PADDED, SHAPE])
    def test_real_half_spectra_and_their_inverse(self, padded):
        """``forward_real`` is the ``rfftn`` of the zero-padded densities and
        ``inverse_real`` the box of the ``irfftn`` (the rVV10 convolution's
        two halves), batch by batch through the workspace."""
        rng = np.random.default_rng(5)
        G = int(np.prod(SHAPE))
        rho = rng.normal(size=(7, G))
        space = FFTWorkspace(SHAPE, padded, batch_bytes=3 * 8
                             * int(np.prod(padded)))
        half = padded[2] // 2 + 1
        spectra = np.empty((7, padded[0], padded[1], half), dtype=complex)
        space.forward_real(rho, spectra)
        pad = np.zeros((7,) + tuple(padded))
        pad[:, :SHAPE[0], :SHAPE[1], :SHAPE[2]] = rho.reshape((7,) + SHAPE)
        reference = sfft.rfftn(pad, axes=(1, 2, 3))
        assert np.abs(spectra - reference).max() < 1e-12 * np.abs(
            reference).max()
        out = space.inverse_real(spectra, np.empty_like(rho), scale=2.0)
        assert np.abs(out - 2.0 * rho).max() < 1e-12 * np.abs(rho).max()

    def test_a_stack_is_the_workspace_buffer(self):
        space = FFTWorkspace(SHAPE)
        first = space.stack(2)
        assert first.shape == (2,) + SHAPE and first.dtype == complex
        assert np.shares_memory(space.stack(1), first)

    def test_in_place_transforms_of_a_stack(self):
        rng = np.random.default_rng(4)
        stack = rng.normal(size=(3,) + SHAPE) + 0j
        reference = np.fft.fftn(stack, axes=(1, 2, 3))
        space = workspace(SHAPE)
        result = space.forward(stack)
        assert np.shares_memory(result, stack)
        assert np.abs(stack - reference).max() < 1e-12
        space.inverse(stack)
        assert np.abs(stack - np.fft.ifftn(reference, axes=(1, 2, 3))
                      ).max() < 1e-12


class TestNoPaddedVolumePerDensity:
    """The buffers are the workspace's: a solve allocates no padded volume,
    whatever the number of densities."""

    def _traced_peak(self, solver, rho, out):
        solver.solve_stack(rho, out=out)            # buffers allocated here
        tracemalloc.start()
        try:
            solver.solve_stack(rho, out=out)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    SHAPE = (22, 24, 20)

    def test_the_complex_route_allocates_nothing_padded(self):
        """Only NumPy's fixed-size ufunc buffers (8192 values per thread)
        remain; one thread keeps the count deterministic."""
        solver = PoissonFFTSolver(self.SHAPE, spacing=0.4, workers=1)
        volume = 16 * int(np.prod(solver.L))
        rng = np.random.default_rng(5)
        G = int(np.prod(self.SHAPE))
        for count in (3, 20):
            rho = rng.normal(size=(count, G)) + 1j * rng.normal(size=(count, G))
            out = np.empty_like(rho)
            assert self._traced_peak(solver, rho, out) < 0.25 * volume

    def test_the_real_route_does_not_grow_with_the_stack(self, monkeypatch):
        """The rfft/irfft stages return box-sized arrays, one batch at a
        time: past one batch, more densities cost no more memory."""
        solver = PoissonFFTSolver(self.SHAPE, spacing=0.4, workers=1)
        assert solver.even
        monkeypatch.setattr(solver.workspace, "batch_bytes",
                            4 * 8 * int(np.prod(solver.L)))
        rng = np.random.default_rng(6)
        G = int(np.prod(self.SHAPE))
        peaks = []
        for count in (20, 80):
            rho = rng.normal(size=(count, G))
            peaks.append(self._traced_peak(solver, rho, np.empty_like(rho)))
        volume = 16 * int(np.prod(solver.L))
        assert peaks[1] <= peaks[0] * 1.05 + 4096
        assert peaks[1] < solver.workspace.batch(8) * volume


class TestTheCache:
    def test_one_workspace_per_grid_and_padding(self):
        assert workspace(SHAPE, PADDED) is workspace(SHAPE, PADDED)
        assert workspace(SHAPE, PADDED) is not workspace(SHAPE)

    def test_the_cache_is_bounded_in_bytes(self, monkeypatch):
        monkeypatch.setattr(fft_mod, "_WORKSPACES", type(
            fft_mod._WORKSPACES)())
        monkeypatch.setattr(fft_mod, "WORKSPACE_CACHE_BYTES", 1)
        first = workspace((5, 5, 5))
        first.convolve(np.zeros((1, 125), complex), np.ones((5, 5, 5)),
                       np.empty((1, 125), complex))
        assert first.nbytes > 0
        workspace((6, 6, 6))
        assert len(fft_mod._WORKSPACES) == 1

    def test_a_padding_smaller_than_the_grid_is_refused(self):
        with pytest.raises(ValueError, match="cover"):
            FFTWorkspace((8, 8, 8), (8, 7, 8))


def test_the_pool_holds_blas_to_one_thread():
    """A pass on the pool runs its matrix products on one core each, so
    BLAS threads never compete with the pool's."""
    from threadpoolctl import threadpool_info

    seen = []

    def probe(lo, hi):
        seen.extend(pool["num_threads"] for pool in threadpool_info()
                    if pool["user_api"] == "blas")

    fft_mod.parallel_chunks(4, probe, 2)
    assert seen and set(seen) == {1}


def test_the_blocks_are_the_problems_not_the_threads():
    """``parallel_blocks`` cuts ``range(count)`` into the same blocks on any
    thread count, and holds BLAS to one thread on one thread too."""
    from threadpoolctl import threadpool_info

    seen = {}
    for threads in (1, 2, 4):
        blocks, limits = [], []

        def record(lo, hi, state):
            blocks.append((lo, hi))
            limits.append(max(info["num_threads"] for info in
                              threadpool_info() if info["user_api"] == "blas"))

        fft_mod.parallel_blocks(23, 5, record, threads, buffers=list)
        seen[threads] = sorted(blocks)
        assert set(limits) == {1}
    assert seen[1] == seen[2] == seen[4] == [(0, 5), (5, 10), (10, 15),
                                             (15, 20), (20, 23)]
    rng = np.random.default_rng(9)
    A, B = rng.normal(size=(37, 50)), rng.normal(size=(50, 6))
    one = fft_mod.parallel_matmul(A, B, 1, block=8)
    assert np.array_equal(one, fft_mod.parallel_matmul(A, B, 3, block=8))
    assert np.abs(one - A @ B).max() < 1e-12


def test_scipy_transforms_in_place_on_views():
    """What the pruned transforms rely on: ``overwrite_x`` on a strided view
    of a complex buffer writes the result into that view."""
    buffer = np.random.default_rng(7).normal(size=(2, 6, 5, 8)) + 0j
    view = buffer[:, :3, :4, :]
    expected = np.fft.fft(view, axis=-1)
    result = sfft.fft(view, axis=-1, overwrite_x=True)
    assert np.shares_memory(result, buffer)
    assert np.abs(buffer[:, :3, :4, :] - expected).max() < 1e-12
