# -*- coding: utf-8 -*-
# file: integrals/fft.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""One FFT workspace per grid: the buffers and transforms every grid
convolution shares.

Every Poisson solve, spectral Laplacian and pair-density transform on a grid
used to allocate its own padded volumes: zero a padded array, copy the
density in, transform into a new array, multiply into another, transform
back into a third.  :class:`FFTWorkspace` owns those buffers once per
``(grid shape, padded shape)`` and runs the transforms in place.

What it does
------------

* **Buffers allocated once.**  A batch of padded volumes, grown to the
  batch a caller needs and then reused; no padded-volume temporary is
  allocated inside a per-density loop.
* **In-place transforms.**  :mod:`scipy.fft` with ``overwrite_x=True`` on
  views of the workspace buffer, and the kernel product in place
  (``spectrum *= kernel``).
* **Pruned transforms for zero-padded convolutions.**  An isolated
  convolution's input is nonzero only in the box ``[:nx, :ny, :nz]`` and only
  the box of its output is read, so the transform runs axis by axis: forward
  along ``z`` for the ``nx ny`` box lines only, then ``y`` for ``nx Lz``
  lines, then ``x`` for all; the inverse is the mirror image.  With ``L``
  about ``2n`` that is :math:`14 n^3` line points per pass instead of
  :math:`24 n^3`, about 40 % less work.  Each stage only needs the slab it
  reads beyond the box to be zero, so only that slab is cleared (the gap of
  a volume the previous inverse overwrote), never the whole padded volume.
* **Real transforms for real input.**  ``rfft`` along ``z`` and the
  half spectrum after it, with a real even kernel
  (:func:`~mandacaru.integrals.poisson.PoissonFFTSolver` makes the isolated
  kernel even; the periodic ones are).  The ``z`` stage reads a real box
  buffer whose padding is never written, so it is never cleared.
* **Batched transforms.**  Several densities per call
  (``axes=(1, 2, 3)``): on a small grid one transform cannot occupy every
  core, a batch of them can.

Using it from another module
----------------------------

The solvers expose it through their ``solve_stack(rho, out=None)``: pass the
caller's own ``(P, G)`` potential buffer as ``out`` and the potentials are
written there, so a loop over blocks of densities reuses one buffer.  A
caller that only needs transforms takes the grid's workspace with
:func:`workspace` and calls :meth:`FFTWorkspace.forward` /
:meth:`FFTWorkspace.inverse` on a stack it owns (in place), or
:meth:`FFTWorkspace.convolve` with its own kernel.  The nuclear-gradient
modules (``algorithms/forces.py``, ``algorithms/pseudo_forces.py``) solve
their pair potentials through ``PoissonFFTSolver.solve_stack`` and can stream
them into one ``out`` block the same way.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
from scipy import fft as sfft

__all__ = ["FFTWorkspace", "workspace", "parallel_chunks", "thread_count",
           "WORKSPACE_BYTES", "WORKSPACE_CACHE_BYTES", "PARALLEL_MIN_VALUES"]

#: Bytes of padded volumes one workspace keeps for a batch: the batch is as
#: many volumes as fit (at least one).
WORKSPACE_BYTES = 64 * 2 ** 20

#: Bytes of buffers all cached workspaces may hold together; the least
#: recently used workspace is dropped beyond it (a solver holding one keeps
#: it alive).
WORKSPACE_CACHE_BYTES = 256 * 2 ** 20


class FFTWorkspace:
    """Transforms and convolutions on one grid, in reusable buffers.

    Parameters
    ----------
    shape : (int, int, int)
        The grid (the box a density lives in).
    padded : (int, int, int), optional
        The transform shape: ``shape`` itself for a circular (periodic)
        convolution, a larger zero-padded one for an isolated one.
    workers : int
        Threads for the transforms (``-1``: all cores).
    batch_bytes : int
        Bytes of padded volumes one batch may take (:data:`WORKSPACE_BYTES`).
    """

    def __init__(self, shape, padded=None, workers: int = -1,
                 batch_bytes: int = WORKSPACE_BYTES):
        self.shape = tuple(int(n) for n in shape)
        self.padded = (self.shape if padded is None
                       else tuple(int(n) for n in padded))
        if len(self.shape) != 3 or len(self.padded) != 3 or any(
                L < n for n, L in zip(self.shape, self.padded)):
            raise ValueError(f"padded shape {self.padded} must cover the "
                             f"grid {self.shape} on all three axes")
        self.workers = workers
        self.batch_bytes = int(batch_bytes)
        #: True when the transform is the grid itself (no padding to prune).
        self.circular = self.padded == self.shape
        self.half = self.padded[2] // 2 + 1
        self._complex = None          # (b, Lx, Ly, Lz) complex
        self._spectrum = None         # (b, Lx, Ly, Lz//2+1) complex
        self._real_box = None         # (b, nx, ny, Lz) real, zero z-tail

    # -- buffers ------------------------------------------------------------ #

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def padded_size(self) -> int:
        return int(np.prod(self.padded))

    def batch(self, itemsize: int = 16) -> int:
        """Volumes per batch at ``itemsize`` bytes per value."""
        return max(1, int(self.batch_bytes // (itemsize * self.padded_size)))

    @property
    def nbytes(self) -> int:
        """Bytes of the buffers currently held."""
        return sum(b.nbytes for b in (self._complex, self._spectrum,
                                      self._real_box) if b is not None)

    def _complex_buffer(self, count: int) -> np.ndarray:
        if self._complex is None or self._complex.shape[0] < count:
            self._complex = np.empty((count,) + self.padded,
                                     dtype=np.complex128)
        return self._complex

    def _real_buffers(self, count: int):
        nx, ny, _nz = self.shape
        if self._real_box is None or self._real_box.shape[0] < count:
            # Only [..., :nz] is ever written: the z-tail stays zero.
            self._real_box = np.zeros((count, nx, ny, self.padded[2]))
        if self._spectrum is None or self._spectrum.shape[0] < count:
            self._spectrum = np.empty(
                (count, self.padded[0], self.padded[1], self.half),
                dtype=np.complex128)
        return self._real_box, self._spectrum

    def release(self) -> None:
        """Drop the buffers (they are rebuilt on the next use)."""
        self._complex = self._spectrum = self._real_box = None

    # -- in-place transforms of a stack ------------------------------------ #

    def forward(self, stack: np.ndarray) -> np.ndarray:
        """Forward 3-D transform of a ``(k, *shape)`` complex stack, in
        place (circular grid)."""
        return sfft.fftn(stack, axes=(1, 2, 3), workers=self.workers,
                         overwrite_x=True)

    def inverse(self, stack: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`forward`, in place."""
        return sfft.ifftn(stack, axes=(1, 2, 3), workers=self.workers,
                          overwrite_x=True)

    def _fft(self, view, axis, inverse=False):
        transform = sfft.ifft if inverse else sfft.fft
        transform(view, axis=axis, workers=self.workers, overwrite_x=True)

    def _planes(self, count: int, function) -> None:
        """``function(lo, hi)`` over chunks of ``range(count)`` (x planes),
        on the workspace's threads: NumPy's element-wise passes are
        single-threaded, the transforms are not, so the passes between them
        are split here."""
        parallel_chunks(count, function, self.threads)

    def _forward_pruned(self, buf):
        """Forward transform of ``buf`` (``(k, *padded)``) whose input is the
        box ``[:, :nx, :ny, :nz]``; clears only the slabs each stage reads."""
        nx, ny, nz = self.shape
        Lx = self.padded[0]

        def tails(lo, hi):
            buf[:, lo:hi, :ny, nz:] = 0.0
            buf[:, lo:hi, ny:, :] = 0.0

        self._planes(nx, tails)
        self._fft(buf[:, :nx, :ny, :], 3)
        self._fft(buf[:, :nx, :, :], 2)

        def gap(lo, hi):
            buf[:, nx + lo:nx + hi] = 0.0

        self._planes(Lx - nx, gap)
        self._fft(buf, 1)

    def _inverse_pruned(self, buf):
        """Inverse transform of ``buf`` valid in the box only."""
        nx, ny, _nz = self.shape
        self._fft(buf, 1, inverse=True)
        self._fft(buf[:, :nx, :, :], 2, inverse=True)
        self._fft(buf[:, :nx, :ny, :], 3, inverse=True)

    def multiply(self, spectrum, kernel) -> None:
        """``spectrum *= kernel`` in place for a ``(k, *shape)`` stack of
        transforms, x planes split over threads."""
        self._planes(spectrum.shape[1], lambda lo, hi: _multiply(
            spectrum[:, lo:hi], kernel[lo:hi]))

    # -- convolutions ------------------------------------------------------- #

    def convolve(self, rho: np.ndarray, kernel: np.ndarray,
                 out: np.ndarray, scale: float = 1.0) -> np.ndarray:
        r"""``out[p] = scale * (K * rho[p])`` restricted to the grid, for a
        ``(P, G)`` stack of complex densities.

        ``kernel`` is the transform on the padded mesh (real or complex,
        shape ``padded``).  The convolution is circular on the padded mesh,
        so it is the isolated one whenever the padding covers the kernel's
        reach.
        """
        nx, ny, nz = self.shape
        P = rho.shape[0]
        step = min(self.batch(16), max(P, 1))
        buf = self._complex_buffer(step)
        complex_out = np.iscomplexobj(out)
        for start in range(0, P, step):
            stop = min(start + step, P)
            k = stop - start
            work = buf[:k]
            source = rho[start:stop].reshape(k, nx, ny, nz)
            target = out[start:stop].reshape(k, nx, ny, nz)

            def load(lo, hi):
                work[:, lo:hi, :ny, :nz] = source[:, lo:hi]

            self._planes(nx, load)
            if self.circular:
                self.forward(work)
                self.multiply(work, kernel)
                self.inverse(work)
            else:
                self._forward_pruned(work)
                self.multiply(work, kernel)
                self._inverse_pruned(work)

            def store(lo, hi):
                box = work[:, lo:hi, :ny, :nz]
                np.multiply(box if complex_out else box.real, scale,
                            out=target[:, lo:hi])

            self._planes(nx, store)
        return out

    def _real_forward(self, rho, start, stop, real_box, spectrum):
        """The half spectrum of real densities ``rho[start:stop]`` in
        ``spectrum[:k]`` (pruned), returned."""
        nx, ny, nz = self.shape
        Lx = self.padded[0]
        k = stop - start
        box = real_box[:k]
        source = rho[start:stop].reshape(k, nx, ny, nz)

        def load(lo, hi):
            box[:, lo:hi, :, :nz] = source[:, lo:hi]

        self._planes(nx, load)
        spec = spectrum[:k]
        lines = sfft.rfft(box, axis=3, workers=self.workers)

        def place(lo, hi):
            spec[:, lo:hi, :ny, :] = lines[:, lo:hi]
            if not self.circular:
                spec[:, lo:hi, ny:, :] = 0.0

        self._planes(nx, place)
        if self.circular:
            self._fft(spec, 2)
        else:
            self._fft(spec[:, :nx, :, :], 2)

            def gap(lo, hi):
                spec[:, nx + lo:nx + hi] = 0.0

            self._planes(Lx - nx, gap)
        self._fft(spec, 1)
        return spec

    def convolve_real(self, rho: np.ndarray, half_kernel: np.ndarray,
                      out: np.ndarray, scale: float = 1.0) -> np.ndarray:
        r""":meth:`convolve` for real densities and a real **even** kernel,
        through ``rfft``: ``half_kernel`` is the kernel on the half spectrum
        ``(Lx, Ly, Lz//2 + 1)``.  ``out`` may be real or complex."""
        nx, ny, nz = self.shape
        Lz = self.padded[2]
        P = rho.shape[0]
        step = min(self.batch(8), max(P, 1))
        real_box, spectrum = self._real_buffers(step)
        for start in range(0, P, step):
            stop = min(start + step, P)
            k = stop - start
            spec = self._real_forward(rho, start, stop, real_box, spectrum)
            self.multiply(spec, half_kernel)
            self._fft(spec, 1, inverse=True)
            self._fft(spec if self.circular else spec[:, :nx, :, :], 2,
                      inverse=True)
            values = sfft.irfft(spec[:, :nx, :ny, :], n=Lz, axis=3,
                                workers=self.workers)
            target = out[start:stop].reshape(k, nx, ny, nz)

            def store(lo, hi):
                np.multiply(values[:, lo:hi, :, :nz], scale,
                            out=target[:, lo:hi])

            self._planes(nx, store)
        return out

    def power(self, rho: np.ndarray, half_kernel: np.ndarray) -> np.ndarray:
        r""":math:`\sum_k |\hat\rho_p(k)|^2 K(k)` of real densities, the
        forward transforms only (pruned), over the full spectrum.

        By Parseval this is :math:`H \sum_x \rho_p(x)(K*\rho_p)(x)`, the
        self-energy of each density without its inverse transform.
        """
        Lx, Ly, Lz = self.padded
        P = rho.shape[0]
        out = np.empty(P)
        # The half spectrum holds kz in [0, Lz/2]: every other kz is the
        # conjugate of one of them, so it counts twice.  Each value enters
        # as its real and imaginary parts squared, hence the repeat.
        weight = np.full(self.half, 2.0)
        weight[0] = 1.0
        if Lz % 2 == 0:
            weight[-1] = 1.0
        weighted = np.repeat((half_kernel * weight).reshape(Lx, -1), 2,
                             axis=1)
        step = min(self.batch(8), max(P, 1))
        real_box, spectrum = self._real_buffers(step)
        for start in range(0, P, step):
            stop = min(start + step, P)
            k = stop - start
            spec = self._real_forward(rho, start, stop, real_box, spectrum)
            values = spec.view(np.float64).reshape(k, Lx, -1)
            partial = np.zeros((Lx, k))

            def contract(lo, hi):
                square = np.square(values[:, lo:hi])
                partial[lo:hi] = np.einsum("pxg,xg->xp", square,
                                           weighted[lo:hi])

            self._planes(Lx, contract)
            out[start:stop] = partial.sum(axis=0)
        return out

    @property
    def threads(self) -> int:
        """Threads of the element-wise passes: those of the transforms, or
        one on a grid too small for a thread to pay for itself."""
        if self.padded_size < PARALLEL_MIN_VALUES:
            return 1
        return thread_count(self.workers)


def _multiply(spectrum: np.ndarray, kernel: np.ndarray) -> None:
    """``spectrum *= kernel`` in place; a real kernel multiplies the real and
    imaginary parts as floats, so no complex copy of it is ever cast."""
    if np.iscomplexobj(kernel):
        spectrum *= kernel
    else:
        pairs = spectrum.view(np.float64).reshape(spectrum.shape + (2,))
        pairs *= kernel[..., None]


#: Below this many values a pass runs inline: a thread's start-up costs more.
PARALLEL_MIN_VALUES = 1 << 16

_POOL = None


def _forget_pool() -> None:
    """A forked child inherits the pool object but not its threads: it
    starts a pool of its own on first use."""
    global _POOL
    _POOL = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_pool)


def thread_count(workers) -> int:
    """Threads behind ``workers`` (``-1`` or ``None``: every core this
    process may run on)."""
    if workers is None or int(workers) < 0:
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except AttributeError:
            return max(1, os.cpu_count() or 1)
    return max(1, int(workers))


_BLAS = None


def _blas_controller():
    """The process's BLAS thread pools (:mod:`threadpoolctl`), inspected
    once, on the first parallel pass (NumPy's BLAS is loaded by then)."""
    global _BLAS
    if _BLAS is None:
        from threadpoolctl import ThreadpoolController

        _BLAS = ThreadpoolController()
    return _BLAS


def parallel_chunks(count: int, function, threads: int) -> None:
    """``function(lo, hi)`` over contiguous chunks of ``range(count)``,
    on a shared thread pool (NumPy releases the GIL in its loops).

    While the pool runs, BLAS is held to one thread: a matrix product inside
    ``function`` is then one core's work, and BLAS threads left spinning
    after an earlier product never compete with the pool for the cores.
    The pool's ``threads`` are the whole parallelism of the pass.
    """
    global _POOL
    threads = min(int(threads), int(count))
    if threads <= 1:
        if count > 0:
            function(0, count)
        return
    if _POOL is None or _POOL._max_workers < threads:
        from concurrent.futures import ThreadPoolExecutor

        if _POOL is not None:
            _POOL.shutdown(wait=False)
        _POOL = ThreadPoolExecutor(max_workers=threads,
                                   thread_name_prefix="mandacaru-fft")
    edges = np.linspace(0, count, threads + 1).astype(int)
    with _blas_controller().limit(limits=1, user_api="blas"):
        futures = [_POOL.submit(function, int(lo), int(hi))
                   for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo]
        for future in futures:
            future.result()


_WORKSPACES: "OrderedDict[tuple, FFTWorkspace]" = OrderedDict()


def workspace(shape, padded=None, workers: int = -1) -> FFTWorkspace:
    """The shared :class:`FFTWorkspace` of a grid (cached per shape, padded
    shape and worker count; the cache is bounded by
    :data:`WORKSPACE_CACHE_BYTES`)."""
    shape = tuple(int(n) for n in shape)
    padded = shape if padded is None else tuple(int(n) for n in padded)
    key = (shape, padded, workers)
    found = _WORKSPACES.pop(key, None)
    if found is None:
        found = FFTWorkspace(shape, padded, workers)
    _WORKSPACES[key] = found
    held = sum(w.nbytes for w in _WORKSPACES.values())
    for other in list(_WORKSPACES):
        if held <= WORKSPACE_CACHE_BYTES or other == key:
            continue
        held -= _WORKSPACES.pop(other).nbytes
    return found
