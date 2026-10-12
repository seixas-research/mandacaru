# -*- coding: utf-8 -*-
# file: integrals/engine.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Basis-agnostic real-space integral engine (Python front-end).

``IntegralEngine`` is a thin orchestration layer: it builds the grid, samples an
arbitrary list of :class:`~mandacaru.basis.base.BasisFunction` objects, evaluates
the external potential, and hands the contiguous arrays to the C backend
(:mod:`mandacaru.integrals._backend`).  It contains **no** knowledge of what the
basis functions are, which is exactly what lets Wannier or numerical orbitals be
dropped in unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ..basis.base import BasisFunction
from ..units import from_hartree
from . import _backend
from .fft import PARALLEL_MIN_VALUES, parallel_chunks, thread_count
from .grid import Grid
from .poisson import PeriodicPoissonSolver, PoissonFFTSolver

#: Default memory budget (MB) of the FFT two-body working set; override per
#: call with ``two_body(max_memory_mb=...)`` or globally with the environment
#: variable ``MANDACARU_ERI_MEMORY_MB``.
ERI_MEMORY_MB = 256.0


def eri_memory_budget_mb() -> float:
    """The two-body working-set budget in MB (``MANDACARU_ERI_MEMORY_MB`` or default)."""
    import os
    value = os.environ.get("MANDACARU_ERI_MEMORY_MB")
    if value is None:
        return ERI_MEMORY_MB
    budget = float(value)
    if not budget > 0:
        raise ValueError("MANDACARU_ERI_MEMORY_MB must be positive")
    return budget

# A potential is any callable V(x, y, z) -> real array on the grid.
PotentialFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def _radial_table(radial, r_max: float = 40.0, points: int = 20001):
    """``(r, R)`` of a radial function on the fine reference grid."""
    r = np.linspace(0.0, r_max, points)
    return r, np.nan_to_num(np.asarray(radial(r), dtype=float))


def radial_norm(radial, r_max: float = 40.0, points: int = 20001) -> float:
    r""":math:`\int R^2 r^2\,dr` of a radial function on that grid."""
    r, R = _radial_table(radial, r_max, points)
    return float(np.trapezoid(R * R * r * r, r))


def radial_kinetic_energy(radial, l: int, r_max: float = 40.0,
                          points: int = 20001) -> float:
    r"""Exact kinetic energy per unit norm of ``R(r) Y_lm``,
    :math:`\tfrac12\int[(rR)'^2 + l(l+1)R^2]\,dr / \int R^2 r^2 dr`,
    from the radial function on a fine one-dimensional grid."""
    r, R = _radial_table(radial, r_max, points)
    u = r * R
    du = np.gradient(u, r)
    norm = float(np.trapezoid(R * R * r * r, r))
    if norm <= 0:
        return 0.0
    centrifugal = l * (l + 1) * np.trapezoid(R[1:] ** 2, r[1:])
    return float(0.5 * (np.trapezoid(du * du, r) + centrifugal) / norm)


def eri_peak_bytes(n_orbitals: int, real: bool = False) -> int:
    """Peak memory of the FFT two-body tensor beyond the Poisson blocks.

    The ``M^4`` result -- real (8 bytes an entry) for a molecule's real
    functions, complex (16) otherwise -- written in place block by block,
    and the ``(nv, M, M)`` contraction results of one block of pair
    potentials (a few ``M^3`` arrays in practice).  The pair densities,
    potentials and contraction tiles are bounded separately by
    ``max_memory_mb``.
    """
    M = int(n_orbitals)
    return (8 if real else 16) * (M ** 4 + 2 * M ** 3)


def refuse_unaffordable_eri(n_orbitals: int, real: bool = False) -> None:
    """Raise :class:`MemoryError` when the two-body tensor cannot fit, before
    the Poisson solves it would otherwise run first."""
    from ..core.matrix_free import available_memory_bytes, format_bytes

    needed = eri_peak_bytes(n_orbitals, real)
    available = available_memory_bytes()
    if needed > available:
        kind = "real" if real else "complex"
        raise MemoryError(
            f"the two-electron tensor of {n_orbitals} orbitals needs "
            f"{format_bytes(needed)} (the {kind} M^4 result) against "
            f"{format_bytes(available)} of available memory; reduce "
            f"the basis (size=, an active_space) before integrating")


#: Most pair potentials solved per block: past this a larger block only
#: holds more memory (the transforms batch by the workspace, the products
#: by tiles).
MAX_PAIR_BLOCK = 1024

#: Target bytes of one contraction tile ``conj(psi) * Phi`` (a few MB keeps
#: it in cache while the matrix product reads it).
CONTRACTION_TILE_BYTES = 8 * 2 ** 20


def contract_pair_potentials(conj_psi, psi, phi, workers: int = -1) -> np.ndarray:
    r"""``X[v, a, c] = sum_g conj(psi_a) Phi_v psi_c`` for a block of pair
    potentials ``phi`` ``(nv, G)`` over the functions ``psi`` ``(M, G)``
    (``conj_psi`` is ``conj(psi)``, passed in so a caller forms it once).

    Tile by tile over the grid: each tile ``conj(psi) * Phi_v`` for a few
    ``v`` (:data:`CONTRACTION_TILE_BYTES`) is one matrix product with
    ``psi^T``, so no ``M x G`` product per potential is ever resident.  Not
    scaled by the volume element.  Real functions and potentials give a real
    ``X``.  ``workers`` threads form the tiles (``-1``: every core), as the
    FFT workers of the solver that made ``phi``.
    """
    M, ngrid = psi.shape
    nv = phi.shape[0]
    dtype = np.result_type(conj_psi, psi, phi)
    X = np.zeros((nv, M, M), dtype=dtype)
    rows = max(1, min(nv, 4096 // max(M, 1)))
    width = int(max(256, min(ngrid, CONTRACTION_TILE_BYTES
                             // (dtype.itemsize * rows * M))))
    tile = np.empty(rows * M * width, dtype=dtype)
    part = np.empty((rows * M, M), dtype=dtype)
    threads = (1 if rows * M * width < PARALLEL_MIN_VALUES
               else thread_count(workers))
    for v0 in range(0, nv, rows):
        v1 = min(v0 + rows, nv)
        k = v1 - v0
        for g0 in range(0, ngrid, width):
            g1 = min(g0 + width, ngrid)
            w = g1 - g0
            t = tile[:k * M * w].reshape(k, M, w)

            def form(lo, hi):
                # The element-wise product is single-threaded in NumPy;
                # its rows are split over threads (the product is not).
                np.multiply(conj_psi[None, :, g0:g1],
                            phi[v0 + lo:v0 + hi, None, g0:g1], out=t[lo:hi])

            parallel_chunks(k, form, threads)
            out = part[:k * M]
            np.matmul(t.reshape(k * M, w), psi[:, g0:g1].T, out=out)
            X[v0:v1] += out.reshape(k, M, M)
    return X


class IntegralEngine:
    """Compute one- and two-body real-space integrals over a localized basis.

    Parameters
    ----------
    basis : sequence of BasisFunction
        The orbitals (HAO, Wannier, ...) spanning the active space.
    grid : Grid
        The shared integration grid.  All functions are sampled on it.
    """

    def __init__(self, basis: Sequence[BasisFunction], grid: Grid,
                 periodic: bool = False):
        """``periodic`` selects the Coulomb boundary condition of the two-body
        tensor: the zero-padded isolated kernel (default, correct for a
        molecule) or the reciprocal-space periodic one, where the grid *is* the
        cell and the density really is repeated on every lattice translation.
        """
        self.periodic = bool(periodic)
        # Resolve the integral backend *before* any integration: prefer the C
        # library, compile it on the spot when it is missing, and only fall
        # back to the NumPy reference kernels when that compile fails (the
        # fallback is announced once per process).
        self.backend_status = _backend.ensure_backend()
        _backend.warn_fallback(self.backend_status)
        self.basis = list(basis)
        self.grid = grid
        # Sample every function once; reuse the (M, ngrid) stack for all
        # integrals.  This is the data actually shipped to C.  Each function
        # is written straight into the preallocated stack, so the samples
        # never exist twice (a list of rows and their stacked copy).
        sample = (self._sample_periodic if self.periodic
                  else (lambda function: function.sample(grid)))
        #: Whether every function is real (a molecule's real harmonics): the
        #: samples, and every integral built from them, are then ``float64``.
        self.real = (not self.periodic and bool(self.basis)
                     and all(getattr(f, "real", False) for f in self.basis))
        self._psi = np.empty((len(self.basis), grid.size),
                             dtype=np.float64 if self.real else np.complex128)
        for row, function in zip(self._psi, self.basis):
            row[:] = sample(function)

        # Profiling: wall-time per integral stage, plus the backend / core count
        # so the driver summary can report how the integration ran.
        from ..utils.profiling import Timings
        self.timings = Timings(
            n_cores=self.backend_status.n_threads,
            backend=self.backend_status.label)

    #: Lattice-image shells tried when sampling a periodic basis function.
    IMAGE_SHELLS = 6
    #: A shell is negligible once it adds less than this, relative to the peak.
    IMAGE_TOLERANCE = 1e-10

    def _sample_periodic(self, function) -> np.ndarray:
        r"""Sample ``phi^per(r) = sum_R phi(r - R)`` on the cell grid.

        A localized orbital truncated at the cell boundary is not the orbital a
        periodic Hamiltonian acts on: its tail belongs to the neighboring
        cell.  The Coulomb kernel and the external potential are both summed
        over the lattice here, so the basis has to be as well, or the two
        halves of the Hamiltonian describe different systems.

        Shells of lattice translations are added until one contributes less
        than :attr:`IMAGE_TOLERANCE` of the peak amplitude.  How many that
        takes is a property of the cell against the orbital's decay length: a
        wide cell needs one, a cell of the order of the orbital needs several.
        """
        import warnings

        grid = self.grid
        if grid.cell is None:
            raise ValueError(
                "a periodic basis needs the cell it is periodic in; build the "
                "Grid with cell=... so the lattice translations are defined.")
        # grid.cell is in the Grid's *user* units while X/Y/Z are in Bohr.
        # Translating by the raw cell puts the images 1.89x too close and drops
        # several copies of every orbital inside the cell.
        from ..units import to_bohr
        cell = np.asarray(
            to_bohr(np.asarray(grid.cell, dtype=float).reshape(3, 3),
                    grid.units), dtype=float)
        total = np.array(
            np.broadcast_to(function.evaluate(grid.X, grid.Y, grid.Z),
                            grid.shape), dtype=np.complex128)
        for shell in range(1, self.IMAGE_SHELLS + 1):
            added = np.zeros_like(total)
            span = range(-shell, shell + 1)
            for i in span:
                for j in span:
                    for k in span:
                        if max(abs(i), abs(j), abs(k)) != shell:
                            continue          # only the new outer layer
                        offset = np.array([i, j, k], dtype=float) @ cell
                        added += np.broadcast_to(
                            function.evaluate(grid.X - offset[0],
                                              grid.Y - offset[1],
                                              grid.Z - offset[2]), grid.shape)
            total += added
            peak = float(np.abs(total).max())
            if float(np.abs(added).max()) <= self.IMAGE_TOLERANCE * max(peak,
                                                                        1e-300):
                break
        else:
            warnings.warn(
                f"the periodic image sum for {type(function).__name__} had not "
                f"converged after {self.IMAGE_SHELLS} shells: the cell is small "
                "against the orbital's range, so the basis tail is truncated.  "
                "Use a larger cell or a more compact basis.",
                RuntimeWarning, stacklevel=2)
        return total.reshape(-1)

    @property
    def uses_c_backend(self) -> bool:
        return self.backend_status.available

    @property
    def n_cores(self) -> int | None:
        """OpenMP threads the C backend uses (``None`` for the NumPy fallback)."""
        return _backend.num_threads()

    def integration_profile(self) -> dict:
        """Timing / core / memory summary of the integrals run so far."""
        return self.timings.as_dict()

    # -- one body ---------------------------------------------------------- #

    def one_body(self, potential: PotentialFn, energy_units: str = "eV",
                 kinetic: str = "fd"):
        """Kinetic ``T`` and potential ``V`` matrices over the basis.

        Parameters
        ----------
        potential : callable
            ``V(x, y, z)`` returning the external potential (real, Hartree) on
            the grid's Bohr coordinates.
        energy_units : {"eV", "Ha"}
            Unit of the returned matrices (default ``"eV"``); the integrals are
            computed in Hartree and converted on return.
        kinetic : {"fd", "spectral"}
            How the Laplacian is discretized.  ``"fd"`` (default) is the
            3-point finite-difference stencil of the C backend.  ``"spectral"``
            applies the exact Laplacian of the trigonometric interpolant of
            the samples (``-k^2`` in Fourier space): the finite-difference
            symbol ``(2 - 2 cos kh)/h^2`` never exceeds ``k^2``, so the stencil
            *under*-estimates the kinetic energy of any function that is not
            smooth on the grid scale -- which is what lets compact functions
            collapse into a deep potential ("ghost states").  The spectral
            operator has no such bias.  Orthogonal grids only.

        Returns
        -------
        (T, V) : tuple of (M, M) complex ndarrays
            ``T[a,b] = <a| -1/2 nabla^2 |b>``, ``V[a,b] = <a| V |b>``.
        """
        Vext = np.ascontiguousarray(
            np.real(potential(self.grid.X, self.grid.Y, self.grid.Z)).reshape(-1),
            dtype=np.float64)
        with self.timings.time("one-body integrals"):
            if kinetic == "fd":
                T, V = _backend.one_body_matrices(self._psi, Vext, self.grid)
            elif kinetic == "spectral":
                T = self._spectral_kinetic()
                V = (np.conj(self._psi) * Vext) @ self._psi.T * self.grid.dV
            else:
                raise ValueError(
                    f"unknown kinetic operator {kinetic!r}; use 'fd' or "
                    "'spectral'")
        return from_hartree(T, energy_units), from_hartree(V, energy_units)

    def _g_squared(self) -> np.ndarray:
        r""":math:`|G|^2` on the grid's FFT mesh.

        For a *general* voxel basis, from the reciprocal lattice of the grid's
        own cell -- the same construction the periodic Poisson kernel uses,
        Nyquist plane included (symmetrized, so :math:`|G|^2` is even in
        :math:`G` and the operator below is real symmetric).  For a diagonal
        step it reduces term by term to the separable ``2 pi fftfreq(n,
        d=dx)`` form (``G_x = 2 pi m1 / (nx dx)``), so an orthogonal grid is
        unaffected, while a hexagonal, rhombohedral or triclinic cell gets the
        cross terms ``b_i . b_j`` it needs instead of being refused.
        """
        from .poisson import fft_g_squared

        shape = tuple(int(n) for n in self.grid.shape)
        step = np.asarray(self.grid.step, dtype=float)
        key = (shape, step.tobytes())
        cached = getattr(self, "_g_squared_cache", None)
        if cached is None or cached[0] != key:
            cell = step @ np.diag(shape)
            cached = (key, fft_g_squared(shape, cell))
            self._g_squared_cache = cached
        return cached[1]

    def spectral_laplacian(self, samples) -> np.ndarray:
        r""":math:`\nabla^2` of each row of ``samples``, by FFT (``-|G|^2``).

        ``samples`` is ``(n, ngrid)`` (or one ``(ngrid,)`` row) on the
        flattened grid; the result has the same shape.  This is the operator
        ``kinetic="spectral"`` integrates: the exact Laplacian of the
        trigonometric interpolant of the samples.  It is real symmetric on the
        grid, which is what lets the nuclear gradient contract
        :math:`\langle\partial\phi_a|\nabla^2|\phi_b\rangle` against the
        Laplacian of the *unmoved* functions only
        (:func:`~mandacaru.algorithms.forces.one_body_cross`).
        """
        samples = np.asarray(samples)
        rows = samples.reshape(-1, self.grid.size)
        shape = tuple(int(n) for n in self.grid.shape)
        space = self._workspace()
        minus_k2 = -self._g_squared()
        if not np.iscomplexobj(rows):
            # Real samples: real transforms, a real Laplacian (|G|^2 is even).
            out = np.empty(rows.shape)
            space.convolve_real(np.ascontiguousarray(rows),
                                minus_k2[..., :space.half], out)
            return out.reshape(samples.shape)
        out = np.empty(rows.shape, dtype=np.complex128)
        # The transforms run in place on `out` itself, a batch of rows at a
        # time, through the grid's shared workspace (circular: no padding).
        step = space.batch(16)
        for start in range(0, len(rows), step):
            block = out[start:start + step]
            block[:] = rows[start:start + step]
            stack = block.reshape((-1,) + shape)
            space.forward(stack)
            space.multiply(stack, minus_k2)
            space.inverse(stack)
        return out.reshape(samples.shape)

    def _workspace(self):
        """The grid's shared FFT workspace (no padding)."""
        from .fft import workspace

        return workspace(self.grid.shape, None, -1)

    def _spectral_kinetic(self) -> np.ndarray:
        r"""``T[a,b] = -1/2 <psi_a | nabla^2 psi_b>`` with the FFT Laplacian.

        A batch of functions at a time (the workspace's batch of
        grid-sized transforms resident), so the memory cost is the basis
        stack already held plus that batch.
        """
        grid = self.grid
        M = self._psi.shape[0]
        T = np.zeros((M, M), dtype=self._psi.dtype)
        step = self._workspace().batch(16)
        conj_psi = np.conj(self._psi)
        for b0 in range(0, M, step):
            lap = self.spectral_laplacian(self._psi[b0:b0 + step])
            T[:, b0:b0 + step] = -0.5 * grid.dV * (conj_psi @ lap.T)
        return 0.5 * (T + T.conj().T)

    def resolution(self, T: np.ndarray | None = None,
                   kinetic: str = "fd") -> np.ndarray:
        r"""Grid-resolution check of every basis function: ``T_grid / T_exact``.

        For each function with a radial part (``radial(r)`` and ``l``), the
        exact radial kinetic energy
        :math:`\tfrac12\int[(rR)'^2 + l(l+1)R^2]\,dr` (per unit norm) is
        compared with the diagonal grid kinetic energy.  A ratio far from one
        means the function is not resolved by the grid -- too compact for the
        spacing -- and any energy it enters is unreliable.  Functions without
        a radial part get ``nan``.
        """
        if T is None:
            T = self.one_body(lambda x, y, z: np.zeros_like(x),
                              energy_units="Ha", kinetic=kinetic)[0]
        norms = np.real(np.einsum("ag,ag->a", np.conj(self._psi), self._psi)
                        * self.grid.dV)
        ratios = np.full(len(self.basis), np.nan)
        for a, fn in enumerate(self.basis):
            radial = getattr(fn, "radial", None)
            l = getattr(fn, "l", None)
            if radial is None or l is None:
                continue
            exact = radial_kinetic_energy(radial, int(l))
            if exact and norms[a] > 0:
                ratios[a] = float(np.real(T[a, a])) / norms[a] / exact
        return ratios

    # -- two body ---------------------------------------------------------- #

    def two_body(self, energy_units: str = "eV",
                 max_memory_mb: float | None = None, solver=None):
        r"""Electron-repulsion tensor over the basis, physicists' notation.

        Returns ``eri[a, b, c, d] = <ab|cd>``,

        .. math::

            \langle ab|cd\rangle = \iint
                \psi_a^*(\mathbf r_1)\,\psi_b^*(\mathbf r_2)\,
                \frac{1}{r_{12}}\,
                \psi_c(\mathbf r_1)\,\psi_d(\mathbf r_2)\,
                d\mathbf r_1\, d\mathbf r_2 ,

        i.e. electron 1 carries the index pair ``(a, c)`` and electron 2 the pair
        ``(b, d)``.  (In chemists' notation this is ``(ac|bd)``.)

        The pair potentials come from the O(N log N) FFT Poisson solver
        (:class:`~mandacaru.integrals.poisson.PoissonFFTSolver`), spectral at
        the :math:`1/r` singularity, so nothing singular is sampled.  It
        follows the grid's own step vectors, so anisotropic *and* skewed
        (non-orthogonal) grids are integrated with the right distances and
        volume.

        Parameters
        ----------
        energy_units : {"eV", "Ha"}
            Unit of the returned tensor (default ``"eV"``); the integrals are
            computed in Hartree and converted on return.
        max_memory_mb : float, optional
            Memory budget of the ``"fft"`` path's working set (pair densities
            and potentials are processed in blocks that fit it); default
            :func:`eri_memory_budget_mb`.
        solver : object, optional
            A Coulomb solver to use instead of the one this engine would pick.
            It must expose ``solve_stack``, ``L`` and ``dV``.  This is how a
            *second* tensor over the same orbitals is built with a different
            kernel -- the Wigner-Seitz-truncated bare ``1/r`` of
            :class:`~mandacaru.core.mpc.TruncatedCoulombSolver`, for the
            exchange-correlation hole -- so the two can be subtracted term by
            term.
        """
        with self.timings.time("two-body integrals (fft)"):
            eri = self._two_body_fft(max_memory_mb, solver=solver)
        # In place: the M^4 tensor is never copied for its units.
        eri *= from_hartree(1.0, energy_units)
        return eri

    def _two_body_fft(self, max_memory_mb: float | None = None, solver=None):
        r"""FFT-Poisson electron-repulsion tensor ``<ab|cd>`` (physicists').

        For every pair ``v = (b, d)`` with ``b <= d`` the density
        ``rho_bd = conj(psi_b) psi_d`` is solved for its potential
        ``Phi_bd``, and its whole ``(a, c)`` slab follows from one product,

            X_v[a, c] = sum_g conj(psi_a) psi_c Phi_bd dV
                      = ((conj(psi) * Phi_bd) @ psi.T) dV ,

        written straight into ``eri[:, b, :, d] = X_v``; the flipped pair is
        ``eri[:, d, :, b] = conj(X_v^T)``, because ``Phi_db = conj(Phi_bd)``
        (the kernel is real).  Only the ``U = M(M+1)/2`` pairs ``b <= d``
        are formed and solved, once each; no Gram matrix and no assembly
        step exist, so the peak is the ``M^4`` result plus the blocks below.

        **Blocking.**  The potentials are solved ``B`` pairs at a time into
        one reused ``B x G`` buffer, bounded by ``max_memory_mb`` (default
        :data:`ERI_MEMORY_MB`, env ``MANDACARU_ERI_MEMORY_MB``).  The products
        ``conj(psi) * Phi`` are formed tile by tile over the grid, so they
        never take more than a few MB.  When all ``U`` pair densities fit in
        half that budget they are formed once and kept, and the block's
        whole contraction is two matrix products against them
        (:meth:`_contract_cached`); otherwise the streamed products above
        (:func:`contract_pair_potentials`).  Both give the dense contraction to
        round-off.

        ``two_body_profile`` holds the seconds of each stage of the last
        call (pair densities, Poisson, contraction) and the route taken.
        """
        from time import perf_counter

        M = len(self.basis)
        ngrid = self.grid.size
        psi = self._psi                                          # (M, ngrid)
        dV = self.grid.dV
        if solver is None:
            solver = (PeriodicPoissonSolver(self.grid.shape, step=self.grid.step)
                      if getattr(self, "periodic", False)
                      else PoissonFFTSolver(self.grid.shape, step=self.grid.step))
        iu, ju = np.triu_indices(M)
        U = iu.size
        # The element-wise passes use as many threads as the solver's FFTs.
        workers = getattr(solver, "workers", -1)
        # Real functions (a molecule) on an even kernel: real pair densities
        # through the real transforms, real potentials, a real tensor.
        real = not np.iscomplexobj(psi) and bool(getattr(solver, "even",
                                                         False))
        dtype = np.float64 if real else np.complex128
        itemsize = np.dtype(dtype).itemsize
        refuse_unaffordable_eri(M, real)
        budget = (eri_memory_budget_mb() if max_memory_mb is None
                  else float(max_memory_mb)) * 2 ** 20
        cached = itemsize * U * ngrid <= 0.5 * budget
        B = self._eri_block_size(U, ngrid, solver, max_memory_mb,
                                 cached=cached, itemsize=itemsize)
        profile = {"pair densities": 0.0, "poisson": 0.0, "contraction": 0.0,
                   "route": "cached" if cached else "streamed",
                   "pair_block": B, "real": real}
        eri = np.empty((M, M, M, M), dtype=dtype)
        rho = (None if cached
               else np.empty((B, ngrid), dtype=psi.dtype))       # reused
        phi = np.empty((B, ngrid), dtype=dtype)                  # reused
        densities = None
        if cached:
            t0 = perf_counter()
            densities = np.empty((U, ngrid), dtype=psi.dtype)
            self._pair_block(psi, iu, ju, densities, workers)
            profile["pair densities"] += perf_counter() - t0
        conj_psi = None if cached else (psi if real else np.conj(psi))
        for v0 in range(0, U, B):
            v1 = min(v0 + B, U)
            nv = v1 - v0
            t0 = perf_counter()
            if cached:
                block = densities[v0:v1]
            else:
                block = self._pair_block(psi, iu[v0:v1], ju[v0:v1], rho[:nv],
                                         workers)
            t1 = perf_counter()
            solver.solve_stack(block, out=phi[:nv])
            t2 = perf_counter()
            if cached:
                X = self._contract_cached(densities, phi[:nv], M, iu, ju)
            else:
                X = contract_pair_potentials(conj_psi, psi, phi[:nv], workers)
            X *= dV
            b, d = iu[v0:v1], ju[v0:v1]
            # The flipped pair first, so a diagonal pair (b == d) keeps X.
            flipped = X.transpose(0, 2, 1)
            eri[:, d, :, b] = flipped if real else np.conj(flipped)
            eri[:, b, :, d] = X
            t3 = perf_counter()
            profile["pair densities"] += t1 - t0
            profile["poisson"] += t2 - t1
            profile["contraction"] += t3 - t2
        self.two_body_profile = profile
        return eri

    @staticmethod
    def _contract_cached(densities, phi, M, iu, ju) -> np.ndarray:
        r"""The same ``X`` from the kept pair densities ``rho_u`` (``u`` over
        ``a <= c``): ``sum_g rho_u Phi_v`` is one matrix product, and the
        entries with ``a > c`` are ``sum_g conj(rho_u) Phi_v``, the conjugate
        of a second product with ``conj(Phi)`` (conjugated in place).  Real
        densities and potentials need the first product only."""
        upper = densities @ phi.T                                 # (U, nv)
        index = np.empty((M, M), dtype=np.int64)
        index[iu, ju] = np.arange(iu.size)
        index[ju, iu] = np.arange(iu.size)
        if not (np.iscomplexobj(densities) or np.iscomplexobj(phi)):
            return upper.T[:, index]
        np.conjugate(phi, out=phi)
        lower = np.conj(densities @ phi.T)
        np.conjugate(phi, out=phi)
        below = (np.arange(M)[:, None] > np.arange(M)[None, :])
        X = np.where(below[None], lower.T[:, index], upper.T[:, index])
        return X

    @staticmethod
    def _pair_block(psi, rows_i, rows_j, out, workers: int = -1):
        """Fill ``out[k] = conj(psi[i_k]) * psi[j_k]`` row by row (no stack
        temps), the rows split over ``workers`` threads."""
        def fill(lo, hi):
            for k in range(lo, hi):
                np.conjugate(psi[rows_i[k]], out=out[k])
                out[k] *= psi[rows_j[k]]

        count = len(rows_i)
        threads = (1 if count * psi.shape[1] < PARALLEL_MIN_VALUES
                   else thread_count(workers))
        parallel_chunks(count, fill, threads)
        return out

    @staticmethod
    def _eri_block_size(n_pairs: int, ngrid: int, solver, max_memory_mb,
                        cached: bool = False, itemsize: int = 16) -> int:
        """Pairs per Poisson block that keep the two-body step within budget.

        Resident per block: the density buffer and the potentials (``2 B
        ngrid`` complex values; with the densities kept, only the potentials
        and the kept stack itself), plus the solver's FFT workspace and one
        contraction tile, independent of ``B``.
        """
        budget = eri_memory_budget_mb() if max_memory_mb is None else float(max_memory_mb)
        space = getattr(solver, "workspace", None)
        fixed = (space.batch_bytes if space is not None
                 else 3 * int(np.prod(solver.L)) * 16)
        fixed += CONTRACTION_TILE_BYTES
        if cached:
            fixed += itemsize * n_pairs * ngrid
        per_pair = (1 if cached else 2) * ngrid * itemsize
        avail = budget * 2 ** 20 - fixed
        B = int(avail // per_pair) if avail > 0 else 1
        return max(1, min(B, n_pairs, MAX_PAIR_BLOCK))
