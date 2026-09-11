# -*- coding: utf-8 -*-
# file: integrals/engine.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Basis-agnostic real-space integral engine (Python front-end).

``IntegralEngine`` is a thin orchestration layer: it builds the grid, samples an
arbitrary list of :class:`~carcara.basis.base.BasisFunction` objects, evaluates
the external potential, and hands the contiguous arrays to the C backend
(:mod:`carcara.integrals._backend`).  It contains **no** knowledge of what the
basis functions are, which is exactly what lets Wannier or numerical orbitals be
dropped in unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ..basis.base import BasisFunction
from ..units import from_hartree
from . import _backend
from .grid import Grid
from .poisson import PoissonFFTSolver

#: Default memory budget (MB) of the FFT two-body working set; override per
#: call with ``two_body(max_memory_mb=...)`` or globally with the environment
#: variable ``CARCARA_ERI_MEMORY_MB``.
ERI_MEMORY_MB = 256.0


def eri_memory_budget_mb() -> float:
    """The two-body working-set budget in MB (``CARCARA_ERI_MEMORY_MB`` or default)."""
    import os
    value = os.environ.get("CARCARA_ERI_MEMORY_MB")
    if value is None:
        return ERI_MEMORY_MB
    budget = float(value)
    if not budget > 0:
        raise ValueError("CARCARA_ERI_MEMORY_MB must be positive")
    return budget

# A potential is any callable V(x, y, z) -> real array on the grid.
PotentialFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def radial_kinetic_energy(radial, l: int, r_max: float = 40.0,
                          points: int = 20001) -> float:
    r"""Exact kinetic energy per unit norm of ``R(r) Y_lm``,
    :math:`\tfrac12\int[(rR)'^2 + l(l+1)R^2]\,dr / \int R^2 r^2 dr`,
    from the radial function on a fine one-dimensional grid."""
    r = np.linspace(0.0, r_max, points)
    R = np.asarray(radial(r), dtype=float)
    if not np.all(np.isfinite(R)):
        R = np.nan_to_num(R)
    u = r * R
    du = np.gradient(u, r)
    norm = float(np.trapezoid(R * R * r * r, r))
    if norm <= 0:
        return 0.0
    centrifugal = l * (l + 1) * np.trapezoid(R[1:] ** 2, r[1:])
    return float(0.5 * (np.trapezoid(du * du, r) + centrifugal) / norm)


class IntegralEngine:
    """Compute one- and two-body real-space integrals over a localized basis.

    Parameters
    ----------
    basis : sequence of BasisFunction
        The orbitals (FAO, Wannier, ...) spanning the active space.
    grid : Grid
        The shared integration grid.  All functions are sampled on it.
    """

    def __init__(self, basis: Sequence[BasisFunction], grid: Grid):
        # Resolve the integral backend *before* any integration: prefer the C
        # library, compile it on the spot when it is missing, and only fall
        # back to the NumPy reference kernels when that compile fails (the
        # fallback is announced once per process).
        self.backend_status = _backend.ensure_backend()
        _backend.warn_fallback(self.backend_status)
        self.basis = list(basis)
        self.grid = grid
        # Sample every function once; reuse the (M, ngrid) stack for all
        # integrals.  This is the data actually shipped to C.
        self._psi = np.ascontiguousarray(
            np.stack([b.sample(grid) for b in self.basis]), dtype=np.complex128)

        # Profiling: wall-time per integral stage, plus the backend / core count
        # so the driver summary can report how the integration ran.
        from ..utils.profiling import Timings
        self.timings = Timings(
            n_cores=self.backend_status.n_threads,
            backend=self.backend_status.label)

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

    def _spectral_kinetic(self) -> np.ndarray:
        r"""``T[a,b] = -1/2 <psi_a | nabla^2 psi_b>`` with the FFT Laplacian.

        One function at a time (one grid-sized transform resident), so the
        memory cost is the basis stack already held plus a single grid.
        """
        from scipy import fft as sfft

        grid = self.grid
        if not grid.is_orthogonal:
            raise NotImplementedError(
                "the spectral kinetic operator needs an orthogonal grid; use "
                "kinetic='fd' for a non-orthogonal cell")
        nx, ny, nz = grid.shape
        kx = 2.0 * np.pi * sfft.fftfreq(nx, d=grid.dx)
        ky = 2.0 * np.pi * sfft.fftfreq(ny, d=grid.dy)
        kz = 2.0 * np.pi * sfft.fftfreq(nz, d=grid.dz)
        k2 = (kx[:, None, None] ** 2 + ky[None, :, None] ** 2
              + kz[None, None, :] ** 2)
        M = self._psi.shape[0]
        T = np.zeros((M, M), dtype=np.complex128)
        for b in range(M):
            psi_b = self._psi[b].reshape(nx, ny, nz)
            lap = sfft.ifftn(-k2 * sfft.fftn(psi_b, workers=-1),
                             workers=-1).reshape(-1)
            T[:, b] = -0.5 * grid.dV * (np.conj(self._psi) @ lap)
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

    def two_body(self, method: str = "fft", softening: float = 0.0,
                 energy_units: str = "eV", max_memory_mb: float | None = None):
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

        Parameters
        ----------
        method : {"fft", "direct"}
            ``"fft"`` (default) uses the O(N log N) FFT Poisson solver with a
            physically correct cell self-energy -- fast and accurate.
            ``"direct"`` uses the O(N^2) real-space double sum in the C backend
            (kept as a reference / for arbitrary non-uniform grids).
        softening : float
            Only used by ``method="direct"``: regularizes ``r12 -> 0``.
        energy_units : {"eV", "Ha"}
            Unit of the returned tensor (default ``"eV"``); the integrals are
            computed in Hartree and converted on return.
        max_memory_mb : float, optional
            Memory budget of the ``"fft"`` path's working set (pair densities
            and potentials are processed in blocks that fit it); default
            :func:`eri_memory_budget_mb`.
        """
        if method == "fft":
            with self.timings.time("two-body integrals (fft)"):
                eri = self._two_body_fft(max_memory_mb)
        elif method == "direct":
            xg, yg, zg = self.grid.flat_coords()
            # The backend already returns the physicists'-ordered tensor
            # eri[a,b,c,d] = <ab|cd> (electron 1 carries indices a, c).
            with self.timings.time("two-body integrals (direct)"):
                eri = _backend.two_body_tensor(self._psi, xg, yg, zg,
                                               self.grid.dV, softening)
        else:
            raise ValueError(f"unknown two-body method {method!r}")
        return from_hartree(eri, energy_units)

    def _two_body_fft(self, max_memory_mb: float | None = None):
        r"""FFT-Poisson electron-repulsion tensor ``<ab|cd>`` (physicists').

        For every pair build the density ``rho_ij = conj(psi_i) psi_j``; solve
        Poisson for ``rho_bd`` to get ``Phi_bd``; then

            eri[a,b,c,d] = <ab|cd> = sum_g rho_ac[g] Phi_bd[g] dV .

        Two things keep this cheap.  **Hermitian pairs:** ``rho_ji =
        conj(rho_ij)`` and the Coulomb kernel is real, so ``Phi_ji =
        conj(Phi_ij)``; only the ``U = M(M+1)/2`` pairs with ``i <= j`` are
        ever built or solved, and the full tensor is assembled from the two
        Gram blocks ``G1[u,v] = sum rho_u Phi_v`` and ``G2[u,v] = sum conj(rho_u)
        Phi_v`` (their conjugates give the flipped cases).  **Blocking:** the
        potentials are solved in blocks of ``B`` pairs and the densities are
        re-formed on the fly per block (an element-wise product, cheap), so
        the resident memory is two ``B x ngrid`` stacks plus the FFT buffers
        -- bounded by ``max_memory_mb`` (default :data:`ERI_MEMORY_MB`, env
        ``CARCARA_ERI_MEMORY_MB``) -- instead of two ``M^2 x ngrid`` stacks.
        The result is bit-for-bit the same tensor as the dense contraction
        up to round-off.
        """
        M = len(self.basis)
        ngrid = self.grid.size
        psi = self._psi                                          # (M, ngrid)
        dV = self.grid.dV
        solver = PoissonFFTSolver(self.grid.shape, self.grid.dx)

        # Unique pairs u = (i, j) with i <= j, and the lookup for any (a, c).
        iu, ju = np.triu_indices(M)
        U = iu.size
        uidx = np.empty((M, M), dtype=np.int64)
        uidx[iu, ju] = np.arange(U)
        uidx[ju, iu] = np.arange(U)
        flip = np.zeros((M, M), dtype=bool)                      # a > c
        flip[ju, iu] = True
        np.fill_diagonal(flip, False)

        B = self._eri_block_size(U, ngrid, solver, max_memory_mb)
        G1 = np.empty((U, U), dtype=np.complex128)
        G2 = np.empty((U, U), dtype=np.complex128)
        rho = np.empty((B, ngrid), dtype=np.complex128)          # reused buffer
        for v0 in range(0, U, B):
            v1 = min(v0 + B, U)
            nv = v1 - v0
            self._pair_block(psi, iu[v0:v1], ju[v0:v1], rho[:nv])
            phi = solver.solve_stack(rho[:nv])                   # (nv, ngrid)
            phi_c = np.conj(phi)
            # G2 = conj(rho) @ phi^T = conj(rho @ conj(phi)^T): one conjugate
            # per block instead of one per (u, v) product.  The transposed
            # operands are BLAS-native (no copies).
            for u0 in range(0, U, B):
                u1 = min(u0 + B, U)
                nu = u1 - u0
                self._pair_block(psi, iu[u0:u1], ju[u0:u1], rho[:nu])
                G1[u0:u1, v0:v1] = rho[:nu] @ phi.T
                np.conjugate(rho[:nu] @ phi_c.T, out=G2[u0:u1, v0:v1])
            del phi, phi_c
        del rho
        G1 *= dV
        G2 *= dV

        # R[(a,c),(b,d)] = sum_g rho_ac Phi_bd dV, with rho_ac = conj(rho_u) when
        # a > c and Phi_bd = conj(Phi_v) when b > d.
        u = uidx.reshape(-1)                                     # over (a, c)
        fa = flip.reshape(-1)
        g1 = G1[u][:, u]
        g2 = G2[u][:, u]
        fa_ = fa[:, None]
        fb_ = fa[None, :]
        R = np.where(~fa_ & ~fb_, g1,
            np.where(fa_ & ~fb_, g2,
            np.where(~fa_ & fb_, np.conj(g2), np.conj(g1))))
        # (a, c, b, d) -> physicists' (a, b, c, d).
        return R.reshape(M, M, M, M).transpose(0, 2, 1, 3).copy()

    @staticmethod
    def _pair_block(psi, rows_i, rows_j, out):
        """Fill ``out[k] = conj(psi[i_k]) * psi[j_k]`` row by row (no stack temps)."""
        for k, (i, j) in enumerate(zip(rows_i, rows_j)):
            np.conjugate(psi[i], out=out[k])
            out[k] *= psi[j]
        return out

    @staticmethod
    def _eri_block_size(n_pairs: int, ngrid: int, solver, max_memory_mb) -> int:
        """Pairs per Poisson block that keep the two-body step within budget.

        Resident per block: the density buffer, the potentials and their
        conjugate (``3 B ngrid`` complex values; the solver's own output
        stack overlaps with the conjugate's lifetime) plus the solver's three
        padded FFT work volumes, independent of ``B``.
        """
        budget = eri_memory_budget_mb() if max_memory_mb is None else float(max_memory_mb)
        padded = 3 * int(np.prod(solver.L)) * 16
        per_pair = 3 * ngrid * 16
        avail = budget * 2 ** 20 - padded
        B = int(avail // per_pair) if avail > 0 else 1
        return max(1, min(B, n_pairs))
