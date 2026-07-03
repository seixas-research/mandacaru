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
from . import _backend
from .grid import Grid
from .poisson import PoissonFFTSolver

# A potential is any callable V(x, y, z) -> real array on the grid.
PotentialFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


class IntegralEngine:
    """Compute one- and two-body real-space integrals over a localized basis.

    Parameters
    ----------
    basis : sequence of BasisFunction
        The orbitals (hydrogenic, Wannier, ...) spanning the active space.
    grid : Grid
        The shared integration grid.  All functions are sampled on it.
    """

    def __init__(self, basis: Sequence[BasisFunction], grid: Grid):
        self.basis = list(basis)
        self.grid = grid
        # Sample every function once; reuse the (M, ngrid) stack for all
        # integrals.  This is the data actually shipped to C.
        self._psi = np.ascontiguousarray(
            np.stack([b.sample(grid) for b in self.basis]), dtype=np.complex128)

    @property
    def uses_c_backend(self) -> bool:
        return _backend.HAS_C_BACKEND

    # -- one body ---------------------------------------------------------- #

    def one_body(self, potential: PotentialFn):
        """Kinetic ``T`` and potential ``V`` matrices over the basis.

        Parameters
        ----------
        potential : callable
            ``V(x, y, z)`` returning the external potential (real) on the grid.

        Returns
        -------
        (T, V) : tuple of (M, M) complex ndarrays
            ``T[a,b] = <a| -1/2 nabla^2 |b>``, ``V[a,b] = <a| V |b>``.
        """
        Vext = np.ascontiguousarray(
            np.real(potential(self.grid.X, self.grid.Y, self.grid.Z)).reshape(-1),
            dtype=np.float64)
        return _backend.one_body_matrices(self._psi, Vext, self.grid.dx,
                                          self.grid.points)

    # -- two body ---------------------------------------------------------- #

    def two_body(self, method: str = "fft", softening: float = 0.0):
        """Electron-repulsion tensor ``(ab|cd)`` over the basis (chemists').

        Parameters
        ----------
        method : {"fft", "direct"}
            ``"fft"`` (default) uses the O(N log N) FFT Poisson solver with a
            physically correct cell self-energy -- fast and accurate.
            ``"direct"`` uses the O(N^2) real-space double sum in the C backend
            (kept as a reference / for arbitrary non-uniform grids).
        softening : float
            Only used by ``method="direct"``: regularizes ``r12 -> 0``.
        """
        if method == "fft":
            return self._two_body_fft()
        if method == "direct":
            xg, yg, zg = self.grid.flat_coords()
            return _backend.two_body_tensor(self._psi, xg, yg, zg,
                                            self.grid.dV, softening)
        raise ValueError(f"unknown two-body method {method!r}")

    def _two_body_fft(self):
        """FFT-Poisson electron-repulsion tensor.

        For every ordered pair build the density ``rho_ij = conj(psi_i) psi_j``;
        solve Poisson for each ``rho_bd`` to get ``Phi_bd``; then the whole
        tensor is a single dense contraction (BLAS GEMM)

            eri[a,b,c,d] = sum_g rho_ac[g] Phi_bd[g] dV .
        """
        M = len(self.basis)
        ngrid = self.grid.size
        psi = self._psi                                          # (M, ngrid)

        # Density pairs, row index p = i*M + j  ->  conj(psi_i) * psi_j.
        pairs = (np.conj(psi)[:, None, :] * psi[None, :, :]).reshape(M * M, ngrid)

        solver = PoissonFFTSolver(self.grid.points, self.grid.dx)
        phi_pairs = solver.solve_stack(pairs)                    # (M*M, ngrid)

        # R[(a,c),(b,d)] = sum_g pairs[(a,c),g] phi[(b,d),g] * dV.
        R = (pairs @ phi_pairs.T) * self.grid.dV                 # (M*M, M*M)
        # Reshape (a,c,b,d) -> transpose to chemists' (a,b,c,d).
        return R.reshape(M, M, M, M).transpose(0, 2, 1, 3).copy()
