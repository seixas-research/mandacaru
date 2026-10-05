# -*- coding: utf-8 -*-
# file: algorithms/periodic_device.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""The per-k-point linear algebra of the periodic Kohn-Sham SCF, behind a device.

Every SCF iteration does the same three things at every k-point of the mesh,
and nothing else that scales with the basis:

* build :math:`H(\mathbf k)` from the potentials on the grid
  (:meth:`CPUDevice.hamiltonians`);
* solve :math:`H(\mathbf k)\,c = \varepsilon\,S(\mathbf k)\,c`
  (:meth:`CPUDevice.eigensolve`);
* form :math:`P^k = C f C^\dagger` (:meth:`CPUDevice.density_matrices`)
  and from it the density;
* for a screened hybrid, the exact-exchange matrices of the mesh's occupied
  states (:meth:`CPUDevice.exchange_matrices`).

A device takes the whole mesh in one call per operation, so the SCF holds no
loop over k-points of its own, and a device that keeps the Bloch sums in
another memory (a GPU) can be put in its place.  :class:`CPUDevice` is the
reference: the grid products run one k-point at a time (each is already a
large BLAS call, and looping keeps the Bloch sums where they are) on one
BLAS thread per physical core, the ``M x M`` algebra for the whole mesh at
once on a single thread (:func:`~mandacaru.integrals._backend.
single_threaded_blas`: matrices that small lose to a thread team).

The generalized eigenproblem is reduced once per device: :math:`S = LL^\dagger`
does not change during the SCF, so :math:`L^{-1}` is kept and each
iteration solves the ordinary problem
:math:`L^{-1} H L^{-\dagger}\,y = \varepsilon y`, :math:`c = L^{-\dagger} y`.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from threadpoolctl import threadpool_limits

from ..integrals._backend import grid_blas_threads


def make_device(crystal, kpoint_data, gradients=None) -> "CPUDevice":
    """The device the per-k-point operations of ``kpoint_data`` run on.

    Every caller -- the SCF, the non-self-consistent bands, the forces'
    re-diagonalization -- takes its device from here, so a new device is
    chosen in one place.
    """
    return CPUDevice(crystal, kpoint_data, gradients)


def kohn_sham_matrix(crystal, data, V, v_tau, moment_operator,
                     gradients=None) -> np.ndarray:
    r""":math:`H(\mathbf k)` of one k-point's :class:`KPointMatrices`.

    ``V`` is the local potential on the grid, ``v_tau`` the meta-GGA
    :math:`\partial e/\partial\tau` (``None`` otherwise, when ``gradients``
    is unused) and ``moment_operator`` the ``(P, P)`` matrix of
    :meth:`~mandacaru.pseudopotentials.periodic_paw.PeriodicPAW.moment_operator`.
    """
    psi, dV = data.psi, crystal.grid.dV
    H = data.fixed + ((psi.conj() * V) @ psi.T) * dV
    if v_tau is not None:
        for d in gradients:
            H = H + 0.5 * ((d.conj() * v_tau) @ d.T) * dV
    C = data.projections
    H = H + C @ moment_operator @ C.conj().T
    return 0.5 * (H + H.conj().T)


class CPUDevice:
    """The per-k-point SCF operations on the host (NumPy, SciPy).

    Parameters
    ----------
    crystal : PeriodicPAW
        The crystal (grid, moment blocks, symmetry).
    kpoint_data : list of KPointMatrices
        The k-points this device works on: the SCF mesh, or a block of a band
        path.
    gradients : list, optional
        Per k-point Bloch gradients (meta-GGA only).
    """

    name = "cpu"

    def __init__(self, crystal, kpoint_data, gradients=None):
        self.crystal = crystal
        self.kpoint_data = list(kpoint_data)
        self.gradients = gradients
        self.grid_threads = grid_blas_threads()
        overlaps = np.stack([d.overlap for d in self.kpoint_data])
        factors = np.linalg.cholesky(overlaps)
        identity = np.eye(overlaps.shape[-1])
        #: ``L^{-1}`` per k-point, ``(nk, M, M)``.
        self._inverse_factor = np.stack([
            solve_triangular(L, identity, lower=True) for L in factors])

    def hamiltonians(self, V, v_tau, w, projector_operator=None) -> np.ndarray:
        """``(nk, M, M)``: :func:`kohn_sham_matrix` at every k-point;
        ``projector_operator`` (``(P, P)``, e.g. a hybrid's one-center terms)
        joins the moments' operator."""
        D = self.crystal.moment_operator(w)
        if projector_operator is not None:
            D = D + projector_operator
        with threadpool_limits(limits=self.grid_threads, user_api="blas"):
            return np.stack([
                kohn_sham_matrix(self.crystal, data, V, v_tau, D,
                                 self.gradients[i] if v_tau is not None
                                 else None)
                for i, data in enumerate(self.kpoint_data)])

    def eigensolve(self, H) -> tuple[np.ndarray, np.ndarray]:
        """``(eps, C)`` of ``H`` (``(..., nk, M, M)``, a leading spin axis
        allowed): eigenvalues ascending, ``(..., nk, M)``; eigenvectors
        ``S``-orthonormal, ``(..., nk, M, M)``."""
        X = self._inverse_factor
        eps, y = np.linalg.eigh(X @ H @ np.swapaxes(X, -1, -2).conj())
        return eps, np.swapaxes(X, -1, -2).conj() @ y

    def density_matrices(self, C, f) -> np.ndarray:
        """``C f C^dagger`` for occupations ``f`` (``(..., nk, M)``)."""
        return (C * f[..., None, :]) @ np.swapaxes(C, -1, -2).conj()

    def exchange_matrices(self, exchange, states) -> np.ndarray:
        """``(nk, M, M)``: the short-range exact exchange of the mesh's
        occupied ``states`` at this device's k-points
        (:meth:`~mandacaru.algorithms.periodic_exchange.PeriodicExchange.matrices`)."""
        return exchange.matrices(self.kpoint_data, states)

    def density(self, matrices) -> tuple[np.ndarray, dict]:
        """``(n~, q)`` of the density matrices of this device's k-points,
        symmetrized
        (:meth:`~mandacaru.pseudopotentials.periodic_paw.PeriodicPAW.density`)."""
        with threadpool_limits(limits=self.grid_threads, user_api="blas"):
            return self.crystal.density(matrices, self.kpoint_data)
