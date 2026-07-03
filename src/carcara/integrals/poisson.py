# -*- coding: utf-8 -*-
# file: integrals/poisson.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""FFT Poisson solver for the two-body Coulomb potential.

The electron-repulsion tensor needs, for every density pair
:math:`\rho(\mathbf r) = \phi_i^*(\mathbf r)\phi_j(\mathbf r)`, its Coulomb
potential on the grid

.. math::

    \Phi(\mathbf r_1) = \sum_{\mathbf r_2} \frac{\rho(\mathbf r_2)}
                                                 {|\mathbf r_1-\mathbf r_2|}\, dV .

That is a discrete convolution of :math:`\rho` with the Green's function
:math:`G(\mathbf d)=1/|\mathbf d|`.  Evaluated directly it costs
:math:`O(N_\text{grid}^2)`; as an FFT convolution it costs
:math:`O(N_\text{grid}\log N_\text{grid})`:

.. math::

    \Phi = \mathrm{IFFT}\big(\mathrm{FFT}(\rho)\cdot \mathrm{FFT}(G)\big) .

Two numerical points make this correct rather than merely fast:

* **Zero-padding.** The FFT computes a *circular* convolution.  Without padding
  each axis to at least :math:`2N-1`, the long Coulomb tail wraps around the box
  and contaminates the potential.  We pad to ``scipy.fft.next_fast_len(2N-1)``.
* **Self term.** The :math:`\mathbf d=0` node is the singular
  self-interaction.  Instead of an ad-hoc softening we use the analytic average
  of :math:`1/r` over one cubic cell, :math:`G(0)=C_\text{cube}/dx` with
  :math:`C_\text{cube}=\int_{[-1/2,1/2]^3} d^3u/|u| \approx 2.3800756`, which is
  the physically correct cell self-energy.

``scipy.fft`` (pocketfft) runs the transforms in threaded C; ``workers=-1``
uses all cores, so this path is parallel without any custom kernel.
"""

from __future__ import annotations

import numpy as np
from scipy import fft as sfft

#: Average of 1/r over a unit cube centered at the origin (see module docstring).
CUBE_SELF_CONSTANT = 2.3800756


class PoissonFFTSolver:
    """Solve the grid Coulomb convolution by zero-padded FFT.

    Parameters
    ----------
    points : int
        Nodes per Cartesian dimension of the (cubic) source grid, ``N``.
    dx : float
        Grid spacing in Bohr.
    self_const : float, optional
        Cell self-energy constant :math:`C_\\text{cube}` used for ``G(0)``.
    workers : int, optional
        Threads for the FFTs (``-1`` uses all cores).
    """

    def __init__(self, points: int, dx: float,
                 self_const: float = CUBE_SELF_CONSTANT, workers: int = -1):
        self.N = int(points)
        self.dx = float(dx)
        self.workers = workers
        # Pad each axis to >= 2N-1 (FFT-friendly length) to avoid wraparound.
        self.L = sfft.next_fast_len(2 * self.N - 1)
        self._Gk = self._build_kernel_transform(self_const)

    def _build_kernel_transform(self, self_const: float) -> np.ndarray:
        """Precompute FFT of the 1/r Green's function on the padded grid."""
        N, L, dx = self.N, self.L, self.dx
        # Signed integer offsets: 0..N-1 positive, top of the array negative.
        idx = np.arange(L)
        offs = np.where(idx < N, idx, idx - L).astype(float)
        SX = offs[:, None, None]
        SY = offs[None, :, None]
        SZ = offs[None, None, :]
        dist = dx * np.sqrt(SX * SX + SY * SY + SZ * SZ)
        with np.errstate(divide="ignore"):
            G = np.where(dist > 0, 1.0 / dist, 0.0)
        G[0, 0, 0] = self_const / dx  # correct cell self-interaction
        return sfft.fftn(G, workers=self.workers)

    def solve(self, rho_flat: np.ndarray) -> np.ndarray:
        """Coulomb potential of a single density on the grid (flattened)."""
        return self.solve_stack(rho_flat[None, :])[0]

    def solve_stack(self, rho_stack: np.ndarray) -> np.ndarray:
        """Coulomb potentials of a stack of ``P`` densities.

        Parameters
        ----------
        rho_stack : (P, N**3) complex
            Densities sampled on the flattened cubic grid.

        Returns
        -------
        (P, N**3) complex
            The corresponding Coulomb potentials ``Phi``.
        """
        N, L, dV = self.N, self.L, self.dx ** 3
        rho_stack = np.ascontiguousarray(rho_stack, dtype=np.complex128)
        P = rho_stack.shape[0]
        out = np.empty((P, N ** 3), dtype=np.complex128)
        pad = np.zeros((L, L, L), dtype=np.complex128)
        for p in range(P):
            pad[:] = 0.0
            pad[:N, :N, :N] = rho_stack[p].reshape(N, N, N)
            spec = sfft.fftn(pad, workers=self.workers)
            phi = sfft.ifftn(spec * self._Gk, workers=self.workers)
            out[p] = phi[:N, :N, :N].reshape(-1) * dV
        return out
