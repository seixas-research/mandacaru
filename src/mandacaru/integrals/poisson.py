# -*- coding: utf-8 -*-
# file: integrals/poisson.py

# This code is part of Mandacaru.
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
  self-interaction.  Instead of an ad-hoc softening we integrate :math:`1/r`
  over the voxel itself, :math:`G(0)=\frac{1}{dV}\int_\text{cell}d^3r/|r|`
  (:func:`cell_self_potential`), the physically correct cell self-energy.  For
  a cube that is :math:`C_\text{cube}/dx` with
  :math:`C_\text{cube}=\int_{[-1/2,1/2]^3} d^3u/|u| \approx 2.3800774`.

* **Voxel geometry.** The displacement between two nodes of *any* Bravais
  sampling is ``step @ (di, dj, dk)``: it depends only on the index
  difference, so the convolution structure holds for anisotropic **and**
  skewed grids alike.  Distances come from the grid's step matrix and the
  source volume is ``|det(step)|``; only the ``d = 0`` voxel needs its own
  treatment (:func:`voxel_self_potential`).

``scipy.fft`` (pocketfft) runs the transforms in threaded C; ``workers=-1``
uses all cores, so this path is parallel without any custom kernel.
"""

from __future__ import annotations

import numpy as np
from scipy import fft as sfft

#: Average of 1/r over a unit cube centered at the origin (see module docstring).
#: Kept as a reference value; :func:`cell_self_potential` is the closed form.
CUBE_SELF_CONSTANT = 2.3800756


def _triangle_potential(a, b, c) -> float:
    r"""``int_T dS/|r|`` over the triangle ``(a, b, c)``, field point at the origin.

    The closed form for a uniformly charged planar polygon: one logarithm along
    each edge plus an arctangent pair for the out-of-plane offset.
    ``perpendicular`` is the in-plane distance from the projected field point to
    the edge line, ``l_*`` the endpoint coordinates along the edge and ``r_*``
    their distances to the field point.
    """
    normal = np.cross(b - a, c - a)
    norm = float(np.linalg.norm(normal))
    if norm == 0.0:
        return 0.0                                    # degenerate triangle
    normal = normal / norm
    offset = float(normal @ a)                        # signed plane distance
    projected = offset * normal
    total = 0.0
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        length = float(np.linalg.norm(edge))
        if length == 0.0:
            continue
        along = edge / length
        outward = np.cross(along, normal)
        perpendicular = float(outward @ (start - projected))
        l_minus = float((start - projected) @ along)
        l_plus = float((end - projected) @ along)
        r_minus = float(np.linalg.norm(start))
        r_plus = float(np.linalg.norm(end))
        r0_sq = perpendicular * perpendicular + offset * offset
        if perpendicular != 0.0 and (r_plus + l_plus) > 0.0 \
                and (r_minus + l_minus) > 0.0:
            total += perpendicular * np.log((r_plus + l_plus)
                                            / (r_minus + l_minus))
        total -= abs(offset) * (
            np.arctan2(perpendicular * l_plus, r0_sq + abs(offset) * r_plus)
            - np.arctan2(perpendicular * l_minus,
                         r0_sq + abs(offset) * r_minus))
    return float(total)


def voxel_self_potential(step) -> float:
    r"""``int_cell d^3r/|r|`` over the voxel of a general ``step`` matrix.

    ``step`` holds the three step vectors as columns, so the voxel is the
    parallelepiped :math:`\{\sum_m u_m s_m : u \in [-1/2, 1/2]^3\}` centered on
    the node.  Because :math:`\nabla^2 r = 2/r`, the divergence theorem turns
    the volume integral into a sum over the six faces,

    .. math::

        \int_V \frac{d^3r}{|r|}
            = \tfrac12 \oint_{\partial V} \hat r\cdot\hat n \, dS
            = \tfrac12 \sum_f h_f \int_f \frac{dS}{|r|} ,

    with :math:`h_f` the (constant) distance from the node to face :math:`f`;
    each parallelogram face splits into two triangles integrated in closed form
    (:func:`_triangle_potential`).  An orthogonal voxel takes the simpler
    rectangular formula :func:`cell_self_potential` instead -- the two agree to
    machine precision.
    """
    step = np.asarray(step, dtype=float)
    if step.shape != (3, 3):
        raise ValueError(f"step must be a 3x3 matrix, got {step.shape}")
    volume = abs(float(np.linalg.det(step)))
    if volume <= 0.0:
        raise ValueError("the voxel has zero volume (degenerate step matrix)")
    diagonal = np.abs(np.diag(step))
    off_diagonal = step - np.diag(np.diag(step))
    if np.all(np.abs(off_diagonal) <= 1e-12 * max(float(diagonal.max()), 1.0)):
        return cell_self_potential(*diagonal)

    corners = {signs: step @ (np.array(signs, dtype=float) - 0.5)
               for signs in np.ndindex(2, 2, 2)}
    total = 0.0
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        for side in (0, 1):
            face = [s for s in np.ndindex(2, 2, 2) if s[axis] == side]
            # Order the four corners around the face (a cycle, not a zig-zag).
            face.sort(key=lambda s: (s[others[0]], s[others[1]] ^ s[others[0]]))
            v = [corners[s] for s in face]
            normal = np.cross(v[1] - v[0], v[2] - v[0])
            normal = normal / np.linalg.norm(normal)
            height = abs(float(normal @ v[0]))
            total += height * (_triangle_potential(v[0], v[1], v[2])
                               + _triangle_potential(v[0], v[2], v[3]))
    return float(0.5 * total)


def cell_self_potential(dx: float, dy: float, dz: float) -> float:
    r"""``int_cell d^3r/|r|`` over a rectangular voxel centered at the origin.

    The closed form of the box integral (Bohr\ :sup:`2`), eight times its
    positive octant.  Divided by the voxel volume it is the Green's function at
    ``d = 0``: the average of ``1/r`` over the cell one source node stands for.
    For a cube it gives ``dx**2 * 2.3800774``.
    """
    a, b, c = 0.5 * float(dx), 0.5 * float(dy), 0.5 * float(dz)
    if min(a, b, c) <= 0.0:
        raise ValueError(f"grid spacings must be positive, got {(dx, dy, dz)}")
    s = np.sqrt(a * a + b * b + c * c)
    octant = (a * b * np.log((c + s) / np.hypot(a, b))
              + b * c * np.log((a + s) / np.hypot(b, c))
              + c * a * np.log((b + s) / np.hypot(c, a))
              - 0.5 * a * a * np.arctan(b * c / (a * s))
              - 0.5 * b * b * np.arctan(c * a / (b * s))
              - 0.5 * c * c * np.arctan(a * b / (c * s)))
    return float(8.0 * octant)


class PoissonFFTSolver:
    """Solve the grid Coulomb convolution by zero-padded FFT.

    Parameters
    ----------
    shape : int or (int, int, int)
        Nodes per Cartesian axis of the source grid.  A scalar means a cubic
        grid ``(N, N, N)``; a triple ``(nx, ny, nz)`` a non-cubic one -- the FFT
        convolution is agnostic to the box shape, only the per-axis lengths and
        padding differ.
    spacing : float or (float, float, float), optional
        Grid spacing in Bohr: one value for every axis, or the per-axis
        ``(dx, dy, dz)`` of an anisotropic grid.  Ignored when ``step`` is given.
    step : (3, 3) array_like, optional
        The full voxel basis (step vectors as columns, i.e. ``Grid.step``), for
        a **skewed** grid: distances then use the lattice displacement
        ``step @ (di, dj, dk)`` and the volume is ``|det(step)|``.
    self_const : float, optional
        Overrides ``G(0)`` with the legacy cubic rule ``self_const / dx``.  By
        default ``G(0)`` is the exact cell average
        (:func:`cell_self_potential`) -- the same thing for a cube.
    workers : int, optional
        Threads for the FFTs (``-1`` uses all cores).
    """

    def __init__(self, shape, spacing=None, self_const: float | None = None,
                 workers: int = -1, step=None):
        if np.isscalar(shape):
            self.shape = (int(shape),) * 3
        else:
            self.shape = tuple(int(s) for s in shape)
        if step is not None:
            self.step = np.asarray(step, dtype=float)
            if self.step.shape != (3, 3):
                raise ValueError(f"step must be a 3x3 matrix, got "
                                 f"{self.step.shape}")
            self.spacing = tuple(float(np.linalg.norm(self.step[:, m]))
                                 for m in range(3))
        else:
            if spacing is None:
                raise ValueError("give either a spacing or a step matrix")
            self.spacing = ((float(spacing),) * 3 if np.isscalar(spacing)
                            else tuple(float(v) for v in spacing))
            if len(self.spacing) != 3 or min(self.spacing) <= 0.0:
                raise ValueError(f"spacing must be three positive lengths, got "
                                 f"{spacing!r}")
            self.step = np.diag(np.asarray(self.spacing, dtype=float))
        self.dx, self.dy, self.dz = self.spacing
        self.dV = abs(float(np.linalg.det(self.step)))
        if self.dV <= 0.0:
            raise ValueError("the voxel has zero volume (degenerate step)")
        self.workers = workers
        # Pad each axis to >= 2N-1 (FFT-friendly length) to avoid wraparound.
        self.L = tuple(sfft.next_fast_len(2 * n - 1) for n in self.shape)
        self._Gk = self._build_kernel_transform(self_const)

    def _build_kernel_transform(self, self_const: float | None) -> np.ndarray:
        """Precompute FFT of the 1/r Green's function on the padded grid."""
        offs = []
        for n, L in zip(self.shape, self.L):
            # Signed integer offsets: 0..n-1 positive, top of the array negative.
            idx = np.arange(L)
            offs.append(np.where(idx < n, idx, idx - L).astype(float))
        # Node (i, j, k) sits at step @ (i, j, k) from the origin node, so the
        # displacement depends only on the index difference -- skew included.
        SX = offs[0][:, None, None]
        SY = offs[1][None, :, None]
        SZ = offs[2][None, None, :]
        dist_sq = np.zeros(self.L, dtype=float)
        for row in range(3):
            component = (self.step[row, 0] * SX + self.step[row, 1] * SY
                         + self.step[row, 2] * SZ)
            dist_sq += component * component
        dist = np.sqrt(dist_sq, out=dist_sq)
        with np.errstate(divide="ignore"):
            G = np.where(dist > 0, 1.0 / dist, 0.0)
        # The d = 0 node carries the voxel's own average of 1/r.
        G[0, 0, 0] = (self_const / self.dx if self_const is not None
                      else voxel_self_potential(self.step) / self.dV)
        return sfft.fftn(G, workers=self.workers)

    def solve(self, rho_flat: np.ndarray) -> np.ndarray:
        """Coulomb potential of a single density on the grid (flattened)."""
        return self.solve_stack(rho_flat[None, :])[0]

    def solve_stack(self, rho_stack: np.ndarray) -> np.ndarray:
        """Coulomb potentials of a stack of ``P`` densities.

        Parameters
        ----------
        rho_stack : (P, nx*ny*nz) complex
            Densities sampled on the flattened grid.

        Returns
        -------
        (P, nx*ny*nz) complex
            The corresponding Coulomb potentials ``Phi``.
        """
        (nx, ny, nz), dV = self.shape, self.dV
        ngrid = nx * ny * nz
        rho_stack = np.ascontiguousarray(rho_stack, dtype=np.complex128)
        P = rho_stack.shape[0]
        out = np.empty((P, ngrid), dtype=np.complex128)
        pad = np.zeros(self.L, dtype=np.complex128)
        for p in range(P):
            pad[:] = 0.0
            pad[:nx, :ny, :nz] = rho_stack[p].reshape(nx, ny, nz)
            spec = sfft.fftn(pad, workers=self.workers)
            phi = sfft.ifftn(spec * self._Gk, workers=self.workers)
            out[p] = phi[:nx, :ny, :nz].reshape(-1) * dV
        return out
