# -*- coding: utf-8 -*-
# file: integrals/grid.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Regular Cartesian integration grid.

The Python front-end owns the grid: it defines the sampling points, the per-axis
spacing and the volume element, then hands the flat coordinate arrays (and the
grid geometry) to the C backend (``csrc/carcara_integrals.c``).

Grid shapes supported, in increasing generality:

* **Cubic** (the historical default) -- a single ``box_size`` half-edge and an
  equal node count and spacing on every axis.
* **Orthorhombic** -- either a length-3 ``box_size`` (per-axis half-extents), a
  length-3 ``h`` (per-axis spacing, i.e. *varying resolution along each axis*),
  or a diagonal ``cell`` tensor.  The three axes stay mutually orthogonal but may
  differ in both extent and spacing.
* **Non-orthogonal** -- a full ``cell`` tensor with ``skew=True`` samples on the
  actual (skewed) lattice: node ``(i, j, k)`` sits at
  ``origin + i s1 + j s2 + k s3`` with the step vectors ``s_m = cell[m]/(n_m-1)``.
  The default (``skew=False``) instead spans the axis-aligned bounding box of the
  cell -- an orthorhombic grid that merely *encloses* the parallelepiped.

The geometry is captured once in the :attr:`step` matrix (columns are the three
grid step vectors).  From it the backend gets the voxel volume ``dV = |det step|``
and the inverse metric ``(step^T step)^{-1}`` that generalizes the finite-
difference Laplacian to anisotropic and non-orthogonal grids.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..units import to_bohr


@dataclass
class Grid:
    """A regular Cartesian grid centered at ``center``.

    The grid is specified by a physical **spacing** rather than a node count: the
    node count on each axis is derived so the realized spacing is as close as
    possible to ``h`` (per axis).

    Lengths are given in ``units`` (**Angstrom** by default).  Internally the grid
    is built in atomic units: the sampled coordinate arrays :attr:`X`, :attr:`Y`,
    :attr:`Z`, the spacings :attr:`dx`/:attr:`dy`/:attr:`dz` and the :attr:`step`
    matrix are always in **Bohr**, which is what the integral backend consumes.

    Parameters
    ----------
    center : array_like, shape (3,)
        Cartesian center of the box, in ``units``.
    box_size : float or array_like, shape (3,)
        Half-edge length of the cube, in ``units`` (the box spans
        ``center ± box_size``).  A length-3 value gives an orthorhombic box with
        per-axis half-extents.  Ignored when ``cell`` is given.
    h : float or array_like, shape (3,)
        Target grid spacing, in ``units`` (default ``0.20``).  A length-3 value
        sets a **different resolution along each axis**.
    units : {"angstrom", "bohr"}
        Unit of ``center``, ``box_size``, ``h`` and ``cell`` (default
        ``"angstrom"``).
    cell : array_like, shape (3, 3), optional
        Full lattice/cell tensor (rows are lattice vectors, ASE convention).
        Overrides ``box_size``.  With ``skew=False`` (default) the grid spans the
        axis-aligned bounding box of the parallelepiped; with ``skew=True`` the
        grid samples the skewed lattice directly (non-orthogonal grid).
    skew : bool
        Sample the non-orthogonal lattice directly instead of its bounding box
        (only meaningful together with ``cell``).  Default ``False``.

    Attributes
    ----------
    points : int
        Number of nodes on the x-axis (== every axis when cubic).  Kept for
        backward compatibility; use :attr:`shape` in the general case.
    shape : (int, int, int)
        Nodes per Cartesian axis ``(nx, ny, nz)``.
    dx, dy, dz : float
        Realized per-axis spacings in Bohr (equal for a cubic grid; ``dx`` is the
        x-axis step-vector length in the skewed case).
    step : (3, 3) ndarray
        Grid step vectors in Bohr; column ``m`` is ``s_m`` so a node at integer
        index ``(i, j, k)`` sits at ``origin + step @ (i, j, k)``.
    """

    center: np.ndarray
    box_size: float = 5.0
    h: float = 0.20
    units: str = "angstrom"
    cell: np.ndarray | None = None
    skew: bool = False

    points: int = field(init=False)
    nx: int = field(init=False)
    ny: int = field(init=False)
    nz: int = field(init=False)
    X: np.ndarray = field(init=False, repr=False)
    Y: np.ndarray = field(init=False, repr=False)
    Z: np.ndarray = field(init=False, repr=False)
    dx: float = field(init=False)
    dy: float = field(init=False)
    dz: float = field(init=False)
    step: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        # Convert the user-facing lengths to the atomic units the backend uses.
        center_bohr = np.asarray(to_bohr(self.center, self.units), dtype=float)
        h_bohr = np.abs(np.asarray(to_bohr(self.h, self.units), dtype=float))
        h_scalar = h_bohr.ndim == 0
        h_axes = np.broadcast_to(h_bohr, (3,)).astype(float)

        box = None if self.cell is not None else np.asarray(self.box_size, float)
        cubic = (self.cell is None and box.ndim == 0 and h_scalar)

        if cubic:
            # -- Cubic path: preserve the historical node/spacing arithmetic
            #    exactly so existing integrals are byte-for-byte unchanged. --
            box_bohr = float(to_bohr(self.box_size, self.units))
            self.points = max(2, int(round(2.0 * box_bohr / h_axes[0])) + 1)
            axis = np.linspace(-box_bohr, box_bohr, self.points)
            self.dx = self.dy = self.dz = float(axis[1] - axis[0])
            self.nx = self.ny = self.nz = self.points
            Xr, Yr, Zr = np.meshgrid(axis, axis, axis, indexing="ij")
            self.step = np.diag([self.dx, self.dy, self.dz])
            self.X = Xr + center_bohr[0]
            self.Y = Yr + center_bohr[1]
            self.Z = Zr + center_bohr[2]
            return

        if self.cell is not None and self.skew:
            self._build_skewed(center_bohr, h_axes)
        else:
            self._build_orthorhombic(center_bohr, h_axes, box)

    # -- orthorhombic (bounding box; possibly per-axis spacing) ----------- #

    def _build_orthorhombic(self, center_bohr, h_axes, box):
        if self.cell is not None:
            half = self._half_extents_from_cell()
        else:
            half = np.abs(np.asarray(to_bohr(box, self.units), dtype=float))
            if half.ndim == 0:                      # scalar box, per-axis h
                half = np.full(3, float(half))
            elif half.shape != (3,):
                raise ValueError(
                    "box_size must be a scalar or a length-3 array of per-axis "
                    f"half-extents, got shape {half.shape}")
        # Keep the per-axis spacing exactly at the target ``h`` (as the historical
        # non-cubic path did): choose the node count nearest that spacing, then
        # realize each half-extent from the count so ``dx`` stays exact.
        counts = [max(2, int(round(2.0 * L / h)) + 1)
                  for L, h in zip(half, h_axes)]
        self.nx, self.ny, self.nz = int(counts[0]), int(counts[1]), int(counts[2])
        self.dx, self.dy, self.dz = (float(h_axes[0]), float(h_axes[1]),
                                     float(h_axes[2]))
        axes = [np.linspace(-d * (n - 1) / 2.0, d * (n - 1) / 2.0, n)
                for d, n in zip((self.dx, self.dy, self.dz), counts)]
        self.points = self.nx
        self.step = np.diag([self.dx, self.dy, self.dz])
        Xr, Yr, Zr = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
        self.X = Xr + center_bohr[0]
        self.Y = Yr + center_bohr[1]
        self.Z = Zr + center_bohr[2]

    # -- non-orthogonal (skewed lattice sampling) ------------------------- #

    def _build_skewed(self, center_bohr, h_axes):
        cell = np.asarray(self.cell, dtype=float)
        if cell.shape != (3, 3):
            raise ValueError(f"cell must be a (3, 3) tensor, got {cell.shape}")
        cell_bohr = np.asarray(to_bohr(cell, self.units), dtype=float)
        lengths = np.linalg.norm(cell_bohr, axis=1)          # lattice-vector norms
        counts = [max(2, int(round(L / h)) + 1)
                  for L, h in zip(lengths, h_axes)]
        self.nx, self.ny, self.nz = int(counts[0]), int(counts[1]), int(counts[2])
        # Step vectors: s_m = a_m / (n_m - 1); columns of `step`.
        step = np.empty((3, 3))
        for m in range(3):
            step[:, m] = cell_bohr[m] / (counts[m] - 1)
        self.step = step
        self.dx = float(np.linalg.norm(step[:, 0]))
        self.dy = float(np.linalg.norm(step[:, 1]))
        self.dz = float(np.linalg.norm(step[:, 2]))
        self.points = self.nx

        idx = np.stack(np.meshgrid(np.arange(self.nx), np.arange(self.ny),
                                   np.arange(self.nz), indexing="ij"), axis=-1)
        # r = origin + step @ (i, j, k); center the parallelepiped on `center`.
        r = idx @ step.T                                     # (nx, ny, nz, 3)
        r_center = 0.5 * (np.array([self.nx - 1, self.ny - 1, self.nz - 1])
                          @ step.T)
        r = r - r_center + center_bohr
        self.X = np.ascontiguousarray(r[..., 0])
        self.Y = np.ascontiguousarray(r[..., 1])
        self.Z = np.ascontiguousarray(r[..., 2])

    def _half_extents_from_cell(self) -> np.ndarray:
        """Per-axis half-extents (Bohr) of the bounding box of the cell.

        The three lattice vectors (rows of ``cell``) span a parallelepiped; its
        axis-aligned bounding box has full extent ``sum_k |cell[k, axis]|`` along
        each Cartesian axis.  Using the bounding box means a non-orthogonal cell
        is still enclosed by the (uniform, rectangular) grid.
        """
        cell = np.asarray(self.cell, dtype=float)
        if cell.shape != (3, 3):
            raise ValueError(f"cell must be a (3, 3) tensor, got {cell.shape}")
        extent = np.sum(np.abs(cell), axis=0)          # full length per axis
        return 0.5 * np.abs(np.asarray(to_bohr(extent, self.units), dtype=float))

    # -- geometry views --------------------------------------------------- #

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    @property
    def is_cubic(self) -> bool:
        """True when every axis has the same node count and spacing (a cube)."""
        return (self.nx == self.ny == self.nz and self.is_orthorhombic
                and np.isclose(self.dx, self.dy) and np.isclose(self.dy, self.dz))

    @property
    def is_orthorhombic(self) -> bool:
        """True when the step vectors are mutually orthogonal (diagonal step)."""
        off = self.step - np.diag(np.diag(self.step))
        return bool(np.allclose(off, 0.0))

    @property
    def size(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def dV(self) -> float:
        """Voxel volume ``|det(step)|`` (``dx*dy*dz`` for an orthorhombic grid)."""
        return float(abs(np.linalg.det(self.step)))

    def metric_inverse(self) -> np.ndarray:
        """Inverse metric ``(step^T step)^{-1}`` for the general FD Laplacian.

        The Laplacian in index coordinates is
        ``nabla^2 f = sum_{a,b} G^{-1}_{ab} d_a d_b f`` with ``G = step^T step``;
        for an orthorhombic grid this is ``diag(1/dx^2, 1/dy^2, 1/dz^2)`` and the
        cross terms vanish.
        """
        G = self.step.T @ self.step
        return np.linalg.inv(G)

    def flat_coords(self):
        """Contiguous flat ``(x, y, z)`` coordinate vectors for the C backend."""
        return (np.ascontiguousarray(self.X.reshape(-1)),
                np.ascontiguousarray(self.Y.reshape(-1)),
                np.ascontiguousarray(self.Z.reshape(-1)))
