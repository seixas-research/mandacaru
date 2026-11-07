# -*- coding: utf-8 -*-
# file: integrals/grid.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Regular Cartesian integration grid.

The Python front-end owns the grid: it defines the sampling points, the per-axis
spacing and the volume element, then hands the flat coordinate arrays (and the
grid geometry) to the packaged Rust extension.

Grid shapes supported, in increasing generality:

* **Cubic** (the historical default) -- a single ``box_size`` half-edge and an
  equal node count and spacing on every axis.
* **Orthorhombic** -- either a length-3 ``box_size`` (per-axis half-extents), a
  length-3 ``grid_spacing`` (per-axis spacing, i.e. *varying resolution along each axis*),
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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from .._native import SpatialGrid
from ..units import DEFAULT_GRID_SPACING


@dataclass
class Grid:
    """A regular Cartesian grid centered at ``center``.

    The grid is specified by a physical **spacing** rather than a node count: the
    node count on each axis is derived so the realized spacing is as close as
    possible to ``grid_spacing`` (per axis).

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
    grid_spacing : float or array_like, shape (3,)
        Target grid spacing, in ``units`` (default ``0.20``).  A length-3 value
        sets a **different resolution along each axis**.
    units : {"angstrom", "bohr"}
        Unit of ``center``, ``box_size``, ``grid_spacing`` and ``cell`` (default
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

    center: Any
    box_size: Any = 5.0
    grid_spacing: Any = DEFAULT_GRID_SPACING
    units: str = "angstrom"
    cell: Any = None
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
    #: Build the grid as a **period** rather than a box: ``n`` nodes spanning the
    #: cell with no repeated endpoint, so ``shape @ step == cell`` exactly.  An
    #: FFT over this grid has the cell as its period, which is what a periodic
    #: Coulomb kernel, a lattice image sum and a spectral kinetic operator all
    #: assume.  The default (``False``) keeps the inclusive-endpoint box every
    #: molecular path uses.
    periodic: bool = False
    #: Round the periodic node count **up to a multiple** of these divisors, one
    #: per lattice vector.  A Born-von Karman supercell of an ``(n1, n2, n3)``
    #: mesh is ``n_i`` primitive cells long, and Bloch's theorem rests on the
    #: grid being invariant under a *primitive* translation -- which it is only
    #: when that translation is a whole number of grid steps.  With 15 nodes
    #: across a 2-cell supercell the primitive translation is 7.5 steps, the
    #: repeated atoms sample the grid differently, and the one-body matrix loses
    #: its translation symmetry: measured on a 2x2x1 square lattice, the four
    #: equivalent sites' on-site energies spread by 0.10 Ha and a degenerate
    #: band pair split by 0.087 Ha.  With 16 nodes both are zero to 1e-14.
    #: ``None`` (the default) leaves the node count alone.
    commensurate: tuple | None = None
    _native: SpatialGrid = field(init=False, repr=False)

    def __post_init__(self):
        grid = SpatialGrid.from_specs(
            center=self.center,
            spacing=self.grid_spacing,
            box_size=self.box_size,
            cell=self.cell,
            units=self.units,
            periodic=self.periodic,
            skew=self.skew,
            commensurate=self.commensurate,
        )
        self._native = grid
        shape = grid.shape
        spacings = grid.spacings
        self.nx, self.ny, self.nz = shape[0], shape[1], shape[2]
        self.dx, self.dy, self.dz = spacings[0], spacings[1], spacings[2]
        self.points = self.nx
        self.step = grid.step_bohr
        self.X, self.Y, self.Z = grid.coordinates()

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    @property
    def is_orthogonal(self) -> bool:
        """True when the voxel basis is diagonal (cubic or orthorhombic)."""
        return self._native.is_orthogonal()

    @property
    def is_cubic(self) -> bool:
        """True when every axis has the same node count and spacing (a cube)."""
        return self._native.is_cubic()

    @property
    def is_orthorhombic(self) -> bool:
        """True when the step vectors are mutually orthogonal (diagonal step)."""
        return self._native.is_orthorhombic()

    @property
    def size(self) -> int:
        return self._native.points

    @property
    def dV(self) -> float:
        """Voxel volume ``|det(step)|`` (``dx*dy*dz`` for an orthorhombic grid)."""
        return self._native.dv_bohr3

    def metric_inverse(self) -> np.ndarray:
        """Inverse metric ``(step^T step)^{-1}`` for the general FD Laplacian.

        The Laplacian in index coordinates is
        ``nabla^2 f = sum_{a,b} G^{-1}_{ab} d_a d_b f`` with ``G = step^T step``;
        for an orthorhombic grid this is ``diag(1/dx^2, 1/dy^2, 1/dz^2)`` and the
        cross terms vanish.
        """
        return self._native.metric_inverse()

    def flat_coords(self):
        """Contiguous flat ``(x, y, z)`` coordinate vectors for the native kernels."""
        return self._native.flat_coordinates()
