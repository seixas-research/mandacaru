# -*- coding: utf-8 -*-
# file: integrals/grid.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Regular Cartesian integration grid.

The Python front-end owns the grid: it defines the sampling points, spacing and
volume element, then hands the flat coordinate arrays to the C backend.  Keeping
the spacing **uniform** (one ``dx`` on every axis) lets the backend compute
finite-difference Laplacians by simple index arithmetic (see
``csrc/carcara_integrals.c``); the grid itself, however, need not be *cubic*.

Two shapes are supported:

* **Cubic** (the historical default) -- a single ``box_size`` half-edge and an
  equal number of nodes on every axis.
* **Non-cubic / orthorhombic** -- either a length-3 ``box_size`` (per-axis
  half-extents) or a full ``cell`` tensor (three lattice vectors, e.g. from an
  ASE ``Atoms.cell``).  The grid then spans the axis-aligned bounding box of the
  cell, still with uniform spacing, so an arbitrary (even skewed) lattice can be
  fed in without changing the integral core.  The node count differs per axis but
  ``dx`` stays common, which keeps the 7-point Laplacian valid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..units import to_bohr


@dataclass
class Grid:
    """A uniform-spacing Cartesian grid centered at ``center``.

    The grid is specified by a physical **spacing** rather than a node count:
    the number of nodes on each axis is derived so the realized spacing is as
    close as possible to ``h`` (and identical across axes).

    Lengths are given in ``units`` (**Angstrom** by default, the user-facing
    convention).  Internally the grid is built in atomic units: the sampled
    coordinate arrays :attr:`X`, :attr:`Y`, :attr:`Z` and the spacing :attr:`dx`
    are always in **Bohr**, which is what the integral backend consumes.

    Parameters
    ----------
    center : array_like, shape (3,)
        Cartesian center of the box, in ``units``.
    box_size : float or array_like, shape (3,)
        Half-edge length of the cube, in ``units`` (the box spans
        ``center ± box_size``).  A length-3 value gives a non-cubic
        (orthorhombic) box with per-axis half-extents.  Ignored when ``cell`` is
        given.
    h : float
        Target grid spacing, in ``units`` (default ``0.20`` Angstrom).
    units : {"angstrom", "bohr"}
        Unit of ``center``, ``box_size``, ``h`` and ``cell`` (default
        ``"angstrom"``).
    cell : array_like, shape (3, 3), optional
        Full lattice/cell tensor (rows are lattice vectors, ASE convention).
        When supplied, the grid spans the axis-aligned bounding box of the
        parallelepiped the three vectors define -- this is how arbitrary,
        possibly non-orthogonal, cells are handled.  Overrides ``box_size``.

    Attributes
    ----------
    points : int
        Number of nodes on the x-axis (== every axis when the grid is cubic).
        Kept for backward compatibility; use :attr:`shape` for the general case.
    shape : (int, int, int)
        Nodes per Cartesian axis ``(nx, ny, nz)``.
    dx : float
        Realized (uniform) spacing in Bohr; the closest achievable value to
        ``h`` (it may differ slightly).
    """

    center: np.ndarray
    box_size: float = 5.0
    h: float = 0.20
    units: str = "angstrom"
    cell: np.ndarray | None = None

    points: int = field(init=False)
    nx: int = field(init=False)
    ny: int = field(init=False)
    nz: int = field(init=False)
    X: np.ndarray = field(init=False, repr=False)
    Y: np.ndarray = field(init=False, repr=False)
    Z: np.ndarray = field(init=False, repr=False)
    dx: float = field(init=False)

    def __post_init__(self):
        # Convert the user-facing lengths to the atomic units the backend uses.
        center_bohr = np.asarray(to_bohr(self.center, self.units), dtype=float)
        h_bohr = float(to_bohr(self.h, self.units))

        box = None if self.cell is not None else np.asarray(self.box_size, float)
        if self.cell is None and box.ndim == 0:
            # -- Cubic path: preserve the historical node/spacing arithmetic
            #    exactly so existing integrals are byte-for-byte unchanged. --
            box_bohr = float(to_bohr(self.box_size, self.units))
            self.points = max(2, int(round(2.0 * box_bohr / h_bohr)) + 1)
            axis = np.linspace(-box_bohr, box_bohr, self.points)
            self.dx = float(axis[1] - axis[0])
            self.nx = self.ny = self.nz = self.points
            Xr, Yr, Zr = np.meshgrid(axis, axis, axis, indexing="ij")
        else:
            # -- Non-cubic path: uniform dx on every axis, per-axis node count. --
            if self.cell is not None:
                half = self._half_extents_from_cell()
            else:
                half = np.abs(np.asarray(to_bohr(box, self.units), dtype=float))
                if half.shape != (3,):
                    raise ValueError(
                        "box_size must be a scalar or a length-3 array of "
                        f"per-axis half-extents, got shape {half.shape}")
            self.dx = h_bohr
            # Keep dx exactly uniform: choose the node count nearest the target
            # spacing per axis, then realize each half-extent from that count.
            counts = [max(2, int(round(2.0 * L / h_bohr)) + 1) for L in half]
            self.nx, self.ny, self.nz = (int(counts[0]), int(counts[1]),
                                         int(counts[2]))
            axes = [np.linspace(-self.dx * (n - 1) / 2.0,
                                self.dx * (n - 1) / 2.0, n) for n in counts]
            self.points = self.nx  # x-axis count (== every axis when cubic)
            Xr, Yr, Zr = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")

        # Absolute Cartesian coordinates of every node (Bohr).
        self.X = Xr + center_bohr[0]
        self.Y = Yr + center_bohr[1]
        self.Z = Zr + center_bohr[2]

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

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    @property
    def is_cubic(self) -> bool:
        """True when every axis has the same number of nodes."""
        return self.nx == self.ny == self.nz

    @property
    def size(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def dV(self) -> float:
        """Volume element ``dx**3`` (spacing is uniform across axes)."""
        return self.dx ** 3

    def flat_coords(self):
        """Contiguous flat ``(x, y, z)`` coordinate vectors for the C backend."""
        return (np.ascontiguousarray(self.X.reshape(-1)),
                np.ascontiguousarray(self.Y.reshape(-1)),
                np.ascontiguousarray(self.Z.reshape(-1)))
