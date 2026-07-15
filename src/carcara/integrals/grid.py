# -*- coding: utf-8 -*-
# file: integrals/grid.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Regular Cartesian integration grid.

The Python front-end owns the grid: it defines the sampling points, spacing and
volume element, then hands the flat coordinate arrays to the C backend.  Keeping
the grid uniform and cubic lets the backend compute finite-difference
Laplacians by simple index arithmetic (see ``csrc/carcara_integrals.c``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..units import to_bohr


@dataclass
class Grid:
    """A uniform cubic grid centered at ``center`` with target spacing ``h``.

    The grid is specified by a physical **spacing** rather than a node count:
    the number of nodes per dimension is derived so the realized spacing is as
    close as possible to ``h``.

    Lengths are given in ``units`` (**Angstrom** by default, the user-facing
    convention).  Internally the grid is built in atomic units: the sampled
    coordinate arrays :attr:`X`, :attr:`Y`, :attr:`Z` and the spacing :attr:`dx`
    are always in **Bohr**, which is what the integral backend consumes.

    Parameters
    ----------
    center : array_like, shape (3,)
        Cartesian center of the box, in ``units``.
    box_size : float
        Half-edge length of the cube, in ``units`` (the box spans
        ``center ± box_size``).
    h : float
        Target grid spacing, in ``units`` (default ``0.20`` Angstrom).
    units : {"angstrom", "bohr"}
        Unit of ``center``, ``box_size`` and ``h`` (default ``"angstrom"``).

    Attributes
    ----------
    points : int
        Derived number of nodes per Cartesian dimension.
    dx : float
        Realized spacing in Bohr; the closest achievable value to ``h`` (it may
        differ slightly).
    """

    center: np.ndarray
    box_size: float = 5.0
    h: float = 0.20
    units: str = "angstrom"

    points: int = field(init=False)
    X: np.ndarray = field(init=False, repr=False)
    Y: np.ndarray = field(init=False, repr=False)
    Z: np.ndarray = field(init=False, repr=False)
    dx: float = field(init=False)

    def __post_init__(self):
        # Convert the user-facing lengths to the atomic units the backend uses.
        center_bohr = to_bohr(self.center, self.units)
        box_bohr = float(to_bohr(self.box_size, self.units))
        h_bohr = float(to_bohr(self.h, self.units))
        # Choose the node count that best matches the target spacing (Bohr).
        self.points = max(2, int(round(2.0 * box_bohr / h_bohr)) + 1)
        axis = np.linspace(-box_bohr, box_bohr, self.points)
        self.dx = float(axis[1] - axis[0])
        Xr, Yr, Zr = np.meshgrid(axis, axis, axis, indexing="ij")
        # Absolute Cartesian coordinates of every node (Bohr).
        self.X = Xr + center_bohr[0]
        self.Y = Yr + center_bohr[1]
        self.Z = Zr + center_bohr[2]

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.points, self.points, self.points)

    @property
    def size(self) -> int:
        return self.points ** 3

    @property
    def dV(self) -> float:
        """Volume element ``dx**3``."""
        return self.dx ** 3

    def flat_coords(self):
        """Contiguous flat ``(x, y, z)`` coordinate vectors for the C backend."""
        return (np.ascontiguousarray(self.X.reshape(-1)),
                np.ascontiguousarray(self.Y.reshape(-1)),
                np.ascontiguousarray(self.Z.reshape(-1)))
