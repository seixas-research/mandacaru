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


@dataclass
class Grid:
    """A uniform cubic grid of ``points**3`` nodes centered at ``center``.

    Parameters
    ----------
    center : array_like, shape (3,)
        Cartesian center of the box in Bohr.
    box_size : float
        Half-edge length of the cube in Bohr (the box spans ``center ± box``).
    points : int
        Number of nodes per Cartesian dimension.
    """

    center: np.ndarray
    box_size: float = 10.0
    points: int = 60

    X: np.ndarray = field(init=False, repr=False)
    Y: np.ndarray = field(init=False, repr=False)
    Z: np.ndarray = field(init=False, repr=False)
    dx: float = field(init=False)

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        axis = np.linspace(-self.box_size, self.box_size, self.points)
        self.dx = float(axis[1] - axis[0])
        Xr, Yr, Zr = np.meshgrid(axis, axis, axis, indexing="ij")
        # Absolute Cartesian coordinates of every node.
        self.X = Xr + self.center[0]
        self.Y = Yr + self.center[1]
        self.Z = Zr + self.center[2]

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
