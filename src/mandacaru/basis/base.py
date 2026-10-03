# -*- coding: utf-8 -*-
# file: basis/base.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Basis-function abstraction.

The whole integral machinery is *agnostic* to the kind of localized basis
function used.  It never manipulates analytic expressions: it only needs to
**sample** a function on a Cartesian grid.  Consequently any localized basis --
hydrogen-like orbitals today, Wannier functions or numerical atomic orbitals
tomorrow -- is injected simply by implementing :meth:`BasisFunction.evaluate`.

This is the single contract the C backend relies on (it receives the sampled
values), so new bases require *zero* changes to the integral core.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

#: Fraction of its peak below which a tabulated radial function counts as
#: zero when its support is measured: far below anything a grid resolves, so
#: sampling only inside the support changes nothing measurable.
SUPPORT_FLOOR = 1e-14

#: Bohr added past the last table value above :data:`SUPPORT_FLOOR`.
SUPPORT_MARGIN = 1.0


class BasisFunction(ABC):
    """A localized single-particle function :math:`\\phi(\\mathbf r)`.

    Subclasses only have to know how to evaluate themselves on a set of
    Cartesian coordinates.  Everything else (grids, integrals, parallel C
    kernels) is shared and basis-independent.
    """

    #: Cartesian center of the function in Bohr, shape ``(3,)``.
    center: np.ndarray

    @abstractmethod
    def evaluate(self, x, y, z) -> np.ndarray:
        """Sample the (possibly complex) function on Cartesian coordinates.

        Parameters
        ----------
        x, y, z : array_like
            Broadcastable arrays of Cartesian coordinates in Bohr.

        Returns
        -------
        numpy.ndarray
            Complex128 array of values, broadcast to the shape of the inputs.
        """
        raise NotImplementedError

    def sample(self, grid) -> np.ndarray:
        """Evaluate on a :class:`~mandacaru.integrals.grid.Grid`, flattened.

        Returns a contiguous ``complex128`` vector of length ``grid.size`` ready
        to be handed (zero-copy) to the C backend.
        """
        radius = self.support_radius
        if radius is None:
            values = self.evaluate(grid.X, grid.Y, grid.Z)
            return np.ascontiguousarray(np.broadcast_to(values, grid.shape),
                                        dtype=np.complex128).reshape(-1)
        # A confined function is exactly zero beyond its radius: evaluate the
        # radial spline and the harmonic only inside, so sampling a basis
        # costs its functions' volumes rather than M whole grids (the largest
        # setup cost of a long molecule).  Bit-identical to the full
        # evaluation.
        center = np.asarray(self.center, dtype=float)
        X, Y, Z = (grid.X.reshape(-1), grid.Y.reshape(-1),
                   grid.Z.reshape(-1))
        dx, dy, dz = X - center[0], Y - center[1], Z - center[2]
        inside = np.flatnonzero(dx * dx + dy * dy + dz * dz
                                <= radius * radius)
        out = np.zeros(grid.size, dtype=np.complex128)
        if inside.size:
            out[inside] = self.evaluate(X[inside], Y[inside], Z[inside])
        return out

    @property
    def support_radius(self) -> float | None:
        """Radius (Bohr) beyond which the function is exactly zero, or
        ``None`` when it never is.

        Confined radial functions carry it in one of two forms: a tabulated
        radial table (``_r``/``_values``, or a multi-zeta ``table`` with
        ``r``/``values``), whose last point above
        :data:`SUPPORT_FLOOR` of the peak is the support -- the table itself
        often runs further, out to 30 Bohr --
        or a confinement radius (``r_c``).  Anything else (Gaussians,
        hydrogenic orbitals) has unbounded support and is sampled on the
        whole grid.
        """
        cached = self.__dict__.get("_support_radius", False)
        if cached is not False:
            return cached
        radius = None
        table, grid_r = getattr(self, "_values", None), getattr(self, "_r",
                                                                None)
        tabulated = getattr(self, "table", None)        # a multi-zeta table
        if table is None and tabulated is not None:
            table = getattr(tabulated, "values", None)
            grid_r = getattr(tabulated, "r", None)
        if table is not None and grid_r is not None:
            # A Fourier-filtered function rings out to the end of its table
            # at round-off level; below SUPPORT_FLOOR of its peak it is
            # treated as zero (C, H PAW-LCAO-SZ: 7.8-9.4 Bohr instead of
            # the 30 Bohr table).
            values = np.abs(np.asarray(table, dtype=float))
            nonzero = np.flatnonzero(values > SUPPORT_FLOOR * values.max())
            r = np.asarray(grid_r, dtype=float)
            # The spline reaches to the next table point beyond the last
            # nonzero value; zero only from there on.
            # A cubic spline is not local: just past the last table value
            # above the floor it still carries ~1e-10 of overshoot, so the
            # support ends SUPPORT_MARGIN further out.
            radius = (min(float(r[min(nonzero[-1] + 1, r.size - 1)])
                          + SUPPORT_MARGIN, float(r[-1]))
                      if nonzero.size else 0.0)
        elif getattr(self, "r_c", None) is not None:
            radius = float(self.r_c)
        self.__dict__["_support_radius"] = radius
        return radius
