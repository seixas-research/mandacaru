# -*- coding: utf-8 -*-
# file: basis/base.py

# This code is part of Carcará.
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
        """Evaluate on a :class:`~carcara.integrals.grid.Grid`, flattened.

        Returns a contiguous ``complex128`` vector of length ``grid.size`` ready
        to be handed (zero-copy) to the C backend.
        """
        values = self.evaluate(grid.X, grid.Y, grid.Z)
        return np.ascontiguousarray(np.broadcast_to(values, grid.shape),
                                    dtype=np.complex128).reshape(-1)
