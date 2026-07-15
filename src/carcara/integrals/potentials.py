# -*- coding: utf-8 -*-
# file: integrals/potentials.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""External potentials sampled on the real-space grid.

:class:`~carcara.integrals.IntegralEngine.one_body` needs an external potential
as a plain callable ``V(x, y, z) -> real array``.  ``Potentials`` collects the
common analytic potentials and exposes them as such callables (bound methods),
so a molecular calculation reads

.. code-block:: python

    potentials = Potentials([(Z_Li, li_pos), (Z_H, h_pos)])
    T, V = engine.one_body(potentials.nuclear_potential)

The class holds only the geometry/charges; each method broadcasts over whatever
grid arrays ``IntegralEngine`` passes in, so the same instance works on any grid.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..units import to_bohr


class Potentials:
    r"""External potentials of a set of point nuclei.

    The nuclear positions are given in ``units`` (Angstrom by default) and are
    stored internally in Bohr.  :meth:`nuclear_potential` is a backend-facing
    callable: it is evaluated on the engine's Bohr grid and returns the
    potential in Hartree, exactly the atomic-unit contract
    :meth:`~carcara.integrals.IntegralEngine.one_body` expects.

    Parameters
    ----------
    nuclei : sequence of ``(Z, center)``
        One entry per nucleus: its charge ``Z`` and Cartesian ``center`` (shape
        ``(3,)``) in ``units``.  For a molecule these are the *true* nuclear
        charges (``Z_Li = 3``, ``Z_H = 1``, ...), independent of any effective
        (Slater) charges used for the basis orbitals.
    softening : float, optional
        Lower bound on ``|r - R_A|`` in Bohr, regularizing the ``r -> R_A``
        Coulomb singularity on the grid (default ``1e-12``).
    units : {"angstrom", "bohr"}
        Unit of the nuclear centers (default ``"angstrom"``).
    """

    def __init__(self, nuclei: Sequence[tuple[float, np.ndarray]],
                 softening: float = 1e-12, units: str = "angstrom"):
        self.nuclei = [(float(Z), to_bohr(center, units))
                       for Z, center in nuclei]
        self.softening = float(softening)

    def nuclear_potential(self, x, y, z) -> np.ndarray:
        r"""Electron-nuclear attraction :math:`V(\mathbf r) = -\sum_A Z_A/|\mathbf r - \mathbf R_A|`.

        Parameters
        ----------
        x, y, z : array_like
            Broadcastable Cartesian coordinates in Bohr (the grid arrays).

        Returns
        -------
        numpy.ndarray
            The real external potential (Hartree) sampled on the input coordinates.
        """
        v = np.zeros_like(x, dtype=float)
        for Z, (Rx, Ry, Rz) in self.nuclei:
            r = np.sqrt((x - Rx) ** 2 + (y - Ry) ** 2 + (z - Rz) ** 2)
            v -= Z / np.maximum(r, self.softening)
        return v
