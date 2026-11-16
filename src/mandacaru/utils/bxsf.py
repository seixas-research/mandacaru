# -*- coding: utf-8 -*-
# file: utils/bxsf.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Band energies on a k-point grid: XCrySDen's ``.bxsf`` (Fermi surfaces).

A ``.bxsf`` file holds every band's energy on a regular grid that spans the
reciprocal cell; XCrySDen (and FermiSurfer) draw the Fermi surface as the
isosurface at the Fermi energy written in its header.  The physics -- the
non-self-consistent diagonalization on the mesh -- lives in
:meth:`~mandacaru.algorithms.dft.KohnSham.fermi_surface`.

Conventions
-----------
* The grid is a *general* grid: its first and last points both lie on the
  spanning vectors, so a periodic ``(n1, n2, n3)`` mesh is written as
  ``(n1 + 1, n2 + 1, n3 + 1)`` points, the last plane repeating the first.
* The spanning vectors are the reciprocal lattice vectors in
  :math:`\text{\AA}^{-1}` **including** the :math:`2\pi`, and the energies are
  in eV.
* Data order is :math:`k_1` outermost and :math:`k_3` innermost (C order),
  one band after another.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..resources import native_context


@dataclass
class BandGrid:
    """The content of a ``.bxsf`` file.

    ``energies`` is ``(n_bands, n1, n2, n3)`` on the periodic mesh (the
    repeated last planes of the file are dropped), in eV;
    ``reciprocal_cell`` holds the three reciprocal vectors as rows, in
    :math:`\\text{\\AA}^{-1}` with the :math:`2\\pi`.
    """

    energies: np.ndarray
    reciprocal_cell: np.ndarray
    fermi_level: float


def write_bxsf(path, energies, reciprocal_cell, fermi_level: float, comment: str='Mandacaru band energies', *, resources=None) -> str:
    """Write the band ``energies`` (eV, ``(n_bands, n1, n2, n3)`` on the
    periodic mesh ``k = (i1/n1, i2/n2, i3/n3)``) as a ``.bxsf`` file.

    ``reciprocal_cell`` is ``(3, 3)``, the reciprocal vectors as rows in
    :math:`\\text{\\AA}^{-1}` (with the :math:`2\\pi`); ``fermi_level`` in
    eV.  Returns the path written.
    """
    native_context(resources).write_bands(str(path), np.ascontiguousarray(energies, dtype=float), np.asarray(reciprocal_cell, dtype=float).reshape(3, 3).tolist(), float(fermi_level), comment=str(comment))
    return str(path)


def read_bxsf(path, *, resources=None) -> BandGrid:
    """Read a ``.bxsf`` file written by :func:`write_bxsf` (or any writer of
    the same single-block layout) back into a :class:`BandGrid`; the file is
    parsed in Rust (``volumetric::read_bxsf``)."""
    raw = native_context(resources).read_bands(str(path))
    return BandGrid(energies=raw["energies"],
                    reciprocal_cell=np.asarray(raw["reciprocal_cell"], dtype=float),
                    fermi_level=float(raw["fermi_level"]))
