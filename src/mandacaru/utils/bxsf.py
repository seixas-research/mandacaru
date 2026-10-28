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

from ..core.atomic import atomic_path

#: Values per line in a band block.
BXSF_VALUES_PER_LINE = 6


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
    from .._native import Context
    Context(**dict(resources or {})).write_bands(str(path), np.ascontiguousarray(energies, dtype=float), np.asarray(reciprocal_cell, dtype=float).reshape(3, 3).tolist(), float(fermi_level), comment=str(comment))
    return str(path)


def read_bxsf(path) -> BandGrid:
    """Read a ``.bxsf`` file written by :func:`write_bxsf` (or any writer of
    the same single-block layout) back into a :class:`BandGrid`."""
    with open(path, encoding="utf-8") as handle:
        tokens = [line.strip() for line in handle]
    fermi = None
    for line in tokens:
        if line.lower().startswith("fermi energy:"):
            fermi = float(line.split(":", 1)[1])
            break
    if fermi is None:
        raise ValueError(f"{path}: no 'Fermi Energy:' line in BEGIN_INFO")
    start = next(i for i, line in enumerate(tokens)
                 if line.upper().startswith("BEGIN_BANDGRID_3D"))
    n_bands = int(tokens[start + 1])
    shape = tuple(int(n) for n in tokens[start + 2].split())
    cell = np.array([[float(x) for x in tokens[start + 4 + i].split()]
                     for i in range(3)])
    count = int(np.prod(shape))
    bands = []
    line = start + 7
    for _band in range(n_bands):
        if not tokens[line].upper().startswith("BAND:"):
            raise ValueError(f"{path}: expected 'BAND:' at line {line + 1}")
        line += 1
        values: list[float] = []
        while len(values) < count:
            values.extend(float(x) for x in tokens[line].split())
            line += 1
        bands.append(np.asarray(values).reshape(shape))
    # Drop the closing planes: the periodic mesh is what was computed.
    energies = np.asarray(bands)[:, :-1, :-1, :-1]
    return BandGrid(energies=energies, reciprocal_cell=cell,
                    fermi_level=fermi)
