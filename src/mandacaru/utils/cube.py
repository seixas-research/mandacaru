# -*- coding: utf-8 -*-
# file: utils/cube.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Volumetric file formats: Gaussian ``.cube`` and XCrySDen ``.xsf``.

This module knows nothing about wavefunctions.  It takes a real scalar field
already sampled on Mandacaru's own :class:`~mandacaru.integrals.grid.Grid`
together with the nuclei, and writes it in a form VESTA, VMD, XCrySDen and
Avogadro open directly.  The physics lives in
:mod:`mandacaru.algorithms.volumetric`.

Why the cube writer is written here rather than taken from ASE
--------------------------------------------------------------
``ase.io.cube.write_cube`` derives the three voxel vectors from
``atoms.cell[i] / n_i``.  Mandacaru's grid is *not* the cell divided by the
node count: it is centered on the molecule, its node count is
``round(L / grid_spacing) + 1`` so that the realized spacing is exactly
``grid_spacing``, and it may
be skewed (:attr:`~mandacaru.integrals.grid.Grid.step` is a full 3x3 matrix of
step *vectors*, not a diagonal).  Handing ASE our data would therefore write a
header describing a different grid from the one the numbers were computed on,
and every atom would sit in the wrong place relative to the density -- the
classic cube-file bug.  The format itself is a dozen lines, so it is written
here, from the grid's own origin and step vectors.

The ``.xsf`` writer *does* go through ASE, because
:func:`ase.io.xsf.write_xsf` accepts the datagrid ``origin`` and
``span_vectors`` explicitly.

Conventions
-----------
* Everything in a cube file is in **Bohr** (a positive node count on the voxel
  lines is what declares that), and a density is in electrons per Bohr^3.  The
  coordinates and step vectors passed in are therefore Bohr as well -- which is
  what :class:`~mandacaru.integrals.grid.Grid` stores natively.
* Data order is ``x`` outermost, ``z`` innermost: exactly C order of the
  ``(nx, ny, nz)`` arrays ``numpy.meshgrid(..., indexing="ij")`` produces, which
  is how the grid samples every field.
* XSF is in **Angstrom**, and its datagrid is a *general* grid whose first and
  last point lie on the spanning vectors, so the span is ``(n - 1) * step``
  rather than ``n * step``.  A crystal's periodic field is closed by a copy
  of its first plane along each axis (``n + 1`` points) and spans the
  lattice vectors; its cube file holds each of the ``n`` nodes once.
"""

from __future__ import annotations

import os

import numpy as np


#: The second comment line.  ``ase.io.cube.read_cube`` parses this exact
#: sentence to learn the axis order, so it is written verbatim.
CUBE_AXIS_COMMENT = "OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z"

#: File extensions this module writes, mapped to the format name.
VOLUMETRIC_FORMATS = {".cube": "cube", ".cub": "cube", ".xsf": "xsf"}


def detect_format(path, format=None) -> str:
    """Resolve the output format from ``format`` or the file extension."""
    if format is not None:
        name = str(format).strip().lower().lstrip(".")
        if name not in set(VOLUMETRIC_FORMATS.values()):
            raise ValueError(
                f"unknown volumetric format {format!r}; use "
                f"{sorted(set(VOLUMETRIC_FORMATS.values()))}")
        return name
    suffix = os.path.splitext(str(path))[1].lower()
    if suffix not in VOLUMETRIC_FORMATS:
        raise ValueError(
            f"cannot tell the volumetric format from the file name {path!r}: "
            f"use one of the extensions {sorted(VOLUMETRIC_FORMATS)} or pass "
            f"format='cube' / 'xsf'")
    return VOLUMETRIC_FORMATS[suffix]


def _checked_field(data, origin, step, numbers, positions):
    """Validate and normalize the arguments shared by both writers."""
    data = np.asarray(data)
    if np.iscomplexobj(data):
        raise TypeError(
            "volumetric files hold real numbers; convert the field first "
            "(real part, imaginary part or modulus) so the choice is explicit")
    data = np.ascontiguousarray(data, dtype=float)
    if data.ndim != 3:
        raise ValueError(f"expected a 3-dimensional field, got shape {data.shape}")
    origin = np.asarray(origin, dtype=float).reshape(3)
    step = np.asarray(step, dtype=float)
    if step.shape != (3, 3):
        raise ValueError(f"step must be a (3, 3) matrix of step vectors "
                         f"(columns), got shape {step.shape}")
    numbers = np.asarray(numbers, dtype=int).reshape(-1)
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    if len(numbers) != len(positions):
        raise ValueError(f"{len(numbers)} atomic numbers but "
                         f"{len(positions)} positions")
    return data, origin, step, numbers, positions


def write_cube(path, data, origin, step, numbers, positions, comment: str='', charges=None, *, resources=None) -> str:
    """Write a Gaussian ``.cube`` file and return the path written.

    Parameters
    ----------
    path : str or Path
        Destination.  Written atomically (a temporary file in the same
        directory, then :func:`os.replace`), so a failed write never truncates
        a previous snapshot.
    data : (nx, ny, nz) array
        The real scalar field, in the grid's own node order.
    origin : (3,) array
        Position of node ``(0, 0, 0)`` in **Bohr**.
    step : (3, 3) array
        Grid step vectors in **Bohr** as *columns*, matching
        :attr:`mandacaru.integrals.grid.Grid.step`: node ``(i, j, k)`` sits at
        ``origin + step @ (i, j, k)``.
    numbers : sequence of int
        Atomic numbers, one per nucleus (the element the viewer will draw).
    positions : (natoms, 3) array
        Nuclear positions in **Bohr**, in the same frame as ``origin``.
    comment : str
        First line of the file.  The second line is fixed
        (:data:`CUBE_AXIS_COMMENT`) because ASE's reader parses it.
    charges : sequence of float, optional
        The per-atom charge column.  Defaults to the atomic numbers; a
        pseudopotential run passes the *valence* charge, which is what the
        density in the file actually integrates to.
    """
    data, origin, step, numbers, positions = _checked_field(data, origin, step, numbers, positions)
    charges = np.asarray(numbers, dtype=float) if charges is None else np.asarray(charges, dtype=float).reshape(-1)
    if len(charges) != len(numbers):
        raise ValueError(f'{len(charges)} charges but {len(numbers)} atoms')
    from .._native import Context
    Context(**dict(resources or {})).write_field(str(path), data, origin.tolist(), step.tolist(), numbers.tolist(), positions.tolist(), format='cube', charges=charges.tolist(), comment=str(comment))
    return str(path)


def write_xsf(path, data, origin, step, numbers, positions, cell=None, periodic: bool=False, *, resources=None) -> str:
    """Write an XCrySDen ``.xsf`` file (VESTA reads these too) via ASE.

    Same arguments as :func:`write_cube`, all in **Bohr**; XSF is an Angstrom
    format, so everything is converted here.  ``cell`` (Bohr, the lattice
    vectors as **rows**) sets the periodic lattice; ``None`` writes a
    molecule.

    ``periodic=True`` declares ``data`` a periodic field over ``cell``
    (``n`` nodes per lattice vector, node ``n`` being node ``0`` of the next
    cell): the general datagrid then gets the closing copy of its first
    plane along each axis, ``n + 1`` points, and spans the lattice vectors
    themselves -- XCrySDen's convention for a crystal's grid.

    ASE formats the data with ``%f`` -- six *decimal* places, not six
    significant digits -- so values below about ``1e-6`` are written as zero.
    That is ample for a density in e/Bohr^3 and for an orbital amplitude, but
    :func:`write_cube` is the more precise of the two.
    """
    data, origin, step, numbers, positions = _checked_field(data, origin, step, numbers, positions)
    from .._native import Context
    Context(**dict(resources or {})).write_field(str(path), data, origin.tolist(), step.tolist(), numbers.tolist(), positions.tolist(), format='xsf', periodic=periodic, cell=None if cell is None else np.asarray(cell, dtype=float).reshape(3, 3).tolist())
    return str(path)


def write_volumetric(path, data, origin, step, numbers, positions, comment: str='', charges=None, cell=None, format=None, periodic: bool=False, *, resources=None) -> str:
    """Write ``data`` in the format named by ``format`` or the file extension.

    ``cell`` and ``periodic`` go to :func:`write_xsf`; a periodic field's
    cube file holds each node once (the cell is ``n`` steps)."""
    name = detect_format(path, format)
    if name == 'cube':
        return write_cube(path, data, origin, step, numbers, positions, comment=comment, charges=charges, resources=resources)
    return write_xsf(path, data, origin, step, numbers, positions, cell=cell, periodic=periodic, resources=resources)
