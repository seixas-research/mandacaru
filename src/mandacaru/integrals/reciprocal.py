# -*- coding: utf-8 -*-
# file: integrals/reciprocal.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Reciprocal space of a real-space grid.

A grid of ``shape = (n1, n2, n3)`` nodes with step vectors ``a_i / n_i`` (the
columns of :attr:`~mandacaru.integrals.Grid.step`) is periodic in the cell
:math:`A = [\mathbf a_1, \mathbf a_2, \mathbf a_3]` it spans, whether that
cell is a crystal's or a molecular box.  Its discrete Fourier transform
represents exactly the wave-vectors

.. math::

    \mathbf G = m_1\mathbf b_1 + m_2\mathbf b_2 + m_3\mathbf b_3, \qquad
    \mathbf a_i\cdot\mathbf b_j = 2\pi\delta_{ij},

with :math:`m_i` the FFT frequencies of axis ``i``, and nothing above them.
This module fixes the conventions every Fourier-space quantity here shares:

.. math::

    f(\mathbf G) = \int_\Omega f(\mathbf r)\,e^{-i\mathbf G\cdot\mathbf r}d^3r
      = \Delta V\, e^{-i\mathbf G\cdot\mathbf r_0}\,\mathrm{FFT}[f]
    \qquad
    f(\mathbf r) = \frac1\Omega\sum_{\mathbf G} f(\mathbf G)
      e^{i\mathbf G\cdot\mathbf r},

where :math:`\mathbf r_0` is the grid's first node, so that atom-centered
quantities are placed by their structure factors
:math:`e^{-i\mathbf G\cdot\mathbf R_A}` in absolute coordinates.

Truncating an atom-centered charge to this set of :math:`\mathbf G` is the
**Fourier filter** of the grid: the charge is then sampled exactly, its
integrals stop depending on where its center falls between nodes, and what the
truncation leaves out is a property of the radial shape alone, recovered
analytically where it matters (the on-site tail of
:meth:`~mandacaru.pseudopotentials.periodic_paw.PeriodicPAW.dense_coulomb`).
"""

from __future__ import annotations

import numpy as np

from ..pseudopotentials.native_context import context as _native_ctx

ISOMETRY_TOLERANCE = 2e-2


def lattice_vectors(grid) -> np.ndarray:
    """The cell the grid spans, lattice vectors as **columns** (Bohr)."""
    return _native_ctx().periodic_lattice_vectors(
        tuple(int(n) for n in grid.shape), np.ascontiguousarray(grid.step, dtype=float))


def reciprocal_vectors(lattice_columns) -> np.ndarray:
    r""":math:`B = 2\pi A^{-T}`: reciprocal vectors as columns."""
    return _native_ctx().periodic_reciprocal_vectors(
        np.ascontiguousarray(lattice_columns, dtype=float))


def cell_volume(grid) -> float:
    """Volume of the spanned cell (Bohr^3)."""
    return _native_ctx().periodic_cell_volume(
        tuple(int(n) for n in grid.shape), np.ascontiguousarray(grid.step, dtype=float))


def grid_origin(grid) -> np.ndarray:
    """The first node of the grid (Bohr)."""
    return np.array([grid.X.flat[0], grid.Y.flat[0], grid.Z.flat[0]], dtype=float)


def wavevectors(grid) -> np.ndarray:
    r"""``(3, n1, n2, n3)`` Cartesian :math:`\mathbf G` in FFT index order."""
    return _native_ctx().periodic_wavevectors(
        tuple(int(n) for n in grid.shape), np.asarray(grid.step, dtype=float))


def lattice_operations(grid) -> list:
    r"""The point operations of the grid's lattice, as integer matrices ``V``."""
    return _native_ctx().periodic_lattice_operations(
        tuple(int(n) for n in grid.shape), grid_origin(grid), np.asarray(grid.step, dtype=float))


def symmetric_aliases(grid) -> tuple[np.ndarray, np.ndarray]:
    r"""``(vectors, norms)``: the FFT box's wave-vectors averaged over operations."""
    return _native_ctx().periodic_symmetric_aliases(
        tuple(int(n) for n in grid.shape), grid_origin(grid), np.asarray(grid.step, dtype=float))


def symmetric_wavevectors(grid) -> np.ndarray:
    """``(3, n1, n2, n3)``: the vectors of :func:`symmetric_aliases`."""
    return symmetric_aliases(grid)[0]


def bloch_aliases(grid, k) -> tuple[np.ndarray, np.ndarray]:
    r"""``(vectors, squares)``: wave-vectors :math:`\mathbf G + \mathbf k` averaged over operations."""
    return _native_ctx().periodic_bloch_aliases(
        tuple(int(n) for n in grid.shape), grid_origin(grid), np.asarray(grid.step, dtype=float),
        (float(k[0]), float(k[1]), float(k[2])))


def alias_table(grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""``(index, vectors, weights)`` of the grid alias table."""
    return _native_ctx().periodic_alias_table(
        tuple(int(n) for n in grid.shape), grid_origin(grid), np.asarray(grid.step, dtype=float))


def alias_combine(grid, box_values, alias_values) -> np.ndarray:
    r"""The mean over the lattice operations' aliases of a function."""
    index, _vectors, weights = alias_table(grid)
    box_arr = np.asarray(box_values)
    return _native_ctx().periodic_alias_combine(
        index, weights, box_arr, np.asarray(alias_values), list(box_arr.shape))


def alias_mean(grid, function) -> np.ndarray:
    r"""``function`` of the wave-vectors ``(3, ...)`` averaged over aliases."""
    _index, vectors, _weights = alias_table(grid)
    return alias_combine(grid, function(wavevectors(grid)), function(vectors))


def to_reciprocal(grid, values) -> np.ndarray:
    r"""``f(G)`` of flat real-space values, in the module's convention."""
    return _native_ctx().periodic_to_reciprocal(
        tuple(int(n) for n in grid.shape), grid_origin(grid), np.ascontiguousarray(grid.step, dtype=float),
        float(grid.dV), np.ascontiguousarray(values))


def to_real(grid, transform) -> np.ndarray:
    r"""Flat ``f(r)`` of ``f(G)`` (inverse of :func:`to_reciprocal`)."""
    return _native_ctx().periodic_to_real(
        tuple(int(n) for n in grid.shape), grid_origin(grid), np.ascontiguousarray(grid.step, dtype=float),
        np.ascontiguousarray(transform, dtype=complex))


def structure_factor(G, center) -> np.ndarray:
    r""":math:`e^{-i\mathbf G\cdot\mathbf R}` for one center (Bohr)."""
    return _native_ctx().periodic_structure_factor(
        np.asarray(G, dtype=float), (float(center[0]), float(center[1]), float(center[2])))


def spherical(G):
    """``(|G|, theta, phi)`` of a ``(3, ...)`` array of vectors."""
    return _native_ctx().periodic_spherical(np.asarray(G, dtype=float))


def radial_transform(r, values, L: int, q) -> np.ndarray:
    r""":math:`F_L(q) = \int_0^\infty f(r)\,j_L(qr)\,r^2\,dr` on a radial table."""
    return _native_ctx().periodic_radial_transform(
        np.ascontiguousarray(r, dtype=float), np.ascontiguousarray(values, dtype=float),
        int(L), np.ascontiguousarray(q, dtype=float))


def multipole_transform(G, center, radial, L: int, M: int) -> np.ndarray:
    r"""Fourier transform of :math:`f_L(r)Y_{LM}(\hat r)` centered at ``center``."""
    return _native_ctx().periodic_multipole_transform(
        np.ascontiguousarray(G, dtype=float), (float(center[0]), float(center[1]), float(center[2])),
        np.ascontiguousarray(radial, dtype=float), int(L), int(M))


def coulomb_kernel(G) -> np.ndarray:
    r""":math:`4\pi/G^2` with the :math:`\mathbf G = 0` term dropped."""
    return _native_ctx().periodic_coulomb_wavevector_kernel(np.asarray(G, dtype=float))


def lattice_translations(lattice_columns, radius: float) -> np.ndarray:
    """All lattice vectors ``R`` with ``|R| <= radius`` (rows, Bohr)."""
    return _native_ctx().periodic_lattice_translations(
        np.asarray(lattice_columns, dtype=float), float(radius))
