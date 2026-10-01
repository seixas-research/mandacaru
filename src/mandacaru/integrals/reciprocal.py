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


def lattice_vectors(grid) -> np.ndarray:
    """The cell the grid spans, lattice vectors as **columns** (Bohr)."""
    return np.asarray(grid.step, dtype=float) @ np.diag(grid.shape)


def reciprocal_vectors(lattice_columns) -> np.ndarray:
    r""":math:`B = 2\pi A^{-T}`: reciprocal vectors as columns."""
    return 2.0 * np.pi * np.linalg.inv(np.asarray(lattice_columns,
                                                  dtype=float)).T


def cell_volume(grid) -> float:
    """Volume of the spanned cell (Bohr^3)."""
    return float(abs(np.linalg.det(lattice_vectors(grid))))


def grid_origin(grid) -> np.ndarray:
    """The first node of the grid (Bohr)."""
    return np.array([grid.X.flat[0], grid.Y.flat[0], grid.Z.flat[0]],
                    dtype=float)


def wavevectors(grid) -> np.ndarray:
    r"""``(3, n1, n2, n3)`` Cartesian :math:`\mathbf G` in FFT index order."""
    B = reciprocal_vectors(lattice_vectors(grid))
    m = [np.fft.fftfreq(n, d=1.0 / n) for n in grid.shape]
    m1, m2, m3 = np.meshgrid(*m, indexing="ij")
    return np.einsum("ci,ixyz->cxyz", B, np.stack([m1, m2, m3]))


def to_reciprocal(grid, values) -> np.ndarray:
    r"""``f(G)`` of flat real-space values, in the module's convention."""
    G = wavevectors(grid)
    phase = np.exp(-1j * np.einsum("c,cxyz->xyz", grid_origin(grid), G))
    return (grid.dV * phase
            * np.fft.fftn(np.asarray(values).reshape(grid.shape)))


def to_real(grid, transform) -> np.ndarray:
    r"""Flat ``f(r)`` of ``f(G)`` (inverse of :func:`to_reciprocal`)."""
    G = wavevectors(grid)
    phase = np.exp(1j * np.einsum("c,cxyz->xyz", grid_origin(grid), G))
    N = int(np.prod(grid.shape))
    return (np.fft.ifftn(transform * phase) * N / cell_volume(grid)).reshape(-1)


def structure_factor(G, center) -> np.ndarray:
    r""":math:`e^{-i\mathbf G\cdot\mathbf R}` for one center (Bohr)."""
    return np.exp(-1j * np.einsum("c,c...->...", np.asarray(center, float), G))


def spherical(G):
    """``(|G|, theta, phi)`` of a ``(3, ...)`` array of vectors."""
    norm = np.sqrt(np.sum(G * G, axis=0))
    safe = np.where(norm > 0.0, norm, 1.0)
    theta = np.arccos(np.clip(G[2] / safe, -1.0, 1.0))
    phi = np.arctan2(G[1], G[0])
    return norm, theta, phi


def radial_transform(r, values, L: int, q) -> np.ndarray:
    r""":math:`F_L(q) = \int_0^\infty f(r)\,j_L(qr)\,r^2\,dr` on a radial table."""
    from scipy.special import spherical_jn

    r = np.asarray(r, dtype=float)
    q = np.asarray(q, dtype=float)
    flat = q.reshape(-1)
    out = np.empty(flat.size)
    weights = np.asarray(values, dtype=float) * r * r
    for start in range(0, flat.size, 2048):
        chunk = flat[start:start + 2048]
        out[start:start + chunk.size] = np.trapezoid(
            spherical_jn(int(L), np.outer(chunk, r)) * weights[None, :], r,
            axis=1)
    return out.reshape(q.shape)


def multipole_transform(G, center, radial, L: int, M: int) -> np.ndarray:
    r"""Fourier transform of :math:`f_L(r)Y_{LM}(\hat r)` centered at ``center``.

    :math:`4\pi(-i)^L Y_{LM}(\hat G)\,F_L(|G|)\,e^{-i\mathbf G\cdot\mathbf R}`,
    from the plane-wave expansion of :math:`e^{-i\mathbf G\cdot\mathbf r}`.
    ``radial`` is ``F_L`` already evaluated on ``|G|``.
    """
    from ..basis._angular import spherical_harmonic

    _norm, theta, phi = spherical(G)
    return (4.0 * np.pi * (-1j) ** int(L)
            * spherical_harmonic(int(L), int(M), theta, phi)
            * radial * structure_factor(G, center))


def coulomb_kernel(G) -> np.ndarray:
    r""":math:`4\pi/G^2` with the :math:`\mathbf G = 0` term dropped."""
    G2 = np.sum(G * G, axis=0)
    return np.where(G2 > 0.0, 4.0 * np.pi / np.where(G2 > 0.0, G2, 1.0), 0.0)


def lattice_translations(lattice_columns, radius: float) -> np.ndarray:
    """All lattice vectors ``R`` with ``|R| <= radius`` (rows, Bohr)."""
    A = np.asarray(lattice_columns, dtype=float)
    B = reciprocal_vectors(A)
    # |n_i| <= radius |b_i| / 2 pi bounds every vector inside the sphere.
    reach = [int(np.ceil(radius * np.linalg.norm(B[:, i]) / (2.0 * np.pi)))
             for i in range(3)]
    n = [np.arange(-r, r + 1) for r in reach]
    n1, n2, n3 = np.meshgrid(*n, indexing="ij")
    integers = np.stack([n1.ravel(), n2.ravel(), n3.ravel()], axis=1)
    R = integers @ A.T
    keep = np.linalg.norm(R, axis=1) <= radius + 1e-12
    return R[keep]
