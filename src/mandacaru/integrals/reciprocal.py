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


#: Largest departure from an isometry, :math:`\|R^TR - 1\|_F`, of a grid
#: operation :func:`lattice_operations` keeps.  Above the strains of a
#: finite-strain stress (1e-4 to 1e-3), so a strained crystal keeps the
#: unstrained grid's operations and its derivatives stay linear in the
#: strain.  A lattice within it of a higher symmetry is averaged over the
#: higher group, which is harmless: its own operations are a subgroup, and
#: every term is still an alias.  A strain that crosses it changes the set,
#: and the derivatives by the difference of two averages, which falls with
#: the grid step like the box's own aliasing.
ISOMETRY_TOLERANCE = 2e-2

#: :func:`symmetric_aliases` of the last few grids, keyed by shape and step.
_SYMMETRIC: dict = {}
_SYMMETRIC_KEEP = 4


def lattice_operations(grid) -> list:
    r"""The point operations of the grid's lattice, as integer matrices
    ``V`` acting on FFT frequencies (:math:`\mathbf m \to V\mathbf m`).

    Those of :math:`x \to Wx` (fractional) with :math:`W` unimodular, its
    Cartesian form :math:`R = AWA^{-1}` an isometry to
    :data:`ISOMETRY_TOLERANCE`, mapping nodes onto nodes (:math:`W_{ij}
    n_i/n_j` integral) and aliases onto aliases; :math:`V = W^{-T}`.  The
    lattice's, not a crystal's: a superset of every space group's point
    part on the grid, and never empty (the identity and the inversion).
    """
    import itertools

    shape = np.asarray(grid.shape, dtype=float)
    A = lattice_vectors(grid)
    inverse = np.linalg.inv(A)

    def kept(W):
        """Which of the ``(k, 3, 3)`` matrices ``W`` are kept."""
        R = np.einsum("ij,kjl,lm->kim", A, W, inverse)
        defect = np.einsum("kji,kjl->kil", R, R) - np.eye(3)
        isometry = np.sqrt(np.sum(defect * defect, axis=(1, 2))) \
            <= ISOMETRY_TOLERANCE
        nodes = W * shape[None, :, None] / shape[None, None, :]
        V = np.transpose(np.linalg.inv(W), (0, 2, 1))
        aliases = V * shape[None, None, :] / shape[None, :, None]
        integral = (np.all(np.abs(nodes - np.round(nodes)) < 1e-9, axis=(1, 2))
                    & np.all(np.abs(aliases - np.round(aliases)) < 1e-9,
                             axis=(1, 2)))
        return isometry & integral

    candidates = np.array(list(itertools.product((-1, 0, 1), repeat=9)),
                          dtype=float).reshape(-1, 3, 3)
    candidates = candidates[np.abs(np.abs(np.linalg.det(candidates)) - 1.0)
                            < 1e-9]
    group = list(candidates[kept(candidates)])
    # Close the set under products: an operation with entries beyond
    # {-1, 0, 1} in a skewed basis is a product of ones within it.
    known = {tuple(np.round(W).astype(int).ravel()) for W in group}
    while True:
        products = np.einsum("aij,bjk->abik", np.array(group),
                             np.array(group)).reshape(-1, 3, 3)
        new = {}
        for W in products[kept(products)]:
            key = tuple(np.round(W).astype(int).ravel())
            if key not in known:
                new[key] = W
        if not new:
            break
        known.update(new)
        group.extend(new.values())
    return [np.round(np.linalg.inv(W).T).astype(int) for W in group]


def symmetric_aliases(grid) -> tuple[np.ndarray, np.ndarray]:
    r"""``(vectors, norms)``: the FFT box's wave-vectors averaged over the
    grid's lattice operations.

    A grid of ``n_i`` nodes per axis cannot tell :math:`\mathbf G` from its
    aliases :math:`\mathbf G + \sum_i j_i n_i\mathbf b_i`; :func:`wavevectors`
    picks the alias in the box :math:`-n_i/2 \le m_i < n_i/2`, and a spectral
    derivative multiplies by it.  An operation that maps the grid onto
    itself maps the box onto itself only when it maps every axis onto an
    axis -- the hexagonal and fcc ones do not -- so the box derivative of a
    symmetric field is not exactly symmetric.  Its average over the
    operations :math:`V` of :func:`lattice_operations`,

    .. math::

        \tilde{\mathbf m} = \frac{1}{|\mathcal P|}\sum_{V\in\mathcal P}
            V^{-1}\,\mathrm{box}(V\mathbf m),

    commutes with every one of them; each term is an alias of
    :math:`\mathbf m`, so a frequency the operations keep inside the box is
    differentiated exactly as before, and only those near its faces get a
    mean of aliases (an even axis' Nyquist plane, through the inversion, the
    mean of :math:`\pm n_i/2`: zero, so a real field's derivative stays
    real).  The average is taken on the integers: it does not depend on the
    metric, so a strain that keeps the operations changes the derivative
    linearly, exactly as the box's -- what the finite-strain stress needs.
    A choice by length (the shortest alias, the Wigner-Seitz cell) is
    equally symmetric but switches aliases where a strain lifts a tie.

    ``vectors`` is :math:`B\tilde{\mathbf m}` (``(3, n1, n2, n3)``),
    ``norms`` the mean length of the averaged aliases (``(n1, n2, n3)``),
    for kernels of :math:`|\mathbf G|`.  On an orthorhombic grid only the
    Nyquist planes differ from :func:`wavevectors`.  Both are cached per
    grid and read-only.
    """
    shape = tuple(int(n) for n in grid.shape)
    key = (shape, np.asarray(grid.step, dtype=float).tobytes())
    cached = _SYMMETRIC.get(key)
    if cached is not None:
        return cached
    B = reciprocal_vectors(lattice_vectors(grid))
    n = np.asarray(shape, dtype=float)[:, None]
    grids = np.meshgrid(*[np.fft.fftfreq(k, d=1.0 / k) for k in shape],
                        indexing="ij")
    m = np.stack([g.ravel() for g in grids])
    box = np.sqrt(np.sum((B @ m) ** 2, axis=0))
    operations = lattice_operations(grid)
    if all(np.all(np.sum(np.abs(V), axis=1) == 1) for V in operations):
        # Signed permutations (an orthorhombic, tetragonal or cubic grid):
        # they keep the box but for an even axis' Nyquist plane, which the
        # inversion averages to zero.
        m = np.where(2 * np.abs(m) == n, 0.0, m)
        vectors = (B @ m).reshape(3, *shape)
        norms = box.reshape(shape)
    else:
        shift = np.zeros(m.shape)
        norms = len(operations) * box
        for V in operations:
            image = V @ m
            # box(V m) - V m: nonzero only where the image left the box.
            wrap = np.mod(image + n // 2, n) - n // 2 - image
            moved = np.any(wrap != 0.0, axis=0)
            back = np.rint(np.linalg.inv(V)) @ wrap[:, moved]
            shift[:, moved] += back
            norms[moved] += (np.sqrt(np.sum((B @ (m[:, moved] + back)) ** 2,
                                            axis=0)) - box[moved])
        count = len(operations)
        vectors = (B @ (m + shift / count)).reshape(3, *shape)
        norms = (norms / count).reshape(shape)
    vectors.setflags(write=False)
    norms.setflags(write=False)
    if len(_SYMMETRIC) >= _SYMMETRIC_KEEP:
        _SYMMETRIC.pop(next(iter(_SYMMETRIC)))
    _SYMMETRIC[key] = (vectors, norms)
    return vectors, norms


def symmetric_wavevectors(grid) -> np.ndarray:
    """``(3, n1, n2, n3)``: the vectors of :func:`symmetric_aliases`."""
    return symmetric_aliases(grid)[0]


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
