# -*- coding: utf-8 -*-
# file: algorithms/tetrahedron.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Brillouin-zone integration by the linear tetrahedron method.

Pure numerics, with no dependence on the rest of Mandacaru: band energies on
a uniform :math:`n_1\times n_2\times n_3` k-mesh in, densities of states and
integration weights out.  The density of states of a crystal is
:math:`g(E) = \sum_n\int_{BZ}\delta(E-\varepsilon_{n\mathbf k})\,d\mathbf k/V_{BZ}`;
a Gaussian broadening of the mesh's levels replaces each delta by a peak
whose width has to beat the mesh spacing.  The tetrahedron method instead
interpolates every band **linearly** inside tetrahedra spanned by mesh points
and integrates the interpolant exactly (Jepsen and Andersen 1971; Lehmann
and Taut 1972), so the spectrum has no width parameter, band edges and van
Hove singularities stay sharp, and a gap is exactly empty.

**The decomposition** (Bloechl, Jepsen and Andersen 1994).  Every subcell
of the mesh -- the parallelepiped spanned by :math:`\mathbf b_a/n_a` at a
mesh point -- is cut into six tetrahedra of equal volume sharing the
shortest of its four main diagonals, so that the tetrahedra are as compact
as the lattice allows.  Corner indices wrap periodically, so the mesh may be
Gamma-centered or shifted alike.  Each tetrahedron carries the fraction
:math:`1/(6N_k)` of the zone.

**The weights.**  For one band in one tetrahedron with sorted corner
energies :math:`e_1\le e_2\le e_3\le e_4`, the fraction of the tetrahedron
below :math:`E` and its share :math:`w_i(E)` on each corner -- the integral
of the barycentric coordinate :math:`\lambda_i` over the occupied part --
have closed forms in the three regimes :math:`e_1<E<e_2`,
:math:`e_2<E<e_3` and :math:`e_3<E<e_4` (Bloechl et al., appendix).  Their
energy derivatives :math:`dw_i/dE` are the corner DOS weights, so a
projected density of states

.. math::

    g_\mu(E) = \sum_{n\mathbf k}\frac{dw_{n\mathbf k}}{dE}(E)\,
               p^\mu_{n\mathbf k}

is the exact integral of the linearly interpolated *product*, not the
tetrahedron DOS times an averaged projection.  The four corner weights sum to
the tetrahedron's own count and DOS.

**The curvature correction** (Bloechl et al., Eq. 22).  Linear interpolation
misses the curvature of a band; to leading order the error of a
corner-weighted integral is removed by

.. math::

    \delta w_i(E) = \frac{D_T(E)}{40}\sum_{j=1}^4 (e_j - e_i),

:math:`D_T` the tetrahedron's DOS.  The correction sums to zero over the
corners, so it leaves the number of states and the total DOS unchanged; it
moves weight between corners and so changes integrals of a k-dependent
quantity (a projection, a band energy).  Its DOS counterpart is the energy
derivative, :math:`D_T'(E)/40\sum_j(e_j-e_i)`, which integrates to zero
over energy: the corrected projected DOS keeps every normalization, but it
is not positive everywhere (it dips below zero where a projection is small
at a band extremum on a mesh point, as the corrected weights do), and at a
band onset its error does not shrink as the mesh is refined.  It is
derived for, and serves, integrated quantities -- the Fermi-sea sums of a
metal at a fixed count, 30 to 1000 times closer with it -- so the
calculator's ``pdos`` leaves it off by default.

**Degenerate corners.**  A regime is entered only on its open interval, so
equal corner energies never divide by zero: an empty regime is skipped, and
every formula is written in ratios bounded by one.  Two cases have no width
at all: four equal corners hold a delta of the whole tetrahedron, and with
three equal corners the DOS steps, so the corrected weights step and their
derivative holds a delta.  A pointwise DOS cannot show a delta; the
cell-averaged one (``cell_average=True``, the difference of the exact counts
across each energy's cell of the axis) keeps both, and conserves the count
on any axis -- the form the calculator reports.

Energies, ``eigenvalues`` and the result share one unit; the densities are
per k-point weight :math:`1/N_k`, so each band integrates to one (one state
per band per k-point, one spin).
"""

from __future__ import annotations

from itertools import permutations

import numpy as np

#: Elements per working array: the energy and band chunking keeps each of
#: ``(energies, pairs, 4)`` and ``(pairs * 4, projections)`` below this, so
#: memory does not grow with ``N_tetrahedra * N_bands * N_energies``.
CHUNK_ELEMENTS = 1 << 20

#: The six tetrahedra of the unit subcell sharing its (0,0,0)-(1,1,1)
#: diagonal: the monotone paths 0 -> e_a -> e_a + e_b -> (1,1,1).
_UNIT_TETRAHEDRA = np.array(
    [[(0, 0, 0),
      tuple(int(axis == p[0]) for axis in range(3)),
      tuple(int(axis in p[:2]) for axis in range(3)),
      (1, 1, 1)] for p in permutations(range(3))], dtype=int)

#: Axes flipped to turn the (0,0,0)-(1,1,1) diagonal into each of the four.
_DIAGONAL_FLIPS = ((), (0,), (1,), (2,))


def mesh_tetrahedra(size, reciprocal=None) -> np.ndarray:
    """Corner indices of the ``6 n1 n2 n3`` tetrahedra of a uniform mesh.

    ``size = (n1, n2, n3)``; the mesh point ``(i1, i2, i3)`` has the flat
    C-order index ``(i1 n2 + i2) n3 + i3``, whatever the mesh's offset.  Each
    subcell is cut along its shortest main diagonal in Cartesian space
    (``reciprocal``: the reciprocal lattice vectors as rows, any common scale)
    -- along (0,0,0)-(1,1,1) without one.  Returns ``(6 N_k, 4)`` integers.
    """
    size = tuple(int(n) for n in size)
    if len(size) != 3 or min(size) < 1:
        raise ValueError(f"size must be three positive integers; got {size!r}")
    flips = ()
    if reciprocal is not None:
        steps = np.asarray(reciprocal, dtype=float) / np.asarray(size)[:, None]
        lengths = []
        for flipped in _DIAGONAL_FLIPS:
            signs = np.array([-1.0 if a in flipped else 1.0 for a in range(3)])
            lengths.append(np.linalg.norm(signs @ steps))
        # The first of equally short diagonals: a stable choice.
        flips = _DIAGONAL_FLIPS[int(np.argmin(np.round(lengths, 12)))]
    corners = _UNIT_TETRAHEDRA.copy()
    for axis in flips:
        corners[..., axis] = 1 - corners[..., axis]
    origins = np.indices(size).reshape(3, -1).T            # (N_k, 3)
    points = (origins[:, None, None, :] + corners[None]) % np.asarray(size)
    flat = np.ravel_multi_index(points.reshape(-1, 3).T, size)
    return flat.reshape(-1, 4)


# --------------------------------------------------------------------------- #
# One tetrahedron.
# --------------------------------------------------------------------------- #

def _regime_masks(E, e):
    """The three open regimes of sorted corners ``e`` (..., 4) at ``E``."""
    e1, e2, e3, e4 = (e[..., i] for i in range(4))
    one = (E > e1) & (E < e2)
    two = (E >= e2) & (E < e3) & (E > e1)
    three = (E >= e3) & (E < e4)
    return one, two, three, E >= e4


def _safe(x):
    """``x`` with zeros replaced by one: a denominator used only where the
    regime mask guarantees it is positive."""
    return np.where(x > 0.0, x, 1.0)


def _sorted_weights(E, e, correction: bool):
    r"""``(w, dw, n, g)`` at energies ``E`` for sorted corners ``e``.

    ``E`` broadcasts against ``e[..., 0]`` (e.g. ``(nE, 1)`` with ``e`` of
    shape ``(P, 4)``); ``w`` and ``dw`` have a trailing corner axis in the
    sorted order, ``n`` and ``g`` are the tetrahedron's fraction below ``E``
    and its DOS (the sums of ``w`` and ``dw`` over the corners).  All are
    fractions of the tetrahedron: ``w`` tends to 1/4 per corner.
    """
    E = np.asarray(E, dtype=float)
    e1, e2, e3, e4 = (e[..., i] for i in range(4))
    one, two, three, full = _regime_masks(E, e)
    shape = np.broadcast_shapes(E.shape, e1.shape)
    w = np.zeros(shape + (4,))
    dw = np.zeros(shape + (4,))
    dg = np.zeros(shape)                      # dD_T/dE, for the correction
    e21, e31, e41 = _safe(e2 - e1), _safe(e3 - e1), _safe(e4 - e1)
    e32, e42, e43 = _safe(e3 - e2), _safe(e4 - e2), _safe(e4 - e3)

    # e1 < E < e2: r_j = (E - e1)/e_j1 in [0, 1].
    a = E - e1
    r2, r3, r4 = a / e21, a / e31, a / e41
    C = r2 * r3 * r4
    q = r2 * r3 / e41                          # C / (E - e1) = g / 3
    reg = [C * (4.0 - r2 - r3 - r4) / 4.0, C * r2 / 4.0, C * r3 / 4.0,
           C * r4 / 4.0]
    dreg = [q * (3.0 - r2 - r3 - r4), q * r2, q * r3, q * r4]
    for i in range(4):
        w[..., i] = np.where(one, reg[i], w[..., i])
        dw[..., i] = np.where(one, dreg[i], dw[..., i])
    dg = np.where(one, 6.0 * r2 / (e31 * e41), dg)

    # e2 < E < e3 (Bloechl et al., appendix): every ratio below is <= 1.
    b, c, d = E - e2, e3 - E, e4 - E
    C1 = (a / e41) * (a / e31)
    C2 = (a / e41) * (b / e32) * (c / e31)
    C3 = (b / e42) * (b / e32) * (d / e41)
    dC1 = 2.0 * a / (e41 * e31)
    dC2 = (b * c + a * c - a * b) / (e41 * e32 * e31)
    dC3 = (2.0 * b * d - b * b) / (e42 * e32 * e41)
    S12, S23, S = C1 + C2, C2 + C3, C1 + C2 + C3
    dS12, dS23, dS = dC1 + dC2, dC2 + dC3, dC1 + dC2 + dC3
    reg = [C1 / 4 + S12 / 4 * c / e31 + S / 4 * d / e41,
           S / 4 + S23 / 4 * c / e32 + C3 / 4 * d / e42,
           S12 / 4 * a / e31 + S23 / 4 * b / e32,
           S / 4 * a / e41 + C3 / 4 * b / e42]
    dreg = [dC1 / 4 + dS12 / 4 * c / e31 - S12 / (4 * e31)
            + dS / 4 * d / e41 - S / (4 * e41),
            dS / 4 + dS23 / 4 * c / e32 - S23 / (4 * e32)
            + dC3 / 4 * d / e42 - C3 / (4 * e42),
            dS12 / 4 * a / e31 + S12 / (4 * e31)
            + dS23 / 4 * b / e32 + S23 / (4 * e32),
            dS / 4 * a / e41 + S / (4 * e41)
            + dC3 / 4 * b / e42 + C3 / (4 * e42)]
    for i in range(4):
        w[..., i] = np.where(two, reg[i], w[..., i])
        dw[..., i] = np.where(two, dreg[i], dw[..., i])
    slope = (6.0 - 6.0 * (b / e32) * ((e3 - e1) + (e4 - e2)) / e42) \
        / (e31 * e41)
    dg = np.where(two, slope, dg)

    # e3 < E < e4: s_j = (e4 - E)/e_4j in [0, 1].
    s1, s2, s3 = d / e41, d / e42, d / e43
    C = s1 * s2 * s3
    q = s2 * s3 / e41                          # C / (e4 - E) = g / 3
    reg = [0.25 - C * s1 / 4.0, 0.25 - C * s2 / 4.0, 0.25 - C * s3 / 4.0,
           0.25 - C * (4.0 - s1 - s2 - s3) / 4.0]
    dreg = [q * s1, q * s2, q * s3, q * (3.0 - s1 - s2 - s3)]
    for i in range(4):
        w[..., i] = np.where(three, reg[i], w[..., i])
        dw[..., i] = np.where(three, dreg[i], dw[..., i])
    dg = np.where(three, -6.0 * s3 / (e41 * e42), dg)

    w = np.where(full[..., None], 0.25, w)
    n = w.sum(axis=-1)
    g = dw.sum(axis=-1)
    if correction:
        # sum_j (e_j - e_i) = sum_j e_j - 4 e_i, zero over the corners.
        spread = (e.sum(axis=-1, keepdims=True) - 4.0 * e) / 40.0
        w = w + g[..., None] * spread
        dw = dw + dg[..., None] * spread
    return w, dw, n, g


def corner_weights(energies, corners, curvature_correction: bool = True):
    """Integration and DOS weights of each corner of each tetrahedron.

    ``energies`` ``(nE,)``; ``corners`` ``(P, 4)`` corner energies of ``P``
    tetrahedra (one band each), in any order.  Returns ``(w, dw)``, each
    ``(nE, P, 4)`` in the corners' own order: ``w[e, p, i]`` is corner
    ``i``'s share of the fraction of tetrahedron ``p`` below ``energies[e]``
    (the four sum to that fraction, 0 below the lowest corner and 1 above the
    highest), ``dw`` its energy derivative (the four sum to the
    tetrahedron's DOS, which integrates to one).  ``curvature_correction``
    adds the Bloechl correction to both.  ``dw`` is the derivative between
    the corner energies; where three corners coincide the DOS, and so the
    corrected ``w``, steps, and the delta of that step is not in ``dw``
    (``tetrahedron_dos(..., cell_average=True)`` keeps it).
    """
    energies = np.atleast_1d(np.asarray(energies, dtype=float))
    corners = np.atleast_2d(np.asarray(corners, dtype=float))
    order = np.argsort(corners, axis=-1, kind="stable")
    e = np.take_along_axis(corners, order, axis=-1)
    w, dw, _n, _g = _sorted_weights(energies[:, None], e[None],
                                    curvature_correction)
    inverse = np.argsort(order, axis=-1)[None]
    return (np.take_along_axis(w, inverse, axis=-1),
            np.take_along_axis(dw, inverse, axis=-1))


# --------------------------------------------------------------------------- #
# The whole mesh.
# --------------------------------------------------------------------------- #

def degenerate_average(eigenvalues, projections, tolerance: float):
    """``projections`` with every degenerate set of levels averaged.

    At a k-point where levels are degenerate (consecutive levels closer than
    ``tolerance``), each state's projection depends on the arbitrary basis
    the eigensolver chose in the degenerate subspace -- a pair at the zone
    face can come out as one state per atom or as two shared ones -- while
    their sum does not.  A tetrahedron interpolates each band's projection
    from its corners, so the arbitrary split would enter the result; the
    set's average is the same in any basis.  ``eigenvalues`` ``(N_k, n)``
    ascending at each k-point, ``projections`` ``(N_k, M, n)``; returns a
    new array, equal to ``projections`` where nothing is degenerate.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    projections = np.asarray(projections, dtype=float)
    n_k, n_bands = eigenvalues.shape
    if n_bands < 2:
        return projections.copy()
    # One label per set: a new set starts where the gap exceeds tolerance.
    starts = np.diff(eigenvalues, axis=1) > tolerance
    labels = np.concatenate([np.zeros((n_k, 1), dtype=int),
                             np.cumsum(starts, axis=1)], axis=1)
    slot = (np.arange(n_k)[:, None] * n_bands + labels).ravel()
    flat = np.moveaxis(projections, 2, 1).reshape(n_k * n_bands, -1)
    sums = np.zeros_like(flat)
    np.add.at(sums, slot, flat)
    counts = np.bincount(slot, minlength=n_k * n_bands)
    averaged = sums[slot] / counts[slot, None]
    return np.moveaxis(averaged.reshape(n_k, n_bands, -1), 1, 2)


def tetrahedron_dos(energies, eigenvalues, tetrahedra, projections=None,
                    curvature_correction: bool = True,
                    integrated: bool = False,
                    cell_average: bool = False) -> np.ndarray:
    r"""Density of states (or number of states) by linear tetrahedra.

    Parameters
    ----------
    energies : array_like, shape (nE,)
        Where to evaluate, in the eigenvalues' unit (any order; distinct
        with ``cell_average``).
    eigenvalues : array_like, shape (N_k, n_bands)
        Band energies at every point of the full mesh, in the flat order
        ``tetrahedra`` indexes (:func:`mesh_tetrahedra`).
    tetrahedra : array_like of int, shape (N_tet, 4)
        Corner indices into the mesh; each tetrahedron is the fraction
        ``1/N_tet`` of the zone.
    projections : array_like, shape (N_k, M, n_bands), optional
        Weights :math:`p^\mu_{n\mathbf k}` of each state (Loewdin weights,
        say); the result is then ``(nE, M)``, one column per projection, each
        the corner-resolved integral of the interpolated product.
    curvature_correction : bool
        Add the Bloechl correction (default).  It leaves the total unchanged
        and acts only on projections, where it serves integrated quantities
        (``integrated=True``: the Fermi-sea sums of a metal) and not a
        projected DOS, which it can make negative near a band edge -- the
        calculator's ``pdos`` leaves it off.
    integrated : bool
        Return the number of states below each energy instead of the DOS.
    cell_average : bool
        Return, at each energy, the DOS averaged over its cell of the axis
        -- from the midpoint to the neighbor below to the midpoint to the
        neighbor above, half a spacing beyond the ends -- as the difference
        of the exact counts at the cell's edges.  The pointwise DOS misses
        what has no width: a band flat across a tetrahedron is a delta of
        its whole weight, and with three equal corners the corrected weights
        step, so their derivative carries a delta too.  The cell average
        keeps them, and the sum of its values times the cell widths is the
        count between the outer edges exactly.

    Returns
    -------
    ndarray, shape (nE,) or (nE, M)
        Per k-point weight :math:`1/N_k` and one spin: the DOS of each band
        integrates to one, and the number of states rises from 0 to
        ``n_bands``.
    """
    energies = np.atleast_1d(np.asarray(energies, dtype=float))
    if cell_average:
        if integrated:
            raise ValueError("cell_average is a DOS; the count is already "
                             "exact at every energy")
        order = np.argsort(energies, kind="stable")
        axis = energies[order]
        if len(axis) < 2 or np.any(np.diff(axis) <= 0.0):
            raise ValueError("cell_average needs two or more distinct "
                             "energies")
        middle = 0.5 * (axis[1:] + axis[:-1])
        edges = np.concatenate([[1.5 * axis[0] - 0.5 * axis[1]], middle,
                                [1.5 * axis[-1] - 0.5 * axis[-2]]])
        count = tetrahedron_dos(edges, eigenvalues, tetrahedra, projections,
                                curvature_correction, integrated=True)
        width = np.diff(edges)
        average = np.diff(count, axis=0) / (width if count.ndim == 1
                                            else width[:, None])
        result = np.empty_like(average)
        result[order] = average
        return result
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    tetrahedra = np.asarray(tetrahedra, dtype=int)
    if eigenvalues.ndim != 2:
        raise ValueError("eigenvalues must be (N_k, n_bands); got shape "
                         f"{eigenvalues.shape}")
    n_k, n_bands = eigenvalues.shape
    if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
        raise ValueError("tetrahedra must be (N_tet, 4) corner indices")
    if tetrahedra.min() < 0 or tetrahedra.max() >= n_k:
        raise ValueError("a tetrahedron corner indexes outside the mesh")
    if projections is not None:
        projections = np.asarray(projections, dtype=float)
        if projections.ndim != 3 or projections.shape[0] != n_k \
                or projections.shape[2] != n_bands:
            raise ValueError("projections must be (N_k, M, n_bands); got "
                             f"shape {projections.shape}")
    n_proj = 1 if projections is None else projections.shape[1]
    order = np.argsort(energies, kind="stable")
    axis = energies[order]
    out = np.zeros((len(axis), n_proj))
    n_tet = len(tetrahedra)

    # Bands in blocks so the (tetrahedron, band) pairs fit the budget.
    band_block = max(1, min(n_bands, CHUNK_ELEMENTS // (16 * n_tet)))
    for start in range(0, n_bands, band_block):
        bands = np.arange(start, min(n_bands, start + band_block))
        # (N_tet, nb, 4) corner energies, sorted with their mesh indices.
        corners = np.moveaxis(eigenvalues[tetrahedra][:, :, bands], 1, 2)
        sort = np.argsort(corners, axis=-1, kind="stable")
        e = np.take_along_axis(corners, sort, axis=-1).reshape(-1, 4)
        k = np.take_along_axis(
            np.broadcast_to(tetrahedra[:, None, :], corners.shape), sort,
            axis=-1).reshape(-1, 4)
        n_of_pair = np.broadcast_to(bands[None, :], corners.shape[:2]
                                    ).reshape(-1)
        _accumulate(out, axis, e, k, n_of_pair, projections,
                    curvature_correction, integrated)
    out /= n_tet
    result = np.empty_like(out)
    result[order] = out
    return result[:, 0] if projections is None else result


def _accumulate(out, axis, e, k, n_of_pair, projections, correction,
                integrated):
    """Add the pairs ``(e, k, n)`` to ``out`` ``(nE, M)`` on the sorted
    ``axis``, energy chunk by energy chunk, over the active pairs only."""
    n_proj = out.shape[1]
    e1, e4 = e[:, 0], e[:, 3]
    filled = np.zeros(n_proj)                 # pairs entirely below a chunk
    filled_mask = np.zeros(len(e), dtype=bool)
    n_axis = len(axis)
    start = 0
    while start < n_axis:
        # Grow the chunk while the active pairs fit the budget.
        stop = min(n_axis, start + 64)
        lo, hi = axis[start], axis[stop - 1]
        active = np.flatnonzero((e4 > lo) & (e1 < hi))
        while stop - start > 1 and (stop - start) * len(active) * 4 \
                > CHUNK_ELEMENTS:
            stop = start + max(1, (stop - start) // 2)
            hi = axis[stop - 1]
            active = np.flatnonzero((e4 > lo) & (e1 < hi))
        E = axis[start:stop, None]
        if integrated:
            below = (e4 <= lo) & ~filled_mask
            if below.any():
                filled += _filled(k[below], n_of_pair[below], projections)
                filled_mask |= below
            out[start:stop] += filled
        block = max(1, CHUNK_ELEMENTS // (4 * max(stop - start, n_proj)))
        for first in range(0, len(active), block):
            pairs = active[first:first + block]
            w, dw, _n, _g = _sorted_weights(E, e[pairs][None], correction)
            weights = w if integrated else dw          # (nc, P, 4)
            if projections is None:
                out[start:stop, 0] += weights.sum(axis=(1, 2))
            else:
                gathered = projections[k[pairs], :, n_of_pair[pairs, None]]
                out[start:stop] += (weights.reshape(len(E), -1)
                                    @ gathered.reshape(-1, n_proj))
        start = stop


def _filled(k, n_of_pair, projections) -> np.ndarray:
    """The contribution of tetrahedra entirely below the energy: a quarter
    of the projection at each corner (one state without projections)."""
    if projections is None:
        return np.array([float(len(k))])
    return projections[k, :, n_of_pair[:, None]].sum(axis=(0, 1)) / 4.0
