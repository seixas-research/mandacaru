# -*- coding: utf-8 -*-
# file: algorithms/kpath.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Band paths for a spectral function that exists only on a mesh.

In one dimension the Brillouin-zone sampling and the band path are the same
object: a chain's mesh is a line of k-points, and plotting the spectral
function against ``k`` is plotting it against the sampling.  In two or three
dimensions they part company.  The sampling is an ``n1 x n2 x n3`` mesh filling
the zone, while a band plot is a *one-dimensional* walk along segments joining
high-symmetry points -- ``G-X-M-G`` for a square lattice.

**The path cannot introduce k-points.**  The Bloch operators
:math:`\hat{c}^{\dagger}_{\mathbf{k}\nu\sigma}` are built as discrete Fourier
transforms over the Born-von Karman supercell's lattice translations, so they
exist at the ``N_c`` commensurate k-points of that supercell and nowhere else;
there is no interacting quantity to evaluate between them, and interpolating
one would be inventing data.  What this module does is therefore **selection**:
given a path, it returns the mesh points that *lie on* it, ordered along it,
with the linear coordinate a band plot needs for its x-axis.

The consequence a caller has to see is that a high-symmetry point can be
missing.  ``X = (0, 1/2, 0)`` is on a ``2 x 2 x 1`` or ``4 x 4 x 1`` mesh and is
*not* on a ``3 x 3 x 1`` mesh, so a ``G-X-M-G`` path over a ``3 x 3 x 1``
sampling genuinely has no X.  :class:`KPointPath` reports that in
:attr:`~KPointPath.missing` and the driver warns, rather than quietly drawing a
plot whose labels name points it never evaluated.

The linear coordinate follows ASE's convention -- :math:`2\pi` times the
reciprocal cell, in inverse Angstrom -- so an axis built here lines up with one
from :meth:`ase.dft.kpoints.BandPath.get_linear_kpoint_axis`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

__all__ = ["KPointPath", "band_path", "resolve_path"]

#: Tolerance for "this mesh point lies on this segment", in inverse Angstrom.
#: The test is done in Cartesian reciprocal space, so this is a real distance
#: and means the same thing for a cube and for a triclinic cell.  Mesh points
#: sit at rational fractions and special points at halves, thirds and eighths,
#: so the comparison is never marginal.
PATH_TOLERANCE = 1e-6

#: Half-width of the reciprocal-lattice image search used when testing whether
#: a mesh point lies on a segment.  The mesh is folded into ``[-1/2, 1/2)``
#: while ASE's special points are not -- FCC puts ``W`` at ``(1/2, 1/4, 3/4)``
#: and ``K`` at ``(3/8, 3/8, 3/4)`` -- so a point may need unfolding by a
#: reciprocal lattice vector to land on the path.  Two shells covers every
#: Bravais lattice's special points with room to spare, and the search is over
#: at most ``125`` offsets per mesh point per segment.
IMAGE_RANGE = (-2, -1, 0, 1, 2)


@dataclass
class KPointPath:
    """The mesh points lying on a band path, ordered along it.

    Attributes
    ----------
    path : str
        The path that was requested, e.g. ``"GXMG"``.  A comma marks a jump
        between disconnected segments, as in ASE.
    points : ndarray, shape (nk, 3)
        Fractional coordinates of the selected mesh points, in path order.  A
        point met twice by the path (the returning ``G`` of ``GXMG``) appears
        twice, at its two different path coordinates.
    indices : ndarray, shape (nk,)
        For each entry of :attr:`points`, its index into the **full mesh** it
        was selected from.  This is what gathers rows of a spectral function:
        ``spectral.weights[kpath.indices]``.
    distances : ndarray, shape (nk,)
        Linear coordinate along the path in inverse Angstrom, for the x-axis.
    labels : list of str
        High-symmetry labels in path order, including repeats.
    label_distances : ndarray
        Their positions on :attr:`distances`, for the x-ticks.
    special_points : dict
        Label to fractional coordinate, as used.
    missing : list of str
        Labels of the requested path that are **not** on the mesh.  Empty is
        the good case; anything here means the plot's axis names a point the
        calculation never evaluated.
    """

    path: str
    points: np.ndarray
    indices: np.ndarray
    distances: np.ndarray
    labels: list
    label_distances: np.ndarray
    special_points: dict
    missing: list

    def __len__(self) -> int:
        return int(len(self.indices))

    @property
    def complete(self) -> bool:
        """Whether every requested high-symmetry point is on the mesh."""
        return not self.missing

    def axis(self):
        """``(x, ticks, labels)`` -- the triple a band plot needs.

        Matches the shape of
        :meth:`ase.dft.kpoints.BandPath.get_linear_kpoint_axis`, so it drops
        into the same plotting code.
        """
        return self.distances, self.label_distances, self.labels

    def summary(self) -> str:
        """One line: how much of the path the mesh actually resolves."""
        text = (f"{self.path}: {len(self)} of the mesh's k-points lie on the "
                f"path ({len(self.labels)} high-symmetry points)")
        if self.missing:
            text += f"; not on the mesh: {', '.join(self.missing)}"
        return text


def resolve_path(cell, pbc=None, path=None):
    """Resolve a path string and its special points for a lattice.

    ``path=None`` asks ASE for the default path of the Bravais lattice the
    cell belongs to, restricted to the periodic directions -- which is what
    turns a square lattice in a tall box into the 2-D ``MGXM`` rather than a
    3-D tetragonal walk through ``Z``, ``R`` and ``A``.

    Returns ``(path, special_points)``.
    """
    from ase.cell import Cell

    cell = Cell(np.asarray(cell, dtype=float))
    kwargs = {} if pbc is None else {"pbc": np.asarray(pbc, dtype=bool)}
    band = cell.bandpath(npoints=0, **kwargs)
    special = {label: np.asarray(point, dtype=float)
               for label, point in band.special_points.items()}
    if path is None:
        path = band.path
    return str(path), special


def _segments(path, special_points):
    """Walk a path string into ``(label_a, label_b, is_continuation)`` triples.

    ``is_continuation`` says whether the segment starts where the previous one
    ended, which is how a shared endpoint is kept from being emitted twice.
    """
    unknown = sorted({label for label in path if label not in ","
                      and label not in special_points})
    if unknown:
        raise ValueError(
            f"unknown high-symmetry point(s) {', '.join(unknown)} in path "
            f"{path!r}; this lattice offers "
            f"{', '.join(sorted(special_points))}.")

    out = []
    for piece in path.split(","):
        labels = list(piece)
        if len(labels) < 2:
            raise ValueError(
                f"a path segment needs at least two points; got {piece!r} in "
                f"{path!r}.")
        for i in range(len(labels) - 1):
            out.append((labels[i], labels[i + 1], i > 0))
    return out


#: Integer offsets tried per mesh point, precomputed once.
_OFFSETS = np.array([[i, j, k] for i in IMAGE_RANGE for j in IMAGE_RANGE
                     for k in IMAGE_RANGE], dtype=float)


def _on_segment(kpt, start, stop, to_cartesian):
    """Where ``kpt`` sits on the segment, or ``None`` if it is not on it.

    Returns the parameter ``t`` in ``[0, 1]``.  Two things make this more than
    a dot product.

    First, the mesh is defined only **modulo a reciprocal lattice vector**, so
    ``kpt`` is tried against every offset in :data:`IMAGE_RANGE`: a mesh point
    folded to ``-1/2`` lies on a segment ending at ``+1/2``, and an FCC ``W`` at
    ``(1/2, 1/4, 3/4)`` is reached only after unfolding.

    Second, the projection is done in **Cartesian** reciprocal space rather
    than on fractional coordinates.  For an orthogonal cell the two agree up to
    a scale per axis; for a hexagonal or triclinic cell they do not, because
    ``b_i . b_j != 0`` means the fractional dot product is not the k-space one.
    Collinearity itself is affine-invariant, so a point exactly on the line
    passes either way -- but ``t`` is then the true fraction of the segment's
    *length*, which is what the plot's x-axis is measured in, and the tolerance
    is a distance in inverse Angstrom instead of a coordinate difference whose
    meaning changes with the lattice.
    """
    origin = to_cartesian(start)
    direction = to_cartesian(stop) - origin
    length = float(direction @ direction)
    if length < PATH_TOLERANCE ** 2:
        return None                        # a zero-length segment has no interior

    offsets = to_cartesian(kpt + _OFFSETS) - origin        # (n_offsets, 3)
    t = (offsets @ direction) / length
    residual = offsets - t[:, None] * direction
    inside = ((t > -PATH_TOLERANCE) & (t < 1.0 + PATH_TOLERANCE)
              & (np.einsum("ij,ij->i", residual, residual)
                 < PATH_TOLERANCE ** 2))
    if not np.any(inside):
        return None
    return float(np.clip(t[inside].min(), 0.0, 1.0))


def band_path(cell, kpoints, path=None, pbc=None, special_points=None,
              warn: bool = True) -> KPointPath:
    """Select the mesh k-points lying on a high-symmetry path.

    Parameters
    ----------
    cell : array_like, shape (3, 3)
        The **primitive** cell, whose reciprocal vectors set the path metric.
    kpoints : array_like, shape (nk, 3)
        Fractional coordinates of the commensurate mesh.
    path : str, optional
        High-symmetry labels, e.g. ``"GXMG"``; ``","`` separates disconnected
        pieces.  Defaults to the lattice's own path from ASE, restricted to
        ``pbc``.
    pbc : array_like of bool, optional
        Periodic directions, used only to pick the default path.
    special_points : dict, optional
        Override the label-to-coordinate table.
    warn : bool
        Emit a ``RuntimeWarning`` naming any requested high-symmetry point that
        the mesh does not contain.

    Returns
    -------
    KPointPath

    Notes
    -----
    Nothing is interpolated.  A segment along which the mesh has only its two
    endpoints contributes exactly those two points, and a coarse mesh gives a
    sparse plot -- which is the honest picture of what was computed.  Resolving
    more of a segment means enlarging the mesh, hence the supercell, hence the
    qubit count.
    """
    mesh = np.atleast_2d(np.asarray(kpoints, dtype=float))
    if mesh.ndim != 2 or mesh.shape[1] != 3:
        raise ValueError(f"kpoints must have shape (nk, 3); got {mesh.shape}")

    resolved, lattice_points = resolve_path(cell, pbc=pbc, path=path)
    path = resolved if path is None else str(path)
    if special_points is None:
        special_points = lattice_points
    else:
        special_points = {label: np.asarray(point, dtype=float)
                          for label, point in special_points.items()}

    from ase.cell import Cell

    reciprocal = 2.0 * np.pi * Cell(np.asarray(cell, dtype=float)).reciprocal()

    def cartesian(frac):
        """Fractional to Cartesian k, in inverse Angstrom.

        ``reciprocal`` holds ``b_i`` as rows, so this is correct for any cell
        shape: a hexagonal or triclinic lattice picks up the ``b_i . b_j``
        cross terms that a per-axis scaling would miss.
        """
        return np.asarray(frac, dtype=float) @ reciprocal

    points, indices, distances = [], [], []
    labels, label_distances = [], []
    missing = []
    travelled = 0.0

    for label_a, label_b, continues in _segments(path, special_points):
        start, stop = special_points[label_a], special_points[label_b]
        span = float(np.linalg.norm(cartesian(stop) - cartesian(start)))

        found = []
        for index, kpt in enumerate(mesh):
            t = _on_segment(kpt, start, stop, cartesian)
            if t is not None:
                found.append((t, index))
        found.sort(key=lambda item: (item[0], item[1]))

        # Record the endpoints as labels whether or not the mesh has them; a
        # label the mesh misses is reported rather than dropped, because an
        # x-tick naming an unevaluated point is the failure mode here.
        for label, target in ((label_a, 0.0), (label_b, 1.0)):
            if not any(abs(t - target) < PATH_TOLERANCE for t, _i in found):
                if label not in missing:
                    missing.append(label)

        if not continues:
            labels.append(label_a)
            label_distances.append(travelled)
        labels.append(label_b)
        label_distances.append(travelled + span)

        for t, index in found:
            if continues and t < PATH_TOLERANCE:
                continue               # already emitted as the previous stop
            points.append(mesh[index])
            indices.append(index)
            distances.append(travelled + t * span)

        travelled += span

    if warn and missing:
        warnings.warn(
            f"the path {path!r} names high-symmetry point(s) "
            f"{', '.join(missing)} that are not on this "
            f"{len(mesh)}-point mesh, so the spectral function was never "
            "evaluated there: the x-ticks name them but no data sits under "
            "them.  Choose a mesh whose divisions land on them (a 2x2x1 or "
            "4x4x1 mesh carries X and M of a square lattice; a 3x3x1 mesh "
            "does not carry X) or restrict the path.",
            RuntimeWarning, stacklevel=2)

    return KPointPath(
        path=path,
        points=(np.asarray(points, dtype=float).reshape(-1, 3)
                if points else np.empty((0, 3))),
        indices=np.asarray(indices, dtype=int),
        distances=np.asarray(distances, dtype=float),
        labels=labels,
        label_distances=np.asarray(label_distances, dtype=float),
        special_points=special_points,
        missing=missing)
