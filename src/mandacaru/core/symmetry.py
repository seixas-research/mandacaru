# -*- coding: utf-8 -*-
# file: core/symmetry.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Crystal symmetry, and the irreducible Brillouin zone it implies.

The space group and its symmetry operations are found natively in Rust.
The Brillouin zone reduction below is then done on Mandacaru's own mesh, in
Mandacaru's own fractional convention.

That split is deliberate. Symmetry detection supplies the **symmetry
operations**; reducing our own array with those rotations ensures every k-point
that comes out is one that went in.

What the reduction is
---------------------

A symmetry operation of the crystal maps an eigenstate at ``k`` onto one at
``k'``, so the spectral function obeys :math:`A(E, \mathbf{k}') = A(E,
\mathbf{k})` for every ``k'`` in the orbit of ``k``.  The irreducible wedge is
one representative per orbit, and the weight is the size of the orbit.  Under a
real-space rotation ``R`` acting on fractional coordinates, the fractional
reciprocal coordinate transforms as

.. math::

    \mathbf{k}' = R^{\mathsf{T}} \mathbf{k} ,

since :math:`\mathbf{k}' \cdot \mathbf{x} = \mathbf{k} \cdot (R\mathbf{x})`.
Translations do not move ``k`` -- they only put a phase on the state -- so only
the rotation part is used.

**Time reversal** adds :math:`A(E, -\mathbf{k}) = A(E, \mathbf{k})` for a
system without spin polarization.  It is a genuine symmetry of the Hamiltonian
here (real integrals, collinear spin), and it is what makes ``k`` and ``-k``
one orbit even in a lattice whose point group lacks inversion.

What it does **not** buy
------------------------

The irreducible zone reduces the number of k-points at which the spectral
function must be evaluated.  It does **not** reduce the qubit count: the
Born-von Karman supercell is fixed by the full ``n1 x n2 x n3`` mesh, and every
k-point of that mesh is already inside the one supercell being solved.  The
saving is in :meth:`~mandacaru.algorithms.bloch._BlochMixin.get_spectral_function`,
whose cost is a Lehmann evaluation per (k-point, orbital, spin, branch), not in
the variational run that precedes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["SymmetryInfo", "IrreducibleZone", "crystal_symmetry",
           "irreducible_kpoints"]

#: Default distance tolerance handed to symmetry search, in Angstrom.
DEFAULT_SYMPREC = 1e-5

#: Fractional coordinates are compared after folding to ``[-1/2, 1/2)``; two
#: k-points closer than this are the same point.
KPOINT_TOLERANCE = 1e-8

_IRREDUCIBLE_CACHE: dict = {}


@dataclass
class SymmetryInfo:
    """The space group of a periodic geometry, as reported natively.

    Attributes
    ----------
    number : int
        International space-group number (1-230).
    international : str
        Hermann-Mauguin symbol, e.g. ``"P4/mmm"``.
    point_group : str
        Point-group symbol of the crystal.
    rotations : ndarray, shape (n_ops, 3, 3)
        Integer rotation matrices acting on **fractional real-space**
        coordinates.  These are what the Brillouin-zone reduction uses.
    translations : ndarray, shape (n_ops, 3)
        Fractional translation parts.  Carried for completeness; they do not
        move a k-point.
    symprec : float
        The tolerance the operations were requested with.
    symprec_used : float | None
        The tolerance actually used after closure verification.
    n_pure_translations : int
        Number of pure translation operations (with W = I).
    """

    number: int
    international: str
    point_group: str
    rotations: np.ndarray
    translations: np.ndarray
    symprec: float
    symprec_used: float | None = None
    n_pure_translations: int = 1

    @property
    def n_operations(self) -> int:
        """Number of symmetry operations of the space group."""
        return int(len(self.rotations))

    def summary(self) -> str:
        """One line naming the group and how many operations it has."""
        return (f"{self.international} (No. {self.number}), point group "
                f"{self.point_group}, {self.n_operations} operations "
                f"(symprec {self.symprec:g})")


@dataclass
class IrreducibleZone:
    """One representative per symmetry orbit of a k-point mesh.

    Attributes
    ----------
    points : ndarray, shape (n_ibz, 3)
        Fractional coordinates of the representatives, in the order their
        orbit was first met while walking the full mesh.
    weights : ndarray, shape (n_ibz,)
        Orbit sizes.  They sum to the number of k-points in the full mesh, and
        ``weights / weights.sum()`` is the normalized Brillouin-zone weight.
    mapping : ndarray, shape (n_kpoints,)
        For each point of the **full** mesh, the index of its representative in
        :attr:`points`.  This is what expands a quantity computed on the wedge
        back onto the mesh: ``full = wedge[mapping]``.
    indices : ndarray, shape (n_ibz,)
        Index into the full mesh of each representative, so a caller can say
        *which* of its own k-points was the one evaluated.
    symmetry : SymmetryInfo
        The operations used.
    time_reversal : bool
        Whether ``k -> -k`` was included among them.
    """

    points: np.ndarray
    weights: np.ndarray
    mapping: np.ndarray
    indices: np.ndarray
    symmetry: SymmetryInfo
    time_reversal: bool = True
    #: Filled in by a consumer that checked the assumption: the largest
    #: relative disagreement found between an orbit member and its
    #: representative.  ``None`` means nobody checked, which is not the same as
    #: zero.
    symmetry_residual: float = None
    _full: np.ndarray = field(default=None, repr=False)

    @property
    def n_kpoints(self) -> int:
        """Size of the full mesh this wedge was reduced from."""
        return int(self.mapping.size)

    @property
    def reduction(self) -> float:
        """Full mesh size divided by wedge size: the factor saved."""
        return self.n_kpoints / len(self.points)

    def expand(self, values) -> np.ndarray:
        """Spread a per-wedge quantity back over the full mesh.

        ``values`` is indexed by irreducible point and may carry any trailing
        shape, so both a per-k scalar and a full ``(n_ibz, nE)`` spectrum
        expand with the same call.
        """
        values = np.asarray(values)
        if len(values) != len(self.points):
            raise ValueError(
                f"expected one entry per irreducible k-point "
                f"({len(self.points)}), got {len(values)}")
        return values[self.mapping]

    def summary(self) -> str:
        """One line: how many points the symmetry removed."""
        reversal = " + time reversal" if self.time_reversal else ""
        return (f"{self.n_kpoints} k-points -> {len(self.points)} irreducible "
                f"({self.reduction:.2f}x) under {self.symmetry.international}"
                f"{reversal}")


def crystal_symmetry(atoms, symprec: float = DEFAULT_SYMPREC) -> SymmetryInfo:
    """Find the space group of an ASE ``Atoms`` natively."""
    from mandacaru import _native

    cell = np.ascontiguousarray(atoms.cell[:], dtype=np.float64)
    positions = np.ascontiguousarray(atoms.get_scaled_positions(), dtype=np.float64)
    numbers = np.ascontiguousarray(atoms.get_atomic_numbers(), dtype=np.int64)

    raw = _native.crystal_symmetry(cell, positions, numbers, float(symprec))

    return SymmetryInfo(
        number=int(raw["number"]),
        international=str(raw["international"]),
        point_group=str(raw["point_group"]),
        rotations=np.asarray(raw["rotations"], dtype=int),
        translations=np.asarray(raw["translations"], dtype=float),
        symprec=float(raw["symprec"]),
        symprec_used=float(raw["symprec_used"]),
        n_pure_translations=int(raw["n_pure_translations"]),
    )


def _fold(k):
    """Fold fractional k-points into ``[-1/2, 1/2)``.

    ``np.round`` breaks its ties to even, so ``+0.5`` and ``-0.5`` both land on
    the same representative -- which is what is wanted, since they differ by a
    reciprocal lattice vector and are the same physical point.
    """
    k = np.asarray(k, dtype=float)
    return k - np.round(k)


def irreducible_kpoints(kpoints, symmetry: SymmetryInfo,
                        time_reversal: bool = True) -> IrreducibleZone:
    """Reduce a k-point mesh to one representative per symmetry orbit.

    Parameters
    ----------
    kpoints : array_like, shape (nk, 3)
        Fractional coordinates of the full mesh, in the caller's own order.
    symmetry : SymmetryInfo
        From :func:`crystal_symmetry`.
    time_reversal : bool
        Include ``k -> -k``.  True for a system without spin polarization,
        which is every case Mandacaru's periodic path currently solves.

    Returns
    -------
    IrreducibleZone

    Notes
    -----
    The orbit of ``k`` is ``{R^T k}`` over every rotation, folded back into the
    first zone, plus its negatives when time reversal is on.  A point is
    assigned to the first representative whose orbit contains it, so walking
    the mesh in order gives a stable, reproducible wedge.
    """
    full = np.atleast_2d(np.asarray(kpoints, dtype=float))
    if full.ndim != 2 or full.shape[1] != 3:
        raise ValueError(
            f"kpoints must have shape (nk, 3); got {full.shape}")

    ir_key = (tuple(np.round(full.flatten(), 8)), id(symmetry), bool(time_reversal))
    if ir_key in _IRREDUCIBLE_CACHE:
        return _IRREDUCIBLE_CACHE[ir_key]

    rotations = np.asarray(symmetry.rotations, dtype=int)
    reps: list[np.ndarray] = []
    rep_indices: list[int] = []
    weights: list[int] = []
    mapping = np.empty(len(full), dtype=int)

    for index, kpt in enumerate(full):
        star = _fold(rotations.transpose(0, 2, 1) @ kpt)
        if time_reversal:
            star = np.concatenate([star, _fold(-star)])

        found = None
        for slot, representative in enumerate(reps):
            # Two k-points are the same when they differ by a reciprocal
            # lattice vector, so compare the folded difference, not the raw one.
            if np.any(np.all(np.abs(_fold(star - representative))
                             < KPOINT_TOLERANCE, axis=1)):
                found = slot
                break

        if found is None:
            found = len(reps)
            reps.append(_fold(kpt))
            rep_indices.append(index)
            weights.append(0)
        mapping[index] = found
        weights[found] += 1

    zone = IrreducibleZone(
        points=np.asarray(reps, dtype=float).reshape(-1, 3),
        weights=np.asarray(weights, dtype=int),
        mapping=mapping,
        indices=np.asarray(rep_indices, dtype=int),
        symmetry=symmetry,
        time_reversal=bool(time_reversal),
        _full=full)
    _IRREDUCIBLE_CACHE[ir_key] = zone
    return zone
