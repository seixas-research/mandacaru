# -*- coding: utf-8 -*-
# file: core/symmetry.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Crystal symmetry, and the irreducible Brillouin zone it implies.

The space group is found with `spglib <https://spglib.readthedocs.io>`_, which
is the one thing here that is not generated natively: recognizing a space group
from a cell and a basis is a well-posed combinatorial problem with a canonical
implementation, and re-deriving it would add a class of silent errors for no
physics.  What spglib supplies is the **symmetry operations**; the Brillouin
zone reduction below is then done on Mandacaru's own mesh, in Mandacaru's own
fractional convention.

That split is deliberate.  spglib also offers ``get_ir_reciprocal_mesh``, but
it carries its own grid-address convention (integer addresses folded to
``[0, 1)``, shifts expressed in half-grid units), and mapping that back onto
the ``[-1/2, 1/2)`` Gamma-centered mesh
:func:`~mandacaru.algorithms._hamiltonian_from_atoms.monkhorst_pack_kpts`
builds is exactly the kind of off-by-a-convention step that produces a plot
that looks plausible and is wrong.  Reducing our own array with spglib's
rotations has no such step: every k-point that comes out is one that went in.

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
           "irreducible_kpoints", "require_spglib"]

#: Default distance tolerance handed to spglib, in Angstrom.  spglib's own
#: default is 1e-5; a looser value merges sites that a relaxation left slightly
#: off their ideal positions, which is usually what is wanted for a *relaxed*
#: geometry and never what is wanted for a pinned test.
DEFAULT_SYMPREC = 1e-5

#: Fractional coordinates are compared after folding to ``[-1/2, 1/2)``; two
#: k-points closer than this are the same point.
KPOINT_TOLERANCE = 1e-8


def require_spglib():
    """Import spglib, configured to raise rather than signal, or explain how
    to get it.

    It is a hard dependency of the package, so the ImportError branch only
    fires in an environment assembled by hand.

    ``spglib.error.OLD_ERROR_HANDLING`` defaults to ``True``, which keeps the
    legacy behavior of stashing a failure in a module-level string and
    returning a sentinel -- and emits a ``DeprecationWarning`` on **every**
    call while it does.  spglib says the flag will be removed, so the forward
    path is to turn it off once, here, at the single point where spglib is
    imported: errors then arrive as exceptions, which is what the callers
    below already expect, and the per-call warning stops.
    """
    try:
        import spglib
    except ImportError as error:                          # pragma: no cover
        raise ImportError(
            "crystal symmetry needs spglib, which Mandacaru lists as a "
            "dependency: install it with `pip install spglib`, or reinstall "
            "the package with `pip install -e '.[dev]'`.") from error

    try:
        from spglib import error as _spglib_error
    except ImportError:                                   # pragma: no cover
        return spglib                                     # older spglib
    if getattr(_spglib_error, "OLD_ERROR_HANDLING", False):
        _spglib_error.OLD_ERROR_HANDLING = False
    return spglib


@dataclass
class SymmetryInfo:
    """The space group of a periodic geometry, as spglib reports it.

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
        The tolerance the operations were found with.  Recorded because the
        answer genuinely depends on it.
    """

    number: int
    international: str
    point_group: str
    rotations: np.ndarray
    translations: np.ndarray
    symprec: float

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
    """Find the space group of an ASE ``Atoms`` with spglib.

    The cell is taken as given.  A slab or a chain carries vacuum along its
    open directions, so spglib reports the space group of the *periodic
    repetition of that cell* -- a square lattice of atoms in a tall box comes
    back as tetragonal ``P4/mmm``, not as a 2-D plane group.  That is the right
    answer for what is being solved (the supercell really is 3-D periodic in
    the basis functions' lattice sum), and the operations it returns are the
    ones that act correctly on the mesh.
    """
    spglib = require_spglib()
    cell = (np.asarray(atoms.cell[:], dtype=float),
            np.asarray(atoms.get_scaled_positions(), dtype=float),
            np.asarray(atoms.get_atomic_numbers(), dtype=int))
    dataset = spglib.get_symmetry_dataset(cell, symprec=float(symprec))
    if dataset is None:                                    # pragma: no cover
        raise ValueError(
            "spglib could not determine the space group of this geometry; "
            "check that the cell is non-degenerate and that symprec "
            f"({symprec:g}) is sensible for the coordinates.")

    # spglib >= 2.5 returns a dataclass, earlier versions a dict.  Both are in
    # the wild, and the attribute names coincide with the dict keys.
    def field_of(name):
        if isinstance(dataset, dict):
            return dataset[name]
        return getattr(dataset, name)

    return SymmetryInfo(
        number=int(field_of("number")),
        international=str(field_of("international")),
        point_group=str(field_of("pointgroup")),
        rotations=np.asarray(field_of("rotations"), dtype=int),
        translations=np.asarray(field_of("translations"), dtype=float),
        symprec=float(symprec))


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

    return IrreducibleZone(
        points=np.asarray(reps, dtype=float).reshape(-1, 3),
        weights=np.asarray(weights, dtype=int),
        mapping=mapping,
        indices=np.asarray(rep_indices, dtype=int),
        symmetry=symmetry,
        time_reversal=bool(time_reversal),
        _full=full)
