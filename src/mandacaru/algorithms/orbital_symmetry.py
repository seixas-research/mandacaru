# -*- coding: utf-8 -*-
# file: algorithms/orbital_symmetry.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Molecular point groups and the symmetry of molecular orbitals.

The ``active_space`` option's ``"symmetry"`` needs two things this module
provides: the symmetry operations of the molecule, and how each molecular
orbital transforms under them.

The operations
--------------

A point-group operation is an orthogonal :math:`3 \times 3` matrix :math:`R`,
about the molecule's charge center, that maps every atom onto an atom of the
same kind.  They are found by construction rather than by guessing axes: two
atoms :math:`a, b` that are not collinear with the center fix an operation
completely once their images :math:`a', b'` are chosen (an orthonormal frame
built from the pair, and its mirror image).  Every candidate pair of images at
the same distances is tried and kept if it maps the whole molecule onto
itself.  This finds every operation of every finite group, including the cubic
and icosahedral ones, with no table of axes.

A **linear** molecule has an infinite group, :math:`C_{\infty v}` or
:math:`D_{\infty h}`.  It is represented by its subgroup of order 16 or 32
built on an 8-fold axis, which keeps :math:`\sigma, \pi, \delta, \phi`
orbitals apart (their characters under the 8-fold rotation differ).  A single
**atom** is represented by the icosahedral group :math:`I_h`, which keeps
every s, p and d shell irreducible; an f shell splits.

How an orbital transforms
-------------------------

Each operation is represented in the atomic-orbital basis by evaluating every
basis function at the transformed points and solving

.. math::

    \chi_\nu(R^{-1}\mathbf r) = \sum_\mu \chi_\mu(\mathbf r)\, D_{\mu\nu}(R)

by least squares on points around the atoms.  The fit is exact up to the
function evaluation, because a complete atomic shell on an atom maps onto the
same shell on the image atom, and it needs no convention for the angular
functions.  In the molecular orbitals it becomes
:math:`U(R) = C^\dagger S D(R) C`.

Orbitals that some :math:`U(R)` mixes form a **degenerate set**, and the trace
of :math:`U(R)` over a set is its character.  The real-space grid breaks the
symmetry slightly, so the matrices are only nearly symmetric; mixing is
recognized above :data:`MIXING_THRESHOLD` and a character is reported as
irreducible only when its norm is within :data:`CHARACTER_TOLERANCE` of one.

Labels
------

The constraints work on the characters.  The Mulliken labels in the run log
are derived from them by the usual rules: A/B for one-dimensional
representations (symmetric or antisymmetric under the principal rotation),
E/T/G/H for dimensions 2 to 5, g/u under inversion, primes under
:math:`\sigma_h`, and subscripts 1/2 under the perpendicular :math:`C_2` (or
the vertical mirror).  Where the textbook convention depends on axis
orientation it is fixed here as follows: in :math:`C_{2v}` and
:math:`D_{2h}`-type groups the plane or axis that contains the most atoms is
:math:`yz` or :math:`z`, and a planar molecule's normal is :math:`x`.  So
water's out-of-plane lone pair is :math:`b_1`, as in Mulliken's 1955
recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product

import numpy as np

__all__ = ["PointGroup", "OrbitalSymmetry", "point_group",
           "orbital_symmetry", "contains"]

#: Largest distance, in Bohr, between an atom's image and the atom it lands on.
POSITION_TOLERANCE = 0.02

#: Largest :math:`|U(R)_{pq}|` taken as "these orbitals do not mix".
MIXING_THRESHOLD = 0.3

#: Largest deviation of a character's norm from 1 for an irreducible set.
CHARACTER_TOLERANCE = 0.3

#: Largest deviation, in direction cosines or matrix elements, between two
#: axes or operations taken as the same.  Coordinates good to 1e-5 Angstrom
#: give axes good to about 1e-5, so this is loose on purpose.
AXIS_TOLERANCE = 1e-3

#: Order of the rotation axis that stands in for :math:`C_\infty`.
LINEAR_ORDER = 8


# --------------------------------------------------------------------------- #
# Operations
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Operation:
    """One point-group operation: its matrix and what kind it is."""

    matrix: np.ndarray
    kind: str                 # "E", "C", "S", "i" or "sigma"
    order: int = 1            # n of C_n / S_n
    axis: np.ndarray | None = None   # rotation axis, or a mirror's normal


def _classify(R: np.ndarray) -> Operation:
    """Name an orthogonal matrix: identity, rotation, mirror, S_n, inversion."""
    det = float(np.linalg.det(R))
    proper = R if det > 0 else -R
    if det > 0 and np.allclose(R, np.eye(3), atol=AXIS_TOLERANCE):
        return Operation(R, "E")
    if det < 0 and np.allclose(R, -np.eye(3), atol=AXIS_TOLERANCE):
        return Operation(R, "i", 2)
    values, vectors = np.linalg.eig(proper)
    axis = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
    axis = axis / np.linalg.norm(axis)
    cos = np.clip(0.5 * (np.trace(proper) - 1.0), -1.0, 1.0)
    angle = float(np.arccos(cos))
    if det > 0:
        return Operation(R, "C", _order(angle), axis)
    if np.isclose(angle, np.pi, atol=AXIS_TOLERANCE):
        # -R is a C2 about the normal: R is the mirror through the normal.
        return Operation(R, "sigma", 2, axis)
    # R = -C(angle) = S(angle + pi) about the same axis.
    return Operation(R, "S", _order(np.pi - angle), axis)


def _order(angle: float) -> int:
    """The smallest n with n * angle a multiple of 2 pi (angle in (0, pi])."""
    for n in range(2, 61):
        k = angle * n / (2.0 * np.pi)
        if abs(k - round(k)) < AXIS_TOLERANCE:
            return n
    return 0


def _frame(u1, u2, handed: float = 1.0) -> np.ndarray:
    """Orthonormal columns from two non-collinear vectors (Gram-Schmidt)."""
    e1 = u1 / np.linalg.norm(u1)
    e2 = u2 - (u2 @ e1) * e1
    e2 = e2 / np.linalg.norm(e2)
    return np.column_stack([e1, e2, handed * np.cross(e1, e2)])


def _maps_onto(R, x, species, tolerance) -> bool:
    """Whether ``R`` sends every atom onto an atom of the same species."""
    image = x @ R.T
    for p, s in zip(image, species):
        distances = np.linalg.norm(x - p, axis=1)
        if not np.any((distances < tolerance) & (species == s)):
            return False
    return True


def _unique(matrices) -> list[np.ndarray]:
    kept: list[np.ndarray] = []
    for R in matrices:
        if not any(np.allclose(R, Q, atol=AXIS_TOLERANCE) for Q in kept):
            kept.append(R)
    return kept


def _finite_operations(x, species, tolerance) -> list[np.ndarray]:
    """Every operation of a non-linear arrangement, by pair mapping."""
    radii = np.linalg.norm(x, axis=1)
    # The frame is best conditioned on atoms far from the center and far from
    # collinear: a is the outermost atom, b the outermost one at an angle.
    order = np.argsort(-radii)
    a = int(order[0])
    b = next(int(j) for j in order
             if np.linalg.norm(np.cross(x[a], x[j]))
             > 0.1 * radii[a] * radii[j])
    frame = _frame(x[a], x[b])
    found = []
    for a2, b2 in product(range(len(x)), repeat=2):
        if species[a2] != species[a] or species[b2] != species[b]:
            continue
        if abs(radii[a2] - radii[a]) > tolerance or \
                abs(radii[b2] - radii[b]) > tolerance:
            continue
        if abs(x[a2] @ x[b2] - x[a] @ x[b]) > tolerance * (radii[a]
                                                          + radii[b]):
            continue
        if np.linalg.norm(np.cross(x[a2], x[b2])) < 1e-8:
            continue
        for handed in (1.0, -1.0):
            R = _frame(x[a2], x[b2], handed) @ frame.T
            if _maps_onto(R, x, species, tolerance):
                found.append(R)
    return _unique(found)


def _rotation(axis, angle) -> np.ndarray:
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _perpendicular(axis) -> np.ndarray:
    trial = np.eye(3)[int(np.argmin(np.abs(axis)))]
    v = np.cross(axis, trial)
    return v / np.linalg.norm(v)


def _linear_operations(axis, centrosymmetric: bool) -> list[np.ndarray]:
    """The order-16 (C8v) or order-32 (D8h) stand-in for C_inf v / D_inf h."""
    n = LINEAR_ORDER
    rotations = [_rotation(axis, 2.0 * np.pi * k / n) for k in range(n)]
    normal = _perpendicular(axis)
    mirror = np.eye(3) - 2.0 * np.outer(normal, normal)
    ops = rotations + [R @ mirror for R in rotations]
    if centrosymmetric:
        ops += [-R for R in ops]
    return _unique(ops)


def _icosahedral_operations() -> list[np.ndarray]:
    """I_h, from the twelve vertices of an icosahedron."""
    phi = 0.5 * (1.0 + np.sqrt(5.0))
    vertices = []
    for s1, s2 in product((1.0, -1.0), repeat=2):
        vertices += [(0.0, s1, s2 * phi), (s1, s2 * phi, 0.0),
                     (s2 * phi, 0.0, s1)]
    x = np.array(vertices)
    return _finite_operations(x, np.zeros(len(x), dtype=int), 1e-6)


# --------------------------------------------------------------------------- #
# The group
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PointGroup:
    """A molecule's point group: its name and its operations.

    Attributes
    ----------
    name : str
        Schoenflies symbol (``"C2v"``, ``"Dinfh"`` for :math:`D_{\\infty h}`,
        ``"Kh"`` for an atom).
    operations : tuple of Operation
        Every operation, the identity first.  For a linear molecule or an atom
        these are the finite stand-in described in the module docstring.
    principal : ndarray or None
        Unit vector of the principal axis (``z``), when there is one.
    secondary : ndarray or None
        Unit vector of the ``y`` axis that fixes the 1/2 subscripts and the
        B1/B2/B3 labels, when the group needs one.
    center : ndarray
        The point the operations act about (the charge center), in the
        coordinates the positions were given in.
    """

    name: str
    operations: tuple[Operation, ...]
    principal: np.ndarray | None = None
    secondary: np.ndarray | None = None
    center: np.ndarray | None = None

    @property
    def order(self) -> int:
        return len(self.operations)

    @property
    def linear(self) -> bool:
        return "inf" in self.name

    @property
    def cubic(self) -> bool:
        return self.name[0] in ("T", "O", "I", "K")


def _axes_of(ops, kind, order=None):
    """Distinct axes (up to sign) of the operations of a kind and order."""
    axes: list[np.ndarray] = []
    for op in ops:
        if op.kind != kind or (order is not None and op.order != order):
            continue
        if not any(abs(abs(op.axis @ a) - 1.0) < AXIS_TOLERANCE for a in axes):
            axes.append(op.axis)
    return axes


def _count_atoms_on(x, axis, plane: bool) -> int:
    """Atoms on an axis (``plane=False``) or in the plane with this normal."""
    if plane:
        return int(np.sum(np.abs(x @ axis) < POSITION_TOLERANCE))
    off = x - np.outer(x @ axis, axis)
    return int(np.sum(np.linalg.norm(off, axis=1) < POSITION_TOLERANCE))


def point_group(numbers, positions, *, signatures=None,
                tolerance: float = POSITION_TOLERANCE) -> PointGroup:
    """The point group of atoms at ``positions`` (Bohr).

    ``numbers`` tell the species apart; ``signatures``, one hashable per atom
    (its basis, say), split a species further, since two atoms of the same
    element with different basis sets are not equivalent for the orbitals.
    """
    positions = np.asarray(positions, dtype=float)
    numbers = np.asarray(numbers, dtype=float)
    keys = [(float(z), None if signatures is None else signatures[i])
            for i, z in enumerate(numbers)]
    index = {k: n for n, k in enumerate(dict.fromkeys(keys))}
    species = np.array([index[k] for k in keys])
    # The center of each species is invariant under every operation, so the
    # mean of the per-species centroids is too.
    center = np.mean([positions[species == k].mean(axis=0)
                      for k in np.unique(species)], axis=0)
    x = positions - center
    return replace(_point_group(x, species, tolerance), center=center)


def _point_group(x, species, tolerance) -> PointGroup:
    if len(x) == 1:
        ops = [_classify(R) for R in _icosahedral_operations()]
        return _named("Kh", ops, x)
    radii = np.linalg.norm(x, axis=1)
    far = int(np.argmax(radii))
    axis = x[far] / radii[far]
    if np.all(np.linalg.norm(np.cross(x, axis), axis=1) < tolerance):
        inversion = _maps_onto(-np.eye(3), x, species, tolerance)
        ops = [_classify(R) for R in _linear_operations(axis, inversion)]
        name = "Dinfh" if inversion else "Cinfv"
        return PointGroup(name, tuple(_identity_first(ops)), axis,
                          _perpendicular(axis))
    ops = [_classify(R) for R in _finite_operations(x, species, tolerance)]
    return _named(None, ops, x)


def _identity_first(ops):
    return sorted(ops, key=lambda op: op.kind != "E")


def _named(name, ops, x) -> PointGroup:
    """Schoenflies name and reference axes from the operations."""
    ops = _identity_first(ops)
    has_i = any(op.kind == "i" for op in ops)
    mirrors = [op for op in ops if op.kind == "sigma"]
    c3 = _axes_of(ops, "C", 3)
    if name is None and len(c3) >= 4:
        if _axes_of(ops, "C", 5):
            name = "Ih" if has_i else "I"
        elif _axes_of(ops, "C", 4):
            name = "Oh" if has_i else "O"
        else:
            name = "Th" if has_i else ("Td" if mirrors else "T")
    if name is not None:
        principal = (_axes_of(ops, "C", 5) or _axes_of(ops, "C", 4)
                     or _axes_of(ops, "C", 2) or [None])[0]
        return PointGroup(name, tuple(ops), principal, None)

    rotations = [op for op in ops if op.kind == "C"]
    n = max((op.order for op in rotations), default=1)
    if n == 1:
        name = "Cs" if mirrors else ("Ci" if has_i else "C1")
        normal = mirrors[0].axis if mirrors else None
        return PointGroup(name, tuple(ops), normal, None)
    candidates = _axes_of(ops, "C", n)
    if n == 2 and len(candidates) > 1:
        # D2-type: z is the C2 axis with the most atoms on it; a tie goes to
        # the axis lying in the plane of the most atoms.
        candidates.sort(key=lambda a: (-_count_atoms_on(x, a, False),
                                       _count_atoms_on(x, a, True)))
    principal = candidates[0]
    c2_perp = [a for a in _axes_of(ops, "C", 2)
               if abs(a @ principal) < AXIS_TOLERANCE]
    sigma_h = any(abs(abs(op.axis @ principal) - 1) < AXIS_TOLERANCE for op in mirrors)
    sigma_v = [op.axis for op in mirrors if abs(op.axis @ principal) < AXIS_TOLERANCE]
    if len(c2_perp) >= n:
        if sigma_h:
            name = f"D{n}h"
        elif len(sigma_v) >= n:
            name = f"D{n}d"
        else:
            name = f"D{n}"
        # y: the perpendicular C2 with the most atoms on it; on a tie, the
        # one lying in the molecule's plane, so a planar molecule's normal
        # is x.
        c2_perp.sort(key=lambda a: (-_count_atoms_on(x, a, False),
                                    _count_atoms_on(x, a, True)))
        secondary = c2_perp[0]
    else:
        if sigma_h:
            name = f"C{n}h"
        elif len(sigma_v) >= n:
            name = f"C{n}v"
        elif _axes_of(ops, "S", 2 * n):
            name = f"S{2 * n}"
        else:
            name = f"C{n}"
        # The vertical mirror containing the most atoms is sigma_v(yz): its
        # normal is x, so y is the in-plane direction perpendicular to z.
        secondary = None
        if sigma_v:
            normal = max(sigma_v, key=lambda a: _count_atoms_on(x, a, True))
            secondary = np.cross(principal, normal)
    return PointGroup(name, tuple(ops), principal, secondary)


# --------------------------------------------------------------------------- #
# Orbitals
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class OrbitalSymmetry:
    """How a set of orbitals transforms under a point group.

    Attributes
    ----------
    group : PointGroup
    sets : tuple of tuple of int
        Degenerate sets: orbitals that some operation mixes, in index order.
    characters : tuple of ndarray
        Each set's character, one value per operation.
    labels : tuple of str
        Each set's Mulliken label, lower case as for orbitals; ``"?"`` when
        the set is not irreducible within the tolerance.
    """

    group: PointGroup
    sets: tuple[tuple[int, ...], ...]
    characters: tuple[np.ndarray, ...] = field(repr=False)
    labels: tuple[str, ...]

    def set_of(self, p: int) -> int:
        for k, members in enumerate(self.sets):
            if p in members:
                return k
        raise KeyError(p)

    def label(self, p: int) -> str:
        return self.labels[self.set_of(p)]


def contains(group: PointGroup, product_character, target) -> bool:
    """Whether a (reducible) representation shares an irreducible component
    with ``target``: :math:`\\frac{1}{|G|}\\sum_R \\chi(R)\\,\\chi_t(R)^* > 1/2`."""
    overlap = np.real(np.vdot(np.asarray(target), np.asarray(
        product_character))) / group.order
    return overlap > 0.5


def operation_matrices(functions, group: PointGroup, center,
                       n_points: int = 400, seed: int = 0):
    """``D(R)`` of every operation in the basis ``functions`` (least squares).

    Points are drawn around every function's center, so the fit sees each
    function where it is not negligible; the same points serve every
    operation.
    """
    rng = np.random.default_rng(seed)
    centers = np.unique(np.round(np.array(
        [np.asarray(f.center, dtype=float) for f in functions]), 8), axis=0)
    points = np.concatenate([c + rng.normal(scale=1.5, size=(n_points, 3))
                             for c in centers])

    def sample(p):
        return np.column_stack([f.evaluate(p[:, 0], p[:, 1], p[:, 2])
                                for f in functions])

    phi = sample(points)
    matrices = []
    for op in group.operations:
        moved = (points - center) @ op.matrix + center   # R^{-1} r
        fit, *_ = np.linalg.lstsq(phi, sample(moved), rcond=None)
        matrices.append(fit)
    return matrices


def orbital_symmetry(group: PointGroup, ao_matrices, overlap,
                     coefficients) -> OrbitalSymmetry:
    """Degenerate sets, characters and labels of the orbitals ``coefficients``.

    ``coefficients`` are the orbitals' columns in the atomic-orbital basis,
    orthonormal in ``overlap``; ``ao_matrices`` come from
    :func:`operation_matrices` in the same basis and group.
    """
    C = np.asarray(coefficients)
    S = np.asarray(overlap)
    U = [C.conj().T @ S @ D @ C for D in ao_matrices]
    K = C.shape[1]
    parent = list(range(K))

    def root(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    mixing = np.max(np.abs(np.array(U)), axis=0)
    for p in range(K):
        for q in range(p + 1, K):
            if mixing[p, q] > MIXING_THRESHOLD:
                parent[root(p)] = root(q)
    groups: dict[int, list[int]] = {}
    for p in range(K):
        groups.setdefault(root(p), []).append(p)
    sets = tuple(tuple(members) for members in
                 sorted(groups.values(), key=lambda m: m[0]))
    characters = tuple(
        np.array([np.real(np.trace(u[np.ix_(m, m)])) for u in U])
        for m in sets)
    labels = tuple(mulliken_label(group, chi) for chi in characters)
    return OrbitalSymmetry(group, sets, characters, labels)


# --------------------------------------------------------------------------- #
# Mulliken labels
# --------------------------------------------------------------------------- #

_DIMENSION_LETTERS = {1: "a", 2: "e", 3: "t", 4: "g", 5: "h"}
_LINEAR_LETTERS = {0: "sigma", 1: "pi", 2: "delta", 3: "phi", 4: "gamma"}


def _value(group, chi, test) -> float | None:
    """The character on the first operation passing ``test``."""
    for op, value in zip(group.operations, chi):
        if test(op):
            return float(value)
    return None


def _parallel(op, axis) -> bool:
    return (op.axis is not None and axis is not None
            and abs(abs(op.axis @ axis) - 1.0) < AXIS_TOLERANCE)


def mulliken_label(group: PointGroup, chi) -> str:
    """Lower-case Mulliken label of an irreducible character, or ``"?"``."""
    chi = np.asarray(chi, dtype=float)
    norm = float(chi @ chi) / group.order
    dim = int(round(chi[0]))
    if abs(norm - 1.0) > CHARACTER_TOLERANCE or dim < 1:
        return "?"
    z, y = group.principal, group.secondary
    has_i = any(op.kind == "i" for op in group.operations)
    parity = ""
    if has_i:
        parity = "g" if _value(group, chi, lambda op: op.kind == "i") > 0 \
            else "u"

    if group.linear:
        if dim == 2:
            # Character of the 2 pi / n rotation: 2 cos(2 pi k / n).
            c = _value(group, chi, lambda op: op.kind == "C"
                       and op.order == LINEAR_ORDER)
            k = int(round(np.degrees(np.arccos(np.clip(c / 2, -1, 1)))
                          / (360.0 / LINEAR_ORDER)))
            return _LINEAR_LETTERS.get(k, "?") + parity
        sv = _value(group, chi, lambda op: op.kind == "sigma"
                    and abs(op.axis @ z) < AXIS_TOLERANCE)
        return "sigma" + ("+" if sv > 0 else "-") + parity

    if group.name in ("C1", "Cs", "Ci"):
        prime = ""
        if group.name == "Cs":
            prime = "'" if chi[1] > 0 else "''"
        return "a" + parity + prime

    letter = _DIMENSION_LETTERS.get(dim, "?")
    subscript = ""
    if group.cubic:
        if dim == 1:
            letter = "a"
        if dim in (1, 3):
            for kind, order in (("C", 4), ("S", 4), ("C", 5)):
                v = _value(group, chi, lambda op, k=kind, o=order:
                           op.kind == k and op.order == o)
                if v is not None:
                    subscript = "1" if v / dim > 0 else "2"
                    break
        return letter + subscript + parity
    n = max((op.order for op in group.operations if op.kind == "C"),
            default=1)
    cn = _value(group, chi, lambda op: op.kind == "C" and op.order == n
                and _parallel(op, z))
    if dim == 1:
        letter = "a" if cn is None or cn > 0 else "b"
        if group.name.startswith("D") and n == 2:
            # D2-type: a is symmetric under all three C2; b1, b2, b3 under
            # C2(z), C2(y), C2(x) only.
            x = np.cross(y, z)
            signs = [_value(group, chi, lambda op, a=axis:
                            op.kind == "C" and _parallel(op, a)) > 0
                     for axis in (z, y, x)]
            letter = "a" if all(signs) else "b"
            if letter == "b":
                subscript = str(1 + signs.index(True))
        else:
            v = None
            if y is not None:
                v = _value(group, chi, lambda op: op.kind == "C"
                           and op.order == 2 and _parallel(op, y))
                if v is None:
                    # sigma_v(xz): its normal is y.
                    v = _value(group, chi, lambda op: op.kind == "sigma"
                               and _parallel(op, y))
            if v is None and z is not None:
                # Any vertical mirror: for n > 2 they are one class.
                v = _value(group, chi, lambda op: op.kind == "sigma"
                           and abs(op.axis @ z) < AXIS_TOLERANCE)
            if v is not None:
                subscript = "1" if v > 0 else "2"
    elif dim == 2 and n >= 5:
        k = int(round(np.degrees(np.arccos(np.clip(cn / 2, -1, 1)))
                      / (360.0 / n)))
        subscript = str(k)
    prime = ""
    if not has_i:
        sh = _value(group, chi, lambda op: op.kind == "sigma"
                    and _parallel(op, z))
        if sh is not None:
            prime = "'" if sh > 0 else "''"
    return letter + subscript + parity + prime
