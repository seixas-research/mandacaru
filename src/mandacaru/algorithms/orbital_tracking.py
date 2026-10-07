# -*- coding: utf-8 -*-
# file: algorithms/orbital_tracking.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Molecular orbitals followed from one geometry to the next.

A Markov-chain ansatz search with ``transfer=True`` starts each geometry from
the ansatz the previous geometry reported.  Its operators are named by
spin-orbital indices and its angles multiply generators built from those
orbitals, but the orbitals come out of an independent mean-field calculation
at each geometry: the eigensolver fixes the sign of a real orbital
arbitrarily, and two orbitals whose energies cross change places.  An
excitation ``S(0->2)`` at one geometry is then a different excitation at the
next, and a carried angle can have the wrong sign.

This module measures the correspondence and rewrites the ansatz in the new
orbitals:

1. :func:`orbital_overlap` -- the overlap of the two geometries' molecular
   orbitals,

   .. math::

       O_{pq} = \langle \phi_p(X_n) | \phi_q(X_{n+1}) \rangle
              = A_n^\dagger \, \langle\chi^{(n)}|\chi^{(n+1)}\rangle \, A_{n+1},

   with :math:`A = S^{-1/2} V` the atomic-orbital coefficients of the
   orbitals the Hamiltonian was built in and the cross overlap of the two
   bases integrated on the new geometry's grid (the previous basis functions
   are sampled there, centered on the previous nuclei).  Each column and row
   is normalized by the orbital's norm in the bare grid metric, so
   :math:`|O_{pq}| \le 1`.  With an overlap-corrected basis (PAW-LCAO) the
   augmentation is left out of the cross overlap: the normalized magnitudes
   still rank the matches, but they are a pseudo-orbital measure.
2. :func:`match_orbitals` -- maximum-overlap matching (an assignment problem
   on :math:`|O|`), **inside** each occupation block of the reference
   (doubly occupied, singly occupied, empty), and the sign of every matched
   overlap.  An occupied orbital whose best partner is now empty is an
   electronic reorganization, not a relabeling; the matched overlaps then
   drop, and :attr:`OrbitalMatch.confidence` -- the smallest matched
   :math:`|O_{p\pi(p)}|` -- says so.
3. :func:`transfer_ansatz` -- the old operators renamed through the matching
   and their angles multiplied by the orbital signs: with
   :math:`\phi^{\rm new}_{\pi(p)} \approx s\,\phi^{\rm old}_p`, the old
   ladder operator is :math:`a_p = s\,a'_{\pi(p)}`, so an excitation picks up
   the product of the signs of its indices.

The renaming is exact for fermionic excitations, which a relabeling of the
orbitals maps onto fermionic excitations.  A qubit excitation drops the
Jordan-Wigner strings, and those depend on the order of the orbitals, so after
a non-trivial permutation it is a good start rather than the same state.  A
Pauli-string or coupled-exchange operator is carried only when the matching
is the identity (signs included, for the coupled ones).

A degenerate set has no orbitals of its own, only a span: any rotation inside
it is as good an eigenbasis, and the eigensolver picks one by round-off.
LiH's pi pair of MP2 natural orbitals came out rotated by 40 degrees between
1.595 and 1.580 A (matched overlaps 0.76, the transfer refused) and by 5
degrees with other last bits.  Matching cannot follow that, so the rotation
is removed where the orbitals are made: :func:`degenerate_gauge` fixes the
orbitals inside every degenerate cluster of the active-space selector's
natural orbitals as the eigenvectors of one fixed real-space operator
(:func:`orbital_gauge`), signs included.  Those vary smoothly with the
geometry, and the gauge changes no physics: the cluster's span, every
occupation and the reference determinant are the same.

Not done: the canonical orbitals of an untruncated or energy-ranked space
keep the eigensolver's gauge, so a degenerate set there still shows as a low
confidence (the chain then rebuilds).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

#: Smallest matched orbital overlap at which an ansatz is still carried over.
DEFAULT_TRANSFER_THRESHOLD = 0.9

_EXCITATION = re.compile(
    r"^(?P<kind>Q?)(?P<rank>[SD])\((?P<occ>[\d,]+)->(?P<virt>[\d,]+)\)$")


@dataclass
class OrbitalMatch:
    """How the previous geometry's active orbitals map onto this one's.

    ``permutation[p]`` is the new index of old active orbital ``p``,
    ``signs[p]`` the sign of their overlap and ``overlaps[p]`` its magnitude.
    """

    permutation: np.ndarray
    signs: np.ndarray
    overlaps: np.ndarray

    @property
    def confidence(self) -> float:
        """The smallest matched overlap magnitude (``1`` for no orbitals)."""
        return float(self.overlaps.min()) if self.overlaps.size else 1.0

    @property
    def is_identity(self) -> bool:
        """No orbital moved and none changed sign."""
        return bool(np.all(self.permutation == np.arange(len(self.permutation)))
                    and np.all(self.signs > 0))

    def describe(self) -> str:
        moved = int(np.count_nonzero(
            self.permutation != np.arange(len(self.permutation))))
        flipped = int(np.count_nonzero(self.signs < 0))
        return (f"orbitals tracked: {moved} reordered, {flipped} sign(s) "
                f"aligned, smallest overlap {self.confidence:.3f}")


@dataclass
class OrbitalSnapshot:
    """What one geometry's orbitals are: enough to overlap them with the next.

    ``coefficients`` are the atomic-orbital coefficients of the **active**
    orbitals (columns), ``basis`` the basis functions they expand in.
    """

    basis: list
    coefficients: np.ndarray
    norms: np.ndarray

    @classmethod
    def from_integrals(cls, integrals, n_active: int):
        """The snapshot of ``integrals``' molecular orbitals, or the reason
        there is none (a string)."""
        if integrals is None or getattr(integrals, "mo_coefficients",
                                        None) is None:
            return "no molecular orbitals (a Hamiltonian given directly)"
        if getattr(integrals, "spinor_basis", False):
            return "spinor orbitals (spin-orbit) are not tracked"
        if getattr(integrals, "periodic", False):
            return "periodic orbitals are not tracked"
        space = getattr(integrals, "active_space", None)
        active = (list(space.active) if space is not None
                  else list(range(integrals.n_orbitals)))
        if len(active) != n_active:
            return (f"{len(active)} active orbitals for a register of "
                    f"{n_active}")
        A = integrals._lowdin_x() @ np.asarray(integrals.mo_coefficients,
                                               dtype=complex)[:, active]
        S = integrals.bare_overlap()
        norms = np.sqrt(np.maximum(
            np.real(np.einsum("mp,mn,np->p", A.conj(), S, A)), 1e-300))
        return cls(basis=list(integrals.basis), coefficients=A, norms=norms)


#: Relative spread of the ranking values (natural occupations) inside which
#: orbitals form one degenerate cluster for :func:`degenerate_gauge`.  The
#: grid keeps a symmetry-degenerate pair degenerate to round-off (LiH's pi
#: pair: 12 digits along a grid axis), and breaks it far above this when it
#: does break it (6e-4 for LiH tilted off the axes at h 0.3).
DEGENERATE_SPREAD = 1e-6
#: Magnitude below which a value is round-off around zero (an empty natural
#: orbital of a single determinant), not a ranking.
ZERO_VALUE = 1e-10

#: The fixed operator :func:`orbital_gauge` diagonalizes inside a cluster:
#: :math:`q(\mathbf u) = \mathbf u^T Q\,\mathbf u + \mathbf b\cdot\mathbf u`
#: with :math:`\mathbf u = \mathbf r - \bar{\mathbf R}` (Bohr; the
#: centroid of the nuclei).  Generic on purpose: three distinct principal
#: values along axes that are no symmetry axis a molecule is likely to have,
#: so its restriction to a degenerate set has distinct eigenvalues.
GAUGE_QUADRATIC = np.array([[1.000, 0.137, 0.271],
                            [0.137, 0.618, 0.089],
                            [0.271, 0.089, 0.382]])
GAUGE_LINEAR = np.array([0.0731, 0.0457, 0.0293])
#: The sign rule's weight :math:`g(\mathbf u) = 1 + \mathbf c\cdot\mathbf u
#: + q(\mathbf u) + (\mathbf d\cdot\mathbf u)^3`: an orbital of angular
#: momentum up to 3 overlaps it, and the sign makes that overlap positive.
SIGN_LINEAR = np.array([0.0613, 0.0389, 0.0521])
SIGN_CUBIC = np.array([0.0577, 0.0493, 0.0651])


def degenerate_clusters(values, indices,
                        spread: float = DEGENERATE_SPREAD) -> list[list[int]]:
    """The clusters (two or more members) of ``indices`` whose ``values``
    agree to ``spread``, relative; each a chain of neighbors in value.  A
    value below :data:`ZERO_VALUE` carries no ranking and joins none."""
    values = np.asarray(values, dtype=float)
    order = sorted((int(p) for p in indices
                    if abs(values[int(p)]) >= ZERO_VALUE),
                   key=lambda p: values[p])
    clusters, run = [], order[:1]
    for p in order[1:]:
        a, b = values[run[-1]], values[p]
        if abs(b - a) <= spread * max(abs(a), abs(b)):
            run.append(p)
        else:
            if len(run) > 1:
                clusters.append(sorted(run))
            run = [p]
    if len(run) > 1:
        clusters.append(sorted(run))
    return clusters


def orbital_gauge(integrals, orbitals):
    r"""``columns -> (q, w)`` for the orbitals ``orbitals @ columns``.

    ``orbitals`` are the molecular orbitals over the orthonormal basis the
    Hamiltonian is built in (``integrals.mo_coefficients``) and ``columns``
    a ``(M, k)`` block of a rotation of them.  ``q`` is the ``(k, k)`` matrix
    of the gauge operator (:data:`GAUGE_QUADRATIC`), ``w`` the ``k`` overlaps
    with the sign weight (:data:`SIGN_CUBIC`), both on the grid with the
    pseudo-orbitals (the augmentation left out: the gauge needs a fixed
    operator, not a physical moment).  ``None`` when the integrals have no
    real-space orbitals to sample.
    """
    psi = getattr(getattr(integrals, "_engine", None), "_psi", None)
    if (orbitals is None or psi is None
            or getattr(integrals, "periodic", False)
            or getattr(integrals, "spinor_basis", False)):
        return None
    grid = integrals.grid
    cache = {}

    def prepare():
        # On the first cluster only: most selections have none.
        X = (integrals._lowdin_x() if getattr(integrals, "orthogonalize", True)
             else np.eye(psi.shape[0]))
        # The Bohr frame of the grid (integrals.nuclei may be Angstrom).
        nuclei = integrals._potentials.nuclei
        center = np.mean([np.asarray(c, dtype=float) for _z, c in nuclei],
                         axis=0)
        u = np.stack([np.asarray(c, dtype=float).reshape(-1) - x0
                      for c, x0 in zip((grid.X, grid.Y, grid.Z), center)])
        q = (np.einsum("ig,ij,jg->g", u, GAUGE_QUADRATIC, u)
             + GAUGE_LINEAR @ u)
        cache.update(A=X @ np.asarray(orbitals), q=q,
                     w=1.0 + SIGN_LINEAR @ u + q + (SIGN_CUBIC @ u) ** 3)

    def evaluate(columns):
        if not cache:
            prepare()
        phi = (cache["A"] @ np.asarray(columns)).T @ psi      # (k, G)
        block = np.real((np.conj(phi) * cache["q"]) @ phi.T) * grid.dV
        return (0.5 * (block + block.T),
                np.real(phi @ cache["w"]) * grid.dV)

    return evaluate


def degenerate_gauge(rotation, values, blocks, gauge,
                     spread: float = DEGENERATE_SPREAD):
    """``rotation`` with every degenerate cluster's orbitals fixed.

    ``rotation`` (``None``: the identity) defines orbitals whose ``values``
    (natural occupations) rank them; inside each of ``blocks`` (index
    lists that must not mix: active occupied, virtual), orbitals whose values
    agree to ``spread`` are rotated among themselves into the eigenvectors of
    the gauge operator (``gauge``, from :func:`orbital_gauge`), each with the
    sign that makes its sign weight positive.  A rotation inside a degenerate
    cluster leaves its span, its values and the reference determinant as they
    were, so only the arbitrary choice of basis inside the span changes.
    Returns ``rotation`` itself when there is nothing to fix.
    """
    if gauge is None or values is None:
        return rotation
    clusters = [c for block in blocks
                for c in degenerate_clusters(values, block, spread)]
    if not clusters:
        return rotation
    n = len(values)
    R = (np.eye(n) if rotation is None
         else np.array(rotation, dtype=float, copy=True))
    for cluster in clusters:
        q, _w = gauge(R[:, cluster])
        _, U = np.linalg.eigh(q)
        R[:, cluster] = R[:, cluster] @ U
        _q, w = gauge(R[:, cluster])
        R[:, cluster] *= np.where(w < 0.0, -1.0, 1.0)
    return R


def orbital_overlap(previous: OrbitalSnapshot, current: OrbitalSnapshot,
                    grid) -> np.ndarray:
    r"""Normalized :math:`O_{pq} = \langle\phi_p^{\rm prev}|\phi_q\rangle`,
    integrated on ``grid`` (the current geometry's)."""
    old = np.stack([fn.sample(grid) for fn in previous.basis])
    new = np.stack([fn.sample(grid) for fn in current.basis])
    cross = (old.conj() @ new.T) * grid.dV
    O = previous.coefficients.conj().T @ cross @ current.coefficients
    return O / np.outer(previous.norms, current.norms)


def occupation_blocks(n_orbitals: int, num_particles) -> list[range]:
    """The reference's doubly occupied, singly occupied and empty orbitals
    (active indices; the occupied ones come first)."""
    na, nb = (int(n) for n in num_particles)
    edges = [0, min(na, nb), max(na, nb), int(n_orbitals)]
    return [range(lo, hi) for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo]


def match_orbitals(overlap, blocks) -> OrbitalMatch:
    """Maximum-overlap matching of old onto new orbitals, block by block."""
    from scipy.optimize import linear_sum_assignment

    O = np.asarray(overlap)
    n = O.shape[0]
    permutation = np.arange(n)
    for block in blocks:
        index = np.asarray(block)
        rows, cols = linear_sum_assignment(-np.abs(O[np.ix_(index, index)]))
        permutation[index[rows]] = index[cols]
    matched = O[np.arange(n), permutation]
    signs = np.where(np.real(matched) < 0.0, -1, 1)
    return OrbitalMatch(permutation=permutation, signs=signs,
                        overlaps=np.abs(matched))


def _rename_excitation(label: str, match: OrbitalMatch, n_orbitals: int,
                       pool_labels):
    """The excitation ``label`` in the new orbitals and the sign its angle
    takes, or ``None`` when the renamed excitation is not in the pool."""
    found = _EXCITATION.match(label)
    if found is None:
        return None
    qubit = bool(found["kind"])
    occ = [int(x) for x in found["occ"].split(",")]
    virt = [int(x) for x in found["virt"].split(",")]
    sign = 1

    def rename(P):
        nonlocal sign
        p, spin = P % n_orbitals, P // n_orbitals
        sign *= int(match.signs[p])
        return int(match.permutation[p]) + spin * n_orbitals

    occ, virt = [rename(P) for P in occ], [rename(P) for P in virt]
    # The pool names each excitation with its indices ascending.  Reordering
    # two fermionic ladder operators changes the sign; qubit ones commute.
    for indices in (occ, virt):
        if len(indices) == 2 and indices[0] > indices[1]:
            indices.reverse()
            if not qubit:
                sign = -sign
    prefix = ("Q" if qubit else "") + found["rank"]
    new = (f"{prefix}({','.join(map(str, occ))}->"
           f"{','.join(map(str, virt))})")
    return (new, sign) if new in pool_labels else None


def transfer_ansatz(operators, angles, match: OrbitalMatch, n_orbitals: int,
                    pool_labels):
    """``(operators, angles)`` rewritten in the new orbitals, or the reason
    they cannot be (a string)."""
    pool_labels = set(pool_labels)
    angles = np.asarray(angles, dtype=float).copy()
    if match.is_identity:
        return list(operators), angles
    renamed = []
    for k, label in enumerate(operators):
        result = _rename_excitation(label, match, n_orbitals, pool_labels)
        if result is None:
            return (f"operator {label} cannot be followed through the "
                    f"orbital matching")
        renamed.append(result[0])
        angles[k] *= result[1]
    return renamed, angles


def carry_ansatz(operators, angles, previous_orbitals, integrals,
                 num_particles, pool_labels,
                 threshold: float = DEFAULT_TRANSFER_THRESHOLD):
    """The previous geometry's ansatz in this geometry's orbitals.

    Returns ``(operators, angles, description)`` -- the operator labels
    renamed and the angles' signs fixed through the orbital matching
    (:func:`transfer_ansatz`), and a line saying how -- or ``(None, None,
    reason)`` when the ansatz cannot be carried: the orbitals of either
    geometry are not available, the best match of some orbital falls below
    ``threshold``, or an operator cannot be followed through the matching.
    ``previous_orbitals`` is the previous geometry's :class:`OrbitalSnapshot`
    (or the reason it has none).
    """
    current = OrbitalSnapshot.from_integrals(
        integrals, int(previous_orbitals.coefficients.shape[1])
        if isinstance(previous_orbitals, OrbitalSnapshot) else 0)
    if isinstance(previous_orbitals, str) or isinstance(current, str):
        reason = (previous_orbitals if isinstance(previous_orbitals, str)
                  else current)
        return None, None, f"orbitals not tracked: {reason}"
    overlap = orbital_overlap(previous_orbitals, current, integrals.grid)
    n_orbitals = len(overlap)
    match = match_orbitals(overlap, occupation_blocks(n_orbitals,
                                                      num_particles))
    if match.confidence < threshold:
        return None, None, (f"smallest matched orbital overlap "
                            f"{match.confidence:.3f} < transfer_threshold "
                            f"{threshold:g}")
    moved = transfer_ansatz(operators, angles, match, n_orbitals,
                            pool_labels)
    if isinstance(moved, str):
        return None, None, moved
    return moved[0], moved[1], match.describe()
