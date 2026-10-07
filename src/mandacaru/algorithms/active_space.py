# -*- coding: utf-8 -*-
# file: algorithms/active_space.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Which spatial orbitals go on the register, and which are spent classically.

A basis set buys accuracy with virtual orbitals and a qubit register pays for
them at two qubits each.  The two are not the same currency, and this module is
where they are exchanged.  Water in PAW-LCAO-TZP has 29 spatial orbitals -- 4
occupied, 25 virtual -- which is 56 qubits under the parity reduction; the same
molecule in PAW-LCAO-SZ has 10.  The TZP number is unreachable and the SZ
number is a poor description of the molecule, and the way out of that is not a
third basis but a **large basis with a small active space**: the basis-set
quality lives in the *shape* of the orbitals, the correlation lives in the few
of them the wavefunction actually mixes.

The three pieces
----------------

Every spatial molecular orbital ends up in exactly one of three places.

**Frozen** orbitals are doubly occupied and are replaced by their mean field:
a constant core energy plus an effective one-body potential on what is left
(:func:`~mandacaru.core.hamiltonian.freeze_core_integrals`).  This is the
frozen-core approximation, and it is what the ``active_space`` option's
``"frozen"`` does.

**Deleted** orbitals are virtual and are simply dropped.  Nothing has to be
folded in: an orbital that is empty in the reference contributes neither to the
core energy nor to the effective potential, so deleting it costs exactly the
correlation it would have carried and nothing else.  This is the new half, and
it is the half that reduces the register.

**Active** orbitals are the rest -- the ones the qubits represent.

Occupied and virtual are ranked differently, on purpose
-------------------------------------------------------

The ``active_space`` option's ``"method"`` ranks the **virtual** orbitals
only.  Occupied truncation
is always by orbital energy, which is to say it is always the chemical core,
because an index-based core (``"frozen": "auto"`` resolves to "the lowest so
many MOs") stops meaning the core the moment the occupied orbitals are
reordered, and because removing an occupied orbital is a far coarser
approximation than removing a virtual one -- it belongs to an explicit request,
not to a selector.  See :func:`~mandacaru.algorithms.mp2.mp2_natural_orbitals`,
which declines to rotate the occupied block for the same reason.

Once the core is settled, ``"mp2"`` does rotate the **active** doubly occupied
orbitals among themselves into natural orbitals of the same density.  That
leaves the reference determinant and its energy unchanged and touches no
frozen index, and it makes every active orbital's occupation an eigenvalue of
the density.  Frozen orbitals keep their canonical form, so their occupations
are diagonal elements.

For the virtuals there are four rankings:

``"energy"``
    Canonical order, lowest orbital energy first.  Free, and it asks "which
    orbital is cheapest to excite into".
``"mp2"``
    The eigenvalues of the second-order density's virtual block -- the frozen
    natural orbitals.  This asks "which orbital does the correlated
    wavefunction actually occupy", which is the question an active space is
    asking, and it answers it in a *rotated* virtual basis, so a few orbitals
    can gather up correlation that canonical ordering leaves spread thinly over
    many.  Costs one MP2 calculation in the full virtual space.  An open shell
    uses :func:`~mandacaru.algorithms.mp2.open_shell_mp2_natural_orbitals`,
    which ranks only the orbitals empty in both spins.
``"dlpno-mp2"``
    The same natural orbitals from **local** MP2
    (:mod:`~mandacaru.algorithms.dlpno_mp2`): localized occupied orbitals,
    projected-atomic-orbital domains and pair natural orbitals, built
    **integral-direct** -- the two-body tensor of the basis is never formed,
    which is what lets a large basis through at all (benzene in
    PAW-LCAO-DZP: RHF in 4 minutes, where the tensor did not finish in 25).
    Equal to ``"mp2"`` in the limit of full domains and a zero PNO cutoff;
    at the defaults it recovers about 99.99% of the MP2 correlation energy.
    Closed shell; the requested ``"frozen"`` core is not correlated.
``"natural"``
    The occupations of the **reference** natural orbitals.  Meaningful only for
    an open-shell (UHF) reference.  For a closed-shell RHF reference the density
    is idempotent, its eigenvalues are exactly 2 and 0, and every ordering of
    the virtuals is as good as any other -- so this is refused by name for a
    closed shell rather than quietly returning the energy ordering under a label
    that claims to be something else.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

#: Recognized values of the ``active_space`` option's ``"method"``.
ACTIVE_SPACE_METHODS = ("energy", "mp2", "dlpno-mp2", "natural")
#: Keys the ``active_space`` option accepts.
ACTIVE_SPACE_KEYS = ("method", "orbitals", "threshold", "frozen",
                     "correlating_pairs", "symmetry")

#: Occupation below which a virtual natural orbital is dropped by
#: ``active_space={..., "threshold": True}``.  Measured MP2 spectra put the useful range between
#: 1e-3 and 1e-4: on LiH / PAW-LCAO-TZP, 1e-3 keeps 4 of 11 virtuals and 94.8 %
#: of the promoted charge while 1e-4 keeps 8 and 99.88 %; on H2O / PAW-LCAO-DZP,
#: 1e-3 keeps 9 of 19 and 96.1 %.  1e-3 is the default because it is the knee of
#: both curves -- the point past which orbitals cost two qubits each and return
#: a few parts per thousand of the correlation.
DEFAULT_OCCUPATION_THRESHOLD = 1e-3

#: Largest deviation from orthogonality accepted in a selector's rotation.
ROTATION_TOLERANCE = 1e-8


@dataclass(frozen=True)
class ActiveSpace:
    """The resolved partition of the spatial molecular orbitals.

    Attributes
    ----------
    n_orbitals : int
        Total number of spatial MOs the partition covers.
    frozen : tuple of int
        Doubly occupied spatial MOs folded into the mean field.
    active : tuple of int
        Spatial MOs the register represents, sorted.  Occupied ones come first
        because occupied MO indices are below virtual ones, which is what
        :func:`~mandacaru.algorithms.volumetric.reference_occupations` relies on
        when it fills the reference determinant.
    deleted : tuple of int
        Virtual spatial MOs dropped from the problem.
    method : str
        Which ranking chose the virtuals (:data:`ACTIVE_SPACE_METHODS`).
    rotation : ndarray or None
        ``(M, M)`` orthogonal matrix taking the incoming MO basis to the one
        ``active`` / ``deleted`` index, or ``None`` when the basis is unchanged.
        Always block diagonal in occupied / virtual.
    occupations : ndarray or None
        The selector's occupation number per MO of the rotated basis, or
        ``None`` for ``"energy"`` (which ranks by orbital energy, not by an
        occupation).
    correlation_energy : float or None
        MP2 correlation energy in Hartree over the **full** virtual space, when
        the MP2 selector ran.  It is what the truncation is measured against:
        the same quantity recomputed in the active space alone says how much of
        it survived.
    n_doubly, n_singly : int
        How many of the reference's spatial MOs are doubly and singly
        occupied, frozen ones included: indices below ``n_doubly`` are doubly
        occupied, the next ``n_singly`` singly, the rest virtual.  ``None``
        when the partition was built without a reference.
    partners : tuple of (int, int)
        ``(occupied, virtual)`` correlating pairs, when ``correlating_pairs``
        was asked for (:func:`correlating_partners`).
    notes : tuple of (str, str)
        ``(constraint, what it did to the selection)``, for the run log.
    point_group : str or None
        Schoenflies name of the molecule's point group, when ``symmetry`` was
        asked for.
    irreps : tuple of str
        Each orbital's symmetry label (``"3a1"``), numbered within its
        irreducible representation; ``"?"`` where the grid broke the symmetry
        too much to tell.
    """

    n_orbitals: int
    frozen: tuple[int, ...] = ()
    active: tuple[int, ...] = ()
    deleted: tuple[int, ...] = ()
    method: str = "energy"
    rotation: np.ndarray | None = None
    occupations: np.ndarray | None = field(default=None, repr=False)
    correlation_energy: float | None = None
    n_doubly: int | None = None
    n_singly: int = 0
    partners: tuple[tuple[int, int], ...] = ()
    notes: tuple[tuple[str, str], ...] = ()
    point_group: str | None = None
    irreps: tuple[str, ...] = ()

    @property
    def truncated(self) -> bool:
        """True when virtual orbitals were deleted.

        This is the flag that matters downstream: a frozen core alone leaves
        ``active`` the complement of ``frozen``, which is what every existing
        density and gradient path assumes.
        """
        return bool(self.deleted)

    @property
    def n_active(self) -> int:
        """Number of active spatial orbitals."""
        return len(self.active)

    def summary(self) -> str:
        """One line for the run log."""
        parts = [f"{self.n_active} active"]
        if self.frozen:
            parts.append(f"{len(self.frozen)} frozen")
        if self.deleted:
            parts.append(f"{len(self.deleted)} deleted")
        line = f"{', '.join(parts)} of {self.n_orbitals} spatial orbitals"
        if self.truncated:
            line += f" (virtuals ranked by {self.method})"
        return line

    def occupancy(self, p: int) -> str | None:
        """``"occupied"``, ``"singly"`` or ``"virtual"`` in the reference."""
        if self.n_doubly is None:
            return None
        if p < self.n_doubly:
            return "occupied"
        return "singly" if p < self.n_doubly + self.n_singly else "virtual"

    def partner(self, p: int) -> int | None:
        """The correlating partner of orbital ``p``, if it has one."""
        for i, a in self.partners:
            if p == i:
                return a
            if p == a:
                return i
        return None

    def role(self, p: int) -> str:
        """``"frozen"``, ``"active"`` or ``"deleted"``."""
        if p in self.frozen:
            return "frozen"
        return "deleted" if p in self.deleted else "active"


@dataclass(frozen=True)
class ActiveSpaceSpec:
    """A resolved ``active_space`` option.

    Attributes
    ----------
    method : str
        How the virtual orbitals are ranked (:data:`ACTIVE_SPACE_METHODS`).
    orbitals : None, int, dict or tuple of int
        The register's size (:func:`normalize_orbitals`); ``None`` when the
        threshold alone decides.
    threshold : float or None
        Occupation a virtual natural orbital must reach to stay active.
    frozen : None, "auto", int or tuple of int
        The frozen core (:func:`normalize_frozen`): ``"auto"`` for the
        chemical (noble-gas) core, a count of the lowest molecular orbitals,
        or explicit spatial orbital indices.  The builder resolves it to
        indices once the atoms and the orbitals exist.
    correlating_pairs : bool
        Keep every active doubly occupied orbital's correlating partner
        active, and no partner of an occupied orbital the count froze; a
        singly occupied orbital has no electron pair and gets no partner
        (:func:`correlating_partners`).
    symmetry : bool
        Keep degenerate sets whole and the irreducible representations the
        target states need (:mod:`mandacaru.algorithms.orbital_symmetry`).
    states : int
        How many states the solver targets (its ``num_states``; 1 for a
        ground-state method).  Set by the solver, not an ``active_space`` key.
    """

    method: str = "energy"
    orbitals: int | dict | tuple | None = None
    threshold: float | None = None
    frozen: str | int | tuple | None = None
    correlating_pairs: bool = False
    symmetry: bool = False
    states: int = 1

    @property
    def truncates(self) -> bool:
        """Whether virtual orbitals may be deleted (a count or a threshold)."""
        return self.orbitals is not None or self.threshold is not None

    def as_dict(self) -> dict:
        """The option as it was resolved, JSON-writable."""
        def plain(value):
            return list(value) if isinstance(value, tuple) else value
        return {"method": self.method, "orbitals": plain(self.orbitals),
                "threshold": self.threshold, "frozen": plain(self.frozen),
                "correlating_pairs": self.correlating_pairs,
                "symmetry": self.symmetry}


def resolve_active_space_spec(spec) -> ActiveSpaceSpec | None:
    """Validate the ``active_space`` option, or raise.

    ``None`` keeps every orbital.  Otherwise a dictionary with keys from
    :data:`ACTIVE_SPACE_KEYS`: ``"method"`` (default ``"energy"``) ranks the
    virtual orbitals, ``"orbitals"`` sizes the register, ``"threshold"``
    keeps the virtuals whose occupation reaches it and ``"frozen"`` folds a
    core into the mean field.  At least one of the last three is needed,
    since a ranking alone changes nothing.  Checked when the calculator is
    constructed, before any integral.
    """
    if spec is None:
        return None
    if isinstance(spec, ActiveSpaceSpec):
        return spec
    if not isinstance(spec, dict):
        raise TypeError(
            "active_space must be a dict such as {'method': 'mp2', "
            f"'orbitals': 8}}, not {type(spec).__name__}")
    unknown = sorted(set(spec) - set(ACTIVE_SPACE_KEYS))
    if unknown:
        raise ValueError(f"unknown active_space key(s) {unknown}; it takes "
                         f"{list(ACTIVE_SPACE_KEYS)}")
    method = resolve_method(spec.get("method"))
    orbitals = normalize_orbitals(spec.get("orbitals"))
    threshold = resolve_threshold(spec.get("threshold"))
    frozen = normalize_frozen(spec.get("frozen"))
    flags = {}
    for key in ("correlating_pairs", "symmetry"):
        value = spec.get(key, False)
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"the active_space {key!r} is a bool, not "
                            f"{type(value).__name__}")
        flags[key] = bool(value)
    if orbitals is None and threshold is None and frozen is None:
        raise ValueError(
            "active_space needs 'orbitals', 'threshold' or 'frozen': a method "
            "only ranks the virtual orbitals, it does not say which to keep.  "
            "Omit active_space to keep every orbital.")
    if threshold is not None and method == "energy":
        raise ValueError(
            f"an active_space threshold ({threshold:g}) needs occupation "
            f"numbers to compare against, and the 'energy' method ranks the "
            f"virtual orbitals by orbital energy instead -- it never computes "
            f"an occupation.  Use 'method': 'mp2' (or 'natural' for an "
            f"open-shell reference) with the threshold, or select by "
            f"'orbitals'.")
    return ActiveSpaceSpec(method=method, orbitals=orbitals,
                           threshold=threshold, frozen=frozen, **flags)


def normalize_frozen(spec):
    """Canonical form of the ``active_space`` ``"frozen"``, or raise.

    ``None`` / ``False`` / ``0`` freeze nothing; ``True`` and ``"auto"`` mean
    the chemical (noble-gas) core, resolved per atom by the builder; an
    ``int`` is that many lowest molecular orbitals; a list is explicit
    spatial orbital indices -- one meaning per type, as for ``"orbitals"``.
    """
    if spec is None or spec is False:
        return None
    if spec is True or (isinstance(spec, str)
                        and spec.strip().lower() == "auto"):
        return "auto"
    if isinstance(spec, (int, np.integer)):
        if int(spec) < 0:
            raise ValueError(f"the active_space 'frozen' count must be >= 0, "
                             f"got {int(spec)}")
        return int(spec) or None
    if isinstance(spec, (list, tuple, set, frozenset, np.ndarray)):
        indices = tuple(sorted({int(i) for i in np.asarray(list(spec)).ravel()}))
        return indices or None
    raise TypeError(
        f"unknown active_space 'frozen' {spec!r}; use 'auto' (the chemical "
        "core), a count of the lowest molecular orbitals, or a list of "
        "spatial orbital indices")


def resolve_method(method) -> str:
    """Normalize the ``active_space`` ``"method"``, or raise."""
    name = str(method if method is not None else "energy")
    name = name.strip().lower().replace("_", "-")
    if name not in ACTIVE_SPACE_METHODS:
        raise ValueError(
            f"unknown active_space method {method!r}; use one of "
            f"{ACTIVE_SPACE_METHODS}")
    return name


def resolve_threshold(threshold):
    """Normalize the ``active_space`` ``"threshold"`` to a float or ``None``.

    ``None`` / ``False`` means no threshold (the count decides), ``True`` takes
    :data:`DEFAULT_OCCUPATION_THRESHOLD`, and a number is used as given.  A
    threshold must be positive: zero would keep every virtual orbital, which is
    not a truncation, and a negative one is meaningless for an occupation.
    """
    if threshold is None or threshold is False:
        return None
    if threshold is True:
        return float(DEFAULT_OCCUPATION_THRESHOLD)
    value = float(threshold)
    if not value > 0.0:
        raise ValueError(
            f"the active_space threshold must be a positive occupation number, "
            f"got {threshold!r}.  Zero keeps every virtual orbital, which is "
            f"not a truncation; give 'orbitals' instead to select by count.")
    if value >= 2.0:
        raise ValueError(
            f"an active_space threshold of {value} is not an occupation "
            f"number: a natural "
            f"orbital's occupation lies in [0, 2], and a *virtual* one's is the "
            f"small charge correlation promotes into it -- of order 1e-2 at the "
            f"very most.  A threshold at or above 2 keeps nothing.")
    return value


def _count_above(occupations, threshold: float, method: str) -> int:
    """How many ranked virtual orbitals clear ``threshold``, or raise."""
    occupations = np.asarray(occupations, dtype=float)
    kept = int(np.count_nonzero(occupations >= threshold))
    if kept == 0:
        largest = float(np.max(occupations)) if occupations.size else 0.0
        raise ValueError(
            f"the active_space threshold {threshold:g} keeps no virtual orbital "
            f"at all: the largest {method} occupation in this problem is "
            f"{largest:.3e}.  An active space with no virtual orbital cannot "
            f"correlate anything -- it would return the Hartree-Fock energy "
            f"through a variational solver.  Virtual natural occupations are "
            f"small by construction (the charge second-order correlation "
            f"promotes, not an electron count), so the useful range is roughly "
            f"1e-3 to 1e-5; {DEFAULT_OCCUPATION_THRESHOLD:g} is the default.")
    return kept


def normalize_orbitals(spec):
    """Canonical, JSON-writable form of the ``active_space`` ``"orbitals"``,
    or raise.

    ``None`` stays ``None``, a count stays an ``int``, a dict becomes a plain
    ``{str: int}`` and any sequence becomes a sorted ``tuple`` of ``int``.  Done
    at construction so a malformed spec -- and, just as usefully, a numpy array
    that would not survive being written into a saved Hamiltonian's metadata --
    fails before the integrals rather than after them.
    """
    if spec is None:
        return None
    if isinstance(spec, dict):
        unknown = sorted(set(spec) - {"occupied", "virtual"})
        if unknown:
            raise ValueError(
                f"unknown key(s) {unknown} in the active_space 'orbitals'; its "
                "dict form takes 'occupied' and 'virtual'")
        return {str(k): int(v) for k, v in spec.items()}
    if isinstance(spec, bool):
        raise TypeError(
            "the active_space 'orbitals' is a number of spatial orbitals, not "
            "a flag; use an int, {'occupied': n, 'virtual': m} or a list of "
            "orbital indices")
    if isinstance(spec, (int, np.integer)):
        return int(spec)
    if isinstance(spec, (list, tuple, set, frozenset, np.ndarray)):
        return tuple(sorted({int(i) for i in np.asarray(list(spec)).ravel()}))
    raise TypeError(
        f"unknown active_space 'orbitals' {spec!r}; use an int (total "
        "spatial orbitals), {'occupied': n, 'virtual': m} or an explicit list "
        "of spatial orbital indices")


def _resolve_counts(orbitals, n_doubly: int, n_singly: int,
                    n_virtual: int, n_frozen: int):
    """``(n_doubly_active, n_virtual_active)`` from a count spec.

    ``n_frozen`` is how many doubly occupied orbitals the ``"frozen"`` core
    already removed; a count spec is read as the register the user wants
    *after* that.
    """
    available_doubly = n_doubly - n_frozen
    if isinstance(orbitals, dict):
        unknown = sorted(set(orbitals) - {"occupied", "virtual"})
        if unknown:
            raise ValueError(
                f"unknown key(s) {unknown} in the active_space 'orbitals'; "
                "its dict form takes 'occupied' (doubly occupied spatial "
                "orbitals kept active) and 'virtual'")
        n_occ_active = int(orbitals.get("occupied", available_doubly))
        n_virt_active = int(orbitals.get("virtual", n_virtual))
        if not 0 <= n_occ_active <= available_doubly:
            raise ValueError(
                f"the active_space asks for {n_occ_active} active doubly "
                f"occupied orbitals, but only {available_doubly} are available "
                f"({n_doubly} doubly occupied, {n_frozen} already frozen)")
        if not 0 <= n_virt_active <= n_virtual:
            raise ValueError(
                f"the active_space asks for {n_virt_active} active virtual "
                f"orbitals of {n_virtual}")
        return n_occ_active, n_virt_active

    total = int(orbitals)
    if total <= 0:
        raise ValueError(
            f"the active_space 'orbitals' must be a positive number of "
            f"spatial orbitals, got {total}")
    # The integer form never touches the occupied space: an occupied orbital is
    # removed only when it is asked for by name.  So the count it fixes is the
    # register width, and the virtuals take whatever is left.
    occupied_kept = available_doubly + n_singly
    n_virt_active = total - occupied_kept
    if n_virt_active < 0:
        raise ValueError(
            f"'orbitals': {total} is smaller than the {occupied_kept} "
            f"occupied spatial orbitals that would stay active "
            f"({available_doubly} doubly occupied"
            + (f" + {n_singly} singly occupied" if n_singly else "")
            + "). The integer form of 'orbitals' never removes an occupied "
              "orbital, because that changes the electron count rather than "
              "the correlation treatment: raise it, freeze a core with "
              "'frozen', or name the split explicitly with "
              "'orbitals': {'occupied': n, 'virtual': m}.")
    if n_virt_active > n_virtual:
        raise ValueError(
            f"'orbitals': {total} exceeds the {occupied_kept + n_virtual} "
            f"spatial orbitals the basis has (after freezing {n_frozen})")
    return available_doubly, n_virt_active


def _virtual_ranking(method: str, h_mo, eri_mo, n_doubly: int,
                     first_virtual: int, n_orbitals: int,
                     reference_occupations=None, open_shell: bool = False,
                     orbital_integrals=None, frozen=()):
    """``(rotation, occupations, correlation_energy, occupied_density)``.

    ``rotation`` is ``None`` when canonical order is already the ranking.
    ``occupations`` covers every orbital of the rotated basis, or is ``None``
    when the ranking is by orbital energy.  ``open_shell`` selects the
    open-shell MP2 expression for ``"mp2"``.  ``occupied_density`` is the MP2
    density's occupied block (``None`` for the other methods), for
    :func:`_rotate_active_occupied`.
    """
    if method == "energy":
        return None, None, None, None

    if method == "dlpno-mp2":
        from .dlpno_mp2 import dlpno_mp2

        if open_shell:
            raise NotImplementedError(
                "the 'dlpno-mp2' active_space method is closed-shell only; "
                "an open-shell reference needs the 'mp2' method")
        if orbital_integrals is None:
            raise ValueError("the 'dlpno-mp2' method needs an orbital-"
                             "integral provider with its basis")
        # The requested core is not correlated (frozen-core local MP2);
        # orbitals a count freezes later are, as canonical MP2 does.
        result = dlpno_mp2(orbital_integrals, n_doubly, frozen=frozen)
        occupations = np.concatenate([result.occupied_occupations,
                                      result.virtual_occupations])
        return (result.rotation, occupations, result.correlation_energy,
                result.occupied_density)

    if method == "mp2":
        from .mp2 import mp2_natural_orbitals, open_shell_mp2_natural_orbitals

        # An open-shell Hamiltonian is built in the UHF natural orbitals, which
        # do not diagonalize any Fock matrix, and its reference may have singly
        # occupied orbitals: the open-shell expression perturbs that determinant
        # and never rotates its occupied orbitals.  The alpha/beta labels do not
        # matter to a spin-summed density, so the larger count goes first.
        if open_shell:
            result = open_shell_mp2_natural_orbitals(h_mo, eri_mo,
                                                     first_virtual, n_doubly)
        else:
            result = mp2_natural_orbitals(h_mo, eri_mo, n_doubly)
        occupations = np.concatenate([result.occupied_occupations,
                                      result.virtual_occupations])
        return (result.rotation, occupations, result.correlation_energy,
                result.occupied_density)

    # method == "natural": the reference's own natural occupations.  The
    # orbitals are already the natural orbitals (that is the basis an open-shell
    # Hamiltonian is built in), so only the order of the virtuals is at stake.
    occupations = np.asarray(reference_occupations, dtype=float).ravel()
    if occupations.size != n_orbitals:
        raise ValueError(
            f"the 'natural' active_space method needs one occupation per "
            f"spatial "
            f"orbital ({n_orbitals}), got {occupations.size}")
    virtual = occupations[first_virtual:]
    order = np.argsort(-virtual)                 # most populated first
    if np.array_equal(order, np.arange(virtual.size)):
        return None, occupations, None, None
    rotation = np.eye(n_orbitals, dtype=float)
    permutation = np.eye(len(virtual), dtype=float)[:, order]
    rotation[first_virtual:, first_virtual:] = permutation
    ranked = np.concatenate([occupations[:first_virtual], virtual[order]])
    return rotation, ranked, None, None


def _rotate_active_occupied(rotation, occupations, occupied_density,
                            indices, n_orbitals: int):
    """Diagonalize the MP2 density over the active doubly occupied orbitals.

    A rotation among doubly occupied orbitals leaves the reference
    determinant, and so the Hartree-Fock energy, unchanged; it makes those
    orbitals natural orbitals, so their occupations are eigenvalues of the
    density as the virtual ones are.  Frozen orbitals are not in ``indices``
    and stay where the energy ordering put them, which is what the frozen-core
    indices were resolved against.  Singly occupied orbitals are never
    included: mixing them with doubly occupied ones changes the determinant.
    The rotated orbitals are ordered by descending occupation.
    """
    indices = list(indices)
    if len(indices) < 2:
        return rotation, occupations
    block = np.asarray(occupied_density, dtype=float)[np.ix_(indices, indices)]
    values, vectors = np.linalg.eigh(0.5 * (block + block.T))
    values, vectors = values[::-1], vectors[:, ::-1]
    turn = np.eye(n_orbitals, dtype=float)
    turn[np.ix_(indices, indices)] = vectors
    occupations = np.array(occupations, dtype=float)
    occupations[indices] = values
    base = np.eye(n_orbitals) if rotation is None else np.asarray(rotation)
    return base @ turn, occupations


def correlating_partners(integrals, occupied, virtual, weights,
                         rotation=None) -> tuple[tuple[int, int], ...]:
    r"""Pair each occupied orbital with the virtual orbital that correlates it.

    The weight of a pair is the size of the first-order amplitude of the pair
    excitation :math:`i^2 \to a^2` -- the excitation that turns a bonding
    orbital's pair into its antibonding one:

    .. math::

        w_{ia} = \frac{|\langle ii|aa\rangle|}{F_{aa} - F_{ii}} ,

    an exchange-type integral, large when :math:`i` and :math:`a` occupy the
    same region of space with a node between them, over the energy it costs.
    The pairs are the one-to-one assignment of maximum total weight
    (:func:`scipy.optimize.linear_sum_assignment`), so two occupied orbitals
    never share a partner.  ``integrals`` is an orbital-integral provider
    (:mod:`~mandacaru.algorithms.orbital_integrals`), asked in the basis
    ``rotation`` defines (``None``: its own orbitals), which is the basis the
    indices refer to; ``weights`` are the reference occupations (2, 1 or 0)
    of every orbital, for the Fock diagonal.
    """
    from scipy.optimize import linear_sum_assignment

    occupied, virtual = list(occupied), list(virtual)
    if not occupied or not virtual:
        return ()
    fock = integrals.fock_diagonal(weights, rotation)
    coupling = integrals.pair_exchange(occupied, virtual, rotation)
    gap = np.maximum(fock[virtual][None, :] - fock[occupied][:, None], 1e-6)
    rows, cols = linear_sum_assignment(-(coupling / gap))
    return tuple(sorted((occupied[r], virtual[c]) for r, c in zip(rows, cols)))


def _select_units(units, budget, eligible, mandatory, threshold_ok):
    """Fill the virtual register unit by unit, the best ranked first.

    ``units`` are lists of virtual indices that enter or leave together (a
    degenerate set), in rank order.  ``mandatory`` units go in first; then
    every other eligible unit whose occupation clears the threshold, in rank
    order, whenever it still fits in ``budget`` (``None``: no cap).  A unit
    that does not fit is skipped, and a smaller one further down may still
    take its place: the register's size is what was asked for, and the
    constraints decide what fills it.
    """
    chosen = [p for unit in units if unit in mandatory for p in unit]
    if budget is not None and len(chosen) > budget:
        raise ValueError(
            f"the active_space constraints need {len(chosen)} virtual "
            f"orbitals ({sorted(chosen)}) but the count leaves room for "
            f"{budget}: raise 'orbitals' by {len(chosen) - budget}, or drop "
            f"the constraint")
    for unit in units:
        if unit in mandatory or not eligible(unit) or not threshold_ok(unit):
            continue
        if budget is not None and len(chosen) + len(unit) > budget:
            continue
        chosen.extend(unit)
    return sorted(chosen)


def resolve_active_space(h_mo=None, eri_mo=None, *, n_orbitals: int,
                         num_particles,
                         spec: ActiveSpaceSpec | None = None, frozen=(),
                         reference_occupations=None,
                         open_shell: bool = False,
                         orbital_symmetry=None,
                         orbital_integrals=None,
                         orbital_gauge=None) -> ActiveSpace:
    """Partition the spatial MOs into frozen / active / deleted.

    Parameters
    ----------
    h_mo, eri_mo : ndarray, optional
        MO-basis one- and two-body integrals.  Needed only by the ``"mp2"``
        selector; ``"energy"`` and an untruncated space need neither.
    n_orbitals : int
        Total number of spatial MOs.
    num_particles : (int, int)
        ``(n_alpha, n_beta)`` of the reference **before** any freezing.  The
        doubly occupied orbitals are ``0 .. n_beta - 1``, the singly occupied
        ones ``n_beta .. n_alpha - 1``, and the virtuals start at ``n_alpha``.
    spec : ActiveSpaceSpec, optional
        The ``active_space`` option (:func:`resolve_active_space_spec`);
        ``None`` keeps every orbital.  Its ``orbitals`` sizes the register: an
        ``int`` is the total number of spatial orbitals it should carry, a
        ``dict`` with keys ``"occupied"`` and ``"virtual"`` names the two
        counts, and a **list or tuple** is explicit spatial MO indices -- never
        an ``(occupied, virtual)`` pair, so there is one meaning per type.  Its
        ``method`` ranks the virtuals (:data:`ACTIVE_SPACE_METHODS`) and its
        ``threshold`` is an occupation criterion on the virtual natural
        orbitals: keep those whose occupation is at least that.  It needs a
        method that *has* occupations, so it is refused for ``"energy"``, and
        it composes with ``orbitals``, which then acts as a hard cap on the
        register: the threshold says which orbitals are worth keeping and the
        count says how many there is room for, and whichever binds first wins.
    frozen : sequence of int
        Doubly occupied spatial MOs already frozen -- the spec's ``frozen``,
        resolved to indices by the builder.
    reference_occupations : sequence of float, optional
        Natural occupations of the reference, for the ``"natural"`` method.
    open_shell : bool
        Whether the reference is the open-shell (UHF natural orbital) one.
    orbital_integrals : provider, optional
        Where the selection's integrals come from
        (:mod:`~mandacaru.algorithms.orbital_integrals`); by default the
        tensors ``h_mo`` / ``eri_mo``.  An integral-direct build passes a
        :class:`~mandacaru.algorithms.orbital_integrals.DirectOrbitalIntegrals`
        and no tensors.
    orbital_symmetry : callable, optional
        ``rotation -> OrbitalSymmetry`` of the orbitals that ``rotation``
        (``None``: the incoming ones) defines, for ``spec.symmetry``.  The
        integrals supply it; without one the symmetry constraint is refused.
    orbital_gauge : callable, optional
        The gauge operator of the incoming orbitals
        (:func:`~mandacaru.algorithms.orbital_tracking.orbital_gauge`).  With
        it, the orbitals of every degenerate set of a selector's natural
        occupations are fixed inside the set
        (:func:`~mandacaru.algorithms.orbital_tracking.degenerate_gauge`)
        instead of left in whichever rotation round-off gave the
        eigensolver, so they follow the geometry smoothly (what
        ``transfer=True`` needs).  The span, the occupations and the
        reference are unchanged.

    With ``correlating_pairs`` or ``symmetry`` the virtual orbitals are no
    longer simply the best-ranked ones: they are filled in units (a degenerate
    set enters whole), the units the constraints require first, then the rest
    in rank order while they fit.  The register keeps the size the count
    asked for, and a count too small for the required units is refused.
    """
    M = int(n_orbitals)
    if orbital_integrals is None and h_mo is not None and eri_mo is not None:
        from .orbital_integrals import TensorOrbitalIntegrals
        orbital_integrals = TensorOrbitalIntegrals(h_mo, eri_mo)
    n_alpha, n_beta = int(num_particles[0]), int(num_particles[1])
    n_doubly, n_singly = min(n_alpha, n_beta), abs(n_alpha - n_beta)
    first_virtual = n_doubly + n_singly
    n_virtual = M - first_virtual
    spec = spec if spec is not None else ActiveSpaceSpec()
    method, orbitals, threshold = spec.method, spec.orbitals, spec.threshold
    frozen = tuple(sorted({int(i) for i in frozen}))
    for i in frozen:
        if not 0 <= i < n_doubly:
            raise ValueError(
                f"frozen spatial orbital {i} is not doubly occupied "
                f"(indices 0..{n_doubly - 1} are)")

    if orbitals is None and threshold is None:
        active = tuple(p for p in range(M) if p not in set(frozen))
        point_group, irreps = None, ()
        if spec.symmetry:
            # Nothing is deleted, so only the frozen core can break symmetry.
            sym = _symmetry_of(orbital_symmetry, None)
            _check_unsplit(sym, frozen, first_virtual, n_doubly)
            labels = _orbital_labels(sym, M)
            point_group = sym.group.name
            irreps = tuple(labels[p] for p in range(M))
        return ActiveSpace(n_orbitals=M, frozen=frozen, active=active,
                           deleted=(), method=method, n_doubly=n_doubly,
                           n_singly=n_singly, point_group=point_group,
                           irreps=irreps)

    if method == "natural" and not open_shell:
        raise ValueError(
            "the 'natural' active_space method carries no information for a "
            "closed-shell reference: the RHF density is idempotent, so its "
            "natural occupations are exactly 2 and 0 and every ordering of the "
            "virtual orbitals is as good as every other.  Use "
            "'method': 'mp2', which ranks the virtuals by the charge "
            "second-order correlation actually puts in them, or "
            "'energy' if the canonical order is what you meant.")

    # -- explicit index list ------------------------------------------------ #
    if isinstance(orbitals, (list, tuple, set, frozenset, np.ndarray)):
        if method != "energy":
            raise ValueError(
                f"an explicit 'orbitals' list cannot be combined with the "
                f"{method!r} active_space method: it rotates the virtual "
                f"orbitals, so the indices would refer to a basis chosen by "
                f"the method rather than the canonical one.  Give a count (an "
                f"int, or {{'occupied': n, 'virtual': m}}) and let the method "
                f"choose, or keep the canonical basis with 'method': "
                f"'energy'.")
        active = tuple(sorted({int(i) for i in orbitals}))
        if not active:
            raise ValueError("the active_space 'orbitals' is an empty list")
        for p in active:
            if not 0 <= p < M:
                raise ValueError(
                    f"active orbital index {p} is out of range [0, {M})")
        overlap = sorted(set(active) & set(frozen))
        if overlap:
            raise ValueError(
                f"spatial orbital(s) {overlap} are both frozen and active")
        missing = [p for p in range(first_virtual)
                   if p not in set(active) | set(frozen)]
        if missing:
            raise ValueError(
                f"occupied spatial orbital(s) {missing} are neither active nor "
                f"frozen.  An occupied orbital that is dropped takes its "
                f"electrons with it, which is a different system rather than a "
                f"smaller correlation treatment; freeze it instead "
                f"('frozen': {missing!r}) if the intent was the frozen-core "
                f"approximation.")
        deleted = tuple(p for p in range(first_virtual, M)
                        if p not in set(active))
        partners = ()
        if spec.correlating_pairs:
            # An explicit list is the user's choice, so it is checked, not
            # changed.
            weights = [2] * n_doubly + [1] * n_singly + [0] * n_virtual
            partners = correlating_partners(
                _need(orbital_integrals, "correlating_pairs"),
                [p for p in range(n_doubly) if p not in set(frozen)],
                range(first_virtual, M), weights)
            missing = [(i, a) for i, a in partners if a not in set(active)]
            if missing:
                raise ValueError(
                    f"the active_space 'orbitals' list leaves out the "
                    f"correlating partner of an active occupied orbital "
                    f"(occupied, virtual): {missing}.  Add those virtual "
                    f"orbitals, or drop 'correlating_pairs'.")
        point_group, irreps = None, ()
        if spec.symmetry:
            sym = _symmetry_of(orbital_symmetry, None)
            _check_unsplit(sym, frozen, first_virtual, n_doubly)
            for members, label in zip(sym.sets, sym.labels):
                kept = [p for p in members if p in set(active)]
                if kept and len(kept) < len(members):
                    raise ValueError(
                        f"the active_space 'orbitals' list keeps {kept} of "
                        f"the degenerate set {list(members)} ({label}); "
                        f"keep all of it or none, or drop 'symmetry'")
            labels = _orbital_labels(sym, M)
            point_group = sym.group.name
            irreps = tuple(labels[p] for p in range(M))
        return ActiveSpace(n_orbitals=M, frozen=frozen, active=active,
                           deleted=deleted, method=method, n_doubly=n_doubly,
                           n_singly=n_singly, partners=partners,
                           point_group=point_group, irreps=irreps)

    # -- the selector ranks, then the criteria cut -------------------------- #
    if orbitals is not None:
        # Refuse an impossible count before the selector runs: an MP2 (or a
        # DLPNO-MP2 on a large basis) is minutes of work to throw away.
        _resolve_counts(orbitals, n_doubly, n_singly, n_virtual, len(frozen))
    tensors = (orbital_integrals.tensors()
               if orbital_integrals is not None else None)
    if method == "mp2" and tensors is None:
        raise ValueError(
            "the 'mp2' active_space method is canonical MP2 over the full "
            "two-body tensor, which an integral-direct build never forms")
    h_mo, eri_mo = tensors if tensors is not None else (h_mo, eri_mo)
    rotation, occupations, correlation, occupied_density = _virtual_ranking(
        method, h_mo, eri_mo, n_doubly, first_virtual, M,
        reference_occupations, open_shell=open_shell,
        orbital_integrals=orbital_integrals, frozen=frozen)
    cap = None
    if orbitals is None:
        # A threshold on its own is a complete criterion: it names how many
        # virtual orbitals are worth keeping, so no count is needed.
        n_occ_active = n_doubly - len(frozen)
        n_virt_active = _count_above(occupations[first_virtual:], threshold,
                                    method)
    else:
        n_occ_active, n_virt_active = _resolve_counts(
            orbitals, n_doubly, n_singly, n_virtual, len(frozen))
        cap = n_virt_active
        if threshold is not None:
            # Both given: the threshold says which orbitals earn their place and
            # the count says how many there is room for.  Whichever binds first
            # wins, and the count is the one that can make a run impossible, so
            # it is the cap.
            n_virt_active = min(
                n_virt_active,
                _count_above(occupations[first_virtual:], threshold, method))
    if rotation is not None:
        rotation = np.asarray(rotation, dtype=float)
        residual = float(np.max(np.abs(
            rotation.T @ rotation - np.eye(M)))) if M else 0.0
        if residual > ROTATION_TOLERANCE:
            raise ValueError(
                f"the {method} selector returned a non-orthogonal rotation "
                f"(|R^T R - 1| = {residual:.2e}); the orbitals it defines "
                f"would not be orthonormal and every integral built from them "
                f"would be wrong")
        coupling = float(np.max(np.abs(
            rotation[:first_virtual, first_virtual:]))) if n_virtual else 0.0
        if coupling > ROTATION_TOLERANCE:
            raise ValueError(
                f"the {method} selector mixes occupied and virtual orbitals "
                f"(largest coupling {coupling:.2e}); that changes the reference "
                f"determinant, so the Hartree-Fock energy and the occupation "
                f"the ansatz prepares would no longer be the ones reported")
        # The frozen indices were resolved against the *incoming* basis (the
        # chemical core is "the lowest so many MOs"), so a selector that
        # reordered the occupied block would leave them pointing at different
        # orbitals.  Nothing here should: the MP2 selector leaves that block
        # alone by construction and only reaches for a semicanonicalization
        # when the Fock matrix is not diagonal, which an RHF/UHF basis makes it.
        # If that ever stops holding, the run must stop rather than freeze
        # whichever orbitals the new labels happen to name.
        occupied_shift = float(np.max(np.abs(
            rotation[:first_virtual, :first_virtual]
            - np.eye(first_virtual)))) if first_virtual else 0.0
        if occupied_shift > ROTATION_TOLERANCE:
            raise NotImplementedError(
                f"the {method} selector had to rotate the occupied orbitals "
                f"(|R_oo - 1| = {occupied_shift:.2e}), which means the incoming "
                f"molecular orbitals did not diagonalize the Fock matrix.  The "
                f"frozen-core indices were resolved against the old labels and "
                f"would now name different orbitals, so the selection is "
                f"refused instead of applied to the wrong ones.  Use "
                f"'method': 'energy' with an explicit 'orbitals' list for such "
                f"a basis.")

    # The ranking is now the index order, so "keep the best m" is "keep the
    # first m".  Occupied truncation stays energy-ordered: the core is the
    # lowest orbitals, and `frozen` already holds any named by the caller.
    still_free = [p for p in range(n_doubly) if p not in set(frozen)]
    extra_frozen = still_free[:len(still_free) - n_occ_active]
    core = frozen
    frozen = tuple(sorted(set(frozen) | set(extra_frozen)))
    active_occupied = [p for p in range(first_virtual) if p not in set(frozen)]
    if occupied_density is not None:
        rotation, occupations = _rotate_active_occupied(
            rotation, occupations, occupied_density,
            [p for p in range(n_doubly) if p not in set(frozen)], M)
    if orbital_gauge is not None and occupations is not None:
        # Before the selection, so a count that cuts a degenerate set (warned
        # below) at least cuts it the same way every time.
        from .orbital_tracking import degenerate_gauge

        rotation = degenerate_gauge(
            rotation, occupations,
            [[p for p in range(n_doubly) if p not in set(frozen)],
             range(first_virtual, M)], orbital_gauge)
    active_virtual = list(range(first_virtual, first_virtual + n_virt_active))
    plain = list(active_virtual)
    partners, notes, labels = (), [], {}
    if spec.correlating_pairs or spec.symmetry:
        values = (None if occupations is None
                  else np.asarray(occupations, dtype=float))
        singles = [[a] for a in range(first_virtual, M)]
        required, barred, sym = set(), set(), None
        if spec.correlating_pairs:
            weights = [2] * n_doubly + [1] * n_singly + [0] * n_virtual
            # The requested core is not a bonding orbital and is never
            # paired; an occupied orbital the count froze is, so its partner
            # is barred.  A singly occupied orbital has no pair to promote.
            partners = correlating_partners(
                _need(orbital_integrals, "correlating_pairs"),
                [p for p in range(n_doubly) if p not in set(core)],
                range(first_virtual, M), weights, rotation)
            required = {a for i, a in partners if i not in set(frozen)}
            barred = {a for i, a in partners if i in set(frozen)}

        def select(units, mandatory):
            return _select_units(
                units, cap,
                eligible=lambda unit: not set(unit) & barred,
                mandatory=mandatory,
                threshold_ok=lambda unit: (
                    threshold is None
                    or float(np.max(values[unit])) >= threshold))

        chosen = plain
        if spec.correlating_pairs:
            chosen = select(singles, [u for u in singles if set(u) & required])
            notes += _change_notes("correlating_pairs", plain, chosen)
        if spec.symmetry:
            sym = _symmetry_of(orbital_symmetry, rotation)
            _check_unsplit(sym, frozen, first_virtual, n_doubly)
            labels = _orbital_labels(sym, M)
            units = [[p for p in members if p >= first_virtual]
                     for members in sym.sets]
            units = [unit for unit in units if unit]
            before = chosen
            mandatory = [u for u in units if set(u) & required]
            chosen = select(units, mandatory)
            targets = _target_excitations(
                orbital_symmetry, orbital_integrals, n_doubly, n_singly,
                core, first_virtual, M, spec.states)
            occupied_sets = [
                k for k, members in enumerate(sym.sets)
                if any(p < first_virtual and p not in set(frozen)
                       for p in members)]
            reference = _reference_character(sym, n_doubly, first_virtual)
            for label, target in targets:
                def reaches(unit, target=target):
                    k_a = sym.set_of(unit[0])
                    return any(_contains(sym, reference
                                         * sym.characters[k_i]
                                         * sym.characters[k_a], target)
                               for k_i in occupied_sets)
                # The unit that meets a target becomes required, so a later
                # target cannot displace it; a count too small for every
                # target is then refused by the selection itself.
                held = next((u for u in units if set(u) <= set(chosen)
                             and reaches(u)), None)
                if held is not None:
                    if held not in mandatory:
                        mandatory.append(held)
                        chosen = select(units, mandatory)
                    continue
                extra = next((u for u in units if reaches(u)
                              and u not in mandatory
                              and not set(u) & barred), None)
                if extra is None:
                    raise ValueError(
                        f"the active_space symmetry targets the {label} "
                        f"excitation, and no active occupied orbital reaches "
                        f"its symmetry with any virtual orbital of this "
                        f"basis")
                mandatory.append(extra)
                chosen = select(units, mandatory)
            detail = ("targets " + ", ".join(t for t, _ in targets)
                      if targets else "")
            if cap is not None and len(chosen) < cap:
                # Every unit left is bigger than the room left: the register
                # comes out smaller than asked rather than splitting a set.
                detail += ("; " if detail else "") + (
                    f"{cap - len(chosen)} virtual slot(s) left empty, no "
                    f"whole set fits")
            notes += _change_notes("symmetry", before, chosen, detail)
        active_virtual = chosen
    if not spec.symmetry and occupations is not None:
        split = _split_degenerate(occupations, active_virtual, first_virtual,
                                  M)
        if split:
            warnings.warn(
                f"the active space takes {len(split[0])} of the "
                f"{len(split[0]) + len(split[1])} degenerate virtual "
                f"orbitals {sorted(split[0] + split[1])} (equal "
                f"occupations): which ones is an arbitrary choice, and so "
                f"is every result in the space.  Pass "
                f"active_space={{..., 'symmetry': True}} to keep "
                f"degenerate sets whole, or change the count",
                RuntimeWarning, stacklevel=2)
            notes.append(("degenerate_split",
                          f"splits the degenerate set "
                          f"{sorted(split[0] + split[1])}"))
    deleted = tuple(p for p in range(first_virtual, M)
                    if p not in set(active_virtual))
    return ActiveSpace(
        n_orbitals=M, frozen=frozen,
        active=tuple(active_occupied + active_virtual), deleted=deleted,
        method=method, rotation=rotation, occupations=occupations,
        correlation_energy=correlation, n_doubly=n_doubly, n_singly=n_singly,
        partners=partners, notes=tuple(notes),
        point_group=None if not labels else sym.group.name,
        irreps=tuple(labels.get(p, "?") for p in range(M)) if labels else ())


def _split_degenerate(occupations, chosen, first_virtual: int, M: int):
    """``(taken, left)`` of the first degenerate set of virtual natural
    occupations (a cluster of
    :func:`~mandacaru.algorithms.orbital_tracking.degenerate_clusters`) the
    selection ``chosen`` cuts through, or ``()``.

    LiH in PAW-LCAO DZP with ``orbitals=4``: the fourth slot took one of the
    two pi orbitals of occupation 0.002352, a choice the last bits of the
    integrals made (ADAPT-VQE energies 52 meV apart).  The gauge of
    :func:`~mandacaru.algorithms.orbital_tracking.degenerate_gauge` makes the
    choice reproducible, not meaningful."""
    from .orbital_tracking import degenerate_clusters

    chosen = set(chosen)
    for same in degenerate_clusters(occupations, range(first_virtual, M)):
        taken = [q for q in same if q in chosen]
        if 0 < len(taken) < len(same):
            return taken, [q for q in same if q not in chosen]
    return ()


def _need(orbital_integrals, constraint: str):
    """The provider, or a refusal naming the constraint that needed it."""
    if orbital_integrals is None:
        raise ValueError(f"the active_space {constraint!r} needs the "
                         f"molecular-orbital integrals")
    return orbital_integrals


def _symmetry_of(orbital_symmetry, rotation):
    """The :class:`~mandacaru.algorithms.orbital_symmetry.OrbitalSymmetry` of
    the ranked orbitals, or a clear refusal when there is no geometry."""
    if orbital_symmetry is None:
        raise ValueError(
            "the active_space 'symmetry' needs the molecule's geometry and "
            "basis functions to find its point group; it is not available "
            "for a Hamiltonian given without them, a periodic system or a "
            "spin-orbit (spinor) basis")
    return orbital_symmetry(rotation)


def _check_unsplit(sym, frozen, first_virtual, n_doubly=None) -> None:
    """Refuse a reference or a frozen core that breaks the symmetry.

    A degenerate set holding both doubly occupied and virtual orbitals means
    the Hartree-Fock determinant is not symmetric (the SCF found a
    broken-symmetry solution), so no active space can be chosen by symmetry.
    A set of singly occupied and virtual orbitals is an open shell's
    partially filled level and is allowed.  A frozen core that takes part of
    a degenerate occupied set is refused as well.
    """
    n_doubly = first_virtual if n_doubly is None else n_doubly
    for members, label in zip(sym.sets, sym.labels):
        doubly = [p for p in members if p < n_doubly]
        virtual = [p for p in members if p >= first_virtual]
        if doubly and virtual:
            raise ValueError(
                f"the Hartree-Fock reference breaks the molecule's "
                f"{sym.group.name} symmetry: doubly occupied orbital(s) "
                f"{doubly} and virtual orbital(s) {virtual} belong to one "
                f"degenerate set ({label}), so the determinant is not "
                f"symmetric and no active space can be chosen by symmetry.  "
                f"The SCF converged to a broken-symmetry solution; a finer "
                f"grid (smaller h) or a larger basis usually restores it.  "
                f"Drop 'symmetry' to run anyway.")
        occupied = [p for p in members if p < first_virtual]
        cut = [p for p in occupied if p in set(frozen)]
        if cut and len(cut) < len(occupied):
            raise ValueError(
                f"the frozen core splits the degenerate occupied set "
                f"{list(occupied)} ({label}): orbitals {cut} are frozen and "
                f"the rest are active, which breaks the molecule's "
                f"symmetry.  Freeze all of them or none (change the "
                f"'occupied' count by {len(cut)} or "
                f"{len(occupied) - len(cut)}).")


def _orbital_labels(sym, M) -> dict[int, str]:
    """``{orbital: "3a1"}``: Mulliken label numbered within its irrep."""
    count: dict[str, int] = {}
    labels = {}
    for members, label in sorted(zip(sym.sets, sym.labels),
                                 key=lambda pair: pair[0][0]):
        count[label] = count.get(label, 0) + 1
        for p in members:
            labels[p] = (label if label == "?"
                         else f"{count[label]}{label}")
    return labels


def _contains(sym, character, target) -> bool:
    from .orbital_symmetry import contains
    return contains(sym.group, character, target)


def _reference_character(sym, n_doubly, first_virtual):
    """Character of the reference determinant: its singly occupied sets."""
    character = np.ones(sym.group.order)
    seen = set()
    for p in range(n_doubly, first_virtual):
        k = sym.set_of(p)
        if k not in seen:
            seen.add(k)
            character = character * sym.characters[k]
    return character


def _target_excitations(orbital_symmetry, orbital_integrals, n_doubly,
                        n_singly, core, first_virtual, M, states):
    """``[(label, character)]`` of the lowest ``states - 1`` excitations.

    The excited states' symmetry is not known before they are solved for, so
    the target is a proxy: the lowest single excitations of the canonical
    (unranked) orbitals, by orbital-energy gap, from a non-core occupied set
    to a virtual set.  Their characters, times the reference's, are what the
    active space must be able to reach.
    """
    if states <= 1:
        return []
    canonical = orbital_symmetry(None)
    weights = [2] * n_doubly + [1] * n_singly + [0] * (M - first_virtual)
    fock = _need(orbital_integrals, "symmetry").fock_diagonal(weights)
    reference = _reference_character(canonical, n_doubly, first_virtual)
    occupied = [k for k, m in enumerate(canonical.sets)
                if any(p < first_virtual and p not in set(core) for p in m)]
    virtual = [k for k, m in enumerate(canonical.sets)
               if any(p >= first_virtual for p in m)]
    names = _orbital_labels(canonical, M)
    excitations = []
    for k_i in occupied:
        for k_a in virtual:
            i_members = [p for p in canonical.sets[k_i] if p < first_virtual]
            a_members = [p for p in canonical.sets[k_a] if p >= first_virtual]
            gap = (float(np.mean(fock[a_members]))
                   - float(np.mean(fock[i_members])))
            character = (reference * canonical.characters[k_i]
                         * canonical.characters[k_a])
            excitations.append(
                (gap, f"{names[i_members[0]]}->{names[a_members[0]]}",
                 character))
    excitations.sort(key=lambda e: e[0])
    return [(label, character)
            for _gap, label, character in excitations[:states - 1]]


def _change_notes(constraint: str, before, after,
                  detail: str = "") -> list[tuple[str, str]]:
    """``[(constraint, what it changed in the virtual selection)]``."""
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    parts = [detail] if detail else []
    if added:
        parts.append(f"brought in {added}")
    if removed:
        parts.append(f"displaced {removed}")
    if not added and not removed:
        parts.append("the ranked selection already satisfied it")
    return [(constraint, f"on ({'; '.join(parts)})")]
