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
frozen-core approximation, and it is what ``frozen_core`` / ``frozen_orbitals``
have always done.

**Deleted** orbitals are virtual and are simply dropped.  Nothing has to be
folded in: an orbital that is empty in the reference contributes neither to the
core energy nor to the effective potential, so deleting it costs exactly the
correlation it would have carried and nothing else.  This is the new half, and
it is the half that reduces the register.

**Active** orbitals are the rest -- the ones the qubits represent.

Occupied and virtual are ranked differently, on purpose
-------------------------------------------------------

``active_selection`` ranks the **virtual** orbitals only.  Occupied truncation
is always by orbital energy, which is to say it is always the chemical core,
because an index-based core (``frozen_core="auto"`` resolves to "the lowest so
many MOs") stops meaning the core the moment the occupied orbitals are
reordered, and because removing an occupied orbital is a far coarser
approximation than removing a virtual one -- it belongs to an explicit request,
not to a selector.  See :func:`~mandacaru.algorithms.mp2.mp2_natural_orbitals`,
which declines to rotate the occupied block for the same reason.

For the virtuals there are three rankings:

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
``"natural"``
    The occupations of the **reference** natural orbitals.  Meaningful only for
    an open-shell (UHF) reference.  For a closed-shell RHF reference the density
    is idempotent, its eigenvalues are exactly 2 and 0, and every ordering of
    the virtuals is as good as any other -- so this is refused by name for a
    closed shell rather than quietly returning the energy ordering under a label
    that claims to be something else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Recognized values of ``active_selection``.
ACTIVE_SELECTIONS = ("energy", "mp2", "natural")

#: Occupation below which a virtual natural orbital is dropped by
#: ``active_threshold=True``.  Measured MP2 spectra put the useful range between
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
    selection : str
        Which ranking chose the virtuals (:data:`ACTIVE_SELECTIONS`).
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
    """

    n_orbitals: int
    frozen: tuple[int, ...] = ()
    active: tuple[int, ...] = ()
    deleted: tuple[int, ...] = ()
    selection: str = "energy"
    rotation: np.ndarray | None = None
    occupations: np.ndarray | None = field(default=None, repr=False)
    correlation_energy: float | None = None

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
            line += f" (virtuals ranked by {self.selection})"
        return line


def resolve_selection(selection) -> str:
    """Normalize an ``active_selection`` spec, or raise."""
    name = str(selection if selection is not None else "energy")
    name = name.strip().lower().replace("_", "-")
    if name not in ACTIVE_SELECTIONS:
        raise ValueError(
            f"unknown active_selection {selection!r}; use one of "
            f"{ACTIVE_SELECTIONS}")
    return name


def resolve_threshold(threshold):
    """Normalize an ``active_threshold`` spec to a float or ``None``.

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
            f"active_threshold must be a positive occupation number, got "
            f"{threshold!r}.  Zero keeps every virtual orbital, which is not a "
            f"truncation; use active_threshold=None to select by count instead.")
    if value >= 2.0:
        raise ValueError(
            f"active_threshold={value} is not an occupation number: a natural "
            f"orbital's occupation lies in [0, 2], and a *virtual* one's is the "
            f"small charge correlation promotes into it -- of order 1e-2 at the "
            f"very most.  A threshold at or above 2 keeps nothing.")
    return value


def _count_above(occupations, threshold: float, selection: str) -> int:
    """How many ranked virtual orbitals clear ``threshold``, or raise."""
    occupations = np.asarray(occupations, dtype=float)
    kept = int(np.count_nonzero(occupations >= threshold))
    if kept == 0:
        largest = float(np.max(occupations)) if occupations.size else 0.0
        raise ValueError(
            f"active_threshold={threshold:g} keeps no virtual orbital at all: "
            f"the largest {selection} occupation in this problem is "
            f"{largest:.3e}.  An active space with no virtual orbital cannot "
            f"correlate anything -- it would return the Hartree-Fock energy "
            f"through a variational solver.  Virtual natural occupations are "
            f"small by construction (the charge second-order correlation "
            f"promotes, not an electron count), so the useful range is roughly "
            f"1e-3 to 1e-5; {DEFAULT_OCCUPATION_THRESHOLD:g} is the default.")
    return kept


def normalize_active_orbitals(spec):
    """Canonical, JSON-writable form of an ``active_orbitals`` spec, or raise.

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
                f"unknown active_orbitals key(s) {unknown}; the dict form takes "
                "'occupied' and 'virtual'")
        return {str(k): int(v) for k, v in spec.items()}
    if isinstance(spec, bool):
        raise TypeError(
            "active_orbitals is a number of spatial orbitals, not a flag; use "
            "an int, {'occupied': n, 'virtual': m}, a list of orbital indices, "
            "or None")
    if isinstance(spec, (int, np.integer)):
        return int(spec)
    if isinstance(spec, (list, tuple, set, frozenset, np.ndarray)):
        return tuple(sorted({int(i) for i in np.asarray(list(spec)).ravel()}))
    raise TypeError(
        f"unknown active_orbitals spec {spec!r}; use an int (total spatial "
        "orbitals), {'occupied': n, 'virtual': m}, an explicit list of spatial "
        "orbital indices, or None")


def _resolve_counts(active_orbitals, n_doubly: int, n_singly: int,
                    n_virtual: int, n_frozen: int):
    """``(n_doubly_active, n_virtual_active)`` from a count spec.

    ``n_frozen`` is how many doubly occupied orbitals an explicit
    ``frozen_orbitals`` / ``frozen_core`` already removed; a count spec is read
    as the register the user wants *after* that.
    """
    available_doubly = n_doubly - n_frozen
    if isinstance(active_orbitals, dict):
        unknown = sorted(set(active_orbitals) - {"occupied", "virtual"})
        if unknown:
            raise ValueError(
                f"unknown active_orbitals key(s) {unknown}; the dict form "
                "takes 'occupied' (doubly occupied spatial orbitals kept "
                "active) and 'virtual'")
        n_occ_active = int(active_orbitals.get("occupied", available_doubly))
        n_virt_active = int(active_orbitals.get("virtual", n_virtual))
        if not 0 <= n_occ_active <= available_doubly:
            raise ValueError(
                f"active_orbitals asks for {n_occ_active} active doubly "
                f"occupied orbitals, but only {available_doubly} are available "
                f"({n_doubly} doubly occupied, {n_frozen} already frozen)")
        if not 0 <= n_virt_active <= n_virtual:
            raise ValueError(
                f"active_orbitals asks for {n_virt_active} active virtual "
                f"orbitals of {n_virtual}")
        return n_occ_active, n_virt_active

    total = int(active_orbitals)
    if total <= 0:
        raise ValueError(
            f"active_orbitals must be a positive number of spatial orbitals, "
            f"got {total}")
    # The integer form never touches the occupied space: an occupied orbital is
    # removed only when it is asked for by name.  So the count it fixes is the
    # register width, and the virtuals take whatever is left.
    occupied_kept = available_doubly + n_singly
    n_virt_active = total - occupied_kept
    if n_virt_active < 0:
        raise ValueError(
            f"active_orbitals={total} is smaller than the {occupied_kept} "
            f"occupied spatial orbitals that would stay active "
            f"({available_doubly} doubly occupied"
            + (f" + {n_singly} singly occupied" if n_singly else "")
            + "). The integer form of active_orbitals never removes an "
              "occupied orbital, because that changes the electron count "
              "rather than the correlation treatment: raise it, freeze a core "
              "with frozen_core=, or name the split explicitly with "
              "active_orbitals={'occupied': n, 'virtual': m}.")
    if n_virt_active > n_virtual:
        raise ValueError(
            f"active_orbitals={total} exceeds the {occupied_kept + n_virtual} "
            f"spatial orbitals the basis has (after freezing {n_frozen})")
    return available_doubly, n_virt_active


def _virtual_ranking(selection: str, h_mo, eri_mo, n_doubly: int,
                     first_virtual: int, n_orbitals: int,
                     reference_occupations=None, open_shell: bool = False):
    """``(rotation, occupations, correlation_energy)`` ranking the virtuals.

    ``rotation`` is ``None`` when canonical order is already the ranking.
    ``occupations`` covers every orbital of the rotated basis, or is ``None``
    when the ranking is by orbital energy.  ``open_shell`` selects the
    open-shell MP2 expression for ``"mp2"``.
    """
    if selection == "energy":
        return None, None, None

    if selection == "mp2":
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
        return result.rotation, occupations, result.correlation_energy

    # selection == "natural": the reference's own natural occupations.  The
    # orbitals are already the natural orbitals (that is the basis an open-shell
    # Hamiltonian is built in), so only the order of the virtuals is at stake.
    occupations = np.asarray(reference_occupations, dtype=float).ravel()
    if occupations.size != n_orbitals:
        raise ValueError(
            f"active_selection='natural' needs one occupation per spatial "
            f"orbital ({n_orbitals}), got {occupations.size}")
    virtual = occupations[first_virtual:]
    order = np.argsort(-virtual)                 # most populated first
    if np.array_equal(order, np.arange(virtual.size)):
        return None, occupations, None
    rotation = np.eye(n_orbitals, dtype=float)
    permutation = np.eye(len(virtual), dtype=float)[:, order]
    rotation[first_virtual:, first_virtual:] = permutation
    ranked = np.concatenate([occupations[:first_virtual], virtual[order]])
    return rotation, ranked, None


def resolve_active_space(h_mo=None, eri_mo=None, *, n_orbitals: int,
                         num_particles, active_orbitals=None,
                         selection="energy", frozen=(),
                         reference_occupations=None,
                         open_shell: bool = False,
                         threshold=None) -> ActiveSpace:
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
    active_orbitals : None, int, dict or sequence of int
        ``None`` keeps everything.  An ``int`` is the total number of spatial
        orbitals the register should carry.  A ``dict`` with keys ``"occupied"``
        and ``"virtual"`` names the two counts.  A **list or tuple** is read as
        explicit spatial MO indices -- never as an ``(occupied, virtual)`` pair,
        so there is one meaning per type.
    selection : str
        How the virtuals are ranked (:data:`ACTIVE_SELECTIONS`).
    frozen : sequence of int
        Doubly occupied spatial MOs already frozen by ``frozen_core`` /
        ``frozen_orbitals``.
    reference_occupations : sequence of float, optional
        Natural occupations of the reference, for ``selection="natural"``.
    open_shell : bool
        Whether the reference is the open-shell (UHF natural orbital) one.
    threshold : float, True or None
        Occupation criterion on the virtual natural orbitals: keep those whose
        occupation is at least ``threshold``.  ``True`` takes
        :data:`DEFAULT_OCCUPATION_THRESHOLD`.  Needs a selector that *has*
        occupations, so it is refused for ``selection="energy"``.  It composes
        with ``active_orbitals``, which then acts as a hard cap on the register:
        the threshold says which orbitals are worth keeping and the count says
        how many there is room for, and whichever binds first wins.
    """
    M = int(n_orbitals)
    n_alpha, n_beta = int(num_particles[0]), int(num_particles[1])
    n_doubly, n_singly = min(n_alpha, n_beta), abs(n_alpha - n_beta)
    first_virtual = n_doubly + n_singly
    n_virtual = M - first_virtual
    selection = resolve_selection(selection)
    threshold = resolve_threshold(threshold)
    if threshold is not None and selection == "energy":
        raise ValueError(
            f"active_threshold={threshold:g} needs occupation numbers to "
            f"compare against, and active_selection='energy' ranks the virtual "
            f"orbitals by orbital energy instead -- it never computes an "
            f"occupation.  Use active_selection='mp2' (or 'natural' for an "
            f"open-shell reference) with the threshold, or select by count.")
    frozen = tuple(sorted({int(i) for i in frozen}))
    for i in frozen:
        if not 0 <= i < n_doubly:
            raise ValueError(
                f"frozen spatial orbital {i} is not doubly occupied "
                f"(indices 0..{n_doubly - 1} are)")

    if active_orbitals is None and threshold is None:
        active = tuple(p for p in range(M) if p not in set(frozen))
        return ActiveSpace(n_orbitals=M, frozen=frozen, active=active,
                           deleted=(), selection=selection)

    if selection == "natural" and not open_shell:
        raise ValueError(
            "active_selection='natural' carries no information for a "
            "closed-shell reference: the RHF density is idempotent, so its "
            "natural occupations are exactly 2 and 0 and every ordering of the "
            "virtual orbitals is as good as every other.  Use "
            "active_selection='mp2', which ranks the virtuals by the charge "
            "second-order correlation actually puts in them, or "
            "'energy' if the canonical order is what you meant.")

    # -- explicit index list ------------------------------------------------ #
    if isinstance(active_orbitals, (list, tuple, set, frozenset, np.ndarray)):
        if selection != "energy":
            raise ValueError(
                f"an explicit active_orbitals list cannot be combined with "
                f"active_selection={selection!r}: that selector rotates the "
                f"virtual orbitals, so the indices would refer to a basis "
                f"chosen by the selector rather than the canonical one.  Give "
                f"a count (an int, or {{'occupied': n, 'virtual': m}}) and let "
                f"the selector choose, or keep the canonical basis with "
                f"active_selection='energy'.")
        active = tuple(sorted({int(i) for i in active_orbitals}))
        if not active:
            raise ValueError("active_orbitals is an empty orbital list")
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
                f"(frozen_orbitals={missing!r}) if the intent was the "
                f"frozen-core approximation.")
        deleted = tuple(p for p in range(first_virtual, M)
                        if p not in set(active))
        return ActiveSpace(n_orbitals=M, frozen=frozen, active=active,
                           deleted=deleted, selection=selection)

    # -- the selector ranks, then the criteria cut -------------------------- #
    rotation, occupations, correlation = _virtual_ranking(
        selection, h_mo, eri_mo, n_doubly, first_virtual, M,
        reference_occupations, open_shell=open_shell)
    if active_orbitals is None:
        # A threshold on its own is a complete criterion: it names how many
        # virtual orbitals are worth keeping, so no count is needed.
        n_occ_active = n_doubly - len(frozen)
        n_virt_active = _count_above(occupations[first_virtual:], threshold,
                                    selection)
    else:
        n_occ_active, n_virt_active = _resolve_counts(
            active_orbitals, n_doubly, n_singly, n_virtual, len(frozen))
        if threshold is not None:
            # Both given: the threshold says which orbitals earn their place and
            # the count says how many there is room for.  Whichever binds first
            # wins, and the count is the one that can make a run impossible, so
            # it is the cap.
            n_virt_active = min(
                n_virt_active,
                _count_above(occupations[first_virtual:], threshold, selection))
    if rotation is not None:
        rotation = np.asarray(rotation, dtype=float)
        residual = float(np.max(np.abs(
            rotation.T @ rotation - np.eye(M)))) if M else 0.0
        if residual > ROTATION_TOLERANCE:
            raise ValueError(
                f"the {selection} selector returned a non-orthogonal rotation "
                f"(|R^T R - 1| = {residual:.2e}); the orbitals it defines "
                f"would not be orthonormal and every integral built from them "
                f"would be wrong")
        coupling = float(np.max(np.abs(
            rotation[:first_virtual, first_virtual:]))) if n_virtual else 0.0
        if coupling > ROTATION_TOLERANCE:
            raise ValueError(
                f"the {selection} selector mixes occupied and virtual orbitals "
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
                f"the {selection} selector had to rotate the occupied orbitals "
                f"(|R_oo - 1| = {occupied_shift:.2e}), which means the incoming "
                f"molecular orbitals did not diagonalize the Fock matrix.  The "
                f"frozen-core indices were resolved against the old labels and "
                f"would now name different orbitals, so the selection is "
                f"refused instead of applied to the wrong ones.  Use "
                f"active_selection='energy' with an explicit "
                f"active_orbitals=[...] for such a basis.")

    # The ranking is now the index order, so "keep the best m" is "keep the
    # first m".  Occupied truncation stays energy-ordered: the core is the
    # lowest orbitals, and `frozen` already holds any named by the caller.
    still_free = [p for p in range(n_doubly) if p not in set(frozen)]
    extra_frozen = still_free[:len(still_free) - n_occ_active]
    frozen = tuple(sorted(set(frozen) | set(extra_frozen)))
    active_occupied = [p for p in range(first_virtual) if p not in set(frozen)]
    active_virtual = list(range(first_virtual, first_virtual + n_virt_active))
    deleted = tuple(range(first_virtual + n_virt_active, M))
    return ActiveSpace(
        n_orbitals=M, frozen=frozen,
        active=tuple(active_occupied + active_virtual), deleted=deleted,
        selection=selection, rotation=rotation, occupations=occupations,
        correlation_energy=correlation)
