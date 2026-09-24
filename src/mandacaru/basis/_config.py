# -*- coding: utf-8 -*-
# file: basis/_config.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Ground-state electron configurations (aufbau filling).

Shared helpers used both by Slater's rules (:mod:`mandacaru.basis.hao`) and
by the numerical-atomic-orbital basis generation (:mod:`mandacaru.basis.nao`), so
the periodic filling order lives in exactly one place.

The same filling order also names the atom's **virtual** levels: the subshells
the aufbau filling never reached, in the order it would have reached them
(:func:`unoccupied_subshells`, used by the ``HAO`` basis's ``virtual_orbitals``
option).
"""

from __future__ import annotations

# Ground-state aufbau (Madelung) filling order as (n, l), with l-subshell
# capacities.
_AUFBAU_ORDER = [
    (1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (3, 2), (4, 1),
    (5, 0), (4, 2), (5, 1), (6, 0), (4, 3), (5, 2), (6, 1), (7, 0),
    (5, 3), (6, 2), (7, 1),
]
_L_CAPACITY = {0: 2, 1: 6, 2: 10, 3: 14}
# Slater groups s and p of the same shell together; d and f stand alone.  The
# order value ranks groups from innermost (screening most) to outermost.
_SLATER_GROUP_ORDER = {"sp": 0, "d": 1, "f": 2}


def _slater_group(l: int) -> str:
    return "sp" if l in (0, 1) else ("d" if l == 2 else "f")


def split_configuration(configuration):
    """Resolve ``{(n, l): q}`` into ``{(n, l, kappa): q}`` for a Dirac atom.

    Each subshell is shared between its two ``j = l +/- 1/2`` levels in
    proportion to their ``2j + 1`` degeneracies, and a **partially** filled
    subshell fills the lower level first -- ``j = l - 1/2`` (``kappa = +l``),
    which lies below ``j = l + 1/2`` for the ordinary (non-inverted) ordering
    of a neutral atom.  An s subshell has one level and is passed through.

    This is the j-j coupled filling; it is not the same state an LS-coupled
    Hund's-rule treatment would occupy for an open shell, and for a spherical
    reference atom -- which is already an average over the multiplet -- the
    difference is absorbed into the spherical average.
    """
    from .relativity import degeneracy, kappa_values

    resolved: dict[tuple[int, int, int], float] = {}
    for (n, l), occupancy in configuration.items():
        left = float(occupancy)
        for k in kappa_values(int(l)):        # ordered j = l-1/2 first
            capacity = float(degeneracy(k))
            resolved[(int(n), int(l), k)] = min(left, capacity)
            left = max(left - capacity, 0.0)
    return resolved


def ground_state_config(atomic_number: int) -> dict[tuple[int, int], int]:
    """Neutral-atom ground-state configuration as ``{(n, l): occupancy}``.

    Fills subshells in aufbau order until ``atomic_number`` electrons are placed.
    """
    Z = int(atomic_number)
    if Z < 1:
        raise ValueError(f"atomic_number must be >= 1, got {atomic_number}")
    config: dict[tuple[int, int], int] = {}
    remaining = Z
    for (n, l) in _AUFBAU_ORDER:
        if remaining <= 0:
            break
        occ = min(_L_CAPACITY[l], remaining)
        remaining -= occ
        config[(n, l)] = occ
    if remaining > 0:
        raise ValueError(
            f"atomic number {Z} is beyond the supported filling table")
    return config


def valence_subshells(atomic_number: int, configuration=None
                      ) -> list[tuple[int, int]]:
    """Outermost occupied ``(n, l)`` subshells -- a minimal-basis valence set.

    Returns every occupied subshell sharing the highest occupied principal
    quantum number, **plus the semicore** ``(n-1)d`` and ``(n-2)f`` shells when
    they are occupied.

    Taking only the highest ``n`` reproduces the chemical valence for the main
    group (C -> ``[(2, 0), (2, 1)]``, Na -> ``[(3, 0)]``, Si -> ``[(3, 0),
    (3, 1)]``) but fails badly for the d and f blocks: iron is
    ``[Ar] 3d^6 4s^2``, whose highest ``n`` is 4, so the rule alone would call
    it a two-electron atom and drop the 3d shell that carries all of its
    chemistry.  The d and f electrons are spatially comparable to the outer s
    shell and must be treated as valence -- both for the basis and for
    pseudopotential generation, where an ``[Ar]``-core iron would have no d
    channel at all.  So Fe -> ``[(3, 2), (4, 0)]`` (8 valence electrons) and
    Ce -> ``[(4, 3), (5, 2), (6, 0)]``.

    ``configuration`` overrides the aufbau filling, and a caller that has
    already solved the atom should pass the atom's own ``occupations``.  The two
    differ wherever :func:`mandacaru.basis.atomic_solver.relaxed_configuration`
    moved an electron: lanthanum's aufbau valence is ``[(4, 3), (6, 0)]``, and
    the configuration LDA actually prefers makes it ``[(5, 2), (6, 0)]``.
    Deriving the valence from the aufbau filling while the orbitals came from
    the relaxed one asks for a ``4f`` the atom never solved.
    """
    config = (ground_state_config(atomic_number) if configuration is None
              else {k: v for k, v in configuration.items() if v > 0})
    n_max = max(n for (n, _l) in config)
    valence = {(n, l) for (n, l) in config if n == n_max}
    for l, offset in ((2, 1), (3, 2)):              # (n-1)d, (n-2)f
        state = (n_max - offset, l)
        if state in config:
            valence.add(state)
    return sorted(valence)


def unoccupied_subshells(atomic_number: int, count: int = 1
                         ) -> list[tuple[int, int]]:
    """The ``count`` lowest **unoccupied** ``(n, l)`` subshells, in aufbau order.

    The neutral atom's ground state fills subshells in aufbau (Madelung) order;
    the ones the filling never reached are its virtual levels, and the order it
    would have reached them in is the order they come back in.

    A **partially filled** subshell counts as occupied: carbon's ``2p^2`` is
    already a basis shell, so carbon's lowest virtual level is ``3s``, not the
    empty half of ``2p``.  Hydrogen's is ``2s``; iron's (``[Ar] 3d^6 4s^2``) is
    ``4p``.

    Parameters
    ----------
    atomic_number : int
        The element.
    count : int
        How many virtual subshells to return (``0`` gives an empty list).

    Raises
    ------
    ValueError
        If ``count`` is negative, or exceeds the virtual levels the filling
        table holds for this element -- the table stops at
        ``_AUFBAU_ORDER[-1]``, so a heavy atom has few levels above its
        occupied set.
    """
    count = int(count)
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    if count == 0:
        return []
    occupied = ground_state_config(atomic_number)
    empty = [state for state in _AUFBAU_ORDER if state not in occupied]
    if len(empty) < count:
        n_last, l_last = _AUFBAU_ORDER[-1]
        raise ValueError(
            f"element Z={int(atomic_number)} has only {len(empty)} unoccupied "
            f"subshell(s) below the end of the aufbau filling table "
            f"({n_last}{'spdf'[l_last]}), but {count} were requested")
    return empty[:count]

#: ``l`` values the aufbau order interleaves, and so can order wrongly: the
#: Madelung rule crosses ``(n-1)d`` with ``ns`` and ``(n-2)f`` with ``ns``, and
#: those are the crossings the neutral-atom anomalies live on.  ``p`` is never
#: in question -- no element's ground state is settled by moving an electron
#: between ``ns`` and ``np``.
_COMPETING_L = (0, 2, 3)


def rearrangements(atomic_number: int, virtuals: int = 3
                   ) -> list[dict[tuple[int, int], int]]:
    """Aufbau, then every **one-electron** rearrangement of its valence.

    The aufbau (Madelung) order is a rule of thumb, and where it is wrong it is
    wrong about exactly one electron: lanthanum is ``5d^1 6s^2`` and not
    ``4f^1 6s^2``, chromium ``3d^5 4s^1`` and not ``3d^4 4s^2``.  Which of the
    two a *functional* prefers is a question with a computable answer, so this
    function enumerates the candidates and
    :func:`mandacaru.basis.atomic_solver.relaxed_configuration` picks the lowest
    in energy.  No table of experimental configurations enters the repository.

    A move takes one electron from an occupied valence subshell to another
    valence subshell or to one of the ``virtuals`` lowest unoccupied ones (so
    lanthanum's empty ``5d`` is reachable), never emptying, never overfilling
    and never entering a principal shell above the outermost occupied one.
    Both subshells must have different ``l`` drawn from
    :data:`_COMPETING_L`, which is what keeps the list short and physical: for
    oxygen it yields no candidate at all, because ``2s -> 2p`` is not a
    crossing the aufbau order can get wrong, and there is no point solving the
    atom twice to discover that.  The aufbau configuration itself is always
    first.

    Single moves are enough for the cases that matter here: they turn La's
    unbound ``4f`` into a bound ``5d``, and thorium's ``5f^2`` into
    ``5f^1 6d^1``, which is what LDA prefers over both ``5f^2`` and ``6d^2``.
    They cannot reach a two-electron anomaly such as palladium's ``4d^10``, and
    this function does not claim to reproduce experiment -- only to let the
    functional choose.
    """
    base = ground_state_config(atomic_number)
    sources = [s for s in valence_subshells(atomic_number) if base.get(s, 0) > 0]
    empty: list[tuple[int, int]] = []
    for count in range(int(virtuals), 0, -1):
        try:                       # the filling table runs out for heavy atoms
            empty = unoccupied_subshells(atomic_number, count)
            break
        except ValueError:
            continue
    # A destination in a *new* principal shell is an excitation, not a
    # reordering: the Madelung crossings that can be wrong are between shells
    # already in play.  Without this bound, lanthanum's lowest-energy candidate
    # came out as 4f -> 7s, whose valence set is the single 7s electron -- a
    # one-electron lanthanum, which is not a reference atom, it is a mistake.
    n_outer = max(n for (n, _l) in base)
    targets = [s for s in dict.fromkeys(list(sources) + list(empty))
               if s[0] <= n_outer]
    out = [dict(base)]
    for src in sources:
        if src[1] not in _COMPETING_L:
            continue
        for dst in targets:
            if dst[1] not in _COMPETING_L or dst[1] == src[1]:
                continue
            if base.get(dst, 0) >= _L_CAPACITY[dst[1]]:
                continue
            moved = dict(base)
            moved[src] = moved[src] - 1
            moved[dst] = moved.get(dst, 0) + 1
            out.append({k: v for k, v in moved.items() if v > 0})
    return out
