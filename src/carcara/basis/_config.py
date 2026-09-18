# -*- coding: utf-8 -*-
# file: basis/_config.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Ground-state electron configurations (aufbau filling).

Shared helpers used both by Slater's rules (:mod:`carcara.basis.fao`) and
by the numerical-atomic-orbital basis generation (:mod:`carcara.basis.nao`), so
the periodic filling order lives in exactly one place.

The same filling order also names the atom's **virtual** levels: the subshells
the aufbau filling never reached, in the order it would have reached them
(:func:`unoccupied_subshells`, used by the ``FAO`` basis's ``virtual_orbitals``
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


def valence_subshells(atomic_number: int) -> list[tuple[int, int]]:
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
    """
    config = ground_state_config(atomic_number)
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
