# -*- coding: utf-8 -*-
# file: algorithms/mcas.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Markov Chain Ansatz Search: the architecture chain, free of any Hamiltonian.

An **architecture** is an ordered tuple of pool indices
:math:`C = (\mu_1, \ldots, \mu_L)`; the first index acts first on the
reference.  Repeated indices are allowed and each occurrence has its own
angle.  This module owns everything about the chain that does not need a
quantum state -- the four architecture moves, their *summed* proposal
probabilities, the Metropolis-Hastings test and the temperature schedule -- so
it can be verified exhaustively against a synthetic cost (the transition matrix
of a small architecture space satisfies detailed balance to round-off) before
any energy enters.  :class:`~mandacaru.algorithms.vasqa.VASQA` drives it with
VQE energies.

Proposal convention
-------------------
A move ``m`` is chosen with :math:`p_m(C) = w_m I_m(C) / \sum_n w_n I_n(C)`,
where :math:`I_m(C)` says whether ``m`` has at least one valid action at ``C``;
the action inside the move is then uniform:

* ``insert`` -- a slot :math:`j \in \{0, \ldots, L\}` and a pool index,
  :math:`p = p_I / [(L+1) M]`; valid while :math:`L < L_{\max}`;
* ``delete`` -- a position, :math:`p = p_D / L`; valid while
  :math:`L > L_{\min}`;
* ``replace`` -- a position and a *different* index,
  :math:`p = p_R / [L (M-1)]`; valid for :math:`L \ge 1`, :math:`M \ge 2`;
* ``swap`` -- an unordered pair of positions holding *different* indices,
  :math:`p = p_S / N_{\mathrm{pairs}}(C)`; valid when such a pair exists.

Pairs of equal indices are excluded from ``swap`` rather than counted as
self-loops, and ``replace`` never re-draws the current index, so every
proposal changes the architecture.  The proposal density of a destination is
the sum over **every** action that produces it,
:math:`q(C'|C) = \sum_{a:\,T(C,a)=C'} p(a|C)` -- inserting ``A`` into ``(A)``
at either slot gives the same ``(A, A)``, and both count.

Acceptance is
:math:`\ell = -\beta[F(C') - F(C)] + \log q(C|C') - \log q(C'|C)`, accepted
when :math:`\log u < \min(0, \ell)`, always in log space, always against the
*current* state and rejected outright when the reverse move has no support.
``beta`` is an inverse *architecture* temperature: it weights circuit
descriptions by their optimized cost and has nothing to do with the
temperature of the physical system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Iterator, Mapping

import numpy as np

#: The four architecture moves, in the order they are reported.
MOVES = ("insert", "delete", "replace", "swap")

#: Equal move weights: each valid move is proposed equally often.
DEFAULT_MOVE_WEIGHTS = {move: 1.0 for move in MOVES}

Architecture = tuple[int, ...]


@dataclass(frozen=True)
class Action:
    """One architecture move with its arguments.

    ``insert``: ``position`` is the slot (``0..L``), ``operator`` the index.
    ``delete``: ``position`` is the occurrence removed.
    ``replace``: ``position`` is the occurrence, ``operator`` its new index.
    ``swap``: ``position < other`` are the two occurrences exchanged.
    """

    move: str
    position: int
    operator: int | None = None
    other: int | None = None

    def apply(self, architecture: Architecture) -> Architecture:
        """The architecture this action produces from ``architecture``."""
        c = tuple(architecture)
        j = self.position
        if self.move == "insert":
            return c[:j] + (self.operator,) + c[j:]
        if self.move == "delete":
            return c[:j] + c[j + 1:]
        if self.move == "replace":
            return c[:j] + (self.operator,) + c[j + 1:]
        if self.move == "swap":
            k = self.other
            out = list(c)
            out[j], out[k] = out[k], out[j]
            return tuple(out)
        raise ValueError(f"unknown move {self.move!r}")

    def transfer(self, parameters) -> np.ndarray:
        r"""Carry the angles of the old architecture to the new one.

        A new occurrence (``insert``, ``replace``) starts at zero angle, which
        for ``insert`` leaves the state exactly unchanged
        (:math:`e^{0 \cdot A} = I`); a deleted occurrence takes its angle with
        it; a swap moves each angle with its operator.
        """
        theta = np.asarray(parameters, dtype=float).ravel().copy()
        j = self.position
        if self.move == "insert":
            return np.insert(theta, j, 0.0)
        if self.move == "delete":
            return np.delete(theta, j)
        if self.move == "replace":
            theta[j] = 0.0
            return theta
        if self.move == "swap":
            k = self.other
            theta[j], theta[k] = theta[k], theta[j]
            return theta
        raise ValueError(f"unknown move {self.move!r}")

    def describe(self, labels=None) -> str:
        """``insert(A@2)``-style text; ``labels`` maps indices to names."""
        def name(index):
            return str(index) if labels is None else str(labels[index])
        if self.move == "insert":
            return f"insert({name(self.operator)}@{self.position})"
        if self.move == "delete":
            return f"delete({self.position})"
        if self.move == "replace":
            return f"replace({self.position}->{name(self.operator)})"
        return f"swap({self.position},{self.other})"


def _distinct_pairs(architecture: Architecture) -> int:
    """Unordered position pairs holding different indices."""
    length = len(architecture)
    counts: dict[int, int] = {}
    for index in architecture:
        counts[index] = counts.get(index, 0) + 1
    same = sum(n * (n - 1) // 2 for n in counts.values())
    return length * (length - 1) // 2 - same


class ProposalKernel:
    """The uniform MCAS proposal over architectures of bounded length.

    Parameters
    ----------
    pool_size : int
        Number of pool operators ``M``; indices run over ``0..M-1``.
    move_weights : dict, optional
        Non-negative weight of each move (default all ``1``).  A move left out
        has weight zero.  ``insert`` and ``delete`` must both be positive: the
        chain would otherwise be irreversible (every insertion rejected) or
        unable to change length.
    min_length, max_length : int
        Allowed architecture lengths, inclusive.
    """

    def __init__(self, pool_size: int, move_weights: Mapping | None = None,
                 min_length: int = 0, max_length: int = 20):
        self.pool_size = int(pool_size)
        if self.pool_size < 1:
            raise ValueError("the operator pool is empty: there is nothing "
                             "for the architecture search to place")
        weights = dict(DEFAULT_MOVE_WEIGHTS if move_weights is None
                       else move_weights)
        unknown = sorted(set(weights) - set(MOVES))
        if unknown:
            raise ValueError(f"unknown move(s) {unknown}; use {list(MOVES)}")
        self.move_weights = {move: float(weights.get(move, 0.0))
                             for move in MOVES}
        for move, weight in self.move_weights.items():
            if not (math.isfinite(weight) and weight >= 0.0):
                raise ValueError(f"move weight of {move!r} must be finite "
                                 f"and non-negative, got {weight!r}")
        if self.move_weights["insert"] <= 0 or self.move_weights["delete"] <= 0:
            raise ValueError(
                "insert and delete need positive weights: without deletion "
                "no insertion has a reverse move and every one is rejected, "
                "and without insertion the architecture can only shrink")
        self.min_length = int(min_length)
        self.max_length = int(max_length)
        if not 0 <= self.min_length < self.max_length:
            raise ValueError(f"need 0 <= min_length < max_length, got "
                             f"{self.min_length} and {self.max_length}")

    # -- validity and move probabilities ---------------------------------- #

    def valid_moves(self, architecture: Architecture) -> dict[str, bool]:
        """:math:`I_m(C)`: which moves have at least one action at ``C``."""
        length = len(architecture)
        return {
            "insert": length < self.max_length,
            "delete": length > self.min_length and length >= 1,
            "replace": length >= 1 and self.pool_size >= 2,
            "swap": _distinct_pairs(architecture) > 0,
        }

    def move_probabilities(self, architecture: Architecture
                           ) -> dict[str, float]:
        """:math:`p_m(C)`, normalized over the moves valid at ``C``."""
        valid = self.valid_moves(architecture)
        weight = {m: self.move_weights[m] * valid[m] for m in MOVES}
        total = sum(weight.values())
        if total <= 0.0:
            return {m: 0.0 for m in MOVES}
        return {m: weight[m] / total for m in MOVES}

    # -- actions ----------------------------------------------------------- #

    def actions(self, architecture: Architecture
                ) -> Iterator[tuple[Action, float]]:
        """Every action at ``C`` with its probability :math:`p(a|C)`.

        Exhaustive, so its cost grows as ``L * M``; :meth:`log_q` computes the
        same sums in closed form and this is what it is checked against.
        """
        c = tuple(architecture)
        length, pool = len(c), self.pool_size
        p = self.move_probabilities(c)
        if p["insert"] > 0:
            each = p["insert"] / ((length + 1) * pool)
            for j in range(length + 1):
                for mu in range(pool):
                    yield Action("insert", j, operator=mu), each
        if p["delete"] > 0:
            each = p["delete"] / length
            for j in range(length):
                yield Action("delete", j), each
        if p["replace"] > 0:
            each = p["replace"] / (length * (pool - 1))
            for j in range(length):
                for mu in range(pool):
                    if mu != c[j]:
                        yield Action("replace", j, operator=mu), each
        if p["swap"] > 0:
            each = p["swap"] / _distinct_pairs(c)
            for i in range(length):
                for k in range(i + 1, length):
                    if c[i] != c[k]:
                        yield Action("swap", i, other=k), each

    def sample(self, architecture: Architecture, rng: np.random.Generator
               ) -> Action | None:
        """Draw one action at ``C``, or ``None`` when no move is valid."""
        c = tuple(architecture)
        length, pool = len(c), self.pool_size
        p = self.move_probabilities(c)
        if sum(p.values()) <= 0.0:
            return None
        move = MOVES[int(rng.choice(len(MOVES), p=[p[m] for m in MOVES]))]
        if move == "insert":
            return Action("insert", int(rng.integers(length + 1)),
                          operator=int(rng.integers(pool)))
        if move == "delete":
            return Action("delete", int(rng.integers(length)))
        if move == "replace":
            j = int(rng.integers(length))
            # Uniform over the M - 1 indices other than the current one.
            mu = int(rng.integers(pool - 1))
            if mu >= c[j]:
                mu += 1
            return Action("replace", j, operator=mu)
        pairs = [(i, k) for i in range(length) for k in range(i + 1, length)
                 if c[i] != c[k]]
        i, k = pairs[int(rng.integers(len(pairs)))]
        return Action("swap", i, other=k)

    # -- summed architecture proposal ------------------------------------- #

    def log_q(self, destination: Architecture, source: Architecture) -> float:
        r""":math:`\log q(C'|C)`, summed over every action from ``source``
        that produces ``destination``; ``-inf`` when none does."""
        c, d = tuple(source), tuple(destination)
        length, pool = len(c), self.pool_size
        p = self.move_probabilities(c)
        total = 0.0
        if len(d) == length + 1 and p["insert"] > 0:
            ways = sum(1 for j in range(len(d)) if d[:j] + d[j + 1:] == c)
            total += p["insert"] * ways / ((length + 1) * pool)
        elif len(d) == length - 1 and p["delete"] > 0:
            ways = sum(1 for j in range(length) if c[:j] + c[j + 1:] == d)
            total += p["delete"] * ways / length
        elif len(d) == length and length:
            differ = [j for j in range(length) if c[j] != d[j]]
            if len(differ) == 1 and p["replace"] > 0:
                total += p["replace"] / (length * (pool - 1))
            elif (len(differ) == 2 and p["swap"] > 0
                  and c[differ[0]] == d[differ[1]]
                  and c[differ[1]] == d[differ[0]]):
                total += p["swap"] / _distinct_pairs(c)
        return math.log(total) if total > 0.0 else -math.inf


# --------------------------------------------------------------------------- #
# Metropolis-Hastings.
# --------------------------------------------------------------------------- #

def log_acceptance(beta: float, cost_current: float, cost_proposed: float,
                   log_q_forward: float, log_q_reverse: float) -> float:
    r"""Log of the Metropolis-Hastings ratio, before the ``min(0, .)``.

    :math:`\ell = -\beta\,[F(C') - F(C)] + \log q(C|C') - \log q(C'|C)`.
    Returns ``-inf`` -- a certain rejection -- when the proposed cost is not
    finite or the reverse move has no support.  ``beta = inf`` is the
    zero-temperature limit: a decrease is accepted, an increase rejected and a
    tie decided by the Hastings ratio alone.
    """
    if not math.isfinite(cost_proposed) or log_q_reverse == -math.inf:
        return -math.inf
    if not math.isfinite(log_q_forward):
        raise ValueError("the forward proposal has no support: the proposed "
                         "architecture cannot have been drawn from the current")
    hastings = log_q_reverse - log_q_forward
    delta = cost_proposed - cost_current
    if math.isinf(beta):
        if delta < 0:
            return math.inf
        if delta > 0:
            return -math.inf
        return hastings
    return -beta * delta + hastings


def metropolis_accept(log_alpha: float, rng: np.random.Generator) -> bool:
    r"""Accept when :math:`\log u < \min(0, \ell)`, ``u ~ U(0, 1)``."""
    if log_alpha >= 0.0:
        # Still draw, so the random stream does not depend on the outcome.
        rng.random()
        return True
    if log_alpha == -math.inf:
        rng.random()
        return False
    return math.log(rng.random()) < log_alpha


# --------------------------------------------------------------------------- #
# Temperature schedule.
# --------------------------------------------------------------------------- #

class TemperatureSchedule:
    r"""The architecture temperature :math:`T_k` at each step (Hartree).

    ``temperature`` is either one number -- a fixed-temperature chain -- or
    ``{"initial": T0, "final": T1}``, a geometric anneal
    :math:`T_k = T_0 (T_1/T_0)^{k/(K-1)}` over ``steps`` = ``K`` steps.
    ``0`` is the zero-temperature (greedy) limit, ``beta = inf``.
    """

    def __init__(self, temperature, steps: int):
        self.steps = max(int(steps), 1)
        if isinstance(temperature, Mapping):
            unknown = sorted(set(temperature) - {"initial", "final"})
            if unknown or "initial" not in temperature \
                    or "final" not in temperature:
                raise ValueError(
                    "an annealed temperature is {'initial': T0, 'final': T1}"
                    + (f"; unknown keys {unknown}" if unknown else ""))
            self.initial = float(temperature["initial"])
            self.final = float(temperature["final"])
            if not (self.initial > 0.0 and self.final > 0.0):
                raise ValueError("annealing temperatures must be positive")
        else:
            self.initial = self.final = float(temperature)
            if not (math.isfinite(self.initial) and self.initial >= 0.0):
                raise ValueError(f"temperature must be finite and >= 0, got "
                                 f"{temperature!r}")

    @property
    def annealed(self) -> bool:
        return self.initial != self.final

    def temperature(self, step: int) -> float:
        if not self.annealed or self.steps == 1:
            return self.initial
        fraction = min(max(int(step), 0), self.steps - 1) / (self.steps - 1)
        return self.initial * (self.final / self.initial) ** fraction

    def beta(self, step: int) -> float:
        t = self.temperature(step)
        return math.inf if t == 0.0 else 1.0 / t
