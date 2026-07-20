# -*- coding: utf-8 -*-
# file: algorithms/vasqe.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""VASQE: the Variational Adaptive Stochastic Quantum Eigensolver.

VASQE is ADAPT-VQE with a **stochastic operator-selection** rule.  Where ADAPT-VQE
greedily appends the pool operator with the largest gradient magnitude, VASQE draws
the operator from a Boltzmann-like softmax of the gradients at a *selection
temperature* :math:`\tau`:

.. math::

    P(i, \tau) = \frac{\exp\!\big(|g_i|/\tau\big)}
                      {\sum_j \exp\!\big(|g_j|/\tau\big)} .

At low :math:`\tau` the probability concentrates on the largest-gradient operator,
so VASQE **reduces to ADAPT-VQE**; at high :math:`\tau` the distribution flattens
toward uniform, letting the ansatz explore operators a greedy rule would never
pick.  The temperature can be held **constant** or **annealed** from a high
initial value to a low final one over the growth iterations, with three cooling
schedules -- **exponential**, **linear** and **logarithmic** -- so early
iterations explore and late ones exploit.

Convergence is unchanged from ADAPT-VQE (``max_i |g_i| < gradient_tolerance``): the
stochastic rule only changes *which* operator is appended, never the stopping
test.  Because selection is the single :meth:`ADAPTVQE._select_operator` hook,
VASQE inherits the excited-state machinery for free -- :meth:`~VASQE.energy_levels`
(deflation) and :class:`SubspaceVASQE` (subspace search) both grow their ansätze
stochastically.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np

from .adapt_vqe import ADAPTVQE, ADAPTVQEResult
from .subspace import SubspaceADAPTVQE

#: Supported selection-temperature schedules.
TEMPERATURE_SCHEDULES = ("constant", "exponential", "linear", "logarithmic")


# --------------------------------------------------------------------------- #
# Selection probabilities and temperature schedule.
# --------------------------------------------------------------------------- #

def softmax_selection_probabilities(gradients, tau: float) -> np.ndarray:
    r"""Softmax selection probabilities ``P(i) = exp(|g_i|/tau) / Z``.

    Uses absolute gradients and is numerically stabilized by subtracting the max
    exponent.  As ``tau -> 0`` this tends to a one-hot on ``argmax|g|`` (the
    ADAPT-VQE choice); as ``tau -> inf`` it tends to uniform.  ``tau <= 0`` returns
    the deterministic one-hot on the largest gradient.
    """
    g = np.abs(np.asarray(gradients, dtype=float))
    n = g.size
    if n == 0:
        return g
    if tau <= 0.0:
        p = np.zeros(n)
        p[int(np.argmax(g))] = 1.0
        return p
    z = g / float(tau)
    z -= z.max()                      # stabilize; largest exponent becomes 0
    p = np.exp(z)
    total = p.sum()
    return p / total if total > 0 else np.full(n, 1.0 / n)


def annealed_temperature(schedule: str, initial: float, final: float,
                         step: int, horizon: int) -> float:
    r"""Selection temperature at growth ``step`` of a ``horizon``-step schedule.

    ``"constant"`` ignores ``final`` and returns ``initial``.  The annealing
    schedules interpolate ``initial -> final`` as ``step`` goes ``0 -> horizon-1``:
    ``"linear"`` in the step, ``"exponential"`` geometrically (constant ratio per
    step), and ``"logarithmic"`` along ``log(1+step)`` (slow early cooling).
    ``step`` is clamped to ``[0, horizon-1]``.
    """
    if schedule == "constant" or horizon <= 1:
        return float(initial)
    k = min(max(int(step), 0), horizon - 1)
    p = k / (horizon - 1)                       # linear progress in [0, 1]
    if schedule == "linear":
        return float(initial + (final - initial) * p)
    if schedule == "exponential":
        return float(initial * (final / initial) ** p)
    if schedule == "logarithmic":
        plog = math.log1p(k) / math.log(horizon)     # in [0, 1], slow at first
        return float(initial + (final - initial) * plog)
    raise ValueError(
        f"unknown schedule {schedule!r}; use one of {TEMPERATURE_SCHEDULES}")


# --------------------------------------------------------------------------- #
# Result container.
# --------------------------------------------------------------------------- #

@dataclass
class VASQEResult(ADAPTVQEResult):
    """Result of a :class:`VASQE` run: an :class:`ADAPTVQEResult` plus the schedule.

    Adds the selection-temperature trace and schedule used to grow the ansatz.
    """

    temperatures: list[float] = field(default_factory=list)  # tau per grown op
    schedule: str = "constant"
    initial_temperature: float = 1.0
    final_temperature: float = 1.0

    def __repr__(self) -> str:
        cnots = self.metrics.cnot_count if self.metrics else None
        return (f"VASQEResult(energy={self.optimal_energy:.6f}, "
                f"n_ops={self.num_operators}, cnots={cnots}, "
                f"schedule={self.schedule!r}, converged={self.converged})")


# --------------------------------------------------------------------------- #
# VASQE driver.
# --------------------------------------------------------------------------- #

class VASQE(ADAPTVQE):
    """Variational Adaptive Stochastic Quantum Eigensolver.

    Subclasses :class:`~carcara.algorithms.adapt_vqe.ADAPTVQE`; every ``ADAPTVQE``
    argument is accepted, plus the stochastic-selection controls below.  All the
    ADAPT machinery (pools, gradients, growable ansatz, circuit profiling, ASE
    calculator mode, frozen core, ...) is reused unchanged -- only the
    operator-selection rule differs.

    Parameters
    ----------
    temperature : float
        Selection temperature :math:`\\tau` (default ``1.0``).  For the
        ``"constant"`` schedule it is used throughout; for an annealing schedule it
        is the **initial** (high, exploratory) temperature.
    final_temperature : float
        Target temperature at the end of annealing (default ``1e-2``; ignored by
        the ``"constant"`` schedule).  A small value makes late iterations behave
        greedily (ADAPT-like).
    schedule : str
        Temperature schedule -- ``"constant"`` (default), ``"exponential"``,
        ``"linear"`` or ``"logarithmic"``.  Annealing runs
        ``temperature -> final_temperature`` over ``annealing_steps`` iterations.
    annealing_steps : int, optional
        Number of iterations the schedule spans (default: ``max_iterations``).
    seed : int
        Seed for the selection RNG (reproducible sampling; default ``0``).

    Notes
    -----
    Selection is the only change: the convergence test is still
    ``max_i |g_i| < gradient_tolerance``.  As ``temperature -> 0`` VASQE reproduces
    ADAPT-VQE exactly.  The excited-state extensions (:meth:`energy_levels` via
    deflation, and :class:`SubspaceVASQE`) inherit the stochastic selection.
    """

    _result_attr = "vasqe_result"

    def __init__(self, hamiltonian=None, pool="fermionic", *,
                 temperature: float = 1.0, final_temperature: float = 1e-2,
                 schedule: str = "constant", annealing_steps: int | None = None,
                 seed: int = 0, **kwargs):
        if schedule not in TEMPERATURE_SCHEDULES:
            raise ValueError(
                f"unknown schedule {schedule!r}; use one of {TEMPERATURE_SCHEDULES}")
        if float(temperature) <= 0 or float(final_temperature) <= 0:
            raise ValueError("temperature and final_temperature must be positive")
        self.initial_temperature = float(temperature)
        self.final_temperature = float(final_temperature)
        self.schedule = schedule
        self.annealing_steps = (None if annealing_steps is None
                                else int(annealing_steps))
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._temperatures: list[float] = []
        super().__init__(hamiltonian, pool, **kwargs)

    # -- stochastic operator selection ----------------------------------- #

    def _temperature(self, iteration: int) -> float:
        """Selection temperature at growth ``iteration`` (0-based)."""
        horizon = (self.annealing_steps if self.annealing_steps is not None
                   else self.max_iterations)
        return annealed_temperature(self.schedule, self.initial_temperature,
                                    self.final_temperature, iteration,
                                    max(int(horizon), 1))

    def _select_operator(self, grads: np.ndarray, iteration: int) -> int:
        """Sample the operator index from the softmax of the pool gradients."""
        tau = self._temperature(iteration)
        self._temperatures.append(tau)
        probs = softmax_selection_probabilities(grads, tau)
        return int(self._rng.choice(probs.size, p=probs))

    # -- run: wrap ADAPT run to emit a VASQEResult ----------------------- #

    def run(self, *args, **kwargs) -> VASQEResult:
        """Grow the ansatz with stochastic selection; return a :class:`VASQEResult`."""
        self._rng = np.random.default_rng(self.seed)     # reproducible per run
        self._temperatures = []
        result = super().run(*args, **kwargs)
        return self._as_vasqe_result(result)

    def _as_vasqe_result(self, result: ADAPTVQEResult) -> VASQEResult:
        fields = {f.name: getattr(result, f.name)
                  for f in dataclasses.fields(ADAPTVQEResult)}
        return VASQEResult(**fields, temperatures=list(self._temperatures),
                           schedule=self.schedule,
                           initial_temperature=self.initial_temperature,
                           final_temperature=self.final_temperature)

    # -- verbose header extra -------------------------------------------- #

    def _extra_header_lines(self) -> list[str]:
        """Report the stochastic-selection temperature schedule in the header."""
        horizon = (self.annealing_steps if self.annealing_steps is not None
                   else self.max_iterations)
        if self.schedule == "constant":
            temp = f"tau = {self.initial_temperature:g} (constant)"
        else:
            temp = (f"tau: {self.initial_temperature:g} -> "
                    f"{self.final_temperature:g} ({self.schedule}, "
                    f"{horizon} steps)")
        return [*super()._extra_header_lines(),
                f"VASQE stochastic selection  |  {temp}  |  seed: {self.seed}"]


# --------------------------------------------------------------------------- #
# Subspace-search VASQE (stochastic selection, one shared ansatz).
# --------------------------------------------------------------------------- #

class SubspaceVASQE(SubspaceADAPTVQE, VASQE):
    """Subspace-search VASQE: SSVQE with stochastic operator selection.

    Combines :class:`~carcara.algorithms.subspace.SubspaceADAPTVQE` (one shared
    adaptively grown ansatz over several orthogonal references, weighted gradient
    screening) with :class:`VASQE`'s softmax selection: the shared ansatz grows by
    sampling operators from the weighted-gradient softmax at the selection
    temperature.  Accepts every ``SubspaceADAPTVQE`` argument (``num_states`` /
    ``weights``) plus every :class:`VASQE` control (``temperature`` /
    ``final_temperature`` / ``schedule`` / ``annealing_steps`` / ``seed``).
    """

    def __init__(self, hamiltonian=None, pool="fermionic", *,
                 num_states: int = 2, weights=None, temperature: float = 1.0,
                 final_temperature: float = 1e-2, schedule: str = "constant",
                 annealing_steps: int | None = None, seed: int = 0, **kwargs):
        # Route the VASQE controls through **kwargs to VASQE.__init__ (which sits
        # after SubspaceADAPTVQE in the MRO), and the subspace controls here.
        super().__init__(hamiltonian, pool, num_states=num_states,
                         weights=weights, temperature=temperature,
                         final_temperature=final_temperature, schedule=schedule,
                         annealing_steps=annealing_steps, seed=seed, **kwargs)
