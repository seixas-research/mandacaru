# -*- coding: utf-8 -*-
# file: optimizers/optim.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Classical optimizers for the variational (hybrid) loop.

:class:`Optimizer` is a thin wrapper exposing the six methods used to drive the
VQE / ADAPT-VQE parameter minimization behind one interface, recording the cost
history so a convergence trace is always available:

* **SPSA** (Simultaneous Perturbation Stochastic Approximation) -- a two-evaluation
  stochastic-gradient method, implemented natively here;
* **COBYLA** (Constrained Optimization BY Linear Approximation) -- SciPy, default;
* **Nelder-Mead** -- SciPy simplex;
* **SLSQP** (Sequential Least Squares Programming) -- SciPy;
* **Adam** (Adaptive Moment Estimation) -- a finite-difference-gradient adaptive
  first-order method, implemented natively here;
* **L-BFGS-B** -- SciPy quasi-Newton.

COBYLA, Nelder-Mead, SLSQP and L-BFGS-B are dispatched to
``scipy.optimize.minimize``; SPSA and Adam are implemented natively (SciPy has no
equivalent) but share the same :meth:`Optimizer.minimize` interface and
:class:`OptimizeResult` output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable, Sequence

import numpy as np
from scipy.optimize import minimize


@dataclass
class OptimizeResult:
    """Outcome of an optimization run."""

    x: np.ndarray                         # optimal parameters
    fun: float                            # optimal cost
    nfev: int                             # number of cost evaluations
    history: list[float] = field(default_factory=list)  # cost per evaluation
    success: bool = True
    message: str = ""


# The optimization methods exposed by name to the variational drivers
# (VQE, ADAPTVQE).  COBYLA is the default everywhere.
NAMED_OPTIMIZERS = ("SPSA", "COBYLA", "Nelder-Mead", "SLSQP", "Adam", "L-BFGS-B")

# Methods routed to scipy.optimize.minimize vs. implemented natively below.
_SCIPY_METHODS = ("COBYLA", "Nelder-Mead", "SLSQP", "L-BFGS-B")
_CUSTOM_METHODS = ("SPSA", "Adam")


def resolve_optimizer(optimizer, allowed=NAMED_OPTIMIZERS,
                      maxiter: int = 2000) -> "Optimizer":
    """Normalize an ``optimizer`` argument to an :class:`Optimizer`.

    Accepts either a pre-built :class:`Optimizer` (returned unchanged) or a
    method name from ``allowed`` (one of ``"SPSA"``, ``"COBYLA"``,
    ``"Nelder-Mead"``, ``"SLSQP"``, ``"Adam"``, ``"L-BFGS-B"``), which is wrapped
    in a fresh :class:`Optimizer`.  Shared by the VQE and ADAPT-VQE drivers so
    both expose the same ``optimizer=`` surface.
    """
    if isinstance(optimizer, Optimizer):
        return optimizer
    if isinstance(optimizer, str):
        if optimizer not in allowed:
            raise ValueError(
                f"unknown optimizer {optimizer!r}; use one of {tuple(allowed)} "
                "or an Optimizer instance")
        return Optimizer(method=optimizer, maxiter=maxiter)
    raise TypeError("optimizer must be a method name or an Optimizer instance")


class Optimizer:
    """Classical optimizer with cost-history tracking.

    Parameters
    ----------
    method : str
        Optimization method (default ``"COBYLA"``).  One of ``"SPSA"``,
        ``"COBYLA"``, ``"Nelder-Mead"``, ``"SLSQP"``, ``"Adam"``, ``"L-BFGS-B"``.
        COBYLA, Nelder-Mead, SLSQP and L-BFGS-B go through
        ``scipy.optimize.minimize``; SPSA and Adam are implemented natively.
    maxiter : int
        Maximum iterations (default ``1000``).  For the SciPy methods this is the
        ``maxiter`` option; for SPSA and Adam it is the number of update steps.
    tol : float, optional
        Convergence tolerance.  Passed to SciPy for the SciPy methods; used as the
        step/cost-change stopping threshold for SPSA and Adam.
    options : dict, optional
        Extra options.  Forwarded to ``scipy.optimize.minimize`` for the SciPy
        methods; the SPSA / Adam hyperparameters (see :meth:`_minimize_spsa` /
        :meth:`_minimize_adam`) are read from here for the native methods.
    seed : int, optional
        Seed for the SPSA perturbation RNG (default ``0``); makes runs
        reproducible.  Ignored by the deterministic methods.
    """

    def __init__(self, method: str = "COBYLA", maxiter: int = 1000,
                 tol: float | None = None, options: dict | None = None,
                 seed: int = 0):
        if method not in NAMED_OPTIMIZERS:
            raise ValueError(
                f"unknown optimizer method {method!r}; use one of "
                f"{NAMED_OPTIMIZERS}")
        self.method = method
        self.maxiter = int(maxiter)
        self.tol = tol
        self.options = dict(options or {})
        self.seed = int(seed)

    def minimize(self, cost: Callable[[np.ndarray], float],
                 x0: Sequence[float]) -> OptimizeResult:
        """Minimize ``cost`` starting from ``x0``."""
        x0 = np.asarray(x0, dtype=float).ravel()
        history: list[float] = []

        def wrapped(x):
            value = float(cost(x))
            history.append(value)
            return value

        # A zero-parameter ansatz has nothing to optimize -- evaluate once.
        if x0.size == 0:
            value = wrapped(x0)
            return OptimizeResult(x=x0, fun=value, nfev=1, history=history,
                                  success=True, message="no free parameters")

        if self.method in _CUSTOM_METHODS:
            if self.method == "SPSA":
                x, fun = self._minimize_spsa(wrapped, x0)
            else:                                    # "Adam"
                x, fun = self._minimize_adam(wrapped, x0)
            return OptimizeResult(
                x=np.asarray(x, dtype=float), fun=float(fun),
                nfev=len(history), history=history, success=True,
                message=f"{self.method} finished ({self.maxiter} iterations)")

        options = {"maxiter": self.maxiter, **self.options}
        res = minimize(wrapped, x0, method=self.method, tol=self.tol,
                       options=options)
        return OptimizeResult(
            x=np.asarray(res.x, dtype=float), fun=float(res.fun),
            nfev=len(history), history=history,
            success=bool(res.success), message=str(res.message))

    # -- native optimizers ------------------------------------------------- #

    def _minimize_spsa(self, cost, x0):
        """SPSA -- Spall's simultaneous perturbation stochastic approximation.

        Each step estimates the gradient from just **two** cost evaluations at
        ``x +/- c_k * delta`` with a random ``+/-1`` perturbation ``delta``, then
        takes a decaying step ``a_k``.  Gain sequences follow Spall's practical
        recommendations, tunable through ``options``: ``a`` (0.2), ``c`` (0.1),
        ``alpha`` (0.602), ``gamma`` (0.101) and the stability constant ``A``
        (``0.1 * maxiter``).
        """
        o = self.options
        a = float(o.get("a", 0.2))
        c = float(o.get("c", 0.1))
        alpha = float(o.get("alpha", 0.602))
        gamma = float(o.get("gamma", 0.101))
        A = float(o.get("A", 0.1 * self.maxiter))
        tol = self.tol if self.tol is not None else 0.0

        rng = np.random.default_rng(self.seed)
        x = np.array(x0, dtype=float)
        best_x, best_f = x.copy(), cost(x)
        for k in range(self.maxiter):
            ak = a / (k + 1 + A) ** alpha
            ck = c / (k + 1) ** gamma
            delta = rng.choice([-1.0, 1.0], size=x.shape)
            fp = cost(x + ck * delta)
            fm = cost(x - ck * delta)
            ghat = (fp - fm) / (2.0 * ck) * (1.0 / delta)
            x = x - ak * ghat
            f = cost(x)
            if f < best_f:
                best_x, best_f = x.copy(), f
            if tol and ak * np.linalg.norm(ghat) < tol:
                break
        return best_x, best_f

    def _minimize_adam(self, cost, x0):
        """Adam -- adaptive moment estimation on a finite-difference gradient.

        The cost is a black box, so the gradient is estimated by central finite
        differences (``2N`` evaluations per step).  Hyperparameters are tunable
        through ``options``: ``lr`` (0.05), ``beta1`` (0.9), ``beta2`` (0.999),
        ``eps`` (1e-8) and the finite-difference step ``fd_eps`` (1e-4).
        """
        o = self.options
        lr = float(o.get("lr", 0.05))
        beta1 = float(o.get("beta1", 0.9))
        beta2 = float(o.get("beta2", 0.999))
        eps = float(o.get("eps", 1e-8))
        fd_eps = float(o.get("fd_eps", 1e-4))
        tol = self.tol if self.tol is not None else 0.0

        x = np.array(x0, dtype=float)
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        best_x, best_f = x.copy(), cost(x)
        for k in range(1, self.maxiter + 1):
            g = self._finite_difference_gradient(cost, x, fd_eps)
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * (g * g)
            mhat = m / (1.0 - beta1 ** k)
            vhat = v / (1.0 - beta2 ** k)
            step = lr * mhat / (np.sqrt(vhat) + eps)
            x = x - step
            f = cost(x)
            if f < best_f:
                best_x, best_f = x.copy(), f
            if tol and np.linalg.norm(step) < tol:
                break
        return best_x, best_f

    @staticmethod
    def _finite_difference_gradient(cost, x, eps):
        """Central-difference gradient of ``cost`` at ``x`` (``2N`` evaluations)."""
        g = np.zeros_like(x)
        for i in range(x.size):
            step = np.zeros_like(x)
            step[i] = eps
            g[i] = (cost(x + step) - cost(x - step)) / (2.0 * eps)
        return g
