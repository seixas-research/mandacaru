# -*- coding: utf-8 -*-
# file: optimizers/optim.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Classical optimizers for the variational (hybrid) loop.

:class:`Optimizer` is a thin wrapper exposing the six methods used to drive the
VQE / ADAPT-VQE parameter minimization behind one interface, recording the cost
history so a convergence trace is always available:

* **SPSA** (Simultaneous Perturbation Stochastic Approximation) -- a two-evaluation
  stochastic-gradient method, implemented natively here;
* **COBYLA** (Constrained Optimization BY Linear Approximation) -- SciPy;
* **Nelder-Mead** -- SciPy simplex;
* **SLSQP** (Sequential Least Squares Programming) -- SciPy, **the default**;
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
    #: Optimizer **steps** -- parameter updates, not cost evaluations.  The two
    #: differ by the method: L-BFGS-B spends several evaluations per step on a
    #: finite-difference gradient and a line search, Adam spends ``2N + 1``,
    #: SPSA two or three, while COBYLA evaluates once per trial point.  It is
    #: the honest measure of "how many moves did this optimizer make", which
    #: ``nfev`` is not.  ``None`` only when a method reports neither a count
    #: nor a per-iteration callback.
    nit: int | None = None


# The optimization methods exposed by name to the variational drivers
# (VQE, ADAPTVQE).
NAMED_OPTIMIZERS = ("SPSA", "COBYLA", "Nelder-Mead", "SLSQP", "Adam", "L-BFGS-B")

#: Default method everywhere (drivers included).  Measured on H2O/PAW-SZ with
#: the qubit pool, where it reaches the same energy on the same circuit as
#: every other method at **one to two orders of magnitude fewer steps and
#: evaluations**, and was one of only two that certified convergence at every
#: growth step.  Its weakness is the small, nearly-converged case, where
#: stopping early on a flat landscape can cost ADAPT an extra operator.
#: See ``docs/source/guide/optimizers.md`` for the measurements.
DEFAULT_OPTIMIZER = "SLSQP"

#: Default iteration budget and convergence tolerance of a bare
#: :class:`Optimizer`.
#:
#: The tolerance is **explicit and very tight on purpose**, and it is chosen
#: together with the method above.
#:
#: SciPy's own defaults are far too loose for a cost measured in Hartree
#: (Nelder-Mead's ``xatol = fatol = 1e-4`` stops the simplex four orders of
#: magnitude above chemical accuracy), and ``tol`` means different things per
#: method: for SLSQP and COBYLA it is a **function-value** criterion, so it has
#: to be tighter than a gradient method's ``tol`` to leave an equally small
#: gradient behind.  That matters because **ADAPT-VQE's own convergence test
#: reads that residual gradient**: at ``1e-8`` SLSQP reaches the H2/FAO ground
#: state to 1e-10 eV but leaves ``max|g| = 4.6e-06``, just above a
#: ``gradient_tolerance`` of ``1e-6``, so the growth loop never stops and piles
#: up 50 redundant operators.  At ``1e-12`` the same run converges after one
#: operator in 4 steps and 9 evaluations.
#:
#: It also gives the native SPSA / Adam a criterion to certify convergence
#: against, which ``None`` does not.
DEFAULT_MAXITER = 1000
DEFAULT_TOL = 1e-12

# Methods routed to scipy.optimize.minimize vs. implemented natively below.
_SCIPY_METHODS = ("COBYLA", "Nelder-Mead", "SLSQP", "L-BFGS-B")
_CUSTOM_METHODS = ("SPSA", "Adam")


def resolve_optimizer(optimizer, allowed=NAMED_OPTIMIZERS,
                      maxiter: int = 2000,
                      tol: float | None = DEFAULT_TOL) -> "Optimizer":
    """Normalize an ``optimizer`` argument to an :class:`Optimizer`.

    Accepts either a pre-built :class:`Optimizer` (returned unchanged, with its
    own ``maxiter`` / ``tol``) or a method name from ``allowed``, which is
    wrapped in a fresh :class:`Optimizer` with the ``maxiter`` / ``tol`` given
    here.  Shared by the VQE and ADAPT-VQE drivers so both expose the same
    ``optimizer=`` surface.

    Pass an :class:`Optimizer` to control the budget and the tolerance; the
    name is the shorthand for the defaults.
    """
    if isinstance(optimizer, Optimizer):
        return optimizer
    if isinstance(optimizer, str):
        if optimizer not in allowed:
            raise ValueError(
                f"unknown optimizer {optimizer!r}; use one of {tuple(allowed)} "
                "or an Optimizer instance")
        return Optimizer(method=optimizer, maxiter=maxiter, tol=tol)
    raise TypeError("optimizer must be a method name or an Optimizer instance")


class Optimizer:
    """Classical optimizer with cost-history tracking.

    Parameters
    ----------
    method : str
        Optimization method (default :data:`DEFAULT_OPTIMIZER`, ``"SLSQP"``).
        One of ``"SPSA"``, ``"COBYLA"``, ``"Nelder-Mead"``, ``"SLSQP"``,
        ``"Adam"``, ``"L-BFGS-B"``.  COBYLA, Nelder-Mead, SLSQP and L-BFGS-B go
        through ``scipy.optimize.minimize``; SPSA and Adam are implemented
        natively.
    maxiter : int
        Maximum iterations (default :data:`DEFAULT_MAXITER`).  For the SciPy
        methods this is the ``maxiter`` option; for SPSA and Adam it is the
        number of update steps.
    tol : float, optional
        Convergence tolerance (default :data:`DEFAULT_TOL`, ``1e-8``).  Passed
        to SciPy for the SciPy methods; used as the step/cost-change stopping
        threshold for SPSA and Adam, which cannot certify convergence without
        one.  ``None`` restores each method's own default -- Nelder-Mead's is
        ``1e-4``, far too loose for a cost in Hartree.
    options : dict, optional
        Extra options.  Forwarded to ``scipy.optimize.minimize`` for the SciPy
        methods; the SPSA / Adam hyperparameters (see :meth:`_minimize_spsa` /
        :meth:`_minimize_adam`) are read from here for the native methods.
    seed : int, optional
        Seed for the SPSA perturbation RNG (default ``0``); makes runs
        reproducible.  Ignored by the deterministic methods.
    """

    def __init__(self, method: str = DEFAULT_OPTIMIZER,
                 maxiter: int = DEFAULT_MAXITER,
                 tol: float | None = DEFAULT_TOL,
                 options: dict | None = None, seed: int = 0):
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
                 x0: Sequence[float], callback=None) -> OptimizeResult:
        """Minimize ``cost`` starting from ``x0``.

        ``callback(x, value, nfev)``, if given, is invoked after **every** cost
        evaluation with the point, its cost and the running evaluation count --
        the hook a driver uses to checkpoint its best-so-far parameters
        mid-optimization, independent of which method is running.
        """
        x0 = np.asarray(x0, dtype=float).ravel()
        history: list[float] = []

        def wrapped(x):
            value = float(cost(x))
            if not np.isfinite(value):
                # A NaN/inf cost makes every downstream decision meaningless:
                # the optimizer would wander and report an arbitrary point.
                raise ValueError(
                    f"the cost function returned {value} at {np.asarray(x)}; "
                    "the optimization cannot continue")
            history.append(value)
            if callback is not None:
                callback(np.array(x, dtype=float).ravel(), value, len(history))
            return value

        # A zero-parameter ansatz has nothing to optimize -- evaluate once.
        if x0.size == 0:
            value = wrapped(x0)
            return OptimizeResult(x=x0, fun=value, nfev=1, history=history,
                                  success=True, message="no free parameters",
                                  nit=0)

        if self.method in _CUSTOM_METHODS:
            if self.method == "SPSA":
                x, fun, converged, steps = self._minimize_spsa(wrapped, x0)
            else:                                    # "Adam"
                x, fun, converged, steps = self._minimize_adam(wrapped, x0)
            # `success` means a convergence test was met, not that the loop ran:
            # the native methods stop on their own step criterion, which needs
            # `tol`.  Without one there is nothing to certify.
            message = (f"{self.method} met tol={self.tol:g}" if converged else
                       f"{self.method} ran {self.maxiter} iterations without "
                       f"meeting a tolerance"
                       + ("" if self.tol else " (no tol set)"))
            return OptimizeResult(
                x=np.asarray(x, dtype=float), fun=float(fun),
                nfev=len(history), history=history, success=bool(converged),
                message=message, nit=int(steps))

        # SciPy calls `callback` once per *iteration*, which is the count we
        # want; `res.nit` is preferred where the method reports it (it is the
        # method's own bookkeeping), and the tally is the fallback for the ones
        # that do not -- COBYLA has no `nit` at all.
        steps = 0

        def count_step(*_args, **_kwargs):
            nonlocal steps
            steps += 1

        options = {"maxiter": self.maxiter, **self.options}
        res = minimize(wrapped, x0, method=self.method, tol=self.tol,
                       options=options, callback=count_step)
        reported = getattr(res, "nit", None)
        return OptimizeResult(
            x=np.asarray(res.x, dtype=float), fun=float(res.fun),
            nfev=len(history), history=history,
            success=bool(res.success), message=str(res.message),
            nit=int(reported) if reported is not None else steps)

    # -- native optimizers ------------------------------------------------- #

    def _minimize_spsa(self, cost, x0):
        """SPSA -- Spall's simultaneous perturbation stochastic approximation.

        Each step estimates the gradient from **two** cost evaluations at
        ``x +/- c_k * delta`` with a random ``+/-1`` perturbation ``delta``, then
        takes a decaying step ``a_k``.  By default a **third** evaluation at the
        new point keeps the best iterate seen, which is what a noiseless
        state-vector cost wants; ``track_best=False`` is the strict
        two-evaluation form (the last iterate is returned, evaluated once at
        the end) -- the one to use when every evaluation is a hardware job.

        Gain sequences follow Spall's practical recommendations, tunable through
        ``options``: ``a`` (0.2), ``c`` (0.1), ``alpha`` (0.602), ``gamma``
        (0.101) and the stability constant ``A`` (``0.1 * maxiter``).  The
        gradient estimate is a single random sample, so one small step proves
        nothing: convergence is claimed only after ``patience`` (5)
        **consecutive** steps below ``tol``.

        Returns ``(x, f, converged, steps)``, where ``steps`` is the number of
        parameter updates taken -- two or three cost evaluations each.
        """
        o = self.options
        a = float(o.get("a", 0.2))
        c = float(o.get("c", 0.1))
        alpha = float(o.get("alpha", 0.602))
        gamma = float(o.get("gamma", 0.101))
        A = float(o.get("A", 0.1 * self.maxiter))
        patience = max(1, int(o.get("patience", 5)))
        track_best = bool(o.get("track_best", True))
        tol = self.tol if self.tol is not None else 0.0

        rng = np.random.default_rng(self.seed)
        x = np.array(x0, dtype=float)
        best_x, best_f = x.copy(), (cost(x) if track_best else None)
        calm, converged, steps = 0, False, 0
        for k in range(self.maxiter):
            steps += 1
            ak = a / (k + 1 + A) ** alpha
            ck = c / (k + 1) ** gamma
            delta = rng.choice([-1.0, 1.0], size=x.shape)
            fp = cost(x + ck * delta)
            fm = cost(x - ck * delta)
            ghat = (fp - fm) / (2.0 * ck) * (1.0 / delta)
            x = x - ak * ghat
            if track_best:
                f = cost(x)
                if f < best_f:
                    best_x, best_f = x.copy(), f
            calm = calm + 1 if tol and ak * np.linalg.norm(ghat) < tol else 0
            if calm >= patience:
                converged = True
                break
        if not track_best:
            best_x, best_f = x, cost(x)
        return best_x, best_f, converged, steps

    def _minimize_adam(self, cost, x0):
        """Adam -- adaptive moment estimation on a finite-difference gradient.

        The cost is a black box, so the gradient is estimated by central finite
        differences (``2N`` evaluations per step).  Hyperparameters are tunable
        through ``options``: ``lr`` (0.05), ``beta1`` (0.9), ``beta2`` (0.999),
        ``eps`` (1e-8) and the finite-difference step ``fd_eps`` (1e-4).

        Returns ``(x, f, converged, steps)``, where ``steps`` is the number of
        parameter updates taken -- ``2N + 1`` cost evaluations each.
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
                return best_x, best_f, True, k
        return best_x, best_f, False, self.maxiter

    @staticmethod
    def _finite_difference_gradient(cost, x, eps):
        """Central-difference gradient of ``cost`` at ``x`` (``2N`` evaluations)."""
        g = np.zeros_like(x)
        for i in range(x.size):
            step = np.zeros_like(x)
            step[i] = eps
            g[i] = (cost(x + step) - cost(x - step)) / (2.0 * eps)
        return g
