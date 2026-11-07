# -*- coding: utf-8 -*-
# file: optimizers/optim.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Classical optimizers for the variational (hybrid) loop.

:class:`Optimizer` is a thin wrapper exposing the methods used to drive the
VQE / ADAPT-VQE parameter minimization behind one interface, recording the cost
history so a convergence trace is always available:

* **SPSA** (Simultaneous Perturbation Stochastic Approximation) -- a two-evaluation
  stochastic-gradient method;
* **COBYLA** (Constrained Optimization BY Linear Approximation);
* **Nelder-Mead** -- simplex search;
* **SLSQP** (Sequential Least Squares Programming) -- **the default**
  on an exact objective (SPSA is, with shots: :data:`SHOT_OPTIMIZER`);
* **BFGS**, **L-BFGS** -- quasi-Newton;
* **CG** -- nonlinear conjugate gradient, Polak-Ribiere.

All methods are executed natively in Rust via :class:`mandacaru._native.Context`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable, Sequence

import numpy as np
from .._native import Context


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
    #: differ by the method: L-BFGS spends several evaluations per step on a
    #: finite-difference gradient and a line search, SPSA two or three, while
    #: COBYLA evaluates once per trial point.  It is
    #: the honest measure of "how many moves did this optimizer make", which
    #: ``nfev`` is not.  ``None`` only when a method reports neither a count
    #: nor a per-iteration callback.
    nit: int | None = None


# The optimization methods exposed by name to the variational drivers
# (VQE, ADAPTVQE).
NAMED_OPTIMIZERS = ("SPSA", "COBYLA", "Nelder-Mead", "SLSQP",
                    "BFGS", "L-BFGS", "CG")

#: Default method on an exact objective (drivers included; with shots the
#: default is :data:`SHOT_OPTIMIZER`).  Measured on H2O/PAW-LCAO-SZ with
#: the qubit pool, where it reaches the same energy on the same circuit as
#: every other method at **one to two orders of magnitude fewer steps and
#: evaluations**, and was one of only two that certified convergence at every
#: growth step.  Its weakness is the small, nearly-converged case, where
#: stopping early on a flat landscape can cost ADAPT an extra operator.
#: See ``docs/source/guide/optimizers.md`` for the measurements.
DEFAULT_OPTIMIZER = "SLSQP"

#: The default on a **shot-based** objective, where every evaluation carries
#: independent sampling noise.  SLSQP (and BFGS, L-BFGS, CG) difference the
#: energy at steps near 1.5e-8; a noise of ``sigma`` then becomes a gradient
#: noise of about ``sqrt(2) sigma / eps`` -- ~10^6 Ha/rad at ``sigma = 0.01``
#: Ha -- and the method walks on noise.  SPSA's perturbation is ``c = 0.1``
#: rad, so the same noise costs ``sigma / c``, and ``track_best=False``
#: spends two evaluations a step instead of three and does not pick a
#: downward noise fluctuation as "the best".  ``tol=None``: a step size is not
#: a convergence certificate under noise, so the run takes its ``maxiter``
#: steps (``2 * maxiter + 1`` evaluations -- each one a job on a processor).
SHOT_OPTIMIZER = {"method": "SPSA", "maxiter": 100, "tol": None,
                  "options": {"track_best": False}}
#: Methods that estimate derivatives by differencing the energy at tiny
#: steps: refused on a shot-based objective (see :data:`SHOT_OPTIMIZER`).
NOISE_INTOLERANT = ("SLSQP", "BFGS", "L-BFGS", "CG")

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
#: reads that residual gradient**: at ``1e-8`` SLSQP reaches the H2/HAO ground
#: state to 1e-10 eV but leaves ``max|g| = 4.6e-06``, just above a
#: gradient convergence threshold of ``1e-6``, so the growth loop never stops
#: and piles up 50 redundant operators.  At ``1e-12`` the same run converges after one
#: operator in 4 steps and 9 evaluations.
#:
#: It also gives the native SPSA a criterion to certify convergence against,
#: which ``None`` does not.
DEFAULT_MAXITER = 1000
DEFAULT_TOL = 1e-12

# Historical compatibility aliases.
_CUSTOM_METHODS = ("SPSA",)
_SCIPY_ALIASES = {"L-BFGS": "L-BFGS-B"}

#: Methods for which SciPy reads ``tol`` as a **gradient norm** (``gtol``)
#: rather than a function-value change.
#:
#: :data:`DEFAULT_TOL` is chosen as a function-value criterion, and handing the
#: same number to a gradient test asks for something a finite-difference
#: gradient cannot deliver: its own accuracy is about ``1e-8``, so ``gtol =
#: 1e-12`` ends every line search in "precision loss" and the run reports
#: non-convergence at every growth step while sitting exactly on the minimum.
#: Near one, ``f - f* ~ |g|^2 / (2 lambda)``, so the gradient criterion of
#: equal strength is **the square root** of the function-value one, and that is
#: what is passed.  Measured on LiH/HAO with the qubit pool: at ``gtol =
#: sqrt(1e-12) = 1e-6`` BFGS and CG certify every step and reach the same
#: energy as SLSQP to 1e-6 eV in a third of the cost evaluations.  An explicit
#: ``options={"gtol": ...}`` is left alone.
_GRADIENT_NORM_METHODS = ("BFGS", "CG")

#: Methods that use a gradient, and so take one from the caller
#: (:meth:`Optimizer.minimize`'s ``gradient``) instead of differencing the
#: cost: one evaluation per parameter per gradient otherwise.
_GRADIENT_METHODS = ("SLSQP", "BFGS", "L-BFGS", "CG")


#: Keys a ``dict`` form of ``optimizer=`` may carry -- the :class:`Optimizer`
#: constructor's own arguments.  ``method`` / ``maxiter`` / ``tol`` are the
#: three that matter; ``options`` and ``seed`` are there so the dict form is
#: never less capable than the object it builds.
OPTIMIZER_KEYS = ("method", "maxiter", "tol", "options", "seed")


def _check_method(method, allowed) -> None:
    """Refuse a method a driver does not offer, however it was spelled."""
    if method not in allowed:
        raise ValueError(
            f"unknown optimizer {method!r}; use one of {tuple(allowed)}, a "
            f"dict of {OPTIMIZER_KEYS}, or an Optimizer instance")


def resolve_optimizer(optimizer, allowed=NAMED_OPTIMIZERS,
                      maxiter: int = 2000,
                      tol: float | None = DEFAULT_TOL,
                      shots: int = 0) -> "Optimizer":
    """Normalize an ``optimizer`` argument to an :class:`Optimizer`.

    Three spellings, all equivalent:

    * a pre-built :class:`Optimizer`, returned unchanged with its own
      ``maxiter`` / ``tol``;
    * a **method name** from ``allowed``, wrapped in a fresh :class:`Optimizer`
      with the ``maxiter`` / ``tol`` given here -- the shorthand for the
      defaults;
    * a **dict** of :data:`OPTIMIZER_KEYS`, e.g.
      ``{"method": "SLSQP", "maxiter": 2000, "tol": 1e-12}``, which builds the
      same object without the caller having to import :class:`Optimizer`.  Keys
      left out fall back to this function's arguments, so
      ``{"maxiter": 500}`` is the default method on a shorter budget.

    Shared by the VQE and ADAPT-VQE drivers, so every method exposes the same
    ``optimizer=`` surface.

    ``None`` is the default for the objective: :data:`DEFAULT_OPTIMIZER` on
    an exact one, :data:`SHOT_OPTIMIZER` when ``shots > 0`` (a dict without a
    ``"method"`` takes the same default).  With ``shots > 0`` a method in
    :data:`NOISE_INTOLERANT` is refused, however it was spelled.
    """
    if optimizer is None:
        optimizer = dict(SHOT_OPTIMIZER) if shots else DEFAULT_OPTIMIZER
    elif isinstance(optimizer, dict) and "method" not in optimizer and shots:
        optimizer = {**SHOT_OPTIMIZER, **optimizer}
    resolved = _resolve_optimizer(optimizer, allowed, maxiter, tol)
    if shots and resolved.method in NOISE_INTOLERANT:
        raise ValueError(
            f"optimizer {resolved.method!r} differences the energy at tiny "
            f"steps, which turns shot noise into gradient noise; with "
            f"shots > 0 use 'SPSA' (the default) or 'COBYLA'")
    return resolved


def _resolve_optimizer(optimizer, allowed, maxiter, tol) -> "Optimizer":
    if isinstance(optimizer, Optimizer):
        return optimizer
    if isinstance(optimizer, dict):
        unknown = sorted(set(optimizer) - set(OPTIMIZER_KEYS))
        if unknown:
            raise ValueError(
                f"unknown optimizer option(s) {unknown}; a dict optimizer "
                f"takes {OPTIMIZER_KEYS}")
        spec = {"method": DEFAULT_OPTIMIZER, "maxiter": maxiter, "tol": tol,
                **optimizer}
        # Same validation as the string form: the dict is a spelling of it, not
        # a way around its checks.
        _check_method(spec["method"], allowed)
        return Optimizer(**spec)
    if isinstance(optimizer, str):
        _check_method(optimizer, allowed)
        return Optimizer(method=optimizer, maxiter=maxiter, tol=tol)
    raise TypeError(
        f"optimizer must be a method name, a dict of {OPTIMIZER_KEYS} or an "
        f"Optimizer instance, got {type(optimizer).__name__}")


class Optimizer:
    """Classical optimizer with cost-history tracking.

    Parameters
    ----------
    method : str
        Optimization method (default :data:`DEFAULT_OPTIMIZER`, ``"SLSQP"``;
        a driver with shots resolves ``None`` to :data:`SHOT_OPTIMIZER`),
        one of :data:`NAMED_OPTIMIZERS`:

        * **derivative-free** -- ``"COBYLA"``, ``"Nelder-Mead"``;
        * **quasi-Newton / gradient** -- ``"SLSQP"``, ``"BFGS"``, ``"L-BFGS"``,
          ``"CG"`` (nonlinear conjugate gradient,
          Polak-Ribiere variant).  None is given an analytic gradient, so each
          builds its own by finite differences;
        * **stochastic** -- ``"SPSA"``.

        Every method goes through native Rust algorithms in :class:`mandacaru._native.Context`.
    maxiter : int
        Maximum iterations (default :data:`DEFAULT_MAXITER`).
    tol : float, optional
        Convergence tolerance (default :data:`DEFAULT_TOL`, ``1e-12``).
    options : dict, optional
        Extra options (e.g. ``track_best``, ``patience``, ``eps``).
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

    def max_evaluations(self) -> int | None:
        """The most cost evaluations :meth:`minimize` can make, or ``None``
        when the method does not bound them by ``maxiter``.

        On a processor each evaluation is a job, so this is what a run
        budget is checked against before the first one is queued.
        """
        if self.method == "SPSA":
            track_best = bool(self.options.get("track_best", True))
            return 2 * self.maxiter + (self.maxiter + 1 if track_best else 1)
        if self.method == "COBYLA":
            return int(self.options.get("maxiter", self.maxiter))
        return None

    def minimize(self, cost: Callable[[np.ndarray], float],
                 x0: Sequence[float], callback=None,
                 gradient=None) -> OptimizeResult:
        """Minimize ``cost`` starting from ``x0``.

        ``callback(x, value, nfev)``, if given, is invoked after **every** cost
        evaluation with the point, its cost and the running evaluation count --
        the hook a driver uses to checkpoint its best-so-far parameters
        mid-optimization, independent of which method is running.

        ``gradient(x) -> (value, grad)``, if given, is the cost together with
        its exact gradient.  The gradient methods (:data:`_GRADIENT_METHODS`)
        then use it instead of a finite difference, and each call counts as
        one evaluation; the gradient-free ones ignore it.
        """
        x0 = np.asarray(x0, dtype=float).ravel()
        history: list[float] = []

        def wrapped(x):
            return wrapped_value(x, cost(x))

        def wrapped_value(x, value):
            value = float(value)
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

        tol = self.tol
        if (tol is not None and self.method in _GRADIENT_NORM_METHODS
                and "gtol" not in self.options):
            # A gradient-norm test of the same strength as a function-value
            # one of `tol`; see _GRADIENT_NORM_METHODS.
            tol = float(np.sqrt(tol))

        native_grad = None
        if gradient is not None and self.method in _GRADIENT_METHODS:
            def native_grad(x):
                value, grad = gradient(x)
                grad = np.asarray(grad, dtype=float).ravel()
                if not np.all(np.isfinite(grad)):
                    raise ValueError(
                        f"the gradient is not finite at {np.asarray(x)}; "
                        "the optimization cannot continue")
                return wrapped_value(x, value), grad

        track_best = bool(self.options.get("track_best", True))
        patience = max(1, int(self.options.get("patience", 5)))
        eps = float(self.options.get("eps", 1.4901161193847656e-8))

        ctx = Context(threads=1)
        res = ctx.minimize(
            self.method,
            x0,
            cost=wrapped,
            gradient=native_grad,
            max_iterations=self.maxiter,
            max_evaluations=self.max_evaluations() or 100000,
            tolerance=tol,
            seed=self.seed,
            track_best=track_best,
            patience=patience,
            eps=eps,
        )

        if self.method == "SPSA":
            message = (f"{self.method} met tol={self.tol:g}" if res["success"] else
                       f"{self.method} ran {self.maxiter} iterations without "
                       f"meeting a tolerance"
                       + ("" if self.tol else " (no tol set)"))
        else:
            message = str(res["message"])

        return OptimizeResult(
            x=np.asarray(res["x"], dtype=float),
            fun=float(res["fun"]),
            nfev=len(history),
            history=history,
            success=bool(res["success"]),
            message=message,
            nit=int(res["steps"]),
        )

