# -*- coding: utf-8 -*-
# file: optimizers/optim.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Classical optimizers for the variational (hybrid) loop.

:class:`Optimizer` is a thin, SciPy-backed wrapper exposing the gradient-free
methods commonly used in VQE -- **COBYLA** (default), **Nelder-Mead**, **SLSQP**,
**Powell**, **L-BFGS-B** -- behind one interface, recording the cost history so a
convergence trace is always available.
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


# The gradient-free / quasi-Newton methods exposed by name to the variational
# drivers (VQE, ADAPTVQE).  COBYLA is the default everywhere.
NAMED_OPTIMIZERS = ("COBYLA", "Nelder-Mead", "BFGS")


def resolve_optimizer(optimizer, allowed=NAMED_OPTIMIZERS,
                      maxiter: int = 2000) -> "Optimizer":
    """Normalize an ``optimizer`` argument to an :class:`Optimizer`.

    Accepts either a pre-built :class:`Optimizer` (returned unchanged) or a
    method name from ``allowed`` (``"COBYLA"``, ``"Nelder-Mead"``, ``"BFGS"``),
    which is wrapped in a fresh :class:`Optimizer`.  Shared by the VQE and
    ADAPT-VQE drivers so both expose the same ``optimizer=`` surface.
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
    """SciPy-backed classical optimizer with cost-history tracking.

    Parameters
    ----------
    method : str
        SciPy ``minimize`` method (default ``"COBYLA"``).
    maxiter : int
        Maximum iterations (default ``1000``).
    tol : float, optional
        Convergence tolerance passed to SciPy.
    options : dict, optional
        Extra options forwarded to ``scipy.optimize.minimize``.
    """

    def __init__(self, method: str = "COBYLA", maxiter: int = 1000,
                 tol: float | None = None, options: dict | None = None):
        self.method = method
        self.maxiter = int(maxiter)
        self.tol = tol
        self.options = dict(options or {})

    def minimize(self, cost: Callable[[np.ndarray], float],
                 x0: Sequence[float]) -> OptimizeResult:
        """Minimize ``cost`` starting from ``x0``."""
        x0 = np.asarray(x0, dtype=float).ravel()
        history: list[float] = []

        def wrapped(x):
            value = float(cost(x))
            history.append(value)
            return value

        options = {"maxiter": self.maxiter, **self.options}
        res = minimize(wrapped, x0, method=self.method, tol=self.tol,
                       options=options)
        return OptimizeResult(
            x=np.asarray(res.x, dtype=float), fun=float(res.fun),
            nfev=len(history), history=history,
            success=bool(res.success), message=str(res.message))
