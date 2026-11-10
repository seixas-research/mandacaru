# -*- coding: utf-8 -*-
# file: optimizers/optim.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Classical optimizers for the variational (hybrid) loop.

:class:`Optimizer` names one of the methods that drive the VQE / ADAPT-VQE
parameter minimization and carries its budget and tolerance; a calculator takes
it in place of the ``optimizer=`` dict.  The minimization itself runs in Rust:

* **SPSA** (Simultaneous Perturbation Stochastic Approximation) -- a two-evaluation
  stochastic-gradient method;
* **COBYLA** (Constrained Optimization BY Linear Approximation);
* **Nelder-Mead** -- simplex search;
* **SLSQP** (Sequential Least Squares Programming) -- **the default**
  on an exact objective (SPSA is, with shots);
* **BFGS**, **L-BFGS** -- quasi-Newton;
* **CG** -- nonlinear conjugate gradient, Polak-Ribiere.

Every method is executed natively in Rust (:class:`mandacaru._native.Context`).
"""

from __future__ import annotations

# The optimization methods exposed by name to the variational drivers.
NAMED_OPTIMIZERS = ("SPSA", "COBYLA", "Nelder-Mead", "SLSQP",
                    "BFGS", "L-BFGS", "CG")

#: Default method on an exact objective (with shots the default is SPSA).
#: Measured on H2O/PAW-LCAO-SZ with
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


class Optimizer:
    """A named classical optimizer with its budget and tolerance.

    Parameters
    ----------
    method : str
        Optimization method (default :data:`DEFAULT_OPTIMIZER`, ``"SLSQP"``;
        a calculator with shots resolves an unset method to ``"SPSA"``),
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
