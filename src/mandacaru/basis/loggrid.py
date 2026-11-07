# -*- coding: utf-8 -*-
# file: basis/loggrid.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Radial eigenstates on a logarithmic grid.

The uniform radial grid of :mod:`mandacaru.basis.atomic_solver` is a
deliberate choice and a good one for everything non-relativistic: it keeps the
eigenproblem a plain symmetric tridiagonal matrix, and a Schrodinger radial
function :math:`u = rR \sim r^{l+1}` is analytic at the origin, so its
three-point Laplacian converges cleanly at second order.

**A relativistic state is not analytic there.**  With a point nucleus
:math:`P \sim r^{\gamma}`, :math:`\gamma = \sqrt{\kappa^2-(Z\alpha)^2}`, which
is *below one* for every s state and every :math:`p_{1/2}`.  A uniform grid
cannot converge that: measured against the closed-form Dirac spectrum, the
error of the 1s level scales as :math:`(Z\alpha)^2` -- **the size of the
relativistic correction itself** -- and the convergence rate never approaches
two.  On a hydrogenic oxygen the discretization error at 64000 points exceeds
the shift it is supposed to be computing.  Refining the grid does not fix it;
the same potential solved non-relativistically converges at a clean rate four,
so it is the cusp and not the :math:`-Z/r` singularity.

A logarithmic grid puts its points where the cusp is.  With
:math:`x = \ln(r/r_0)` uniform, :math:`r^{\gamma}` becomes
:math:`e^{\gamma x}` -- a smooth exponential, which a uniform :math:`x` grid
represents to fourth order.

The transformation
------------------

:math:`d/dr = r^{-1}d/dx` turns the radial equation into

.. math::

    u_{xx} - u_x = \big[l(l+1) + 2r^2(V-\varepsilon)\big]u ,

and the substitution :math:`u = e^{x/2}y` cancels the first derivative
identically, leaving :math:`y_{xx} = g\,y` with

.. math::

    g = l(l+1) + \tfrac14 + 2r^2(V-\varepsilon) .

Relativistically the same substitution works with
:math:`P = e^{x/2}M^{1/2}y`, which is the log-grid counterpart of the
:math:`P = M^{1/2}W` cancellation on the uniform grid.  Writing
:math:`w = M_x/M`,

.. math::

    g = l(l+1) + \kappa w + \frac{(1+w)^2}{4} - \frac{w_x}{2}
        - 2Mr^2(\varepsilon-V) ,

and :math:`M` depends on :math:`\varepsilon`, so :math:`g` is rebuilt at every
trial energy.  (The equivalent form with :math:`w^2/2 - M_{xx}/2M` in place of
:math:`-w_x/2` is the same expression; both are verified in the tests.)

Why this is solved by shooting and not by a matrix
--------------------------------------------------

Written on the grid, :math:`y_{xx} = gy` looks like a **generalized**
symmetric-definite tridiagonal problem :math:`Ay = \varepsilon By` with
:math:`B = 2r^2` diagonal and positive, which invites the obvious reduction
:math:`\tilde H = B^{-1/2}AB^{-1/2}` back to the ordinary tridiagonal solver
the uniform grid uses.  **That reduction is numerically fatal here and was
tried first.**  The whole point of a log grid is that :math:`r` spans many
decades, so :math:`B` does too: over the default grid :math:`2r^2` runs from
about 1e-16 to 1e3, and :math:`\tilde H` acquires a leading diagonal entry
near 1e19.  The physical eigenvalues, a few Hartree, are then far below the
roundoff of the matrix that is supposed to carry them, and the solver returns
degenerate garbage.  The conditioning is intrinsic to the transform, not to
any one grid, so refining or re-scaling does not rescue it.

Shooting never forms that matrix.  It integrates :math:`y_{xx}=gy` outward
from the origin and inward from the tail with Numerov's fourth-order
recurrence, and asks for the one energy at which the two halves join
smoothly -- so the dynamic range of :math:`r` costs nothing, and the fourth
order in :math:`x` is exactly what the log grid was chosen to buy.

A second, quieter gain: the combination :math:`\kappa w - w_x/2` is
:math:`O(1)` at the origin on this grid.  Its uniform-grid counterpart,
:math:`\kappa M'/(Mr) - M''/2M`, is a difference of two terms that each
diverge as :math:`1/r^3` and cancel, which costs seven digits at small
:math:`r` and gets *worse* as the uniform grid is refined.  The log
transformation removes that cancellation analytically rather than working
around it.

Scope
-----

This module solves; it does not replace the uniform grid.  The reference atom,
the pseudopotential generators and everything downstream stay on the uniform
grid they were written for.

It is **not wired into** :func:`~mandacaru.basis.atomic_solver.solve_atom` or
:func:`~mandacaru.basis.relativity.solve_radial_relativistic` any more: the native
reference atom is uniform-grid only, and this module (Python numerics, awaiting a port
or removal) is reachable only by calling :func:`solve_radial_log` directly.  Making it their default would change every relativistic eigenvalue they
produce, and so every shipped dataset -- a deliberate step to take on its own,
not a side effect of adding a solver.  What it would buy is an accurate **eigenvalue**; the wave on
the uniform grid still could not represent the cusp, but nothing downstream
needs it to -- the pseudization works at :math:`r_c`, where the wave is
smooth, and the cusp region carries almost no charge.

Measured against the closed-form Dirac spectrum (:func:`test_loggrid.py
<test.basis.test_loggrid>` pins these):

===============  ==================  =====================
hydrogenic 1s    uniform, 64000 pts  logarithmic, 1200 pts
===============  ==================  =====================
:math:`Z=1`      4.6e-5              2.4e-8
:math:`Z=8`      1.3e-3              4.0e-8
:math:`Z=20`     3.2e-3              1.6e-6
convergence      never reaches 2     4.00
===============  ==================  =====================
"""

from __future__ import annotations

import numpy as np

#: Innermost radius of the default grid, as a fraction of ``1/Z`` -- the scale
#: of the 1s orbital.  The boundary condition is ``y = 0`` there, and
#: :math:`P \sim r^{\gamma}` makes that an error of order
#: ``(r_min Z)^gamma``, i.e. 1e-8 at this value.
DEFAULT_R_MIN_FRACTION = 1e-8

#: Points of the default grid.  A log grid needs far fewer than a uniform one:
#: 1200 here resolves what 64000 uniform points cannot.
DEFAULT_POINTS = 1200

#: Bisections available to the energy search.  Each one halves the window,
#: so this reaches the floating-point floor of any physical bracket with room
#: to spare; the loop exits on ``tolerance`` long before exhausting it.
MAX_BISECTIONS = 200

#: Rescale the running solution when it exceeds this, so that the exponential
#: growth through a thick forbidden region cannot overflow.
_RESCALE_AT = 1e120

#: A point counts toward the node total only if the solution there exceeds
#: this fraction of its own largest value.  Below it the wave has underflowed
#: and its sign is roundoff.
_NODE_FLOOR = 1e-10


def log_grid(r_max: float, atomic_number: float = 1.0,
             points: int = DEFAULT_POINTS, r_min: float | None = None):
    r"""``(r, x, dx)`` of a grid uniform in :math:`\ln r`.

    ``r_min`` defaults to :data:`DEFAULT_R_MIN_FRACTION` over the nuclear
    charge, so the innermost point scales with the 1s shell rather than with
    the box.
    """
    from .._native import loggrid_make_grid

    r_min_f = None if r_min is None else float(r_min)
    return loggrid_make_grid(float(r_max), float(atomic_number), int(points), r_min_f)


def _fourth_order_derivatives(f, dx):
    r"""``(f_x, f_xx)`` to fourth order on a uniform grid.

    :func:`numpy.gradient` is second order, and on this grid that -- not
    Numerov -- is what limits the relativistic branch: measured against the
    closed-form Dirac 1s of oxygen, second-order derivatives converge at rate
    2.00 and fourth-order ones at 4.01, a factor of 500 in accuracy at 1200
    points.  The interior stencils are the standard five-point ones; the two
    points at each edge use the matching one-sided forms.
    """
    from .. import _native

    return _native.loggrid_fourth_order_derivatives(
        np.ascontiguousarray(f, dtype=float), float(dx)
    )


def radial_g(r, potential, l: int, energy: float, *, kappa: int | None = None,
             relativistic: bool = False, c: float | None = None,
             dx: float | None = None, derivatives=None):
    r"""The :math:`g` of :math:`y_{xx} = g\,y`, and the mass factor with it.

    Returns ``(g, M)``; ``M`` is ``None`` when ``relativistic`` is false.  The
    relativistic branch differentiates :math:`V` rather than :math:`M`, which
    is the same thing (:math:`M_x = -V_x/2c^2` at fixed :math:`\varepsilon`)
    and keeps the mass floor out of the derivative.

    ``derivatives`` optionally supplies :math:`(V_x, V_{xx})` analytically.
    A caller that knows them exactly should pass them: the finite-difference
    default is what caps the accuracy here, not the integrator.
    """
    from .. import _native

    derivs = (
        (np.ascontiguousarray(derivatives[0], dtype=float),
         np.ascontiguousarray(derivatives[1], dtype=float))
        if derivatives is not None else None
    )
    return _native.loggrid_radial_g(
        np.ascontiguousarray(r, dtype=float),
        np.ascontiguousarray(potential, dtype=float),
        int(l),
        float(energy),
        None if kappa is None else int(kappa),
        bool(relativistic),
        None if c is None else float(c),
        None if dx is None else float(dx),
        derivs,
    )


def solve_radial_log(r, potential, l: int, n_nodes: int, *,
                     kappa: int | None = None, treatment: str = "none",
                     atomic_number: float = 0.0, c: float | None = None,
                     energy_guess: float | None = None,
                     max_iterations: int = MAX_BISECTIONS,
                     tolerance: float = 1e-13, derivatives=None):
    r"""Bound state on a logarithmic grid: ``(P, eps)``.

    ``r`` must be uniform in :math:`\ln r` (:func:`log_grid`).  ``P = rR`` is
    returned on that same grid, normalized :math:`\int P^2\,dr = 1` with the
    logarithmic measure :math:`dr = r\,dx`.

    The state is selected by its node count, not by a matrix index, so
    ``n_nodes`` picks the state unambiguously even where levels crowd.
    ``energy_guess`` only seeds the bracket; it cannot select a different
    state, and a poor guess costs iterations rather than correctness.
    """
    from .. import _native

    derivs = (
        (np.ascontiguousarray(derivatives[0], dtype=float),
         np.ascontiguousarray(derivatives[1], dtype=float))
        if derivatives is not None else None
    )
    return _native.solve_radial_log(
        np.ascontiguousarray(r, dtype=float),
        np.ascontiguousarray(potential, dtype=float),
        int(l),
        int(n_nodes),
        None if kappa is None else int(kappa),
        str(treatment),
        float(atomic_number),
        None if c is None else float(c),
        None if energy_guess is None else float(energy_guess),
        int(max_iterations),
        float(tolerance),
        derivs,
    )
