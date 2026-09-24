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

It is reached only on request, through ``grid="log"`` of
:func:`~mandacaru.basis.relativity.solve_radial_relativistic` and
:func:`~mandacaru.basis.atomic_solver.solve_atom`; the generators never pass
it.  Making it their default would change every relativistic eigenvalue they
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
    Z = max(float(atomic_number), 1.0)
    if r_min is None:
        r_min = DEFAULT_R_MIN_FRACTION / Z
    x = np.linspace(np.log(float(r_min)), np.log(float(r_max)), int(points))
    return np.exp(x), x, float(x[1] - x[0])


def _fourth_order_derivatives(f, dx):
    """``(f_x, f_xx)`` to fourth order on a uniform grid.

    :func:`numpy.gradient` is second order, and on this grid that -- not
    Numerov -- is what limits the relativistic branch: measured against the
    closed-form Dirac 1s of oxygen, second-order derivatives converge at rate
    2.00 and fourth-order ones at 4.01, a factor of 500 in accuracy at 1200
    points.  The interior stencils are the standard five-point ones; the two
    points at each edge use the matching one-sided forms.
    """
    f = np.asarray(f, dtype=float)
    n = f.size
    fx = np.empty(n)
    fxx = np.empty(n)
    if n < 6:                      # too short for the stencils; stay 2nd order
        fx = np.gradient(f, dx, edge_order=2)
        return fx, np.gradient(fx, dx, edge_order=2)

    i = slice(2, n - 2)
    fx[i] = (-f[4:] + 8.0 * f[3:-1] - 8.0 * f[1:-3] + f[:-4]) / (12.0 * dx)
    fxx[i] = (-f[4:] + 16.0 * f[3:-1] - 30.0 * f[2:-2]
              + 16.0 * f[1:-3] - f[:-4]) / (12.0 * dx * dx)

    fx[0] = (-25*f[0] + 48*f[1] - 36*f[2] + 16*f[3] - 3*f[4]) / (12*dx)
    fx[1] = (-3*f[0] - 10*f[1] + 18*f[2] - 6*f[3] + f[4]) / (12*dx)
    fx[-1] = (25*f[-1] - 48*f[-2] + 36*f[-3] - 16*f[-4] + 3*f[-5]) / (12*dx)
    fx[-2] = (3*f[-1] + 10*f[-2] - 18*f[-3] + 6*f[-4] - f[-5]) / (12*dx)

    h2 = 12.0 * dx * dx
    fxx[0] = (45*f[0] - 154*f[1] + 214*f[2] - 156*f[3] + 61*f[4] - 10*f[5]) / h2
    fxx[1] = (10*f[0] - 15*f[1] - 4*f[2] + 14*f[3] - 6*f[4] + f[5]) / h2
    fxx[-1] = (45*f[-1] - 154*f[-2] + 214*f[-3] - 156*f[-4]
               + 61*f[-5] - 10*f[-6]) / h2
    fxx[-2] = (10*f[-1] - 15*f[-2] - 4*f[-3] + 14*f[-4]
               - 6*f[-5] + f[-6]) / h2
    return fx, fxx


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
    from .relativity import SPEED_OF_LIGHT, MASS_FLOOR

    r = np.asarray(r, dtype=float)
    potential = np.asarray(potential, dtype=float)
    if dx is None:
        dx = float(np.log(r[1]) - np.log(r[0]))
    c = SPEED_OF_LIGHT if c is None else float(c)
    if not relativistic:
        return (l * (l + 1) + 0.25
                + 2.0 * r * r * (potential - energy)), None

    k = -1 if kappa is None else int(kappa)
    M = np.maximum(1.0 + (energy - potential) / (2.0 * c * c), MASS_FLOOR)
    if derivatives is None:
        Vx, Vxx = _fourth_order_derivatives(potential, dx)
    else:
        Vx, Vxx = (np.asarray(d, dtype=float) for d in derivatives)
    Mx = -Vx / (2.0 * c * c)
    Mxx = -Vxx / (2.0 * c * c)
    w = Mx / M
    wx = Mxx / M - w * w
    # NB: l(l+1) here, *not* ``centrifugal``.  The 1/4 that the
    # non-relativistic branch carries in ``centrifugal`` is produced here by
    # (1+w)^2/4, which is 1/4 when w -> 0; adding both double-counts it and
    # shifts g by a constant 1/4 at every point.  That is invisible to a
    # convergence study -- the error does not shrink with the grid -- and it
    # was how this bug was found.
    g = (l * (l + 1) + k * w + (1.0 + w) ** 2 / 4.0 - wx / 2.0
         - 2.0 * M * r * r * (energy - potential))
    return g, M


def _numerov_outward(f, dx, stop, slope):
    """``y`` from the origin to ``stop``, seeded with ``y ~ exp(slope x)``."""
    y = np.zeros(len(f))
    y[0] = 1.0
    y[1] = float(np.exp(slope * dx))
    for n in range(1, stop + 1):
        y[n + 1] = ((12.0 - 10.0 * f[n]) * y[n] - f[n - 1] * y[n - 1]) / f[n + 1]
        if abs(y[n + 1]) > _RESCALE_AT:
            y[:n + 2] /= _RESCALE_AT
    return y


def _numerov_inward(f, g, dx, stop):
    """``y`` from the outer edge down to ``stop``, seeded by the decaying tail."""
    y = np.zeros(len(f))
    n_last = len(f) - 1
    y[n_last] = 1.0
    y[n_last - 1] = float(np.exp(np.sqrt(max(g[n_last], 0.0)) * dx))
    for n in range(n_last - 1, stop - 1, -1):
        y[n - 1] = ((12.0 - 10.0 * f[n]) * y[n] - f[n + 1] * y[n + 1]) / f[n - 1]
        if abs(y[n - 1]) > _RESCALE_AT:
            y[n - 1:] /= _RESCALE_AT
    return y


def _match(r, potential, l, energy, *, kappa, relativistic, c, dx,
           derivatives=None):
    """``(residual, nodes, y, M)`` of the two-sided solution at ``energy``.

    ``residual`` is the Numerov equation's own defect at the matching point,
    which vanishes exactly when the outward and inward solutions are one
    solution.  It is ``None`` when the energy leaves no classically allowed
    region for the state to live in.
    """
    g, M = radial_g(r, potential, l, energy, kappa=kappa,
                    relativistic=relativistic, c=c, dx=dx,
                    derivatives=derivatives)
    allowed = np.flatnonzero(g < 0.0)
    if allowed.size == 0:
        return None, 0, None, M
    m = int(np.clip(allowed[-1], 2, len(g) - 3))

    f = 1.0 - dx * dx * g / 12.0
    out = _numerov_outward(f, dx, m, float(np.sqrt(max(g[0], 1e-12))))
    inw = _numerov_inward(f, g, dx, m)
    if out[m] == 0.0 or inw[m] == 0.0:
        return None, 0, None, M

    y = np.empty(len(f))
    y[:m + 1] = out[:m + 1]
    y[m:] = inw[m:] * (out[m] / inw[m])

    # Nodes are counted on the *outward* solution only, and only where it is
    # numerically meaningful.  Both restrictions matter.  Inward of the first
    # few points y ~ r^(l+1/2) has underflowed to roundoff, whose sign flips
    # freely: counting raw sign changes over the whole array reports 5 nodes
    # for a 1s state.  And past the matching point the two halves do not join
    # except at an eigenvalue, so sign changes there count the mismatch, not
    # the state.
    head = out[:m + 1]
    scale = np.max(np.abs(head))
    meaningful = np.abs(head) > _NODE_FLOOR * scale
    nodes = int(np.count_nonzero(np.diff(np.signbit(head[meaningful]))))

    residual = ((f[m - 1] * y[m - 1] + f[m + 1] * y[m + 1]
                 - (12.0 - 10.0 * f[m]) * y[m]) / (dx * dx * y[m]))
    return float(residual), nodes, y, M


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
    from .relativity import SCALAR_KAPPA, _resolve

    r = np.asarray(r, dtype=float)
    potential = np.asarray(potential, dtype=float)
    x = np.log(r)
    dx = float(x[1] - x[0])
    key = _resolve(treatment)
    relativistic = key != "none"
    k = (SCALAR_KAPPA if key == "scalar"
         else (SCALAR_KAPPA if kappa is None else int(kappa)))

    def probe(energy):
        return _match(r, potential, l, energy, kappa=k,
                      relativistic=relativistic, c=c, dx=dx,
                      derivatives=derivatives)

    # A bound state sits between the bottom of the well and zero.  The lower
    # bound is deliberately *not* min(V): on a log grid the innermost point
    # sits at 1e-8/Z, where a Coulomb tail is about -1e8 Z^2, which is a real
    # value of the potential but nowhere near any eigenvalue.  Bracketing from
    # there would spend the whole bisection crossing empty energy.
    Z = max(float(atomic_number), 1.0)
    lo, hi = -(2.0 * Z * Z + 10.0), -1e-12

    # A guess (the relativistic loop supplies the previous iterate) is worth a
    # narrow bracket first: if it holds the state, phase 1 starts decades in.
    if energy_guess is not None:
        guess = float(energy_guess)
        width = 0.25 * max(abs(guess), 1.0)
        near_lo, near_hi = guess - width, min(guess + width, hi)
        if lo < near_lo < near_hi < 0.0:
            r_lo, n_lo, _, _ = probe(near_lo)
            r_hi, n_hi, _, _ = probe(near_hi)
            if (r_lo is not None and r_hi is not None
                    and n_lo <= n_nodes < n_hi):
                lo, hi = near_lo, near_hi

    # The search bisects on one combined criterion, because neither half is
    # sufficient alone.  The node count selects *which* state -- it is monotone
    # in the energy and cannot be fooled by crowding -- but it increments
    # somewhere above the eigenvalue, not at it, so bisecting on it converges
    # to the transition rather than to the level.  The matching residual
    # locates the level exactly, crossing zero there, but it also has a pole
    # between consecutive levels, so its sign alone would not say which one
    # was found.  Together they are monotone over the whole range:
    #
    #   too low   <=>  nodes < n_nodes, or nodes == n_nodes and residual < 0
    #   too high  <=>  nodes > n_nodes, or nodes == n_nodes and residual > 0
    #
    # An energy under the well has no allowed region at all, which is "too
    # low".
    def too_low(energy):
        residual, nodes, _, _ = probe(energy)
        if residual is None or nodes < n_nodes:
            return True
        if nodes > n_nodes:
            return False
        return residual < 0.0

    for _ in range(int(max_iterations)):
        mid = 0.5 * (lo + hi)
        if too_low(mid):
            lo = mid
        else:
            hi = mid
        if hi - lo <= tolerance * max(1.0, abs(lo)):
            break
    energy = 0.5 * (lo + hi)

    residual, nodes, y, M = probe(energy)
    if y is None:
        raise ValueError(
            f"no bound state with {n_nodes} node(s) for l={l} in this "
            f"potential: the search left no classically allowed region")

    P = np.exp(x / 2.0) * y
    if relativistic and M is not None:
        P = P * np.sqrt(M)
    norm = np.sqrt(np.trapezoid(P * P * r, x))
    P = P / norm
    return (-P if P[len(P) // 4] < 0 else P), float(energy)
