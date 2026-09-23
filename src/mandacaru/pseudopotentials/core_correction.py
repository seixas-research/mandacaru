# -*- coding: utf-8 -*-
# file: pseudopotentials/core_correction.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Nonlinear core correction (Louie, Froyen and Cohen).

Unscreening a pseudopotential subtracts the Hartree and exchange-correlation
potentials of the *valence* density alone:

.. math::

    V^{\text{ion}}_{\text{loc}} = V^{\text{scr}}_{\text{loc}}
        - V_H[\tilde\rho_v] - V_{xc}[\tilde\rho_v] .

That step is exact for :math:`V_H`, which is linear in the density, and
**wrong** for :math:`V_{xc}`, which is not: the all-electron potential was
screened by :math:`V_{xc}[\rho_c + \rho_v]`, and
:math:`V_{xc}[\rho_c+\rho_v] - V_{xc}[\rho_v] \neq V_{xc}[\rho_c]`.  The error is
whatever nonlinearity of :math:`V_{xc}` the core and valence densities sample
together, so it is largest exactly where core and valence overlap -- alkali and
alkaline-earth metals, and the 3d series, where the semicore and valence shells
sit on top of each other.

Louie, Froyen and Cohen (Phys. Rev. B **26**, 1738 (1982)) fixed this by keeping
a **partial core density**: unscreen with :math:`V_{xc}[\tilde\rho_c+\tilde\rho_v]`
and carry :math:`\tilde\rho_c` with the pseudopotential, so the same core
density can be added back wherever :math:`V_{xc}` is evaluated later.  The full
:math:`\rho_c` would be useless -- it is sharply peaked at the nucleus and would
demand the plane-wave cutoff the pseudopotential exists to avoid -- so inside a
radius :math:`r_{\text{nlcc}}` it is replaced by a smooth function that matches
the true core density in value and slope:

.. math::

    \tilde\rho_c(r) = \begin{cases}
        A\,\dfrac{\sin(Br)}{r}, & r < r_{\text{nlcc}} \\[4pt]
        \rho_c(r),              & r \ge r_{\text{nlcc}}
      \end{cases}

with :math:`A` and :math:`B` fixed by those two conditions
(:func:`smooth_core_density`).  The form is the original paper's; it is nodeless
on :math:`(0, r_{\text{nlcc}})`, finite at the origin, and its Fourier transform
decays like the pseudo valence density's.

.. note::

   **What this does and does not buy in Mandacaru.**  The correction enters the
   *generation* of the pseudopotential, where it belongs: the unscreening above
   is where the nonlinearity is committed, and fixing it there changes the ionic
   local potential every later calculation uses.  But Mandacaru's many-body
   Hamiltonian is a wavefunction method -- it has no exchange-correlation
   functional at run time to add :math:`\tilde\rho_c` back into.  So the run-time
   half of the correction, which in a DFT code re-evaluates
   :math:`V_{xc}[\rho_c+\rho_v]` at every step, has nothing to act on here and is
   not performed.  :attr:`core_density` is stored and reported so a DFT
   consumer of a Mandacaru-generated pseudopotential can do it.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

#: The partial core density is matched where ``rho_core`` falls to this
#: multiple of the valence density.  1.0 is the crossing point, the usual
#: choice; a larger value moves the match outward and makes the core smoother.
DEFAULT_CROSSOVER = 1.0

#: Bracket of the ``x cot x`` root, kept strictly inside ``(0, pi)`` where the
#: function is monotone and the smooth core stays nodeless.
_X_BRACKET = (1e-9, np.pi - 1e-9)


def crossover_radius(r, core, valence, ratio: float = DEFAULT_CROSSOVER):
    r"""Where the core density falls to ``ratio`` times the valence density.

    The outermost crossing is taken: inside it the core dominates and is worth
    smoothing, outside it the core is a small tail that must be left alone.
    Returns ``None`` when the core never dominates -- a valence-only atom such
    as hydrogen, which needs no correction.
    """
    r = np.asarray(r, dtype=float)
    excess = np.asarray(core, dtype=float) - float(ratio) * np.asarray(
        valence, dtype=float)
    sign_change = np.nonzero((excess[:-1] > 0.0) & (excess[1:] <= 0.0))[0]
    if sign_change.size == 0:
        return None
    i = int(sign_change[-1])
    # Linear interpolation of the crossing between the bracketing nodes.
    span = excess[i] - excess[i + 1]
    if abs(span) < 1e-300:
        return float(r[i])
    return float(r[i] + (r[i + 1] - r[i]) * excess[i] / span)


def smooth_core_density(r, core, r_nlcc: float):
    r"""The Louie-Froyen-Cohen partial core density and its ``(A, B)``.

    ``A sin(Br)/r`` inside ``r_nlcc``, the true ``core`` outside, matched in
    value and first derivative.  With :math:`x = Br_{\text{nlcc}}` the slope
    condition is

    .. math::

        x\cot x = 1 + r_{\text{nlcc}}\,
            \frac{\rho_c'(r_{\text{nlcc}})}{\rho_c(r_{\text{nlcc}})} ,

    whose left side falls monotonically from 1 to :math:`-\infty` on
    :math:`(0,\pi)`.  A decaying core makes the right side less than 1, so the
    root exists and is unique, and keeping :math:`x < \pi` keeps
    :math:`\sin(Br)` from reaching a node inside the matching radius.

    Returns
    -------
    (rho_tilde, A, B) : (ndarray, float, float)
    """
    r = np.asarray(r, dtype=float)
    core = np.asarray(core, dtype=float)
    r_nlcc = float(r_nlcc)

    index = int(np.argmin(np.abs(r - r_nlcc)))
    r_match = float(r[index])
    value = float(core[index])
    if value <= 0.0:
        raise ValueError(
            f"the core density vanishes at r_nlcc = {r_match:.4f} Bohr, so "
            "there is nothing to match a smooth continuation to")
    slope = float(np.gradient(core, r, edge_order=2)[index])
    target = 1.0 + r_match * slope / value
    if target >= 1.0:
        raise ValueError(
            f"the core density is not decreasing at r_nlcc = {r_match:.4f} "
            "Bohr, so the sin(Br)/r form has no nodeless match there")

    def condition(x):
        return x / np.tan(x) - target

    x = brentq(condition, *_X_BRACKET, xtol=1e-14, rtol=1e-15)
    B = x / r_match
    A = value * r_match / np.sin(x)

    smooth = np.where(r < r_match, A * np.sin(B * r) / r, core)
    return smooth, float(A), float(B)


def partial_core_density(r, core, valence, r_nlcc=None,
                         ratio: float = DEFAULT_CROSSOVER):
    """``(rho_tilde, details)``: the density to unscreen with, and how it was made.

    ``r_nlcc`` defaults to :func:`crossover_radius`.  When the core never
    dominates the valence density the correction is not worth making and the
    returned density is zero, with ``details["applied"] = False`` -- the same
    answer as not asking for a core correction at all.
    """
    r = np.asarray(r, dtype=float)
    core = np.asarray(core, dtype=float)
    radius = (crossover_radius(r, core, valence, ratio) if r_nlcc is None
              else float(r_nlcc))
    if radius is None:
        return np.zeros_like(r), {"applied": False, "r_nlcc": None,
                                  "reason": "the core density never exceeds "
                                            "the valence density"}
    smooth, A, B = smooth_core_density(r, core, radius)
    shell = 4.0 * np.pi * r * r
    return smooth, {
        "applied": True,
        "r_nlcc": float(radius),
        "amplitude": A,
        "wavevector": B,
        "core_electrons": float(np.trapezoid(core * shell, r)),
        "partial_core_electrons": float(np.trapezoid(smooth * shell, r)),
    }
