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

from .. import _native
from .native_context import context

#: The partial core density is matched where ``rho_core`` falls to this
#: multiple of the valence density.  1.0 is the crossing point, the usual
#: choice; a larger value moves the match outward and makes the core smoother.
DEFAULT_CROSSOVER = 1.0


def _array(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=float)


def crossover_radius(r, core, valence, ratio: float = DEFAULT_CROSSOVER):
    r"""Where the core density falls to ``ratio`` times the valence density.

    The outermost crossing is taken: inside it the core dominates and is worth
    smoothing, outside it the core is a small tail that must be left alone.
    Returns ``None`` when the core never dominates -- a valence-only atom such
    as hydrogen, which needs no correction.
    """
    return _native.paw_crossover_radius(context(), _array(r), _array(core),
                                        _array(valence), float(ratio))


def smooth_core_density(r, core, r_nlcc: float):
    r"""The Louie-Froyen-Cohen partial core density and its ``(A, B)``.

    ``A sin(Br)/r`` inside ``r_nlcc``, the true ``core`` outside, matched in
    value and first derivative.  With :math:`x = Br_{\text{nlcc}}` the slope
    condition is

    .. math::

        x\cot x = 1 + r_{\text{nlcc}}\,
            \frac{\rho_c'(r_{\text{nlcc}})}{\rho_c(r_{\text{nlcc}})} ,

    whose left side falls monotonically from 1 to :math:`-\infty` on
    :math:`(0,\pi)`, so the root exists and is unique and :math:`\sin(Br)`
    reaches no node inside the matching radius.

    Returns
    -------
    (rho_tilde, A, B) : (ndarray, float, float)
    """
    return _native.paw_smooth_core_density(context(), _array(r), _array(core),
                                           float(r_nlcc))


def partial_core_density(r, core, valence, r_nlcc=None,
                         ratio: float = DEFAULT_CROSSOVER):
    """``(rho_tilde, details)``: the density to unscreen with, and how it was made.

    ``r_nlcc`` defaults to :func:`crossover_radius`.  When the core never
    dominates the valence density the correction is not worth making and the
    returned density is zero, with ``details["applied"] = False`` -- the same
    answer as not asking for a core correction at all.
    """
    out = _native.paw_partial_core_density(
        context(), _array(r), _array(core), _array(valence),
        None if r_nlcc is None else float(r_nlcc), float(ratio))
    if out["r_nlcc"] is None:
        return out["density"], {"applied": False, "r_nlcc": None,
                                "reason": "the core density never exceeds "
                                          "the valence density"}
    return out["density"], {
        "applied": True,
        "r_nlcc": out["r_nlcc"],
        "amplitude": out["amplitude"],
        "wavevector": out["wavevector"],
        "core_electrons": out["core_electrons"],
        "partial_core_electrons": out["partial_core_electrons"],
    }
