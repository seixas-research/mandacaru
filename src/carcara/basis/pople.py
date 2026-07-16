# -*- coding: utf-8 -*-
# file: basis/pople.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Native Pople split-valence basis (6-31G / 6-31G(d)), generated from scratch.

Where :mod:`carcara.basis.sto_ng` builds a *minimal* basis (one contracted
Gaussian per occupied subshell), a **split-valence** basis gives the valence
shells radial flexibility by splitting each into a tight contracted part and a
loose single primitive.  The 6-31G basis of Pople and co-workers uses:

* **core** subshells -- one contraction of **6** primitives (as in STO-6G);
* **valence** subshells -- split ``3 + 1``: an inner contraction of **3**
  primitives plus **1** uncontracted outer primitive (two basis functions).

Adding a set of uncontracted **d** polarization functions on non-hydrogen atoms
gives **6-31G(d)** (a.k.a. 6-31G*).

As everywhere in Carcará, nothing is read from a basis-set table.  The primitive
exponents and contraction coefficients come from the same on-the-fly STO-nG
least-squares fit to Slater-type orbitals used by :mod:`carcara.basis.sto_ng`
(:func:`~carcara.basis.sto_ng._fit_reference`, rescaled by the subshell's Slater
exponent :math:`\zeta`): the 6-primitive core is the STO-6G contraction, and the
valence split is read off a 4-primitive fit -- its three tightest primitives form
the contracted inner function and its single most diffuse primitive the outer
function.  The polarization exponent is a Slater-based heuristic,
:math:`\alpha_d = 0.4\,\zeta_{\text{val}}^2` (e.g. Li :math:`\approx 0.17`,
C :math:`\approx 1.0`), of the right order as the tabulated 6-31G(d) values.

The result is not bit-for-bit the published 6-31G(d) (which itself comes from a
molecular energy optimization), but a faithful, self-contained split-valence +
polarization basis of the same structure.
"""

from __future__ import annotations

import numpy as np

from ._config import ground_state_config
from .sto_ng import (_fit_reference, effective_principal_number, slater_exponent,
                     sto_ng_contraction)

# A "shell" spec is (l, exponents, coefficients): a single contracted Gaussian.
ShellSpec = tuple[int, np.ndarray, np.ndarray]

CORE_PRIMITIVES = 6          # the "6" of 6-31G: core contraction length
VALENCE_INNER = 3            # the "3": inner (contracted) valence primitives
VALENCE_FIT = 4              # inner (3) + outer (1) come from a 4-Gaussian fit
POLARIZATION_FACTOR = 0.4    # alpha_d = POLARIZATION_FACTOR * zeta_valence^2


def _valence_principal(atomic_number: int) -> int:
    """Highest occupied principal quantum number (the valence shell)."""
    return max(n for (n, _l) in ground_state_config(atomic_number))


def _core_shell(Z: int, n: int, l: int) -> list[ShellSpec]:
    """One 6-primitive (STO-6G) contraction for a core subshell."""
    exps, coeffs = sto_ng_contraction(Z, n, l, CORE_PRIMITIVES)
    return [(l, np.asarray(exps), np.asarray(coeffs))]


def _split_valence_shell(Z: int, n: int, l: int) -> list[ShellSpec]:
    """Split a valence subshell ``3 + 1`` into inner contraction and outer primitive.

    A 4-Gaussian STO fit (exponents tightest-first) supplies both parts: the three
    tightest primitives are the contracted inner function, the single most diffuse
    primitive the uncontracted outer function.
    """
    alpha0, coeff0, _ = _fit_reference(int(n), int(l), VALENCE_FIT)
    zeta = slater_exponent(Z, n, l)
    exps = np.asarray(alpha0) * zeta ** 2
    coeffs = np.asarray(coeff0)

    inner = (l, exps[:VALENCE_INNER].copy(), coeffs[:VALENCE_INNER].copy())
    outer = (l, exps[VALENCE_INNER:].copy(), np.ones(1))
    return [inner, outer]


def _polarization_shell(Z: int) -> list[ShellSpec]:
    """One uncontracted ``d`` (l=2) polarization function on non-H/He atoms."""
    if Z <= 2:
        return []
    n_val = _valence_principal(Z)
    # Slater exponent of the valence s subshell sets the length scale.
    zeta = slater_exponent(Z, n_val, 0)
    alpha_d = POLARIZATION_FACTOR * zeta ** 2
    return [(2, np.array([alpha_d]), np.ones(1))]


def pople_631g_shells(atomic_number: int,
                      polarization: bool = True) -> list[ShellSpec]:
    """All 6-31G(d) contracted-shell specs ``(l, exponents, coefficients)`` for an atom.

    Core subshells give one 6-primitive contraction each; valence subshells give a
    contracted inner + uncontracted outer function; ``polarization=True`` appends a
    single ``d`` shell on non-hydrogen atoms (the ``(d)`` of 6-31G(d)).
    """
    Z = int(atomic_number)
    n_val = _valence_principal(Z)
    shells: list[ShellSpec] = []
    for (n, l) in sorted(ground_state_config(Z)):
        if n < n_val:
            shells.extend(_core_shell(Z, n, l))
        else:
            shells.extend(_split_valence_shell(Z, n, l))
    if polarization:
        shells.extend(_polarization_shell(Z))
    return shells
