# -*- coding: utf-8 -*-
# file: basis/gaussian_families.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Named Gaussian basis-set families, generated natively from their structure.

Mandacaru never reads a basis-set table.  The STO-nG contractions are least-squares
fits to Slater orbitals (:mod:`mandacaru.basis.sto_ng`) and 6-31G(d) is assembled
from those fits (:mod:`mandacaru.basis.pople`).  This module generalizes that
construction to the standard **named** families -- minimal, Pople split-valence,
Dunning correlation-consistent and Karlsruhe ``def2`` -- by separating a basis
set into two things:

1. its **structure** -- a :class:`GaussianRecipe`: how many primitives contract
   each core shell, how the valence shells are split, which polarization,
   diffuse and core-correlating functions are added and on which atoms.  That
   structure is what a name like ``6-311+G(2df,2p)`` or ``aug-cc-pVTZ``
   encodes, and :func:`parse_basis_name` recovers it from the name alone;
2. its **numbers** -- exponents and contraction coefficients, generated for the
   atom at hand from the cached Slater-orbital fits, Slater's rules for the
   orbital exponents, and even-tempered heuristics for the polarization,
   diffuse and tight functions.

The result has the **same shell structure and function count** as the published
basis of the same name (``cc-pVTZ`` carbon is ``4s3p2d1f``, 30 functions), but
its exponents are Mandacaru's own, not the published, molecule-optimized values.
That is the same relationship the native 6-31G(d) already bears to Pople's.

Provenance
----------
Because the numbers differ, the name alone would be misleading, so every basis
built here carries a :attr:`GaussianRecipe.provenance` qualifier --
``native:cc-pVTZ-recipe`` -- and reports under it (the dry run prints it as the
basis).  Published basis-set convergence results and term-by-term comparisons
with another package's ``cc-pVTZ`` do **not** transfer.  ``published:`` is a
reserved namespace that deliberately resolves to nothing, so asking for
tabulated data fails loudly instead of being served a recipe, and
:func:`basis_set_data` writes out every exponent, coefficient and the angular
convention they are meant in, which is what an actual comparison needs.

Construction rules
------------------
* A shell ``(n, l)`` with contraction *pattern* ``(k_1, k_2, ...)`` is built
  from one Slater-orbital fit of ``sum(k_i)`` Gaussians, partitioned
  tightest-first: the first ``k_1`` primitives form a contracted function, the
  next ``k_2`` the next, and so on.  ``(3, 1)`` is the 6-31G valence split,
  ``(3, 1, 1)`` the 6-311G triple split, ``(6,)`` an STO-6G core.
* **Core** shells are those below the valence principal quantum number;
  hydrogen and helium have no core.
* **Polarization** functions of angular momentum ``l`` are uncontracted, with
  exponents ``f_l * zeta_val^2`` spread geometrically (ratio 3) when several
  are requested; ``f_2 = 0.4`` reproduces the native 6-31G(d) choice.
* **Diffuse** functions take the most diffuse exponent the atom already has in
  that angular momentum, divided by 3.5 -- the ratio the published augmented
  sets follow.
* **Core-correlating (tight)** functions (``cc-pCVXZ``) sit at three times the
  tightest valence exponent of their angular momentum.

Everything is deterministic, cached, and produced in milliseconds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ._config import ground_state_config
from .sto_ng import _fit_reference, slater_exponent

#: A contracted shell: ``(l, exponents, coefficients)``.
ShellSpec = tuple[int, np.ndarray, np.ndarray]

from ..models.basis import (_L_LETTERS, _L_OF, HEAVY_POLARIZATION_FACTOR, HYDROGEN_POLARIZATION_FACTOR, POLARIZATION_RATIO, DIFFUSE_RATIO, TIGHT_RATIO, MAX_CONTRACTION, GaussianRecipe, _STO_RE, _POPLE_RE, _DUNNING_RE, _KARLSRUHE_RE, _ZETA_OF, NATIVE_NAMESPACE, PUBLISHED_NAMESPACE, _pol_spec, _parse_pople, _parse_dunning, _KARLSRUHE_TABLE, _parse_karlsruhe, parse_basis_name, NAMED_BASIS_SETS, available_basis_names)


def _valence_principal(Z: int) -> int:
    return max(n for (n, _l) in ground_state_config(Z))


def pattern_shells(Z: int, n: int, l: int,
                   pattern: tuple[int, ...]) -> list[ShellSpec]:
    """Contract one Slater-orbital fit into the functions of ``pattern``.

    A fit of ``sum(pattern)`` Gaussians to the ``(n, l)`` Slater orbital is
    partitioned tightest-first; each group with more than one primitive keeps
    the fit's coefficients (a contracted function), a single primitive becomes
    an uncontracted function.
    """
    total = int(sum(pattern))
    alpha0, coeff0, _ = _fit_reference(int(n), int(l), total)
    zeta = slater_exponent(Z, n, l)
    exps = np.asarray(alpha0, dtype=float) * zeta ** 2
    coeffs = np.asarray(coeff0, dtype=float)
    shells: list[ShellSpec] = []
    start = 0
    for k in pattern:
        stop = start + int(k)
        if k == 1:
            shells.append((l, exps[start:stop].copy(), np.ones(1)))
        else:
            shells.append((l, exps[start:stop].copy(), coeffs[start:stop].copy()))
        start = stop
    return shells


def _polarization_exponents(Z: int, l: int, count: int) -> np.ndarray:
    n_val = _valence_principal(Z)
    zeta = slater_exponent(Z, n_val, 0)
    table = HYDROGEN_POLARIZATION_FACTOR if Z <= 2 else HEAVY_POLARIZATION_FACTOR
    factor = table.get(l, 0.2 * l)
    base = factor * zeta ** 2
    k = np.arange(count)
    return base * POLARIZATION_RATIO ** ((count - 1) / 2.0 - k)   # tightest first


def gaussian_shells(atomic_number: int, recipe: GaussianRecipe) -> list[ShellSpec]:
    """Every contracted shell ``(l, exponents, coefficients)`` of ``recipe`` for an atom.

    Order: core shells, valence shells (by ``(n, l)``), core-correlating tight
    functions, polarization shells (by ``l``), diffuse functions (by ``l``).
    """
    Z = int(atomic_number)
    n_val = _valence_principal(Z)
    heavy = Z >= 3
    shells: list[ShellSpec] = []
    valence_shells: list[ShellSpec] = []

    for (n, l) in sorted(ground_state_config(Z)):
        if n < n_val:
            shells.extend(pattern_shells(Z, n, l, recipe.core))
        else:
            valence_shells.extend(pattern_shells(Z, n, l, recipe.valence))
    shells.extend(valence_shells)

    # Core-correlating tight functions (heavy atoms only).
    if heavy:
        for l, count in recipe.tight:
            tightest = max((float(e.max()) for (ll, e, _c) in valence_shells
                            if ll == l), default=None)
            if tightest is None:
                continue
            for k in range(count):
                shells.append((l, np.array([tightest * TIGHT_RATIO ** (k + 1)]),
                               np.ones(1)))

    # Polarization.
    pol = (recipe.polarization if Z >= recipe.polarization_min_z
           else (recipe.polarization_h if Z <= 2 else ()))
    for l, count in sorted(pol):
        for alpha in _polarization_exponents(Z, l, count):
            shells.append((l, np.array([alpha]), np.ones(1)))

    # Diffuse.
    if recipe.diffuse_present:
        ls = sorted({l for (l, _e, _c) in shells if l <= recipe.diffuse_l_max})
    else:
        ls = list(recipe.diffuse if heavy else recipe.diffuse_h)
    for l in ls:
        present = [float(e.min()) for (ll, e, _c) in shells if ll == l]
        if present:
            alpha = min(present) / DIFFUSE_RATIO
        else:
            # No function of this l yet (a diffuse p on lithium): take the
            # scale from the most diffuse s function instead.
            alpha = min(float(e.min()) for (ll, e, _c) in shells
                        if ll == 0) / DIFFUSE_RATIO
        shells.append((l, np.array([alpha]), np.ones(1)))
    return shells


def count_functions(atomic_number: int, recipe: GaussianRecipe) -> int:
    """Number of basis functions (``2l + 1`` per shell) for one atom."""
    return sum(2 * l + 1 for (l, _e, _c) in gaussian_shells(atomic_number, recipe))


def shell_notation(atomic_number: int, recipe: GaussianRecipe) -> str:
    """The ``[3s2p1d]`` contracted-shell notation of an atom in ``recipe``."""
    counts: dict[int, int] = {}
    for (l, _e, _c) in gaussian_shells(atomic_number, recipe):
        counts[l] = counts.get(l, 0) + 1
    return "[" + "".join(f"{counts[l]}{_L_LETTERS[l]}" for l in sorted(counts)) + "]"


# --------------------------------------------------------------------------- #
# Serializing what was actually generated.
# --------------------------------------------------------------------------- #

#: How the generated numbers are meant: what the coefficients multiply, what the
#: angular factor is, and which units everything is in.  Serialized with every
#: export so the numbers cannot be read under another package's conventions.
ANGULAR_CONVENTION = {
    "harmonics": "complex orthonormal spherical harmonics Y_l^m(theta, phi)",
    "functions_per_shell": "2l + 1, m = -l ... +l (spherical, never Cartesian)",
    "radial": "r^l * sum_i d_i exp(-alpha_i r^2)",
    "primitive_normalization": (
        "coefficients are for unit-normalized primitives; the contraction is "
        "renormalized to <phi|phi> = 1 in 3D"),
    "units": "exponents in Bohr^-2",
}


def shells_record(elements, shells_of, name: str, family: str,
                  structure: str) -> dict:
    """Assemble a provenance record from any ``shells_of(Z)`` generator.

    The shared body of :func:`basis_set_data`: every native Gaussian family
    reports the same way, whatever generated its shells.
    """
    from ase.data import atomic_numbers, chemical_symbols

    atoms = {}
    for element in elements:
        if isinstance(element, str):
            symbol = element.capitalize()
            if symbol not in atomic_numbers:
                raise ValueError(f"unknown element symbol {element!r}")
            Z = int(atomic_numbers[symbol])
        else:
            Z, symbol = int(element), chemical_symbols[int(element)]
        shells = [(int(l), np.asarray(e).ravel(), np.asarray(c).ravel())
                  for (l, e, c) in shells_of(Z)]
        counts: dict[int, int] = {}
        for (l, _e, _c) in shells:
            counts[l] = counts.get(l, 0) + 1
        atoms[symbol] = {
            "atomic_number": Z,
            "notation": "[" + "".join(f"{counts[l]}{_L_LETTERS[l]}"
                                      for l in sorted(counts)) + "]",
            "n_functions": sum(2 * l + 1 for (l, _e, _c) in shells),
            "shells": [{"l": l, "shell": _L_LETTERS[l],
                        "exponents": [float(x) for x in e],
                        "coefficients": [float(x) for x in c]}
                       for (l, e, c) in shells],
        }
    return {
        "name": name,
        "provenance": f"{NATIVE_NAMESPACE}:{name}-recipe",
        "generated": True,
        "published_data": False,
        "family": family,
        "structure": structure,
        "convention": dict(ANGULAR_CONVENTION),
        "elements": atoms,
    }


def basis_set_data(elements, recipe: GaussianRecipe) -> dict:
    """A full, self-describing record of the basis generated for ``elements``.

    Carries the provenance, the recipe that produced the structure, the angular
    and normalization conventions, and every exponent and coefficient per
    element -- so a calculation can be reproduced, or compared against a
    published basis set, without reading Mandacaru's source.

    Parameters
    ----------
    elements : iterable
        Chemical symbols or atomic numbers.
    recipe : GaussianRecipe
        The recipe to generate from.
    """
    return shells_record(elements,
                         lambda Z: gaussian_shells(Z, recipe),
                         recipe.name, recipe.family, recipe.summary())
