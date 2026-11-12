# -*- coding: utf-8 -*-
# file: basis/multizeta.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Multiple-zeta and polarization orbitals for the numerical atomic basis.

A single confined numerical atomic orbital (NAO) per valence subshell -- a
**single-zeta** (SZ) basis -- has no radial freedom at all: the orbital can only
be occupied, never reshaped.  Chemistry needs both.

Radial flexibility: the split-valence scheme
--------------------------------------------
The extra zetas are generated from an existing zeta rather than from new
eigenvalue problems, by the *split-valence* construction (Artacho *et al.*,
1999).  Given a radial function :math:`R_1`, pick a **split radius**
:math:`r_s` such that the norm carried by the tail beyond it is a prescribed
fraction (the split norm):

.. math::

    \int_{r_s}^{r_c} |R_1(r)|^2 r^2\,dr = \texttt{split\_norm} .

Inside :math:`r_s` replace the orbital by the smooth polynomial
:math:`r^{l}(a - b r^2)` matched in value and slope at :math:`r_s`, and take the
**difference**

.. math::

    R_2(r) = \begin{cases}
      R_1(r) - r^{l}\,(a - b r^{2}), & r < r_s \\
      0, & r \ge r_s .
    \end{cases}

:math:`R_2` is strictly shorter-ranged than :math:`R_1`, which is exactly the
point: the variational space gains the ability to contract or expand the shell,
and the added function is cheap because it vanishes early.  Triple and quadruple
zeta repeat the construction, each new function shorter-ranged than the last.
``zeta_split`` says *which* function every extra zeta is split from
(:data:`ZETA_SPLITS`):

* ``"first_zeta"`` (the default) -- every extra zeta is split from the
  **first** one, at the radius leaving a tail of **norm**
  :data:`DEFAULT_TAIL_NORMS` ``[k]`` outside;
* ``"last_zeta"`` -- each extra zeta is split from the **previous** one, at
  the radius leaving :data:`DEFAULT_SPLIT_NORM` of its **squared** norm
  outside, the fraction halved for every further zeta.

The scheme carries its parameter: :func:`resolve_split_scheme` derives the
tail norms or the split norm from ``zeta_split``, and neither is a basis
option of its own.

Angular flexibility: polarization
---------------------------------
Bond formation distorts an orbital in directions its own :math:`l` cannot
describe -- a hydrogen 1s polarizes toward its partner, which needs :math:`p`
character.  A **polarization shell** adds :math:`l+1` for the highest occupied
:math:`l`, solved in the same confining sphere so it stays compact.

The result is the usual notation: ``SZ``, ``DZ``, ``DZP``, ``TZP``, ``TZ2P``,
``QZP`` ... all generated from scratch, with no tabulated data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import _native
from ..models.radial import DEFAULT_SPLIT_NORM

# --------------------------------------------------------------------------- #
# Radial tables.
# --------------------------------------------------------------------------- #

@dataclass
class RadialTable:
    """A tabulated radial function with its quantum numbers."""

    r: np.ndarray
    values: np.ndarray
    n: int
    l: int
    zeta: int = 1
    polarization: bool = False
    #: A diffuse shell (:mod:`mandacaru.pseudopotentials.diffuse`): neither a
    #: zeta of an occupied state nor a polarization shell.
    diffuse: bool = False


# --------------------------------------------------------------------------- #
# Shell construction.
# --------------------------------------------------------------------------- #

def zeta_tables(r: np.ndarray, radial: np.ndarray, n: int, l: int,
                n_zeta: int, split_norm: float = DEFAULT_SPLIT_NORM,
                tail_norms=None, *, fractions: bool = False,
                context=None) -> list[RadialTable]:
    r"""First-zeta table plus ``n_zeta - 1`` split-valence refinements.

    The construction is native (``_native.radial_zetas``): the extra zeta is
    :math:`R_1 - r^l(a - b r^2)` below a split radius, with ``a`` and ``b``
    fixed by matching value and slope of the previous zeta there, so it and
    its slope vanish at the split and it is zero beyond; it is normalized.
    Two schemes share it and differ in *what* is split and *where*:

    * ``last_zeta`` (``tail_norms=None``) -- each successive zeta splits the
      *previous* one, at the radius leaving ``split_norm`` of its **squared
      norm** outside, halved each time;
    * ``first_zeta``, selected by ``tail_norms`` (a sequence, one entry per
      extra zeta) -- every zeta splits the **first** one, at the radius
      leaving a tail of **norm** ``tail_norms[k]`` (so a squared-norm
      fraction of its square) outside; with ``fractions`` the entries are
      that squared-norm fraction itself.

    Either way the added functions become progressively shorter-ranged, and
    fewer than ``n_zeta`` tables come back when a split leaves nothing.  ``r``
    is any increasing grid from ``r >= 0``.
    """
    if tail_norms is not None and int(n_zeta) - 1 > len(tail_norms):
        raise ValueError(
            f"{int(n_zeta)} zetas need {int(n_zeta) - 1} tail norms, got "
            f"{tuple(tail_norms)!r}")
    first = np.asarray(radial, dtype=float)
    refined = _native.radial_zetas(
        _native.Context(threads=1) if context is None else context,
        np.asarray(r, dtype=float), first, int(l), int(n_zeta),
        float(split_norm),
        None if tail_norms is None else np.asarray(tail_norms, dtype=float),
        fractions)
    values = [first] + list(refined[1:])
    return [RadialTable(r=r, values=v, n=n, l=l, zeta=zeta)
            for zeta, v in enumerate(values, start=1)]
