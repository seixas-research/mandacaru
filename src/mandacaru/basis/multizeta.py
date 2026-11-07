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
from ..units import to_bohr
from ._angular import sample_dtype as _dtype
from ._angular import (radial_angular_gradient, spherical_coords,
                       spherical_harmonic)
from .base import BasisFunction

from ..models.radial import (ZETA_SPLITS, DEFAULT_ZETA_SPLIT, DEFAULT_SPLIT_NORM, DEFAULT_TAIL_NORMS, validate_zeta_split, resolve_split_scheme, DEFAULT_NAO_SIZE, ZETA_NAMES, resolve_zeta)

# --------------------------------------------------------------------------- #
# Basis functions.
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


class TabulatedOrbital(BasisFunction):
    r"""A basis function ``R(r) Y_lm`` from a tabulated radial function.

    Used for every orbital the multiple-zeta machinery produces -- the extra
    zetas and the polarization shells alike -- so they all sample identically in
    the integral engine.
    """

    def __init__(self, table: RadialTable, m: int, center=None,
                 units: str = "angstrom", *, real: bool = True):
        if abs(int(m)) > int(table.l):
            raise ValueError(f"require |m| <= l, got l={table.l}, m={m}")
        self.table = table
        self.n, self.l, self.m = int(table.n), int(table.l), int(m)
        #: Real (molecule) or complex (crystal) spherical harmonics.
        self.real = bool(real)
        self.zeta = int(table.zeta)
        self.polarization = bool(table.polarization)
        self.diffuse = bool(table.diffuse)
        self.center = (np.zeros(3) if center is None
                       else to_bohr(center, units))
        self.r_c = float(table.r[-1])

    @property
    def state(self) -> tuple[int, int, int]:
        return (self.n, self.l, self.m)

    @property
    def kind(self) -> str:
        """``"diffuse"``, ``"polarization"`` or ``"zeta"``."""
        return ("diffuse" if self.diffuse
                else "polarization" if self.polarization else "zeta")

    def radial(self, r) -> np.ndarray:
        """Interpolated radial function; zero outside the confining sphere."""
        shape = np.shape(r)
        res = _native.radial_orbital_evaluate(
            np.ascontiguousarray(self.table.r, dtype=float),
            np.ascontiguousarray(self.table.values, dtype=float),
            np.ascontiguousarray(np.asarray(r, dtype=float).ravel(), dtype=float),
            int(self.l), float(self.r_c), False,
        )
        return res.reshape(shape)

    def radial_slope(self, r) -> np.ndarray:
        """``dR/dr`` of :meth:`radial`: the spline's derivative."""
        shape = np.shape(r)
        res = _native.radial_orbital_evaluate(
            np.ascontiguousarray(self.table.r, dtype=float),
            np.ascontiguousarray(self.table.values, dtype=float),
            np.ascontiguousarray(np.asarray(r, dtype=float).ravel(), dtype=float),
            int(self.l), float(self.r_c), True,
        )
        return res.reshape(shape)

    def evaluate(self, x, y, z) -> np.ndarray:
        r, theta, phi = spherical_coords(x, y, z, self.center)
        return np.asarray(self.radial(r)
                          * spherical_harmonic(self.l, self.m, theta, phi,
                                               real=self.real),
                          dtype=_dtype(self.real))

    def gradient(self, x, y, z, delta: float = 1e-3) -> np.ndarray:
        """Analytic gradient (``delta`` is unused)."""
        return radial_angular_gradient(
            self.radial, self.radial_slope, self.l, self.m,
            np.asarray(x, dtype=float) - self.center[0],
            np.asarray(y, dtype=float) - self.center[1],
            np.asarray(z, dtype=float) - self.center[2], real=self.real)

    def __repr__(self) -> str:
        kind = ("diffuse" if self.diffuse
                else "pol" if self.polarization else f"zeta{self.zeta}")
        return (f"TabulatedOrbital(n={self.n}, l={self.l}, m={self.m}, "
                f"{kind}, r_c={self.r_c:.2f})")


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


def polarization_tables(r: np.ndarray, l_max: int, n_polarization: int,
                        solver, n_zeta: int = 1,
                        split_norm: float = DEFAULT_SPLIT_NORM,
                        tail_norms=None) -> list[RadialTable]:
    """Polarization shells at ``l_max + 1`` (and beyond), from ``solver``.

    ``solver(n, l)`` returns ``(r, R)`` for a confined orbital.  The polarization
    shell uses the lowest principal quantum number allowed for its angular
    momentum, so it is the most compact function of that symmetry.  Each
    polarization shell may itself be split into multiple zetas.

    ``solver`` must accept an *unoccupied* subshell: a polarization shell has no
    electrons in it by definition.  Failures are not swallowed -- a silently
    absent polarization shell is indistinguishable from an unpolarized basis and
    would quietly change the answer.
    """
    tables: list[RadialTable] = []
    for offset in range(int(n_polarization)):
        l = int(l_max) + 1 + offset
        n = l + 1                              # lowest allowed principal number
        r_pol, radial = solver(n, l)
        for table in zeta_tables(r_pol, radial, n, l, n_zeta, split_norm,
                                 tail_norms=tail_norms):
            table.polarization = True
            tables.append(table)
    return tables


def build_shells(valence, solver, n_zeta: int = 1, n_polarization: int = 0,
                 split_norm: float = DEFAULT_SPLIT_NORM,
                 tail_norms=None) -> list[RadialTable]:
    """All radial tables of a multiple-zeta polarized basis for one atom.

    ``valence`` is a sequence of ``(n, l)`` subshells and ``solver(n, l)``
    returns ``(r, R)`` for each.
    """
    tables: list[RadialTable] = []
    l_max = -1
    for (n, l) in valence:
        r, radial = solver(n, l)
        tables.extend(zeta_tables(r, radial, n, l, n_zeta, split_norm,
                                  tail_norms=tail_norms))
        l_max = max(l_max, l)
    if n_polarization and l_max >= 0:
        tables.extend(polarization_tables(r, l_max, n_polarization, solver,
                                          n_zeta=1, split_norm=split_norm,
                                          tail_norms=tail_norms))
    return tables


def orbitals_from_tables(tables, center=(0.0, 0.0, 0.0),
                         units: str = "angstrom", *,
                         real: bool = True) -> list[BasisFunction]:
    """Expand radial tables over all ``m`` into basis functions, with real
    spherical harmonics (a molecule) or complex ones (``real=False``, a
    crystal)."""
    orbitals: list[BasisFunction] = []
    for table in tables:
        for m in range(-table.l, table.l + 1):
            orbitals.append(TabulatedOrbital(table, m, center=center,
                                             units=units, real=real))
    return orbitals
