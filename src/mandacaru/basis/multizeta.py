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
from scipy.interpolate import CubicSpline

from ..units import to_bohr
from ._angular import sample_dtype as _dtype
from ._angular import (radial_angular_gradient, spherical_coords,
                       spherical_harmonic)
from .base import BasisFunction

#: The split-valence schemes ``zeta_split`` names: which function every extra
#: zeta is split from -- the first zeta, or the previous (last) one.
ZETA_SPLITS = ("first_zeta", "last_zeta")
DEFAULT_ZETA_SPLIT = "first_zeta"

#: ``zeta_split="last_zeta"``: the fraction of the orbital's **squared** norm
#: left outside the split radius, each higher zeta splitting the previous one
#: with the fraction halved.
DEFAULT_SPLIT_NORM = 0.15

#: ``zeta_split="first_zeta"``: the **norm** -- not the squared norm -- of the
#: tail left outside the split radius of the second, third and fourth zeta,
#: every one of them split from the *first* zeta.  0.16 is a squared-norm
#: fraction of 0.0256, so it is not comparable with
#: :data:`DEFAULT_SPLIT_NORM` digit for digit.
DEFAULT_TAIL_NORMS = (0.16, 0.3, 0.6)


def validate_zeta_split(spec):
    """Normalize a ``zeta_split`` option to one of :data:`ZETA_SPLITS`, or
    ``None`` when it was not written."""
    if spec is None:
        return None
    key = str(spec).strip().lower().replace("-", "_")
    if key not in ZETA_SPLITS:
        raise ValueError(f"zeta_split must be one of {list(ZETA_SPLITS)}, "
                         f"got {spec!r}")
    return key


def resolve_split_scheme(zeta_split=None):
    """``(split_norm, tail_norms)`` of the split-valence scheme ``zeta_split``.

    The scheme fixes its own parameter, which is never written by hand:
    ``"first_zeta"`` (the default) returns ``(None, DEFAULT_TAIL_NORMS)`` --
    every extra zeta split from the first by the tail's **norm** -- and
    ``"last_zeta"`` returns ``(DEFAULT_SPLIT_NORM, None)`` -- each zeta split
    from the previous one by the tail's **squared** norm, halved per zeta.
    """
    scheme = validate_zeta_split(zeta_split) or DEFAULT_ZETA_SPLIT
    if scheme == "first_zeta":
        return None, DEFAULT_TAIL_NORMS
    return DEFAULT_SPLIT_NORM, None

#: Default NAO size.  Double zeta plus polarization: single zeta gives a shell
#: no radial freedom (it cannot contract or expand) and no angular freedom (it
#: cannot bend into a bond), and both matter for chemistry.  DZP is also where
#: the real-space grid stops being the limiting approximation -- see the module
#: docstring on why TZ and beyond need a finer mesh to be worth their cost.
DEFAULT_NAO_SIZE = "DZP"

#: Recognized basis-size names -> (number of zetas, polarization shells).
ZETA_NAMES = {
    "SZ": (1, 0), "SZP": (1, 1),
    "DZ": (2, 0), "DZP": (2, 1), "DZ2P": (2, 2),
    "TZ": (3, 0), "TZP": (3, 1), "TZ2P": (3, 2),
    "QZ": (4, 0), "QZP": (4, 1), "QZ2P": (4, 2),
}


def resolve_zeta(name):
    """Resolve a basis-size spec to ``(n_zeta, n_polarization)``.

    Accepts a name from :data:`ZETA_NAMES` (case-insensitive) or an explicit
    ``(n_zeta, n_polarization)`` pair.
    """
    if isinstance(name, (tuple, list)) and len(name) == 2:
        return int(name[0]), int(name[1])
    key = str(name).strip().upper().replace("-", "").replace(" ", "")
    if key not in ZETA_NAMES:
        raise ValueError(
            f"unknown basis size {name!r}; use one of "
            f"{sorted(ZETA_NAMES)} or an (n_zeta, n_polarization) pair")
    return ZETA_NAMES[key]


# --------------------------------------------------------------------------- #
# Split-valence construction.
# --------------------------------------------------------------------------- #

def split_radius(r: np.ndarray, radial: np.ndarray, split_norm: float) -> float:
    r"""Radius outside which the orbital carries ``split_norm`` of its norm.

    Solves :math:`\int_{r_s}^{r_c}|R|^2r^2dr = \texttt{split\_norm}` by scanning
    the cumulative tail norm.  A larger ``split_norm`` pushes :math:`r_s` inward,
    giving a longer-ranged (softer) extra zeta.
    """
    from scipy.integrate import cumulative_trapezoid

    density = np.asarray(radial) ** 2 * r ** 2
    total = np.trapezoid(density, r)
    if total <= 0:
        raise ValueError("the radial function has zero norm")
    inner = cumulative_trapezoid(density, r, initial=0.0)
    tail = (total - inner) / total
    # tail decreases monotonically from 1 to 0; find where it crosses.
    index = int(np.argmin(np.abs(tail - float(split_norm))))
    return float(r[max(index, 1)])


def split_valence_tail(r: np.ndarray, radial: np.ndarray, l: int,
                       r_split: float) -> np.ndarray:
    r"""The split-valence extra-zeta function ``R_1 - r^l(a - b r^2)``.

    ``a`` and ``b`` are fixed by matching value and slope of :math:`R_1/r^l` at
    ``r_split``, so the difference and its first derivative vanish there and the
    result is zero beyond it.
    """
    r = np.asarray(r, dtype=float)
    radial = np.asarray(radial, dtype=float)

    spline = CubicSpline(r, radial)
    value = float(spline(r_split))
    slope = float(spline(r_split, 1))

    # R(r) = r^l (a - b r^2)  =>  match R and R' at r_split.
    #   value = rs^l (a - b rs^2)
    #   slope = l rs^(l-1) (a - b rs^2) - 2 b rs^(l+1)
    #         = l * value / rs - 2 b rs^(l+1)
    b = (l * value / r_split - slope) / (2.0 * r_split ** (l + 1))
    a = value / r_split ** l + b * r_split ** 2

    inside = r < r_split
    smooth = np.where(inside, r ** l * (a - b * r * r), 0.0)
    return np.where(inside, radial - smooth, 0.0)


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
        self._spline = CubicSpline(table.r, table.values, extrapolate=False)
        self._slope_spline = self._spline.derivative()

    @property
    def state(self) -> tuple[int, int, int]:
        return (self.n, self.l, self.m)

    @property
    def kind(self) -> str:
        """``"diffuse"``, ``"polarization"`` or ``"zeta"``."""
        return ("diffuse" if self.diffuse
                else "polarization" if self.polarization else "zeta")

    def radial(self, r) -> np.ndarray:
        """Interpolated radial function; zero outside the confining sphere.

        Below the first table point the function continues as
        ``R(r_0) (r / r_0)^l``.  Without that, a table starting at ``r_0 > 0``
        made every s function vanish within ``r_0`` of its center, and a nucleus
        sitting exactly on a grid node lost the orbital's peak sample (LiH
        PAW-LCAO-DZP moved by 3 eV).
        """
        r = np.asarray(r, dtype=float)
        r0 = float(self.table.r[0])
        values = self._spline(np.maximum(r, r0))
        if r0 > 0.0:
            values = np.where(r < r0, values * (np.maximum(r, 0.0) / r0) ** self.l,
                              values)
        return np.where(np.isnan(values) | (r >= self.r_c), 0.0, values)

    def radial_slope(self, r) -> np.ndarray:
        """``dR/dr`` of :meth:`radial`: the spline's derivative, and the
        derivative of the power-law continuation below the first table
        point."""
        r = np.asarray(r, dtype=float)
        r0 = float(self.table.r[0])
        slope = self._slope_spline(np.maximum(r, r0))
        if r0 > 0.0:
            below = (self._spline(r0) * self.l
                     * np.maximum(r, 0.0) ** max(self.l - 1, 0) / r0 ** self.l
                     if self.l else np.zeros_like(r))
            slope = np.where(r < r0, below, slope)
        return np.where(np.isnan(slope) | (r >= self.r_c), 0.0, slope)

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
                tail_norms=None) -> list[RadialTable]:
    """First-zeta table plus ``n_zeta - 1`` split-valence refinements.

    Two schemes share the split-valence polynomial and differ in *what* is
    split and *where*:

    * ``last_zeta`` (``tail_norms=None``) -- each successive zeta splits the
      *previous* one, at the radius leaving ``split_norm`` of its **squared
      norm** outside, halved each time;
    * ``first_zeta``, selected by ``tail_norms`` (a sequence, one entry per
      extra zeta) -- every zeta splits the **first** one, at the radius
      leaving a tail of **norm** ``tail_norms[k]`` (so a squared-norm
      fraction of its square) outside.

    Either way the added functions become progressively shorter-ranged.
    """
    first = np.asarray(radial, dtype=float)
    tables = [RadialTable(r=r, values=first, n=n, l=l, zeta=1)]
    if tail_norms is not None and int(n_zeta) - 1 > len(tail_norms):
        raise ValueError(
            f"{int(n_zeta)} zetas need {int(n_zeta) - 1} tail norms, got "
            f"{tuple(tail_norms)!r}")
    current = first
    norm = float(split_norm)
    for zeta in range(2, int(n_zeta) + 1):
        if tail_norms is not None:
            current, norm = first, float(tail_norms[zeta - 2]) ** 2
        r_split = split_radius(r, current, norm)
        values = split_valence_tail(r, current, l, r_split)
        if not np.any(np.abs(values) > 1e-12):
            break                              # nothing left to split
        # Normalize the extra zeta.  Its span is unchanged, but a function
        # carrying a few percent of the norm makes the overlap matrix (and the
        # grid kinetic energy that screens it) needlessly ill-conditioned.
        values = values / np.sqrt(np.trapezoid(values * values * r * r, r))
        tables.append(RadialTable(r=r, values=values, n=n, l=l, zeta=zeta))
        current, norm = values, norm * 0.5
    return tables


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
