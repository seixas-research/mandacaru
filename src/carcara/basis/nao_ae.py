# -*- coding: utf-8 -*-
# file: basis/nao_ae.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""All-electron numerical atomic orbitals (NAO-AE).

The **NAO-AE** family is the all-electron counterpart of the confined
:mod:`~carcara.basis.nao` orbitals: every function is a numerically tabulated
radial part times a spherical harmonic, :math:`\phi(\mathbf r) = u(r)/r \cdot
Y_{lm}`, generated from scratch for each element by solving one-dimensional
radial problems -- no tabulated basis-set data anywhere.  Three ideas define it.

**1. The minimal basis is the atom itself.**  A spherical, self-consistent
all-electron LDA atom (:func:`~carcara.basis.atomic_solver.solve_atom`) supplies
the effective potential :math:`v_\text{free}(r)`.  Every occupied shell
:math:`(n, l)` -- core shells included, which is what makes the family
*all-electron* -- is then re-solved as a bound state of the **basis-defining
potential**

.. math::

    v_\text{basis}(r) = v_\text{free}(r) + v_\text{cut}(r),

so the minimal functions are the atom's own orbitals, gently localized.

**2. Localization by a smooth wall, not a hard sphere.**  The confining
potential is an *exponential wall* that is exactly zero up to an **onset**
radius :math:`r_0`, rises smoothly (all derivatives vanish at :math:`r_0`) and
diverges at :math:`r_0 + w`:

.. math::

    v_\text{cut}(r) = s\,\frac{\exp\!\big[-w/(r - r_0)\big]}{(r_0 + w - r)^2},
    \qquad r_0 < r < r_0 + w ,

with :math:`v_\text{cut} = 0` for :math:`r \le r_0` and :math:`+\infty` (capped)
beyond the wall.  Inside the onset an orbital is untouched; the tail is bent to
zero over the width :math:`w` without the kink a hard sphere introduces.  Every
function is therefore **strictly zero** beyond :math:`r_0 + w`.

**3. Radial and angular flexibility come from hydrogen-like tiers.**  Extra
functions are bound states of :math:`-z/r + v_\text{cut}(r)` -- hydrogen-like
orbitals with an *effective charge* :math:`z` that sets their size.  Rather than
tabulating :math:`z` per element, the value is derived from the atom: a
hydrogen-like :math:`(n, l)` state has mean radius
:math:`\langle r\rangle = [3n^2 - l(l+1)]/(2z)`, so :math:`z` is chosen to give
the function a prescribed extent relative to the atom's valence shell
(:func:`effective_charge_for_radius`).  The functions are organised in
**tiers** (:func:`tier_specification`):

* *tier 1* -- one **polarization** function at :math:`l_\max + 1` (nodeless,
  as compact as the valence shell) plus one **diffuse** function per occupied
  valence channel (one more node than the valence shell, twice its extent);
* *tier 2* -- a second polarization channel at :math:`l_\max + 2`, a noded
  polarization function at :math:`l_\max + 1`, and a **contracted** function
  per valence channel (same nodes as the valence shell, 0.6 of its extent).

Explicit ``(n, l, z)`` triples can be added on top (``extra=``) for full control.

Finally, all radial functions of one element are **orthonormalized within
each** :math:`l` **channel** by Gram-Schmidt, minimal functions first, and a
candidate whose norm after projection falls below ``linear_dependence_tol`` is
dropped as linearly dependent.  Tier functions are additionally shortened so
that at most ``tail_norm`` of their norm lies beyond the onset (the wall is
re-centred and the state re-solved).

The result goes through the same :class:`~carcara.basis.multizeta.TabulatedOrbital`
as the multiple-zeta NAOs, so the integral engine sees nothing new.  Lengths
are user-facing in Ångström (``onset`` / ``width``) and Bohr internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ..units import to_bohr
from ._config import ground_state_config, valence_subshells
from .atomic_solver import DEFAULT_POINTS, DEFAULT_R_MAX, solve_atom, solve_radial
from .multizeta import RadialTable

#: Default confinement onset / width (Ångström).  The wall sits at
#: ``onset + width``; keep the real-space box at least that large around
#: every atom.
DEFAULT_ONSET = 3.0
DEFAULT_WIDTH = 1.0
#: Wall strength ``s`` (Hartree * Bohr^2).
DEFAULT_SCALE = 1.0
#: Default tier (0 = minimal / all-electron atomic orbitals only).
DEFAULT_TIER = 1
#: Gram-Schmidt rejection threshold on the residual norm.
DEFAULT_LINEAR_DEPENDENCE_TOL = 1e-4
#: Tier functions are shortened until at most this norm lies beyond the onset.
DEFAULT_TAIL_NORM = 1e-4
#: Value the wall potential is capped at (Hartree) -- effectively infinite.
WALL_CAP = 1.0e5

_L_LETTERS = "spdfghik"


# --------------------------------------------------------------------------- #
# The confining wall.
# --------------------------------------------------------------------------- #

def confinement_potential(r, r_onset: float, width: float,
                          scale: float = DEFAULT_SCALE,
                          cap: float = WALL_CAP) -> np.ndarray:
    r"""Smooth exponential-wall confinement (Hartree; ``r`` in Bohr).

    Zero for ``r <= r_onset``, then
    :math:`s\,e^{-w/(r-r_0)}/(r_0+w-r)^2`, capped at ``cap`` and held there
    for ``r >= r_onset + width``.  Every derivative vanishes at the onset, so
    the confined orbital is smooth there, and the wall is impenetrable.
    """
    r = np.asarray(r, dtype=float)
    r0, w = float(r_onset), float(width)
    if w <= 0.0 or r0 < 0.0:
        raise ValueError("require width > 0 and r_onset >= 0")
    v = np.full(r.shape, float(cap))
    inside = r <= r0
    ramp = (r > r0) & (r < r0 + w)
    v[inside] = 0.0
    x = r[ramp]
    v[ramp] = np.minimum(
        float(scale) * np.exp(-w / (x - r0)) / (r0 + w - x) ** 2, float(cap))
    return v


# --------------------------------------------------------------------------- #
# Hydrogen-like sizing.
# --------------------------------------------------------------------------- #

def hydrogenic_mean_radius(n: int, l: int, z: float) -> float:
    r"""``<r>`` of a hydrogen-like state: :math:`[3n^2 - l(l+1)]/(2z)` (Bohr)."""
    if not (0 <= l < n) or z <= 0:
        raise ValueError("require 0 <= l < n and z > 0")
    return (3.0 * n * n - l * (l + 1)) / (2.0 * float(z))


def effective_charge_for_radius(n: int, l: int, target_radius: float) -> float:
    r"""The ``z`` giving a hydrogen-like ``(n, l)`` state mean radius ``target``.

    Inverts :func:`hydrogenic_mean_radius`.  This is how the tiers are sized
    from the atom instead of from a table.
    """
    if target_radius <= 0:
        raise ValueError("target_radius must be > 0")
    return (3.0 * n * n - l * (l + 1)) / (2.0 * float(target_radius))


def mean_radius(r: np.ndarray, u: np.ndarray) -> float:
    """``<r> = int u^2 r dr`` of a normalized ``u = r R``."""
    return float(np.trapezoid(u * u * r, r) / np.trapezoid(u * u, r))


# --------------------------------------------------------------------------- #
# Radial functions.
# --------------------------------------------------------------------------- #

@dataclass
class RadialFunction:
    """One radial function ``u(r) = r R(r)`` of an NAO-AE species."""

    r: np.ndarray                 # Bohr, strictly positive, up to the wall
    u: np.ndarray                 # normalized: int u^2 dr = 1
    n: int
    l: int
    kind: str                     # minimal | diffuse | polarization | contracted | custom
    energy: float = 0.0           # eigenvalue in its defining potential (Ha)
    z: float | None = None        # effective charge of a hydrogen-like function
    onset: float = 0.0            # confinement onset actually used (Bohr)
    details: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        tag = f"{self.n}{_L_LETTERS[self.l]}"
        if self.kind == "minimal":
            return f"{tag} (atomic)"
        z = "" if self.z is None else f", z={self.z:.2f}"
        return f"{tag} ({self.kind}{z})"

    def radial(self) -> np.ndarray:
        """``R(r) = u/r`` on the grid."""
        return self.u / self.r

    def outer_radius(self, threshold: float = 1e-6) -> float:
        """Largest ``r`` where ``|u|`` still exceeds ``threshold``."""
        idx = np.nonzero(np.abs(self.u) > threshold)[0]
        return float(self.r[idx[-1]]) if idx.size else 0.0


def _solve_on(r, potential, l, n_nodes, wall_radius):
    """Bound state of ``potential`` on the part of the grid inside the wall."""
    keep = r < wall_radius
    u, eps = solve_radial(r[keep], potential[keep], l, n_nodes)
    full = np.zeros_like(r)
    full[keep] = u
    return full, eps


def _tail_radius(r, u, tail_norm: float) -> float:
    """Radius beyond which ``u`` carries at most ``tail_norm`` of its norm."""
    from scipy.integrate import cumulative_trapezoid

    density = u * u
    total = float(np.trapezoid(density, r))
    inner = cumulative_trapezoid(density, r, initial=0.0)
    tail = 1.0 - inner / total
    idx = np.nonzero(tail < float(tail_norm))[0]
    return float(r[idx[0]]) if idx.size else float(r[-1])


def hydrogenic_function(r, n: int, l: int, z: float, onset: float,
                        width: float, scale: float = DEFAULT_SCALE,
                        kind: str = "custom",
                        tail_norm: float | None = DEFAULT_TAIL_NORM
                        ) -> RadialFunction:
    r"""Confined hydrogen-like state of ``-z/r + v_cut`` with ``n - l - 1`` nodes.

    With ``tail_norm`` set, the onset is pulled inward to the radius outside
    which the function carries ``tail_norm`` of its norm (if that is shorter
    than ``onset``) and the state is re-solved with the wall re-centred there,
    so compact functions stay compact instead of inheriting the species-wide
    range.
    """
    if not (0 <= l < n):
        raise ValueError(f"require 0 <= l < n, got n={n}, l={l}")
    r = np.asarray(r, dtype=float)
    coulomb = -float(z) / r
    r0 = float(onset)

    def solve(r_onset):
        v = coulomb + confinement_potential(r, r_onset, width, scale)
        return _solve_on(r, v, l, n - l - 1, r_onset + width)

    u, eps = solve(r0)
    if tail_norm:
        r_tail = _tail_radius(r, u, tail_norm)
        if r_tail < r0:
            r0 = r_tail
            u, eps = solve(r0)
    return RadialFunction(r=r, u=u, n=int(n), l=int(l), kind=kind,
                          energy=float(eps), z=float(z), onset=r0,
                          details={"width": float(width)})


# --------------------------------------------------------------------------- #
# The free atom and its minimal basis.
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=None)
def confined_atom(atomic_number: int, onset: float, width: float,
                  scale: float = DEFAULT_SCALE, points: int = DEFAULT_POINTS,
                  r_max: float = DEFAULT_R_MAX):
    """Self-consistent LDA atom whose orbitals feel the wall (cached).

    ``onset`` / ``width`` in Bohr.  The confinement enters only the orbital
    solves, so the returned effective potential is that of the confined
    density without the wall -- ready to have the wall added again as the
    basis-defining potential.
    """
    def wall(r):
        return confinement_potential(r, onset, width, scale)
    return solve_atom(int(atomic_number), points=points, r_max=r_max,
                      confinement=wall)


def minimal_functions(atom, onset: float, width: float,
                      scale: float = DEFAULT_SCALE) -> list[RadialFunction]:
    """Every occupied shell of ``atom`` re-solved in ``v_free + v_cut``.

    Core shells are included: this is the all-electron minimal basis.
    """
    r = atom.r
    v_basis = atom.v_effective + confinement_potential(r, onset, width, scale)
    out = []
    for (n, l) in sorted(atom.occupations):
        if atom.occupations[(n, l)] <= 0:
            continue
        u, eps = _solve_on(r, v_basis, l, n - l - 1, onset + width)
        out.append(RadialFunction(r=r, u=u, n=n, l=l, kind="minimal",
                                  energy=float(eps), onset=float(onset)))
    return out


# --------------------------------------------------------------------------- #
# Tiers.
# --------------------------------------------------------------------------- #

#: Extent of each tier function relative to the valence shell's ``<r>``.
TIER_RADIUS_RATIOS = {"polarization": 1.0, "diffuse": 2.0,
                      "contracted": 0.6, "noded-polarization": 1.5}
MAX_TIER = 2


def valence_radii(atom) -> dict[int, float]:
    """``{l: <r>}`` of the valence shells, plus ``-1`` -> the outermost one."""
    radii = {}
    for (n, l) in valence_subshells(atom.atomic_number):
        u = atom.orbitals.get((n, l))
        if u is not None:
            radii[l] = mean_radius(atom.r, u)
    radii[-1] = max(radii.values())
    return radii


def tier_specification(atom, tier: int) -> list[tuple[int, int, float, str]]:
    r"""``[(n, l, z, kind), ...]`` of the hydrogen-like functions up to ``tier``.

    Sizes come from the atom (:func:`valence_radii`); see the module docstring
    for the rule.  ``tier=0`` is the minimal basis alone.
    """
    tier = int(tier)
    if tier < 0 or tier > MAX_TIER:
        raise ValueError(f"tier must be in [0, {MAX_TIER}], got {tier}")
    if tier == 0:
        return []
    valence = valence_subshells(atom.atomic_number)
    n_of_l = {l: n for (n, l) in valence}
    l_max = max(n_of_l)
    radii = valence_radii(atom)
    spec: list[tuple[int, int, float, str]] = []

    def add(n, l, ratio_key, l_ref):
        target = TIER_RADIUS_RATIOS[ratio_key] * radii.get(l_ref, radii[-1])
        spec.append((n, l, effective_charge_for_radius(n, l, target),
                     ratio_key if ratio_key != "noded-polarization"
                     else "polarization"))

    # Tier 1: polarization at l_max + 1, diffuse in every valence channel.
    l_pol = l_max + 1
    add(l_pol + 1, l_pol, "polarization", -1)
    for l in sorted(n_of_l):
        add(n_of_l[l] + 1, l, "diffuse", l)
    if tier == 1:
        return spec
    # Tier 2: second polarization channel, a noded polarization function,
    # and a contracted function per valence channel.
    add(l_pol + 2, l_pol + 1, "polarization", -1)
    add(l_pol + 2, l_pol, "noded-polarization", -1)
    for l in sorted(n_of_l):
        add(n_of_l[l], l, "contracted", l)
    return spec


# --------------------------------------------------------------------------- #
# Orthonormalization.
# --------------------------------------------------------------------------- #

def orthonormalize(functions: list[RadialFunction],
                   tol: float = DEFAULT_LINEAR_DEPENDENCE_TOL
                   ) -> list[RadialFunction]:
    """Gram-Schmidt within each ``l`` channel, in list order.

    A candidate whose residual norm after projecting out the accepted
    functions of its channel is below ``tol`` is dropped as linearly
    dependent.  A minimal function can never be dropped: that would mean the
    atom's own orbital is missing, which is an error.
    """
    accepted: list[RadialFunction] = []
    for f in functions:
        r = f.r
        u = f.u.copy()
        same = [g for g in accepted if g.l == f.l and g.r.shape == r.shape]
        for _pass in range(3):
            for g in same:
                u -= float(np.trapezoid(g.u * u, r)) * g.u
            norm2 = float(np.trapezoid(u * u, r))
            if _pass == 0 and norm2 < tol:
                if f.kind == "minimal":
                    raise RuntimeError(
                        f"minimal function {f.label} is linearly dependent")
                u = None
                break
            u /= np.sqrt(norm2)
        if u is None:
            continue
        if u[np.argmax(np.abs(u))] < 0:
            u = -u
        accepted.append(RadialFunction(r=r, u=u, n=f.n, l=f.l, kind=f.kind,
                                       energy=f.energy, z=f.z, onset=f.onset,
                                       details=dict(f.details)))
    return accepted


# --------------------------------------------------------------------------- #
# Building a species.
# --------------------------------------------------------------------------- #

def build_species(atomic_number: int, *, tier: int = DEFAULT_TIER,
                  onset: float = DEFAULT_ONSET, width: float = DEFAULT_WIDTH,
                  scale: float = DEFAULT_SCALE, extra=None,
                  linear_dependence_tol: float = DEFAULT_LINEAR_DEPENDENCE_TOL,
                  tail_norm: float | None = DEFAULT_TAIL_NORM,
                  points: int = DEFAULT_POINTS, r_max: float = DEFAULT_R_MAX,
                  units: str = "angstrom") -> list[RadialFunction]:
    """All orthonormalized radial functions of one element.

    ``onset`` / ``width`` are in ``units`` (Ångström by default); ``extra`` is
    a sequence of explicit hydrogen-like ``(n, l, z)`` triples appended after
    the tiers.  Returns the accepted :class:`RadialFunction` list, minimal
    functions first, each channel orthonormal.
    """
    r0 = float(to_bohr(onset, units))
    w = float(to_bohr(width, units))
    if r0 + w > r_max:
        raise ValueError(
            f"the wall ({r0 + w:.2f} Bohr) lies beyond the radial grid "
            f"(r_max={r_max} Bohr); shorten onset/width or raise r_max")
    atom = confined_atom(int(atomic_number), r0, w, float(scale), int(points),
                         float(r_max))
    functions = minimal_functions(atom, r0, w, scale)
    for (n, l, z, kind) in tier_specification(atom, tier):
        functions.append(hydrogenic_function(atom.r, n, l, z, r0, w, scale,
                                             kind=kind, tail_norm=tail_norm))
    for item in (extra or ()):
        n, l, z = item
        functions.append(hydrogenic_function(atom.r, int(n), int(l), float(z),
                                             r0, w, scale, kind="custom",
                                             tail_norm=tail_norm))
    # Compact functions first within each channel: better-conditioned
    # Gram-Schmidt, and the minimal functions always lead.
    minimal = [f for f in functions if f.kind == "minimal"]
    others = sorted((f for f in functions if f.kind != "minimal"),
                    key=lambda f: f.outer_radius())
    return orthonormalize(minimal + others, linear_dependence_tol)


def radial_tables(functions: list[RadialFunction],
                  wall_radius: float) -> list[RadialTable]:
    """Convert to :class:`~carcara.basis.multizeta.RadialTable` (``R = u/r``).

    The table runs from ``r = 0`` (extrapolated) to the wall, where the
    function is exactly zero, so the resulting
    :class:`~carcara.basis.multizeta.TabulatedOrbital` has ``r_c`` at the wall.
    Zeta indices count the functions of each ``l`` channel; ``polarization``
    flags channels above the highest occupied ``l``.
    """
    tables: list[RadialTable] = []
    l_occupied = {f.l for f in functions if f.kind == "minimal"}
    zeta_of_l: dict[int, int] = {}
    for f in functions:
        keep = f.r < wall_radius
        r = f.r[keep]
        R = f.u[keep] / r
        # R(0): finite for l = 0 (linear extrapolation), zero otherwise.
        R0 = R[0] - (R[1] - R[0]) / (r[1] - r[0]) * r[0] if f.l == 0 else 0.0
        r_full = np.concatenate(([0.0], r, [wall_radius]))
        R_full = np.concatenate(([R0], R, [0.0]))
        zeta_of_l[f.l] = zeta_of_l.get(f.l, 0) + 1
        tables.append(RadialTable(r=r_full, values=R_full, n=f.n, l=f.l,
                                  zeta=zeta_of_l[f.l],
                                  polarization=f.l not in l_occupied))
    return tables


def describe_species(functions: list[RadialFunction]) -> str:
    """One line per radial function: label, energy, extent."""
    lines = []
    for f in functions:
        lines.append(f"  {f.label:<28s} eps = {f.energy:+9.4f} Ha   "
                     f"<r> = {mean_radius(f.r, f.u):5.2f} a0   "
                     f"r_out = {f.outer_radius():5.2f} a0   "
                     f"({2 * f.l + 1} functions)")
    return "\n".join(lines)
