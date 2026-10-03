# -*- coding: utf-8 -*-
# file: src/mandacaru/pseudopotentials/confinement.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Confined pseudo-atomic orbitals: the ``energy_shift`` of a pseudopotential
basis (PAW-LCAO, UPAW-LCAO and ONCVPSP).

Without this module the first zeta of such a basis is the dataset's bound smooth
partial wave -- the valence orbital of the **free** atom, which has no range of
its own (a lithium 2s still carries 1e-4 of its norm beyond 14 Bohr).  LCAO
codes instead commonly use the orbital of the atom inside a confining
potential, and name the confinement by what it costs: the ``energy_shift`` is
how far the confined eigenvalue lies above the free one, and it *defines* the
cutoff radius :math:`r_c` of each orbital (Sankey and Niklewski 1989; Artacho
*et al.* 1999).  One number then gives every channel of every element a radius
that is tight for a compact orbital and generous for a diffuse one.

The operator is untouched.  The projectors, the couplings ``D``, the overlap
``q`` and the local potential all come from the dataset; only the *trial
function* changes: the atom is confined, the dataset is kept.  The confined
orbital is the lowest solution of the same generalized radial problem the
dataset's bound wave solves,

.. math::

    \Bigl(T_l + \tilde v^{scr} + v_{conf}
          + \sum_{ij}|\tilde p_i\rangle D^{scr}_{ij}\langle\tilde p_j|\Bigr)u
    = \varepsilon\Bigl(1 + \sum_{ij}|\tilde p_i\rangle q_{ij}
                       \langle\tilde p_j|\Bigr)u ,
    \qquad u(r_c) = 0 ,

with the smooth confining potential of Junquera *et al.* (PRB 64, 235111),

.. math::

    v_{conf}(r) = \frac{A}{r_c - r}\,
                  \exp\!\Bigl(-\frac{r_c - r_i}{r - r_i}\Bigr)
    \quad (r_i < r < r_c), \qquad 0 \ \ (r \le r_i),

which is flat (all derivatives zero) at the inner radius :math:`r_i` and
diverges at :math:`r_c`, so the orbital and every derivative of it vanish at
the cutoff -- a hard wall leaves a kink there, which a real-space grid pays
for.  The amplitude :math:`A` = 12 Ha and :math:`r_i = 0.6\,r_c` are a common
default for this potential, chosen on purpose: with the same ``energy_shift``
and the same ``size``, a PAW-LCAO basis built here follows the same recipe as
the LCAO bases it is compared with.  (The *datasets* still differ -- each code
pseudizes its own atom -- so the radii agree closely, not identically.)

For the norm-conserving family the overlap is the identity (``q = 0``):
ONCVPSP solves with its own local potential and projectors.  Each dataset supplies its operator through
``channel_operator_on(l, r)``, which returns the grid, the potential, the
``u``-form projectors, ``D`` and ``q``.

``energy_shift`` is in **eV**, like the ``energy_shift`` of the
all-electron ``"NAO"`` family.  It is on by default at
:data:`DEFAULT_ENERGY_SHIFT` for every family that offers it; ``None`` /
``False`` / ``0`` keep the free-atom orbital.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..units import EV_TO_HARTREE, HARTREE_TO_EV

#: Amplitude ``A`` (Hartree) of the confining potential.
CONFINEMENT_AMPLITUDE = 12.0

#: Inner radius of the confining potential as a fraction of ``r_c``.  Below
#: it the atom is untouched.
CONFINEMENT_INNER_FRACTION = 0.6

#: The ``energy_shift`` (eV) a pseudopotential basis uses when the basis dict
#: does not say.  ``"energy_shift": None``
#: restores the unconfined free-atom orbital.
DEFAULT_ENERGY_SHIFT = 0.1

#: The basis options this module provides, shared by every family whose
#: datasets define ``channel_operator_on`` -- PAW-LCAO, UPAW-LCAO and
#: ONCVPSP -- and their defaults (the polarization shell is derived from the
#: confinement, :func:`resolve_polarization`, so it has none of its own).
CONFINEMENT_OPTIONS = ("energy_shift", "confinement", "polarization")
CONFINEMENT_DEFAULT_OPTIONS = {"energy_shift": DEFAULT_ENERGY_SHIFT}

#: Which polarization shell a pseudopotential basis builds (``polarization=``).
#: ``"gaussian"``: the quasi-Gaussian of :func:`gaussian_polarization` -- the
#: default **wherever the orbital is confined**, since the Gaussian takes its
#: cutoff from the confined orbital.  ``"orbital"``: :math:`r^k R_{outer}(r)`,
#: split off the outermost channel -- Mandacaru's original, and what an
#: unconfined element falls back to when the option is left unwritten.
POLARIZATION_KINDS = ("gaussian", "orbital")

#: The polarization Gaussian is sized from the cutoff the base orbital has at
#: a **fixed** 0.3 eV shift -- whatever ``energy_shift`` the basis itself
#: uses -- confined with the default potential: ``r_char = 0.25 * r_c(0.3 eV)``.
POLARIZATION_REFERENCE_SHIFT = 0.3
POLARIZATION_CHARACTER_FRACTION = 0.25

#: Smallest ``energy_shift`` (eV) accepted.  Below it the cutoff radius runs
#: past the end of the dataset's radial table, and the request is
#: indistinguishable from "no confinement", which has its own spelling.
MIN_ENERGY_SHIFT = 1e-3

#: Target spacing (Bohr) of the finer of the two radial grids the confined
#: problem is solved on; the coarser is twice it, and the two are
#: Richardson-extrapolated.  The actual spacings are ``r_c / n`` so that the
#: wall sits exactly on a node -- otherwise the eigenvalue is a staircase in
#: ``r_c`` and the root search below has nothing smooth to converge on.
RADIAL_SPACING = 0.01

#: Growth factor of the outward bracketing scan, and the tolerance (Bohr) of
#: the root search on ``r_c``.
BRACKET_GROWTH = 1.25
RADIUS_TOLERANCE = 1e-4

#: Fraction of the radial table the cutoff may reach; beyond it the dataset's
#: potential is no longer tabulated.
TABLE_FRACTION = 0.95


# --------------------------------------------------------------------------- #
# The option.
# --------------------------------------------------------------------------- #

def _validate_value(value):
    if value is None or value is False:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"energy_shift must be a number of eV (or None / False for the "
            f"free-atom orbital), got {value!r}")
    value = float(value)
    if value == 0.0:
        return None
    if not np.isfinite(value) or value < MIN_ENERGY_SHIFT:
        raise ValueError(
            f"energy_shift must be at least {MIN_ENERGY_SHIFT} eV, got "
            f"{value!r}; use None for the unconfined free-atom orbital")
    return value


def validate_energy_shift(spec):
    """Normalize an ``energy_shift`` option.

    ``None`` / ``False`` / ``0`` mean *unconfined* and return ``None``; a
    positive number of **eV** is returned as a float; a ``{symbol: eV}``
    mapping (``"*"`` = default) gives each element its own and returns the
    normalized mapping (or ``None`` when every entry is off).
    """
    if isinstance(spec, dict):
        table = {(k if k == "*" else str(k).capitalize()): _validate_value(v)
                 for k, v in spec.items()}
        return table if any(v is not None for v in table.values()) else None
    return _validate_value(spec)


def validate_polarization(kind):
    """Normalize the ``polarization`` option to one of
    :data:`POLARIZATION_KINDS`, or ``None`` when it was not written (the
    default then depends on the confinement: :func:`resolve_polarization`)."""
    if kind is None:
        return None
    key = str(kind).strip().lower().replace("-", "_")
    key = {"quasi_gaussian": "gaussian"}.get(key, key)
    if key not in POLARIZATION_KINDS:
        raise ValueError(
            f"polarization must be one of {list(POLARIZATION_KINDS)}, got "
            f"{kind!r}")
    return key


def resolve_polarization(kind, energy_shift, symbol=None) -> str:
    """The polarization shell actually built.

    Written explicitly, it is what was written.  Left unwritten, it is
    ``"gaussian"`` where the orbital is confined and ``"orbital"`` where it is
    not -- for ``symbol`` when one is given, else for the basis as a whole
    (Gaussian as soon as any element is confined).
    """
    kind = validate_polarization(kind)
    if kind is not None:
        return kind
    spec = validate_energy_shift(energy_shift)
    confined = (energy_shift_of(spec, symbol) is not None
                if symbol is not None else spec is not None)
    return "gaussian" if confined else "orbital"


def validate_confinement(spec) -> tuple[float, float]:
    """Normalize the ``confinement`` option, ``(amplitude_Ha, r_i / r_c)``.

    ``None`` is the default, ``(12.0, 0.6)``.
    """
    if spec is None:
        return CONFINEMENT_AMPLITUDE, CONFINEMENT_INNER_FRACTION
    try:
        amplitude, inner = (float(v) for v in spec)
    except (TypeError, ValueError):
        raise ValueError(
            "confinement must be a pair (amplitude in Hartree, r_i / r_c), "
            f"got {spec!r}") from None
    if not (np.isfinite(amplitude) and amplitude > 0.0):
        raise ValueError(f"the confinement amplitude must be positive, got "
                         f"{amplitude!r}")
    if not 0.0 < inner < 1.0:
        raise ValueError("the confinement inner radius is a fraction of r_c in "
                         f"(0, 1), got {inner!r}")
    return amplitude, inner


def energy_shift_of(spec, symbol):
    """The ``energy_shift`` (eV, or ``None``) of one element under ``spec``."""
    spec = validate_energy_shift(spec)
    if isinstance(spec, dict):
        return spec.get(str(symbol).capitalize(), spec.get("*"))
    return spec


def energy_shift_label(spec) -> str:
    """How a basis label names the confinement: always, on or off."""
    spec = validate_energy_shift(spec)
    if spec is None:
        return "unconfined"
    if isinstance(spec, dict):
        inner = ", ".join(f"{k}: {'off' if v is None else format(v, 'g')}"
                          for k, v in sorted(spec.items()))
        return f"energy_shift {{{inner}}} eV"
    return f"energy_shift {spec:g} eV"


# --------------------------------------------------------------------------- #
# The confined radial problem.
# --------------------------------------------------------------------------- #

def confinement_potential(r, r_c: float,
                          amplitude: float = CONFINEMENT_AMPLITUDE,
                          inner_fraction: float = CONFINEMENT_INNER_FRACTION
                          ) -> np.ndarray:
    """Junquera's smooth confining potential (Hartree) on ``r`` (Bohr).

    Zero up to ``r_i = inner_fraction * r_c`` and ``+inf`` from ``r_c`` on; the
    solver never samples the wall itself (its grid ends one node short).
    """
    r = np.asarray(r, dtype=float)
    r_c = float(r_c)
    r_i = float(inner_fraction) * r_c
    out = np.zeros_like(r)
    inside = (r > r_i) & (r < r_c)
    x = r[inside]
    out[inside] = (float(amplitude) / (r_c - x)
                   * np.exp(-(r_c - r_i) / (x - r_i)))
    out[r >= r_c] = np.inf
    return out


def _grid(r_c: float, doubling: int) -> np.ndarray:
    """Interior nodes of a uniform grid whose node ``n`` is the wall."""
    half = max(8, int(round(r_c / (2.0 * RADIAL_SPACING))))
    n = half * (2 if doubling == 0 else 1)             # fine, then coarse
    return (r_c / n) * np.arange(1, n)


def _lowest_eigenvalue(r, v, p_u, D, q) -> float:
    r"""Lowest eigenvalue of the pencil of
    :func:`~.paw._generalized_matrices`, without forming it.

    ``H = T + U D U^T`` and ``S = I + U q U^T``, with ``T`` tridiagonal and
    ``U = \sqrt h\,X^T`` one column per projector.  The number of
    eigenvalues below :math:`\sigma` is the inertia of
    :math:`H - \sigma S` (``S`` is positive definite), which Haynsworth's
    formula splits into that of :math:`T - \sigma` -- from ``T``'s own
    eigenvalues -- and of a matrix the size of the projector count; the
    count is bisected to the eigenvalue.  A dense generalized solve of the
    free level (the wall at the table's end, 2849 nodes) cost 1.4 s per
    channel and was most of a crystal's setup; this costs milliseconds and
    agrees to round-off.
    """
    from scipy.linalg import eigh, eigvalsh_tridiagonal, solve_banded

    h = float(r[1] - r[0])
    n = r.size
    diagonal = 1.0 / h ** 2 + np.asarray(v, dtype=float)
    off = np.full(n - 1, -0.5 / h ** 2)
    levels = eigvalsh_tridiagonal(diagonal, off)
    if not p_u:
        return float(levels[0])
    U = np.sqrt(h) * np.array(p_u).T
    D = np.asarray(D, dtype=float)
    q = np.asarray(q, dtype=float)
    band = np.zeros((3, n))
    band[0, 1:] = off
    band[2, :-1] = off

    def below(sigma):
        """How many eigenvalues of the pencil lie below ``sigma``."""
        beta, V = eigh(D - sigma * q)
        keep = np.abs(beta) > 1e-14 * max(1.0, float(np.max(np.abs(beta))))
        count = int(np.searchsorted(levels, sigma))
        if not np.any(keep):
            return count
        W = U @ V[:, keep]
        band[1] = diagonal - sigma
        solved = solve_banded((1, 1), band, W)
        inner = -np.diag(1.0 / beta[keep]) - W.T @ solved
        count += int(np.sum(np.linalg.eigvalsh(inner) < 0.0))
        count -= int(np.sum(beta[keep] > 0.0))        # -1/beta < 0
        return count

    scale = max(1.0, abs(float(levels[0])))
    lo = float(levels[0]) - scale
    while below(lo) > 0:
        lo -= 2.0 * (abs(lo) + 1.0)
    hi = float(levels[0])
    while below(hi) < 1:
        hi += 2.0 * (abs(hi) + 1.0)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid in (lo, hi):
            break
        if below(mid) >= 1:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _solve(pp, l: int, r_c: float, amplitude: float, inner_fraction: float,
           vectors: bool):
    """Lowest generalized eigenpair on the fine and the coarse grid."""
    from scipy.linalg import eigh

    from .paw import _generalized_matrices

    out = []
    for doubling in (0, 1):
        r = _grid(r_c, doubling)
        r, v, p_u, D, q = pp.channel_operator_on(l, r)
        v = v + confinement_potential(r, r_c, amplitude, inner_fraction)
        if not vectors:
            out.append((r, None, None, _lowest_eigenvalue(r, v, p_u, D, q)))
            continue
        H, S = _generalized_matrices(r, v, p_u, D, q)
        values, states = eigh(H, S, subset_by_index=[0, 0])
        out.append((r, states[:, 0], S, float(values[0])))
    return out


def confined_energy(pp, l: int, r_c: float, *,
                    amplitude: float = CONFINEMENT_AMPLITUDE,
                    inner_fraction: float = CONFINEMENT_INNER_FRACTION
                    ) -> float:
    """Lowest eigenvalue (Hartree) of channel ``l`` confined at ``r_c``,
    Richardson-extrapolated from the two grids."""
    (_r, _u, _S, fine), (_r2, _u2, _S2, coarse) = _solve(
        pp, int(l), float(r_c), amplitude, inner_fraction, vectors=False)
    return (4.0 * fine - coarse) / 3.0


def confined_eigenstate(pp, l: int, r_c: float, *,
                        amplitude: float = CONFINEMENT_AMPLITUDE,
                        inner_fraction: float = CONFINEMENT_INNER_FRACTION):
    """``(r, u, energy)`` of the confined channel: ``u = r R`` on the coarse
    grid (wall excluded), Richardson-extrapolated pointwise, normalized in the
    dataset's own metric (``int u S u = 1``, the normalization of the stored
    bound wave) and signed like it."""
    from scipy.interpolate import CubicSpline

    l = int(l)
    bound = CubicSpline(pp.r, pp.channels[l].pseudo_radial)
    solved = []
    for r, u, S, energy in _solve(pp, l, float(r_c), amplitude,
                                  inner_fraction, vectors=True):
        h = r[1] - r[0]
        u = u / np.sqrt(h * u @ S @ u)
        if np.trapezoid(u * bound(r) * r, dx=h) < 0:
            u = -u
        solved.append((r, u, energy))
    (_r_f, u_f, e_f), (r_c_grid, u_c, e_c) = solved
    return r_c_grid, (4.0 * u_f[1::2] - u_c) / 3.0, (4.0 * e_f - e_c) / 3.0


#: ``(id(dataset), l) -> (dataset, free level)``, as for :data:`_CACHE`.
_FREE_CACHE: dict = {}


def free_energy(pp, l: int) -> float:
    """Eigenvalue (Hartree) of channel ``l``'s free bound state **in the
    dataset's own operator**, solved exactly as a confined one is but with the
    wall at the end of the radial table.

    Not the all-electron reference level: a dataset reproduces it to 1e-7 Ha
    for the first rows but only to 1-6 mHa for a deep semicore channel (gold's
    4f: -3.1036 against -3.1075 Ha), which is more than a 0.1 eV shift -- the
    confinement radius was then measured from the wrong level and could not be
    found.  Solving both levels with one discretization also cancels its
    error from the shift.
    """
    l = int(l)
    entry = _FREE_CACHE.get((id(pp), l))
    if entry is not None and entry[0] is pp:
        return entry[1]
    energy = confined_energy(pp, l, TABLE_FRACTION * float(pp.r[-1]))
    _FREE_CACHE[(id(pp), l)] = (pp, energy)
    return energy


def confinement_radius(pp, l: int, energy_shift: float, *,
                       amplitude: float = CONFINEMENT_AMPLITUDE,
                       inner_fraction: float = CONFINEMENT_INNER_FRACTION,
                       tightest: bool = False) -> float:
    """Cutoff radius ``r_c`` (Bohr) at which channel ``l``'s eigenvalue lies
    ``energy_shift`` (**eV**) above the free atom's.

    The eigenvalue falls monotonically as the wall recedes, so the radius is
    bracketed by scanning outward from the channel's cutoff radius and then
    refined with Brent's method.  A shift too large to reach without cutting
    into the sphere, or too small to reach inside the radial table, is
    refused with the range that *is* reachable -- unless ``tightest``, when a
    shift too large returns the tightest wall the sphere allows instead.  That
    is the case of a deep semicore channel (the 4f of Lu-Hg, 1-3.5 Ha deep in
    a 2.9-3.9 Bohr sphere): its orbital lies inside the sphere, so no wall
    outside it can move its energy by 0.1 eV, and it needs no confining.
    """
    from scipy.optimize import brentq

    l = int(l)
    shift = _validate_value(energy_shift)
    if shift is None:
        raise ValueError("confinement_radius needs a positive energy_shift")
    target = free_energy(pp, l) + shift * EV_TO_HARTREE

    def excess(r_c):
        return confined_energy(pp, l, r_c, amplitude=amplitude,
                               inner_fraction=inner_fraction) - target

    r_lo = float(pp.channels[l].r_cut) / float(inner_fraction)
    r_end = TABLE_FRACTION * float(pp.r[-1])
    f_lo = excess(r_lo)
    if f_lo <= 0.0 and tightest:
        return float(r_lo)
    if f_lo <= 0.0:
        largest = (f_lo + shift * EV_TO_HARTREE) * HARTREE_TO_EV
        raise ValueError(
            f"energy_shift = {shift:g} eV is too large for the {pp.symbol} "
            f"l = {l} channel: it would put the confinement inside the "
            f"channel's cutoff radius (r_cut = {pp.channels[l].r_cut:.2f} "
            f"Bohr); "
            f"the largest usable value is about {largest:.2f} eV")
    r_hi = r_lo
    while True:
        r_hi = min(BRACKET_GROWTH * r_hi, r_end)
        if excess(r_hi) < 0.0:
            break
        if r_hi >= r_end:
            raise ValueError(
                f"energy_shift = {shift:g} eV is too small for the "
                f"{pp.symbol} l = {l} channel: its cutoff radius lies beyond "
                f"the dataset's radial table ({r_end:.1f} Bohr); use None for "
                "the unconfined orbital")
        r_lo = r_hi
    return float(brentq(excess, r_lo, r_hi, xtol=RADIUS_TOLERANCE))


# --------------------------------------------------------------------------- #
# The orbital a basis uses.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ConfinedOrbital:
    """A confined first zeta, tabulated on the dataset's own radial grid."""

    symbol: str
    l: int
    energy_shift: float          # eV, as requested
    r_c: float                   # Bohr
    energy: float                # Hartree, confined eigenvalue
    free_energy: float           # Hartree, free-atom eigenvalue
    radial: np.ndarray           # R(r) on pp.r, zero from r_c on

    @property
    def achieved_shift(self) -> float:
        """The realized shift (eV); equals :attr:`energy_shift` to the root
        search's tolerance."""
        return (self.energy - self.free_energy) * HARTREE_TO_EV


#: ``(id(dataset), l, shift) -> (dataset, ConfinedOrbital)``.  The dataset is
#: kept in the entry so an ``id`` recycled after a garbage collection cannot
#: serve another dataset's orbital.
_CACHE: dict = {}


def confined_orbital(pp, l: int, energy_shift: float,
                     confinement=None) -> ConfinedOrbital:
    """The confined orbital of channel ``l`` for ``energy_shift`` (eV), cached
    per dataset -- the root search costs a second or two per channel and a
    relaxation would otherwise repeat it at every geometry.  ``confinement`` is
    the ``(amplitude, r_i / r_c)`` pair (``None`` = the default)."""
    from scipy.interpolate import CubicSpline

    l = int(l)
    shift = _validate_value(energy_shift)
    if shift is None:
        raise ValueError("confined_orbital needs a positive energy_shift")
    amplitude, inner = validate_confinement(confinement)
    key = (id(pp), l, round(shift, 12), amplitude, inner)
    entry = _CACHE.get(key)
    if entry is not None and entry[0] is pp:
        return entry[1]

    # A channel whose orbital already lies inside its sphere (a deep semicore
    # shell) takes the tightest wall; `achieved_shift` then reports the
    # smaller shift it actually got.
    r_c = confinement_radius(pp, l, shift, amplitude=amplitude,
                             inner_fraction=inner, tightest=True)
    r, u, energy = confined_eigenstate(pp, l, r_c, amplitude=amplitude,
                                       inner_fraction=inner)
    # u(0) = u(r_c) = 0 close the table; R = u / r is then regular at the
    # origin (u ~ r^(l+1)) and the dataset's first node is never exactly 0.
    spline = CubicSpline(np.concatenate([[0.0], r, [r_c]]),
                         np.concatenate([[0.0], u, [0.0]]))
    table = np.asarray(pp.r, dtype=float)
    radial = np.where(table < r_c, spline(np.minimum(table, r_c)) / table, 0.0)
    orbital = ConfinedOrbital(symbol=str(pp.symbol), l=l, energy_shift=shift,
                              r_c=float(r_c), energy=float(energy),
                              free_energy=free_energy(pp, l), radial=radial)
    _CACHE[key] = (pp, orbital)
    return orbital


def first_zeta_factory(energy_shift, record: dict | None = None,
                       confinement=None):
    """The ``first_zeta`` hook of :func:`~.orbitals.pseudo_basis` for an
    ``energy_shift`` option, or ``None`` when the option is off.

    ``record`` (optional) is filled with ``{symbol: {l: ConfinedOrbital}}`` so
    a builder can report the radii it used.
    """
    spec = validate_energy_shift(energy_shift)
    if spec is None:
        return None
    confinement = validate_confinement(confinement)

    def first_zeta(symbol, pp, l):
        shift = energy_shift_of(spec, symbol)
        if shift is None:
            return None
        orbital = confined_orbital(pp, l, shift, confinement)
        if record is not None:
            record.setdefault(symbol, {})[int(l)] = orbital
        return orbital.radial

    return first_zeta


# --------------------------------------------------------------------------- #
# The quasi-Gaussian polarization function.
# --------------------------------------------------------------------------- #

def quasi_gaussian(r, alpha: float, r_cut: float) -> np.ndarray:
    r"""The quasi-Gaussian :math:`e^{-\alpha r^2} - (a - b r^2)` inside
    ``r_cut`` and zero beyond, with :math:`a = (1 + \alpha r_c^2)
    e^{-\alpha r_c^2}`, :math:`b = \alpha e^{-\alpha r_c^2}` so that the value
    *and* the slope vanish at the cutoff."""
    r = np.asarray(r, dtype=float)
    decay = np.exp(-float(alpha) * float(r_cut) ** 2)
    a = (1.0 + alpha * r_cut ** 2) * decay
    b = alpha * decay
    inside = r < r_cut
    r2 = np.where(inside, r * r, 0.0)
    return np.where(inside, np.exp(-alpha * r2) - (a - b * r2), 0.0)


def polarization_channel(channels) -> int:
    """Angular momentum of the polarization shell: the first
    ``l`` **missing** among the valence channels, else ``l_max + 1`` (so a
    4s/3d transition metal is polarized with a p shell, not an f shell)."""
    present = {int(l) for l in channels}
    for l in range(max(present) + 1):
        if l not in present:
            return l
    return max(present) + 1


@dataclass(frozen=True)
class GaussianPolarization:
    """A quasi-Gaussian polarization shell on the dataset's radial grid."""

    symbol: str
    l: int                       # of the polarization shell
    base_l: int                  # the valence channel it is sized from
    r_cut: float                 # Bohr: the base orbital's cutoff
    r_char: float                # Bohr: 1 / sqrt(alpha)
    reference_radius: float      # Bohr: the base orbital's cutoff at 0.3 eV
    radial: np.ndarray           # R(r) = r^l f(r), unit norm


def gaussian_polarization(pp, energy_shift: float, confinement=None
                          ) -> GaussianPolarization:
    r"""The quasi-Gaussian polarization function for the dataset ``pp``.

    The shell has the angular momentum of
    :func:`polarization_channel`; it is based on the valence channel one below,
    whose cutoff (at the basis's own ``energy_shift``) becomes the function's
    ``r_cut``; its width is ``r_char = 0.25 * r_c(0.3 eV)`` with that reference
    cutoff always taken at 0.3 eV in the default potential; and
    :math:`R(r) = r^l\,[e^{-r^2/r_{char}^2} - (a - b r^2)]`, normalized.
    """
    l_pol = polarization_channel(pp.channels)
    base = l_pol - 1
    if base not in pp.channels:
        raise ValueError(
            f"a Gaussian polarization shell with l = {l_pol} for "
            f"{pp.symbol} needs an l = {base} valence channel to size it from")
    r_cut = confined_orbital(pp, base, energy_shift, confinement).r_c
    reference = confined_orbital(pp, base, POLARIZATION_REFERENCE_SHIFT).r_c
    r_char = POLARIZATION_CHARACTER_FRACTION * reference
    table = np.asarray(pp.r, dtype=float)
    radial = table ** l_pol * quasi_gaussian(table, 1.0 / r_char ** 2, r_cut)
    radial = radial / np.sqrt(np.trapezoid(radial * radial * table * table,
                                           table))
    return GaussianPolarization(symbol=str(pp.symbol), l=l_pol, base_l=base,
                                r_cut=float(r_cut), r_char=float(r_char),
                                reference_radius=float(reference),
                                radial=radial)


def polarization_factory(kind, energy_shift, record: dict | None = None,
                         confinement=None):
    """The ``polarization_shape`` hook of :func:`~.orbitals.pseudo_basis`, or
    ``None`` for the ``"orbital"`` shell that function builds itself.

    The Gaussian shell is sized from the confined orbital's cutoff, so it
    exists only for a confined orbital.  *Asking* for it without an
    ``energy_shift`` is refused rather than given an invented radius; when the
    option was left unwritten, an unconfined element quietly keeps the
    ``"orbital"`` shell instead (the hook returns ``None`` for it).
    """
    if resolve_polarization(kind, energy_shift) != "gaussian":
        return None
    explicit = validate_polarization(kind) is not None
    spec = validate_energy_shift(energy_shift)
    confinement = validate_confinement(confinement)

    def polarization_shape(symbol, pp):
        shift = energy_shift_of(spec, symbol)
        if shift is None and not explicit:
            return None
        if shift is None:
            raise ValueError(
                f"polarization='gaussian' needs an energy_shift for "
                f"{symbol!r}: the Gaussian shell takes its cutoff from the "
                "confined orbital, and an unconfined orbital has none")
        shell = gaussian_polarization(pp, shift, confinement)
        if record is not None:
            record[symbol] = shell
        return shell.l, shell.radial

    return polarization_shape
