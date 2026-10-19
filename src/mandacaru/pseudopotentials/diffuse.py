# -*- coding: utf-8 -*-
# file: src/mandacaru/pseudopotentials/diffuse.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Diffuse functions for a PAW-LCAO or UPAW-LCAO basis: the ``diffuse`` option.

A pseudopotential basis is built from the atom's own orbitals, and no
setting of the confinement makes them diffuse: at any ``energy_shift`` the
first zeta is at most the free atom's orbital (hydrogen's 1s keeps an rms
radius of 1.97 Bohr at 0.001 eV and unconfined alike).  A response
property (the polarizability), a bound anion or a weak complex needs the
density further out than that.  This module **adds** functions there; the
ordinary basis (``size``, ``energy_shift``, polarization) is unchanged.

Each diffuse shell is the quasi-Gaussian of the polarization function
(:func:`~.confinement.quasi_gaussian`),

.. math::

    R(r) = N\,r^l\,\bigl[e^{-\alpha r^2} - (a - b r^2)\bigr]
    \quad (r < r_{cut}), \qquad 0 \ \ (r \ge r_{cut}),

whose value and slope vanish at :math:`r_{cut}`.  Two numbers fix it:

* the **width**, as an rms radius :math:`\sqrt{\langle r^2\rangle}` of the
  cut function: ``rms_ratio`` times the free-atom valence rms radius of the
  same :math:`l` (the dataset's own bound orbital, unconfined), or of the
  atom's outermost valence orbital for an :math:`l` with no valence state
  (hydrogen's p, oxygen's d); ``rms_radius`` (Angstrom) gives it directly;
* the **support**, :math:`r_{cut}`, where the uncut Gaussian
  :math:`r^l e^{-\alpha r^2}` leaves ``cutoff_norm`` of its squared norm
  outside.  Width and support are independent: a narrow Gaussian stays
  narrow however far its cutoff.

For a Gaussian both are closed forms, and both scale as
:math:`\alpha^{-1/2}`: the tail fraction is the regularized gamma function
:math:`Q(l + 3/2, 2\alpha r_{cut}^2)`, so :math:`r_{cut} = c_l/\sqrt\alpha`,
and the cut function's rms radius is :math:`\rho_l/\sqrt\alpha` with
:math:`\rho_l` evaluated once at :math:`\alpha = 1`.  No root search is
needed.  ``count`` shells per channel follow a geometric sequence of rms
radii, ``ratio`` apart (default :math:`\sqrt{3.5}`, the exponent ratio
:data:`~mandacaru.basis.gaussian_families.DIFFUSE_RATIO` of the Gaussian
families as a length).

Each shell is tabulated on **its own radial grid**: the dataset's grid,
continued past its 30-Bohr end when the support needs it (a lithium s shell
reaches 30-40 Bohr).  It is Fourier-filtered like every other function of
the basis, then **Gram-Schmidt orthogonalized on its atom** against the
atom's functions of the same :math:`l`, compact ones first, and dropped
when less than :data:`LINEAR_DEPENDENCE_TOL` of its squared norm is left
(the rule of the all-electron NAO family, :mod:`mandacaru.basis.nao_ae`).
That step depends only on the atom, never on the geometry, so it cannot make
an energy scan discontinuous; the dependence between atoms is left to the
Loewdin guard, which warns and refuses loudly.

The PAW-LCAO argument is unchanged: a diffuse function belongs to the trial
space only.  It is projected on the dataset's projectors like any other
function, enters the augmented overlap :math:`S + C Q C^\dagger`, and adds no
projector, occupation or dataset change.

Spellings (:func:`validate_diffuse`): ``False`` / ``None`` (the default, the
basis of today), ``True`` (the recipe :data:`DEFAULT_DIFFUSE`), an ``int``
(that many shells per channel with the default recipe), a dict of
:data:`DIFFUSE_KEYS`, or ``{symbol: any of these}`` with a ``"*"`` default.

The construction follows the augmented correlation-consistent sets in what
it adds (one diffuse shell per angular momentum present, Kendall, Dunning and
Harrison 1992) and the numerical-orbital bases in how (a finite-support
radial function, Artacho *et al.* 1999; Junquera *et al.* 2001).  Molecules
only: a crystal is refused (:func:`refuse_crystal`).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from ..units import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM

#: Keys a ``diffuse`` recipe dict accepts.
DIFFUSE_KEYS = ("shells", "count", "rms_ratio", "ratio", "rms_radius",
                "cutoff_norm")

#: ``shells="present"``: one shell for every angular momentum the atom's
#: ordinary basis has (valence and polarization), up to this ``l`` -- the cap
#: of the augmented Gaussian families (``diffuse_l_max``).
PRESENT_L_MAX = 2

#: Highest angular momentum an explicit ``shells`` list may name.
DIFFUSE_L_LIMIT = 3

_L_LETTERS = "spdf"

#: rms radius of the first diffuse shell over the free-atom valence rms radius
#: of its ``l`` (the default recipe).  Measured on DZP (HISTORY.md, "Diffuse
#: functions: the option, Phases 1-4"): 2.5 against 2.0 gives the F UHF
#: affinity 1.17 against 1.00 eV (all-electron 1.18), H- 0.59 against 0.54 eV,
#: water's RHF alpha 8.58 against 8.54 a.u. and twice the smallest overlap
#: eigenvalue; 3.0 opens a hole between the valence orbital and the shell
#: (Cl 2.25 against 2.36 eV).  2.0 is better on the hydrogen-rich H2 and CH4
#: (hydrogen's p shell, sized from its 1s, comes out wide); over six molecules
#: the PBE alpha error has the same mean at both and a lower median at 2.5.  The aug-cc-pVDZ diffuse primitives sit at
#: 2.2-2.8 times the same free-atom radii.  The price is the box: 7.2 A of
#: vacuum around hydrogen instead of 5.8.
DEFAULT_RMS_RATIO = 2.5

#: Ratio between successive rms radii of one channel (``count > 1``): the
#: Gaussian families' exponent ratio 3.5 as a length.
DEFAULT_SHELL_RATIO = float(np.sqrt(3.5))

#: The uncut Gaussian's squared norm left beyond ``r_cut``.  1e-4 instead of
#: the filter's 1e-6 shortens every support by 14-15 % (the box needed by as
#: much); measured, it moved water's alpha by 0.01 a.u. (Phase 0) and the F
#: and H- affinities by 7 and 10 meV.
DEFAULT_CUTOFF_NORM = 1e-4

#: Largest ``cutoff_norm`` accepted: beyond it the cut removes a visible part
#: of the function.
MAX_CUTOFF_NORM = 1e-2

#: One-center Gram-Schmidt: a shell keeping less than this squared norm
#: after its atom's same-``l`` functions are projected out is dropped.
LINEAR_DEPENDENCE_TOL = 1e-4

#: The diffuse grid runs to this multiple of ``r_cut``, past the window the
#: Fourier filter places at 1.5 times the function's support.
GRID_REACH = 1.6

#: A function whose norm on the real-space grid falls below ``1 - BOX_TOL``
#: of its radial norm is cut by the box: warned about.
BOX_TOL = 1e-4


@dataclass(frozen=True)
class DiffuseRecipe:
    """One element's normalized ``diffuse`` recipe.

    ``shells`` is ``"present"`` or a sorted tuple of ``l``; ``rms_radius`` is
    in **Angstrom** (``None``: sized by ``rms_ratio``).
    """

    shells: object = "present"
    count: int = 1
    rms_ratio: float = DEFAULT_RMS_RATIO
    ratio: float = DEFAULT_SHELL_RATIO
    rms_radius: float | None = None
    cutoff_norm: float = DEFAULT_CUTOFF_NORM

    def label(self) -> str:
        shells = (self.shells if self.shells == "present"
                  else "".join(_L_LETTERS[l] for l in self.shells))
        width = (f"rms {self.rms_radius:g} A" if self.rms_radius is not None
                 else f"rms_ratio {self.rms_ratio:g}")
        parts = [f"shells {shells}", f"count {self.count}", width]
        if self.count > 1:
            parts.append(f"ratio {self.ratio:.4g}")
        parts.append(f"cutoff_norm {self.cutoff_norm:g}")
        return ", ".join(parts)


#: The recipe ``diffuse=True`` selects: one shell per angular momentum present,
#: rms radius 2.5 times the free-atom valence one, cut at a 1e-4 tail.
DEFAULT_DIFFUSE = DiffuseRecipe()


@dataclass(frozen=True)
class DiffuseShell:
    """One diffuse radial function as built (the ``[BASIS]`` record).

    Lengths in Bohr; ``alpha`` in Bohr^-2.  ``residual`` is the squared norm
    left after the one-center Gram-Schmidt; ``accepted`` is whether it
    cleared :data:`LINEAR_DEPENDENCE_TOL`.
    """

    symbol: str
    l: int
    index: int
    n: int
    reference_l: int
    reference_rms: float
    rms: float
    alpha: float
    r_cut: float
    residual: float
    accepted: bool
    r_max: float = field(default=0.0)       # the end of its radial grid


# --------------------------------------------------------------------------- #
# The option.
# --------------------------------------------------------------------------- #

def _shells(value):
    if isinstance(value, str):
        key = value.strip().lower()
        if key == "present":
            return "present"
        letters = list(key)
    elif isinstance(value, (list, tuple)):
        letters = list(value)
    else:
        raise ValueError(
            "diffuse 'shells' is 'present', a string of letters such as "
            f"'sp', or a list of l (or letters), got {value!r}")
    out = set()
    for item in letters:
        if isinstance(item, str) and item.strip().lower() in _L_LETTERS:
            out.add(_L_LETTERS.index(item.strip().lower()))
        elif (isinstance(item, (int, np.integer)) and not isinstance(item, bool)
              and 0 <= int(item) <= DIFFUSE_L_LIMIT):
            out.add(int(item))
        else:
            raise ValueError(
                f"diffuse 'shells' entries are l = 0..{DIFFUSE_L_LIMIT} or "
                f"the letters {_L_LETTERS!r}, got {item!r}")
    if not out:
        raise ValueError("diffuse 'shells' names no angular momentum")
    return tuple(sorted(out))


def _positive(key, value, *, above=0.0):
    if isinstance(value, bool) or not isinstance(value, (int, float,
                                                         np.floating,
                                                         np.integer)):
        raise ValueError(f"diffuse {key!r} must be a number, got {value!r}")
    value = float(value)
    if not np.isfinite(value) or value <= above:
        raise ValueError(f"diffuse {key!r} must be greater than {above:g}, "
                         f"got {value!r}")
    return value


def _recipe(spec):
    """One element's spec (no mapping) -> :class:`DiffuseRecipe` or ``None``."""
    if spec is None or spec is False:
        return None
    if isinstance(spec, DiffuseRecipe):
        return spec
    if spec is True:
        return DEFAULT_DIFFUSE
    if isinstance(spec, (int, np.integer)):
        count = int(spec)
        if count < 0:
            raise ValueError(f"a diffuse shell count must be >= 0, got {spec!r}")
        return None if count == 0 else DiffuseRecipe(count=count)
    if not isinstance(spec, dict):
        raise ValueError(
            "diffuse must be True / False / None, a shell count, a recipe "
            f"dict with keys {list(DIFFUSE_KEYS)}, or {{symbol: ...}} with an "
            f"optional '*' default; got {spec!r}")
    unknown = sorted(set(spec) - set(DIFFUSE_KEYS))
    if unknown:
        raise ValueError(f"unknown diffuse key(s) {unknown}; a recipe takes "
                         f"{list(DIFFUSE_KEYS)}")
    count = spec.get("count", 1)
    if (isinstance(count, bool) or not isinstance(count, (int, np.integer))
            or int(count) < 1):
        raise ValueError(f"diffuse 'count' is a positive int, got {count!r}")
    cutoff_norm = _positive("cutoff_norm",
                            spec.get("cutoff_norm", DEFAULT_CUTOFF_NORM))
    if cutoff_norm > MAX_CUTOFF_NORM:
        raise ValueError(f"diffuse 'cutoff_norm' must be at most "
                         f"{MAX_CUTOFF_NORM:g}, got {cutoff_norm!r}")
    rms_radius = spec.get("rms_radius")
    return DiffuseRecipe(
        shells=_shells(spec.get("shells", "present")),
        count=int(count),
        rms_ratio=_positive("rms_ratio", spec.get("rms_ratio",
                                                  DEFAULT_RMS_RATIO),
                            above=1.0),
        ratio=_positive("ratio", spec.get("ratio", DEFAULT_SHELL_RATIO),
                        above=1.0),
        rms_radius=(None if rms_radius is None
                    else _positive("rms_radius", rms_radius)),
        cutoff_norm=cutoff_norm)


def _is_element_mapping(spec: dict) -> bool:
    from ase.data import chemical_symbols

    if not spec:
        return False
    recipe = [k for k in spec if k in DIFFUSE_KEYS]
    elements = [k for k in spec
                if k == "*" or str(k).capitalize() in chemical_symbols[1:]]
    if elements and len(elements) == len(spec):
        return True
    if not elements:
        return False                # a recipe; unknown keys are refused there
    if not recipe:
        unknown = sorted(str(k) for k in spec if k not in elements)
        raise ValueError(f"unknown diffuse key(s) {unknown}: neither a recipe "
                         f"key {list(DIFFUSE_KEYS)} nor an element symbol")
    raise ValueError(
        f"a diffuse dict is either a recipe with keys {list(DIFFUSE_KEYS)} or "
        f"an element mapping {{symbol: recipe, '*': default}}, not a mix; got "
        f"keys {sorted(map(str, spec))}")


def validate_diffuse(spec):
    """Normalize a ``diffuse`` option.

    ``None`` / ``False`` / ``0`` are off and return ``None``.  ``True``, a
    shell count or a recipe dict return a :class:`DiffuseRecipe`; a
    ``{symbol: spec}`` mapping (``"*"`` = default) returns ``{symbol:
    DiffuseRecipe or None}``, or ``None`` when every entry is off.
    Idempotent: a normalized value validates to itself.
    """
    if isinstance(spec, dict) and _is_element_mapping(spec):
        table = {}
        for key, value in spec.items():
            if isinstance(value, dict) and value and _is_element_mapping(value):
                raise ValueError("a diffuse element mapping cannot nest "
                                 f"another one (under {key!r})")
            table[key if key == "*" else str(key).capitalize()] = _recipe(value)
        return table if any(v is not None for v in table.values()) else None
    return _recipe(spec)


def diffuse_of(spec, symbol) -> DiffuseRecipe | None:
    """One element's recipe under ``spec`` (``None``: no diffuse shells)."""
    spec = validate_diffuse(spec)
    if isinstance(spec, dict):
        return spec.get(str(symbol).capitalize(), spec.get("*"))
    return spec


def diffuse_recipe_text(spec) -> str:
    """The normalized option in words: ``"off"``, one recipe, or
    ``{symbol: recipe}``."""
    spec = validate_diffuse(spec)
    if spec is None:
        return "off"
    if isinstance(spec, dict):
        return "{" + "; ".join(f"{k}: {'off' if v is None else v.label()}"
                               for k, v in sorted(spec.items())) + "}"
    return spec.label()


def diffuse_label(spec) -> str:
    """How a basis label names the option: ``"no diffuse"`` or the recipe."""
    if validate_diffuse(spec) is None:
        return "no diffuse"
    return f"diffuse ({diffuse_recipe_text(spec)})"


def refuse_crystal(spec) -> None:
    """Raise when a crystal is asked for diffuse shells."""
    if validate_diffuse(spec) is None:
        return
    raise ValueError(
        "the 'diffuse' basis option is for molecules: in a crystal a diffuse "
        "shell is nearly linearly dependent on the neighbors' valence "
        "functions, and its Bloch sum runs over several times the images of "
        "a confined orbital.  Drop 'diffuse' for a periodic geometry")


# --------------------------------------------------------------------------- #
# The radial function.
# --------------------------------------------------------------------------- #

def _gaussian_cut(l: int, cutoff_norm: float) -> float:
    """``c_l`` with ``r_cut = c_l / sqrt(alpha)``: the radius leaving
    ``cutoff_norm`` of ``r^l exp(-alpha r^2)``'s squared norm outside."""
    from scipy.special import gammainccinv

    return float(np.sqrt(0.5 * gammainccinv(l + 1.5, float(cutoff_norm))))


def _rms(r, values) -> float:
    density = np.asarray(values, dtype=float) ** 2 * np.asarray(r) ** 2
    return float(np.sqrt(np.trapezoid(density * r * r, r)
                         / np.trapezoid(density, r)))


_UNIT_RMS: dict = {}


def _unit_rms(l: int, cutoff_norm: float) -> float:
    """rms radius of the cut quasi-Gaussian at ``alpha = 1`` (Bohr)."""
    from .confinement import quasi_gaussian

    key = (int(l), float(cutoff_norm))
    if key not in _UNIT_RMS:
        c = _gaussian_cut(l, cutoff_norm)
        r = np.linspace(0.0, c, 20001)
        _UNIT_RMS[key] = _rms(r, r ** l * quasi_gaussian(r, 1.0, c))
    return _UNIT_RMS[key]


def diffuse_parameters(l: int, rms: float, cutoff_norm: float = DEFAULT_CUTOFF_NORM
                       ) -> tuple[float, float]:
    """``(alpha, r_cut)`` (Bohr^-2, Bohr) of the cut quasi-Gaussian of
    angular momentum ``l`` whose rms radius is ``rms`` (Bohr)."""
    alpha = (_unit_rms(l, cutoff_norm) / float(rms)) ** 2
    return alpha, _gaussian_cut(l, cutoff_norm) / np.sqrt(alpha)


def diffuse_radial(r, l: int, rms: float,
                   cutoff_norm: float = DEFAULT_CUTOFF_NORM):
    """``(R, alpha, r_cut)``: the unit-norm diffuse shell on ``r`` (Bohr)."""
    from .confinement import quasi_gaussian

    r = np.asarray(r, dtype=float)
    alpha, r_cut = diffuse_parameters(l, rms, cutoff_norm)
    values = r ** l * quasi_gaussian(r, alpha, r_cut)
    values = values / np.sqrt(np.trapezoid(values * values * r * r, r))
    return values, alpha, r_cut


def free_valence_rms(pp) -> dict:
    """``{l: rms}`` (Bohr) of the dataset's free-atom valence orbitals: the
    stored bound partial wave of each occupied valence state (the
    outermost when a channel has two), and an empty channel's bound level."""
    from .orbitals import basis_states, bound_radial

    r = np.asarray(pp.r, dtype=float)
    out: dict = {}
    for state in basis_states(pp):
        if state.semicore:
            continue
        radial = (bound_radial(pp, state.l) if state.empty
                  else pp.channels[state.l].pseudo_waves[state.index])
        out[state.l] = max(out.get(state.l, 0.0), _rms(r, radial))
    return out


def _extended_grid(r, reach: float) -> np.ndarray:
    """``r`` continued with its last spacing up to ``reach`` (Bohr)."""
    r = np.asarray(r, dtype=float)
    if reach <= r[-1]:
        return r
    h = float(r[-1] - r[-2])
    extra = r[-1] + h * np.arange(1, int(np.ceil((reach - r[-1]) / h)) + 1)
    return np.concatenate([r, extra])


def _on_grid(r_new, r_old, values) -> np.ndarray:
    """A table on ``r_old`` resampled on ``r_new``, zero past its end."""
    from scipy.interpolate import CubicSpline

    r_old = np.asarray(r_old, dtype=float)
    values = np.asarray(values, dtype=float)
    if r_new.size >= r_old.size and np.array_equal(r_new[:r_old.size], r_old):
        return np.concatenate([values, np.zeros(r_new.size - r_old.size)])
    spline = CubicSpline(r_old, values, extrapolate=False)
    out = spline(np.clip(r_new, r_old[0], r_old[-1]))
    return np.where((r_new > r_old[-1]) | np.isnan(out), 0.0, out)


def diffuse_tables(symbol, pp, existing, recipe: DiffuseRecipe,
                   filter_wavevector=None):
    """``(tables, shells)``: one atom's diffuse
    :class:`~mandacaru.basis.multizeta.RadialTable` s and their records.

    ``existing`` are the atom's ordinary radial functions as ``(n, l, r,
    values)``, final (filtered) form, in basis order.  The shells of each
    channel are built compact first, filtered (``filter_wavevector``, Bohr^-1)
    and Gram-Schmidt orthogonalized against ``existing`` and the shells
    accepted before them; a rejected shell is recorded, not built.
    """
    from ..basis.filtering import filter_radial
    from ..basis.multizeta import RadialTable

    present = sorted({int(l) for _n, l, _r, _v in existing})
    channels = ([l for l in present if l <= PRESENT_L_MAX]
                if recipe.shells == "present" else list(recipe.shells))
    reference = free_valence_rms(pp)
    if not reference:
        raise ValueError(f"{symbol} has no valence orbital to size diffuse "
                         "shells from")
    outermost = max(reference, key=reference.get)

    tables, shells = [], []
    for l in channels:
        ref_l = l if l in reference else outermost
        ref_rms = reference[ref_l]
        first = (recipe.rms_radius * ANGSTROM_TO_BOHR
                 if recipe.rms_radius is not None
                 else recipe.rms_ratio * ref_rms)
        targets = [first * recipe.ratio ** k for k in range(recipe.count)]
        _alpha, widest = diffuse_parameters(l, targets[-1], recipe.cutoff_norm)
        r = _extended_grid(pp.r, GRID_REACH * widest)
        same = [_on_grid(r, r_old, values)
                for _n, l_old, r_old, values in existing if l_old == l]
        n_next = max([n for n, l_old, _r, _v in existing if l_old == l],
                     default=l) + 1
        for k, rms in enumerate(targets):
            values, alpha, r_cut = diffuse_radial(r, l, rms,
                                                  recipe.cutoff_norm)
            if filter_wavevector is not None:
                values, _info = filter_radial(r, values, l,
                                              float(filter_wavevector))
                values = values / np.sqrt(np.trapezoid(values * values * r * r,
                                                       r))
            candidate = values.copy()
            norms = [float(np.trapezoid(g * g * r * r, r)) for g in same]
            basis = [g / np.sqrt(norm) for g, norm in zip(same, norms)
                     if norm > 0.0]
            # Modified Gram-Schmidt against an orthonormalized copy of the
            # channel: the existing zetas are not orthogonal among themselves.
            ortho = []
            for g in basis:
                g = g.copy()
                for _pass in range(2):
                    for o in ortho:
                        g -= float(np.trapezoid(o * g * r * r, r)) * o
                norm = float(np.trapezoid(g * g * r * r, r))
                if norm > 1e-12:
                    ortho.append(g / np.sqrt(norm))
            for _pass in range(2):
                for o in ortho:
                    candidate -= float(np.trapezoid(o * candidate * r * r, r)) * o
            residual = float(np.trapezoid(candidate * candidate * r * r, r))
            accepted = residual >= LINEAR_DEPENDENCE_TOL
            shells.append(DiffuseShell(
                symbol=str(symbol), l=int(l), index=k, n=n_next + k,
                reference_l=int(ref_l), reference_rms=float(ref_rms),
                rms=float(rms), alpha=float(alpha), r_cut=float(r_cut),
                residual=residual, accepted=bool(accepted),
                r_max=float(r[-1])))
            if not accepted:
                continue
            candidate = candidate / np.sqrt(residual)
            if np.trapezoid(candidate * values * r * r, r) < 0.0:
                candidate = -candidate
            table = RadialTable(r=r, values=candidate, n=n_next + k, l=int(l),
                                zeta=1, diffuse=True)
            tables.append(table)
            same.append(candidate)
    return tables, shells


def diffuse_factory(spec, record: dict | None = None):
    """The ``diffuse_shells`` hook of :func:`~.orbitals.pseudo_basis`, or
    ``None`` when the option is off.

    The hook is ``(symbol, pp, existing, filter_wavevector) -> tables``;
    ``record`` (optional) receives ``{symbol: [DiffuseShell, ...]}``.  The
    tables are built once per element and shared by its atoms.
    """
    spec = validate_diffuse(spec)
    if spec is None:
        return None
    built: dict = {}

    def diffuse_shells(symbol, pp, existing, filter_wavevector=None):
        recipe = diffuse_of(spec, symbol)
        if recipe is None:
            return []
        key = (symbol, None if filter_wavevector is None
               else float(filter_wavevector))
        if key not in built:
            built[key] = diffuse_tables(symbol, pp, existing, recipe,
                                        filter_wavevector)
        tables, shells = built[key]
        if record is not None:
            record[symbol] = list(shells)
        return list(tables)

    return diffuse_shells


# --------------------------------------------------------------------------- #
# What the option demands of the rest of a calculation.
# --------------------------------------------------------------------------- #

def is_diffuse(function) -> bool:
    """Whether a basis function is a diffuse shell."""
    return bool(getattr(function, "diffuse", False))


def refuse_energy_ranking(method, n_diffuse: int, n_virtual: int,
                          n_virtual_kept: int) -> None:
    """Refuse an ``"energy"``-ranked truncation of a basis with diffuse shells.

    Diffuse shells make the lowest virtual orbitals, and the ``"energy"``
    ranking keeps the lowest first: a truncated register then fills with
    orbitals that carry almost none of the correlation.
    """
    if (n_diffuse and str(method) == "energy"
            and int(n_virtual_kept) < int(n_virtual)):
        raise ValueError(
            f"the basis has {n_diffuse} diffuse function(s) and the "
            f"active_space keeps {n_virtual_kept} of {n_virtual} virtual "
            "orbitals ranked by 'energy': diffuse shells make the lowest "
            "virtual orbitals, so that ranking fills the register with them "
            "whatever they contribute.  Rank the virtuals by occupation with "
            "'method': 'mp2' (or 'natural' for an open-shell reference), or "
            "keep every virtual orbital")


def check_active_space(active_space, basis_fns, num_particles) -> None:
    """:func:`refuse_energy_ranking` for a resolved basis and register
    request (``active_space`` as the builder receives it)."""
    from ..algorithms.active_space import (_resolve_counts,
                                           resolve_active_space_spec)

    n_diffuse = sum(1 for f in basis_fns if is_diffuse(f))
    if not n_diffuse or active_space is None:
        return
    spec = resolve_active_space_spec(active_space)
    if spec is None or spec.orbitals is None or isinstance(spec.orbitals,
                                                           tuple):
        return            # nothing ranked: no truncation, or explicit indices
    n_doubly = min(num_particles)
    n_singly = abs(num_particles[0] - num_particles[1])
    n_virtual = len(basis_fns) - n_doubly - n_singly
    _occupied, n_kept = _resolve_counts(spec.orbitals, n_doubly, n_singly,
                                        n_virtual, 0)
    refuse_energy_ranking(spec.method, n_diffuse, n_virtual, n_kept)


def _radial_table_norm(function) -> float:
    table = function.table
    r = np.asarray(table.r, dtype=float)
    values = np.asarray(table.values, dtype=float)
    return float(np.trapezoid(values * values * r * r, r))


def box_diagnostic(integrals, basis_fns, atom_of_orbital, symbols,
                   shells: dict | None = None):
    """``(ratios, rows)`` for the diffuse functions of a molecule.

    ``ratios[i]`` is basis function ``i``'s norm on the real-space grid over
    its radial norm (``nan`` for a function that is not diffuse); a ratio
    below ``1 - BOX_TOL`` is warned about -- the cell, which is a molecule's
    whole box, cuts the tail.  ``rows`` are one per atom with diffuse
    shells: its largest ``r_cut`` against the distance from the atom to the
    nearest face of the grid, and the smallest ratio among its functions.
    """
    psi = integrals._engine._psi
    grid = integrals.grid
    ratios = np.full(len(basis_fns), np.nan)
    for i, fn in enumerate(basis_fns):
        if is_diffuse(fn):
            grid_norm = float(np.real(np.vdot(psi[i], psi[i]))) * grid.dV
            ratios[i] = grid_norm / _radial_table_norm(fn)
    # The faces of the grid: node 0 plus the steps spanned by its nodes.
    origin = np.array([grid.X.flat[0], grid.Y.flat[0], grid.Z.flat[0]])
    span = np.asarray(grid.step, dtype=float) @ np.diag(
        np.asarray(grid.shape, dtype=float) - 1.0)
    inverse = np.linalg.inv(span)
    widths = 1.0 / np.linalg.norm(inverse, axis=1)
    owners = np.asarray(atom_of_orbital, dtype=int)
    rows = []
    for atom in sorted({int(owners[i]) for i in np.flatnonzero(
            np.isfinite(ratios))}):
        mine = [i for i in np.flatnonzero(owners == atom)
                if np.isfinite(ratios[i])]
        center = np.asarray(basis_fns[mine[0]].center, dtype=float)
        frac = inverse @ (center - origin)
        face = float(np.min(np.minimum(frac, 1.0 - frac) * widths))
        symbol = symbols[atom]
        r_cut = max((s.r_cut for s in (shells or {}).get(symbol, ())
                     if s.accepted), default=float("nan"))
        rows.append((atom, symbol, r_cut, face, float(np.min(ratios[mine]))))
    short = np.flatnonzero(ratios < 1.0 - BOX_TOL)
    if short.size:
        worst = int(short[np.argmin(ratios[short])])
        warnings.warn(
            f"the cell cuts {short.size} diffuse basis function(s): the "
            f"worst keeps {ratios[worst]:.6f} of its norm on the grid "
            f"(function {worst}, on atom {int(owners[worst])}).  A "
            "molecule's box is its cell; enlarge the vacuum until every "
            "atom's diffuse r_cut fits inside it (the [BASIS] diffuse_box "
            "table lists r_cut against each atom's distance to the cell "
            "face)", RuntimeWarning, stacklevel=3)
    return ratios, rows


def box_table(rows) -> list:
    """The ``[BASIS]`` ``diffuse_box`` table from :func:`box_diagnostic`'s
    rows (lengths in Bohr and Angstrom)."""
    return [("atom", "symbol", "r_cut_Bohr", "r_cut_Angstrom",
             "face_distance_Bohr", "face_distance_Angstrom", "grid_norm")] + [
        (atom, symbol, f"{r_cut:.4f}", f"{r_cut * BOHR_TO_ANGSTROM:.4f}",
         f"{face:.4f}", f"{face * BOHR_TO_ANGSTROM:.4f}", f"{ratio:.6f}")
        for atom, symbol, r_cut, face, ratio in rows]


def shells_table(shells: dict) -> list:
    """The ``[BASIS]`` ``diffuse_shells`` table (lengths in Bohr and
    Angstrom, ``alpha`` in Bohr^-2)."""
    rows = [("symbol", "l", "index", "rms_Bohr", "rms_Angstrom",
             "reference_l", "reference_rms_Bohr", "alpha_Bohr^-2",
             "r_cut_Bohr", "r_cut_Angstrom", "residual", "status")]
    for symbol, records in shells.items():
        for s in records:
            rows.append((symbol, s.l, s.index, f"{s.rms:.4f}",
                         f"{s.rms * BOHR_TO_ANGSTROM:.4f}", s.reference_l,
                         f"{s.reference_rms:.4f}", f"{s.alpha:.6f}",
                         f"{s.r_cut:.4f}", f"{s.r_cut * BOHR_TO_ANGSTROM:.4f}",
                         f"{s.residual:.6f}",
                         "accepted" if s.accepted else "rejected"))
    return rows
