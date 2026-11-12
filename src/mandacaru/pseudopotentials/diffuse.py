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
radii, ``ratio`` apart (default :math:`\sqrt{3.5}`, the diffuse exponent ratio of the Gaussian
families as a length).

Each shell is tabulated on **its own radial grid**: the dataset's grid,
continued past its 30-Bohr end when the support needs it (a lithium s shell
reaches 30-40 Bohr).  It is Fourier-filtered like every other function of
the basis, then **Gram-Schmidt orthogonalized on its atom** against the
atom's functions of the same :math:`l`, compact ones first, and dropped
when less than :data:`LINEAR_DEPENDENCE_TOL` of its squared norm is left
(the rule of the all-electron NAO family of the Rust engine).
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

import math
import numbers
import warnings
from dataclasses import dataclass

import numpy as np

from .. import _native
from ..units import to_angstrom

#: Keys a ``diffuse`` recipe dict accepts.
DIFFUSE_KEYS = ("shells", "count", "rms_ratio", "ratio", "rms_radius",
                "cutoff_norm")


#: Highest angular momentum an explicit ``shells`` list may name.
DIFFUSE_L_LIMIT = 3

_L_LETTERS = "spdf"

#: The defaults of the diffuse recipe, one definition each, in Rust (``basis::diffuse``): the rms
#: radius of the first shell over the free-atom valence rms radius of its ``l`` (2.5: measured on
#: DZP, F UHF affinity 1.17 against 1.00 eV, all-electron 1.18; the price is the box, 7.2 A of vacuum
#: around hydrogen), the ratio between successive rms radii of one channel (the Gaussian families'
#: exponent ratio 3.5 as a length), the uncut Gaussian's squared norm left beyond ``r_cut`` and the
#: largest ``cutoff_norm`` accepted.
DEFAULT_RMS_RATIO = _native.DEFAULT_RMS_RATIO
DEFAULT_SHELL_RATIO = _native.DEFAULT_SHELL_RATIO
DEFAULT_CUTOFF_NORM = _native.DEFAULT_CUTOFF_NORM
MAX_CUTOFF_NORM = _native.MAX_CUTOFF_NORM

#: One-center Gram-Schmidt: a shell keeping less than this squared norm after its atom's same-``l``
#: functions are projected out is dropped.
LINEAR_DEPENDENCE_TOL = _native.LINEAR_DEPENDENCE_TOL


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
        elif (isinstance(item, numbers.Integral)
              and not isinstance(item, bool)
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
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"diffuse {key!r} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value <= above:
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
    if isinstance(spec, numbers.Integral):
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
    if (isinstance(count, bool) or not isinstance(count, numbers.Integral)
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
        items = sorted(spec.items())
        return "{" + "; ".join(f"{k}: {'off' if v is None else v.label()}"
                               for k, v in items) + "}"
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
# The radial function (Rust: ``basis::diffuse``).
# --------------------------------------------------------------------------- #

def _array(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=float)


def diffuse_parameters(l: int, rms: float, cutoff_norm: float = DEFAULT_CUTOFF_NORM
                       ) -> tuple[float, float]:
    """``(alpha, r_cut)`` (Bohr^-2, Bohr) of the cut quasi-Gaussian of
    angular momentum ``l`` whose rms radius is ``rms`` (Bohr)."""
    return _native.diffuse_parameters(int(l), float(rms), float(cutoff_norm))


def diffuse_radial(r, l: int, rms: float,
                   cutoff_norm: float = DEFAULT_CUTOFF_NORM):
    """``(R, alpha, r_cut)``: the unit-norm diffuse shell on ``r`` (Bohr)."""
    return _native.diffuse_radial(_array(r), int(l), float(rms),
                                  float(cutoff_norm))


# --------------------------------------------------------------------------- #
# What the option demands of the rest of a calculation.
# --------------------------------------------------------------------------- #

def is_diffuse(function) -> bool:
    """Whether a basis function is a diffuse shell."""
    return bool(getattr(function, "diffuse", False))


def warn_if_cut(box) -> None:
    """Warn when the cell cuts diffuse functions: ``box`` is the engine's box diagnostic
    (``_native.paw_diffuse_box``, ``paw::molecular::Setup::diffuse_box``), whose ``worst`` function
    keeps less than ``1 - BOX_TOL`` of its norm on the grid."""
    if box["short"]:
        function, atom, ratio = box["worst"]
        warnings.warn(
            f"the cell cuts {box['short']} diffuse basis function(s): the "
            f"worst keeps {ratio:.6f} of its norm on the grid "
            f"(function {function}, on atom {atom}).  A "
            "molecule's box is its cell; enlarge the vacuum until every "
            "atom's diffuse r_cut fits inside it (the [BASIS] diffuse_box "
            "table lists r_cut against each atom's distance to the cell "
            "face)", RuntimeWarning, stacklevel=2)


def box_table(box, symbols) -> list:
    """The ``[BASIS]`` ``diffuse_box`` table from the engine's box diagnostic (``box["rows"]``:
    atom, largest accepted ``r_cut``, distance to the nearest face of the cell, smallest grid norm;
    lengths in Bohr and Angstrom)."""
    return [("atom", "symbol", "r_cut_Bohr", "r_cut_Angstrom",
             "face_distance_Bohr", "face_distance_Angstrom", "grid_norm")] + [
        (atom, symbols[atom], f"{r_cut:.4f}", f"{to_angstrom(r_cut):.4f}",
         f"{face:.4f}", f"{to_angstrom(face):.4f}", f"{ratio:.6f}")
        for atom, r_cut, face, ratio in box["rows"]]


def shells_table(shells: dict) -> list:
    """The ``[BASIS]`` ``diffuse_shells`` table (lengths in Bohr and
    Angstrom, ``alpha`` in Bohr^-2)."""
    rows = [("symbol", "l", "index", "rms_Bohr", "rms_Angstrom",
             "reference_l", "reference_rms_Bohr", "alpha_Bohr^-2",
             "r_cut_Bohr", "r_cut_Angstrom", "residual", "status")]
    for symbol, records in shells.items():
        for s in records:
            rows.append((symbol, s.l, s.index, f"{s.rms:.4f}",
                         f"{to_angstrom(s.rms):.4f}", s.reference_l,
                         f"{s.reference_rms:.4f}", f"{s.alpha:.6f}",
                         f"{s.r_cut:.4f}", f"{to_angstrom(s.r_cut):.4f}",
                         f"{s.residual:.6f}",
                         "accepted" if s.accepted else "rejected"))
    return rows
