# -*- coding: utf-8 -*-
# file: basis/gaussian_families.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Named Gaussian basis-set families, generated natively from their structure.

Carcará never reads a basis-set table.  The STO-nG contractions are least-squares
fits to Slater orbitals (:mod:`carcara.basis.sto_ng`) and 6-31G(d) is assembled
from those fits (:mod:`carcara.basis.pople`).  This module generalizes that
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
its exponents are Carcará's own, not the published, molecule-optimized values.
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
from dataclasses import dataclass, field

import numpy as np

from ._config import ground_state_config
from .sto_ng import _fit_reference, slater_exponent

#: A contracted shell: ``(l, exponents, coefficients)``.
ShellSpec = tuple[int, np.ndarray, np.ndarray]

_L_LETTERS = "spdfghik"
_L_OF = {c: i for i, c in enumerate(_L_LETTERS)}

#: Polarization exponent factors ``alpha_l = f_l * zeta_val^2``, by ``l``.
HEAVY_POLARIZATION_FACTOR = {2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0, 6: 1.2}
HYDROGEN_POLARIZATION_FACTOR = {1: 0.9, 2: 1.2, 3: 1.5, 4: 1.8, 5: 2.1}
#: Geometric ratio between successive polarization exponents of one ``l``.
POLARIZATION_RATIO = 3.0
#: A diffuse function sits at (most diffuse exponent of that l) / this.
DIFFUSE_RATIO = 3.5
#: A core-correlating function sits at (tightest valence exponent) * this.
TIGHT_RATIO = 3.0
#: Longest contraction the Slater-orbital fit is asked for.  The published
#: cc-pV5Z core runs to 14 primitives, but past ten the primitive overlap
#: matrix is numerically singular and the fit gains nothing (1 - <S|P|S> is
#: already 4e-9 at ten); the function count does not depend on it.
MAX_CONTRACTION = 10


# --------------------------------------------------------------------------- #
# The structure of a basis set.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GaussianRecipe:
    """How a named Gaussian basis set is put together.

    Attributes
    ----------
    name : str
        Canonical spelling (``"6-31+G*"``, ``"aug-cc-pVDZ"``, ``"def2-SVP"``).
    family : str
        ``"sto"``, ``"pople"``, ``"dunning"`` or ``"karlsruhe"``.
    core : tuple of int
        Contraction pattern of every core shell, e.g. ``(6,)``; ``(8, 1)`` adds
        an uncontracted tight primitive (the extra core ``s`` of def2-TZVP).
    valence : tuple of int
        Contraction pattern of every valence shell, e.g. ``(3, 1)``.
    polarization : tuple of tuple of int
        Polarisation shells as ``(l, count)`` pairs on heavy atoms
        (``Z >= polarization_min_z``).
    polarization_h : tuple of tuple of int
        Polarisation shells as ``(l, count)`` pairs on hydrogen and helium.
    diffuse : tuple of int
        Angular momenta receiving one diffuse function on heavy atoms.
    diffuse_h : tuple of int
        The same for hydrogen and helium.
    diffuse_present : bool
        Instead of the explicit lists, add one diffuse function for every
        angular momentum the atom already carries (``aug-`` / ``def2-...D``),
        up to ``diffuse_l_max``.
    diffuse_l_max : int
        Cap for ``diffuse_present``.
    tight : tuple of tuple of int
        Core-correlating ``(l, count)`` pairs on heavy atoms (``cc-pCVXZ``).
    polarization_min_z : int
        Lowest atomic number that counts as "heavy" for polarization: ``3``
        (anything beyond helium), or ``11`` for Pople's ``3-21G*``, whose
        ``*`` adds ``d`` functions to second-row atoms only.

    Notes
    -----
    A recipe is the *structure* of the named basis, not its published numbers:
    :attr:`provenance` says so in every label it appears in.
    """

    name: str
    family: str
    core: tuple[int, ...]
    valence: tuple[int, ...]
    polarization: tuple[tuple[int, int], ...] = ()
    polarization_h: tuple[tuple[int, int], ...] = ()
    diffuse: tuple[int, ...] = ()
    diffuse_h: tuple[int, ...] = ()
    diffuse_present: bool = False
    diffuse_l_max: int = 99
    tight: tuple[tuple[int, int], ...] = ()
    polarization_min_z: int = 3
    description: str = ""

    @property
    def zeta(self) -> int:
        """Number of functions per valence shell."""
        return len(self.valence)

    @property
    def provenance(self) -> str:
        """``"native:<name>-recipe"`` -- where these numbers come from.

        The shell structure is the published basis set's; the exponents and
        contraction coefficients are generated here (Slater fits, Slater's
        rules, even-tempered heuristics).  A calculation in ``native:cc-pVTZ-
        recipe`` is therefore **not** comparable term by term with another
        package's published ``cc-pVTZ``, and the published basis-set
        convergence literature does not transfer to it.  See
        :data:`PUBLISHED_NAMESPACE` for the namespace kept free for real
        tabulated data.
        """
        return f"{NATIVE_NAMESPACE}:{self.name}-recipe"

    def summary(self) -> str:
        parts = [f"{self.name} ({self.family})",
                 f"core {'+'.join(map(str, self.core))}" if self.core else "no core",
                 f"valence {'+'.join(map(str, self.valence))}"]
        if self.polarization:
            parts.append("heavy polarization " + " ".join(
                f"{c}{_L_LETTERS[l]}" for l, c in self.polarization))
        if self.polarization_h:
            parts.append("H polarization " + " ".join(
                f"{c}{_L_LETTERS[l]}" for l, c in self.polarization_h))
        if self.diffuse_present:
            parts.append("diffuse: one per l present")
        elif self.diffuse or self.diffuse_h:
            parts.append("diffuse heavy " + "".join(_L_LETTERS[l] for l in self.diffuse)
                         + (" / H " + "".join(_L_LETTERS[l] for l in self.diffuse_h)
                            if self.diffuse_h else ""))
        if self.tight:
            parts.append("tight " + " ".join(
                f"{c}{_L_LETTERS[l]}" for l, c in self.tight))
        return "; ".join(parts)


# --------------------------------------------------------------------------- #
# Name grammar.
# --------------------------------------------------------------------------- #

_STO_RE = re.compile(r"^STO-?(\d)G$", re.IGNORECASE)
_POPLE_RE = re.compile(
    r"^(\d)-(\d{2,3})(\+{0,2})G(\*{0,2})(?:\((.+)\))?$", re.IGNORECASE)
_DUNNING_RE = re.compile(r"^(AUG-)?CC-P(C?)V(D|T|Q|5|6)Z$", re.IGNORECASE)
_KARLSRUHE_RE = re.compile(
    r"^DEF2-(SV\(P\)|SVP|SVPD|TZVP|TZDP|TZVPD|TZVPP|TZVPPD|QZVP|QZVPD|QZVPP|QZVPPD)$",
    re.IGNORECASE)

_ZETA_OF = {"D": 2, "T": 3, "Q": 4, "5": 5, "6": 6}

#: Namespace of the recipes generated here -- ``"native:cc-pVTZ-recipe"``.
NATIVE_NAMESPACE = "native"

#: Namespace reserved for published, tabulated basis-set data.  Nothing
#: resolves in it: Carcará ships no basis-set tables, and a name written with
#: this prefix is refused rather than silently served from a recipe.
PUBLISHED_NAMESPACE = "published"


def _pol_spec(text: str) -> tuple[tuple[int, int], ...]:
    """``"3df"`` -> ``((2, 3), (3, 1))``; ``"d"`` -> ``((2, 1),)``; ``"2p"`` -> ``((1, 2),)``."""
    out = []
    for count, letter in re.findall(r"(\d*)([spdfghi])", text.strip().lower()):
        out.append((_L_OF[letter], int(count) if count else 1))
    if not out and text.strip():
        raise ValueError(f"cannot read the polarization spec {text!r}")
    return tuple(out)


def _parse_pople(match) -> GaussianRecipe:
    core_k, valence_digits, plus, stars, paren = match.groups()
    core = (int(core_k),)
    valence = tuple(int(d) for d in valence_digits)
    plus_count = len(plus)
    # Polarization: "*" -> d on heavy; "**" -> plus p on H; "(d)", "(d,p)",
    # "(3df,3pd)", "(2df,2p)" -> explicit per-atom-class shells.
    pol_heavy: tuple = ()
    pol_h: tuple = ()
    if stars:
        pol_heavy = ((2, 1),)
        if len(stars) == 2:
            pol_h = ((1, 1),)
    elif paren:
        groups = [g.strip() for g in paren.split(",")]
        pol_heavy = _pol_spec(groups[0]) if groups and groups[0] else ()
        pol_h = _pol_spec(groups[1]) if len(groups) > 1 else ()
    # Pople's 3-21G* puts d functions on second-row atoms only.
    min_z = 11 if (core_k == "3" and valence_digits == "21") else 3
    pol_text = ("*" * len(stars)) if stars else (f"({paren})" if paren else "")
    name = f"{core_k}-{valence_digits}{'+' * plus_count}G{pol_text}"
    return GaussianRecipe(
        name=name, family="pople", core=core, valence=valence,
        polarization=pol_heavy, polarization_h=pol_h,
        diffuse=(0, 1) if plus_count >= 1 else (),
        diffuse_h=(0,) if plus_count == 2 else (),
        polarization_min_z=min_z,
        description=(f"Pople split-valence: {core_k}-primitive core, valence "
                     f"{'+'.join(map(str, valence))}"
                     + (", diffuse sp on heavy atoms" if plus_count else "")
                     + (" and s on H" if plus_count == 2 else "")))


def _parse_dunning(match) -> GaussianRecipe:
    aug, core_corr, zeta_letter = match.groups()
    X = _ZETA_OF[zeta_letter.upper()]
    aug = bool(aug)
    core_corr = bool(core_corr)
    name = f"{'aug-' if aug else ''}cc-p{'C' if core_corr else ''}V{zeta_letter.upper()}Z"
    # Heavy atoms: (X-1) d, (X-2) f, ...; hydrogen: (X-1) p, (X-2) d, ...
    pol_heavy = tuple((2 + k, X - 1 - k) for k in range(X - 1))
    pol_h = tuple((1 + k, X - 1 - k) for k in range(X - 1))
    tight = tuple((l, X - 1) for l in (0, 1)) if core_corr else ()
    return GaussianRecipe(
        name=name, family="dunning",
        core=(min(4 + 2 * X, MAX_CONTRACTION),), valence=(4,) + (1,) * (X - 1),
        polarization=pol_heavy, polarization_h=pol_h,
        diffuse_present=aug, tight=tight,
        description=(f"Dunning correlation-consistent, {X}-zeta valence"
                     + (", augmented with one diffuse function per l" if aug else "")
                     + (", plus core-correlating tight functions" if core_corr else "")))


_KARLSRUHE_TABLE = {
    # key: (core pattern, valence pattern, heavy pol, H pol, diffuse)
    "SV(P)":  ((5,),    (3, 1),       ((2, 1),),                 (),                        False),
    "SVP":    ((5,),    (3, 1),       ((2, 1),),                 ((1, 1),),                 False),
    "SVPD":   ((5,),    (3, 1),       ((2, 1),),                 ((1, 1),),                 True),
    "TZVP":   ((6, 1),  (3, 1, 1),    ((2, 2), (3, 1)),          ((1, 1),),                 False),
    "TZDP":   ((6, 1),  (3, 1, 1),    ((2, 2), (3, 1)),          ((1, 1),),                 False),
    "TZVPD":  ((6, 1),  (3, 1, 1),    ((2, 2), (3, 1)),          ((1, 1),),                 True),
    "TZVPP":  ((6, 1),  (3, 1, 1),    ((2, 2), (3, 1)),          ((1, 2), (2, 1)),          False),
    "TZVPPD": ((6, 1),  (3, 1, 1),    ((2, 2), (3, 1)),          ((1, 2), (2, 1)),          True),
    "QZVP":   ((7, 1, 1), (3, 1, 1, 1), ((2, 3), (3, 2), (4, 1)), ((1, 3), (2, 2)),          False),
    "QZVPD":  ((7, 1, 1), (3, 1, 1, 1), ((2, 3), (3, 2), (4, 1)), ((1, 3), (2, 2)),          True),
    "QZVPP":  ((7, 1, 1), (3, 1, 1, 1), ((2, 3), (3, 2), (4, 1)), ((1, 3), (2, 2), (3, 1)),  False),
    "QZVPPD": ((7, 1, 1), (3, 1, 1, 1), ((2, 3), (3, 2), (4, 1)), ((1, 3), (2, 2), (3, 1)),  True),
}


def _parse_karlsruhe(match) -> GaussianRecipe:
    key = match.group(1).upper()
    core, valence, pol_heavy, pol_h, diffuse = _KARLSRUHE_TABLE[key]
    zeta = {2: "double", 3: "triple", 4: "quadruple"}[len(valence)]
    note = " (accepted as a spelling of def2-TZVP)" if key == "TZDP" else ""
    return GaussianRecipe(
        name=f"def2-{key}", family="karlsruhe", core=core, valence=valence,
        polarization=pol_heavy, polarization_h=pol_h,
        diffuse_present=diffuse, diffuse_l_max=2,
        description=(f"Karlsruhe def2, {zeta}-zeta valence with polarization"
                     + (", diffuse-augmented (D)" if diffuse else "") + note))


def parse_basis_name(name: str) -> GaussianRecipe:
    """Recover the structure of a named Gaussian basis set from its name.

    Understands ``STO-nG``, the Pople grammar ``K-NL[M][+|++]G[*|**|(...)]``,
    Dunning ``[aug-]cc-p[C]VXZ`` and the Karlsruhe ``def2-...`` names.  Raises
    :class:`ValueError` for anything else.
    """
    key = str(name).strip()
    prefix, _, rest = key.partition(":")
    if rest:
        prefix, rest = prefix.strip().lower(), rest.strip()
        if prefix == PUBLISHED_NAMESPACE:
            raise ValueError(
                f"{name!r}: Carcará ships no published basis-set tables. The "
                f"{PUBLISHED_NAMESPACE!r} namespace is reserved for them; what "
                f"the plain name {rest!r} builds is the natively generated "
                f"recipe {NATIVE_NAMESPACE}:{rest}-recipe -- same shell "
                "structure, Carcará's own exponents")
        if prefix == NATIVE_NAMESPACE:
            # Explicit spelling of what a bare name gives; "-recipe" optional.
            key = rest[:-len("-recipe")] if rest.lower().endswith("-recipe") \
                else rest
        else:
            raise ValueError(
                f"unknown basis-set namespace {prefix!r} in {name!r}; use "
                f"{NATIVE_NAMESPACE!r} or no prefix at all")
    match = _STO_RE.match(key)
    if match:
        n = int(match.group(1))
        return GaussianRecipe(name=f"STO-{n}G", family="sto", core=(n,),
                              valence=(n,),
                              description=f"minimal STO-{n}G, one {n}-primitive "
                                          "contraction per occupied subshell")
    match = _POPLE_RE.match(key)
    if match:
        return _parse_pople(match)
    match = _DUNNING_RE.match(key)
    if match:
        return _parse_dunning(match)
    match = _KARLSRUHE_RE.match(key)
    if match:
        return _parse_karlsruhe(match)
    raise ValueError(
        f"unknown Gaussian basis set {key!r}; expected STO-nG, a Pople name "
        "such as '6-31+G*' or '6-311G(2df,2p)', a Dunning name such as "
        "'aug-cc-pVTZ' / 'cc-pCVDZ', or a Karlsruhe name such as 'def2-TZVP'")


#: The named basis sets this module is validated against.
NAMED_BASIS_SETS = (
    "STO-3G", "STO-4G", "STO-5G", "STO-6G",
    "3-21G", "3-21G*", "3-21G**", "3-21+G", "3-21++G", "3-21+G*", "3-21+G**",
    "4-21G", "4-31G", "6-21G", "6-31G", "6-31G*", "6-31+G*", "6-31G(3df,3pd)",
    "6-311G", "6-311G*", "6-311+G*", "6-311+G(2df,2p)",
    "cc-pVDZ", "cc-pVTZ", "cc-pVQZ", "cc-pV5Z", "aug-cc-pVDZ", "cc-pCVDZ",
    "def2-SV(P)", "def2-SVP", "def2-SVPD", "def2-TZDP", "def2-TZVP", "def2-TZVPD",
    "def2-TZVPP", "def2-TZVPPD", "def2-QZVP", "def2-QZVPD", "def2-QZVPP",
    "def2-QZVPPD",
)


def available_basis_names() -> tuple[str, ...]:
    """The validated named Gaussian basis sets (the grammar accepts more)."""
    return NAMED_BASIS_SETS


# --------------------------------------------------------------------------- #
# Building the shells of one atom.
# --------------------------------------------------------------------------- #

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


def shell_data(atomic_number: int, recipe: GaussianRecipe) -> list[dict]:
    """The generated shells of one atom as plain data.

    Each entry is ``{"l", "shell", "exponents", "coefficients"}`` -- the numbers
    this basis actually uses, not a published table.  ``coefficients`` are as
    passed to :class:`~carcara.basis.gaussian.GaussianOrbital`, i.e. for
    unit-normalized primitives (see :data:`ANGULAR_CONVENTION`).
    """
    return [{"l": int(l), "shell": _L_LETTERS[int(l)],
             "exponents": [float(e) for e in np.asarray(exps).ravel()],
             "coefficients": [float(c) for c in np.asarray(coeffs).ravel()]}
            for (l, exps, coeffs) in gaussian_shells(atomic_number, recipe)]


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
    published basis set, without reading Carcará's source.

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
