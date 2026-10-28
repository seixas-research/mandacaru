"""Declarative Gaussian recipes and name grammar; no numerical dependencies."""

from __future__ import annotations
import re
from dataclasses import dataclass

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
        Polarization shells as ``(l, count)`` pairs on heavy atoms
        (``Z >= polarization_min_z``).
    polarization_h : tuple of tuple of int
        Polarization shells as ``(l, count)`` pairs on hydrogen and helium.
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
        parts = [
            f"{self.name} ({self.family})",
            f"core {'+'.join(map(str, self.core))}" if self.core else "no core",
            f"valence {'+'.join(map(str, self.valence))}",
        ]
        if self.polarization:
            parts.append(
                "heavy polarization "
                + " ".join(f"{c}{_L_LETTERS[l]}" for l, c in self.polarization)
            )
        if self.polarization_h:
            parts.append(
                "H polarization "
                + " ".join(f"{c}{_L_LETTERS[l]}" for l, c in self.polarization_h)
            )
        if self.diffuse_present:
            parts.append("diffuse: one per l present")
        elif self.diffuse or self.diffuse_h:
            parts.append(
                "diffuse heavy "
                + "".join(_L_LETTERS[l] for l in self.diffuse)
                + (
                    " / H " + "".join(_L_LETTERS[l] for l in self.diffuse_h)
                    if self.diffuse_h
                    else ""
                )
            )
        if self.tight:
            parts.append(
                "tight " + " ".join(f"{c}{_L_LETTERS[l]}" for l, c in self.tight)
            )
        return "; ".join(parts)


# --------------------------------------------------------------------------- #
# Name grammar.
# --------------------------------------------------------------------------- #

_STO_RE = re.compile(r"^(?:STO-(\d)G|STO([456])G)$", re.IGNORECASE)
_POPLE_RE = re.compile(
    r"^(\d)-(\d{2,3})(\+{0,2})G(\*{0,2})(?:\((.+)\))?$", re.IGNORECASE
)
_DUNNING_RE = re.compile(r"^(AUG-)?CC-P(C?)V(D|T|Q|5|6)Z$", re.IGNORECASE)
_KARLSRUHE_RE = re.compile(
    r"^DEF2-(SV\(P\)|SVP|SVPD|TZVP|TZDP|TZVPD|TZVPP|TZVPPD|QZVP|QZVPD|QZVPP|QZVPPD)$",
    re.IGNORECASE,
)

_ZETA_OF = {"D": 2, "T": 3, "Q": 4, "5": 5, "6": 6}

#: Namespace of the recipes generated here -- ``"native:cc-pVTZ-recipe"``.
NATIVE_NAMESPACE = "native"

#: Namespace reserved for published, tabulated basis-set data.  Nothing
#: resolves in it: Mandacaru ships no basis-set tables, and a name written with
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
        name=name,
        family="pople",
        core=core,
        valence=valence,
        polarization=pol_heavy,
        polarization_h=pol_h,
        diffuse=(0, 1) if plus_count >= 1 else (),
        diffuse_h=(0,) if plus_count == 2 else (),
        polarization_min_z=min_z,
        description=(
            f"Pople split-valence: {core_k}-primitive core, valence "
            f"{'+'.join(map(str, valence))}"
            + (", diffuse sp on heavy atoms" if plus_count else "")
            + (" and s on H" if plus_count == 2 else "")
        ),
    )


def _parse_dunning(match) -> GaussianRecipe:
    aug, core_corr, zeta_letter = match.groups()
    X = _ZETA_OF[zeta_letter.upper()]
    aug = bool(aug)
    core_corr = bool(core_corr)
    name = (
        f"{'aug-' if aug else ''}cc-p{'C' if core_corr else ''}V{zeta_letter.upper()}Z"
    )
    # Heavy atoms: (X-1) d, (X-2) f, ...; hydrogen: (X-1) p, (X-2) d, ...
    pol_heavy = tuple((2 + k, X - 1 - k) for k in range(X - 1))
    pol_h = tuple((1 + k, X - 1 - k) for k in range(X - 1))
    tight = tuple((l, X - 1) for l in (0, 1)) if core_corr else ()
    return GaussianRecipe(
        name=name,
        family="dunning",
        core=(min(4 + 2 * X, MAX_CONTRACTION),),
        valence=(4,) + (1,) * (X - 1),
        polarization=pol_heavy,
        polarization_h=pol_h,
        diffuse_present=aug,
        tight=tight,
        description=(
            f"Dunning correlation-consistent, {X}-zeta valence"
            + (", augmented with one diffuse function per l" if aug else "")
            + (", plus core-correlating tight functions" if core_corr else "")
        ),
    )


_KARLSRUHE_TABLE = {
    # key: (core pattern, valence pattern, heavy pol, H pol, diffuse)
    "SV(P)": ((5,), (3, 1), ((2, 1),), (), False),
    "SVP": ((5,), (3, 1), ((2, 1),), ((1, 1),), False),
    "SVPD": ((5,), (3, 1), ((2, 1),), ((1, 1),), True),
    "TZVP": ((6, 1), (3, 1, 1), ((2, 2), (3, 1)), ((1, 1),), False),
    "TZDP": ((6, 1), (3, 1, 1), ((2, 2), (3, 1)), ((1, 1),), False),
    "TZVPD": ((6, 1), (3, 1, 1), ((2, 2), (3, 1)), ((1, 1),), True),
    "TZVPP": ((6, 1), (3, 1, 1), ((2, 2), (3, 1)), ((1, 2), (2, 1)), False),
    "TZVPPD": ((6, 1), (3, 1, 1), ((2, 2), (3, 1)), ((1, 2), (2, 1)), True),
    "QZVP": (
        (7, 1, 1),
        (3, 1, 1, 1),
        ((2, 3), (3, 2), (4, 1)),
        ((1, 3), (2, 2)),
        False,
    ),
    "QZVPD": (
        (7, 1, 1),
        (3, 1, 1, 1),
        ((2, 3), (3, 2), (4, 1)),
        ((1, 3), (2, 2)),
        True,
    ),
    "QZVPP": (
        (7, 1, 1),
        (3, 1, 1, 1),
        ((2, 3), (3, 2), (4, 1)),
        ((1, 3), (2, 2), (3, 1)),
        False,
    ),
    "QZVPPD": (
        (7, 1, 1),
        (3, 1, 1, 1),
        ((2, 3), (3, 2), (4, 1)),
        ((1, 3), (2, 2), (3, 1)),
        True,
    ),
}


def _parse_karlsruhe(match) -> GaussianRecipe:
    key = match.group(1).upper()
    core, valence, pol_heavy, pol_h, diffuse = _KARLSRUHE_TABLE[key]
    zeta = {2: "double", 3: "triple", 4: "quadruple"}[len(valence)]
    note = " (accepted as a spelling of def2-TZVP)" if key == "TZDP" else ""
    return GaussianRecipe(
        name=f"def2-{key}",
        family="karlsruhe",
        core=core,
        valence=valence,
        polarization=pol_heavy,
        polarization_h=pol_h,
        diffuse_present=diffuse,
        diffuse_l_max=2,
        description=(
            f"Karlsruhe def2, {zeta}-zeta valence with polarization"
            + (", diffuse-augmented (D)" if diffuse else "")
            + note
        ),
    )


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
                f"{name!r}: Mandacaru ships no published basis-set tables. The "
                f"{PUBLISHED_NAMESPACE!r} namespace is reserved for them; what "
                f"the plain name {rest!r} builds is the natively generated "
                f"recipe {NATIVE_NAMESPACE}:{rest}-recipe -- same shell "
                "structure, Mandacaru's own exponents"
            )
        if prefix == NATIVE_NAMESPACE:
            # Explicit spelling of what a bare name gives; "-recipe" optional.
            key = rest[: -len("-recipe")] if rest.lower().endswith("-recipe") else rest
        else:
            raise ValueError(
                f"unknown basis-set namespace {prefix!r} in {name!r}; use "
                f"{NATIVE_NAMESPACE!r} or no prefix at all"
            )
    match = _STO_RE.match(key)
    if match:
        n = int(match.group(1) or match.group(2))
        return GaussianRecipe(
            name=f"STO-{n}G",
            family="sto",
            core=(n,),
            valence=(n,),
            description=f"minimal STO-{n}G, one {n}-primitive "
            "contraction per occupied subshell",
        )
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
        "'aug-cc-pVTZ' / 'cc-pCVDZ', or a Karlsruhe name such as 'def2-TZVP'"
    )


#: The named basis sets this module is validated against.
NAMED_BASIS_SETS = (
    "STO-3G",
    "STO-4G",
    "STO-5G",
    "STO-6G",
    "3-21G",
    "3-21G*",
    "3-21G**",
    "3-21+G",
    "3-21++G",
    "3-21+G*",
    "3-21+G**",
    "4-21G",
    "4-31G",
    "6-21G",
    "6-31G",
    "6-31G*",
    "6-31+G*",
    "6-31G(3df,3pd)",
    "6-311G",
    "6-311G*",
    "6-311+G*",
    "6-311+G(2df,2p)",
    "cc-pVDZ",
    "cc-pVTZ",
    "cc-pVQZ",
    "cc-pV5Z",
    "aug-cc-pVDZ",
    "cc-pCVDZ",
    "def2-SV(P)",
    "def2-SVP",
    "def2-SVPD",
    "def2-TZDP",
    "def2-TZVP",
    "def2-TZVPD",
    "def2-TZVPP",
    "def2-TZVPPD",
    "def2-QZVP",
    "def2-QZVPD",
    "def2-QZVPP",
    "def2-QZVPPD",
)


def available_basis_names() -> tuple[str, ...]:
    """The validated named Gaussian basis sets (the grammar accepts more)."""
    return NAMED_BASIS_SETS


# --------------------------------------------------------------------------- #
# Building the shells of one atom.
# --------------------------------------------------------------------------- #
