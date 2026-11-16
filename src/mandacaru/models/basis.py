"""Declarative Gaussian recipes and name grammar; no numerical dependencies."""

from __future__ import annotations
import re
from dataclasses import dataclass

from .. import _native

_L_LETTERS = "spdfghik"


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


def _parse_pople(match) -> GaussianRecipe:
    core_k, valence_digits, plus, stars, paren = match.groups()
    core = (int(core_k),)
    valence = tuple(int(d) for d in valence_digits)
    # The marks' shells ("*", "**", "+", "++", "(2df,2p)") and the 3-21G* row limit are the
    # engine's rule; this only reads the name.
    ladder = _native.pople_ladder(core[0], list(valence), len(plus), len(stars), paren)
    pol_text = stars or (f"({paren})" if paren else "")
    return GaussianRecipe(
        name=f"{core_k}-{valence_digits}{plus}G{pol_text}",
        family="pople",
        core=core,
        valence=valence,
        polarization=tuple(map(tuple, ladder["polarization"])),
        polarization_h=tuple(map(tuple, ladder["polarization_h"])),
        diffuse=tuple(ladder["diffuse"]),
        diffuse_h=tuple(ladder["diffuse_h"]),
        polarization_min_z=ladder["polarization_min_z"],
        description=(
            f"Pople split-valence: {core_k}-primitive core, valence "
            f"{'+'.join(map(str, valence))}"
            + (", diffuse sp on heavy atoms" if ladder["diffuse"] else "")
            + (" and s on H" if ladder["diffuse_h"] else "")
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
    # The ladder (valence contractions, polarization and core-correlating shells) is the
    # engine's rule; this only names the family.
    ladder = _native.dunning_ladder(X, core_corr)
    return GaussianRecipe(
        name=name,
        family="dunning",
        core=(ladder["core"],),
        valence=tuple(ladder["valence"]),
        polarization=tuple(map(tuple, ladder["polarization"])),
        polarization_h=tuple(map(tuple, ladder["polarization_h"])),
        diffuse_present=aug,
        tight=tuple(map(tuple, ladder["tight"])),
        description=(
            f"Dunning correlation-consistent, {X}-zeta valence"
            + (", augmented with one diffuse function per l" if aug else "")
            + (", plus core-correlating tight functions" if core_corr else "")
        ),
    )


def _parse_karlsruhe(match) -> GaussianRecipe:
    key = match.group(1).upper()
    ladder = _native.karlsruhe_ladder(key)
    valence = tuple(ladder["valence"])
    zeta = {2: "double", 3: "triple", 4: "quadruple"}[len(valence)]
    note = " (accepted as a spelling of def2-TZVP)" if key == "TZDP" else ""
    return GaussianRecipe(
        name=f"def2-{key}",
        family="karlsruhe",
        core=tuple(ladder["core"]),
        valence=valence,
        polarization=tuple(map(tuple, ladder["polarization"])),
        polarization_h=tuple(map(tuple, ladder["polarization_h"])),
        diffuse_present=ladder["diffuse_present"],
        diffuse_l_max=ladder["diffuse_l_max"],
        description=(
            f"Karlsruhe def2, {zeta}-zeta valence with polarization"
            + (", diffuse-augmented (D)" if ladder["diffuse_present"] else "")
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
            key = re.sub(r"-recipe$", "", rest, flags=re.IGNORECASE)
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
