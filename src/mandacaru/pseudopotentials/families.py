# -*- coding: utf-8 -*-
# file: pseudopotentials/families.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Pseudopotential **families** and their registry.

A family is everything that distinguishes one kind of pseudopotential from
another once the integral engine is reached: how a potential is *generated*
for an element, how it is *loaded* from a library, and how a molecule's
valence-only Hamiltonian is *built* from it -- the basis, the local
potential, the projectors and, in the general separable form of
:meth:`mandacaru.core.hamiltonian.MolecularIntegrals.kb_nonlocal`,

.. math::

    H^{NL} = C\,D\,C^\dagger, \qquad C_{\mu p} = \langle\phi_\mu|\chi_p\rangle ,

the block-diagonal coupling matrix :math:`D` (one block per ``(atom, l, m)``)
and, for families whose projectors also carry an overlap correction (PAW-LCAO), the
block matrix :math:`Q` that turns the basis overlap into
:math:`S + C\,Q\,C^\dagger`.

The registry :data:`PSEUDO_FAMILIES` maps a family name to its
:class:`FamilySpec`.  **A family is selected through the** ``basis``
**argument of any driver**, exactly like an all-electron family:
``basis="PAW-LCAO"``, ``basis={"name": "PAW-LCAO", "size": "DZP"}``, or a per-element
mapping ``{"O": {"name": "PAW-LCAO", "size": "DZP"}, "H": "PAW-LCAO"}``.  The drivers
only ever see the spec: :class:`~mandacaru.Mandacaru` recognizes a registered family name
(:func:`lookup_family`) and checks the basis dict against it
(:meth:`FamilySpec.validate_options`) in its constructor.  A new family is added with
:func:`register_family` and never needs the driver edited -- its name becomes
a basis name.

Registered today:

``"paw-lcao"`` (:mod:`.paw`)
    Bloechl's **projector augmented-wave** datasets -- two partial waves and
    projectors per channel, a :math:`2\times2` coupling block **and** a
    :math:`2\times2` overlap-correction block :math:`q` (``norm_conserving
    = False``), compensation multipoles in the one- and two-body terms and a
    frozen one-center constant.  Registered when :mod:`.paw` is imported
    (the package does so).  It and its ``"upaw-lcao"`` variant declare
    ``default_options = {"filter_cutoff": "auto"}``: their smooth partial waves are
    built to be band-limited, so filtering them to the grid's Nyquist
    wave-vector is nearly free and removes most of the egg-box.

Per-family defaults
-------------------
:attr:`FamilySpec.default_options` is where a family says what it does when
the user does not.  :meth:`FamilySpec.resolved_options` is the single merge
(defaults first, the user's basis dict on top), consulted by
:func:`build_valence_hamiltonian` and by the dry run, so a default is visible
in one place, a new family declares its own, and writing the option explicitly
always wins -- ``basis={"name": "PAW-LCAO", "filter_cutoff": False}`` is
exactly the unfiltered calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

#: The family a name-less request resolves to: PAW-LCAO (library:
#: ``$MANDACARU_PAW_PATH``).
DEFAULT_FAMILY = "paw-lcao"

#: Basis options every family accepts: the size hierarchy built on the
#: pseudo-orbitals and its split-valence scheme, the PAW dataset folder and the
#: cutoff of the Fourier filter (:mod:`mandacaru.basis.filtering`) that
#: removes from each radial function the wave-vectors the real-space grid
#: cannot represent.
COMMON_OPTIONS = ("size", "zeta_split", "paw_dataset", "filter_cutoff")


@dataclass(frozen=True)
class FamilySpec:
    """Everything the drivers need to know about one pseudopotential family.

    Parameters
    ----------
    name : str
        Canonical registry key (lower case).
    description : str
        One line for error messages and listings.
    generate : callable
        ``generate(symbol, **options) -> PseudoPotential``.
    get : callable
        ``get(symbol, paw_dataset=None) -> PseudoPotential`` -- the (cached)
        library loader.
    build : callable
        ``build(atoms, grid, grid_spacing, charge, spin, options, kinetic, *,
        active_space=None, hamiltonian=True)`` returning the driver
        5-tuple ``(hamiltonian, num_particles, n_spatial_orbitals,
        integration_profile, context)`` -- the same tuple
        the all-electron path returns (with ``hamiltonian=False``, the
        integrals only and ``None`` in place of the operator).  The keywords
        are always passed, so a builder must accept them (``**kwargs`` forwarded to
        :func:`build_valence_hamiltonian` is enough, and is what every built-in
        family does): a valence-only basis has no core to freeze but plenty of
        virtual orbitals to drop, so unlike a ``"frozen"`` core, truncation is
        a request a pseudopotential family has to be able to honor.
    norm_conserving : bool
        Whether the family's projectors leave the basis overlap untouched
        (``False`` for PAW-LCAO, which carries an overlap correction).
    aliases : tuple of str
        Alternative names resolving to this family.
    options : tuple of str
        The keys a ``{"name": <family>, ...}`` basis dict may carry for this
        family (default :data:`COMMON_OPTIONS`); anything else is refused
        before any integral is computed.
    default_options : dict
        Option values this family uses when the user does not say
        (``{"filter_cutoff": "auto"}`` for PAW-LCAO / UPAW-LCAO, whose smooth partial waves are
        built to be band-limited and lose little by being filtered).  Declaring the default **here** rather
        than in the builder is what keeps it discoverable: one place says what
        every family does, a new family declares its own, and
        :meth:`resolved_options` is the only merge.  Every key must be in
        :attr:`options`.
    """

    name: str
    description: str
    generate: Callable
    get: Callable
    norm_conserving: bool = True
    aliases: tuple = field(default_factory=tuple)
    options: tuple = COMMON_OPTIONS
    default_options: dict = field(default_factory=dict)

    def __post_init__(self):
        unknown = sorted(set(self.default_options) - set(self.options))
        if unknown:
            raise ValueError(
                f"family {self.name!r} declares default(s) for option(s) "
                f"{unknown} it does not accept; its options are "
                f"{list(self.options)}")

    @property
    def label(self) -> str:
        """The family name as a basis name (upper case)."""
        return self.name.upper()

    def resolved_options(self, options=None) -> dict:
        """The family's :attr:`default_options` with the user's ``options`` on top.

        Every consumer of a basis dict goes through this, so
        ``basis="PAW-LCAO"`` and ``basis={"name": "PAW-LCAO", "filter_cutoff":
        "auto"}`` are the same calculation and ``{"name": "PAW-LCAO",
        "filter_cutoff": False}`` is exactly the unfiltered one.
        """
        merged = dict(self.default_options)
        merged.update(options or {})
        return merged

    def validate_options(self, options=None) -> None:
        """Refuse a basis dict this family cannot honor; return ``None``.

        ``options`` is the user's basis dict without its ``"name"``.  A key
        outside :attr:`options` raises ``ValueError`` naming it and listing
        the valid ones; each value then goes through the validator its
        option owns (:data:`OPTION_VALIDATORS`).  Cheap by construction: it
        reads no dataset and builds no basis, so the ``Mandacaru``
        constructor can call it.
        """
        options = dict(options or {})
        unknown = sorted(str(key) for key in options if key not in self.options)
        if unknown:
            raise ValueError(
                f"unknown option(s) {unknown} for the {self.label} basis; "
                f"valid options: {list(self.options)}")
        for key, value in options.items():
            if key not in OPTION_VALIDATORS:
                raise ValueError(
                    f"option {key!r} of the {self.label} family has no "
                    "validator in OPTION_VALIDATORS")
            OPTION_VALIDATORS[key](value)


def _check_size(value):
    from ..models.radial import resolve_zeta

    if isinstance(value, dict):
        if not value:
            raise ValueError("a size mapping needs at least one element or '*'")
        for entry in value.values():
            resolve_zeta(entry)
    else:
        resolve_zeta(value)


def _check_zeta_split(value):
    from ..models.radial import validate_zeta_split

    validate_zeta_split(value)


def _check_paw_dataset(value):
    import os

    if value is not None and not isinstance(value, (str, os.PathLike)):
        raise ValueError(
            f"paw_dataset is a folder name or path, got {value!r}")


def _check_filter_cutoff(value):
    from ..basis.filtering import validate_filter_cutoff

    validate_filter_cutoff(value)


def _check_projector_basis(value):
    if value not in ("raw", "dual"):
        raise ValueError(f"projector_basis must be 'raw' or 'dual', got {value!r}")


def _check_energy_shift(value):
    from .confinement import validate_energy_shift

    validate_energy_shift(value)


def _check_confinement_potential(value):
    from .confinement import validate_confinement_potential

    validate_confinement_potential(value)


def _check_polarization_method(value):
    from .confinement import validate_polarization_method

    validate_polarization_method(value)


def _check_diffuse(value):
    from .diffuse import validate_diffuse

    validate_diffuse(value)


#: The validator each basis option owns, ``option -> callable(value)``; a
#: family's :attr:`FamilySpec.options` selects which of them apply.  Each
#: validates what the builder will later normalize and raises ``ValueError``,
#: so a typo fails at construction instead of mid-run.  A new option is
#: declared in a family's ``options`` and given its validator here.
OPTION_VALIDATORS: dict[str, Callable] = {
    "size": _check_size,
    "zeta_split": _check_zeta_split,
    "paw_dataset": _check_paw_dataset,
    "filter_cutoff": _check_filter_cutoff,
    "projector_basis": _check_projector_basis,
    "energy_shift": _check_energy_shift,
    "confinement_potential": _check_confinement_potential,
    "polarization_method": _check_polarization_method,
    "diffuse": _check_diffuse,
}

#: Registry ``name -> FamilySpec`` (canonical names only; see :func:`resolve_family`).
PSEUDO_FAMILIES: dict[str, FamilySpec] = {}
_ALIASES: dict[str, str] = {}


def _key(name) -> str:
    return str(name).strip().lower().replace("_", "-").replace(" ", "-")


def register_family(spec: FamilySpec, replace: bool = False) -> FamilySpec:
    """Add ``spec`` (and its aliases) to :data:`PSEUDO_FAMILIES`."""
    name = _key(spec.name)
    for key in (name, *map(_key, spec.aliases)):
        owner = _ALIASES.get(key)
        if owner is not None and owner != name and not replace:
            raise ValueError(
                f"pseudopotential family name {key!r} is already taken by "
                f"{owner!r}")
    PSEUDO_FAMILIES[name] = spec
    _ALIASES[name] = name
    for alias in spec.aliases:
        _ALIASES[_key(alias)] = name
    return spec


def unregister_family(name) -> FamilySpec:
    """Remove a family (and its aliases) from the registry; returns the spec."""
    spec = resolve_family(name)
    if spec.name == DEFAULT_FAMILY:
        raise ValueError(f"the default family {DEFAULT_FAMILY!r} cannot be "
                         "unregistered")
    del PSEUDO_FAMILIES[spec.name]
    for key in [k for k, v in _ALIASES.items() if v == spec.name]:
        del _ALIASES[key]
    return spec


def family_names() -> list[str]:
    """Every accepted name: the default family, the other canonical names,
    then the aliases."""
    canonical = [DEFAULT_FAMILY] + sorted(k for k in PSEUDO_FAMILIES
                                          if k != DEFAULT_FAMILY)
    aliases = sorted(k for k, v in _ALIASES.items() if k != v)
    return canonical + aliases


def family_listing() -> str:
    """One line naming every registered family and its aliases."""
    return ", ".join(
        f"{spec.name!r}" + (f" (aliases: {', '.join(map(repr, spec.aliases))})"
                            if spec.aliases else "")
        for spec in PSEUDO_FAMILIES.values())


def lookup_family(name) -> FamilySpec | None:
    """The :class:`FamilySpec` registered under ``name``, or ``None``.

    Case-insensitive, aliases accepted (``"PAW-LCAO"``, ``"unitary-paw-lcao"``);
    anything that is not a registered family name -- ``"HAO"``, ``"cc-pVTZ"``,
    a per-element mapping -- gives ``None``.  This is how
    :class:`~mandacaru.Mandacaru` tells a pseudopotential basis from an
    all-electron one.
    """
    if isinstance(name, FamilySpec):
        return name
    if not isinstance(name, str):
        return None
    if _key(name) in ("paw-lcao", "upaw-lcao", "unitary-paw-lcao") and _key(name) not in _ALIASES:
        # Built-in registration must not depend on which public module a user
        # happened to import first.
        from importlib import import_module
        import_module(".paw", __package__)
    canonical = _ALIASES.get(_key(name))
    return None if canonical is None else PSEUDO_FAMILIES[canonical]


def canonical_family_name(name) -> str:
    """The canonical registry key for ``name`` (an alias, any case); an
    unregistered name is returned normalized but otherwise unchanged."""
    key = _key(name)
    return _ALIASES.get(key, key)


def resolve_family(name=None) -> FamilySpec:
    """The :class:`FamilySpec` for ``name`` (case-insensitive; aliases accepted).

    ``None`` selects :data:`DEFAULT_FAMILY`; an unknown name raises
    ``ValueError`` listing the registered families.
    """
    if name is None:
        name = DEFAULT_FAMILY
    if isinstance(name, FamilySpec):
        return name
    if not isinstance(name, str):
        raise TypeError(
            f"a pseudopotential family is named by a string, got {name!r}")
    spec = lookup_family(name)
    if spec is None:
        raise ValueError(
            f"unknown pseudopotential family {name!r}; registered families: "
            f"{family_listing()}")
    return spec


def pseudo_basis_arguments(family, options, *, confinement=None,
                           polarization=None, diffuse=None) -> dict:
    """Keyword arguments of :func:`~.orbitals.pseudo_basis` for ``options``.

    The one place a basis dict is turned into a construction, shared by the
    Hamiltonian builder and the dry run so the two cannot count different
    bases.  ``confinement`` / ``polarization`` (optional dicts) receive what
    the hooks resolve -- each orbital's cutoff radius, the polarization
    shell's -- for the run log, and ``diffuse`` (optional) the diffuse shells
    (:mod:`.diffuse`).

    The split-valence scheme is ``zeta_split`` (``"first_zeta"`` unless
    written), and the scheme fixes its own tail norms or split norm
    (:func:`~mandacaru.basis.multizeta.resolve_split_scheme`).
    """
    from ..basis.multizeta import resolve_split_scheme

    spec = resolve_family(family)
    split_norm, tail_norms = resolve_split_scheme(options.get("zeta_split"))
    arguments = {"size": options.get("size", "SZ"), "split_norm": split_norm,
                 "tail_norms": tail_norms}
    if "energy_shift" in spec.options:
        from .confinement import first_zeta_factory, polarization_factory
        arguments["first_zeta"] = first_zeta_factory(
            options.get("energy_shift"), confinement,
            options.get("confinement_potential"))
        arguments["polarization_shape"] = polarization_factory(
            options.get("polarization_method"), options.get("energy_shift"),
            polarization, options.get("confinement_potential"))
    if "diffuse" in spec.options:
        from .diffuse import diffuse_factory
        hook = diffuse_factory(options.get("diffuse"), diffuse)
        if hook is not None:
            arguments["diffuse_shells"] = hook
    return arguments
