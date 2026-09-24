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
``basis="PAW-LCAO"``, ``basis={"name": "NCPP", "size": "DZP"}``, or a per-element
mapping ``{"O": {"name": "PAW-LCAO", "size": "DZP"}, "H": "PAW-LCAO"}``.  The drivers
only ever see the spec: :func:`mandacaru.algorithms._hamiltonian_from_atoms.resolve_basis`
recognizes a registered family name (:func:`lookup_family`) and
:func:`~mandacaru.algorithms._hamiltonian_from_atoms._pseudopotential_hamiltonian`
dispatches to ``spec.build``.  A new family is added with
:func:`register_family` and never needs the driver edited -- its name becomes
a basis name.

Registered today:

``"ncpp"`` (aliases ``"tm"``, ``"ncpp-tm"``)
    Norm-conserving **Troullier-Martins** potentials in Kleinman-Bylander
    separable form -- one projector per channel, :math:`D = \mathrm{diag}(E^{KB})`,
    no overlap correction.  The bundled library (``library/ncpp/``) covers
    every element up to uranium.

``"oncvpsp"`` (alias ``"oncv"``; :mod:`.oncv`)
    Hamann's **optimized norm-conserving Vanderbilt** potentials -- two
    projectors per channel with a :math:`2\times2` coupling block, a
    polynomial local potential that is not a channel, no overlap correction.
    Registered when :mod:`.oncv` is imported (the package does so).

``"paw-lcao"`` (:mod:`.paw`)
    Bloechl's **projector augmented-wave** datasets -- two partial waves and
    projectors per channel, a :math:`2\times2` coupling block **and** a
    :math:`2\times2` overlap-correction block :math:`q` (``norm_conserving
    = False``), compensation multipoles in the one- and two-body terms and a
    frozen one-center constant.  Registered when :mod:`.paw` is imported
    (the package does so).  It and its ``"upaw-lcao"`` variant declare
    ``default_options = {"filter": True}``: their smooth partial waves are
    built to be band-limited, so filtering them to the grid's Nyquist
    wave-vector is nearly free and removes most of the egg-box.

Per-family defaults
-------------------
:attr:`FamilySpec.default_options` is where a family says what it does when
the user does not.  :meth:`FamilySpec.resolved_options` is the single merge
(defaults first, the user's basis dict on top), consulted by
:func:`build_valence_hamiltonian` and by the dry run, so a default is visible
in one place, a new family declares its own, and writing the option explicitly
always wins -- ``basis={"name": "PAW-LCAO", "filter": False}`` is exactly the
unfiltered calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

#: The canonical name of the Troullier-Martins family (the bundled library).
DEFAULT_FAMILY = "ncpp"

#: Basis options every family accepts: the size hierarchy built on the
#: pseudo-orbitals, the library directory and the Fourier filter
#: (:mod:`mandacaru.basis.filtering`) that removes from each radial function
#: the wave-vectors the real-space grid cannot represent.
COMMON_OPTIONS = ("size", "split_norm", "tail_norm", "directory", "filter")


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
        ``get(symbol, directory=None) -> PseudoPotential`` -- the (cached)
        library loader.
    build : callable
        ``build(atoms, grid, h, charge, spin, options, kinetic, *,
        active_orbitals=None, active_selection="energy",
        active_threshold=None)`` returning the driver
        5-tuple ``(hamiltonian, num_particles, n_spatial_orbitals,
        integration_profile, context)`` -- exactly what
        :func:`~mandacaru.algorithms._hamiltonian_from_atoms.build_basis_hamiltonian`
        returns for the all-electron path.  The two keyword arguments are always
        passed, so a builder must accept them (``**kwargs`` forwarded to
        :func:`build_valence_hamiltonian` is enough, and is what every built-in
        family does).  They arrived with virtual-orbital truncation: a
        valence-only basis has no core to freeze but plenty of virtual orbitals
        to drop, so unlike ``frozen_core`` this is a request a pseudopotential
        family has to be able to honour.
    norm_conserving : bool
        Whether the family's projectors leave the basis overlap untouched
        (``True`` for TM/ONCVPSP; PAW-LCAO carries an overlap correction).
    aliases : tuple of str
        Alternative names resolving to this family.
    options : tuple of str
        The keys a ``{"name": <family>, ...}`` basis dict may carry for this
        family (default :data:`COMMON_OPTIONS`); anything else is refused
        before any integral is computed.
    default_options : dict
        Option values this family uses when the user does not say
        (``{"filter": True}`` for PAW-LCAO / UPAW-LCAO, whose smooth partial waves are
        built to be band-limited and lose little by being filtered, while
        NCPP / ONCVPSP leave it off).  Declaring the default **here** rather
        than in the builder is what keeps it discoverable: one place says what
        every family does, a new family declares its own, and
        :meth:`resolved_options` is the only merge.  Every key must be in
        :attr:`options`.
    """

    name: str
    description: str
    generate: Callable
    get: Callable
    build: Callable
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
        ``basis="PAW-LCAO"`` and ``basis={"name": "PAW-LCAO", "filter": True}`` are the
        same calculation and ``{"name": "PAW-LCAO", "filter": False}`` is exactly
        the unfiltered one.
        """
        merged = dict(self.default_options)
        merged.update(options or {})
        return merged


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

    Case-insensitive, aliases accepted (``"PAW-LCAO"``, ``"oncv"``, ``"NCPP-TM"``);
    anything that is not a registered family name -- ``"HAO"``, ``"cc-pVTZ"``,
    a per-element mapping -- gives ``None``.  This is how
    :func:`~mandacaru.algorithms._hamiltonian_from_atoms.resolve_basis` tells a
    pseudopotential basis from an all-electron one.
    """
    if isinstance(name, FamilySpec):
        return name
    if not isinstance(name, str):
        return None
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


# --------------------------------------------------------------------------- #
# Family "ncpp": norm-conserving Troullier-Martins, Kleinman-Bylander form.
# --------------------------------------------------------------------------- #

def _generate_tm(symbol, **options):
    from .generation import generate_pseudopotential
    return generate_pseudopotential(symbol, **options)


def _get_tm(symbol, directory=None):
    """Library loader; refuses a file that belongs to another family."""
    from .io import get_pseudopotential
    pp = get_pseudopotential(symbol, directory)
    if canonical_family_name(getattr(pp, "family", DEFAULT_FAMILY)) != "ncpp":
        location = directory or "the bundled library"
        raise ValueError(
            f"the pseudopotential for {symbol!r} in {location} belongs to "
            f"family {pp.family!r}, not 'ncpp'")
    return pp


def pseudo_basis_arguments(family, options, *, confinement=None,
                           polarization=None) -> dict:
    """Keyword arguments of :func:`~.orbitals.pseudo_basis` for ``options``.

    The one place a basis dict is turned into a construction, shared by the
    Hamiltonian builder and the dry run so the two cannot count different
    bases.  ``confinement`` / ``polarization`` (optional dicts) receive what
    the hooks resolve -- each orbital's cutoff radius, the polarization
    shell's -- for the run log.

    The split-valence scheme is GPAW's unless ``split_norm`` is written
    (:func:`~mandacaru.basis.multizeta.resolve_split_scheme`).
    """
    from ..basis.multizeta import resolve_split_scheme

    spec = resolve_family(family)
    split_norm, tail_norms = resolve_split_scheme(options.get("split_norm"),
                                                  options.get("tail_norm"))
    arguments = {"size": options.get("size", "SZ"), "split_norm": split_norm,
                 "tail_norms": tail_norms}
    if "energy_shift" in spec.options:
        from .confinement import first_zeta_factory, polarization_factory
        arguments["first_zeta"] = first_zeta_factory(
            options.get("energy_shift"), confinement,
            options.get("confinement"))
        arguments["polarization_shape"] = polarization_factory(
            options.get("polarization"), options.get("energy_shift"),
            polarization, options.get("confinement"))
    return arguments


def build_valence_hamiltonian(atoms, grid, h, charge, spin, options, kinetic, *,
                              active_orbitals=None,
                              active_selection: str = "energy",
                              active_threshold=None,
                              family: str, load, projectors, coupling,
                              overlap=None, spin_orbit=None,
                              integrals_class=None,
                              potentials_keyword: str = "pseudos"):
    """The driver 5-tuple of a pseudopotential family -- the part every family
    shares.

    A family differs only in how a dataset is loaded (``load(symbol,
    directory)``), which projectors it samples (``projectors(symbols, positions,
    potentials, options)``), their coupling blocks (``coupling(projectors,
    symbols, potentials)``), an optional overlap correction (``overlap``, same
    signature; PAW-LCAO) and the integral class (``integrals_class``, default
    :class:`~mandacaru.core.MolecularIntegrals`; ``potentials_keyword`` names
    its datasets argument).  The basis (with its ``size`` hierarchy and its
    optional ``filter``), the grid, the electron count, the spin state and the
    returned ``context`` are built here once.
    """
    from ..algorithms._hamiltonian_from_atoms import (
        DEFAULT_KINETIC, _num_particles, _warn_unresolved, coherent_positions,
        grid_from_cell, resolve_num_unpaired)
    from ..basis.filtering import filter_cutoff
    from ..core import MolecularIntegrals
    from .orbitals import pseudo_basis, valence_electrons

    # The family's own defaults, with the caller's options on top.  Doing it
    # here (and not only in the driver's dispatcher) means a family built
    # directly -- `build_paw(atoms, ...)` in a script or a test -- gets the
    # same basis the calculator would give it.  Merging twice is a no-op.
    options = resolve_family(family).resolved_options(options)

    directory = options.get("directory")
    symbols = atoms.get_chemical_symbols()
    positions = coherent_positions(atoms)
    potentials = {symbol: load(symbol, directory) for symbol in set(symbols)}

    n_el = int(round(valence_electrons(symbols, potentials))) - int(charge)
    # The grid comes first now: ``filter="auto"`` ties its cutoff to the
    # *realized* spacing (the coarsest axis -- an anisotropic grid can only
    # represent what its worst direction can), which is known only once the
    # grid exists.  Nothing else about the grid depends on the basis.
    g = (grid if grid is not None
         else grid_from_cell(atoms, h, center=positions.mean(axis=0)))
    k_c = filter_cutoff(options.get("filter"),
                        max(g.dx, g.dy, g.dz))

    confinement: dict = {}
    polarization: dict = {}
    basis_fns, atom_of_orbital = pseudo_basis(
        symbols, positions, potentials, filter_cutoff=k_c,
        **pseudo_basis_arguments(family, options, confinement=confinement,
                                 polarization=polarization))
    kb = projectors(symbols, positions, potentials, options)
    coupling_blocks = coupling(kb, symbols, potentials)
    overlap_blocks = (None if overlap is None
                      else overlap(kb, symbols, potentials))
    # Spin-orbit blocks, keyed by (atom, l) rather than (atom, l, m): the
    # term couples different m, so it is not part of the block-diagonal D.
    spin_orbit_blocks = (None if spin_orbit is None
                         else spin_orbit(kb, symbols, potentials))
    nuclei = [(potentials[symbol].valence_charge, position)
              for symbol, position in zip(symbols, positions)]

    n_unpaired = resolve_num_unpaired(atoms, spin, n_el)
    num_particles = _num_particles(n_el, n_unpaired, family.upper())
    integrals = (integrals_class or MolecularIntegrals)(
        nuclei, basis_fns, g, softening=0.0,
        kb_projectors=kb, nonlocal_coupling=coupling_blocks,
        nonlocal_overlap=overlap_blocks,
        spin_orbit_coupling=spin_orbit_blocks,
        kinetic=kinetic or DEFAULT_KINETIC["pseudopotentials"],
        **{potentials_keyword: [potentials[s] for s in symbols]})
    hamiltonian = integrals.molecular_hamiltonian(
        mo_basis=True, n_electrons=n_el, num_particles=num_particles,
        active_orbitals=active_orbitals, active_selection=active_selection,
        active_threshold=active_threshold)
    _warn_unresolved(integrals, basis_fns, h)

    # A valence-only basis has no core to freeze, but it does have virtual
    # orbitals to delete, and the selector may also fold valence occupied
    # orbitals into a mean field when the register is named as a count.  Either
    # way the active electron count comes from the partition, not the request.
    space = integrals.active_space
    frozen = tuple(space.frozen) if space is not None else ()
    if frozen:
        num_particles = _num_particles(n_el - 2 * len(frozen), n_unpaired,
                                       family.upper())
    context = {"integrals": integrals, "atom_of_orbital": atom_of_orbital,
               "frozen": frozen, "n_electrons": n_el,
               "pseudopotentials": potentials, "kb_projectors": kb,
               "nonlocal_coupling": coupling_blocks, "family": family,
               "filter_cutoff": k_c, "options": dict(options),
               "confinement": confinement, "polarization": polarization,
               "active_space": space}
    if space is not None:
        context["active"] = tuple(space.active)
        context["deleted"] = tuple(space.deleted)
    if overlap_blocks is not None:
        context["nonlocal_overlap"] = overlap_blocks
    n_active = space.n_active if space is not None else len(basis_fns)
    return (hamiltonian, num_particles, n_active,
            integrals.integration_profile(), context)


def _build_tm(atoms, grid, h, charge, spin, options, kinetic=None, **active):
    r"""Valence-only Hamiltonian from Troullier-Martins pseudopotentials.

    The core electrons are gone entirely: the basis is the set of valence
    pseudo-atomic orbitals, the external potential is the smooth local
    channel, and the Kleinman-Bylander projectors supply the nonlocal part
    through the general separable form with the :math:`1\times1` blocks
    :math:`[E^{KB}_l]` (:func:`~.orbitals.kb_coupling_blocks`) and no overlap
    correction.  The "nuclei" carry the *ionic* charges, so the constant term
    is the ion-ion repulsion.
    """
    from .orbitals import kb_coupling_blocks, kb_projectors

    return build_valence_hamiltonian(
        atoms, grid, h, charge, spin, options, kinetic, family="ncpp",
        load=_get_tm, **active,
        projectors=lambda symbols, positions, potentials, _options:
            kb_projectors(symbols, positions, potentials),
        coupling=lambda projectors, _symbols, _potentials:
            kb_coupling_blocks(projectors))


NCPP_FAMILY = register_family(FamilySpec(
    name="ncpp",
    description="norm-conserving Troullier-Martins, Kleinman-Bylander "
                "separable form (one projector per channel)",
    generate=_generate_tm,
    get=_get_tm,
    build=_build_tm,
    norm_conserving=True,
    aliases=("tm", "ncpp-tm"),
))
