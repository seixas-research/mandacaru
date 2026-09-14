# -*- coding: utf-8 -*-
# file: experimental/pseudopotentials/families.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Pseudopotential **families** and their registry -- experimental.

A family is everything that distinguishes one kind of pseudopotential from
another once the integral engine is reached: how a potential is *generated*
for an element, how it is *loaded* from a library, and how a molecule's
valence-only Hamiltonian is *built* from it -- the basis, the local
potential, the projectors and, in the general separable form of
:meth:`carcara.core.hamiltonian.MolecularIntegrals.kb_nonlocal`,

.. math::

    H^{NL} = C\,D\,C^\dagger, \qquad C_{\mu p} = \langle\phi_\mu|\chi_p\rangle ,

the block-diagonal coupling matrix :math:`D` (one block per ``(atom, l, m)``)
and, for families whose projectors also carry an overlap correction (PAW), the
block matrix :math:`Q` that turns the basis overlap into
:math:`S + C\,Q\,C^\dagger`.

The registry :data:`PSEUDO_FAMILIES` maps a family name to its
:class:`FamilySpec`.  The drivers only ever see the spec: ``pseudopotentials=``
on any driver accepts ``True`` (the default family), a family name or a
``{"family": name, ...options}`` dict, and
:func:`carcara.algorithms._hamiltonian_from_atoms._pseudopotential_hamiltonian`
dispatches to ``spec.build``.  A new family is added with
:func:`register_family` and never needs the driver edited.

Registered today:

``"tm"`` (aliases ``"ncpp"``, ``"ncpp-tm"``)
    Norm-conserving **Troullier-Martins** potentials in Kleinman-Bylander
    separable form -- one projector per channel, :math:`D = \mathrm{diag}(E^{KB})`,
    no overlap correction.  This is what ``pseudopotentials=True`` has always
    meant.

``"oncvpsp"`` (alias ``"oncv"``; :mod:`.oncv`)
    Hamann's **optimized norm-conserving Vanderbilt** potentials -- two
    projectors per channel with a :math:`2\times2` coupling block, a
    polynomial local potential that is not a channel, no overlap correction.
    Registered when :mod:`.oncv` is imported (the package does so).

``"paw"`` (:mod:`.paw`)
    Bloechl's **projector augmented-wave** datasets -- two partial waves and
    projectors per channel, a :math:`2\times2` coupling block **and** a
    :math:`2\times2` overlap-correction block :math:`q` (``norm_conserving
    = False``), monopole compensation charges in the two-body tensor and a
    frozen one-center constant.  Registered when :mod:`.paw` is imported
    (the package does so).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

#: The family ``pseudopotentials=True`` selects.
DEFAULT_FAMILY = "tm"


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
        ``build(atoms, grid, h, charge, spin, options, kinetic)`` returning the
        driver 5-tuple ``(hamiltonian, num_particles, n_spatial_orbitals,
        integration_profile, context)`` -- exactly what
        :func:`~carcara.algorithms._hamiltonian_from_atoms.build_basis_hamiltonian`
        returns for the all-electron path.
    norm_conserving : bool
        Whether the family's projectors leave the basis overlap untouched
        (``True`` for TM/ONCVPSP; PAW carries an overlap correction).
    aliases : tuple of str
        Alternative names resolving to this family.
    """

    name: str
    description: str
    generate: Callable
    get: Callable
    build: Callable
    norm_conserving: bool = True
    aliases: tuple = field(default_factory=tuple)


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


def resolve_family(name=None) -> FamilySpec:
    """The :class:`FamilySpec` for ``name`` (case-insensitive; aliases accepted).

    ``None`` and ``True`` select :data:`DEFAULT_FAMILY`; an unknown name raises
    ``ValueError`` listing the registered families.
    """
    if name is None or name is True:
        name = DEFAULT_FAMILY
    if isinstance(name, FamilySpec):
        return name
    if not isinstance(name, str):
        raise TypeError(
            f"a pseudopotential family is named by a string, got {name!r}")
    canonical = _ALIASES.get(_key(name))
    if canonical is None:
        listing = ", ".join(
            f"{spec.name!r}" + (f" (aliases: {', '.join(map(repr, spec.aliases))})"
                                if spec.aliases else "")
            for spec in PSEUDO_FAMILIES.values())
        raise ValueError(
            f"unknown pseudopotential family {name!r}; registered families: "
            f"{listing}")
    return PSEUDO_FAMILIES[canonical]


def normalize_pseudopotentials(spec) -> dict:
    """Normalize a driver's ``pseudopotentials=`` argument to an options dict.

    Accepts ``True`` (the default family), a family name string
    (``"tm"``, ``"ncpp"``, ``"ncpp-tm"``, ...) or a dict with an optional
    ``"family"`` key plus the family's options (``directory``, ``size``, ...).
    Returns a **new** dict whose ``"family"`` is the canonical name, so the
    rest of the pipeline never sees an alias.  ``False``/``None`` raise --
    callers only reach this when pseudopotentials were requested.
    """
    if spec is None or spec is False:
        raise ValueError("pseudopotentials were not requested")
    if spec is True:
        options = {}
    elif isinstance(spec, str):
        options = {"family": spec}
    elif isinstance(spec, dict):
        options = dict(spec)
    else:
        raise TypeError(
            "pseudopotentials must be True, a family name such as 'tm', or a "
            f"dict like {{'family': 'tm', 'directory': ...}}; got {spec!r}")
    family = resolve_family(options.get("family"))
    options["family"] = family.name
    return options


# --------------------------------------------------------------------------- #
# Family "tm": norm-conserving Troullier-Martins, Kleinman-Bylander form.
# --------------------------------------------------------------------------- #

def _generate_tm(symbol, **options):
    from .generation import generate_pseudopotential
    return generate_pseudopotential(symbol, **options)


def _get_tm(symbol, directory=None):
    """Library loader; refuses a file that belongs to another family."""
    from .io import get_pseudopotential
    pp = get_pseudopotential(symbol, directory)
    if _key(getattr(pp, "family", DEFAULT_FAMILY)) != "tm":
        raise ValueError(
            f"the pseudopotential for {symbol!r} in {directory or 'the bundled '
            'library'} belongs to family {pp.family!r}, not 'tm'")
    return pp


def _build_tm(atoms, grid, h, charge, spin, options, kinetic=None):
    r"""Valence-only Hamiltonian from Troullier-Martins pseudopotentials.

    The core electrons are gone entirely: the basis is the set of valence
    pseudo-atomic orbitals, the external potential is the smooth local
    channel, and the Kleinman-Bylander projectors supply the nonlocal part
    through the general separable form with the :math:`1\times1` blocks
    :math:`[E^{KB}_l]` (:func:`~.orbitals.kb_coupling_blocks`) and no overlap
    correction.  The "nuclei" carry the *ionic* charges, so the constant term
    is the ion-ion repulsion.
    """
    from ...algorithms._hamiltonian_from_atoms import (
        DEFAULT_KINETIC, _num_particles, _warn_unresolved, coherent_positions,
        grid_from_cell, resolve_num_unpaired)
    from ...core import MolecularIntegrals
    from .orbitals import (kb_coupling_blocks, kb_projectors, pseudo_basis,
                           valence_electrons)

    directory = options.get("directory")
    symbols = atoms.get_chemical_symbols()
    positions = coherent_positions(atoms)
    potentials = {symbol: _get_tm(symbol, directory) for symbol in set(symbols)}

    basis_fns, atom_of_orbital = pseudo_basis(
        symbols, positions, potentials, size=options.get("size", "SZ"),
        split_norm=options.get("split_norm"))
    projectors = kb_projectors(symbols, positions, potentials)
    nuclei = [(potentials[symbol].valence_charge, position)
              for symbol, position in zip(symbols, positions)]

    n_el = int(round(valence_electrons(symbols, potentials))) - int(charge)
    g = (grid if grid is not None
         else grid_from_cell(atoms, h, center=positions.mean(axis=0)))

    n_unpaired = resolve_num_unpaired(atoms, spin, n_el)
    num_particles = _num_particles(n_el, n_unpaired, "PP")
    integrals = MolecularIntegrals(
        nuclei, basis_fns, g, softening=0.0,
        pseudopotentials=[potentials[s] for s in symbols],
        kb_projectors=projectors,
        nonlocal_coupling=kb_coupling_blocks(projectors),
        nonlocal_overlap=None,
        kinetic=kinetic or DEFAULT_KINETIC["pseudopotentials"])
    hamiltonian = integrals.molecular_hamiltonian(mo_basis=True,
                                                  n_electrons=n_el,
                                                  num_particles=num_particles)
    _warn_unresolved(integrals, basis_fns, h)

    context = {"integrals": integrals, "atom_of_orbital": atom_of_orbital,
               "frozen": (), "n_electrons": n_el,
               "pseudopotentials": potentials, "kb_projectors": projectors,
               "family": "tm"}
    return (hamiltonian, num_particles, len(basis_fns),
            integrals.integration_profile(), context)


TM_FAMILY = register_family(FamilySpec(
    name="tm",
    description="norm-conserving Troullier-Martins, Kleinman-Bylander "
                "separable form (one projector per channel)",
    generate=_generate_tm,
    get=_get_tm,
    build=_build_tm,
    norm_conserving=True,
    aliases=("ncpp", "ncpp-tm"),
))
