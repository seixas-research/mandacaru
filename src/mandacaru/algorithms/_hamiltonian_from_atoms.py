# -*- coding: utf-8 -*-
# file: algorithms/_hamiltonian_from_atoms.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Build a molecular qubit Hamiltonian from an ASE ``Atoms`` geometry.

Shared by the ASE-calculator paths of :class:`~mandacaru.algorithms.vqe.VQE` and
:class:`~mandacaru.algorithms.adapt_vqe.ADAPTVQE` so both drive quantum simulations
the same way: a geometry (elements, positions, unit cell) plus a basis name is
turned into an RHF molecular-orbital :class:`~mandacaru.core.mapping.Fermion`
Hamiltonian, and the real-space integration is profiled (time / cores / memory)
along the way.

**Placement is irrelevant.**  mandacaru solves an *isolated-molecule* (Gamma-point,
open-boundary) electronic-structure problem: the cell only sets the size of the
real-space box.  The box is centered on the molecule -- and, when the geometry is
periodic (``pbc``), the molecule is first made whole under the minimum-image
convention (ASE :func:`~ase.geometry.find_mic`) -- so it does not matter *where*
in the cell the atoms sit, nor whether the molecule straddles a cell face.  This
is a real-space grid convenience, not a periodic (Bloch / k-point) treatment of
the electrons; see the ``kpts`` argument of the drivers.
"""

from __future__ import annotations

import numpy as np


def coherent_positions(atoms) -> np.ndarray:
    """Angstrom positions with the molecule made whole (minimum-image aware).

    With periodic boundary conditions a molecule can straddle a cell face and
    come back as two far-apart fragments; ASE's minimum-image convention unwraps
    every atom relative to the first, giving one connected fragment whose centroid
    is meaningful.  Without ``pbc`` (or without a cell) the positions are returned
    unchanged.
    """
    pos = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(atoms.get_cell(), dtype=float)
    pbc = np.asarray(atoms.get_pbc())
    if pos.shape[0] > 1 and pbc.any() and np.any(cell):
        from ase.geometry import find_mic
        disps, _ = find_mic(pos - pos[0], cell, pbc)
        pos = pos[0] + disps
    return pos


def grid_from_cell(atoms, h: float, center=None):
    """Build the real-space integration grid from the ASE ``atoms.cell``.

    The cell's lattice vectors fix the extent (and shape) of the box and ``h``
    (Angstrom) sets the uniform node spacing.  The box is **centered on the
    molecule** (``center``, defaulting to the minimum-image centroid) rather than
    on the cell, so wherever the atoms are placed the orbitals stay inside the
    grid.  The same grid feeds both the one- and two-body integral kernels.

    A unit cell is **required** (or pass an explicit ``grid=``); raises
    ``ValueError`` otherwise.
    """
    from ..integrals import Grid

    cell = np.asarray(atoms.get_cell(), dtype=float)      # Angstrom (ASE)
    if not np.any(cell):
        raise ValueError(missing_cell_message(h))
    if center is None:
        center = coherent_positions(atoms).mean(axis=0)   # center on the molecule
    return Grid(center=center, box_size=0.0, h=h, units="angstrom", cell=cell)


def missing_cell_message(h=None) -> str:
    """The error text for a geometry without a unit cell.

    The real-space box is the geometry's cell -- nothing else pads it -- so
    the cell must be defined **in the geometry**: ``atoms.center(vacuum=3.0)``
    on the ASE side, ``atoms.cell = [Lx, Ly, Lz]``, or an extended-XYZ file
    whose header carries a ``Lattice``.
    """
    spacing = f" at h = {h:g} Angstrom" if h is not None else ""
    return (
        "the geometry has no unit cell, and the real-space box is built from "
        f"the cell{spacing}.  Define the cell in the geometry: "
        "atoms.center(vacuum=3.0) pads the molecule with 3 Angstrom of empty "
        "space on every side, atoms.cell = [Lx, Ly, Lz] sets an explicit box, "
        "and an extended-XYZ file carries it in its Lattice=\"...\" header.  "
        "Alternatively pass an explicit grid=.")


#: The ``name`` :func:`resolve_basis` returns for a per-element mapping.
PER_ELEMENT = "per-element"
#: Key of a per-element mapping that supplies the basis of unlisted elements.
DEFAULT_ELEMENT_KEY = "*"


def is_per_element_basis(basis) -> bool:
    """True for a ``{"O": ..., "H": ..., "*": ...}`` mapping of element -> basis.

    Distinguished from a single-family dict by the absence of a ``"name"`` key:
    every key must be a chemical symbol (or ``"*"`` for the default).
    """
    if not isinstance(basis, dict) or "name" in basis or not basis:
        return False
    from ase.data import atomic_numbers
    return all(key == DEFAULT_ELEMENT_KEY
               or (isinstance(key, str) and key.capitalize() in atomic_numbers)
               for key in basis)


def per_element_basis(basis, symbols) -> dict:
    """``{symbol: (name, options)}`` for every symbol, from a per-element mapping.

    An element with no entry takes the ``"*"`` entry; without one it is an
    error -- a silently defaulted basis on one atom would be invisible in the
    result.  Plane waves are not atom-centered and cannot be mixed in.
    """
    if not is_per_element_basis(basis):
        raise TypeError("expected a per-element basis mapping")
    table = {(k if k == DEFAULT_ELEMENT_KEY else k.capitalize()): v
             for k, v in basis.items()}
    resolved = {}
    for symbol in dict.fromkeys(symbols):
        spec = table.get(symbol.capitalize(), table.get(DEFAULT_ELEMENT_KEY))
        if spec is None:
            raise ValueError(
                f"no basis given for element {symbol!r} in the per-element "
                f"mapping {sorted(k for k in table if k != DEFAULT_ELEMENT_KEY)}; "
                f"add it, or a '{DEFAULT_ELEMENT_KEY}' default entry")
        name, options = resolve_basis(spec)
        if name == PER_ELEMENT:
            raise ValueError("per-element mappings cannot be nested")
        if _is_plane_wave(name):
            raise ValueError(
                "the plane-wave basis is not atom-centered and cannot be "
                "assigned to a single element")
        resolved[symbol.capitalize()] = (name, options)
    return resolved


def _is_plane_wave(name) -> bool:
    return isinstance(name, str) and \
        name.upper().replace("-", "").replace(" ", "") in ("PW", "PLANEWAVE")


def resolve_basis(basis):
    """Normalize a ``basis`` spec to ``(name, options)``.

    Accepts a plain method name (``"HAO"``) or a dict giving the name plus that
    family's options: ``{"name": "HAO"}``, ``{"name": "NAO", "energy_shift": 0.03}``,
    ``{"name": "NAO", "size": "DZP"}`` (multiple-zeta and polarized -- see
    :mod:`mandacaru.basis.multizeta`), ``{"name": "NAO-AE", "tier": 1,
    "onset": 3.0}`` (all-electron NAOs, :mod:`mandacaru.basis.nao_ae`),
    ``{"name": "GTO", "n_gaussians": 3}``,
    ``{"name": "6-31G(d)"}``, the plane-wave basis
    ``{"name": "PW", "energy_cutoff": 300}``, or a **pseudopotential family**
    -- ``"NCPP"`` (aliases ``"TM"``, ``"NCPP-TM"``), ``"ONCVPSP"`` (alias
    ``"ONCV"``), ``"PAW"`` -- with the same size hierarchy as its options,
    ``{"name": "PAW", "size": "DZP"}`` (see :func:`pseudopotential_family`).
    Returns the name string and a dict of the remaining keyword options.

    A **per-element mapping** -- a dict keyed by chemical symbols (plus an
    optional ``"*"`` default), each value itself a basis spec, e.g.
    ``{"O": {"name": "NAO", "size": "DZP"}, "H": "6-31G", "*": "HAO"}`` --
    returns ``(PER_ELEMENT, mapping)``; see :func:`per_element_basis`.
    """
    if isinstance(basis, str):
        _check_retired_basis_name(basis)
        return basis, {}
    if isinstance(basis, dict):
        if is_per_element_basis(basis):
            return PER_ELEMENT, dict(basis)
        options = dict(basis)
        name = options.pop("name", None)
        if name is None:
            raise ValueError(
                "a basis dict must include a 'name' key, e.g. {'name': 'HAO'} "
                "or {'name': 'PW', 'energy_cutoff': 300}, or be a per-element "
                "mapping such as {'O': 'HAO', 'H': '6-31G'}")
        _check_retired_basis_name(name)
        return name, options
    raise TypeError(
        "basis must be a name string, a dict like {'name': 'HAO', ...}, or a "
        "per-element mapping like {'O': 'HAO', 'H': '6-31G'}")


# Noble-gas core: (highest Z of the row, core electrons of that row's atoms).
_NOBLE_CORE_THRESHOLDS = ((2, 0), (10, 2), (18, 10), (36, 18),
                          (54, 36), (86, 54), (118, 86))


def core_electrons(atomic_number: int) -> int:
    """Number of noble-gas core electrons for an atom of ``atomic_number``.

    The chemical (frozen) core is the electron count of the preceding noble gas:
    ``0`` for H/He, ``2`` (He) for Li--Ne, ``10`` (Ne) for Na--Ar, ``18`` (Ar) for
    K--Kr, and so on.
    """
    Z = int(atomic_number)
    for zmax, core in _NOBLE_CORE_THRESHOLDS:
        if Z <= zmax:
            return core
    return 86


def _auto_frozen_count(frozen_core, numbers) -> int:
    """Number of lowest MOs to freeze from a ``frozen_core`` spec (no explicit list).

    ``False``/``None``/``0`` -> freeze nothing; ``True``/``"auto"`` -> the chemical
    (noble-gas) core, ``sum(core_electrons(Z)) // 2`` spatial orbitals; an integer
    -> that many lowest MOs.
    """
    if frozen_core is None or frozen_core is False:
        return 0
    if frozen_core is True or (isinstance(frozen_core, str)
                              and frozen_core.strip().lower() == "auto"):
        return sum(core_electrons(int(z)) for z in numbers) // 2
    if isinstance(frozen_core, (int, np.integer)):
        n = int(frozen_core)
        if n < 0:
            raise ValueError(f"frozen_core count must be >= 0, got {n}")
        return n
    raise ValueError(
        f"unknown frozen_core spec {frozen_core!r}; use False, True/'auto', or an "
        "integer number of core spatial orbitals")


def resolve_frozen_core(frozen_core, frozen_orbitals, numbers, n_el: int,
                        n_orbitals: int, n_doubly=None) -> list[int]:
    """Resolve the frozen-core spec to a sorted list of frozen spatial-MO indices.

    ``frozen_orbitals`` (an explicit list of spatial MO indices) takes precedence;
    otherwise the lowest ``_auto_frozen_count(frozen_core, numbers)`` MOs are
    frozen.  Every frozen orbital must be doubly occupied in the reference
    (index ``< n_doubly``, the number of doubly occupied orbitals -- ``n_beta``,
    defaulting to ``n_el // 2``), since the frozen-core approximation removes
    doubly occupied core orbitals.
    """
    n_occ = int(n_doubly) if n_doubly is not None else n_el // 2
    if frozen_orbitals is not None:
        frozen = sorted({int(i) for i in frozen_orbitals})
    else:
        frozen = list(range(_auto_frozen_count(frozen_core, numbers)))
    for i in frozen:
        if not (0 <= i < n_orbitals):
            raise ValueError(
                f"frozen orbital index {i} is out of range [0, {n_orbitals})")
        if i >= n_occ:
            raise ValueError(
                f"cannot freeze spatial orbital {i}: only the {n_occ} doubly "
                f"occupied orbitals (indices 0..{n_occ - 1}) may be frozen")
    return frozen


def resolve_num_unpaired(atoms, spin, n_el: int) -> int:
    """Number of unpaired electrons ``2S = n_alpha - n_beta`` for the reference.

    The **initial spin state** of the molecule is read primarily from the ASE
    ``Atoms`` initial magnetic moments (``Atoms(..., magmoms=...)`` /
    :meth:`ase.Atoms.set_initial_magnetic_moments`): their rounded total is the
    number of unpaired electrons -- e.g. a triplet O₂ with ``magmoms=[1, 1]``
    gives ``2``.  When no magnetic moments are set, falls back to the boolean
    ``spin`` flag.  An odd electron count always has at least one unpaired
    electron, so with no magnetic moments it is a doublet whatever ``spin``
    says; for an even count ``spin`` keeps the closed-shell singlet (``0``) --
    a high-spin even-electron state is requested through the magnetic moments.
    """
    total = 0.0
    if atoms is not None:
        try:
            total = float(np.sum(atoms.get_initial_magnetic_moments()))
        except Exception:
            total = 0.0
    if abs(total) > 1e-8:
        return int(round(abs(total)))
    return n_el % 2


def _num_particles(n_el: int, n_unpaired: int, basis) -> tuple[int, int]:
    """Reference occupation ``(n_alpha, n_beta)`` for ``n_unpaired = n_alpha - n_beta``.

    The requested spin state must share the parity of the electron count (an
    odd count is a doublet, quartet, ...; an even count a singlet, triplet,
    ...) and cannot exceed it.  Odd counts are built in the open-shell
    (UHF natural-orbital) basis -- see
    :meth:`~mandacaru.core.hamiltonian.MolecularIntegrals.molecular_hamiltonian`.
    """
    if n_el < 0:
        raise ValueError(f"negative electron count {n_el}")
    if n_unpaired < 0 or n_unpaired > n_el or (n_unpaired - n_el) % 2 != 0:
        parity = "odd" if n_el % 2 else "even"
        raise ValueError(
            f"the requested spin state (n_unpaired={n_unpaired}) is incompatible "
            f"with {n_el} electrons: n_unpaired must be {parity} and in "
            f"[0, {n_el}] (basis {basis!r}).")
    return ((n_el + n_unpaired) // 2, (n_el - n_unpaired) // 2)


#: Basis names of the retired ``pseudopotentials`` driver argument; refused
#: with a pointer to the family names rather than silently aliased to one.
RETIRED_PSEUDO_BASIS_NAMES = ("PP", "PSEUDO")


def _basis_key(name) -> str:
    return str(name).upper().replace("-", "").replace(" ", "")


def _check_retired_basis_name(name):
    if isinstance(name, str) and _basis_key(name) in RETIRED_PSEUDO_BASIS_NAMES:
        raise ValueError(
            f"basis {name!r} is no longer a basis name: the pseudopotential "
            "family is now selected through the basis itself -- "
            "basis='NCPP' (Troullier-Martins, aliases 'TM' / 'NCPP-TM'), "
            "basis='ONCVPSP' (alias 'ONCV') or basis='PAW', with the size "
            "hierarchy as options: basis={'name': 'PAW', 'size': 'DZP'}")


def pseudopotential_family(name):
    """The :class:`~mandacaru.pseudopotentials.families.FamilySpec` a basis
    name selects, or ``None`` for an all-electron (or plane-wave) family.

    The registry :data:`~mandacaru.pseudopotentials.families.PSEUDO_FAMILIES`
    is the single source of truth: ``"NCPP"`` / ``"TM"`` / ``"NCPP-TM"``,
    ``"ONCVPSP"`` / ``"ONCV"`` and ``"PAW"`` today, plus anything added with
    :func:`~mandacaru.pseudopotentials.families.register_family`.  Names are
    case-insensitive.
    """
    if not isinstance(name, str):
        return None
    _check_retired_basis_name(name)
    from ..pseudopotentials.families import lookup_family
    return lookup_family(name)


#: Pseudo-basis options that describe the *construction* (one recipe for the
#: whole basis): a per-element basis may write them on any element, but every
#: element that does must say the same thing.
SHARED_PSEUDO_OPTIONS = ("tail_norm", "polarization", "confinement")


def resolve_pseudo_basis(name, options, symbols):
    """``(family, options)`` for a resolved basis spec, ``(None, options)``
    when it is all-electron.

    A single family name passes its options through (``size``,
    ``split_norm``, ``directory``, ... -- validated against
    ``family.options`` by :func:`_pseudopotential_hamiltonian`).  A
    **per-element mapping** must name one pseudopotential family for every
    element (the ``"*"`` default counts): the first zeta of each atom is the
    potential's own orbital, so only the size hierarchy can differ from atom
    to atom -- the entries become ``options["size"] = {symbol: size}``.
    Mixing a pseudopotential family with an all-electron family across
    elements, or two different pseudopotential families, raises.
    """
    if name != PER_ELEMENT:
        return pseudopotential_family(name), dict(options)

    resolved = per_element_basis(options, symbols)      # {symbol: (name, opts)}
    families = {symbol: pseudopotential_family(sub_name)
                for symbol, (sub_name, _o) in resolved.items()}
    pseudo = {sym: fam for sym, fam in families.items() if fam is not None}
    if not pseudo:
        return None, dict(options)
    if len(pseudo) != len(families):
        all_electron = sorted(sym for sym, fam in families.items() if fam is None)
        raise ValueError(
            "a per-element basis cannot mix a pseudopotential family with an "
            f"all-electron family: {sorted(pseudo)} use "
            f"{sorted({f.label for f in pseudo.values()})} while "
            f"{all_electron} use "
            f"{sorted({resolved[s][0] for s in all_electron})}.  A "
            "pseudopotential replaces the core and the -Z/r potential of its "
            "atom, so every element must carry one (the size may still differ "
            "per element).")
    labels = {fam.label for fam in pseudo.values()}
    if len(labels) > 1:
        raise ValueError(
            "a per-element basis must use one pseudopotential family for "
            f"every element, got {sorted(labels)}: the families differ in "
            "their projectors and overlap treatment and cannot share one "
            "Hamiltonian")
    family = next(iter(pseudo.values()))
    sizes, splits, merged = {}, {}, {}
    shifts = {}
    shared = {}
    filters = {}
    for symbol, (_sub_name, sub_options) in resolved.items():
        extra = {k: v for k, v in sub_options.items()
                 if k not in ("size", "split_norm", "filter",
                              "energy_shift") + SHARED_PSEUDO_OPTIONS}
        if extra:
            raise ValueError(
                f"per-element basis options {extra!r} for {symbol!r} cannot "
                f"differ per element with the {family.label} family; only "
                "'size', 'split_norm', 'energy_shift' and 'filter' may be "
                "written per element "
                "(give the other options in a single {'name': ..., ...} basis "
                "dict)")
        sizes[symbol] = sub_options.get("size", "SZ")
        if "split_norm" in sub_options:
            splits[symbol] = sub_options["split_norm"]
        if "energy_shift" in sub_options:
            if "energy_shift" not in family.options:
                raise ValueError(
                    f"the {family.label} family has no 'energy_shift' option; "
                    f"it accepts {list(family.options)}")
            shifts[symbol] = sub_options["energy_shift"]
        if "filter" in sub_options:
            filters[symbol] = sub_options["filter"]
        for key in SHARED_PSEUDO_OPTIONS:
            if key in sub_options:
                if key not in family.options:
                    raise ValueError(
                        f"the {family.label} family has no {key!r} option; "
                        f"it accepts {list(family.options)}")
                shared.setdefault(key, {})[symbol] = sub_options[key]
    if filters:
        # The filter cutoff is a property of the *grid*, which every atom
        # shares, so it may be written per element for convenience but must
        # agree -- and it must be written for all of them, since an element
        # left out would silently keep its unrepresentable components.
        distinct = {repr(v) for v in filters.values()}
        if len(distinct) > 1 or len(filters) != len(resolved):
            raise ValueError(
                f"the basis filter must be the same for every element, got "
                f"{filters!r}: the cutoff is set by the real-space grid, which "
                "all the atoms share.  Write it once as "
                "basis={'name': ..., 'filter': ..., 'size': {<per element>}}.")
        merged["filter"] = next(iter(filters.values()))
    for key, values in shared.items():
        if len({repr(v) for v in values.values()}) > 1:
            raise ValueError(
                f"the basis option {key!r} describes how every atom's basis is "
                f"constructed and must agree where it is written, got "
                f"{values!r}")
        merged[key] = next(iter(values.values()))
    merged["size"] = sizes
    if shifts:
        # A confinement is a property of each atom's basis (unlike the
        # filter's grid cutoff), so it may differ per element.  An element
        # left out gets the family's default -- not "unconfined": leaving an
        # option out never means turning it off.
        default = family.default_options.get("energy_shift")
        merged["energy_shift"] = ({**shifts} if default is None
                                  else {"*": default, **shifts})
    if splits:
        # Keep a scalar when every element agrees; otherwise the per-element
        # mapping travels on (it used to be silently overwritten by the last
        # element in the loop).
        distinct = set(splits.values())
        merged["split_norm"] = (distinct.pop() if len(distinct) == 1
                                and len(splits) == len(resolved) else splits)
    return family, merged


#: Default Laplacian per path.  Both keep the finite-difference stencil: it
#: is what the nuclear-gradient code differentiates, so energies and forces
#: stay consistent.  ``kinetic="spectral"`` is the more accurate choice for a
#: single-point energy on a coarse grid (the stencil under-estimates the
#: kinetic energy of compact functions by up to ~15% at 0.2-0.3 Angstrom).
DEFAULT_KINETIC = {"all-electron": "fd", "pseudopotentials": "fd"}


def _warn_unresolved(integrals, basis_fns, h):
    """Warn about basis functions / projectors the grid does not resolve."""
    import warnings
    functions, projectors = integrals.unresolved()
    if not functions and not projectors:
        return
    parts = []
    if functions:
        labels = [f"{i}:{type(basis_fns[i]).__name__}(l={getattr(basis_fns[i], 'l', '?')})"
                  for i in functions[:6]]
        ratios = [f"{integrals.resolution_ratios[i]:.2f}" for i in functions[:6]]
        parts.append(f"{len(functions)} basis function(s) whose grid kinetic "
                     f"energy is off by more than {int(100 * 0.25)}% of the exact "
                     f"value ({', '.join(labels)}; T_grid/T_exact = "
                     f"{', '.join(ratios)}{', ...' if len(functions) > 6 else ''})")
    if projectors:
        ratios = [f"{integrals.kb_resolution_ratios[i]:.2f}" for i in projectors[:6]]
        parts.append(f"{len(projectors)} Kleinman-Bylander projector(s) whose "
                     f"grid norm is off ({', '.join(ratios)}"
                     f"{', ...' if len(projectors) > 6 else ''})")
    warnings.warn(
        f"the real-space grid (h = {h:g} Angstrom) does not resolve "
        + " and ".join(parts)
        + ".  Energies involving them are not trustworthy: refine h, or drop "
        "the compact functions (a smaller basis size / no polarization).",
        RuntimeWarning, stacklevel=3)


def _pseudopotential_hamiltonian(atoms, grid, h, charge, spin, family,
                                 options, kinetic=None):
    """Valence-only Hamiltonian from a pseudopotential **family**.

    A thin dispatcher: ``family`` is the
    :class:`~mandacaru.pseudopotentials.families.FamilySpec` the basis name
    selected (see :func:`pseudopotential_family`), ``options`` the basis
    dict's remaining keys -- checked against ``family.options`` so a typo
    fails before any integral -- and ``family.build`` returns the same
    5-tuple as :func:`build_basis_hamiltonian`.  A new family is registered
    with ``register_family`` and needs nothing here.
    """
    unknown = sorted(set(options) - set(family.options))
    if unknown:
        raise ValueError(
            f"unknown option(s) {unknown} for the {family.label} basis; it "
            f"accepts {list(family.options)}")
    return family.build(atoms, grid, h, charge, spin, dict(options), kinetic)


def build_basis_hamiltonian(atoms, basis, grid, h: float, charge: int,
                            n_electrons, spin: bool = False,
                            frozen_core=False, frozen_orbitals=None,
                            kinetic=None):
    """Build the RHF MO Hamiltonian from ``atoms`` using ``basis``.

    ``basis`` is a name string or a ``{"name": ..., <options>}`` dict (see
    :func:`resolve_basis`).  The plane-wave family (``"PW"``) uses the periodic
    :class:`~mandacaru.core.PlaneWaveIntegrals` engine; a **pseudopotential
    family** (``"NCPP"`` / ``"ONCVPSP"`` / ``"PAW"``, see
    :func:`pseudopotential_family`) builds the valence-only Hamiltonian of
    that family -- the core electrons are removed, the basis is the smooth
    pseudo-atomic orbitals and the ``-Z/r`` potential is replaced by the
    family's local channel plus projectors; every other family uses an
    all-electron localized basis on the real-space grid.

    Returns ``(hamiltonian, num_particles, n_spatial_orbitals,
    integration_profile, context)``, where ``context`` carries the objects a
    *nuclear gradient* needs -- the live
    :class:`~mandacaru.core.MolecularIntegrals` (basis, grid, nuclei, softening)
    and ``atom_of_orbital``, which atom each basis function is centered on.  It is
    ``None`` for the plane-wave family, whose basis does not move with the
    nuclei.

    The reference occupation comes from the geometry's magnetic moments (see
    :func:`resolve_num_unpaired`): an even electron count defaults to the
    closed-shell singlet, an **odd** count to the doublet.  Odd-electron (and
    any ``n_alpha != n_beta``) systems are built in the open-shell
    **UHF natural-orbital** basis, even counts in the closed-shell RHF basis --
    see :meth:`~mandacaru.core.hamiltonian.MolecularIntegrals.molecular_hamiltonian`.

    ``frozen_core`` / ``frozen_orbitals`` apply the frozen-core approximation (see
    :func:`resolve_frozen_core`): the resolved core spatial MOs are removed from
    the active space, so the returned ``num_particles`` and ``n_spatial_orbitals``
    describe the reduced active space.

    ``kinetic`` selects the Laplacian discretization (``"fd"`` or
    ``"spectral"``); ``None`` takes :data:`DEFAULT_KINETIC` for the path.  After
    the integrals, basis functions and projectors the grid does not resolve
    (see :meth:`~mandacaru.core.hamiltonian.MolecularIntegrals.unresolved`) raise a
    :class:`RuntimeWarning` naming them.
    """
    name, options = resolve_basis(basis)
    symbols = atoms.get_chemical_symbols()
    family, options = resolve_pseudo_basis(name, options, symbols)
    if family is not None:
        if frozen_core or frozen_orbitals:
            raise ValueError(
                f"frozen_core is redundant with the {family.label} basis -- "
                "the core is already absent from the valence-only pseudo "
                "basis")
        if n_electrons is not None:
            # The valence count follows the datasets' valence charges; an
            # explicit count would silently disagree with the projectors and
            # one-center terms built for those charges.
            raise ValueError(
                f"n_electrons is not accepted with the {family.label} basis: "
                "the valence electron count comes from the pseudopotentials "
                "themselves.  Use `charge` to add or remove electrons.")
        return _pseudopotential_hamiltonian(atoms, grid, h, charge, spin,
                                            family, options, kinetic=kinetic)

    numbers = atoms.get_atomic_numbers()
    n_el = (int(n_electrons) if n_electrons is not None
            else int(sum(int(z) for z in numbers)) - int(charge))

    if _is_plane_wave(name):
        return _plane_wave_hamiltonian(atoms, options, n_el, spin, name,
                                       frozen_core, frozen_orbitals)

    from ..basis import BasisSet
    from ..core import MolecularIntegrals

    positions = coherent_positions(atoms)                 # minimum-image whole
    if name == PER_ELEMENT:
        bset = BasisSet.build(options)          # validated by resolve_pseudo_basis
    else:
        bset = BasisSet.build(name, **options)
    basis_fns, nuclei, atom_of_orbital = [], [], []
    for atom_index, (Z, sym, pos) in enumerate(zip(numbers, symbols, positions)):
        functions = bset.atom(sym, center=pos, units="angstrom")
        basis_fns += functions
        atom_of_orbital += [atom_index] * len(functions)
        nuclei.append((float(Z), pos))

    g = (grid if grid is not None
         else grid_from_cell(atoms, h, center=positions.mean(axis=0)))

    n_unpaired = resolve_num_unpaired(atoms, spin, n_el)
    n_alpha, n_beta = _num_particles(n_el, n_unpaired, name)
    frozen = resolve_frozen_core(frozen_core, frozen_orbitals, numbers, n_el,
                                 len(basis_fns), n_doubly=n_beta)
    n_active_el = n_el - 2 * len(frozen)
    num_particles = _num_particles(n_active_el, n_unpaired, name)

    # Soften the -Z/r cusp to half a grid step (Bohr): a nucleus that lands on a
    # grid node would otherwise sample -Z/r at r->0 and produce a ~1e12 garbage
    # core integral.  Half a step keeps the well-resolved region untouched while
    # bounding the on-node case, so heavier-atom cores stay finite on a coarse grid.
    softening = 0.5 * float(min(g.dx, g.dy, g.dz))
    integrals = MolecularIntegrals(
        nuclei, basis_fns, g, softening=softening,
        kinetic=kinetic or DEFAULT_KINETIC["all-electron"])
    hamiltonian = integrals.molecular_hamiltonian(
        mo_basis=True, n_electrons=n_el, num_particles=(n_alpha, n_beta),
        frozen_orbitals=frozen if frozen else None)
    _warn_unresolved(integrals, basis_fns, h)
    context = {"integrals": integrals, "atom_of_orbital": atom_of_orbital,
               "frozen": tuple(frozen), "n_electrons": n_el}
    return (hamiltonian, num_particles, len(basis_fns) - len(frozen),
            integrals.integration_profile(), context)


def _plane_wave_hamiltonian(atoms, options, n_el, spin, name,
                            frozen_core=False, frozen_orbitals=None):
    """Build the periodic plane-wave (PW) MO Hamiltonian from ``atoms``."""
    from ..core import PlaneWaveIntegrals

    if frozen_core or frozen_orbitals:
        raise NotImplementedError(
            "the frozen-core approximation is not supported for the plane-wave "
            "(PW) basis: plane waves are delocalized and have no localized core "
            "to freeze.  Use a localized basis (HAO / GTO / 6-31G(d) / NAO).")

    cell = np.asarray(atoms.get_cell(), dtype=float)
    if not np.any(cell):
        raise ValueError(
            "the plane-wave (PW) basis requires a periodic unit cell; set "
            "atoms.cell (or atoms.set_cell(...)).")
    positions = coherent_positions(atoms)
    numbers = atoms.get_atomic_numbers()
    nuclei = [(float(Z), pos) for Z, pos in zip(numbers, positions)]

    pw = PlaneWaveIntegrals(nuclei, cell, units="angstrom", **options)
    n_unpaired = resolve_num_unpaired(atoms, spin, n_el)
    num_particles = _num_particles(n_el, n_unpaired, name)
    hamiltonian = pw.molecular_hamiltonian(mo_basis=True, n_electrons=n_el,
                                           num_particles=num_particles)
    # No context: the plane-wave basis is not atom-centered, so it contributes
    # no Pulay forces and the gradient machinery does not apply to it.
    return (hamiltonian, num_particles, pw.n_orbitals,
            pw.integration_profile(), None)


def resolve_initial_state(initial_state):
    """Normalize the ``initial_state`` spec (currently ``"hartree-fock"`` only).

    ``None`` and ``"hartree-fock"`` / ``"hf"`` (case-insensitive) map to
    ``"hartree-fock"`` -- the Hartree-Fock determinant used as the ansatz
    reference.  Anything else raises ``ValueError``.
    """
    if initial_state is None:
        return "hartree-fock"
    key = str(initial_state).strip().lower().replace("_", "-").replace(" ", "-")
    if key in ("hartree-fock", "hartree", "hf"):
        return "hartree-fock"
    raise ValueError(
        f"unknown initial_state {initial_state!r}; only 'hartree-fock' "
        "(the Hartree-Fock determinant) is supported")


def monkhorst_pack_kpts(kpts):
    """Resolve a k-point spec to a Monkhorst-Pack mesh via ASE.

    ``kpts`` may be

    * ``None`` or ``(1, 1, 1)`` -- a single Gamma point;
    * a triple ``(n1, n2, n3)`` -- a Monkhorst-Pack grid;
    * a dict ``{"size": (n1, n2, n3), "gamma": True}`` -- the ASE spelling, where
      ``gamma=True`` shifts the mesh so it is Gamma-centered (includes the Gamma
      point even for even mesh sizes).

    Returns ``(size, gamma, mesh)``: the ``(n1, n2, n3)`` size, whether the mesh is
    Gamma-centered, and the ``(Nk, 3)`` array of fractional k-point coordinates
    built with :func:`ase.dft.kpoints.monkhorst_pack`.
    """
    from ase.dft.kpoints import monkhorst_pack

    gamma = None
    if kpts is None:
        size = (1, 1, 1)
    elif isinstance(kpts, dict):
        size = tuple(int(k) for k in kpts.get("size", (1, 1, 1)))
        if kpts.get("gamma", None) is not None:
            gamma = bool(kpts["gamma"])
    else:
        size = tuple(int(k) for k in kpts)
    if len(size) != 3 or any(k < 1 for k in size):
        raise ValueError(
            f"kpts size must be three positive integers (n1, n2, n3); got {kpts!r}")

    mesh = monkhorst_pack(size)
    if gamma:
        # Gamma-centered: shift by 0.5/n on even axes so Gamma is on the mesh.
        offset = np.array([0.5 / n if n % 2 == 0 else 0.0 for n in size])
        mesh = mesh + offset
    return size, bool(gamma), mesh
