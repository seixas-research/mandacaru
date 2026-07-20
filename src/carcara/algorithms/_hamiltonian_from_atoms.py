# -*- coding: utf-8 -*-
# file: algorithms/_hamiltonian_from_atoms.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Build a molecular qubit Hamiltonian from an ASE ``Atoms`` geometry.

Shared by the ASE-calculator paths of :class:`~carcara.algorithms.vqe.VQE` and
:class:`~carcara.algorithms.adapt_vqe.ADAPTVQE` so both drive quantum simulations
the same way: a geometry (elements, positions, unit cell) plus a basis name is
turned into an RHF molecular-orbital :class:`~carcara.core.mapping.Fermion`
Hamiltonian, and the real-space integration is profiled (time / cores / memory)
along the way.

**Placement is irrelevant.**  carcará solves an *isolated-molecule* (Gamma-point,
open-boundary) electronic-structure problem: the cell only sets the size of the
real-space box.  The box is centred on the molecule -- and, when the geometry is
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
    (Angstrom) sets the uniform node spacing.  The box is **centred on the
    molecule** (``center``, defaulting to the minimum-image centroid) rather than
    on the cell, so wherever the atoms are placed the orbitals stay inside the
    grid.  The same grid feeds both the one- and two-body integral kernels.

    A unit cell is **required** (or pass an explicit ``grid=``); raises
    ``ValueError`` otherwise.
    """
    from ..integrals import Grid

    cell = np.asarray(atoms.get_cell(), dtype=float)      # Angstrom (ASE)
    if not np.any(cell):
        raise ValueError(
            "cannot auto-generate a grid: the geometry has no unit cell.  Set "
            "one (e.g. atoms.cell = [[Lx,0,0],[0,Ly,0],[0,0,Lz]] or "
            "atoms.set_cell(...)), or pass an explicit `grid=`.  The grid is then "
            f"built from the cell at resolution h={h:g} Angstrom.")
    if center is None:
        center = coherent_positions(atoms).mean(axis=0)   # centre on the molecule
    return Grid(center=center, box_size=0.0, h=h, units="angstrom", cell=cell)


def resolve_basis(basis):
    """Normalize a ``basis`` spec to ``(name, options)``.

    Accepts a plain method name (``"FAO"``) or a dict giving the name plus that
    family's options: ``{"name": "FAO"}``, ``{"name": "NAO", "energy_shift": 0.03}``,
    ``{"name": "NAO", "size": "DZP"}`` (multiple-zeta and polarized -- see
    :mod:`carcara.basis.multizeta`), ``{"name": "GTO", "n_gaussians": 3}``,
    ``{"name": "6-31G(d)"}`` or the plane-wave basis
    ``{"name": "PW", "energy_cutoff": 300}``.  Returns the name string and a dict
    of the remaining keyword options.
    """
    if isinstance(basis, str):
        return basis, {}
    if isinstance(basis, dict):
        options = dict(basis)
        name = options.pop("name", None)
        if name is None:
            raise ValueError(
                "a basis dict must include a 'name' key, e.g. {'name': 'FAO'} "
                "or {'name': 'PW', 'energy_cutoff': 300}")
        return name, options
    raise TypeError(
        "basis must be a name string or a dict like {'name': 'FAO', ...}")


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
                        n_orbitals: int) -> list[int]:
    """Resolve the frozen-core spec to a sorted list of frozen spatial-MO indices.

    ``frozen_orbitals`` (an explicit list of spatial MO indices) takes precedence;
    otherwise the lowest ``_auto_frozen_count(frozen_core, numbers)`` MOs are
    frozen.  Every frozen orbital must be doubly occupied in the reference
    (index ``< n_el // 2``), since the frozen-core approximation removes doubly
    occupied core orbitals.
    """
    n_occ = n_el // 2
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
    ``spin`` flag: ``spin=True`` requests a single unpaired electron for an
    odd-electron count (a high-spin doublet) and, for an even count, keeps the
    closed-shell singlet (``0``); ``spin=False`` is always ``0``.
    """
    total = 0.0
    if atoms is not None:
        try:
            total = float(np.sum(atoms.get_initial_magnetic_moments()))
        except Exception:
            total = 0.0
    if abs(total) > 1e-8:
        return int(round(abs(total)))
    if spin:
        return 1 if n_el % 2 == 1 else 0
    return 0


def _num_particles(n_el: int, n_unpaired: int, basis) -> tuple[int, int]:
    """Reference occupation ``(n_alpha, n_beta)`` for ``n_unpaired = n_alpha - n_beta``.

    Validates the shell for the restricted (closed-shell RHF) integral builders:
    the electron count must be even (odd-electron open shells are not yet
    supported) and the requested spin state ``n_unpaired`` must share its parity
    and not exceed ``n_el``.
    """
    if n_el % 2 != 0:
        raise NotImplementedError(
            f"the built-in {basis!r} builder uses restricted (closed-shell RHF) "
            f"integrals and needs an even electron count; got {n_el}.  Open-shell "
            "odd-electron systems are not yet supported (pass a "
            "hamiltonian_builder for those).")
    if n_unpaired < 0 or n_unpaired > n_el or n_unpaired % 2 != 0:
        raise ValueError(
            f"the requested spin state (n_unpaired={n_unpaired}) is incompatible "
            f"with {n_el} electrons: n_unpaired must be even and in [0, {n_el}].")
    return ((n_el + n_unpaired) // 2, (n_el - n_unpaired) // 2)


#: Basis families whose radial functions the pseudopotential path can honour.
#: ``"PP"`` names the pseudo-atomic family explicitly; ``"NAO"`` is accepted
#: because the multiple-zeta construction is shared with it.
_PSEUDO_BASIS_NAMES = ("PP", "PSEUDO", "NAO")


def _merge_pseudo_basis_options(basis, options):
    """Fold the ``basis`` spec into the pseudopotential options, or refuse it.

    A pseudopotential fixes its own first zeta -- the Troullier-Martins
    pseudo-orbitals the Kleinman-Bylander projectors were built from -- so an
    all-electron family such as ``"FAO"`` or ``"6-31G(d)"`` cannot be combined
    with it.  What *can* carry over is the size hierarchy (``size``,
    ``split_norm``), which refines that pseudo-orbital rather than replacing it.

    Silently ignoring the argument, as this path used to, meant
    ``basis={"name": "NAO", "size": "DZP"}`` ran a minimal single-zeta basis with
    no indication anything had been dropped.
    """
    name, basis_options = resolve_basis(basis)
    key = str(name).upper().replace("-", "").replace(" ", "")

    unusable = {k: v for k, v in basis_options.items()
                if k not in ("size", "split_norm")}
    if unusable or key not in _PSEUDO_BASIS_NAMES:
        if key in ("FAO",) and not unusable:
            # The historical default: nothing was actually requested, so there
            # is nothing to honour or refuse.
            return options
        raise ValueError(
            f"basis {basis!r} cannot be used with pseudopotentials. The "
            "pseudopotential supplies its own valence radial functions (the "
            "pseudized orbitals its Kleinman-Bylander projectors were built "
            "from), so an all-electron family cannot replace them. Use "
            "basis={'name': 'PP', 'size': 'DZP'} to refine them instead, or "
            "drop pseudopotentials=True to run all-electron.")

    merged = dict(options)
    for option in ("size", "split_norm"):
        if option in basis_options:
            merged[option] = basis_options[option]
    return merged


def _pseudopotential_hamiltonian(atoms, grid, h, charge, spin, options):
    """Valence-only Hamiltonian from norm-conserving pseudopotentials.

    The core electrons are gone entirely: the basis is the set of valence
    pseudo-atomic orbitals, the external potential is the smooth local channel,
    and the Kleinman-Bylander projectors supply the nonlocal part.  The
    "nuclei" carry the *ionic* charges, so the constant term is the ion-ion
    repulsion.
    """
    from ..basis.pseudo_io import get_pseudopotential
    from ..basis.pseudo_orbital import (kb_projectors, pseudo_basis,
                                        valence_electrons)
    from ..core import MolecularIntegrals

    directory = options.get("directory")
    symbols = atoms.get_chemical_symbols()
    positions = coherent_positions(atoms)
    potentials = {symbol: get_pseudopotential(symbol, directory)
                  for symbol in set(symbols)}

    basis_fns, atom_of_orbital = pseudo_basis(
        symbols, positions, potentials, size=options.get("size", "SZ"),
        split_norm=options.get("split_norm"))
    projectors = kb_projectors(symbols, positions, potentials)
    nuclei = [(potentials[symbol].valence_charge, position)
              for symbol, position in zip(symbols, positions)]

    n_el = int(round(valence_electrons(symbols, potentials))) - int(charge)
    g = (grid if grid is not None
         else grid_from_cell(atoms, h, center=positions.mean(axis=0)))

    integrals = MolecularIntegrals(nuclei, basis_fns, g, softening=0.0,
                                   pseudopotentials=[potentials[s]
                                                     for s in symbols],
                                   kb_projectors=projectors)
    hamiltonian = integrals.molecular_hamiltonian(mo_basis=True,
                                                  n_electrons=n_el)
    n_unpaired = resolve_num_unpaired(atoms, spin, n_el)
    num_particles = _num_particles(n_el, n_unpaired, "PP")

    context = {"integrals": integrals, "atom_of_orbital": atom_of_orbital,
               "frozen": (), "n_electrons": n_el,
               "pseudopotentials": potentials, "kb_projectors": projectors}
    return (hamiltonian, num_particles, len(basis_fns),
            integrals.integration_profile(), context)


def build_basis_hamiltonian(atoms, basis, grid, h: float, charge: int,
                            n_electrons, spin: bool = False,
                            frozen_core=False, frozen_orbitals=None,
                            pseudopotentials=None):
    """Build the RHF MO Hamiltonian from ``atoms`` using ``basis``.

    ``basis`` is a name string or a ``{"name": ..., <options>}`` dict (see
    :func:`resolve_basis`).  The plane-wave family (``"PW"``) uses the periodic
    :class:`~carcara.core.PlaneWaveIntegrals` engine; every other family uses a
    localized basis on the real-space grid.

    Returns ``(hamiltonian, num_particles, n_spatial_orbitals,
    integration_profile, context)``, where ``context`` carries the objects a
    *nuclear gradient* needs -- the live
    :class:`~carcara.core.MolecularIntegrals` (basis, grid, nuclei, softening)
    and ``atom_of_orbital``, which atom each basis function is centred on.  It is
    ``None`` for the plane-wave family, whose basis does not move with the
    nuclei.

    ``spin`` selects the reference occupation: ``False`` (default) is closed-shell
    (``n_alpha == n_beta``, requires an even electron count); ``True`` is a
    spin-polarized (high-spin) reference.  A genuinely open-shell (odd-electron)
    system raises ``NotImplementedError`` (RHF-only integrals).

    ``frozen_core`` / ``frozen_orbitals`` apply the frozen-core approximation (see
    :func:`resolve_frozen_core`): the resolved core spatial MOs are removed from
    the active space, so the returned ``num_particles`` and ``n_spatial_orbitals``
    describe the reduced active space.
    """
    if pseudopotentials:
        if frozen_core or frozen_orbitals:
            raise ValueError(
                "frozen_core is redundant with pseudopotentials -- the core is "
                "already absent from the valence-only pseudo basis")
        options = ({} if pseudopotentials is True else dict(pseudopotentials))
        options = _merge_pseudo_basis_options(basis, options)
        return _pseudopotential_hamiltonian(atoms, grid, h, charge, spin,
                                            options)

    name, options = resolve_basis(basis)
    numbers = atoms.get_atomic_numbers()
    n_el = (int(n_electrons) if n_electrons is not None
            else int(sum(int(z) for z in numbers)) - int(charge))

    if name.upper().replace("-", "").replace(" ", "") in ("PW", "PLANEWAVE"):
        return _plane_wave_hamiltonian(atoms, options, n_el, spin, name,
                                       frozen_core, frozen_orbitals)

    from ..basis import BasisSet
    from ..core import MolecularIntegrals

    symbols = atoms.get_chemical_symbols()
    positions = coherent_positions(atoms)                 # minimum-image whole
    bset = BasisSet.build(name, **options)
    basis_fns, nuclei, atom_of_orbital = [], [], []
    for atom_index, (Z, sym, pos) in enumerate(zip(numbers, symbols, positions)):
        functions = bset.atom(sym, center=pos, units="angstrom")
        basis_fns += functions
        atom_of_orbital += [atom_index] * len(functions)
        nuclei.append((float(Z), pos))

    g = (grid if grid is not None
         else grid_from_cell(atoms, h, center=positions.mean(axis=0)))

    frozen = resolve_frozen_core(frozen_core, frozen_orbitals, numbers, n_el,
                                 len(basis_fns))
    n_active_el = n_el - 2 * len(frozen)
    n_unpaired = resolve_num_unpaired(atoms, spin, n_el)
    num_particles = _num_particles(n_active_el, n_unpaired, name)

    # Soften the -Z/r cusp to half a grid step (Bohr): a nucleus that lands on a
    # grid node would otherwise sample -Z/r at r->0 and produce a ~1e12 garbage
    # core integral.  Half a step keeps the well-resolved region untouched while
    # bounding the on-node case, so heavier-atom cores stay finite on a coarse grid.
    softening = 0.5 * float(min(g.dx, g.dy, g.dz))
    integrals = MolecularIntegrals(nuclei, basis_fns, g, softening=softening)
    hamiltonian = integrals.molecular_hamiltonian(
        mo_basis=True, n_electrons=n_el,
        frozen_orbitals=frozen if frozen else None)
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
            "to freeze.  Use a localized basis (FAO / GTO / 6-31G(d) / NAO).")

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
    hamiltonian = pw.molecular_hamiltonian(mo_basis=True, n_electrons=n_el)
    # No context: the plane-wave basis is not atom-centred, so it contributes
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
      ``gamma=True`` shifts the mesh so it is Gamma-centred (includes the Gamma
      point even for even mesh sizes).

    Returns ``(size, gamma, mesh)``: the ``(n1, n2, n3)`` size, whether the mesh is
    Gamma-centred, and the ``(Nk, 3)`` array of fractional k-point coordinates
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
        # Gamma-centred: shift by 0.5/n on even axes so Gamma is on the mesh.
        offset = np.array([0.5 / n if n % 2 == 0 else 0.0 for n in size])
        mesh = mesh + offset
    return size, bool(gamma), mesh
