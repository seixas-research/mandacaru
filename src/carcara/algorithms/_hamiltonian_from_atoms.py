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
    ``{"name": "GTO", "n_gaussians": 3}``, ``{"name": "6-31G(d)"}`` or the
    plane-wave basis ``{"name": "PW", "energy_cutoff": 300}``.  Returns the name
    string and a dict of the remaining keyword options.
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


def _num_particles(n_el: int, spin: bool, basis) -> tuple[int, int]:
    """Reference occupation ``(n_alpha, n_beta)``; validates the shell for RHF."""
    if n_el % 2 != 0:
        if spin:
            raise NotImplementedError(
                f"open-shell spin-polarized Hamiltonian construction ({n_el} "
                "electrons) is not yet implemented: the built-in builders use "
                "closed-shell RHF.  Pass a hamiltonian_builder for open shells.")
        raise ValueError(
            f"the built-in {basis!r} builder assumes a closed shell; got an odd "
            f"electron count ({n_el}). Use spin=True (open shell) or pass a "
            "hamiltonian_builder.")
    return ((n_el + 1) // 2, n_el // 2) if spin else (n_el // 2, n_el // 2)


def build_basis_hamiltonian(atoms, basis, grid, h: float, charge: int,
                            n_electrons, spin: bool = False):
    """Build the RHF MO Hamiltonian from ``atoms`` using ``basis``.

    ``basis`` is a name string or a ``{"name": ..., <options>}`` dict (see
    :func:`resolve_basis`).  The plane-wave family (``"PW"``) uses the periodic
    :class:`~carcara.core.PlaneWaveIntegrals` engine; every other family uses a
    localized basis on the real-space grid.

    Returns ``(hamiltonian, num_particles, n_spatial_orbitals, integration_profile)``.

    ``spin`` selects the reference occupation: ``False`` (default) is closed-shell
    (``n_alpha == n_beta``, requires an even electron count); ``True`` is a
    spin-polarized (high-spin) reference.  A genuinely open-shell (odd-electron)
    system raises ``NotImplementedError`` (RHF-only integrals).
    """
    name, options = resolve_basis(basis)
    numbers = atoms.get_atomic_numbers()
    n_el = (int(n_electrons) if n_electrons is not None
            else int(sum(int(z) for z in numbers)) - int(charge))

    if name.upper().replace("-", "").replace(" ", "") in ("PW", "PLANEWAVE"):
        return _plane_wave_hamiltonian(atoms, options, n_el, spin, name)

    from ..basis import BasisSet
    from ..core import MolecularIntegrals

    symbols = atoms.get_chemical_symbols()
    positions = coherent_positions(atoms)                 # minimum-image whole
    bset = BasisSet.build(name, **options)
    basis_fns, nuclei = [], []
    for Z, sym, pos in zip(numbers, symbols, positions):
        basis_fns += bset.atom(sym, center=pos, units="angstrom")
        nuclei.append((float(Z), pos))

    g = (grid if grid is not None
         else grid_from_cell(atoms, h, center=positions.mean(axis=0)))
    num_particles = _num_particles(n_el, spin, name)

    integrals = MolecularIntegrals(nuclei, basis_fns, g)
    hamiltonian = integrals.molecular_hamiltonian(mo_basis=True, n_electrons=n_el)
    return (hamiltonian, num_particles, len(basis_fns),
            integrals.integration_profile())


def _plane_wave_hamiltonian(atoms, options, n_el, spin, name):
    """Build the periodic plane-wave (PW) MO Hamiltonian from ``atoms``."""
    from ..core import PlaneWaveIntegrals

    cell = np.asarray(atoms.get_cell(), dtype=float)
    if not np.any(cell):
        raise ValueError(
            "the plane-wave (PW) basis requires a periodic unit cell; set "
            "atoms.cell (or atoms.set_cell(...)).")
    positions = coherent_positions(atoms)
    numbers = atoms.get_atomic_numbers()
    nuclei = [(float(Z), pos) for Z, pos in zip(numbers, positions)]

    pw = PlaneWaveIntegrals(nuclei, cell, units="angstrom", **options)
    num_particles = _num_particles(n_el, spin, name)
    hamiltonian = pw.molecular_hamiltonian(mo_basis=True, n_electrons=n_el)
    return (hamiltonian, num_particles, pw.n_orbitals, pw.integration_profile())


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
