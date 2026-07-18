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
"""

from __future__ import annotations

import numpy as np


def grid_from_cell(atoms, h: float):
    """Build the real-space integration grid from the ASE ``atoms.cell``.

    The grid is generated automatically from the **unit cell** and the target
    resolution ``h`` (Angstrom): the cell's three lattice vectors fix the extent
    (and shape) of the box, and ``h`` sets the uniform node spacing.  The same
    grid feeds both the one- and two-body integral kernels over the chosen basis
    (the engine is basis-agnostic).

    A unit cell is **required**: attach one to the geometry (``atoms.cell = ...``
    / ``atoms.set_cell(...)``), or pass an explicit ``grid=``.  Raises
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
    center = 0.5 * cell.sum(axis=0)                        # geometric cell center
    return Grid(center=center, box_size=0.0, h=h, units="angstrom", cell=cell)


def build_basis_hamiltonian(atoms, basis: str, grid, h: float, charge: int,
                            n_electrons):
    """Build the RHF MO Hamiltonian from ``atoms`` using the named ``basis``.

    Returns ``(hamiltonian, num_particles, n_spatial_orbitals, integration_profile)``,
    where ``integration_profile`` is the timing / cores / peak-memory dict from
    the real-space integral engine (see
    :meth:`carcara.core.MolecularIntegrals.integration_profile`).
    """
    from ..basis import BasisSet
    from ..core import MolecularIntegrals

    numbers = atoms.get_atomic_numbers()
    symbols = atoms.get_chemical_symbols()
    positions = np.asarray(atoms.get_positions(), dtype=float)

    bset = BasisSet.build(basis)
    basis_fns, nuclei = [], []
    for Z, sym, pos in zip(numbers, symbols, positions):
        basis_fns += bset.atom(sym, center=pos, units="angstrom")
        nuclei.append((float(Z), pos))

    g = grid if grid is not None else grid_from_cell(atoms, h)
    n_el = (int(n_electrons) if n_electrons is not None
            else int(sum(int(z) for z in numbers)) - int(charge))
    if n_el % 2 != 0:
        raise ValueError(
            f"the built-in {basis!r} builder assumes a closed shell; got an odd "
            f"electron count ({n_el}). Pass a hamiltonian_builder.")

    integrals = MolecularIntegrals(nuclei, basis_fns, g)
    hamiltonian = integrals.molecular_hamiltonian(mo_basis=True, n_electrons=n_el)
    return (hamiltonian, (n_el // 2, n_el // 2), len(basis_fns),
            integrals.integration_profile())
