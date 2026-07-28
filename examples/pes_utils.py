# -*- coding: utf-8 -*-
# file: examples/pes_utils.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Shared helpers for the molecular potential-energy-curve examples.

Builds dissociation curves by scanning the bond length of a diatomic.  Each
curve is referenced to the **sum of isolated-atom energies** (computed with
unrestricted Hartree-Fock so open-shell atoms like H and Li are handled), so
``E = 0`` is the separated-atom limit and the well depth is a binding energy.
That atomic sum -- not a correlated benchmark -- is the absolute reference used
throughout the potential-energy-curve examples and tests.

.. note::

   The engine integrates on a *uniform* real-space grid, so a nucleus that sits
   between grid nodes samples its ``-Z/r`` cusp differently than one on a node --
   the "egg-box" effect.  For a light, core-less atom (H) it is negligible; for a
   tight ``1s`` core (Li) it is large.  Two measures keep the curves smooth and
   comparable:

   * **grid-commensurate bond lengths** -- distances are stepped by *twice* the
     grid spacing (:func:`commensurate_distances`), so both nuclei move by a whole
     number of grid nodes at each step and their sub-node alignment (hence the
     core error) is *constant* along the scan and cancels in the curve *shape*;
   * **alignment-matched atomic references** -- each isolated-atom energy is
     computed on the same grid with the atom at a position it occupies in the
     molecule, so the residual core error also cancels in the *zero*.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor

import numpy as np
from ase.data import atomic_numbers

from carcara.algorithms import RHF, UHF
from carcara.core import MolecularIntegrals
from carcara.integrals import Grid


@dataclass
class GridSpec:
    """Uniform-grid parameters for a scan."""

    box_size: float          # cubic box edge (Angstrom)
    spacing: float           # grid spacing h (Angstrom)

    def build(self) -> Grid:
        """The origin-centered :class:`~carcara.integrals.Grid` of this spec."""
        return Grid(center=[0.0, 0.0, 0.0], box_size=self.box_size,
                    h=self.spacing)


def commensurate_distances(r_min: float, r_max: float,
                           grid_spec: GridSpec) -> np.ndarray:
    """Bond lengths stepped by ``2 * spacing`` so both nuclei move whole grid nodes.

    Keeping the step a multiple of ``2h`` makes each nucleus (at ``±R/2``) shift by
    an integer number of grid spacings between points, so the core "egg-box" error
    is constant along the scan and cancels in the curve shape.
    """
    step = 2.0 * grid_spec.spacing
    n = int(np.floor((r_max - r_min) / step + 1e-9)) + 1
    return r_min + step * np.arange(n)


def molecule_positions(distance: float) -> list[np.ndarray]:
    """Symmetric placement of a diatomic on the z-axis about the origin."""
    return [np.array([0.0, 0.0, -distance / 2]),
            np.array([0.0, 0.0, +distance / 2])]


def cusp_softening(grid: Grid) -> float:
    """The half-grid-step Coulomb softening used by the calculator drivers.

    The ASE-calculator path softens ``-Z/r`` to half a grid step so a nucleus on
    a grid node cannot produce a divergent integral; the isolated-atom reference
    must use the **same** softening or it would not cancel against the scan.
    """
    return 0.5 * float(min(grid.dx, grid.dy, grid.dz))


def rhf_total_energy(symbols, positions, basis_set, grid: Grid) -> float:
    """RHF total energy (electronic + nuclear repulsion) of a molecule, in Hartree."""
    nuclei = [(float(atomic_numbers[s]), np.asarray(p, dtype=float))
              for s, p in zip(symbols, positions)]
    n_electrons = sum(atomic_numbers[s] for s in symbols)
    basis = []
    for sym, pos in zip(symbols, positions):
        basis.extend(basis_set.atom(sym, center=pos))
    integrals = MolecularIntegrals(nuclei, basis, grid,
                                   softening=cusp_softening(grid))
    rhf = RHF(integrals.one_body(), integrals.two_body(), n_electrons).run()
    return rhf.electronic_energy + integrals.nuclear_repulsion


def atom_energy(symbol, basis_set, grid: Grid, position) -> float:
    """UHF energy of an isolated atom (Hartree) placed at ``position`` on ``grid``.

    The energy is translation-invariant physically; the placement is chosen only
    to match the atom's grid alignment in the molecular scan, so the core grid
    error cancels in the reference.  The Coulomb softening matches the
    calculator drivers (:func:`cusp_softening`) for the same reason.
    """
    Z = atomic_numbers[symbol]
    integrals = MolecularIntegrals([(float(Z), np.asarray(position, dtype=float))],
                                   basis_set.atom(symbol, center=position), grid,
                                   softening=cusp_softening(grid))
    n_alpha, n_beta = ceil(Z / 2), floor(Z / 2)
    return UHF(integrals.one_body(), integrals.two_body(), n_alpha, n_beta).run()


def atomic_reference(symbols, basis_set, grid: Grid, ref_positions) -> float:
    """Sum of isolated-atom UHF energies, each atom at its molecular grid position.

    This is the absolute energy reference of every dissociation curve: the
    energy of the fully separated, non-interacting atoms in the same basis on
    the same grid.
    """
    return sum(atom_energy(s, basis_set, grid, p)
               for s, p in zip(symbols, ref_positions))
