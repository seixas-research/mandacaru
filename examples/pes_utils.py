# -*- coding: utf-8 -*-
# file: examples/pes_utils.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Shared helpers for the molecular potential-energy-surface (PES) examples.

Builds dissociation curves by scanning the bond length of a diatomic and, at each
geometry, computing the restricted Hartree-Fock (RHF) total energy over a chosen
localized basis via Carcará's real-space integral engine.  Each curve is
referenced to the **sum of isolated-atom energies** (computed with unrestricted
Hartree-Fock so open-shell atoms like H and Li are handled), so ``E = 0`` is the
separated-atom limit and the well depth is a binding energy.

Three basis sets are compared:

* ``FAO``        -- minimal analytic Full Atomic Orbitals;
* ``STO-3G``     -- native minimal Gaussian (STO-3G);
* ``6-31G(d)``   -- native Pople split-valence with ``d`` polarization.

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
     computed on the same grid with the atom at the position it occupies in the
     molecule, so the residual core error also cancels in the *zero*.

   With these, H2 is quantitative and LiH is a smooth, physically reasonable curve
   (bound minimum near the experimental 1.6 A); LiH absolute well depths remain
   grid-limited by the under-resolved Li core and should be read as qualitative.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import ceil, floor

import numpy as np
from ase.data import atomic_numbers

from carcara.algorithms import RHF, UHF
from carcara.basis import BasisSet
from carcara.core import MolecularIntegrals
from carcara.integrals import Grid

HARTREE_TO_EV = 27.211386245988

# Basis sets compared in every curve: (key, human label, BasisSet).
BASES = [
    ("FAO", "FAO", BasisSet.build("FAO")),
    ("sto-3g", "STO-3G", BasisSet.build("GTO", n_gaussians=3)),
    ("6-31g(d)", "6-31G(d)", BasisSet.build("6-31G(d)")),
]


@dataclass
class GridSpec:
    """Uniform-grid parameters for a scan."""

    box_size: float          # cubic box edge (Angstrom)
    spacing: float           # grid spacing h (Angstrom)


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


def _molecule_positions(R: float):
    """Symmetric placement of a diatomic on the z-axis about the origin."""
    return [np.array([0.0, 0.0, -R / 2]), np.array([0.0, 0.0, +R / 2])]


def rhf_total_energy(symbols, positions, basis_set, grid: Grid) -> float:
    """RHF total energy (electronic + nuclear repulsion) of a molecule, in Hartree."""
    nuclei = [(float(atomic_numbers[s]), np.asarray(p, dtype=float))
              for s, p in zip(symbols, positions)]
    n_electrons = sum(atomic_numbers[s] for s in symbols)
    basis = []
    for sym, pos in zip(symbols, positions):
        basis.extend(basis_set.atom(sym, center=pos))
    integrals = MolecularIntegrals(nuclei, basis, grid)
    rhf = RHF(integrals.one_body(), integrals.two_body(), n_electrons).run()
    return rhf.electronic_energy + integrals.nuclear_repulsion


def atom_energy(symbol, basis_set, grid: Grid, position) -> float:
    """UHF energy of an isolated atom (Hartree) placed at ``position`` on ``grid``.

    The energy is translation-invariant physically; the placement is chosen only
    to match the atom's grid alignment in the molecular scan, so the core grid
    error cancels in the reference.
    """
    Z = atomic_numbers[symbol]
    integrals = MolecularIntegrals([(float(Z), np.asarray(position, dtype=float))],
                                    basis_set.atom(symbol, center=position), grid)
    n_alpha, n_beta = ceil(Z / 2), floor(Z / 2)
    return UHF(integrals.one_body(), integrals.two_body(), n_alpha, n_beta).run()


def atomic_reference(symbols, basis_set, grid: Grid, ref_positions) -> float:
    """Sum of isolated-atom UHF energies, each atom at its molecular grid position."""
    return sum(atom_energy(s, basis_set, grid, p)
               for s, p in zip(symbols, ref_positions))


def scan_curve(symbols, distances, basis_set, grid: Grid, e_atoms: float):
    """RHF total and atom-referenced relative energy over a bond-length scan.

    ``symbols`` is the diatomic ``(A, B)`` placed symmetrically on the z-axis.
    Returns ``(total_Ha, relative_eV)`` arrays; failed points are ``NaN``.
    """
    total = np.full(len(distances), np.nan)
    for i, R in enumerate(distances):
        try:
            total[i] = rhf_total_energy(symbols, _molecule_positions(R),
                                        basis_set, grid)
        except Exception as exc:                       # keep the scan going
            print(f"    ! {basis_set!r} R={R:.3f}: {type(exc).__name__}: {exc}")
    relative = (total - e_atoms) * HARTREE_TO_EV
    return total, relative


def generate_pes(name, symbols, distances, grid_spec: GridSpec,
                 csv_path: str) -> dict:
    """Run every basis for a molecule and write a tidy CSV; return the data.

    The CSV has one row per bond length with, per basis, the RHF total energy
    (Hartree) and the atom-referenced relative energy (eV).  Header comment lines
    record the isolated-atom reference energy of each basis.
    """
    distances = np.asarray(distances, dtype=float)
    grid = Grid(center=[0.0, 0.0, 0.0], box_size=grid_spec.box_size,
                h=grid_spec.spacing)
    # Reference atoms sit where they do in the molecule at the first geometry,
    # so their grid alignment matches the (commensurate) scan.
    ref_positions = _molecule_positions(float(distances[0]))

    columns = {"distance_angstrom": distances}
    references = {}
    print(f"== {name}: scanning {len(distances)} bond lengths "
          f"[{distances[0]:.2f}, {distances[-1]:.2f}] A ==")
    for key, label, basis_set in BASES:
        e_atoms = atomic_reference(symbols, basis_set, grid, ref_positions)
        references[label] = e_atoms
        print(f"  {label:9s}: E(atoms) = {e_atoms:+.5f} Ha ; scanning ...")
        total, relative = scan_curve(symbols, distances, basis_set, grid, e_atoms)
        columns[f"{key}_total_Ha"] = total
        columns[f"{key}_rel_eV"] = relative

    _write_csv(csv_path, name, symbols, grid_spec, references, columns)
    print(f"  -> wrote {csv_path}\n")
    return columns


def _write_csv(path, name, symbols, grid_spec, references, columns):
    field_names = list(columns)
    rows = len(columns["distance_angstrom"])
    with open(path, "w", newline="") as f:
        f.write(f"# Carcara potential-energy surface: {name} ({'-'.join(symbols)})\n")
        f.write(f"# method: RHF total energy; reference E=0 at sum of isolated "
                f"atom UHF energies\n")
        f.write(f"# grid: box_size={grid_spec.box_size} A, spacing="
                f"{grid_spec.spacing} A (uniform real-space)\n")
        for label, e in references.items():
            f.write(f"# E(atoms) [{label}] = {e:.8f} Ha\n")
        writer = csv.writer(f)
        writer.writerow(field_names)
        for i in range(rows):
            writer.writerow([f"{columns[c][i]:.8f}" for c in field_names])
