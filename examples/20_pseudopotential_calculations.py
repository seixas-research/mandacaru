# -*- coding: utf-8 -*-
# file: examples/20_pseudopotential_calculations.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Running the variational solvers with norm-conserving pseudopotentials.

``pseudopotentials=True`` switches any solver from an all-electron calculation to
a **valence-only** one:

* the core electrons are removed (oxygen keeps 6 of its 8);
* the basis becomes the smooth pseudo-atomic orbitals;
* the singular :math:`-Z/r` external potential is replaced by a bounded local
  channel plus Kleinman-Bylander projectors.

.. code-block:: python

    atoms.calc = QuantumCalculator(method="adapt-vqe", basis="FAO",
                                   pseudopotentials=True, h=0.15)

The library under ``pseudos/`` covers every element with Z < 90 (H through
Ac) and is loaded automatically.

What this script measures
-------------------------
1. the library and the size reduction it buys;
2. H2 end to end, all-electron vs pseudopotential;
3. the **isolated-atom force test** -- the exact answer is zero, and it is the
   sharpest probe of the grid pathology that motivated pseudopotentials;
4. H2O with a valence-only Hamiltonian.

Output: ``examples/data/pseudopotential_forces.png``.
"""

from __future__ import annotations

import os
import time

import numpy as np
from ase import Atoms

from carcara.algorithms import QuantumCalculator
from carcara.basis.pseudo_io import available_elements, get_pseudopotential
from carcara.integrals import Grid
from carcara.units import BOHR_TO_ANGSTROM

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)
PNG_PATH = os.path.join(DATA, "pseudopotential_forces.png")
RULE = "=" * 76

# --------------------------------------------------------------------------- #
# 1. The library.
# --------------------------------------------------------------------------- #

print(RULE)
print("1. The bundled pseudopotential library (pseudos/)")
print(RULE)
elements = available_elements()
print(f"{len(elements)} elements: {' '.join(elements)}\n")
print(f"{'atom':>5}{'Z':>4}{'Z_ion':>7}{'core removed':>15}{'V_loc(0) Ha':>14}")
for symbol in ("H", "C", "O", "Si", "Cl", "Fe"):
    pp = get_pseudopotential(symbol)
    removed = pp.atomic_number - pp.valence_charge
    print(f"{symbol:>5}{pp.atomic_number:>4}{pp.valence_charge:>7.0f}"
          f"{removed:>15.0f}{pp.v_local[0]:>14.2f}")

# --------------------------------------------------------------------------- #
# 2. H2 end to end.
# --------------------------------------------------------------------------- #

print(f"\n{RULE}")
print("2. H2: all-electron vs pseudopotential")
print(RULE)

grid = Grid(center=[0.0, 0.0, 0.0], box_size=6.0, h=0.20)
for label, options in (("all-electron", {}),
                       ("pseudopotential", {"pseudopotentials": True})):
    atoms = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]])
    start = time.perf_counter()
    atoms.calc = QuantumCalculator(method="vqe", basis="FAO", grid=grid,
                                   verbose=False, **options)
    energy = atoms.get_potential_energy()
    print(f"  {label:<17} E = {energy:>12.4f} eV   "
          f"{atoms.calc.n_qubits} qubits   "
          f"{time.perf_counter() - start:.1f}s")
print("  (hydrogen has no core, so the two differ only by the pseudization")
print("   of its 1s -- the absolute energies are not comparable.)")

# --------------------------------------------------------------------------- #
# 3. Isolated-atom force: the decisive test.
# --------------------------------------------------------------------------- #

print(f"\n{RULE}")
print("3. Force on an ISOLATED oxygen atom.  Exact answer: zero.")
print(RULE)


def isolated_force(spacing, use_pseudopotentials):
    """Largest force on a lone O atom placed off a grid node (eV/Angstrom)."""
    box = Grid(center=[0, 0, 0], box_size=6.0, h=spacing)
    shift = 0.37 * box.dx * BOHR_TO_ANGSTROM
    atoms = Atoms("O", positions=[[shift, 0.0, 0.0]])
    atoms.calc = QuantumCalculator(
        method="adapt-vqe", basis="FAO", grid=box,
        pseudopotentials=use_pseudopotentials,
        frozen_core=not use_pseudopotentials, pool="qeb",
        max_iterations=6, gradient_tolerance=1e-3, profile=False,
        verbose=False)
    atoms.get_potential_energy()
    return float(np.abs(atoms.get_forces()).max())


spacings = (0.20, 0.15, 0.12)
all_electron, pseudo = [], []
print(f"{'h (A)':>8}{'all-electron':>16}{'pseudopotential':>18}{'ratio':>9}")
for spacing in spacings:
    bare = isolated_force(spacing, False)
    smooth = isolated_force(spacing, True)
    all_electron.append(bare)
    pseudo.append(smooth)
    print(f"{spacing:>8.2f}{bare:>16.1f}{smooth:>18.1f}{bare / smooth:>8.0f}x")

print("\nThe columns behave *qualitatively* differently.  All-electron gets")
print("worse as the grid is refined: the oxygen 1s (length scale a0/Z = 0.066 A)")
print("is never resolved, and the nearest grid node moves into an unresolved")
print("cusp faster than the sampling improves.  The pseudopotential column")
print("converges, because there is no cusp left to resolve.")
print("\nRemaining error is basis-set incompleteness -- a minimal valence")
print("s+p shell per atom -- not the core.  It shrinks with the grid rather")
print("than growing, which is what makes relaxation tractable in principle.")

# --------------------------------------------------------------------------- #
# 4. H2O, valence only.
# --------------------------------------------------------------------------- #

print(f"\n{RULE}")
print("4. H2O with a valence-only Hamiltonian")
print(RULE)
water = Atoms("OH2", positions=[[0.0, 0.0, 0.0],
                                [0.0, 0.7634, 0.5921],
                                [0.0, -0.7634, 0.5921]])
water_grid = Grid(center=water.get_positions().mean(axis=0), box_size=8.0,
                  h=0.15)
start = time.perf_counter()
water.calc = QuantumCalculator(method="adapt-vqe", basis="FAO", grid=water_grid,
                               pseudopotentials=True, pool="qeb",
                               max_iterations=12, gradient_tolerance=1e-3,
                               verbose=False, profile=False)
energy = water.get_potential_energy()
print(f"  E = {energy:.4f} eV   {water.calc.n_qubits} qubits   "
      f"num_particles = {water.calc.num_particles}   "
      f"{time.perf_counter() - start:.1f}s")
print("  valence space: O(2s + 2p) + 2 x H(1s) = 6 orbitals, 8 electrons")
print("  the O 1s pair never enters the calculation at all.")

# --------------------------------------------------------------------------- #
# 5. Plot.
# --------------------------------------------------------------------------- #

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:                                     # pragma: no cover
    raise SystemExit("matplotlib is not installed; the tables were still printed")

fig, axis = plt.subplots(figsize=(7.5, 5.0))
axis.semilogy(spacings, all_electron, marker="o", markersize=7, lw=2.2,
              color="#D55E00", label="all-electron (frozen core)")
axis.semilogy(spacings, pseudo, marker="s", markersize=7, lw=2.2,
              color="#009E73", label="norm-conserving pseudopotential")
axis.invert_xaxis()
axis.set_xlabel("grid spacing h (Angstrom)   [finer to the right]")
axis.set_ylabel("spurious force on an isolated O atom (eV/Angstrom)")
axis.set_title("Exact answer is zero: the pseudopotential converges,\n"
               "the all-electron calculation does not")
axis.grid(True, which="both", color="0.92", lw=0.8)
axis.legend(frameon=False)
for spine in ("top", "right"):
    axis.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(PNG_PATH, dpi=150)
print(f"\nwrote {PNG_PATH}")

print(f"\n{RULE}")
print("STATUS")
print(RULE)
print("Working: the library (H-Ac, Z < 90), the valence-only Hamiltonian, the")
print("Kleinman-Bylander nonlocal term (C-accelerated), the calculator argument,")
print("and forces that now converge with grid refinement instead of diverging.")
print()
print("Not yet good enough for production geometry optimization: the residual")
print("force on an isolated atom is still ~30 eV/A at h = 0.10 A, limited by")
print("the minimal valence basis (one s + one p shell per atom).  A polarized")
print("multiple-zeta basis is now available -- basis={'name': 'NAO',")
print("'size': 'DZP'} -- and example 21 measures what it buys.")
print(RULE)
