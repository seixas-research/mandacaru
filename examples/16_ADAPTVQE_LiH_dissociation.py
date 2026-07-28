# -*- coding: utf-8 -*-
# file: examples/16_ADAPTVQE_LiH_dissociation.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""LiH potential energy curve with ADAPT-VQE: operator pools and qubit mappings.

Scans the Li--H bond distance and, at every geometry, solves the *same*
electronic problem with ``QuantumCalculator(method="adapt-vqe", ...)`` under

* four **operator pools** -- ``fermionic``, ``qubit``, ``qeb`` and ``ceo`` -- all
  under Jordan-Wigner, and
* three **fermion-to-qubit mappings** -- Jordan-Wigner, parity and Bravyi-Kitaev
  -- all with the mapping-general ``fermionic`` pool.

(The ``qubit`` / ``qeb`` / ``ceo`` pools are built from Jordan-Wigner-mapped
excitations, so they are JW-specific; only the ``fermionic`` pool is
mapping-general.  That is why the two sweeps are factored this way rather than
run as a full 4x3 grid.)

Every energy is referenced to the **sum of the isolated-atom energies**
``E(Li) + E(H)`` (unrestricted Hartree-Fock in the same basis, on the same grid,
with the same Coulomb softening), so the lower panels show the **binding
energy** ``E - E_atoms`` and ``E = 0`` is the separated-atom limit.  Both sweeps
must trace the **same** curve: a pool and a mapping change how the ansatz is
built and how the Hamiltonian is encoded, never the physics.  The figure is
laid out in **two columns of subplots** -- pools on the left, mappings on the
right -- with the absolute energies on top and the binding energies underneath,
so any disagreement between solvers would be immediately visible.

Output
------
``examples/data/lih_dissociation.png`` and the raw scan as
``examples/data/lih_dissociation.csv``.

.. warning::

   **The distance set below is curated, and this curve is not a converged
   potential energy surface.**

   On a uniform real-space grid the Li 1s core cusp is sampled differently
   depending on where the nucleus falls relative to the grid nodes.  For LiH that
   shifts the total energy by :math:`\sim 0.2` Ha between neighboring bond
   lengths, and the effect does **not** vanish as the grid is refined (checked
   from :math:`h = 0.30` down to :math:`0.10` Angstrom).  The distances used here
   were chosen because they sample the core consistently; nearby values such as
   1.25 or 2.4 Angstrom are off by :math:`\sim 0.2` Ha.  H\ :sub:`2`, which has
   no tight core, is smooth and grid-convergent at *any* distance under the same
   code -- confirming the Li core as the cause.

   So read the panels as a **solver comparison**, which is what they test
   rigorously: at each geometry every pool and every mapping must produce the
   same ground-state energy of the Hamiltonian it is given, to within chemical
   accuracy.  Do not read the absolute energies or the well depth as
   spectroscopy.
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np
from ase import Atoms

from carcara.algorithms import QuantumCalculator
from carcara.basis import BasisSet
from carcara.integrals import Grid
from carcara.units import HARTREE_TO_EV

from pes_utils import atomic_reference

# All generated files (logs, CSV, plots) go to examples/data/.
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

CSV_PATH = os.path.join(DATA, "lih_dissociation.csv")
PNG_PATH = os.path.join(DATA, "lih_dissociation.png")

#: Li--H distances (Angstrom).  **Curated** -- see the module warning: these
#: sample the Li 1s core consistently on the grid, whereas nearby values do not.
DISTANCES = np.array([1.0, 1.3, 1.6, 1.9, 2.2, 2.6, 2.8])
POOLS = ("fermionic", "qubit", "qeb", "ceo")
MAPPINGS = ("jordan_wigner", "parity", "bravyi_kitaev")

#: Contracted Gaussians (STO-3G-like): smoother on a grid than the bare-Z FAO
#: cusp, which matters for a geometry scan.
BASIS = {"name": "GTO", "n_gaussians": 3}
CELL = 15.0                 # cubic cell edge (Angstrom)
GRID_SPACING = 0.15         # grid resolution h (Angstrom)
MAX_ITERATIONS = 14
CHEMICAL_ACCURACY = 1.6e-3  # Ha


def lih(distance):
    """LiH at ``distance``, centered in the cell (the grid is cell-centered)."""
    center = CELL / 2.0
    return Atoms("LiH",
                 positions=[[center, center, center - distance / 2.0],
                            [center, center, center + distance / 2.0]],
                 cell=[[CELL, 0.0, 0.0], [0.0, CELL, 0.0], [0.0, 0.0, CELL]],
                 pbc=True)


def solve(distance, pool, mapping):
    """ADAPT-VQE total energy (Hartree) and operator count at one geometry."""
    atoms = lih(distance)
    atoms.calc = QuantumCalculator(method="adapt-vqe", pool=pool, basis=BASIS,
                                   mapping=mapping, h=GRID_SPACING,
                                   optimizer="L-BFGS-B", verbose=False,
                                   profile=False, max_iterations=MAX_ITERATIONS,
                                   gradient_tolerance=1e-5)
    atoms.get_total_energy()
    result = atoms.calc.result
    return result.optimal_energy, result.num_operators


# --------------------------------------------------------------------------- #
# 1. The absolute reference: isolated Li + H on the same grid.
# --------------------------------------------------------------------------- #

# The calculator grids the cell centered on the molecule; the isolated atoms
# are placed at their first-geometry positions on an identical grid so the
# core sampling error cancels in the reference (see pes_utils).
center = CELL / 2.0
grid = Grid(center=[center, center, center], box_size=CELL, h=GRID_SPACING)
first = float(DISTANCES[0])
ref_positions = [np.array([center, center, center - first / 2.0]),
                 np.array([center, center, center + first / 2.0])]
basis_set = BasisSet.build("GTO", n_gaussians=3)
e_atoms = atomic_reference(["Li", "H"], basis_set, grid, ref_positions)
print(f"reference: E(Li) + E(H) (UHF) = {e_atoms:+.6f} Ha")

# --------------------------------------------------------------------------- #
# 2. Scan.
# --------------------------------------------------------------------------- #

print(f"\nLiH dissociation curve: {len(DISTANCES)} distances "
      f"x ({len(POOLS)} pools + {len(MAPPINGS)} mappings), h = {GRID_SPACING} A")
print(f"{'series':<26}{'d (A)':>8}{'E (Ha)':>16}{'E - E_atoms':>14}{'ops':>6}")
print("-" * 70)

rows = []
# Left column: pools, all under Jordan-Wigner.
pool_curves = {pool: [] for pool in POOLS}
# Right column: mappings, all with the mapping-general fermionic pool.
mapping_curves = {mapping: [] for mapping in MAPPINGS}

t0 = time.perf_counter()
for distance in DISTANCES:
    jw_fermionic = None
    for pool in POOLS:
        energy, n_ops = solve(distance, pool, "jordan_wigner")
        if pool == "fermionic":
            # The JW column of the mapping sweep is this very calculation.
            jw_fermionic = (energy, n_ops)
        pool_curves[pool].append(energy)
        rows.append({"sweep": "pool", "series": pool, "mapping": "jordan_wigner",
                     "distance_A": distance, "energy_Ha": energy,
                     "atoms_Ha": e_atoms,
                     "binding_eV": (energy - e_atoms) * HARTREE_TO_EV,
                     "num_operators": n_ops})
        print(f"{'pool ' + pool:<26}{distance:>8.3f}{energy:>16.8f}"
              f"{energy - e_atoms:>+14.5f}{n_ops:>6}")

    for mapping in MAPPINGS:
        if mapping == "jordan_wigner":
            energy, n_ops = jw_fermionic          # reuse, don't recompute
        else:
            energy, n_ops = solve(distance, "fermionic", mapping)
        mapping_curves[mapping].append(energy)
        rows.append({"sweep": "mapping", "series": mapping,
                     "mapping": mapping, "distance_A": distance,
                     "energy_Ha": energy, "atoms_Ha": e_atoms,
                     "binding_eV": (energy - e_atoms) * HARTREE_TO_EV,
                     "num_operators": n_ops})
        print(f"{'map  ' + mapping:<26}{distance:>8.3f}{energy:>16.8f}"
              f"{energy - e_atoms:>+14.5f}{n_ops:>6}")

print(f"\nscan finished in {time.perf_counter() - t0:.1f} s")

with open(CSV_PATH, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["sweep", "series", "mapping",
                                            "distance_A", "energy_Ha",
                                            "atoms_Ha", "binding_eV",
                                            "num_operators"])
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {CSV_PATH}")

# --------------------------------------------------------------------------- #
# 3. Consistency checks.
# --------------------------------------------------------------------------- #

# Every pool and every mapping solves the same physics: all series must trace
# one curve.  The mapping-general fermionic/JW series is the yardstick.
reference = np.array(pool_curves["fermionic"])
for family, curves in (("pool", pool_curves), ("mapping", mapping_curves)):
    for label, curve in curves.items():
        worst = float(np.abs(np.array(curve) - reference).max())
        assert worst < CHEMICAL_ACCURACY, \
            f"{family} {label!r} deviates from the shared curve by {worst:.2e} Ha"
print("All pools and mappings trace the same curve within chemical accuracy "
      f"({CHEMICAL_ACCURACY:.1e} Ha).")

binding_reference = (reference - e_atoms) * HARTREE_TO_EV
equilibrium = DISTANCES[int(np.argmin(reference))]
print(f"minimum of the computed curve: d = {equilibrium:.3f} A, "
      f"binding {binding_reference.min():+.3f} eV "
      f"(experimental LiH: 1.595 A)")
print("NOTE: the distance set is curated -- the real-space grid samples the "
      "Li 1s\n      core inconsistently at other distances (see the module "
      "docstring).\n      Read the panels as a solver comparison, not as "
      "spectroscopy.")

# --------------------------------------------------------------------------- #
# 4. Plot: two columns of subplots (pools | mappings).
# --------------------------------------------------------------------------- #

try:
    import matplotlib
    matplotlib.use("Agg")                       # headless: write a PNG
    import matplotlib.pyplot as plt
except ImportError:                             # pragma: no cover
    raise SystemExit("matplotlib is not installed; the CSV was still written")

POOL_COLORS = {"fermionic": "#0072B2", "qubit": "#D55E00",
               "qeb": "#009E73", "ceo": "#CC79A7"}
MAPPING_COLORS = {"jordan_wigner": "#0072B2", "parity": "#E69F00",
                  "bravyi_kitaev": "#009E73"}
MARKERS = ("o", "s", "^", "D")

fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), sharex=True)
(ax_pool, ax_map), (ax_pool_bind, ax_map_bind) = axes

for ax, ax_bind, curves, colors, title in (
        (ax_pool, ax_pool_bind, pool_curves, POOL_COLORS,
         "(a) Operator pools  (Jordan-Wigner)"),
        (ax_map, ax_map_bind, mapping_curves, MAPPING_COLORS,
         "(b) Fermion-to-qubit mappings  (fermionic pool)")):

    for marker, (label, curve) in zip(MARKERS, curves.items()):
        curve = np.array(curve)
        ax.plot(DISTANCES, curve, marker=marker, markersize=6,
                lw=1.6, color=colors[label], markerfacecolor="none",
                markeredgewidth=1.5, label=label, zorder=3)
        ax_bind.plot(DISTANCES, (curve - e_atoms) * HARTREE_TO_EV,
                     marker=marker, markersize=6, lw=1.6,
                     color=colors[label], markerfacecolor="none",
                     markeredgewidth=1.5, label=label, zorder=3)

    ax.set_ylabel("total energy  (Ha)")
    ax.set_title(title)
    ax.grid(True, color="0.92", lw=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=9)

    ax_bind.axhline(0.0, color="0.35", ls="--", lw=1.2, zorder=2)
    ax_bind.text(0.99, 0.02, "Li + H (separated atoms)",
                 transform=ax_bind.get_yaxis_transform(), ha="right",
                 va="bottom", fontsize=9, color="0.35")
    ax_bind.set_xlabel("Li--H distance  (Angstrom)")
    ax_bind.set_ylabel(r"$E - E_{\mathrm{atoms}}$  (eV)")
    ax_bind.grid(True, color="0.92", lw=0.8, zorder=0)

for ax in axes.ravel():
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

fig.suptitle("LiH with ADAPT-VQE: operator pools and qubit mappings trace the "
             "same curve\n"
             "(binding energies referenced to isolated Li + H; curated distance "
             "set -- see the module docstring)", fontsize=12)
fig.tight_layout()
fig.savefig(PNG_PATH, dpi=150)
print(f"wrote {PNG_PATH}")
