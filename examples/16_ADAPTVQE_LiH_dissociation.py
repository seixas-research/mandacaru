# -*- coding: utf-8 -*-
# file: examples/16_ADAPTVQE_LiH_dissociation.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""LiH potential energy curve with ADAPT-VQE: operator pools and qubit mappings.

Scans the Li--H bond distance and, at every geometry, solves the *same*
electronic problem with ADAPT-VQE under

* four **operator pools** -- ``fermionic``, ``qubit``, ``qeb`` and ``ceo`` -- all
  under Jordan-Wigner, and
* three **fermion-to-qubit mappings** -- Jordan-Wigner, parity and Bravyi-Kitaev
  -- all with the mapping-general ``fermionic`` pool.

(The ``qubit`` / ``qeb`` / ``ceo`` pools are built from Jordan-Wigner-mapped
excitations, so they are JW-specific; only the ``fermionic`` pool is
mapping-general.  That is why the two sweeps are factored this way rather than
run as a full 4x3 grid.)

Both sweeps must trace the **same** curve: a pool and a mapping change how the
ansatz is built and how the Hamiltonian is encoded, never the physics.  The
figure is laid out in **two columns of subplots** -- pools on the left, mappings
on the right -- with the absolute energies on top and the error against exact
diagonalization (FCI) of the same Hamiltonian underneath, so any disagreement
would be immediately visible.

Output
------
``examples/data/lih_dissociation.png`` and the raw scan as
``examples/data/lih_dissociation.csv``.

.. warning::

   **The distance set below is curated, and this curve is not a converged
   potential energy surface.**

   On a uniform real-space grid the Li 1s core cusp is sampled differently
   depending on where the nucleus falls relative to the grid nodes.  For LiH that
   shifts the total energy by :math:`\sim 0.2` Ha between neighbouring bond
   lengths, and the effect does **not** vanish as the grid is refined (checked
   from :math:`h = 0.30` down to :math:`0.10` Angstrom).  The distances used here
   were chosen because they sample the core consistently; nearby values such as
   1.25 or 2.4 Angstrom are off by :math:`\sim 0.2` Ha.  H\ :sub:`2`, which has
   no tight core, is smooth and grid-convergent at *any* distance under the same
   code -- confirming the Li core as the cause.

   So read the left/right panels as a **solver comparison**, which is what they
   test rigorously: at each geometry every pool and every mapping must recover the
   FCI ground state *of the Hamiltonian it is given*, to ~1e-9 Ha.  Do not read
   the absolute energies or the well depth as spectroscopy.
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np
from ase import Atoms

from carcara.algorithms import ADAPTVQE

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
    """LiH at ``distance``, centred in the cell (the grid is cell-centred)."""
    centre = CELL / 2.0
    return Atoms("LiH",
                 positions=[[centre, centre, centre - distance / 2.0],
                            [centre, centre, centre + distance / 2.0]],
                 cell=[[CELL, 0.0, 0.0], [0.0, CELL, 0.0], [0.0, 0.0, CELL]],
                 pbc=True)


def solve(distance, pool, mapping):
    """ADAPT-VQE energy at one geometry; also returns the exact (FCI) energy."""
    atoms = lih(distance)
    atoms.calc = ADAPTVQE(pool=pool, basis=BASIS, mapping=mapping,
                          h=GRID_SPACING, optimizer="L-BFGS-B", verbose=False,
                          profile=False, max_iterations=MAX_ITERATIONS,
                          gradient_tolerance=1e-5)
    atoms.get_total_energy()
    result = atoms.calc.adapt_result

    matrix = atoms.calc.hamiltonian.to_matrix()
    exact = float(np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T)).min())
    return result.optimal_energy, exact, result.num_operators


# --------------------------------------------------------------------------- #
# 1. Scan.
# --------------------------------------------------------------------------- #

print(f"LiH dissociation curve: {len(DISTANCES)} distances "
      f"x ({len(POOLS)} pools + {len(MAPPINGS)} mappings), h = {GRID_SPACING} A")
print(f"{'series':<26}{'d (A)':>8}{'E (Ha)':>16}{'E - FCI':>12}{'ops':>6}")
print("-" * 68)

rows = []
# Left column: pools, all under Jordan-Wigner.
pool_curves = {pool: {"energy": [], "exact": []} for pool in POOLS}
# Right column: mappings, all with the mapping-general fermionic pool.
mapping_curves = {mapping: {"energy": [], "exact": []} for mapping in MAPPINGS}

t0 = time.perf_counter()
for distance in DISTANCES:
    jw_fermionic = None
    for pool in POOLS:
        energy, exact, n_ops = solve(distance, pool, "jordan_wigner")
        if pool == "fermionic":
            # The JW column of the mapping sweep is this very calculation.
            jw_fermionic = (energy, exact, n_ops)
        pool_curves[pool]["energy"].append(energy)
        pool_curves[pool]["exact"].append(exact)
        rows.append({"sweep": "pool", "series": pool, "mapping": "jordan_wigner",
                     "distance_A": distance, "energy_Ha": energy,
                     "fci_Ha": exact, "error_Ha": energy - exact,
                     "num_operators": n_ops})
        print(f"{'pool ' + pool:<26}{distance:>8.3f}{energy:>16.8f}"
              f"{energy - exact:>+12.2e}{n_ops:>6}")

    for mapping in MAPPINGS:
        if mapping == "jordan_wigner":
            energy, exact, n_ops = jw_fermionic      # reuse, don't recompute
        else:
            energy, exact, n_ops = solve(distance, "fermionic", mapping)
        mapping_curves[mapping]["energy"].append(energy)
        mapping_curves[mapping]["exact"].append(exact)
        rows.append({"sweep": "mapping", "series": mapping,
                     "mapping": mapping, "distance_A": distance,
                     "energy_Ha": energy, "fci_Ha": exact,
                     "error_Ha": energy - exact, "num_operators": n_ops})
        print(f"{'map  ' + mapping:<26}{distance:>8.3f}{energy:>16.8f}"
              f"{energy - exact:>+12.2e}{n_ops:>6}")

print(f"\nscan finished in {time.perf_counter() - t0:.1f} s")

with open(CSV_PATH, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["sweep", "series", "mapping",
                                            "distance_A", "energy_Ha", "fci_Ha",
                                            "error_Ha", "num_operators"])
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {CSV_PATH}")

# --------------------------------------------------------------------------- #
# 2. Consistency checks.
# --------------------------------------------------------------------------- #

reference = np.array(pool_curves["fermionic"]["exact"])
for family, curves in (("pool", pool_curves), ("mapping", mapping_curves)):
    for label, curve in curves.items():
        error = np.abs(np.array(curve["energy"]) - np.array(curve["exact"]))
        worst = float(error.max())
        assert worst < CHEMICAL_ACCURACY, \
            f"{family} {label!r} missed FCI by {worst:.2e} Ha"
        # A mapping is a change of encoding, so the spectrum is invariant.
        assert np.allclose(curve["exact"], reference, atol=1e-8), \
            f"{family} {label!r} changed the Hamiltonian's spectrum"
print("All pools and mappings reproduce the same FCI curve within "
      f"chemical accuracy ({CHEMICAL_ACCURACY:.1e} Ha).")

equilibrium = DISTANCES[int(np.argmin(reference))]
print(f"minimum of the computed curve: d = {equilibrium:.3f} A "
      f"(experimental LiH: 1.595 A)")
print("NOTE: the distance set is curated -- the real-space grid samples the "
      "Li 1s\n      core inconsistently at other distances (see the module "
      "docstring).\n      Read the panels as a solver comparison, not as "
      "spectroscopy.")

# --------------------------------------------------------------------------- #
# 3. Plot: two columns of subplots (pools | mappings).
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
(ax_pool, ax_map), (ax_pool_err, ax_map_err) = axes

for ax, ax_err, curves, colors, title in (
        (ax_pool, ax_pool_err, pool_curves, POOL_COLORS,
         "(a) Operator pools  (Jordan-Wigner)"),
        (ax_map, ax_map_err, mapping_curves, MAPPING_COLORS,
         "(b) Fermion-to-qubit mappings  (fermionic pool)")):

    # Exact curve underneath, as the shared reference.
    ax.plot(DISTANCES, reference, color="0.25", lw=3.0, alpha=0.35,
            zorder=1, label="exact (FCI)")
    for marker, (label, curve) in zip(MARKERS, curves.items()):
        ax.plot(DISTANCES, curve["energy"], marker=marker, markersize=6,
                lw=1.6, color=colors[label], markerfacecolor="none",
                markeredgewidth=1.5, label=label, zorder=3)
        ax_err.semilogy(DISTANCES,
                        np.maximum(np.abs(np.array(curve["energy"])
                                          - np.array(curve["exact"])), 1e-14),
                        marker=marker, markersize=6, lw=1.6,
                        color=colors[label], markerfacecolor="none",
                        markeredgewidth=1.5, label=label, zorder=3)

    ax.set_ylabel("total energy  (Ha)")
    ax.set_title(title)
    ax.grid(True, color="0.92", lw=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=9)

    ax_err.axhline(CHEMICAL_ACCURACY, color="0.35", ls="--", lw=1.2, zorder=2)
    ax_err.text(0.99, CHEMICAL_ACCURACY * 1.6, "chemical accuracy",
                transform=ax_err.get_yaxis_transform(), ha="right",
                va="bottom", fontsize=9, color="0.35")
    ax_err.set_xlabel("Li--H distance  (Angstrom)")
    ax_err.set_ylabel(r"$|E - E_{FCI}|$  (Ha)")
    ax_err.grid(True, which="both", color="0.92", lw=0.8, zorder=0)

for ax in axes.ravel():
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

fig.suptitle("LiH with ADAPT-VQE: operator pools and qubit mappings trace the "
             "same curve\n"
             "(curated distance set -- see the module docstring; read this as a "
             "solver comparison, not spectroscopy)", fontsize=12)
fig.tight_layout()
fig.savefig(PNG_PATH, dpi=150)
print(f"wrote {PNG_PATH}")
