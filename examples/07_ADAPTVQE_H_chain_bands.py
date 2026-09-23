# -*- coding: utf-8 -*-
# file: examples/07_ADAPTVQE_H_chain_bands.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Periodic hydrogen chain: k-point-summed total energy + Bloch bands.

A **linear chain of hydrogen atoms 1.0 Angstrom apart**, periodic along *x* with a
10 Angstrom vacuum gap in *y* and *z*.  The geometry is an ASE ``Atoms`` object with
the **one-atom primitive cell** of the chain (one H per cell -> one 1s band); basis
**HAO**, fermion-to-qubit map **Jordan-Wigner**.

The calculation uses the periodic methods ``"bloch-vqe"`` /
``"bloch-adapt-vqe"``, reached through ``Mandacaru`` with a ``kpts`` mesh:

* **Quasiparticle bands** -- ``calc.get_spectral_function()`` builds the interacting
  spectral function ``A(E, k)`` from the correlated ground state, and
  ``calc.bands(spectral=...)`` reads off the dominant pole per k-point, with
  ``calc.band_weights(spectral=...)`` reporting how much of the peak it carries.
  It is resolved only at the mesh's own commensurate k-points -- there is no
  continuous band path for an interacting calculation, so a finer band means a
  larger ``kpts`` mesh, hence a larger supercell and more qubits.

* **Total energy using all k-points** -- ``atoms.get_potential_energy()``.  A
  correlated solver cannot be run independently per k-point and summed (the
  two-electron interaction couples crystal momenta), so this uses the Born-von
  Karman equivalence: an ``Nk``-point mesh is a Gamma-point calculation on the
  ``Nk``-cell supercell, and ``E/cell = E(supercell) / Nk``.  The correlated
  supercell energy is grown adaptively with ``method="bloch-adapt-vqe"``.
  Running ``Nk = 2, 4, 6`` converges it toward the bulk limit.

The quasiparticle band points are written to ``examples/data/h_chain_bands.csv``;
plotting is a **separate** script, ``examples/plot_h_chain_bands.py``.
"""

from __future__ import annotations

import csv
import os

import numpy as np
from ase import Atoms

from mandacaru.algorithms import Mandacaru

SPACING = 1.0            # H-H distance = lattice constant a (Angstrom)
VACUUM = 10.0            # y/z vacuum gap (Angstrom)
BAND_MESH = 4            # k-points for the quasiparticle band (= cells = qubits/2)
MESH_SERIES = (2, 4, 6)  # k-point counts for the total-energy convergence
# All generated files (logs, CSV, plots) go to examples/data/.
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)
CSV_PATH = os.path.join(DATA, "h_chain_bands.csv")

# One-atom primitive cell of the chain (periodic along x only).
def primitive():
    return Atoms("H", positions=[[0.0, 0.0, 0.0]],
                 cell=[[SPACING, 0.0, 0.0], [0.0, VACUUM, 0.0], [0.0, 0.0, VACUUM]],
                 pbc=[True, False, False])


settings = dict(basis="HAO", mapping="jordan_wigner")

# A throwaway calculator, just to report the lattice's dimension and band count --
# both are properties of the basis and the mesh, so they answer without a run.
# The actual bands come from the mesh built further down.
atoms = primitive()
atoms.calc = Mandacaru(method="bloch-vqe",
                       kpts={"size": (1, 1, 1), "gamma": True},
                       h=0.20, **settings)
bands = atoms.calc
print(f"Periodic H chain: {SPACING:.2f} A spacing (one-atom cell), "
      f"{VACUUM:.0f} A vacuum, {bands.dimension}-D, {bands.n_bands} band(s)")

# --- Total energy using ALL k-points (Born-von Karman supercell series). ----- #
print("\n--- Total energy using all k-points (ADAPT-VQE, BvK supercell) ---")
print("  Nk k-points  ==  Nk-cell supercell at Gamma;  E/cell = E(supercell) / Nk")
for n_k in MESH_SERIES:
    chain = primitive()
    # The mesh IS the supercell, so each member needs its own calculator.
    chain.calc = Mandacaru(method="bloch-adapt-vqe",
                           kpts={"size": (n_k, 1, 1), "gamma": True},
                           h=0.35, max_iterations=10, gradient_tolerance=1e-3,
                           trace=False, **settings)
    e_cell = chain.get_potential_energy()
    result = chain.calc.result
    print(f"  ({n_k:>2d}, 1, 1) mesh -> {n_k:>2d} atoms, "
          f"E/cell = {e_cell:+.4f} eV  (operators {result.num_operators})")
    assert result.optimal_energy < result.reference_energy, "ADAPT did not lower E"
print("  (the finite supercells converge toward the bulk total energy.)")

# --- Quasiparticle bands, from the interacting spectral function. ----------- #
# There is no continuous k-path to plot: a finite Born-von Karman supercell
# defines Bloch operators at its own commensurate k-points and nowhere else, so
# the band is as many points as the mesh has, and a finer one costs qubits.
band_atoms = primitive()
band_atoms.calc = Mandacaru(method="bloch-adapt-vqe",
                            kpts={"size": (BAND_MESH, 1, 1), "gamma": True},
                            h=0.35, max_iterations=10, gradient_tolerance=1e-3,
                            trace=False, **settings)
band_atoms.get_potential_energy()
spectral = band_atoms.calc.get_spectral_function()
band = band_atoms.calc.bands(spectral=spectral)[:, 0]
weight = band_atoms.calc.band_weights(spectral=spectral)[:, 0]
frac = band_atoms.calc.kpoints[:, 0]
order = np.argsort(frac)

print(f"\n--- Quasiparticle band ({BAND_MESH} k-points, from A(E,k)) ---")
print(f"  {'k (frac)':>9} {'k (1/A)':>9} {'E-E0 (eV)':>11} {'weight':>7}")
for index in order:
    print(f"  {frac[index]:>9.3f} {2.0 * np.pi * frac[index] / SPACING:>9.3f} "
          f"{band[index]:>11.3f} {weight[index]:>7.3f}")
print(f"  chemical potential {spectral.chemical_potential:+.3f} eV, "
      f"sum rule {spectral.sum_rule:.1e}")
print("  (a weight below 1 is correlation: the rest sits on satellites)")

# --- Save the band points to CSV (plotting is a separate script). ----------- #
with open(CSV_PATH, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["k_fractional", "k_invAng", "band_eV", "weight"])
    for index in order:
        writer.writerow([f"{frac[index]:.6f}",
                         f"{2.0 * np.pi * frac[index] / SPACING:.6f}",
                         f"{band[index]:.6f}", f"{weight[index]:.6f}"])
print(f"\nband points written to {CSV_PATH}")
