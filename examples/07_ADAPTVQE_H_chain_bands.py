# -*- coding: utf-8 -*-
# file: examples/07_ADAPTVQE_H_chain_bands.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Periodic hydrogen chain: k-point-summed total energy + Bloch bands.

A **linear chain of hydrogen atoms 1.0 Angstrom apart**, periodic along *x* with a
10 Angstrom vacuum gap in *y* and *z*.  The geometry is an ASE ``Atoms`` object with
the **one-atom primitive cell** of the chain (one H per cell -> one 1s band); basis
**FAO**, fermion-to-qubit map **Jordan-Wigner**.

The calculation uses the Bloch crystal driver family (all sharing the
:class:`carcara.algorithms.BlochVariationalDriver` base):

* **Band structure** -- ``bloch.bands(...)`` solves the single-particle generalized
  Bloch eigenproblem ``H(k) c = eps(k) S(k) c`` at each **ASE**-generated k-point (a
  dense Gamma--X path plus the Monkhorst-Pack mesh).  The band structure is
  *single-particle*, hence identical for every driver, so we use the lightweight
  fixed-ansatz :class:`~carcara.algorithms.BlochVQE`.

* **Total energy using all k-points** -- ``bloch.total_energy((Nk, 1, 1))``.  A
  correlated solver cannot be run independently per k-point and summed (the
  two-electron interaction couples crystal momenta), so this uses the Born-von
  Karman equivalence: an ``Nk``-point mesh is a Gamma-point calculation on the
  ``Nk``-cell supercell, and ``E/cell = E(supercell) / Nk``.  The correlated
  supercell energy is grown adaptively with
  :class:`~carcara.algorithms.BlochADAPTVQE`.  Running ``Nk = 2, 4, 6`` converges it
  toward the bulk limit; the requested ``(10, 1, 1)`` mesh is the ``Nk = 10``
  (20-qubit) member of the same call.

The band points are written to ``h_chain_bands.csv``; plotting is a **separate**
script, ``examples/plot_h_chain_bands.py``.
"""

from __future__ import annotations

import csv
import os

import numpy as np
from ase import Atoms

from carcara.algorithms import BlochADAPTVQE, BlochVQE

SPACING = 1.0            # H-H distance = lattice constant a (Angstrom)
VACUUM = 10.0            # y/z vacuum gap (Angstrom)
KSIZE = (10, 1, 1)       # Monkhorst-Pack mesh
MESH_SERIES = (2, 4, 6)  # k-point counts for the total-energy convergence
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "h_chain_bands.csv")

# One-atom primitive cell of the chain (periodic along x only).
atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]],
              cell=[[SPACING, 0.0, 0.0], [0.0, VACUUM, 0.0], [0.0, 0.0, VACUUM]],
              pbc=[True, False, False])
settings = dict(basis="FAO", mapping="jordan_wigner", n_cells=4, n_images=7, h=0.20)

# Correlated total energy -> adaptive driver; single-particle bands -> fixed-ansatz
# driver (bands are solver-independent, so BlochVQE is the cheapest choice).
adapt = BlochADAPTVQE(atoms, **settings)
bands = BlochVQE(atoms, **settings)
print(f"Periodic H chain: {SPACING:.2f} A spacing (one-atom cell), "
      f"{VACUUM:.0f} A vacuum, {bands.dimension}-D, {bands.n_bands} band(s)")

# --- Total energy using ALL k-points (Born-von Karman supercell series). ----- #
print("\n--- Total energy using all k-points (ADAPT-VQE, BvK supercell) ---")
print("  Nk k-points  ==  Nk-cell supercell at Gamma;  E/cell = E(supercell) / Nk")
for n_k in MESH_SERIES:
    e_cell, result = adapt.total_energy((n_k, 1, 1), h=0.35, max_iterations=10,
                                        gradient_tolerance=1e-3)
    print(f"  ({n_k:>2d}, 1, 1) mesh -> {n_k:>2d} atoms, "
          f"E/cell = {e_cell:+.4f} eV  (operators {result.num_operators})")
    assert result.optimal_energy < result.reference_energy, "ADAPT did not lower E"
print(f"  (the requested {KSIZE} mesh is the Nk=10 / 20-qubit member of this call;")
print("   the finite supercells converge toward the bulk total energy.)")

# --- Band structure: solve the Bloch Hamiltonian at each ASE k-point. -------- #
frac = np.linspace(-0.5, 0.5, 201)                      # fractional k along Gamma-X
k_path = 2.0 * np.pi * frac / SPACING                    # cartesian k_x (1/Angstrom)
band_path = bands.bands(np.c_[frac, 0 * frac, 0 * frac])[:, 0]

mp = bands.monkhorst_pack(KSIZE)                          # ASE Monkhorst-Pack mesh
k_mp = 2.0 * np.pi * mp[:, 0] / SPACING
band_mp = bands.bands(mp)[:, 0]

print("\n--- Bloch band structure (single 1s band, per k-point) ---")
print(f"  band edges: Gamma {band_path[len(band_path) // 2]:.3f} eV, "
      f"X {band_path[0]:.3f} eV  (width {band_path.max() - band_path.min():.3f} eV)")

# --- Save band points to CSV (plotting is a separate script). --------------- #
with open(CSV_PATH, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["series", "k_invAng", "band_eV"])
    for k, e in zip(k_path, band_path):
        writer.writerow(["path", f"{k:.6f}", f"{e:.6f}"])
    for k, e in zip(k_mp, band_mp):
        writer.writerow(["mp", f"{k:.6f}", f"{e:.6f}"])
print(f"\nband points written to {CSV_PATH}")
print("plot them with:  python examples/plot_h_chain_bands.py")
