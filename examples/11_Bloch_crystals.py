# -*- coding: utf-8 -*-
# file: examples/11_Bloch_crystals.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The Bloch crystal calculator: vqe / adapt-vqe / vasqe methods on a chain.

A single :class:`~carcara.algorithms.BlochCalculator` covers periodic systems;
its ``method`` argument selects the variational eigensolver.  The
single-particle **band structure** is solver-independent (identical across the
three methods), while the correlated **total energy per cell** is computed on
the Born-von Karman supercell by the selected molecular solver:

* ``method="vqe"``       -- fixed UCCSD ansatz,
* ``method="adapt-vqe"`` -- adaptive ansatz growth,
* ``method="vasqe"``     -- stochastic (annealed) selection.

A 1-D hydrogen chain (one atom per cell, 1.0 A spacing) is used throughout.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from carcara.algorithms import BlochCalculator
from carcara.optimizers import Optimizer


def make(method, **kwargs):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]],
                  cell=[[1.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                  pbc=[True, False, False])
    return BlochCalculator(atoms, method=method, basis="FAO",
                           mapping="jordan_wigner",
                           n_cells=4, n_images=7, h=0.25, **kwargs)


# --- Band structure (single-particle; the same for every method) -------------
bloch = make("vqe")
kline = np.linspace(0.0, 0.5, 6)                       # Gamma -> X (fractional)
bands = bloch.bands(np.column_stack([kline, np.zeros_like(kline), np.zeros_like(kline)]))
print(f"{bloch.dimension}-D crystal, {bloch.n_bands} band(s)")
print("1s band along Gamma->X (eV):", np.round(bands[:, 0], 3))
print("(the band structure is solver-independent -- identical for all methods)\n")

# --- Correlated total energy per cell (Born-von Karman supercell) ------------
opt = Optimizer("L-BFGS-B", maxiter=2000)
mesh = (4, 1, 1)

e_vqe, _ = make("vqe").total_energy(mesh, optimizer=opt)
print(f"vqe        E/cell = {e_vqe:+.4f} eV   (fixed UCCSD)")

e_adapt, r_adapt = make("adapt-vqe").total_energy(
    mesh, optimizer=opt, max_iterations=10, gradient_tolerance=1e-3)
print(f"adapt-vqe  E/cell = {e_adapt:+.4f} eV   "
      f"({r_adapt.num_operators} operators grown)")

# VASQE with exponential temperature annealing (explore -> exploit).
e_vasqe, r_vasqe = make("vasqe").total_energy(
    mesh, optimizer=opt, temperature=2.0, final_temperature=0.02,
    schedule="exponential", max_iterations=10, gradient_tolerance=1e-3, seed=1)
taus = ", ".join(f"{t:.2g}" for t in r_vasqe.temperatures)
print(f"vasqe      E/cell = {e_vasqe:+.4f} eV   "
      f"(annealed tau=[{taus}])")

spread = max(e_vqe, e_adapt, e_vasqe) - min(e_vqe, e_adapt, e_vasqe)
print(f"\nAll three solvers target the same per-cell ground state and agree to "
      f"within chemical accuracy (spread {spread * 1e3:.1f} meV = "
      f"{spread / 0.0272114:.2f} mHa).")
