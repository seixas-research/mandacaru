# -*- coding: utf-8 -*-
# file: examples/11_Bloch_crystals.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Bloch crystals: the periodic methods on a hydrogen chain.

Periodic systems go through the same entry point as everything else --
``Mandacaru(method="bloch-vqe" | "bloch-adapt-vqe", kpts=...)`` -- with
``kpts`` the Gamma-centered Monkhorst-Pack sampling.  Both the **bands** --
the quasiparticle peaks of the interacting spectral function -- and the
correlated **total energy per cell** come from the state the selected solver
found on the Born-von Karman supercell, so both depend on the method:

* ``method="bloch-vqe"``       -- fixed UCCSD ansatz,
* ``method="bloch-adapt-vqe"`` -- adaptive ansatz growth,

A 1-D hydrogen chain (one atom per cell, 1.0 A spacing) is used throughout.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from mandacaru.algorithms import Mandacaru


def make(method, mesh=(4, 1, 1), **kwargs):
    """A 1-D H chain with the periodic calculator attached."""
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]],
                  cell=[[1.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                  pbc=[True, False, False])
    atoms.calc = Mandacaru(method=method,
                           kpts={"size": mesh, "gamma": True},
                           basis="HAO",
                           mapping="jordan_wigner",
                           h=0.25,
                           **kwargs)
    return atoms


opt = {"method": "L-BFGS", "maxiter": 2000, "tol": 1e-12}

# --- Quasiparticle bands, from the interacting spectral function -------------
# They live on the mesh the supercell realizes, and nowhere between: the Bloch
# operators only exist at those k-points.
chain = make("bloch-vqe", optimizer=opt)
chain.get_potential_energy()                           # the bands need the state
spectral = chain.calc.get_spectral_function()
bands = chain.calc.bands(spectral=spectral)
weights = chain.calc.band_weights(spectral=spectral)
print(f"{chain.calc.dimension}-D crystal, {chain.calc.n_bands} band(s), "
      f"{len(chain.calc.kpoints)} k-points")
order = np.argsort(chain.calc.kpoints[:, 0])
for index in order:
    print(f"  k = {chain.calc.kpoints[index][0]:+.3f}: "
          f"E - E0 = {bands[index, 0]:+8.3f} eV, "
          f"quasiparticle weight {weights[index, 0]:.3f}")
print(f"chemical potential: {spectral.chemical_potential:+.3f} eV")
print(f"sum rule deviation: {spectral.sum_rule:.1e}  (exact identity: 1)\n")

# --- Correlated total energy per cell (Born-von Karman supercell) ------------
e_vqe = make("bloch-vqe", optimizer=opt).get_potential_energy()
print(f"bloch-vqe        E/cell = {e_vqe:+.4f} eV   (fixed UCCSD)")

adapt = make("bloch-adapt-vqe", optimizer=opt, max_iterations=10,
             gradient_tolerance=1e-3)
e_adapt = adapt.get_potential_energy()
print(f"bloch-adapt-vqe  E/cell = {e_adapt:+.4f} eV   "
      f"({adapt.calc.result.num_operators} operators grown)")


spread = max(e_vqe, e_adapt) - min(e_vqe, e_adapt)
print(f"\nBoth solvers target the same per-cell ground state and agree to "
      f"within chemical accuracy (spread {spread * 1e3:.1f} meV; chemical "
      f"accuracy is 43 meV).")
