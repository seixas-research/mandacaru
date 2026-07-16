# -*- coding: utf-8 -*-
# file: examples/basis_sets.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Building localized basis sets with the ``BasisSet`` factory.

Demonstrates the two localized basis-set families supported by Carcará:

* **NAO** -- confined Numerical Atomic Orbitals (Sankey/SIESTA-type), whose
  cutoff radius follows from an ``energy_shift``;
* **GTO** -- a minimal STO-nG Contracted Gaussian basis, generated from scratch
  by least-squares fitting ``n_gaussians`` primitives to Slater-type orbitals
  (no tabulated basis-set data).

Both plug straight into :class:`~carcara.integrals.IntegralEngine`.  Lengths are
in Angstrom and energies in eV (the user-facing units).

Run with::

    python examples/basis_sets.py
"""

from __future__ import annotations

import numpy as np

from carcara.basis import BasisSet, energy_shift_to_rc
from carcara.integrals import Grid, IntegralEngine, Potentials

np.set_printoptions(precision=3, suppress=True)

# --- 1) NAO: confinement from the energy shift ---------------------------- #
print("== Numerical Atomic Orbitals (confined) ==")
for shift in (0.30, 0.03):
    print(f"  energy_shift = {shift:.2f} eV  ->  r_c = "
          f"{energy_shift_to_rc(shift):7.3f} Bohr")

nao = BasisSet.build(method="NAO", energy_shift=0.30)
h_nao = nao.atom("H", center=[0.0, 0.0, 0.0])
print(f"  H valence NAO: {len(h_nao)} orbital(s); "
      f"1s confined energy = {h_nao[0].energy:.4f} Ha, r_c = {h_nao[0].r_c:.2f} a0")

# --- 2) GTO: native STO-nG minimal basis, fit on the fly ------------------ #
print("\n== Gaussian-Type Orbitals (native STO-nG) ==")
for n_g in (2, 3, 6):
    gto = BasisSet.build(method="GTO", n_gaussians=n_g)
    print(f"  {gto.name:7s}: H -> {len(gto.atom('H')):2d}   C -> {len(gto.atom('C')):2d}   "
          f"O -> {len(gto.atom('O')):2d} basis functions "
          f"({n_g} primitives/contraction)")

# --- 3) Integrals over a GTO basis (H2, STO-3G) --------------------------- #
print("\n== One- and two-body integrals: H2 in STO-3G ==")
Z, R = 1.0, 0.74  # Angstrom
proton_a = np.array([0.0, 0.0, -R / 2])
proton_b = np.array([0.0, 0.0, +R / 2])

gto = BasisSet.build(method="GTO", n_gaussians=3)
basis = (gto.atom("H", center=proton_a) + gto.atom("H", center=proton_b))
potentials = Potentials([(Z, proton_a), (Z, proton_b)])
grid = Grid(center=[0.0, 0.0, 0.0], box_size=6.0, h=0.12)

engine = IntegralEngine(basis, grid)
print(f"  basis size: {len(basis)} functions;  C backend: {engine.uses_c_backend}")

T, V = engine.one_body(potentials.nuclear_potential)
h_core = T + V
eri = engine.two_body(method="fft")
print("  Core Hamiltonian h = T + V (eV), diagonal:")
print("   ", np.diag(h_core.real))
print(f"  <00|00> on-site repulsion = {eri[0, 0, 0, 0].real:.3f} eV")
