# -*- coding: utf-8 -*-
# file: examples/new/08_DFT_Si_bands.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

# Band structure, density of states and projected density of states of
# diamond silicon with Kohn-Sham DFT (LDA, PAW-LCAO DZP).
#
# One self-consistent run on a 4x4x4 Gamma-centered mesh fixes the potential.
# Everything after it is non-self-consistent, on that frozen potential:
#
#   * band_structure() diagonalizes H(k) at 80 points of the L-G-X-W-K-G path,
#     so the bands are continuous and are not limited to the SCF mesh;
#   * dos() and pdos() use a denser 8x8x8 mesh, reduced by the crystal's
#     symmetry, for a smooth curve.  Their energies are on the eigenvalue
#     reference, so the Fermi level is subtracted below.
#
# The gap of the SCF mesh is an upper bound: silicon's conduction-band minimum
# lies between Gamma and X, on no point of a small mesh, so the gap along the
# path is the smaller number.  Both are Kohn-Sham orbital-energy gaps, not
# quasiparticle gaps, and LDA underestimates them.
#
# Slow: expect minutes, not seconds.

import os

import matplotlib

matplotlib.use("Agg")                                     # write the file, no window
import matplotlib.pyplot as plt
import numpy as np
from ase.build import bulk

from mandacaru import Mandacaru

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

PATH = "LGXWKG"                                           # high-symmetry path
NPOINTS = 80                                              # k-points along it
DOS_KPTS = (8, 8, 8)                                      # denser mesh for the DOS
SIGMA = 0.15                                              # Gaussian broadening (eV)
WINDOW = (-14.0, 8.0)                                     # plotted range about E_F (eV)

# Crystal: diamond silicon, two atoms per cell
atoms = bulk("Si", "diamond", a=5.43)

atoms.calc = Mandacaru(method="dft",
                       xc="lda",
                       basis={"name": "PAW-LCAO", "size": "DZP"},
                       h=0.25,                            # Grid spacing (Angstrom)
                       kpts={"size": (4, 4, 4), "gamma": True},
                       txt=os.path.join(OUT, "output_08.txt"),
                       references=os.path.join(OUT, "references_08.bib"))

atoms.get_potential_energy()
fermi = atoms.calc.get_fermi_level()                      # eV, eigenvalue reference

# Gap on the SCF mesh (an upper bound): the levels of every irreducible k-point
mesh_levels = np.concatenate([atoms.calc.get_eigenvalues(kpt=k)
                              for k in range(len(atoms.calc.get_ibz_k_points()))])
mesh_gap = (mesh_levels[mesh_levels > fermi].min()
            - mesh_levels[mesh_levels < fermi].max())

# Bands along the path, non-self-consistent.  energies: (spin, k-point, band) in eV
bands = atoms.calc.band_structure(path=PATH, npoints=NPOINTS)
energies = bands.energies[0]
valence_top = energies[energies < fermi].max()
conduction_bottom = energies[energies > fermi].min()
path_gap = conduction_bottom - valence_top

# Density of states and its projection on the two atoms (all shells of each)
energy, dos = atoms.calc.dos(width=SIGMA, kpts=DOS_KPTS)
energy, pdos = atoms.calc.pdos(width=SIGMA, kpts=DOS_KPTS)
energy = energy - fermi
# The two Si atoms are equivalent, so split the DOS by angular momentum.
per_shell = {}
for (_atom, l), curve in pdos.items():
    per_shell[l] = per_shell.get(l, 0.0) + curve

fig, (ax_bands, ax_dos) = plt.subplots(
    1, 2, figsize=(8.0, 5.0), sharey=True,
    gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05})

bands.subtract_reference().plot(ax=ax_bands, emin=WINDOW[0],
                                emax=WINDOW[1])           # zero at the Fermi level
ax_bands.set_title(f"Si, LDA, path {PATH}")

ax_dos.fill_betweenx(energy, dos, color="lightgray", alpha=0.6)
ax_dos.plot(dos, energy, color="black", label="total")
for l, curve in sorted(per_shell.items()):
    ax_dos.plot(curve, energy, label="spd"[l], ls="--", lw=1.0)
ax_dos.axhline(0.0, color="gray", ls=":", lw=0.8)
ax_dos.set_xlim(left=0.0)
ax_dos.set_ylim(*WINDOW)
ax_dos.set_xlabel("states / eV per cell")
ax_dos.legend(frameon=False, fontsize=8)

fig.savefig(os.path.join(OUT, "08_si_bands.png"), dpi=200, bbox_inches="tight")

print(f"Fermi level       : {fermi:.3f} eV (eigenvalue reference)")
print(f"gap on the mesh   : {mesh_gap:.3f} eV (4x4x4, upper bound)")
print(f"gap on the path   : {path_gap:.3f} eV ({PATH}, {NPOINTS} points)")
