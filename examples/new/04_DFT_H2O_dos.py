# -*- coding: utf-8 -*-
# file: examples/new/04_DFT_H2O_dos.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

# The Kohn-Sham eigenvalue spectrum of H2O as a molecular density of states.
#
# A molecule has discrete levels, so the "density of states" here is the list of
# Kohn-Sham eigenvalues with each level (two electrons per spatial orbital)
# smeared by a Gaussian of width SIGMA.  It is a picture of the orbital
# energies of one isolated molecule at the Gamma point: not a band structure,
# and the gap between the highest occupied and lowest unoccupied level is an
# orbital-energy gap, not an excitation energy.
#
# The PAW-LCAO datasets are LDA datasets, so XC = "lda" is the consistent
# choice; "pbe" or "r2scan" run too, with a RuntimeWarning about the mismatch.

import os

import matplotlib

matplotlib.use("Agg")                                     # write the file, no window
import matplotlib.pyplot as plt
import numpy as np
from ase import units
from ase.build import molecule

from mandacaru import Mandacaru

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

XC = "lda"
SIGMA = 0.3                                               # Gaussian broadening (eV)
ABOVE_HOMO = 20.0                                         # plotted range above the HOMO (eV)

# Molecule: H2O, 3 Angstrom of vacuum on every side
atoms = molecule("H2O")
atoms.center(vacuum=3.0)

atoms.calc = Mandacaru(method="dft",
                       xc=XC,
                       basis={"name": "PAW-LCAO", "size": "DZP"},
                       h=0.2,                             # Grid spacing (Angstrom)
                       txt=os.path.join(OUT, "output_04.txt"),
                       references=os.path.join(OUT, "references_04.bib"))

atoms.get_potential_energy()
scf = atoms.calc.result.scf

eigenvalues = scf.mo_energies * units.Hartree             # Kohn-Sham levels (eV)
n_occupied = scf.n_occupied                               # doubly occupied orbitals
homo = eigenvalues[n_occupied - 1]
lumo = eigenvalues[n_occupied]

low = eigenvalues[0] - 4.0 * SIGMA
high = homo + ABOVE_HOMO
energy = np.linspace(low, high, 2000)


def broadened(levels):
    """Sum of Gaussians, two electrons per spatial level (states / eV)."""
    levels = np.asarray(levels)[:, None]
    peaks = np.exp(-0.5 * ((energy[None, :] - levels) / SIGMA) ** 2)
    return 2.0 * peaks.sum(axis=0) / (SIGMA * np.sqrt(2.0 * np.pi))


occupied = broadened(eigenvalues[:n_occupied])
unoccupied = broadened(eigenvalues[n_occupied:])

fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.fill_between(energy, occupied, color="tab:blue", alpha=0.35,
                label="occupied")
ax.plot(energy, occupied, color="tab:blue")
ax.fill_between(energy, unoccupied, color="tab:orange", alpha=0.35,
                label="unoccupied")
ax.plot(energy, unoccupied, color="tab:orange")
ax.vlines(eigenvalues[eigenvalues < high], 0.0, -0.4, color="black", lw=0.8)
ax.axvline(homo, color="gray", ls="--", lw=0.8)
ax.axvline(lumo, color="gray", ls=":", lw=0.8)
ax.set_xlim(low, high)
ax.set_xlabel("Kohn-Sham eigenvalue (eV)")
ax.set_ylabel("density of states (states / eV)")
ax.set_title(f"H$_2$O, {XC.upper()}, Gaussian broadening {SIGMA} eV")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "04_h2o_dos.png"), dpi=200)

print(f"functional        : {XC.upper()}")
print(f"occupied levels   : {n_occupied}")
print(f"HOMO              : {homo:.3f} eV")
print(f"LUMO              : {lumo:.3f} eV")
print(f"HOMO-LUMO gap     : {lumo - homo:.3f} eV")
print(f"levels plotted    : {int(np.sum(eigenvalues < high))} of {len(eigenvalues)}")
