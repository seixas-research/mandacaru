# -*- coding: utf-8 -*-
# file: examples/new/03_DFT_H2O_functionals.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

# H2O with Kohn-Sham DFT: LDA, PBE+D4 and r2SCAN+D4 on one PAW-LCAO DZP basis.
#
# The three runs share the geometry, the grid and the basis, so the totals and
# the HOMO-LUMO gaps differ only through the exchange-correlation functional
# (and the D4 term).  Compare the totals only at one grid spacing: an absolute
# total on a real-space grid drifts with h, a difference at fixed h does not.
#
# The PAW-LCAO datasets are LDA datasets.  PBE and r2SCAN on them therefore
# print a RuntimeWarning (the molecule and its datasets use different
# functionals); the warning is expected here and is left visible on purpose.
# D4 needs the dftd4 package:  pip install 'mandacaru[dispersion]'

import os

from ase import units
from ase.build import molecule

from mandacaru import Mandacaru

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

# Molecule: H2O, 3 Angstrom of vacuum on every side
water = molecule("H2O")
water.center(vacuum=3.0)

BASIS = {"name": "PAW-LCAO", "size": "DZP"}

# (label, xc, dispersion)
RUNS = [("LDA", "lda", None),
        ("PBE+D4", "pbe", "d4"),
        ("r2SCAN+D4", "r2scan", "d4")]

print(f"{'functional':<11} {'E_total (Ha)':>14} {'D4 (mHa)':>10} "
      f"{'gap (eV)':>9} {'SCF iterations':>15}")

for label, xc, dispersion in RUNS:
    tag = xc + ("_" + dispersion if dispersion else "")
    atoms = water.copy()
    atoms.calc = Mandacaru(method="dft",                  # closed-shell Kohn-Sham
                           xc=xc,                         # "lda" | "pbe" | "r2scan"
                           dispersion=dispersion,         # None | "d4" (PBE and r2SCAN only)
                           basis=BASIS,
                           h=0.2,                         # Grid spacing (Angstrom)
                           txt=os.path.join(OUT, f"output_03_{tag}.txt"),
                           references=os.path.join(OUT, f"references_03_{tag}.bib"))

    energy = atoms.get_potential_energy()                 # eV
    scf = atoms.calc.result.scf                           # the Kohn-Sham determinant

    print(f"{label:<11} {energy / units.Hartree:>14.6f} "
          f"{1000.0 * scf.dispersion_energy:>10.3f} "
          f"{scf.homo_lumo_gap * units.Hartree:>9.3f} "
          f"{scf.n_iterations:>15d}")
