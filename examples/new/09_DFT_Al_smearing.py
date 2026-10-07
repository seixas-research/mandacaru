# -*- coding: utf-8 -*-
# file: examples/new/09_DFT_Al_smearing.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

# Occupation smearing in a metal: fcc aluminum with Kohn-Sham DFT (LDA,
# PAW-LCAO DZP).
#
# A metal has partly filled bands: the Fermi surface cuts through the
# Brillouin zone, and with sharp (0 or 1) occupations the energy jumps every
# time a band crosses the Fermi level between two k-points of the mesh.
# Smearing the occupations over a width sigma makes the energy a smooth
# function of the mesh, at the price of a fictitious electronic temperature:
#
#   * the variational quantity is the free energy F = E - sigma S, which
#     atoms.get_potential_energy(force_consistent=True) returns;
#   * atoms.get_potential_energy() returns the sigma -> 0 estimate.  For
#     Fermi-Dirac and Gaussian smearing F and E move away from the sharp
#     energy by about the same amount in opposite directions, quadratic in
#     sigma, so (E + F) / 2 cancels the leading term.  Methfessel-Paxton
#     occupations (first order) are built so that F itself has no
#     sigma^2 term, and are reported as they are.
#
# The sweep below solves the same crystal for three methods and four widths.
# On a mesh fine enough for the metal, the sigma -> 0 estimates move far less
# with sigma than F of the Fermi-Dirac and Gaussian runs, which bends away
# quadratically; at the widest smearing the next order shows, most for
# Fermi-Dirac, whose occupations have the longest tails.  A coarse mesh
# needs a wider smearing to hide its k-point sampling, which is where
# Methfessel-Paxton earns its place.  The density of states of the reference run (Methfessel-
# Paxton, 0.2 eV) is drawn on a denser mesh, non-self-consistently: no gap at
# the Fermi level, the signature of a metal.
#
# The lattice constant is the experimental one, not this model's equilibrium,
# so the absolute energies describe a slightly strained crystal; only their
# differences with sigma are the point here.
#
# Slow: twelve self-consistent runs, expect several minutes.

import os

import matplotlib

matplotlib.use("Agg")                                     # write the file, no window
import matplotlib.pyplot as plt
from ase.build import bulk

from mandacaru import Mandacaru

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

LATTICE = 4.05                                            # experimental a (Angstrom)
KPTS = {"size": (8, 8, 8), "gamma": True}                 # SCF mesh
METHODS = ("fermi-dirac", "gaussian", "methfessel-paxton")
WIDTHS = (0.05, 0.1, 0.2, 0.4)                            # smearing widths (eV)
REFERENCE = ("methfessel-paxton", 0.2)                    # the run that is logged
DOS_KPTS = (16, 16, 16)                                   # denser mesh for the DOS
SIGMA = 0.15                                              # DOS Gaussian broadening (eV)
WINDOW = (-12.0, 6.0)                                     # DOS range about E_F (eV)


def solve(method, width, **logging):
    """One self-consistent run; returns the crystal with its calculator."""
    atoms = bulk("Al", "fcc", a=LATTICE)                  # one atom per cell
    atoms.calc = Mandacaru(method="dft",
                           xc="lda",
                           basis={"name": "PAW-LCAO", "size": "DZP"},
                           h=0.25,                        # Grid spacing (Angstrom)
                           kpts=KPTS,
                           smearing={"method": method, "width": width},
                           **logging)
    atoms.get_potential_energy()
    return atoms


rows = []                                                 # (method, width, E0, F, E_F)
reference = None
for method in METHODS:
    for width in WIDTHS:
        logged = (method, width) == REFERENCE
        atoms = solve(method, width, **(
            {"txt": os.path.join(OUT, "output_09.txt"),
             "references": os.path.join(OUT, "references_09.bib")}
            if logged else {}))
        estimate = atoms.get_potential_energy()           # sigma -> 0 (eV)
        free = atoms.get_potential_energy(force_consistent=True)
        rows.append((method, width, estimate, free,
                     atoms.calc.get_fermi_level()))
        if logged:
            reference = atoms

# Energies relative to the reference run's sigma -> 0 estimate, in meV
zero = next(r[2] for r in rows if (r[0], r[1]) == REFERENCE)

with open(os.path.join(OUT, "09_al_smearing.dat"), "w") as table:
    n1, n2, n3 = KPTS["size"]
    table.write(f"# fcc Al, LDA, PAW-LCAO DZP, {n1}x{n2}x{n3} "
                "Gamma-centered mesh\n")
    table.write("# energies in eV per cell; E0 = sigma -> 0 estimate, "
                "F = E - sigma S\n")
    table.write(f"# {'method':<18} {'width':>6} {'E0':>16} {'F':>16} "
                f"{'E_Fermi':>10}\n")
    for method, width, estimate, free, fermi in rows:
        table.write(f"  {method:<18} {width:6.3f} {estimate:16.8f} "
                    f"{free:16.8f} {fermi:10.4f}\n")

# Density of states of the reference run, non-self-consistent on a denser mesh
fermi = reference.calc.get_fermi_level()
energy, dos = reference.calc.dos(width=SIGMA, kpts=DOS_KPTS)
energy = energy - fermi

fig, (ax_energy, ax_dos) = plt.subplots(
    1, 2, figsize=(9.0, 4.0), gridspec_kw={"width_ratios": [3, 2],
                                           "wspace": 0.3})
for method, color in zip(METHODS, ("C0", "C1", "C2")):
    mine = [r for r in rows if r[0] == method]
    widths = [r[1] for r in mine]
    ax_energy.plot(widths, [1e3 * (r[2] - zero) for r in mine], "o-",
                   color=color, label=f"{method}, sigma -> 0")
    ax_energy.plot(widths, [1e3 * (r[3] - zero) for r in mine], "s--",
                   color=color, mfc="none", label=f"{method}, F")
ax_energy.axhline(0.0, color="gray", ls=":", lw=0.8)
ax_energy.set_xlabel("smearing width sigma (eV)")
ax_energy.set_ylabel("energy relative to the reference (meV / atom)")
ax_energy.set_title("fcc Al: free energy and its sigma -> 0 estimate")
ax_energy.legend(frameon=False, fontsize=7)

inside = (energy >= WINDOW[0]) & (energy <= WINDOW[1])
ax_dos.fill_between(energy[inside], dos[inside], color="lightgray", alpha=0.6)
ax_dos.plot(energy[inside], dos[inside], color="black")
ax_dos.axvline(0.0, color="gray", ls=":", lw=0.8)
ax_dos.set_xlim(*WINDOW)
ax_dos.set_ylim(bottom=0.0)
ax_dos.set_xlabel("E - E_F (eV)")
ax_dos.set_ylabel("states / eV per cell")
ax_dos.set_title(f"DOS ({REFERENCE[0]}, {REFERENCE[1]} eV)")

fig.savefig(os.path.join(OUT, "09_al_smearing.png"), dpi=200,
            bbox_inches="tight")

print(f"{'method':<18} {'sigma (eV)':>10} {'E0 - ref (meV)':>15} "
      f"{'F - ref (meV)':>14}")
for method, width, estimate, free, _fermi in rows:
    print(f"{method:<18} {width:10.3f} {1e3 * (estimate - zero):15.3f} "
          f"{1e3 * (free - zero):14.3f}")
