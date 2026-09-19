# -*- coding: utf-8 -*-
# file: examples/plot_lih_relaxation.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Plot the LiH relaxation of example 28 over the VASP (PBE) energy curve.

Reads examples/data/relax.traj (written by 28_LiH_relaxation_PAW.py) and
examples/data/lih_pbe_scan_vasp.csv; each curve is shown relative to its own
minimum, since the two methods have different absolute energies.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# VASP scan and its minimum (quartic fit around the lowest point).
d_vasp, e_vasp = np.loadtxt(os.path.join(DATA, "lih_pbe_scan_vasp.csv"),
                            delimiter=",", skiprows=1, usecols=(1, 2), unpack=True)
near = (d_vasp >= 1.3) & (d_vasp <= 2.0)
fit = np.polyfit(d_vasp[near], e_vasp[near], 4)
d_fine = np.linspace(1.3, 2.0, 2001)
d_eq_vasp = d_fine[np.argmin(np.polyval(fit, d_fine))]
e_min_vasp = np.polyval(fit, d_fine).min()

# Mandacaru relaxation steps.
steps = read(os.path.join(DATA, "relax.traj"), index=":")
d_steps = np.array([a.get_distance(0, 1) for a in steps])
e_steps = np.array([a.get_potential_energy() for a in steps])
e_steps -= e_steps[-1]

plt.plot(d_vasp, e_vasp - e_min_vasp, "o-", color="gray", ms=4,
         label="VASP, PBE (plane waves)")
plt.plot(d_steps, e_steps, "--", color="tab:red", lw=0.8)
plt.scatter(d_steps, e_steps, c=np.arange(len(steps)), cmap="autumn", zorder=3,
            edgecolors="k", label="Mandacaru BFGS steps (ADAPT-VQE, PAW-DZP)")
labeled = None                  # label steps that moved, not the converged cluster
for i, (d, e) in enumerate(zip(d_steps, e_steps)):
    if labeled is None or abs(d - labeled) > 0.015 or i == len(steps) - 1:
        plt.annotate(str(i), (d, e), textcoords="offset points", xytext=(4, 4), fontsize=7)
        labeled = d
plt.axvline(d_eq_vasp, color="gray", ls=":", label=f"PBE minimum {d_eq_vasp:.3f} Å")
plt.axvline(d_steps[-1], color="tab:red", ls=":",
            label=f"Mandacaru relaxed {d_steps[-1]:.3f} Å")
plt.xlim(1.2, 2.5)
plt.ylim(-0.05, 1.0)
plt.xlabel("Li–H distance (Å)")
plt.ylabel("E − E_min (eV)")
plt.title("LiH relaxation from 2.0 Å")
plt.legend(fontsize=8)
plt.savefig(os.path.join(DATA, "lih_relaxation_vs_vasp.png"), dpi=150)
print(f"PBE minimum {d_eq_vasp:.4f} A, Mandacaru relaxed {d_steps[-1]:.4f} A "
      f"({len(steps) - 1} steps)")
