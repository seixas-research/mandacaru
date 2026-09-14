# -*- coding: utf-8 -*-
# file: examples/plot_lih_ibm.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Plot the LiH curve of ``24_ADAPTVQE_LiH_IBM.py``: local state vector vs.
the energies measured on IBM hardware, from ``data/lih_dissociation_ibm.csv``."""

import csv
import os

import matplotlib.pyplot as plt

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CSV_PATH = os.path.join(DATA, "lih_dissociation_ibm.csv")
PNG_PATH = os.path.join(DATA, "lih_dissociation_ibm.png")

with open(CSV_PATH, newline="") as fh:
    rows = list(csv.DictReader(fh))

d = [float(r["distance_A"]) for r in rows]
local = [float(r["energy_local_eV"]) for r in rows]
hardware = [r for r in rows if r["energy_measured_eV"]]

fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.plot(d, local, "o-", color="#0072B2", label="local state vector (exact)")
if hardware:
    ax.errorbar([float(r["distance_A"]) for r in hardware],
                [float(r["energy_measured_eV"]) for r in hardware],
                yerr=[float(r["std_measured_eV"]) for r in hardware],
                fmt="s--", color="#D55E00", capsize=3,
                label=f"{hardware[0]['backend']} ({hardware[0]['shots']} shots, "
                      f"job {hardware[0]['job_id']})")
ax.set_xlabel("Li-H distance (Angstrom)")
ax.set_ylabel("energy (eV)")
ax.set_title("LiH, ADAPT-VQE (ceo pool, Jordan-Wigner)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(PNG_PATH, dpi=150)
print(f"wrote {PNG_PATH}")
