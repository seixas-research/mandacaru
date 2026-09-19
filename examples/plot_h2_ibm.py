# -*- coding: utf-8 -*-
# file: examples/plot_h2_ibm.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Overlay the H2 curves measured on IBM processors (``25_ADAPTVQE_H2_IBM.py``)
on the exact local curve, from the CSV files in ``data/``.

    python plot_h2_ibm.py data/h2_dissociation_ibm_fez.csv data/h2_dissociation_ibm_level2.csv
"""

import csv
import os
import sys

import matplotlib.pyplot as plt

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
files = sys.argv[1:] or [os.path.join(DATA, "h2_dissociation_ibm_fez.csv"),
                         os.path.join(DATA, "h2_dissociation_ibm_level2.csv")]
PNG_PATH = os.path.join(DATA, "h2_dissociation_ibm_compare.png")


def read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


fig, ax = plt.subplots(figsize=(6.4, 4.4))
first = read(files[0])
ax.plot([float(r["distance_A"]) for r in first],
        [float(r["energy_local_eV"]) for r in first], "o-", color="#0072B2",
        label="local state vector (exact)")
for path, color, marker in zip(files, ("#D55E00", "#009E73", "#CC79A7"), "s^D"):
    rows = [r for r in read(path) if r["energy_measured_eV"]]
    if not rows:
        continue
    ax.errorbar([float(r["distance_A"]) for r in rows],
                [float(r["energy_measured_eV"]) for r in rows],
                yerr=[float(r["std_measured_eV"]) for r in rows],
                fmt=marker + "--", color=color, capsize=3,
                label=f"{rows[0]['backend']} ({rows[0]['shots']} shots)")
ax.set_xlabel("H-H distance (Angstrom)")
ax.set_ylabel("energy (eV)")
ax.set_title("H2 (FAO), ADAPT-VQE, parity mapping, 2 qubits, resilience level 2")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(PNG_PATH, dpi=150)
print(f"wrote {PNG_PATH}")
