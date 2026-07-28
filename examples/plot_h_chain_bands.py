# -*- coding: utf-8 -*-
# file: examples/plot_h_chain_bands.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Plot the periodic H-chain band structure from ``examples/data/h_chain_bands.csv``.

Companion to ``examples/07_ADAPTVQE_H_chain_bands.py``, which writes the band
points (dense Gamma--X path + Monkhorst-Pack samples) to CSV.  Keeping the plot in
its own script means the figure can be restyled without re-running the ADAPT-VQE /
Bloch calculation.

Usage::

    python examples/07_ADAPTVQE_H_chain_bands.py     # writes h_chain_bands.csv
    python examples/plot_h_chain_bands.py            # writes h_chain_bands.png
"""

from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")                      # headless; write a PNG
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)
CSV_PATH = os.path.join(DATA, "h_chain_bands.csv")
PNG_PATH = os.path.join(DATA, "h_chain_bands.png")


def load(csv_path):
    """Read the band CSV into ``{series: (k, band)}`` arrays."""
    rows = {"path": [], "mp": []}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows[row["series"]].append((float(row["k_invAng"]),
                                        float(row["band_eV"])))
    out = {}
    for series, data in rows.items():
        arr = np.array(sorted(data))
        out[series] = (arr[:, 0], arr[:, 1])
    return out


def main():
    if not os.path.exists(CSV_PATH):
        raise SystemExit(
            f"{CSV_PATH} not found -- run "
            "'python examples/07_ADAPTVQE_H_chain_bands.py' first.")
    data = load(CSV_PATH)
    kp, band_path = data["path"]
    km, band_mp = data["mp"]

    k_boundary = float(np.abs(kp).max())
    # Half filling (one electron per cell): Fermi level between the two central
    # Monkhorst-Pack band energies.
    sorted_mp = np.sort(band_mp)
    e_fermi = float(np.mean(sorted_mp[len(sorted_mp) // 2 - 1:
                                      len(sorted_mp) // 2 + 1]))

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(kp, band_path, color="#1f77b4", lw=2, label="1s band")
    ax.plot(km, band_mp, "o", color="#1f77b4", ms=6,
            label="Monkhorst-Pack k-points")
    ax.axhline(e_fermi, ls="--", color="gray", lw=1,
               label=f"Fermi level ({e_fermi:.1f} eV)")

    ax.set_xticks([-k_boundary, 0.0, k_boundary])
    ax.set_xticklabels([r"$-X$", r"$\Gamma$", r"$X$"])
    ax.set_xlabel(r"crystal momentum $k_x$")
    ax.set_ylabel("energy (eV)")
    ax.set_title("Periodic H chain (1.0 A) - FAO 1s Bloch band, Jordan-Wigner")
    ax.legend(loc="upper center", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)
    print(f"band structure written to {PNG_PATH}")


if __name__ == "__main__":
    main()
