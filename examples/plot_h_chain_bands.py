# -*- coding: utf-8 -*-
# file: examples/plot_h_chain_bands.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Plot the periodic H-chain band structure from ``examples/data/h_chain_bands.csv``.

Companion to ``examples/07_ADAPTVQE_H_chain_bands.py``, which writes the
**quasiparticle** band -- the dominant pole of the interacting spectral function
``A(E, k)`` at each of the mesh's own commensurate k-points, together with the
spectral weight that pole carries -- to CSV.  Keeping the plot in its own script
means the figure can be restyled without re-running the ADAPT-VQE calculation.

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
    """Read the quasiparticle band CSV: one row per mesh k-point."""
    k, band, weight = [], [], []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            k.append(float(row["k_invAng"]))
            band.append(float(row["band_eV"]))
            weight.append(float(row["weight"]))
    order = np.argsort(k)
    return (np.asarray(k)[order], np.asarray(band)[order],
            np.asarray(weight)[order])


def main():
    if not os.path.exists(CSV_PATH):
        raise SystemExit(
            f"{CSV_PATH} not found -- run "
            "'python examples/07_ADAPTVQE_H_chain_bands.py' first.")
    k, band, weight = load(CSV_PATH)

    k_boundary = float(np.abs(k).max())
    # Half filling (one electron per cell): the chemical potential sits between
    # the two central k-points' quasiparticle energies.
    sorted_band = np.sort(band)
    e_fermi = float(np.mean(sorted_band[len(sorted_band) // 2 - 1:
                                        len(sorted_band) // 2 + 1]))

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(k, band, "--", color="#1f77b4", lw=1, alpha=0.6,
            label="quasiparticle band")
    # Marker area scales with the spectral weight the peak carries: a small
    # marker at a k-point means the "band" there is not a sharp excitation.
    ax.scatter(k, band, s=140 * weight, facecolors="none",
               edgecolors="#1f77b4", linewidths=1.4, zorder=3,
               label="commensurate k-points (size = weight)")
    ax.axhline(e_fermi, ls="--", color="gray", lw=1,
               label=f"chemical potential ({e_fermi:.1f} eV)")

    ax.set_xticks([-k_boundary, 0.0, k_boundary])
    ax.set_xticklabels([r"$-X$", r"$\Gamma$", r"$X$"])
    ax.set_xlabel(r"crystal momentum $k_x$ (1/A)")
    ax.set_ylabel(r"$E - E_0$ (eV)")
    ax.set_title("Periodic H chain (1.0 A) - quasiparticle 1s band, Jordan-Wigner")
    ax.legend(loc="upper center", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)
    print(f"band structure written to {PNG_PATH}")


if __name__ == "__main__":
    main()
