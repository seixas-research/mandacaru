# -*- coding: utf-8 -*-
# file: examples/plot_pes.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Plot the H2 and LiH potential-energy curves from their CSV files.

Reads ``data/h2_pes_data.csv`` and ``data/lih_pes_data.csv`` (produced by
``generate_h2_pes.py`` / ``generate_lih_pes.py``) and draws, for each molecule, the
three basis-set dissociation curves on a single graph -- relative energy (eV, with
``E = 0`` the separated-atom limit) versus bond length.  One PNG per available CSV
is written next to it in ``data/``.

Kept deliberately separate from the calculation scripts so the (slow) energy scans
run once and the plot styling can be tweaked freely.  Colours are the
colourblind-safe Okabe-Ito set, assigned to basis sets in a fixed order.

Run with::

    python examples/plot_pes.py
"""

from __future__ import annotations

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")

# Basis key -> (legend label, Okabe-Ito colourblind-safe colour, marker).
SERIES = [
    ("FAO", "FAO", "#0072B2", "o"),
    ("sto-3g", "STO-3G", "#E69F00", "s"),
    ("6-31g(d)", "6-31G(d)", "#009E73", "^"),
]


def _read_csv(path):
    """Return ``(header_comments, {column: np.array})`` for a PES CSV."""
    comments, rows, header = [], [], None
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                comments.append(",".join(row).lstrip("# "))
            elif header is None:
                header = row
            else:
                rows.append([float(x) for x in row])
    data = np.array(rows, dtype=float)
    return comments, {name: data[:, i] for i, name in enumerate(header)}


def plot_molecule(name, csv_path, out_path) -> bool:
    """Draw the three basis curves for one molecule; return True if plotted."""
    if not os.path.exists(csv_path):
        print(f"  (skip {name}: {os.path.basename(csv_path)} not found -- run the "
              f"generator first)")
        return False
    _, cols = _read_csv(csv_path)
    R = cols["distance_angstrom"]

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.axhline(0.0, color="0.6", lw=1.0, ls="--", zorder=1)  # separated-atom limit

    for key, label, color, marker in SERIES:
        col = f"{key}_rel_eV"
        if col not in cols:
            continue
        y = cols[col]
        good = np.isfinite(y)
        # Encode the well depth (and its bond length) into the legend label so
        # identity and the headline number travel together, no colliding
        # per-point annotations.
        legend_label = label
        min_point = None
        if good.any():
            i = int(np.nanargmin(np.where(good, y, np.nan)))
            min_point = (R[i], y[i])
            legend_label = f"{label}   (min {y[i]:.2f} eV @ {R[i]:.2f} Å)"
        ax.plot(R[good], y[good], color=color, lw=2.0, marker=marker,
                markersize=6, markeredgecolor="white", markeredgewidth=0.6,
                label=legend_label, zorder=3)
        if min_point is not None:
            ax.scatter([min_point[0]], [min_point[1]], s=90, facecolors="none",
                       edgecolors=color, linewidths=1.6, zorder=4)

    ax.set_xlabel("bond length  R  (Å)")
    ax.set_ylabel("energy relative to separated atoms  (eV)")
    ax.set_title(f"{name} dissociation curve — basis-set comparison (RHF)")
    ax.grid(True, color="0.9", lw=0.8, zorder=0)
    ax.legend(title="basis set", frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> wrote {out_path}")
    return True


print("Plotting potential-energy curves:")
plotted = 0
plotted += plot_molecule("H₂", os.path.join(DATA_DIR, "h2_pes_data.csv"),
                         os.path.join(DATA_DIR, "h2_pes.png"))
plotted += plot_molecule("LiH", os.path.join(DATA_DIR, "lih_pes_data.csv"),
                         os.path.join(DATA_DIR, "lih_pes.png"))
if not plotted:
    print("No CSVs found. Run generate_h2_pes.py / generate_lih_pes.py first.")
