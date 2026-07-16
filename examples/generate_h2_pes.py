# -*- coding: utf-8 -*-
# file: examples/generate_h2_pes.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Generate the H2 potential-energy curve (three basis sets) and export a CSV.

Scans the H-H bond length from 0.5 to 2.5 A and computes the RHF total energy over

* the minimal analytic **hydrogenic** basis,
* the native minimal **STO-3G** Gaussian basis, and
* the native Pople split-valence **6-31G(d)** basis,

each referenced to the sum of isolated-atom energies (``2 * E_H``) so ``E = 0`` is
the two separated hydrogen atoms.  Results are written to ``data/h2_pes_data.csv``;
plot them with ``examples/plot_pes.py``.

H2 has no atomic core, so the uniform real-space grid resolves it well and these
curves are quantitative (the expected basis-set ordering
``hydrogenic > STO-3G > 6-31G(d)`` in energy is visible).

Run with::

    python examples/generate_h2_pes.py
"""

from __future__ import annotations

import os

import numpy as np

from pes_utils import GridSpec, generate_pes

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    # H has no atomic core, so the grid resolves it well at every geometry and a
    # fine, freely-spaced scan is both smooth and quantitative (unlike LiH, which
    # needs grid-commensurate spacing -- see generate_lih_pes.py).
    grid = GridSpec(box_size=7.0, spacing=0.12)
    distances = np.linspace(0.5, 2.5, 21)
    csv_path = os.path.join(DATA_DIR, "h2_pes_data.csv")
    generate_pes("H2", ["H", "H"], distances, grid, csv_path)
    print("Done. Plot with:  python examples/plot_pes.py")


if __name__ == "__main__":
    main()
