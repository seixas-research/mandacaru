# -*- coding: utf-8 -*-
# file: examples/generate_lih_pes.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Generate the LiH potential-energy curve (three basis sets) and export a CSV.

Scans the Li-H bond length from 1.0 to 3.5 A and computes the RHF total energy over
the minimal **hydrogenic**, native **STO-3G** and native **6-31G(d)** bases, each
referenced to the sum of isolated-atom energies (``E_Li + E_H``).  Results go to
``data/lih_pes_data.csv``; plot them with ``examples/plot_pes.py``.

.. warning::

   LiH has a tight lithium ``1s`` core that a *uniform* real-space grid cannot
   fully resolve.  The resulting grid error is a few tenths of a Hartree and is
   geometry-dependent, so it does not cancel exactly against the isolated-atom
   reference.  These curves are therefore **qualitative** -- read them for
   basis-set trends and the presence of a bound minimum, not for a quantitative
   well depth.  (H2, having no core, is quantitative; see ``generate_h2_pes.py``.)
   6-31G(d) LiH is the largest case (10 spatial orbitals) and is the slowest to
   run.

Run with::

    python examples/generate_lih_pes.py
"""

from __future__ import annotations

import os

from pes_utils import GridSpec, commensurate_distances, generate_pes

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")


os.makedirs(DATA_DIR, exist_ok=True)
grid = GridSpec(box_size=10.0, spacing=0.16)
distances = commensurate_distances(1.0, 3.5, grid)   # step = 2h = 0.32 A
csv_path = os.path.join(DATA_DIR, "lih_pes_data.csv")
print("Note: LiH well depth is grid-limited (Li 1s core); read qualitatively.\n")
generate_pes("LiH", ["Li", "H"], distances, grid, csv_path)
print("Done. Plot with:  python examples/plot_pes.py")
