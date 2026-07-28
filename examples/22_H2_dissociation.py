# -*- coding: utf-8 -*-
# file: examples/22_H2_dissociation.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""H2 potential energy curve with ADAPT-VQE and VASQE, referenced to the atoms.

Scans the H--H bond length with the :class:`~carcara.algorithms.QuantumCalculator`
(``method="adapt-vqe"`` and ``method="vasqe"``) and references every point to the
**sum of the isolated hydrogen atom energies**, so ``E = 0`` is the
separated-atom limit and the well depth is the binding energy:

.. math::

    \Delta E(R) = E_{\mathrm{H}_2}(R) - 2\,E_{\mathrm{H}} .

The atomic reference is computed with unrestricted Hartree-Fock (exact for a
one-electron atom within the basis and grid) **on the same grid, at the same
grid alignment** as the molecular scan, and the bond lengths are stepped by
twice the grid spacing (:func:`pes_utils.commensurate_distances`) -- see
:mod:`pes_utils` for why both measures are needed on a uniform real-space grid.
The same fixed grid is passed to the calculator for every distance
(``grid=...``), so molecule and reference are integrated identically.

Output
------
``examples/data/h2_dissociation.csv`` (bond length, absolute and atom-referenced
energies per method) and ``examples/data/h2_dissociation.png``.
"""

from __future__ import annotations

import csv
import os

import numpy as np
from ase import Atoms

from carcara.algorithms import QuantumCalculator
from carcara.basis import BasisSet
from carcara.units import HARTREE_TO_EV

from pes_utils import (GridSpec, atomic_reference, commensurate_distances,
                       molecule_positions)

# All generated files (logs, CSV, plots) go to examples/data/.
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

CSV_PATH = os.path.join(DATA, "h2_dissociation.csv")
PNG_PATH = os.path.join(DATA, "h2_dissociation.png")

#: The grid of the whole scan (shared by every geometry and by the reference).
GRID_SPEC = GridSpec(box_size=8.0, spacing=0.16)
#: Bond lengths stepped by 2h (0.42, 0.74, 1.06, ... A).
DISTANCES = commensurate_distances(0.42, 3.0, GRID_SPEC)

#: The two adaptive methods compared on the same curve.
METHOD_OPTIONS = {
    "adapt-vqe": {},
    "vasqe": {"temperature": 0.05, "seed": 7},
}
SOLVER = dict(pool="qeb", basis="FAO", optimizer="L-BFGS-B", verbose=False,
              profile=False, max_iterations=10, gradient_tolerance=1e-5)


def h2(distance: float) -> Atoms:
    """H2 on the z-axis about the origin (the scan grid is origin-centered)."""
    return Atoms("H2", positions=molecule_positions(distance))


# --------------------------------------------------------------------------- #
# 1. The absolute reference: two isolated hydrogen atoms on the same grid.
# --------------------------------------------------------------------------- #

grid = GRID_SPEC.build()
e_atoms = atomic_reference(["H", "H"], BasisSet.build("FAO"), grid,
                           molecule_positions(float(DISTANCES[0])))
print(f"reference: E(2 x H, UHF) = {e_atoms:+.6f} Ha "
      f"({e_atoms * HARTREE_TO_EV:+.4f} eV)")

# --------------------------------------------------------------------------- #
# 2. Scan the bond length with each method.
# --------------------------------------------------------------------------- #

curves: dict[str, np.ndarray] = {}
print(f"\n{'method':<12}{'d (A)':>8}{'E (Ha)':>14}{'E - E_atoms (eV)':>18}")
print("-" * 52)
for method, options in METHOD_OPTIONS.items():
    total = np.empty(len(DISTANCES))
    for i, distance in enumerate(DISTANCES):
        atoms = h2(float(distance))
        atoms.calc = QuantumCalculator(method=method, grid=grid,
                                       **SOLVER, **options)
        atoms.get_total_energy()
        total[i] = atoms.calc.result.optimal_energy          # Hartree
        print(f"{method:<12}{distance:>8.2f}{total[i]:>14.6f}"
              f"{(total[i] - e_atoms) * HARTREE_TO_EV:>18.4f}")
    curves[method] = total

# --------------------------------------------------------------------------- #
# 3. Consistency checks against the separated-atom reference.
# --------------------------------------------------------------------------- #

binding = {m: (curves[m] - e_atoms) * HARTREE_TO_EV for m in curves}
for method, rel in binding.items():
    i_min = int(np.argmin(rel))
    r_eq = float(DISTANCES[i_min])
    print(f"\n{method}: minimum {rel[i_min]:+.3f} eV at d = {r_eq:.2f} A "
          f"(experimental H2: 0.741 A, -4.75 eV)")
    # A bound minimum in the interior of the scan.  The fixed-zeta minimal FAO
    # basis overstretches the bond (the 1s exponent cannot contract on bonding),
    # so the minimum lands near ~1 A rather than at the experimental 0.741 A.
    assert 0 < i_min < len(DISTANCES) - 1, f"{method}: no interior minimum"
    assert 0.6 < r_eq < 1.2, f"{method}: equilibrium far off the H2 bond length"
    assert -8.0 < rel[i_min] < -2.0, f"{method}: unphysical well depth"
    # The curve must return to the separated-atom limit E = 0 at large R.
    assert abs(rel[-1]) < 0.4, f"{method}: no dissociation to the atoms"

# The two adaptive methods solve the same Hamiltonians: same curve.
spread = np.abs(curves["adapt-vqe"] - curves["vasqe"]).max()
assert spread < 1.6e-3, f"ADAPT-VQE and VASQE disagree by {spread:.2e} Ha"
print(f"\nADAPT-VQE and VASQE agree along the curve to {spread:.1e} Ha")

with open(CSV_PATH, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["distance_A", "e_atoms_Ha"]
                    + [f"{m}_Ha" for m in curves]
                    + [f"{m}_binding_eV" for m in curves])
    for i, d in enumerate(DISTANCES):
        writer.writerow([f"{d:.4f}", f"{e_atoms:.8f}"]
                        + [f"{curves[m][i]:.8f}" for m in curves]
                        + [f"{binding[m][i]:.6f}" for m in curves])
print(f"wrote {CSV_PATH}")

# --------------------------------------------------------------------------- #
# 4. Plot.
# --------------------------------------------------------------------------- #

try:
    import matplotlib
    matplotlib.use("Agg")                       # headless: write a PNG
    import matplotlib.pyplot as plt
except ImportError:                             # pragma: no cover
    raise SystemExit("matplotlib is not installed; the CSV was still written")

COLORS = {"adapt-vqe": "#0072B2", "vasqe": "#D55E00"}
MARKERS = {"adapt-vqe": "o", "vasqe": "s"}

fig, ax = plt.subplots(figsize=(7.0, 4.6))
ax.axhline(0.0, color="0.35", lw=1.2, ls="--", zorder=1)
ax.text(0.99, 0.02, "2 x H (separated atoms)", transform=ax.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=9, color="0.35")
for method, rel in binding.items():
    ax.plot(DISTANCES, rel, marker=MARKERS[method], markersize=6, lw=1.6,
            color=COLORS[method], markerfacecolor="none", markeredgewidth=1.5,
            label=method, zorder=3)
ax.set_xlabel("H--H distance  (Angstrom)")
ax.set_ylabel(r"$E - 2E_{\mathrm{H}}$  (eV)")
ax.set_title("H$_2$ dissociation referenced to the isolated atoms")
ax.grid(True, color="0.92", lw=0.8, zorder=0)
ax.legend(frameon=False)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(PNG_PATH, dpi=150)
print(f"wrote {PNG_PATH}")
