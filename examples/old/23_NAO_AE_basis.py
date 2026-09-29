# -*- coding: utf-8 -*-
# file: examples/23_NAO_AE_basis.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""All-electron numerical atomic orbitals (NAO-AE).

The ``NAO-AE`` family builds every function from the atom itself: the
self-consistent LDA atom's occupied shells -- core included -- re-solved under
a smooth exponential-wall confinement, plus hydrogen-like *tiers* whose
effective charges are derived from the atom's valence radius rather than read
from a table (see :mod:`mandacaru.basis.nao_ae`).

.. code-block:: python

    atoms.calc = Mandacaru(basis={"name": "NAO-AE", "tier": 1})

What this script shows
----------------------
1. the radial functions of H and O per tier, with their sizes;
2. the confining wall and how the tier functions sit inside it;
3. the variational payoff at Hartree-Fock on H2 -- tier 0 (the bare LDA 1s)
   versus tier 1 (polarized, radially flexible) -- against the analytic HAO;
4. the qubit count each choice implies, via the dry-run estimator.

Output: ``examples/data/nao_ae_basis.png``.
"""

from __future__ import annotations

import os

import numpy as np
from ase import Atoms

from mandacaru.algorithms import estimate_qubits
from mandacaru.basis import BasisSet
from mandacaru.basis.nao_ae import confinement_potential
from mandacaru.core import MolecularIntegrals
from mandacaru.integrals import Grid
from mandacaru.units import BOHR_TO_ANGSTROM, from_hartree, to_bohr

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)
PNG_PATH = os.path.join(DATA, "nao_ae_basis.png")
RULE = "=" * 76

ONSET, WIDTH = 2.5, 1.0          # Angstrom: a wall at 3.5 A suits a 7 A box

# --------------------------------------------------------------------------- #
# 1. Species listings.
# --------------------------------------------------------------------------- #
print(RULE)
print("NAO-AE radial functions per tier")
print(RULE)
for symbol in ("H", "O"):
    for tier in (0, 1, 2):
        bset = BasisSet.build("NAO-AE", tier=tier, onset=ONSET, width=WIDTH)
        n = len(bset.atom(symbol))
        print(f"\n{symbol}, tier {tier}: {n} basis functions")
        print(bset.describe(symbol))

# --------------------------------------------------------------------------- #
# 2. The wall and the functions inside it.
# --------------------------------------------------------------------------- #
bset = BasisSet.build("NAO-AE", tier=1, onset=ONSET, width=WIDTH)
hydrogen = bset.species("H")
r0, w = to_bohr(ONSET, "angstrom"), to_bohr(WIDTH, "angstrom")

# --------------------------------------------------------------------------- #
# 3. Hartree-Fock payoff on H2.
# --------------------------------------------------------------------------- #
print("\n" + RULE)
print("H2 at 0.74 A, restricted Hartree-Fock, h = 0.20 A")
print(RULE)
positions = ([0.0, 0.0, -0.37], [0.0, 0.0, 0.37])
grid = Grid(center=[0, 0, 0], box_size=7.0, h=0.20)


def rhf_energy(basis):
    functions, nuclei = [], []
    for position in positions:
        functions += basis.atom("H", center=position, units="angstrom")
        nuclei.append((1.0, np.asarray(position)))
    integrals = MolecularIntegrals(
        nuclei, functions, grid,
        softening=0.5 * min(grid.dx, grid.dy, grid.dz))
    result = integrals.hartree_fock(2)
    # The SCF runs in Hartree; convert once, on return.
    return (float(from_hartree(result.electronic_energy
                               + integrals.nuclear_repulsion, "eV")),
            len(functions))


energies = {}
for label, basis in (("HAO (analytic 1s)", BasisSet.build("HAO")),
                     ("NAO-AE tier 0", BasisSet.build("NAO-AE", tier=0,
                                                       onset=ONSET, width=WIDTH)),
                     ("NAO-AE tier 1", bset)):
    energy, n_fn = rhf_energy(basis)
    energies[label] = energy
    print(f"  {label:<20s}  {n_fn:2d} functions   E_RHF = {energy:+.4f} eV")
gain = energies["NAO-AE tier 0"] - energies["NAO-AE tier 1"]
print(f"\n  tier 1 lowers the RHF energy by {gain:.3f} eV over tier 0")
assert energies["NAO-AE tier 1"] < energies["NAO-AE tier 0"]

# --------------------------------------------------------------------------- #
# 4. Qubit budget.
# --------------------------------------------------------------------------- #
print("\n" + RULE)
print("Qubits (dry run, no integrals)")
print(RULE)
h2 = Atoms("H2", positions=positions, cell=[7.0] * 3)
for tier in (0, 1, 2):
    est = estimate_qubits(h2, basis={"name": "NAO-AE", "tier": tier,
                                     "onset": ONSET, "width": WIDTH})
    print(f"  tier {tier}: {est.n_qubits:3d} qubits  "
          f"({est.n_basis_functions} spatial functions)")

# --------------------------------------------------------------------------- #
# Plot.
# --------------------------------------------------------------------------- #
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax_r, ax_v) = plt.subplots(1, 2, figsize=(10.5, 4.0))
r = hydrogen[0].r                                   # Bohr (radial table)
mask = r < r0 + w + 0.5
r_ang = r * BOHR_TO_ANGSTROM                        # plot in Angstrom
for f in hydrogen:
    ax_r.plot(r_ang[mask], f.u[mask], label=f.label)
ax_r.axvspan(ONSET, ONSET + WIDTH, color="0.85", label="wall ramp")
ax_r.set_xlabel(r"$r$ (Angstrom)")
ax_r.set_ylabel(r"$u(r) = r\,R(r)$")
ax_r.set_title("Hydrogen, NAO-AE tier 1 (orthonormalized)")
ax_r.legend(fontsize=8)

ax_v.plot(r_ang[mask], from_hartree(confinement_potential(r[mask], r0, w), "eV"))
ax_v.set_ylim(0, from_hartree(5.0, "eV"))
ax_v.axvline(ONSET, color="k", ls="--", lw=0.8)
ax_v.set_xlabel(r"$r$ (Angstrom)")
ax_v.set_ylabel(r"$v_\mathrm{cut}(r)$ (eV)")
ax_v.set_title(f"Exponential wall: onset {ONSET} A, width {WIDTH} A")
fig.tight_layout()
fig.savefig(PNG_PATH, dpi=150)
print(f"\nPlot written to {PNG_PATH}")
