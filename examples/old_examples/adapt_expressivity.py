# -*- coding: utf-8 -*-
# file: examples/adapt_expressivity.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Track the expressibility of a PQC as ADAPT-VQE grows it (linear H4).

Expressibility (Sim et al., 2019) measures how uniformly a parameterized circuit's
output states cover the accessible Hilbert space -- scored by the KL divergence of
its random-parameter fidelity distribution from the Haar distribution (lower =
more expressive).  This example:

1. builds the H4 chain Hamiltonian in the Hartree-Fock MO basis (8 qubits, 4
   electrons);
2. runs ADAPT-VQE with the fermionic and QEB pools, attaching an
   ``ADAPTExpressivityTracker`` so expressibility is measured after every operator
   the ansatz gains;
3. plots (a) expressibility E vs number of ADAPT operators -- the "grows then
   saturates" curve -- and (b) the final ansatz's fidelity distribution against the
   analytical Haar curve.

Because the fermionic/QEB ansätze conserve particle number and Sz, the Haar
reference uses the number-conserving sector dimension
``d = C(M, n_alpha) * C(M, n_beta)`` (here 36), not the full 2**8 = 256 -- see
``carcara.algorithms.expressivity``.

Run with::

    python examples/adapt_expressivity.py
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib.pyplot as plt

from carcara.algorithms import (ADAPTVQE, active_space_dimension,
                                compute_expressibility, plot_expressivity_growth,
                                plot_fidelity_distribution,
                                track_adapt_expressivity)
from carcara.circuits import AdaptAnsatz
from carcara.core import MolecularIntegrals, minimal_fao_basis
from carcara.integrals import Grid

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")
os.makedirs(DATA_DIR, exist_ok=True)

POOL_COLORS = {"fermionic": "#0072B2", "qeb": "#009E73"}

# Linear H4 Hamiltonian in the RHF molecular-orbital (FAO) basis (8 qubits).
spacing = 1.0
zs = [(-1.5 + i) * spacing for i in range(4)]
nuclei = [(1.0, np.array([0.0, 0.0, z])) for z in zs]
grid = Grid(center=[0.0, 0.0, 0.0], box_size=9.0, h=0.22)
H = MolecularIntegrals(nuclei, minimal_fao_basis(nuclei), grid).molecular_hamiltonian(
    mo_basis=True, n_electrons=4)

num_particles = (2, 2)
d = active_space_dimension(8, num_particles)
print(f"Linear H4: 8 qubits, 4 electrons; number-conserving sector d = {d} "
      f"(full space would be {2 ** 8}).\n")

rng = np.random.default_rng(7)
histories = {}
finals = {}
for pool in ("fermionic", "qeb"):
    adapt = ADAPTVQE(H, pool, num_particles=num_particles,
                     n_spatial_orbitals=4, profile=False, verbose=False,
                     max_iterations=8, gradient_tolerance=1e-3)
    result, history = track_adapt_expressivity(
        adapt, num_samples=600, bins=75, rng=rng)
    histories[pool] = history
    finals[pool] = (adapt, result)
    print(f"== {pool} pool ==")
    for s in history:
        print(f"  #ops={s.num_operators:2d}  E={s.kl_divergence:7.4f}  "
              f"energy={s.energy:+.6f}  ({s.operator_label})")
    print()

# --- Plot 1: expressibility growth (both pools on one graph) -------------- #
fig, ax = plt.subplots(figsize=(7.0, 4.5))
for pool, history in histories.items():
    plot_expressivity_growth(history, ax=ax, color=POOL_COLORS[pool],
                             label=f"{pool} pool")
ax.set_title("Expressibility growth during ADAPT-VQE (linear H₄)")
ax.legend(frameon=False)
fig.tight_layout()
growth_path = os.path.join(DATA_DIR, "adapt_expressivity.png")
fig.savefig(growth_path, dpi=150)
plt.close(fig)
print(f"-> wrote {growth_path}")

# --- Plot 2: final-ansatz fidelity distribution vs Haar ------------------- #
adapt, _ = finals["fermionic"]
# Rebuild the fully-grown fermionic ansatz to measure its distribution.
ansatz = AdaptAnsatz(adapt.n_qubits, adapt.pool.occupied_orbitals)
by_label = {op.label: op for op in adapt.pool.operators()}
for label in finals["fermionic"][1].operators:
    ansatz.append(by_label[label])
res = compute_expressibility(ansatz, num_samples=4000, bins=60,
                             num_particles=num_particles, rng=rng)
print(f"\nFinal fermionic ansatz ({ansatz.num_parameters} operators): {res}")

ax = plot_fidelity_distribution(res)
ax.figure.tight_layout()
hist_path = os.path.join(DATA_DIR, "fidelity_distribution.png")
ax.figure.savefig(hist_path, dpi=150)
plt.close(ax.figure)
print(f"-> wrote {hist_path}")
print("\nLower E = more expressive; the curve falls then saturates as the "
      "ansatz grows.")
