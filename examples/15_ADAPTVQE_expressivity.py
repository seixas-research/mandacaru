# -*- coding: utf-8 -*-
# file: examples/15_ADAPTVQE_expressivity.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""How expressibility grows as ADAPT-VQE enlarges its parameterization.

The *expressibility* of a parameterized quantum circuit (Sim, Johnson &
Aspuru-Guzik, 2019) measures how uniformly its output states cover the accessible
Hilbert space.  It is scored by comparing the distribution of fidelities between
randomly parameterized state pairs,

.. math::

    F = \bigl|\langle\psi(\vec\theta_a)|\psi(\vec\theta_b)\rangle\bigr|^2 ,

with the Haar-random distribution :math:`P_{\text{Haar}}(F) = (d-1)(1-F)^{d-2}`,
via the Kullback-Leibler divergence

.. math::

    E = D_{\mathrm{KL}}\!\bigl(P_{\text{PQC}}\,\|\,P_{\text{Haar}}\bigr).

A **smaller** :math:`E` means a more expressive circuit; :math:`E = 0` is the
maximally expressive (Haar-like) limit.

This example runs ADAPT-VQE on LiH and records, after every accepted operator,
both the energy and the expressibility of the ansatz grown so far.  It shows the
central trade-off of an adaptive ansatz: each new operator buys correlation
energy *and* expressibility, until the ansatz saturates its symmetry sector.

Important: the Haar reference dimension is **not** :math:`2^N`
-----------------------------------------------------------
Carcará's fermionic ansätze conserve particle number and :math:`S_z`, so they
never leave the Hartree-Fock symmetry sector, whose dimension is
:math:`d = \binom{M}{n_\alpha}\binom{M}{n_\beta}` -- for LiH here,
:math:`d = \binom{3}{2}^2 = 9`, not :math:`2^6 = 64`.  Scoring against a
:math:`2^N`-dimensional Haar distribution would label *every* number-conserving
ansatz hopelessly inexpressive.  The fixed sector dimension is used throughout so
the scores are comparable across growth steps.

Output
------
``examples/data/adapt_expressivity.png`` -- a 2x2 figure: the expressibility and
energy growth curves, and the sampled PQC fidelity distributions at three ansatz
sizes overlaid on the analytical Haar density.  The trace is also written to
``examples/data/adapt_expressivity.csv``.
"""

from __future__ import annotations

import csv
import os

import numpy as np
from ase import Atoms

from carcara.algorithms import QuantumCalculator
from carcara.algorithms.expressivity import (active_space_dimension,
                                             calculate_kl_divergence,
                                             haar_density,
                                             sample_pqc_fidelities)

# All generated files (logs, CSV, plots) go to examples/data/.
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

HAMILTONIAN_FILE = os.path.join(DATA, "lih_expressivity.parquet")
CSV_PATH = os.path.join(DATA, "adapt_expressivity.csv")
PNG_PATH = os.path.join(DATA, "adapt_expressivity.png")

POOL = "qeb"
MAX_ITERATIONS = 10
NUM_SAMPLES = 3000          # fidelity pairs per snapshot
NUM_BINS = 60
#: Ansatz sizes whose full fidelity distribution is kept for the plot.
SNAPSHOT_STEPS = (1, 3, 8)

# --------------------------------------------------------------------------- #
# 1. Build LiH once and cache the Hamiltonian.
# --------------------------------------------------------------------------- #

atoms = Atoms("LiH",
              positions=[[7.5, 7.5, 7.5 - 0.7975], [7.5, 7.5, 7.5 + 0.7975]],
              cell=[[15.0, 0.0, 0.0], [0.0, 15.0, 0.0], [0.0, 0.0, 15.0]],
              pbc=True)
atoms.calc = QuantumCalculator(method="adapt-vqe", pool=POOL,
                               basis={"name": "FAO"}, h=0.25, verbose=False,
                               profile=False, max_iterations=1,
                               save_hamiltonian=HAMILTONIAN_FILE)
atoms.get_total_energy()

n_qubits = atoms.calc.n_qubits
num_particles = atoms.calc.num_particles
dimension = active_space_dimension(n_qubits, num_particles)

matrix = atoms.calc.hamiltonian.to_matrix()
exact = float(np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T)).min())

print(f"LiH: {n_qubits // 2} spatial orbitals ({n_qubits} qubits), "
      f"num_particles={num_particles}, pool={POOL!r}")
print(f"Haar reference dimension: d = {dimension} "
      f"(number-conserving sector, not 2^{n_qubits} = {2 ** n_qubits})")
print(f"exact FCI ground state  : {exact:.8f} Ha\n")

# --------------------------------------------------------------------------- #
# 2. Grow the ansatz, sampling expressibility after every operator.
# --------------------------------------------------------------------------- #

rng = np.random.default_rng(11)
history: list[dict] = []
snapshots: dict[int, np.ndarray] = {}


def record(info):
    """ADAPT-VQE callback: score the ansatz grown so far.

    ``info["ansatz"]`` is the live ansatz at its current size, so the fidelities
    must be sampled *now* -- it keeps growing after this returns.
    """
    ansatz = info["ansatz"]
    fidelities = sample_pqc_fidelities(ansatz, NUM_SAMPLES, rng)
    kl = calculate_kl_divergence(fidelities, n_qubits, NUM_BINS, dimension)
    step = int(info["iteration"])
    history.append({"step": step,
                    "num_parameters": int(info["num_operators"]),
                    "expressibility": kl,
                    "energy": float(info["energy"]),
                    "operator": info["operator_label"]})
    if step in SNAPSHOT_STEPS:
        snapshots[step] = fidelities


calc = QuantumCalculator(method="adapt-vqe", pool=POOL,
                         load_hamiltonian=HAMILTONIAN_FILE, verbose=False,
                         profile=False, optimizer="L-BFGS-B",
                         max_iterations=MAX_ITERATIONS, gradient_tolerance=1e-6)
result = calc.run(callback=record)

print(f"{'step':>5}  {'#params':>8}  {'E (Ha)':>15}  {'E - FCI':>11}  "
      f"{'expressibility':>15}  operator")
print("-" * 82)
for row in history:
    print(f"{row['step']:>5}  {row['num_parameters']:>8}  "
          f"{row['energy']:>15.8f}  {row['energy'] - exact:>+11.2e}  "
          f"{row['expressibility']:>15.4f}  {row['operator']}")

scores = [row["expressibility"] for row in history]
print(f"\nexpressibility: {scores[0]:.4f} (1 operator) -> {scores[-1]:.4f} "
      f"({len(scores)} operators);  lower = more expressive")
assert scores[-1] < scores[0], "the ansatz did not become more expressive"

with open(CSV_PATH, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["step", "num_parameters",
                                            "expressibility", "energy",
                                            "operator"])
    writer.writeheader()
    writer.writerows(history)
print(f"wrote {CSV_PATH}")

# --------------------------------------------------------------------------- #
# 3. Plot: growth curves + PQC fidelity distributions against Haar.
# --------------------------------------------------------------------------- #

try:
    import matplotlib
    matplotlib.use("Agg")                       # headless: write a PNG
    import matplotlib.pyplot as plt
except ImportError:                             # pragma: no cover
    raise SystemExit("matplotlib is not installed; the CSV was still written")

steps = np.array([row["step"] for row in history])
energies = np.array([row["energy"] for row in history])

fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
(ax_expr, ax_energy), (ax_dist, ax_overlay) = axes

# (a) Expressibility vs ansatz size.
ax_expr.plot(steps, scores, marker="o", markersize=6, lw=2.0, color="#009E73",
             markeredgecolor="white", markeredgewidth=0.6, zorder=3)
ax_expr.set_xlabel("number of ADAPT operators (= parameters)")
ax_expr.set_ylabel(r"expressibility  $E = D_{KL}(P_{PQC}\,\|\,P_{Haar})$")
ax_expr.set_title("(a) Expressibility improves as the ansatz grows")
ax_expr.grid(True, color="0.92", lw=0.8, zorder=0)

# (b) Energy convergence, for context: expressibility buys correlation energy.
ax_energy.semilogy(steps, np.maximum(np.abs(energies - exact), 1e-12),
                   marker="o", markersize=6, lw=2.0, color="#0072B2",
                   markeredgecolor="white", markeredgewidth=0.6, zorder=3)
ax_energy.axhline(1.6e-3, color="0.35", ls="--", lw=1.2, zorder=2)
ax_energy.text(0.99, 1.6e-3 * 1.4, "chemical accuracy",
               transform=ax_energy.get_yaxis_transform(), ha="right",
               va="bottom", fontsize=9, color="0.35")
ax_energy.set_xlabel("number of ADAPT operators")
ax_energy.set_ylabel(r"$|E - E_{FCI}|$  (Ha)")
ax_energy.set_title("(b) ... and so does the energy")
ax_energy.grid(True, which="both", color="0.92", lw=0.8, zorder=0)

# (c) Fidelity distributions at several ansatz sizes vs the Haar density.
grid = np.linspace(0.0, 1.0, 400)
palette = ["#E69F00", "#0072B2", "#009E73"]
for color, step in zip(palette, sorted(snapshots)):
    fidelities = snapshots[step]
    kl = next(r["expressibility"] for r in history if r["step"] == step)
    ax_dist.hist(fidelities, bins=NUM_BINS, range=(0.0, 1.0), density=True,
                 histtype="step", lw=2.0, color=color, zorder=3,
                 label=f"{step} operator{'s' if step > 1 else ''}  (E = {kl:.3f})")
ax_dist.plot(grid, haar_density(grid, dimension), color="#D55E00", lw=2.5,
             ls="--", zorder=4, label=f"Haar  (d = {dimension})")
ax_dist.set_xlabel(r"fidelity  $F = |\langle\psi(\theta_a)|\psi(\theta_b)\rangle|^2$")
ax_dist.set_ylabel("probability density")
ax_dist.set_title("(c) PQC fidelity distribution approaches Haar")
ax_dist.grid(True, color="0.92", lw=0.8, zorder=0)
ax_dist.legend(frameon=False, fontsize=9)
# The F ~ 0 bin is tall (a grown ansatz produces near-orthogonal states more
# often than Haar); clip the axis so the Haar curve stays legible.
ax_dist.set_ylim(0.0, 1.15 * (dimension - 1))

# (d) The largest ansatz alone, filled, against Haar -- the cleanest comparison.
largest = max(snapshots)
kl_largest = next(r["expressibility"] for r in history if r["step"] == largest)
ax_overlay.hist(snapshots[largest], bins=NUM_BINS, range=(0.0, 1.0),
                density=True, color="#0072B2", alpha=0.55, edgecolor="white",
                linewidth=0.5, zorder=2,
                label=f"PQC, {largest} operators  (E = {kl_largest:.3f})")
ax_overlay.plot(grid, haar_density(grid, dimension), color="#D55E00", lw=2.5,
                zorder=3, label=f"Haar  (d = {dimension})")
ax_overlay.set_xlabel(r"fidelity  $F$")
ax_overlay.set_ylabel("probability density")
ax_overlay.set_title("(d) Grown ansatz vs the Haar distribution")
ax_overlay.grid(True, color="0.92", lw=0.8, zorder=0)
ax_overlay.legend(frameon=False, fontsize=9)
ax_overlay.set_ylim(0.0, 1.15 * (dimension - 1))

for ax in axes.ravel():
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

fig.suptitle(f"ADAPT-VQE expressibility on LiH  ({n_qubits} qubits, "
             f"{POOL.upper()} pool, sector dimension d = {dimension})",
             fontsize=12)
fig.tight_layout()
fig.savefig(PNG_PATH, dpi=150)
print(f"wrote {PNG_PATH}")
