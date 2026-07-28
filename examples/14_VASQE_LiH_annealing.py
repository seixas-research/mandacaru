# -*- coding: utf-8 -*-
# file: examples/14_VASQE_LiH_annealing.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""LiH with VASQE and an **exponential** temperature-annealing schedule.

VASQE grows an ADAPT-style ansatz but *samples* the next operator from a softmax
of the pool gradients at a selection temperature :math:`\tau`:

.. math::

    P(i, \tau) = \frac{\exp(|g_i|/\tau)}{\sum_j \exp(|g_j|/\tau)} .

**Exponential annealing** cools :math:`\tau` geometrically from a high initial
value to a low final one over the growth iterations, so early steps *explore*
(any operator with appreciable gradient can be picked) and late steps *exploit*
(the largest gradient dominates, reproducing greedy ADAPT-VQE).  LiH is a good
demonstration: its QEB pool has several near-degenerate gradients, so a purely
greedy rule commits early, whereas annealed sampling can reach the same FCI
energy along a different operator sequence.

What this script shows
----------------------
1. the **convergence trace** -- energy and selection temperature per growth
   step -- for the exponential schedule, printed as a table;
2. a **comparison** against the greedy (``tau -> 0``), constant-temperature,
   linear and logarithmic schedules, all against the exact FCI energy of the same
   Hamiltonian;
3. a **convergence plot** (energy vs. ADAPT step, with the annealed temperature
   on a twin axis) written to ``examples/data/vasqe_lih_annealing.png``, plus the
   raw trace as CSV.

The Hamiltonian is built once and cached to Parquet, so the five schedules are
compared on *exactly* the same operator -- and cost almost nothing to re-run.

.. note::

   LiH on a uniform real-space grid is **qualitative** (the contracted Li 1s core
   is only partially resolved).  The invariant checked is self-consistent: every
   schedule must reach the FCI ground state *of the Hamiltonian it is given*.
"""

from __future__ import annotations

import csv
import os

import numpy as np
from ase import Atoms

from carcara.algorithms import QuantumCalculator

# All generated files (logs, CSV, plots) go to examples/data/.
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

HAMILTONIAN_FILE = os.path.join(DATA, "lih_vasqe.parquet")
CSV_PATH = os.path.join(DATA, "vasqe_lih_annealing.csv")
PNG_PATH = os.path.join(DATA, "vasqe_lih_annealing.png")

POOL = "qeb"
MAX_ITERATIONS = 14
SEED = 7
CHEMICAL_ACCURACY = 1.6e-3          # Ha

# --------------------------------------------------------------------------- #
# 1. Build LiH's Hamiltonian once and cache it.
# --------------------------------------------------------------------------- #

atoms = Atoms("LiH",
              positions=[[7.5, 7.5, 7.5 - 0.7975], [7.5, 7.5, 7.5 + 0.7975]],
              cell=[[15.0, 0.0, 0.0], [0.0, 15.0, 0.0], [0.0, 0.0, 15.0]],
              pbc=True)
atoms.calc = QuantumCalculator(method="vasqe", basis={"name": "FAO"}, h=0.25,
                               pool=POOL, verbose=True, profile=False,
                               max_iterations=1, temperature=1e-3,
                               save_hamiltonian=HAMILTONIAN_FILE)
atoms.get_total_energy()

hamiltonian = atoms.calc.hamiltonian.to_matrix()
exact = float(np.linalg.eigvalsh(0.5 * (hamiltonian
                                        + hamiltonian.conj().T)).min())
n_qubits = atoms.calc.n_qubits

print(f"LiH: {n_qubits // 2} spatial orbitals ({n_qubits} qubits), "
      f"num_particles={atoms.calc.num_particles}, pool={POOL!r}")
print(f"exact FCI ground state = {exact:.8f} Ha\n")


def run(label, **kwargs):
    """Run VASQE from the cached Hamiltonian and report its convergence."""
    calc = QuantumCalculator(method="vasqe", pool=POOL,
                             load_hamiltonian=HAMILTONIAN_FILE, verbose=True,
                             profile=False, optimizer="L-BFGS-B",
                             max_iterations=MAX_ITERATIONS,
                             gradient_tolerance=1e-5, seed=SEED, **kwargs)
    result = calc.run()
    error = result.optimal_energy - exact
    status = "FCI" if abs(error) < CHEMICAL_ACCURACY else "above FCI"
    print(f"{label:<34}E = {result.optimal_energy:+.8f} Ha  "
          f"({error:+.1e}, {status})  {result.num_operators} ops")
    return result


# --------------------------------------------------------------------------- #
# 2. The exponential schedule, step by step.
# --------------------------------------------------------------------------- #

print("Exponential annealing: tau 2.0 -> 0.001 over the growth steps")
annealed = run("VASQE exponential (tau 2->1e-3)", schedule="exponential",
               temperature=2.0, final_temperature=1e-3)

print(f"\n{'step':>5}  {'tau':>10}  {'E (Ha)':>15}  {'E - FCI':>11}  "
      f"{'max|grad|':>11}  operator")
print("-" * 82)
for step, iteration in enumerate(annealed.iterations):
    tau = (annealed.temperatures[step] if step < len(annealed.temperatures)
           else float("nan"))
    print(f"{step + 1:>5}  {tau:>10.4g}  {iteration.energy:>15.8f}  "
          f"{iteration.energy - exact:>+11.2e}  "
          f"{iteration.max_gradient:>11.3e}  {iteration.operator_label}")

# --------------------------------------------------------------------------- #
# 3. Compare every schedule on the same Hamiltonian.
# --------------------------------------------------------------------------- #

print("\nSchedule comparison (same Hamiltonian, same seed):")
runs = {
    "exponential": annealed,
    "greedy (tau=1e-6)": run("VASQE greedy limit (ADAPT-VQE)",
                             schedule="constant", temperature=1e-6),
    "constant tau=0.5": run("VASQE constant tau=0.5",
                            schedule="constant", temperature=0.5),
    "linear": run("VASQE linear (tau 2->1e-3)", schedule="linear",
                  temperature=2.0, final_temperature=1e-3),
    "logarithmic": run("VASQE logarithmic (tau 2->1e-3)",
                       schedule="logarithmic", temperature=2.0,
                       final_temperature=1e-3),
}

for label, result in runs.items():
    assert result.optimal_energy - exact < CHEMICAL_ACCURACY, \
        f"{label} did not reach chemical accuracy"
print("\nEvery schedule reached the FCI ground state within chemical accuracy.")

# The point of annealing: exploration can change *which* operators are chosen
# while still landing on the same energy.
greedy_ops = runs["greedy (tau=1e-6)"].operators
if annealed.operators != greedy_ops:
    print("Exponential annealing found a different operator sequence than "
          "greedy ADAPT-VQE, at the same energy:")
    print(f"  greedy      : {' -> '.join(greedy_ops[:5])} ...")
    print(f"  exponential : {' -> '.join(annealed.operators[:5])} ...")
else:
    print("Exponential annealing converged onto the greedy operator sequence.")

# --------------------------------------------------------------------------- #
# 4. Write the trace and plot the convergence.
# --------------------------------------------------------------------------- #

with open(CSV_PATH, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["schedule", "step", "temperature", "energy_Ha",
                     "error_Ha", "max_gradient", "operator"])
    for label, result in runs.items():
        for step, iteration in enumerate(result.iterations):
            tau = (result.temperatures[step]
                   if step < len(result.temperatures) else "")
            writer.writerow([label, step + 1, tau, f"{iteration.energy:.10f}",
                             f"{iteration.energy - exact:.3e}",
                             f"{iteration.max_gradient:.6e}",
                             iteration.operator_label])
print(f"\nwrote {CSV_PATH}")

try:
    import matplotlib
    matplotlib.use("Agg")                       # headless: write a PNG
    import matplotlib.pyplot as plt
except ImportError:                             # pragma: no cover
    raise SystemExit("matplotlib is not installed; the CSV was still written")

fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.0, 7.5), sharex=True,
                                  height_ratios=[2.0, 1.0])
colors = {"exponential": "#0072B2", "greedy (tau=1e-6)": "#D55E00",
          "constant tau=0.5": "#009E73", "linear": "#CC79A7",
          "logarithmic": "#E69F00"}

for label, result in runs.items():
    steps = np.arange(1, len(result.iterations) + 1)
    errors = np.abs(np.array(result.energy_history) - exact)
    # Clip at machine precision so the log axis stays readable.
    top.semilogy(steps, np.maximum(errors, 1e-12), marker="o", markersize=5,
                 lw=1.8, color=colors[label], label=label,
                 markeredgecolor="white", markeredgewidth=0.6, zorder=3)

top.axhline(CHEMICAL_ACCURACY, color="0.35", ls="--", lw=1.2, zorder=2)
top.text(0.99, CHEMICAL_ACCURACY * 1.4, "chemical accuracy (1.6 mHa)",
         transform=top.get_yaxis_transform(), ha="right", va="bottom",
         fontsize=9, color="0.35")
top.set_ylabel("|E - E$_{FCI}$|  (Ha)")
top.set_title("VASQE convergence on LiH: temperature schedules")
top.grid(True, which="both", color="0.92", lw=0.8, zorder=0)
top.legend(frameon=False, fontsize=9)

for label, result in runs.items():
    if not result.temperatures:
        continue
    steps = np.arange(1, len(result.temperatures) + 1)
    bottom.semilogy(steps, result.temperatures, marker="s", markersize=4,
                    lw=1.6, color=colors[label], label=label, zorder=3)
bottom.set_xlabel("ADAPT growth step")
bottom.set_ylabel(r"selection temperature $\tau$")
bottom.grid(True, which="both", color="0.92", lw=0.8, zorder=0)

for ax in (top, bottom):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

fig.tight_layout()
fig.savefig(PNG_PATH, dpi=150)
print(f"wrote {PNG_PATH}")
