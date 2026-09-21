# -*- coding: utf-8 -*-
# file: examples/33_optimizer_comparison.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Every classical optimizer on one problem: LiH, qubit pool, ADAPT-VQE.

Every growth step of ADAPT-VQE ends in a classical minimization over the
ansatz parameters, and which optimizer runs it decides most of the classical
cost of a run.  This compares all of
:data:`~mandacaru.optimizers.NAMED_OPTIMIZERS` on the same 6-qubit LiH problem,
each configured through the ``optimizer={"method": ..., "maxiter": ...,
"tol": ...}`` dict so the iteration budget and the tolerance are the same for
all of them.

Two different currencies are reported, and they are not interchangeable:

* **steps** -- parameter updates, the optimizer's own moves
  (``OptimizeResult.nit``, summed over the growth steps and also available per
  step on ``result.iterations[i].optimizer_steps``);
* **cost evaluations** -- energy evaluations, what a QPU would be billed for
  (``result.num_evaluations``).

Set ``SYSTEM = "H2O"`` for the realistic check (12 qubits, a 640-operator pool,
~20 minutes): it **reverses the ranking**, because water stops on
``max_iterations`` rather than on the screening gradient and every method then
grows the same circuit.  Those measurements are committed as
``data/h2o_optimizers.csv`` (capped run) and ``data/h2o_optimizers_qeb.csv``
(run to convergence); the discussion is in
``docs/source/guide/optimizers.md``.

A gradient-based method takes few steps but spends several evaluations on each
one (a finite-difference gradient plus a line search); a direct-search method
takes many cheap steps.  The ``steps`` column of the ``output.txt``
``[ITERATIONS]`` table reports the first of those per growth step, and the
summary block the run's total.
"""

import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.build import molecule

from mandacaru import Mandacaru
from mandacaru.optimizers import (DEFAULT_MAXITER, DEFAULT_TOL,
                                  NAMED_OPTIMIZERS)
from mandacaru.units import HARTREE_TO_EV, from_hartree

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

# "LiH" runs in seconds and is the default; "H2O" is the realistic check --
# 12 qubits, a 640-operator pool, ~30 minutes for every method -- and it
# reverses the ranking (see docs/source/guide/optimizers.md).
SYSTEM = "LiH"

DISTANCE = 1.6                  # Angstrom (LiH)
CELL = 10.0                     # cubic cell edge (Angstrom)
# The shipped defaults, so the comparison is of methods and nothing else.
MAXITER = DEFAULT_MAXITER       # inner-optimizer iteration budget (1000)
TOL = DEFAULT_TOL               # inner-optimizer tolerance (1e-12)

SETUPS = {
    "LiH": dict(basis="FAO", h=0.30, pool="qubit", max_iterations=12),
    "H2O": dict(basis={"name": "PAW", "size": "SZ"}, h=0.25, pool="qubit",
                max_iterations=40),
}


def geometry():
    """The molecule under test, centered in a cubic cell."""
    if SYSTEM == "H2O":
        atoms = molecule("H2O")
        atoms.set_cell([CELL, CELL, CELL])
    else:
        atoms = Atoms("LiH",
                      positions=[[0.0, 0.0, 0.0], [0.0, 0.0, DISTANCE]],
                      cell=[CELL, CELL, CELL])
    atoms.pbc = True
    atoms.center()
    return atoms


def run(method):
    """One ADAPT-VQE run driven by ``method``; returns its row of the table."""
    atoms = geometry()
    atoms.calc = Mandacaru(method="adapt-vqe",
                           mapping="jordan_wigner",
                           optimizer={"method": method,
                                      "maxiter": MAXITER,
                                      "tol": TOL},
                           gradient_tolerance=1e-4,
                           trace=False,
                           **SETUPS[SYSTEM])
    start = time.perf_counter()
    energy = atoms.get_total_energy()              # eV
    seconds = time.perf_counter() - start
    result = atoms.calc.result
    return atoms.calc, {
        "optimizer": method,
        "optimizer_steps": result.optimizer_steps,
        "cost_evaluations": result.num_evaluations,
        "num_operators": result.num_operators,
        "energy_eV": energy,
        "max_gradient": result.final_max_gradient,
        # Two different convergences: whether ADAPT's screening gradient fell
        # below its tolerance, and in how many growth steps the *inner*
        # optimizer failed to certify its own.  The second one is the column
        # that judges the optimizer.
        "adapt_converged": result.converged,
        "inner_failures": len(result.optimizer_failures),
        "cnot_count": result.metrics.cnot_count,
        "seconds": seconds,
    }


rows = []
reference = None
register = None
for name in NAMED_OPTIMIZERS:
    calc, row = run(name)
    if reference is None:
        # Exact reference: the lowest eigenvalue of the qubit Hamiltonian.  It
        # is the same operator in every run -- the optimizer changes nothing
        # about the problem -- so it is diagonalized once.
        matrix = calc.hamiltonian.to_matrix()
        reference = float(from_hartree(
            np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T)).min(), "eV"))
        register = calc.n_qubits
    row["error_eV"] = row["energy_eV"] - reference
    rows.append(row)

setup = SETUPS[SYSTEM]
print(f"{SYSTEM}, basis={setup['basis']}, h={setup['h']}, "
      f"pool={setup['pool']}, Jordan-Wigner ({register} qubits); "
      f"FCI = {reference:.6f} eV\n")
print(f"{'optimizer':<12}{'steps':>8}{'evals':>8}{'ops':>5}{'E (eV)':>16}"
      f"{'E-FCI (eV)':>13}{'cnot':>6}{'s':>7}{'ADAPT':>8}{'failed':>8}")
for row in rows:
    print(f"{row['optimizer']:<12}{row['optimizer_steps']:>8}"
          f"{row['cost_evaluations']:>8}{row['num_operators']:>5}"
          f"{row['energy_eV']:>16.6f}{row['error_eV']:>13.2e}"
          f"{row['cnot_count']:>6}{row['seconds']:>7.1f}"
          f"{'yes' if row['adapt_converged'] else 'no':>8}"
          f"{row['inner_failures']:>8}")

# Every method must land on the same state; only the effort differs.
worst = max(abs(row["error_eV"]) for row in rows)
assert worst < 1e-3 * HARTREE_TO_EV, f"an optimizer missed FCI by {worst:.2e} eV"

with open(os.path.join(DATA, f"{SYSTEM.lower()}_optimizers.csv"), "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

# Left: the two currencies side by side, log scale -- they differ by two orders
# of magnitude between methods.  Right: how close each one got to FCI.
names = [row["optimizer"] for row in rows]
index = np.arange(len(names))
figure, (effort, accuracy) = plt.subplots(1, 2, figsize=(11, 4))
effort.bar(index - 0.2, [row["optimizer_steps"] for row in rows], 0.4,
           label="optimizer steps")
effort.bar(index + 0.2, [row["cost_evaluations"] for row in rows], 0.4,
           label="cost evaluations")
effort.set_yscale("log")
effort.set_xticks(index, names, rotation=30, ha="right")
effort.set_ylabel("count (whole run)")
effort.set_title("classical effort")
effort.legend()

accuracy.bar(index, [max(row["error_eV"], 1e-12) for row in rows], 0.5,
             color="tab:red")
accuracy.set_yscale("log")
accuracy.set_xticks(index, names, rotation=30, ha="right")
accuracy.set_ylabel("E - E(FCI) (eV)")
accuracy.set_title("accuracy")
figure.suptitle(f"{SYSTEM}, ADAPT-VQE, {setup['pool']} pool")

figure.tight_layout()
figure.savefig(os.path.join(DATA, f"{SYSTEM.lower()}_optimizers.png"), dpi=150)
print(f"\nwrote {os.path.join(DATA, f'{SYSTEM.lower()}_optimizers.csv')}")
