# -*- coding: utf-8 -*-
# file: examples/37_active_space_selection.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Fitting a PAW-LCAO-TZP basis on a register a processor can hold.

A basis set buys accuracy with virtual orbitals and a qubit register pays for
them at two qubits each.  Water in PAW-LCAO-TZP is 29 spatial orbitals -- 56
qubits after the parity reduction -- and the same molecule in SZ is 10.  The
first is unreachable on hardware and the second is a poor description of the
molecule.  The way out is a **large basis with a small active space**: the
basis-set quality lives in the shape of the orbitals, the correlation lives in
the few of them the wavefunction actually mixes.

This script shows the three things that decide which those are.

1. **How many orbitals**, by count (``active_orbitals``) or by an occupation
   criterion on the natural orbitals (``active_threshold``).
2. **Which ones**, by ``active_selection``: canonical orbital energy, or the
   frozen natural orbitals of the MP2 density.
3. **What it costs**, measured against the untruncated answer.

The headline is part 2.  Ranking virtuals by orbital energy asks "which is
cheapest to excite into"; ranking them by the second-order density asks "which
one does the correlated wavefunction actually occupy", and for a basis with
diffuse or polarization functions those are different orbitals -- a diffuse
virtual is low in energy and spatially wrong for correlation.  On LiH in TZP the
difference at 16 qubits is 412 meV against 1.4 meV.

Run:  python examples/37_active_space_selection.py
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from mandacaru import Mandacaru

#: One geometry for the whole script: LiH near equilibrium, in a box the
#: real-space grid can hold.  Twelve spatial orbitals in TZP, of which one is
#: occupied -- so eleven virtuals to choose among.
DISTANCE = 1.60
CELL = (9.0, 9.0, 11.0)
GRID = 0.25
BASIS = {"name": "PAW-LCAO", "size": "TZP"}
OPTIMIZER = {"method": "SLSQP", "maxiter": 400, "tol": 1e-10}


def lih() -> Atoms:
    """LiH at ``DISTANCE``, centered in ``CELL``."""
    atoms = Atoms("LiH", positions=[(0, 0, 0), (0, 0, DISTANCE)], cell=CELL)
    atoms.center()
    return atoms


def solve(**active) -> tuple[float, int, int, object]:
    """``(energy_eV, n_qubits, n_operators, ActiveSpace|None)`` for one spec."""
    atoms = lih()
    atoms.calc = Mandacaru(method="adapt-vqe", basis=BASIS, h=GRID,
                           optimizer=OPTIMIZER, trace=False, **active)
    energy = atoms.get_potential_energy()
    solver = atoms.calc.solver
    space = (getattr(solver, "_gradient_context", None) or {}).get(
        "active_space")
    operators = len(getattr(solver.result, "iterations", []) or [])
    return energy, solver.n_qubits, operators, space


# --------------------------------------------------------------------------- #
# 1. What the untruncated basis costs.
# --------------------------------------------------------------------------- #

print(__doc__.split("Run:")[0])
print("=" * 74)
print("The untruncated basis")
print("=" * 74)

reference, full_qubits, full_ops, _space = solve()
print("  PAW-LCAO-TZP, every orbital on the register")
print(f"    qubits        : {full_qubits}")
print(f"    energy        : {reference:.6f} eV")
print(f"    state vector  : 2^{full_qubits} amplitudes "
      f"= {2 ** full_qubits * 16 / 2 ** 20:.0f} MiB")

# --------------------------------------------------------------------------- #
# 2. Look at the occupation numbers before choosing a threshold.
# --------------------------------------------------------------------------- #

print()
print("=" * 74)
print("The MP2 natural occupation numbers, which are what a threshold cuts")
print("=" * 74)

# Any truncation runs the selector, so this is how to see the spectrum: ask for
# one orbital more than the occupied count and read the occupations back.
_e, _q, _ops, space = solve(active_orbitals=2, active_selection="mp2")
n_occupied = len(space.active) - 1
occupations = np.sort(np.asarray(space.occupations)[n_occupied:])[::-1]
total = float(occupations.sum())

print(f"  {len(occupations)} virtual orbitals; "
      f"{total:.4e} electrons promoted in total")
print("  NOON: " + "  ".join(f"{value:.2e}" for value in occupations))
print()
print("  A threshold keeps the orbitals above it:")
cumulative = np.cumsum(occupations) / total
for threshold in (2e-2, 1e-3, 1e-4, 1e-5):
    kept = int((occupations >= threshold).sum())
    share = 100.0 * cumulative[kept - 1] if kept else 0.0
    register = 2 * (n_occupied + kept)
    verdict = "REFUSED: nothing clears it" if kept == 0 else \
        f"{kept:2d} virtuals, {share:6.2f}% of the charge, {register:2d} qubits"
    print(f"    occupation >= {threshold:.0e}  ->  {verdict}")

print()
print("  Note the first row.  A virtual natural occupation is the small charge")
print("  correlation promotes -- not an electron count -- so it is of order")
print("  1e-2 at the very most.  A threshold of 0.02 is above the whole")
print("  spectrum and keeps nothing, which Mandacaru refuses by name rather")
print("  than returning Hartree-Fock through a variational solver.")

# --------------------------------------------------------------------------- #
# 3. The same register width, chosen two ways.
# --------------------------------------------------------------------------- #

print()
print("=" * 74)
print("Which orbitals: canonical energy order against the MP2 density")
print("=" * 74)
print(f"  {'spec':34} {'qubits':>6} {'ops':>5} {'error vs. full':>16}")

for width in (4, 8):
    for selection in ("energy", "mp2"):
        energy, qubits, operators, space = solve(active_orbitals=width,
                                                active_selection=selection)
        error = (energy - reference) * 1000.0
        print(f"  active_orbitals={width}, {selection:<14} {qubits:>6} "
              f"{operators:>5} {error:>13.2f} meV")

print()
print("  Energy ordering keeps the lowest-lying virtuals.  In a basis with")
print("  polarization functions those are not the ones that correlate the")
print("  occupied orbital: on H2 in 6-31G the largest first-order amplitude")
print("  sits on the *highest* virtual, and the leading natural orbital is a")
print("  mixture of the first and third.  That is why the two columns differ")
print("  by two orders of magnitude and not by a few percent.")

# --------------------------------------------------------------------------- #
# 4. Choosing by occupation instead of by count.
# --------------------------------------------------------------------------- #

print()
print("=" * 74)
print("Choosing by occupation: active_threshold")
print("=" * 74)
print("  A count says how wide a register you are willing to pay for; a")
print("  threshold says which orbitals are worth paying for and lets the width")
print("  follow.  Use the threshold when the question is 'how small can this")
print("  get', and the count when the register size is fixed by the hardware.")
print()
print(f"  {'threshold':>12} {'qubits':>7} {'active':>7} {'ops':>6} "
      f"{'error vs. full':>16}")

for threshold in (1e-3, 1e-4, 1e-5):
    energy, qubits, operators, space = solve(active_threshold=threshold,
                                            active_selection="mp2")
    error = (energy - reference) * 1000.0
    print(f"  {threshold:12.0e} {qubits:>7} {space.n_active:>7} "
          f"{operators:>6} {error:>13.2f} meV")

print()
print("  `active_threshold=True` takes the default, 1e-3 -- the knee of the")
print("  curve above, past which each orbital costs two qubits and returns a")
print("  few parts per thousand of the correlation.")
print()
print("  The two compose: active_orbitals caps the register and")
print("  active_threshold decides what earns a place inside it, so")
print("  `active_orbitals=8, active_threshold=1e-5` means 'the orbitals worth")
print("  keeping, but never more than eight'.")

# --------------------------------------------------------------------------- #
# 5. A caveat worth more than the numbers above.
# --------------------------------------------------------------------------- #

print()
print("=" * 74)
print("Reading these errors honestly")
print("=" * 74)
print("  Deleting a virtual orbital removes variational freedom, so the exact")
print("  energy of a truncated space is always ABOVE the untruncated one and")
print("  falls monotonically as orbitals are added back.  By exact")
print("  diagonalization on this system: +22.65, +3.69, +0.24 meV at")
print("  thresholds 1e-3, 1e-4, 1e-5.")
print()
print("  The errors printed above come from ADAPT-VQE runs instead, and at")
print("  these widths the truncation error is SMALLER than the optimizer's own")
print("  residual on the 24-qubit run -- which has fifty-odd operators to")
print("  converge against the truncated run's handful.  So a column above can")
print("  come out negative without anything being wrong.  Compare truncations")
print("  by exact diagonalization, or with a convergence tolerance tight")
print("  enough that the solver is not the largest error in the comparison.")
print()
print("  Also: nuclear forces and the stress are refused with a truncated")
print("  virtual space.  The selection moves with the nuclei, and that")
print("  response is not in the Hellmann-Feynman and Pulay sums, so the")
print("  gradient would be the derivative of a different energy than the one")
print("  reported.  Relax with the full virtual space.")
