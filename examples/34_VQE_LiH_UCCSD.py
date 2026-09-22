# -*- coding: utf-8 -*-
# file: examples/34_VQE_LiH_UCCSD.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""LiH ground state with plain VQE on the **UCCSD** ansatz.

``Mandacaru(method="vqe", ...)`` is the fixed-ansatz eigensolver: the circuit is
decided *before* the optimization from the problem's own shape -- every
single and double excitation out of the Hartree-Fock determinant, the unitary
coupled-cluster ansatz -- and the classical optimizer then varies its
amplitudes.  **UCCSD is the default**, so a calculator-mode run needs nothing
but a geometry:

    atoms.calc = Mandacaru(method="vqe", basis="HAO", h=0.25)
    atoms.get_total_energy()

That is the whole example.  The rest of the script is the comparison that makes
the choice meaningful: the same molecule, the same Hamiltonian and the same
answer through **ADAPT-VQE**, which builds its circuit one operator at a time
instead.  The two differ in *where* they spend:

* **UCCSD knows its ansatz up front.**  No pool, no gradient screening, so it
  reaches the answer in roughly a quarter of the energy evaluations.
* **UCCSD includes every excitation**, whether or not it matters, so its circuit
  is about 30 % longer in CNOTs.

On LiH in the HAO basis the full singles-and-doubles set is only eight
operators, so ADAPT ends up selecting essentially all of them and the two
ansatze coincide in size.  The adaptive method earns its screening cost when
the pool is far larger than the set that matters -- water's 640-operator qubit
pool, for instance (``examples/03_ADAPTVQE_H2O.py``).

.. note::

   LiH on a uniform real-space grid is **qualitative**: the tightly contracted
   Li 1s core is only partially resolved at a practical grid spacing.  What is
   checked here is *self-consistent* -- VQE must reproduce the exact (FCI)
   ground state **of the Hamiltonian it is given**, which it does to ~1e-11 eV.
"""

from __future__ import annotations

import csv
import os

import numpy as np
from ase import Atoms

from mandacaru.algorithms import Mandacaru
from mandacaru.circuits import UCCSD
from mandacaru.units import HARTREE_TO_EV, from_hartree

# All generated files (logs, CSV, plots) go to examples/data/.
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

BASIS = "HAO"                   # Li {1s, 2s} + H {1s} = 3 orbitals -> 6 qubits
SPACING = 0.25                  # grid spacing (Angstrom)
CELL = 10.0                     # cubic cell edge (Angstrom)
DISTANCES = (1.2, 1.6, 2.0, 2.6)        # Angstrom; 1.6 is near equilibrium
OPTIMIZER = {"method": "SLSQP", "maxiter": 1000, "tol": 1e-12}


def lih(distance: float) -> Atoms:
    """LiH at ``distance``, centered in its cell (the grid is cut from it)."""
    atoms = Atoms("LiH", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, distance)],
                  cell=[CELL, CELL, CELL])
    atoms.center()
    return atoms


def exact_energy(calc) -> float:
    """Lowest eigenvalue of the qubit Hamiltonian this run was given, in eV."""
    matrix = calc.hamiltonian.to_matrix()
    ground = np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T)).min()
    return float(from_hartree(ground, "eV"))


def cnot_count(calc) -> int:
    """CNOTs of the optimized circuit, compiled to a native ``{cx, u}`` set.

    ``ansatz_problem()`` is the register-level description of the prepared
    state -- ``(n_qubits, occupied, generators, thetas)`` -- which is exactly
    what a circuit provider consumes, so the same three lines work for either
    method.
    """
    from qiskit import transpile

    from mandacaru.backends import QiskitProvider

    circuit = QiskitProvider(device="statevector").build(
        *calc.solver.ansatz_problem()[:4])
    native = transpile(circuit, basis_gates=["cx", "u"], optimization_level=1)
    return int(native.count_ops().get("cx", 0))


# --------------------------------------------------------------------------- #
# 1. The example proper: VQE on the default UCCSD ansatz, at equilibrium.
# --------------------------------------------------------------------------- #

atoms = lih(1.6)
# `txt=` is not offered here: the structured block log is ADAPT-VQE's, whose
# run() goes through it; `method="vqe"` prints its own summary instead, and
# Mandacaru refuses the option rather than accepting a file it would leave empty.
atoms.calc = Mandacaru(method="vqe",
                       basis=BASIS,
                       h=SPACING,
                       mapping="jordan_wigner",
                       optimizer=OPTIMIZER)

energy_ev = atoms.get_total_energy()          # eV (ASE convention)
result = atoms.calc.result
exact_ev = exact_energy(atoms.calc)

print(f"LiH at 1.60 A, basis {BASIS}, h = {SPACING} A")
print(f"  {atoms.calc.n_qubits // 2} spatial orbitals "
      f"({atoms.calc.n_qubits} qubits), num_particles="
      f"{atoms.calc.num_particles}")
print(f"  ansatz          : {type(atoms.calc.solver.ansatz).__name__} "
      f"(the default for method='vqe'), "
      f"{result.num_parameters} amplitudes")
print(f"  Hartree-Fock    : {result.reference_energy:+.6f} eV")
print(f"  VQE             : {energy_ev:+.6f} eV  [qualitative]")
print(f"  correlation     : {result.correlation_energy:+.6f} eV")
print(f"  exact (FCI)     : {exact_ev:+.6f} eV  "
      f"-> error {energy_ev - exact_ev:+.1e} eV")
print(f"  cost            : {result.num_evaluations} energy evaluations, "
      f"{result.optimizer_steps} optimizer steps")

assert abs(energy_ev - exact_ev) < 1e-6 * HARTREE_TO_EV, "UCCSD missed FCI"

# --------------------------------------------------------------------------- #
# 2. The same answer, built adaptively: what the fixed ansatz costs and buys.
# --------------------------------------------------------------------------- #
#
# UCCSD is a circuit only in its Trotter product form, so the gate count needs
# `ansatz_builder` to ask for one; the energy above does not (the default is
# the exact UCC exponential, which no circuit realizes).

rows = []
print(f"\n{'d (A)':>6} | {'UCCSD par':>9} {'evals':>6} {'CNOTs':>6} "
      f"| {'ADAPT ops':>9} {'evals':>6} {'CNOTs':>6} | {'E(VQE)-E(ADAPT)':>16}")
for distance in DISTANCES:
    vqe = lih(distance)
    vqe.calc = Mandacaru(
        method="vqe",
        basis=BASIS,
        h=SPACING,
        optimizer=OPTIMIZER,
        trace=False,
        ansatz_builder=lambda n_orbitals, particles, mapping: UCCSD(
            n_orbitals, particles, mapping=mapping, trotter=True))
    e_vqe = vqe.get_total_energy()

    adapt = lih(distance)
    adapt.calc = Mandacaru(method="adapt-vqe",
                           basis=BASIS,
                           h=SPACING,
                           pool="qeb",
                           optimizer=OPTIMIZER,
                           trace=False)
    e_adapt = adapt.get_total_energy()

    row = dict(distance_A=distance,
               uccsd_parameters=vqe.calc.result.num_parameters,
               uccsd_evaluations=vqe.calc.result.num_evaluations,
               uccsd_cnots=cnot_count(vqe.calc),
               adapt_operators=adapt.calc.result.num_operators,
               adapt_evaluations=adapt.calc.result.num_evaluations,
               adapt_cnots=cnot_count(adapt.calc),
               energy_vqe_eV=e_vqe, energy_adapt_eV=e_adapt)
    rows.append(row)
    print(f"{distance:>6.2f} | {row['uccsd_parameters']:>9} "
          f"{row['uccsd_evaluations']:>6} {row['uccsd_cnots']:>6} "
          f"| {row['adapt_operators']:>9} {row['adapt_evaluations']:>6} "
          f"{row['adapt_cnots']:>6} | {e_vqe - e_adapt:>16.2e}")

path = os.path.join(DATA, "lih_vqe_uccsd.csv")
with open(path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"\nwrote {path}")

# The two methods must agree: they are two ways of reaching the same FCI state
# of the same Hamiltonian, so the difference is optimizer noise, not physics.
worst = max(abs(r["energy_vqe_eV"] - r["energy_adapt_eV"]) for r in rows)
assert worst < 1e-8, f"VQE and ADAPT-VQE disagree by {worst:.2e} eV"
print(f"VQE and ADAPT-VQE agree to {worst:.1e} eV at every distance.")
