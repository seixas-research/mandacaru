# -*- coding: utf-8 -*-
# file: examples/run_adapt_vqe.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""ADAPT-VQE on H2 with the four operator pools, compared on hardware cost.

Runs ADAPT-VQE (Grimsley et al., 2019) on H2 with each of the four operator pools

    fermionic  -- spin-adapted fermionic excitations (Jordan-Wigner)
    qubit      -- individual JW Pauli strings (qubit-ADAPT)
    qeb        -- qubit-excitation generators (Z-strings dropped)
    ceo        -- coupled-exchange operators (shared entangling structure)

and prints a comparative summary of convergence rate, CNOT count and final circuit
depth.  The whole simulation is driven the ASE-native way: the molecule is an ASE
``Atoms`` object and each pool is run by attaching ``ADAPTVQE`` as its calculator
and calling ``atoms.get_total_energy()`` (the Hamiltonian is rebuilt from the same
geometry each time, so every pool sees the same problem).

Run with::

    python examples/run_adapt_vqe.py
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from carcara.algorithms import ADAPTVQE
from carcara.optimizers import Optimizer

POOLS = ["fermionic", "qubit", "qeb", "ceo"]

# H2 in a cubic cell; ADAPTVQE builds the FAO/RHF-MO Hamiltonian from the geometry
# (the grid is generated from atoms.cell at resolution h).
atoms = Atoms("H2", positions=[[4.0, 4.0, 3.63], [4.0, 4.0, 4.37]],
              cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]], pbc=True)

header = (f"{'pool':10s} {'E (Ha)':>13s} {'E-FCI':>11s} {'#ops':>5s} "
          f"{'#params':>7s} {'CNOTs':>6s} {'depth':>6s} {'nfev':>6s}  "
          f"{'converged':>9s}")

summary = {}
exact = None
for name in POOLS:
    atoms.calc = ADAPTVQE(
        pool=name, basis="FAO", h=0.20,
        optimizer=Optimizer("L-BFGS-B", maxiter=2000),
        max_iterations=15, gradient_tolerance=1e-6, verbose=False)
    atoms.get_total_energy()                     # runs the ADAPT-VQE simulation
    summary[name] = atoms.calc.adapt_result
    if exact is None:
        # Exact reference: lowest eigenvalue of the (shared) qubit Hamiltonian.
        h_matrix = atoms.calc.hamiltonian.to_matrix()
        exact = float(np.linalg.eigvalsh(0.5 * (h_matrix + h_matrix.conj().T)).min())
        print(f"H2 (FAO basis, RHF-MO), FCI energy = {exact:+.8f} Ha\n")
        print(header)
        print("-" * len(header))

for name in POOLS:
    res = summary[name]
    print(f"{name:10s} {res.optimal_energy:+13.8f} "
          f"{res.optimal_energy - exact:+11.2e} "
          f"{res.num_operators:5d} {len(res.optimal_parameters):7d} "
          f"{res.metrics.cnot_count:6d} {res.metrics.depth:6d} "
          f"{res.num_evaluations:6d} {str(res.converged):>9s}")

print("\nOperator selection order (label per ADAPT iteration):")
for name in POOLS:
    labels = " -> ".join(it.operator_label for it in summary[name].iterations)
    print(f"  {name:10s}: {labels}")

# Headline benchmark: same ground state, fewer CNOTs for hardware-minded pools.
ferm = summary["fermionic"].metrics.cnot_count
print("\nCNOTs to reach FCI, relative to the fermionic pool:")
for name in POOLS:
    c = summary[name].metrics.cnot_count
    print(f"  {name:10s}: {c:3d} CNOTs  ({c / ferm:.2f}x fermionic)")

for name in POOLS:
    assert abs(summary[name].optimal_energy - exact) < 1e-6, name
print("\nAll four pools reached the FCI ground state to < 1e-6 Ha.")
