# -*- coding: utf-8 -*-
# file: examples/01_ADAPTVQE_H2.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""H2 ground state with ADAPT-VQE (qubit pool) through the QuantumCalculator.

End-to-end demonstration wiring the pieces together through the ASE interface:

* the molecule is defined once as an ASE :class:`ase.Atoms` object and
  :class:`~carcara.algorithms.QuantumCalculator` is attached to it as an ASE
  *calculator* (``atoms.calc = QuantumCalculator(method="adapt-vqe", ...)``);
* the calculator builds the Hamiltonian from the current geometry using the
  chosen ``basis`` (here ``"FAO"`` -- Full Atomic Orbitals), so no manual
  integral wiring is needed;
* calling ``atoms.get_total_energy()`` runs ADAPT-VQE with the qubit pool and
  returns the ground-state energy in **eV** (the ASE convention); the full run
  result is on ``atoms.calc.result``.

The argument surface exercised here is ``method`` / ``pool`` / ``basis`` /
``mapping`` / ``gradient`` / ``device``.
"""

from __future__ import annotations

import os

import numpy as np
from ase import Atoms

from carcara.algorithms import QuantumCalculator
from carcara.units import from_hartree

# All generated files (logs, CSV, plots) go to examples/data/.
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)


atoms = Atoms("H2",
              positions=[[6.0, 6.0, 5.63], [6.0, 6.0, 6.37]],
              cell=[[12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 12.0]],
              pbc=True)

atoms.calc = QuantumCalculator(
              method="adapt-vqe",
              pool="qubit",
              basis={"name": "FAO"},
              mapping="jordan_wigner",
              optimizer="COBYLA",
              gradient="parameter-shift",
              device="AER_simulator",
              h=0.10,
              max_iterations=15,
              gradient_tolerance=1e-4,
              output=os.path.join(DATA, "output_H2.txt"))

energy_ev = atoms.get_total_energy()
result = atoms.calc.result
energy_ha = result.optimal_energy

# Exact reference: lowest eigenvalue of the qubit Hamiltonian (FCI).
h_matrix = atoms.calc.hamiltonian.to_matrix()
exact_ha = float(np.linalg.eigvalsh(0.5 * (h_matrix + h_matrix.conj().T)).min())
exact_ev = float(from_hartree(exact_ha, "eV"))

assert abs(energy_ha - exact_ha) < 1e-4, "ADAPT missed FCI"
print(f"H2  E = {energy_ev:.6f} eV ({energy_ha:.8f} Ha), error vs FCI "
      f"{energy_ev - exact_ev:+.1e} eV")
print(f"    {result.num_operators} operators, "
      f"{result.metrics.cnot_count} CNOTs, converged={result.converged}")
