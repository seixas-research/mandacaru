# -*- coding: utf-8 -*-
# file: examples/H2_vqe.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Ground-state energy of H2 by VQE with a UCCSD ansatz, driven through ASE.

The whole quantum simulation is run the ASE-native way: define the molecule as an
``Atoms`` object (with a unit cell) and attach :class:`~carcara.algorithms.VQE`
as its calculator.  ``atoms.get_total_energy()`` then runs the end-to-end
pipeline

    geometry -> real-space integrals -> second-quantized Hamiltonian
             -> Jordan-Wigner qubit Hamiltonian
             -> UCCSD ansatz -> VQE optimization -> ground-state energy (eV)

and prints the qubit Hamiltonian (Pauli strings) plus a timing / memory / cores
summary.  The result is validated against exact diagonalization.

Run with::

    python examples/H2_vqe.py
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from carcara.algorithms import VQE
from carcara.units import from_hartree

# --- 1) Molecule: H2 at equilibrium in a cubic cell (the grid is generated from
#        the cell at resolution h; no explicit Grid needed).
atoms = Atoms("H2", positions=[[4.0, 4.0, 3.63], [4.0, 4.0, 4.37]],
              cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
              pbc=True)

# --- 2) Attach VQE as an ASE calculator.  basis="FAO" builds the Hamiltonian and
#        a default UCCSD ansatz from the geometry; optimizer is named by string.
atoms.calc = VQE(basis="FAO", mapping="jordan_wigner", optimizer="COBYLA",
                 h=0.20)

# --- 3) Asking ASE for the energy runs the whole VQE simulation (returns eV).
energy_ev = atoms.get_total_energy()
result = atoms.calc.vqe_result
energy_ha = result.optimal_energy

# --- 4) Validate against exact diagonalization of the built qubit Hamiltonian.
h_matrix = atoms.calc.hamiltonian.to_matrix()
exact_ha = float(np.linalg.eigvalsh(0.5 * (h_matrix + h_matrix.conj().T)).min())
exact_ev = float(from_hartree(exact_ha, "eV"))

assert abs(energy_ha - exact_ha) < 1.6e-3, "VQE missed chemical accuracy"
print(f"H2  VQE E = {energy_ha:.6f} Ha ({energy_ev:.6f} eV), "
      f"error vs exact {energy_ha - exact_ha:+.1e} Ha")
print(f"    correlation {result.correlation_energy:+.6f} Ha, "
      f"{result.num_evaluations} evaluations, "
      f"{result.timings['n_cores']} cores")
