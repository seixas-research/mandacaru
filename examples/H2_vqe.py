# -*- coding: utf-8 -*-
# file: examples/H2_vqe.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Ground-state energy of H2 by VQE with a UCCSD ansatz (Jordan-Wigner).

End-to-end pipeline on a minimal FAO basis:

    geometry -> real-space integrals -> second-quantized Hamiltonian
             -> Jordan-Wigner qubit Hamiltonian
             -> UCCSD ansatz -> VQE optimization -> ground-state energy

The result is validated against exact diagonalization of the qubit Hamiltonian.

Run with::

    python examples/H2_vqe.py
"""

from __future__ import annotations

import numpy as np

from carcara.algorithms import VQE
from carcara.circuits import UCCSD
from carcara.core import MolecularIntegrals, minimal_fao_basis
from carcara.integrals import Grid
from carcara.optimizers import Optimizer

# --- 1) Molecule: H2 at equilibrium, minimal FAO 1s basis (Angstrom).
R = 0.74
nuclei = [(1.0, np.array([0.0, 0.0, -R / 2])),
          (1.0, np.array([0.0, 0.0, +R / 2]))]
basis = minimal_fao_basis(nuclei)
grid = Grid(center=[0.0, 0.0, 0.0], box_size=6.0, h=0.15)

# --- 2) Second-quantized molecular Hamiltonian (a Fermion), 4 spin-orbitals.
integrals = MolecularIntegrals(nuclei, basis, grid)
H = integrals.molecular_hamiltonian()
print(f"Fermionic Hamiltonian: {H}")

# --- 3) UCCSD ansatz on 4 qubits, 2 electrons (1 alpha + 1 beta), JW mapping.
ansatz = UCCSD(n_spatial_orbitals=2, num_particles=(1, 1),
               mapping="jordan_wigner")
print(f"Ansatz: {ansatz}")

# --- 4) VQE: minimize <psi(theta)| H |psi(theta)> from the HF reference.
vqe = VQE(H, ansatz, optimizer=Optimizer(method="COBYLA", maxiter=2000))
result = vqe.run()

# --- 5) Report and validate against exact diagonalization.
h_matrix = H.map_to_qubits("jordan_wigner").to_matrix()
exact = float(np.linalg.eigvalsh(0.5 * (h_matrix + h_matrix.conj().T)).min())

print(f"\nHartree-Fock reference energy : {result.reference_energy:+.6f} Ha")
print(f"VQE ground-state energy       : {result.optimal_energy:+.6f} Ha")
print(f"Exact diagonalization         : {exact:+.6f} Ha")
print(f"VQE - exact                   : {result.optimal_energy - exact:+.2e} Ha")
print(f"Correlation energy recovered  : "
      f"{result.reference_energy - result.optimal_energy:.6f} Ha")
print(f"Optimizer evaluations         : {result.num_evaluations}")
print(f"Optimal parameters            : "
      f"{np.array2string(result.optimal_parameters, precision=5)}")

chemical_accuracy = 1.6e-3  # Ha
assert abs(result.optimal_energy - exact) < chemical_accuracy
print("\nVQE reproduces the exact ground state to within chemical accuracy.")
