# -*- coding: utf-8 -*-
# file: examples/new/01_VQE_LiH.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

# LiH ground state with VQE on the UCCSD ansatz.

import numpy as np
from ase import Atoms

from mandacaru import Mandacaru

# Molecule: LiH
atoms = Atoms("LiH",
              positions=[[0.0, 0.0, 0.0],   # Li
                         [0.0, 0.0, 1.6]],  # H
              cell=[10.0, 10.0, 10.0])


atoms.calc = Mandacaru(method="vqe",                       # Variational Quantum Eigensolver (VQE)
                       ansatz="uccsd",                     # UCCSD ansatz
                       basis="HAO",                        # Hydrogenic Atomic Orbitals (HAO)
                       h=0.10,                             # Grid spacing (Angstrom)
                       mapping="jordan_wigner",
                       optimizer={"method": "SLSQP",
                                  "maxiter": 1000,
                                  "tol": 1e-10},
                       txt="./outputs/output_01.txt")

energy = atoms.get_total_energy()
result = atoms.calc.result

print(f"qubits            : {atoms.calc.n_qubits}")
print(f"UCCSD amplitudes  : {result.num_parameters}")
print(f"Hartree-Fock      : {result.reference_energy:.6f} eV")
print(f"VQE (UCCSD)       : {energy:.6f} eV")
