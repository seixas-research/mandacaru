# -*- coding: utf-8 -*-
# file: examples/H2_mapping.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Fermionic Hamiltonian and fermion-to-qubit mapping for H2.

Assembles the second-quantized molecular Hamiltonian of H2 from the real-space
integral engine (:class:`~carcara.core.MolecularIntegrals`) as a ``Fermion``
operator, then maps it to a qubit (Pauli) operator with each of the three
mappings -- Jordan-Wigner (default), Bravyi-Kitaev and parity (with the optional
two-qubit reduction).  All three share the same spectrum; the parity reduction
tapers the 4-qubit operator to 2 qubits.

Run with::

    python examples/H2_mapping.py
"""

from __future__ import annotations

import numpy as np

from carcara.core import MolecularIntegrals, minimal_fao_basis
from carcara.integrals import Grid

# H2: a minimal Slater-screened 1s basis, one orbital per atom (Angstrom).
R = 0.74
nuclei = [(1.0, np.array([0.0, 0.0, -R / 2])),
          (1.0, np.array([0.0, 0.0, +R / 2]))]
basis = minimal_fao_basis(nuclei)
grid = Grid(center=[0.0, 0.0, 0.0], box_size=6.0, h=0.15)

# Second-quantized molecular Hamiltonian over spin-orbitals (a Fermion):
#   H = sum_pq h_pq a+_p a_q + 1/2 sum_pqrs <pq|rs> a+_p a+_q a_s a_r
integrals = MolecularIntegrals(nuclei, basis, grid)
H = integrals.molecular_hamiltonian()          # 2 spatial -> 4 spin-orbitals
print(f"Fermionic Hamiltonian: {H}")

# Map to a qubit operator (a PauliSum of Pauli strings).
H_jw = H.map_to_qubits(method="jordan_wigner")           # default
H_bk = H.map_to_qubits(method="bravyi_kitaev")
H_parity = H.map_to_qubits(method="parity",
                           two_qubit_reduction=True, num_particles=(1, 1))

print(f"\nJordan-Wigner : {H_jw.num_qubits} qubits, "
      f"{len(H_jw.simplify().terms)} Pauli terms")
print(f"Bravyi-Kitaev : {H_bk.num_qubits} qubits")
print(f"parity (2-qubit reduction): {H_parity.num_qubits} qubits")

# The three encodings are different operators but share the same spectrum.
e_jw = np.linalg.eigvalsh(H_jw.to_matrix())
e_bk = np.linalg.eigvalsh(H_bk.to_matrix())
e_par = np.linalg.eigvalsh(H_parity.to_matrix())
print("\nJW and BK spectra agree:", np.allclose(e_jw, e_bk))
print(f"ground-state energy (JW, all sectors) = {e_jw.min():.4f} Ha")
print(f"ground-state energy (parity, tapered) = {e_par.min():.4f} Ha")

# Convert to a Qiskit SparsePauliOp for downstream VQE/estimation.
# sparse = H_jw.to_sparse_pauli_op()
