# -*- coding: utf-8 -*-
# file: examples/06_ADAPTVQE_O2_triplet.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Triplet O2 ground state with spin-polarized ADAPT-VQE.

Molecular oxygen has a **triplet** (spin-polarized) ground state -- two unpaired
electrons in the degenerate pi* orbitals.  That initial spin state is set the ASE
way, through the atoms' **initial magnetic moments** (``magmoms=[1, 1]`` -> a
total moment of 2, i.e. two unpaired electrons); the
:class:`~carcara.algorithms.QuantumCalculator` (``method="adapt-vqe"``) reads
it and builds the reference with ``n_alpha - n_beta = 2`` (see
:func:`carcara.algorithms._hamiltonian_from_atoms.resolve_num_unpaired`).  The
grown ansatz then conserves ``S_z``, so the whole simulation stays in the triplet
sector.

O2 with Full Atomic Orbitals is 10 spatial orbitals / 16 electrons; to keep the
exact state-vector simulation tractable it is run in a compact active space
(``frozen_orbitals`` removes the lowest molecular orbitals, leaving a 5-orbital /
10-qubit ``(4, 2)`` triplet with 6 active electrons).

.. note::

   This is a **qualitative** demonstration of the spin-polarized machinery, not a
   quantitative O2 calculation: both the coarse real-space grid and the aggressive
   active-space truncation are approximations.  The checks below are
   self-consistent -- the triplet reference is correctly set up, and ADAPT-VQE
   lowers the energy within the triplet sector toward its (sector) FCI.
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



def sector_fci(pauli_hamiltonian, n_qubits, na, nb):
    """Lowest eigenvalue in the ``(na, nb)`` particle/spin sector (sparse assembly)."""
    M = n_qubits // 2
    keep = [i for i in range(2 ** n_qubits)
            if sum((i >> (n_qubits - 1 - q)) & 1 for q in range(M)) == na
            and sum((i >> (n_qubits - 1 - q)) & 1 for q in range(M, n_qubits)) == nb]
    keep = np.array(keep)
    sub = pauli_hamiltonian.to_sparse_matrix()[np.ix_(keep, keep)].toarray()
    return float(np.linalg.eigvalsh(0.5 * (sub + sub.conj().T)).min())


# 1. Triplet O2: the initial spin state lives on the Atoms object via magmoms.
d = 1.208
atoms = Atoms("O2",
              positions=[[4.0, 4.0, 4.0 - d / 2], [4.0, 4.0, 4.0 + d / 2]],
              cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
              pbc=True,
              magmoms=[1.0, 1.0])                 # two unpaired electrons -> triplet

atoms.calc = QuantumCalculator(
              method="adapt-vqe",
              pool="fermionic",
              basis={"name": "FAO"},
              mapping="jordan_wigner",
              frozen_orbitals=[0, 1, 2, 3, 4],     # compact active space (tractable)
              h=0.25,
              max_iterations=12,
              gradient_tolerance=1e-3,
              output=os.path.join(DATA, "output_O2.txt"))

# 2. Asking ASE for the energy runs the whole ADAPT-VQE simulation.
energy_ev = atoms.get_total_energy()               # eV (ASE convention)
result = atoms.calc.result
energy_ha = result.optimal_energy
na, nb = atoms.calc.num_particles

# The reference must be a genuine triplet (n_alpha - n_beta = 2), and ADAPT must
# lower the energy below that spin-polarized Hartree-Fock reference.
assert na - nb == 2, "the reference is not a triplet"
assert result.optimal_energy < result.reference_energy, "ADAPT did not lower E"

exact_ha = sector_fci(atoms.calc.hamiltonian, atoms.calc.n_qubits, na, nb)
exact_ev = float(from_hartree(exact_ha, "eV"))

print(f"O2 (triplet)  {atoms.calc.n_qubits // 2} active orbitals "
      f"({atoms.calc.n_qubits} qubits), num_particles=({na}, {nb}), "
      f"2Sz = {na - nb}")
print(f"     E = {energy_ev:.6f} eV ({energy_ha:.8f} Ha) [qualitative]")
correlation = from_hartree(result.optimal_energy - result.reference_energy, "eV")
print(f"     correlation energy = {correlation:.4f} eV below the triplet HF "
      f"reference")
print(f"     triplet-sector FCI = {exact_ev:.6f} eV "
      f"(gap {energy_ev - exact_ev:+.3f} eV)")
print(f"     {result.num_operators} operators, converged={result.converged}")
