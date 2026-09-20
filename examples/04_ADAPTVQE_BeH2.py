# -*- coding: utf-8 -*-
# file: examples/04_ADAPTVQE_BeH2.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Linear BeH2 ground state with ADAPT-VQE and a frozen Be core.

Linear (D∞h) BeH2 is built as an ASE :class:`ase.Atoms` object with the two
hydrogens symmetric about the central beryllium along ``z``.  With
``basis={"name": "FAO"}`` each atom contributes its Full Atomic Orbitals
(Be {1s, 2s} + 2 H {1s} = 4 spatial orbitals), and ``frozen_core=True`` freezes
the Be ``1s`` core -- leaving a 3-orbital / 6-qubit active space with 4 active
electrons (a ``(2, 2)`` closed shell).  The solver is attached through
:class:`~mandacaru.algorithms.Mandacaru`
(``atoms.calc = Mandacaru(method="adapt-vqe", ...)``);
``atoms.get_total_energy()`` then drives ADAPT-VQE with the CEO pool and returns
the energy in **eV**, with the full run result on ``atoms.calc.result``.

.. note::

   On a uniform real-space grid this is **qualitative**: the contracted cores are
   only partially resolved.  The invariant checked here is *self-consistent* --
   ADAPT-VQE must recover the exact (FCI) ground state **of the active-space
   Hamiltonian it is given**.
"""

from __future__ import annotations

import os

import numpy as np
from ase import Atoms

from mandacaru.algorithms import Mandacaru
from mandacaru.units import HARTREE_TO_EV, from_hartree

# All generated files (logs, CSV, plots) go to examples/data/.
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)


# 1. Linear BeH2 (Be at the cell center, H atoms at +/- 1.334 A along z).
d = 1.334
atoms = Atoms("BeH2",
              positions=[[5.0, 5.0, 5.0],
                         [5.0, 5.0, 5.0 - d],
                         [5.0, 5.0, 5.0 + d]],
              cell=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
              pbc=True)

atoms.calc = Mandacaru(method="adapt-vqe",
                       pool="ceo",
                       basis={"name": "FAO"},
                       sparse=True,
                       mapping="jordan_wigner",
                       gradient="parameter_shift",
                       frozen_core=True,   # freeze the Be 1s core
                       h=0.10,
                       max_iterations=25,
                       gradient_tolerance=1e-3,
                       output=os.path.join(DATA, "output_BeH2.txt"))

# 2. Asking ASE for the energy runs the whole ADAPT-VQE simulation.
energy_ev = atoms.get_total_energy()               # eV (ASE convention)
result = atoms.calc.result                         # result.optimal_energy is eV
n_orbitals = atoms.calc.n_qubits // 2

# Exact reference: lowest eigenvalue of the (active-space) qubit Hamiltonian
# (FCI).  The qubit Hamiltonian is internal (Hartree): converted once, here.
h_matrix = atoms.calc.hamiltonian.to_matrix()
exact_ev = float(from_hartree(
    np.linalg.eigvalsh(0.5 * (h_matrix + h_matrix.conj().T)).min(), "eV"))

assert abs(energy_ev - exact_ev) < 1e-4 * HARTREE_TO_EV, \
    "CEO ADAPT missed active-space FCI"
print(f"BeH2 (linear)  {n_orbitals} active orbitals "
      f"({atoms.calc.n_qubits} qubits, Be 1s frozen), "
      f"num_particles={atoms.calc.num_particles}")
print(f"     E = {energy_ev:.6f} eV [qualitative], "
      f"error vs FCI {energy_ev - exact_ev:+.1e} eV")
print(f"     {result.num_operators} operators, "
      f"{result.metrics.cnot_count} CNOTs, converged={result.converged}")
