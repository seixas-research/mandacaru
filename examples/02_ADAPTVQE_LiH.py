# -*- coding: utf-8 -*-
# file: examples/02_ADAPTVQE_LiH.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""LiH ground state with ADAPT-VQE (CEO pool) through the Mandacaru.

LiH is defined as an ASE :class:`ase.Atoms` object and
:class:`~mandacaru.algorithms.Mandacaru` is attached as its *calculator*
(``atoms.calc = Mandacaru(method="adapt-vqe", ...)``); with
``basis={"name": "HAO"}`` the Hydrogenic Atomic Orbitals of
each atom (Li {1s, 2s} + H {1s} = 3 spatial orbitals -> 6 qubits) are generated
from the geometry, and ``atoms.get_total_energy()`` drives ADAPT-VQE with the CEO
pool, returning the energy in **eV**; the full run result is on
``atoms.calc.result``.

.. note::

   LiH on a uniform real-space grid is **qualitative**: the tightly contracted
   Li 1s core is only partially resolved at a practical grid spacing.  The
   invariant checked here is *self-consistent* -- ADAPT-VQE must recover the exact
   (FCI) ground state **of the Hamiltonian it is given**.
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


# 1. Define LiH via ASE and attach the Mandacaru.  The molecule is
#    placed at the center of the cell (7.5, 7.5, 7.5): the auto-generated grid is
#    centered on the cell, so the atoms must sit inside it -- putting them at the
#    origin would leave the orbitals hanging off the box corner and wreck the
#    integrals.
atoms = Atoms("LiH",
              positions=[[7.5, 7.5, 7.5 - 0.7975], [7.5, 7.5, 7.5 + 0.7975]],
              cell=[[15.0, 0.0, 0.0], [0.0, 15.0, 0.0], [0.0, 0.0, 15.0]],
              pbc=True)

atoms.calc = Mandacaru(method="adapt-vqe",
                       pool="ceo",
                       basis={"name": "HAO"},
                       mapping="jordan_wigner",
                       gradient="parameter_shift",
                       device="AER_simulator",
                       h=0.10,
                       max_iterations=25,
                       gradient_tolerance=1e-3,
                       txt=os.path.join(DATA, "output_LiH.txt"))

# 2. Asking ASE for the energy runs the whole ADAPT-VQE simulation.
energy_ev = atoms.get_total_energy()               # eV (ASE convention)
result = atoms.calc.result                         # result.optimal_energy is eV
n_orbitals = atoms.calc.n_qubits // 2

# Exact reference: lowest eigenvalue of the qubit Hamiltonian (FCI).  The
# qubit Hamiltonian is internal (Hartree), so its FCI energy is converted once.
h_matrix = atoms.calc.hamiltonian.to_matrix()
exact_ev = float(from_hartree(
    np.linalg.eigvalsh(0.5 * (h_matrix + h_matrix.conj().T)).min(), "eV"))
print(f"Exact energy: {exact_ev:.6f} eV")

assert abs(energy_ev - exact_ev) < 1e-4 * HARTREE_TO_EV, "CEO ADAPT missed FCI"
print(f"LiH  {n_orbitals} orbitals ({atoms.calc.n_qubits} qubits), "
      f"num_particles={atoms.calc.num_particles}")
print(f"     E = {energy_ev:.6f} eV [qualitative], "
      f"error vs FCI {energy_ev - exact_ev:+.1e} eV")
print(f"     {result.num_operators} operators, "
      f"{result.metrics.cnot_count} CNOTs, converged={result.converged}")
