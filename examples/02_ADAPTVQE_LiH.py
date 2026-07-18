# -*- coding: utf-8 -*-
# file: examples/lih_adapt_ceo_ase.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""LiH ground state with ADAPT-VQE (CEO pool) as an ASE calculator.

LiH is defined as an ASE :class:`ase.Atoms` object and :class:`~carcara.algorithms.ADAPTVQE`
is attached as its *calculator*; with ``basis={"name": "FAO"}`` the Full Atomic Orbitals of
each atom (Li {1s, 2s} + H {1s} = 3 spatial orbitals -> 6 qubits) are generated
from the geometry, and ``atoms.get_total_energy()`` drives ADAPT-VQE with the CEO
pool, returning the energy in **eV**.

.. note::

   LiH on a uniform real-space grid is **qualitative**: the tightly contracted
   Li 1s core is only partially resolved at a practical grid spacing.  The
   invariant checked here is *self-consistent* -- ADAPT-VQE must recover the exact
   (FCI) ground state **of the Hamiltonian it is given**.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from carcara.algorithms import ADAPTVQE
from carcara.units import from_hartree

# 1. Define LiH via ASE and attach ADAPTVQE as its calculator.  The molecule is
#    placed at the centre of the cell (7.5, 7.5, 7.5): the auto-generated grid is
#    centred on the cell, so the atoms must sit inside it -- putting them at the
#    origin would leave the orbitals hanging off the box corner and wreck the
#    integrals.
atoms = Atoms("LiH",
              positions=[[7.5, 7.5, 7.5 - 0.7975], [7.5, 7.5, 7.5 + 0.7975]],
              cell=[[15.0, 0.0, 0.0], [0.0, 15.0, 0.0], [0.0, 0.0, 15.0]],
              pbc=True)

atoms.calc = ADAPTVQE(
              pool="ceo",
              basis={"name": "FAO"},
              mapping="jordan_wigner",
              gradient="parameter-shift_rule",
              device="AER_simulator",
              h=0.10,
              max_iterations=25,
              gradient_tolerance=1e-3,
              output="output_LiH.txt")

# 2. Asking ASE for the energy runs the whole ADAPT-VQE simulation.
energy_ev = atoms.get_total_energy()               # eV (ASE convention)
result = atoms.calc.adapt_result
energy_ha = result.optimal_energy
n_orbitals = atoms.calc.n_qubits // 2

# Exact reference: lowest eigenvalue of the qubit Hamiltonian (FCI).
h_matrix = atoms.calc.hamiltonian.to_matrix()
exact_ha = float(np.linalg.eigvalsh(0.5 * (h_matrix + h_matrix.conj().T)).min())
exact_ev = float(from_hartree(exact_ha, "eV"))
print(f"Exact energy: {exact_ev:.8f}")

assert abs(energy_ha - exact_ha) < 1e-4, "CEO ADAPT missed FCI"
print(f"LiH  {n_orbitals} orbitals ({atoms.calc.n_qubits} qubits), "
      f"num_particles={atoms.calc.num_particles}")
print(f"     E = {energy_ev:.6f} eV ({energy_ha:.8f} Ha) [qualitative], "
      f"error vs FCI {energy_ev - exact_ev:+.1e} eV")
print(f"     {result.num_operators} operators, "
      f"{result.metrics.cnot_count} CNOTs, converged={result.converged}")
