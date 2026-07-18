# -*- coding: utf-8 -*-
# file: examples/lih_adapt_ceo_ase.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""LiH ground state with ADAPT-VQE (CEO pool) as an ASE calculator.

Companion to ``h2_adapt_ceo_ase.py``, for the four-electron LiH molecule.  LiH is
defined as an ASE :class:`ase.Atoms` object and :class:`~carcara.algorithms.ADAPTVQE`
is attached as its *calculator*; with ``basis="FAO"`` the Full Atomic Orbitals of
each atom (Li {1s, 2s} + H {1s} = 3 spatial orbitals -> 6 qubits) are generated
from the geometry, and ``atoms.get_total_energy()`` drives ADAPT-VQE with the CEO
pool, returning the energy in **eV**.

.. note::

   LiH on a uniform real-space grid is **qualitative**: the tightly contracted
   Li 1s core is only partially resolved at a practical grid spacing.  The
   invariant checked here is *self-consistent* -- ADAPT-VQE must recover the exact
   (FCI) ground state **of the Hamiltonian it is given**.

Run with::

    PYTHONPATH=src python examples/lih_adapt_ceo_ase.py
"""

from __future__ import annotations

import os

import numpy as np
from ase import Atoms

from carcara.algorithms import ADAPTVQE
from carcara.integrals import Grid
from carcara.units import from_hartree
from carcara.utils import parse_output

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data")
os.makedirs(DATA_DIR, exist_ok=True)
output_file = os.path.join(DATA_DIR, "output_lih.txt")

# 1. Define LiH via ASE and attach ADAPTVQE as its calculator.  gradient_tol is
#    matched to COBYLA (the default optimizer): a tighter screen than the
#    gradient-free optimizer resolves would bloat the ansatz with re-selected
#    operators after the energy has already converged.
atoms = Atoms("LiH", positions=[[0.0, 0.0, -0.7975], [0.0, 0.0, +0.7975]])
atoms.calc = ADAPTVQE(
    pool="ceo",
    basis="FAO",
    mapping="jordan_wigner",
    gradient="parameter-shift_rule",
    device="AER_simulator",
    grid=Grid(center=[0.0, 0.0, 0.0], box_size=7.0, h=0.18),
    max_iterations=25,
    gradient_tolerance=1e-3,
    output=output_file)

# 2. Asking ASE for the energy runs the whole ADAPT-VQE simulation.
energy_ev = atoms.get_total_energy()               # eV (ASE convention)
result = atoms.calc.adapt_result
energy_ha = result.optimal_energy
n_orbitals = atoms.calc.n_qubits // 2

# Exact reference: lowest eigenvalue of the qubit Hamiltonian (FCI).
h_matrix = atoms.calc.hamiltonian.to_matrix()
exact_ha = float(np.linalg.eigvalsh(0.5 * (h_matrix + h_matrix.conj().T)).min())
exact_ev = float(from_hartree(exact_ha, "eV"))

print("LiH from ASE Atoms:", atoms.get_chemical_symbols(), "@ d = 1.595 Angstrom")
print(f"Active space: {n_orbitals} orbitals ({atoms.calc.n_qubits} qubits), "
      f"4 electrons -> num_particles = {atoms.calc.num_particles}")
print(f"pool=ceo  basis=FAO  mapping={atoms.calc.mapping}  "
      f"gradient={atoms.calc.gradient}  device={atoms.calc.device}")
print(f"\natoms.get_total_energy() = {energy_ev:+.6f} eV  "
      f"({energy_ha:+.8f} Ha)  [qualitative]")
print(f"FCI reference            = {exact_ev:+.6f} eV  ({exact_ha:+.8f} Ha)")
print(f"error vs FCI             = {energy_ev - exact_ev:+.2e} eV")

print("\nADAPT-VQE (CEO pool) final ansatz")
print("-" * 48)
print(f"  operators     = {result.num_operators}")
print(f"  CNOTs / depth = {result.metrics.cnot_count} / {result.metrics.depth}")
print(f"  total gates   = {result.metrics.total_gates}")
print(f"  converged     = {result.converged}")

assert abs(energy_ha - exact_ha) < 1e-4, "CEO ADAPT missed FCI"
print("\nReached the FCI ground state of this Hamiltonian to < 1e-4 Ha.")

# 3. Read back the structured (eV / Angstrom) trace the loop wrote live.
parsed = parse_output(output_file)
print(f"\nWrote structured trace -> {output_file}")
print(f"  units          : {parsed['setup'].get('energy_unit')} / "
      f"{parsed['metadata'].get('units')}")
for it in parsed["iterations"]:
    print(f"  iteration {it['index']:>2}: {it['selected_operator']:<28} "
          f"E={it.get('expressivity_E')}  energy={it['energy']:+.4f} "
          f"{it['energy_unit']}")
