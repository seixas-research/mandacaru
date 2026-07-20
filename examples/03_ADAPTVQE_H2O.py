# -*- coding: utf-8 -*-
# file: examples/03_ADAPTVQE_H2O.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Water (H2O) ground state with ADAPT-VQE in a frozen-core active space.

Water is the textbook frozen-core minimal-basis problem.  With
``basis={"name": "FAO"}`` each atom contributes its Full Atomic Orbitals
(O {1s, 2s, 2p} + 2 H {1s} = 7 spatial orbitals) and ``frozen_core=True`` freezes
the oxygen ``1s`` core -- so oxygen enters as a frozen ``[1s^2]`` core plus an
active ``[2s^2, 2p^4]`` valence, giving a **6-orbital / 12-qubit** active space
with 8 active electrons (a ``(4, 4)`` closed shell).

At 12 qubits a *dense* operator pool would need tens of GB, so ADAPT-VQE
automatically switches to its **sparse** pool (``sparse="auto"``): the generators
are kept as sparse matrices and screened with the exact analytic gradient, and
only the few selected operators are ever densified.  ``atoms.get_total_energy()``
returns the energy in **eV**.

.. note::

   On a uniform real-space grid this is **qualitative** (the contracted cores are
   only partially resolved).  This is a strongly correlated active-space
   Hamiltonian, and ADAPT-VQE with the fermionic pool converges to a *stationary
   point* that recovers part of the correlation energy rather than the full FCI
   ground state -- a known behaviour of adaptive VQE on such systems.  The example
   verifies the frozen-core active space, the sparse pool and a genuine energy
   lowering, and reports the gap to the sector FCI for context.
"""

from __future__ import annotations

import os

import numpy as np
from ase import Atoms

from carcara.algorithms import ADAPTVQE
from carcara.units import from_hartree

# All generated files (logs, CSV, plots) go to examples/data/.
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)



def sector_fci(pauli_hamiltonian, n_qubits, na, nb):
    """Lowest eigenvalue in the ``(na, nb)`` particle/spin sector.

    The ADAPT ansatz starts from the Hartree-Fock determinant and grows with
    spin- and number-conserving excitations, so it lives in the sector with ``na``
    alpha (qubits ``0..M-1``) and ``nb`` beta (qubits ``M..2M-1``) electrons; that
    sector's FCI is the correct reference (qubit 0 is the most significant bit).
    The 12-qubit Hamiltonian is assembled *sparsely* -- a dense ``2^12 x 2^12``
    build would be needlessly slow -- and only the small sector block is densified.
    """
    M = n_qubits // 2
    keep = [i for i in range(2 ** n_qubits)
            if sum((i >> (n_qubits - 1 - q)) & 1 for q in range(M)) == na
            and sum((i >> (n_qubits - 1 - q)) & 1 for q in range(M, n_qubits)) == nb]
    keep = np.array(keep)
    h_sparse = pauli_hamiltonian.to_sparse_matrix()
    sub = h_sparse[np.ix_(keep, keep)].toarray()
    return float(np.linalg.eigvalsh(0.5 * (sub + sub.conj().T)).min())


# 1. Water geometry (O at the cell centre; O-H = 0.958 A, angle 104.5 deg).
theta = np.deg2rad(104.5) / 2.0
r = 0.958
atoms = Atoms("OH2",
              positions=[[5.0, 5.0, 5.0],
                         [5.0, 5.0 + r * np.sin(theta), 5.0 + r * np.cos(theta)],
                         [5.0, 5.0 - r * np.sin(theta), 5.0 + r * np.cos(theta)]],
              cell=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
              pbc=True)

atoms.calc = ADAPTVQE(
              pool="fermionic",
              basis={"name": "FAO"},
              mapping="jordan_wigner",
              frozen_core=True,              # freeze the oxygen 1s core
              h=0.30,
              max_iterations=20,
              gradient_tolerance=1e-3,
              profile=False,                 # skip per-iteration circuit transpile
              output=os.path.join(DATA, "output_H2O.txt"),
              # expressibility sampling is costly at 12 qubits; skip it in the log
              run_options={"log_expressivity": False})

# 2. Asking ASE for the energy runs the whole ADAPT-VQE simulation.
energy_ev = atoms.get_total_energy()               # eV (ASE convention)
result = atoms.calc.adapt_result
energy_ha = result.optimal_energy
na, nb = atoms.calc.num_particles

# The frozen core must have shrunk the space to 12 qubits (O [1s^2] frozen, O
# [2s^2, 2p^4] + 2 H(1s) active) with a (4, 4) closed shell, the sparse pool must
# have kicked in, and ADAPT must lower the energy below the Hartree-Fock reference.
assert atoms.calc.n_qubits == 12 and (na, nb) == (4, 4), "unexpected active space"
assert atoms.calc._sparse, "the 12-qubit run should use the sparse pool"
assert result.optimal_energy < result.reference_energy, "ADAPT did not lower E"

# Sector FCI, for context (ADAPT with the fermionic pool converges to a
# stationary point that recovers part of the correlation energy -- a known
# behaviour on a strongly correlated Hamiltonian like this qualitative water).
exact_ha = sector_fci(atoms.calc.hamiltonian, atoms.calc.n_qubits, na, nb)
recovered = from_hartree(result.reference_energy - energy_ha, "eV")
print(f"H2O  {atoms.calc.n_qubits // 2} active orbitals "
      f"({atoms.calc.n_qubits} qubits, O [1s^2] frozen), "
      f"num_particles=({na}, {nb}), sparse pool={atoms.calc._sparse}")
print(f"     E = {energy_ev:.6f} eV ({energy_ha:.8f} Ha) [qualitative]")
print(f"     correlation recovered = {recovered:.4f} eV below the HF reference")
print(f"     sector-FCI = {from_hartree(exact_ha, 'eV'):.6f} eV "
      f"(gap {energy_ev - from_hartree(exact_ha, 'eV'):+.3f} eV)")
print(f"     {result.num_operators} operators, converged={result.converged}")
