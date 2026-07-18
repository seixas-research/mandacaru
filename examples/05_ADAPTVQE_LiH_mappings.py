# -*- coding: utf-8 -*-
# file: examples/05_ADAPTVQE_LiH_mappings.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""LiH ground state with ADAPT-VQE across fermion-to-qubit mappings.

The same LiH molecule and fermionic pool are run through each of Carcará's three
fermion-to-qubit mappings -- **Jordan-Wigner**, **parity** and **Bravyi-Kitaev**
-- selected with the ``mapping`` argument of
:class:`~carcara.algorithms.ADAPTVQE`.  The mappings encode the fermionic
Hamiltonian into *different* qubit Pauli operators, but all describe the same
physics, so ADAPT-VQE must recover the **same** ground-state energy (the FCI
eigenvalue of the mapping's qubit Hamiltonian) in every case.

The ``fermionic`` pool is used here because it maps its excitation generators
through the *same* ``mapping`` as the Hamiltonian; the qubit-tailored pools
(``qubit`` / ``qeb`` / ``ceo``) are defined via Jordan-Wigner Pauli strings and
are meant for the default Jordan-Wigner mapping.

.. note::

   LiH on a uniform real-space grid is **qualitative** (the contracted Li 1s core
   is only partially resolved); the mapping-independence checked here is exact for
   the Hamiltonian each run is given.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from carcara.algorithms import ADAPTVQE
from carcara.units import from_hartree

# LiH centred in the cell so the auto-generated grid covers both orbitals.
atoms = Atoms("LiH",
              positions=[[7.5, 7.5, 7.5 - 0.7975], [7.5, 7.5, 7.5 + 0.7975]],
              cell=[[15.0, 0.0, 0.0], [0.0, 15.0, 0.0], [0.0, 0.0, 15.0]],
              pbc=True)

energies = {}
for mapping in ("jordan_wigner", "parity", "bravyi_kitaev"):
    atoms.calc = ADAPTVQE(
                  pool="fermionic",
                  basis={"name": "FAO"},
                  mapping=mapping,
                  gradient="parameter-shift_rule",
                  h=0.10,
                  max_iterations=25,
                  gradient_tolerance=1e-3,
                  verbose=False)                    # keep the loop output compact

    energy_ev = atoms.get_total_energy()            # eV (ASE convention)
    result = atoms.calc.adapt_result
    energy_ha = result.optimal_energy

    # FCI reference: lowest eigenvalue of *this* mapping's qubit Hamiltonian.
    h_matrix = atoms.calc.hamiltonian.to_matrix()
    exact_ha = float(np.linalg.eigvalsh(0.5 * (h_matrix + h_matrix.conj().T)).min())
    assert abs(energy_ha - exact_ha) < 1e-4, f"{mapping}: ADAPT missed FCI"

    energies[mapping] = energy_ha
    print(f"{mapping:16s}  {atoms.calc.n_qubits} qubits  "
          f"E = {energy_ev:.6f} eV ({energy_ha:.8f} Ha)  "
          f"{result.num_operators} ops, {result.metrics.cnot_count} CNOTs")

# All three mappings describe the same physics -> the same ground-state energy.
spread = max(energies.values()) - min(energies.values())
assert spread < 1e-4, "mappings disagree on the ground-state energy"
print(f"\nmapping spread = {from_hartree(spread, 'eV'):.2e} eV "
      f"(all mappings agree on the FCI ground state)")
