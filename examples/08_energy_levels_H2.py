# -*- coding: utf-8 -*-
# file: examples/08_energy_levels_H2.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""H2 molecular energy levels (ground + excited state) via variational deflation.

:class:`~carcara.algorithms.Carcara` exposes ``energy_levels``, which
computes the low-lying spectrum with variational quantum deflation (VQD): each
excited level minimizes ``<H> + beta * sum_j |<psi_j|psi>|^2`` over the
previously found states, so it is pushed orthogonal to them.

Here H2 is defined once as an ASE :class:`ase.Atoms` object; attaching
``Carcara(method="vqe", ...)`` and calling ``get_potential_energy()``
builds the Hamiltonian and configures the solver, after which ``energy_levels``
returns the ground state and the first excited state.  Every returned level is a
true eigenvalue of the qubit Hamiltonian (checked against exact
diagonalization).
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from carcara.algorithms import Carcara
from carcara.units import from_hartree


atoms = Atoms("H2",
              positions=[[4.0, 4.0, 3.63], [4.0, 4.0, 4.37]],
              cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
              pbc=True)

atoms.calc = Carcara(method="vqe",
                     basis="FAO",
                     mapping="jordan_wigner",
                     h=0.20,
                     verbose=False)

# get_potential_energy() builds the Hamiltonian + UCCSD ansatz and runs the
# ground-state VQE, leaving the calculator configured for energy_levels().
ground_ev = atoms.get_potential_energy()

levels = atoms.calc.energy_levels(num_states=2, restarts=4)

# Exact reference spectrum: eigenvalues of the qubit Hamiltonian.  The qubit
# Hamiltonian is internal (Hartree); the levels, like every result, are eV.
h = atoms.calc.hamiltonian.to_matrix()
exact = from_hartree(np.sort(np.linalg.eigvalsh(0.5 * (h + h.conj().T)).real), "eV")

TOLERANCE = from_hartree(1e-5, "eV")               # 2.7e-4 eV
print(f"H2 ground state   = {ground_ev:.6f} eV")
print(f"energy levels (eV): {np.round(levels.energies, 4)}")
print(f"excitation  (eV)  : {np.round(levels.excitation_energies, 4)}")
for e in levels.energies:
    match = float(np.min(np.abs(exact - e)))
    assert match < TOLERANCE, "energy level is not an eigenvalue of H"
print(f"all levels match exact diagonalization (<= {TOLERANCE:.1e} eV).")
