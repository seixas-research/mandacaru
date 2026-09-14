# -*- coding: utf-8 -*-
# file: examples/09_SubspaceVQE_H2.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""H2 ground + first excited state at once with subspace-search VQE (SSVQE).

Unlike deflation (which finds excited states one after another), **subspace-search
VQE** finds several states in a *single* optimization: it sends orthogonal
reference determinants through one shared unitary ``U(theta)`` and minimizes a
weighted energy sum ``sum_j w_j <phi_j|U' H U|phi_j>`` with descending weights, so
``U|phi_0>`` becomes the ground state, ``U|phi_1>`` the first excited state, etc.

Both subspace methods are driven through
:class:`~carcara.algorithms.Carcara` (``method="subspace-vqe"`` and
``method="subspace-adapt-vqe"``); here H2 is defined once and a subspace
calculator returns the whole low-lying spectrum on ``atoms.calc.result``.

The returned energies are variational **upper bounds** on the exact levels
(Hylleraas-Undheim): the ground state is recovered exactly, while excited-state
accuracy depends on how expressive the shared ansatz is.
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

# Exact reference spectrum (built after the first calculation, below).
def report(name, result, exact):
    """``result.energies`` and ``exact`` are both in eV."""
    print(f"\n{name}: {result.num_states} states in one optimization")
    for i, e_ev in enumerate(result.energies):
        tag = "ground " if i == 0 else f"excited{i}"
        bound = exact[i]
        print(f"  E[{i}] ({tag}) = {e_ev:10.4f} eV   "
              f"(exact {bound:10.4f} eV, "
              f"E_i >= lambda_i: {e_ev >= bound - from_hartree(1e-6, 'eV')})")
    print(f"  first excitation energy = {result.excitation_energies[1]:.3f} eV")


# --- Subspace-search VQE (fixed UCCSD ansatz) --------------------------------
atoms.calc = Carcara(method="subspace-vqe", basis="FAO", h=0.20,
                               mapping="jordan_wigner", num_states=2,
                               weights=[2.0, 1.0], verbose=False)
atoms.get_potential_energy()
ssvqe = atoms.calc.result

# Exact eigenvalues of the qubit Hamiltonian for comparison.  The qubit
# Hamiltonian is internal (Hartree); every result energy is eV, so convert once.
h = atoms.calc.hamiltonian.to_matrix()
exact = from_hartree(np.sort(np.linalg.eigvalsh(0.5 * (h + h.conj().T)).real), "eV")

report("Subspace-VQE", ssvqe, exact)

# --- Subspace-search ADAPT-VQE (one shared, adaptively grown ansatz) ---------
atoms.calc = Carcara(method="subspace-adapt-vqe", basis="FAO", h=0.20,
                               pool="fermionic", num_states=2, verbose=False,
                               profile=False, gradient_tolerance=1e-4,
                               max_iterations=20)
atoms.get_potential_energy()
ss_adapt = atoms.calc.result
report("Subspace-ADAPT-VQE", ss_adapt, exact)
print(f"  ansatz grew {ss_adapt.num_operators} operator(s)")

# The ground state is exact; every level is a valid variational upper bound.
assert abs(ssvqe.energies[0] - exact[0]) < from_hartree(1e-4, "eV")
for i, e in enumerate(ssvqe.energies):
    assert e >= exact[i] - from_hartree(1e-6, "eV"), "Hylleraas-Undheim bound violated"
print("\nground state exact; all levels satisfy the Hylleraas-Undheim bounds.")
