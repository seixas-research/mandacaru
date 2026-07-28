# -*- coding: utf-8 -*-
# file: examples/10_VASQE_H2.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""H2 ground state with VASQE (stochastic ADAPT-VQE) and temperature annealing.

VASQE grows an ADAPT-style ansatz but **samples** the next operator from a softmax
of the pool gradients at a selection temperature ``tau``:

    P(i, tau) = exp(|g_i| / tau) / sum_j exp(|g_j| / tau)

At low ``tau`` this concentrates on the largest-gradient operator, so VASQE
reduces to ADAPT-VQE; a high initial ``tau`` (optionally **annealed** down)
explores operators a greedy rule would skip.  The solver runs through the ASE
interface of :class:`~carcara.algorithms.QuantumCalculator`
(``method="vasqe"``), so H2 is defined once and the calculator is attached to
it; the run result is on ``atoms.calc.result``.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from carcara.algorithms import QuantumCalculator
from carcara.units import from_hartree


atoms = Atoms("H2",
              positions=[[4.0, 4.0, 3.63], [4.0, 4.0, 4.37]],
              cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
              pbc=True)


def run(label, **vasqe_kwargs):
    atoms.calc = QuantumCalculator(method="vasqe", basis="FAO", h=0.20,
                                   mapping="jordan_wigner",
                                   optimizer="L-BFGS-B", verbose=False,
                                   profile=False, max_iterations=12,
                                   gradient_tolerance=1e-5, **vasqe_kwargs)
    energy_ev = atoms.get_total_energy()
    res = atoms.calc.result
    taus = ", ".join(f"{t:.3g}" for t in res.temperatures)
    print(f"{label:34s} E = {energy_ev:10.6f} eV  "
          f"({res.num_operators} ops, schedule={res.schedule}, tau=[{taus}])")
    return res


# Exact FCI reference (lowest eigenvalue of the qubit Hamiltonian).
atoms.calc = QuantumCalculator(method="vasqe", basis="FAO", h=0.20,
                               temperature=1e-3, verbose=False, profile=False,
                               max_iterations=1, gradient_tolerance=1e-5)
atoms.get_total_energy()
h = atoms.calc.hamiltonian.to_matrix()
fci_ev = float(from_hartree(np.linalg.eigvalsh(0.5 * (h + h.conj().T)).min(), "eV"))
print(f"exact FCI ground state: {fci_ev:.6f} eV\n")

# (1) Low temperature -> reduces to ADAPT-VQE (greedy, reaches FCI).
run("VASQE  tau=0.001 (ADAPT limit)", temperature=1e-3)

# (2) Constant, warmer temperature -> still converges, explores more.
run("VASQE  tau=1.0 (constant)", temperature=1.0, seed=1)

# (3) Annealing: hot -> cold over the growth iterations (three schedules).
for schedule in ("exponential", "linear", "logarithmic"):
    run(f"VASQE  anneal 2.0->0.01 ({schedule})", temperature=2.0,
        final_temperature=0.01, schedule=schedule, seed=1)

print("\nAll runs converge to the ADAPT-VQE / FCI ground state; higher initial "
      "temperatures explore more operators before settling.")
