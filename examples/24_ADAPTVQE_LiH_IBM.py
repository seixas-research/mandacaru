# -*- coding: utf-8 -*-
# file: examples/24_ADAPTVQE_LiH_IBM.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""LiH dissociation curve with ADAPT-VQE (CEO pool, Jordan-Wigner mapping).

Runs locally as written.  To run on IBM Quantum hardware, set

    DEVICE = "ibm-quantum"      # least-busy QPU of your account, or e.g. "ibm_torino"
    SHOTS = 4096                # hardware never returns a state vector

after saving your account once with ``QiskitRuntimeService.save_account(...)``.
``DEVICE = "fake_torino"`` with ``SHOTS > 0`` rehearses the hardware run locally.
"""

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms

from carcara.algorithms import Carcara

DEVICE = "AER_simulator"
SHOTS = 0

DISTANCES = [1.0, 1.3, 1.6, 1.9, 2.2, 2.6, 2.8]     # Angstrom
CELL = 15.0                                         # cubic cell edge (Angstrom)


def lih(distance):
    """LiH along z, centered in a cubic cell."""
    c = CELL / 2.0
    return Atoms("LiH",
                 positions=[[c, c, c - distance / 2.0], [c, c, c + distance / 2.0]],
                 cell=[CELL, CELL, CELL], pbc=True)


def calculator():
    return Carcara(
        method="adapt-vqe",
        pool="ceo",
        mapping="jordan_wigner",
        basis={"name": "GTO", "n_gaussians": 3},
        h=0.15,
        charge=0,
        spin=False,
        frozen_core=False,
        initial_state="hartree-fock",
        optimizer="COBYLA",
        gradient="finite_difference",
        max_iterations=14,
        gradient_tolerance=1e-3,
        quenching=True,
        sparse="auto",
        device=DEVICE,
        shots=SHOTS,
        backend_provider="qiskit",
        execute_circuits=SHOTS > 0,
        profile=True,
        verbose=False,
    )


energies = []
print(f"{'d (A)':>8}{'E (Ha)':>16}{'ops':>6}")
for distance in DISTANCES:
    atoms = lih(distance)
    atoms.calc = calculator()
    atoms.get_total_energy()
    result = atoms.calc.result
    energies.append(result.optimal_energy)
    print(f"{distance:>8.2f}{result.optimal_energy:>16.8f}{result.num_operators:>6}")

plt.plot(DISTANCES, energies, "o-")
plt.xlabel("Li-H distance (Angstrom)")
plt.ylabel("energy (Ha)")
plt.title(f"LiH, ADAPT-VQE (ceo pool, Jordan-Wigner) on {DEVICE}")
plt.savefig("lih_dissociation_ibm.png", dpi=150)
