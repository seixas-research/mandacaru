# -*- coding: utf-8 -*-
# file: examples/24_ADAPTVQE_LiH_IBM.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""LiH dissociation curve with ADAPT-VQE (CEO pool, Jordan-Wigner mapping),
optimized locally and measured on IBM Quantum hardware.

The pool is ``"ceo-ovp"``: coupled exchange operators (Ramoa *et al.*, npj
Quantum Inf. **11**, 86, 2025) kept in their one-parameter form.  A CEO couples
the qubit excitations acting on the same spin-orbitals, and their sum or
difference is a combination of four Pauli strings where a single qubit
excitation needs eight -- which is what halves the gate count.  On this curve it
costs nothing in energy and gives **104 CNOTs against 208 for ``"qeb"``**.  The
adaptive ``"ceo"`` is the paper's full algorithm and would be the better choice
with its specialized circuits; without them its multi-parameter growth steps
compile as separate excitations and cost *more* gates (248 here), so a hardware
example uses the one-parameter variant.

The variational optimization runs on the local state vector.  The optimized
states are then measured with the Qiskit Runtime ``Estimator`` on a real
processor -- one job for the whole curve -- when ``HARDWARE`` names one:

    HARDWARE = "ibm_kingston,ibm_fez,ibm_marrakesh"   # least busy of these
    HARDWARE = "fake_kingston"                        # local rehearsal
    HARDWARE = None                                   # local only

Your IBM account must be saved once with ``QiskitRuntimeService.save_account``.
"""

import csv
import os

import matplotlib.pyplot as plt
from ase import Atoms

from mandacaru.algorithms import Mandacaru
from mandacaru.algorithms.base import measure_energies
from mandacaru.backends.providers import QiskitProvider

# Outputs go to examples/data/ like every other example.
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)
from mandacaru.units import HARTREE_TO_EV

HARDWARE = None
from mandacaru.optimizers import Optimizer
SHOTS = 4096

DISTANCES = [1.0, 1.3, 1.6, 1.9, 2.2, 2.6, 2.8]     # Angstrom
CELL = 15.0                                         # cubic cell edge (Angstrom)


def lih(distance):
    """LiH along z, centered in a cubic cell."""
    c = CELL / 2.0
    return Atoms("LiH",
                 positions=[[c, c, c - distance / 2.0], [c, c, c + distance / 2.0]],
                 cell=[CELL, CELL, CELL], pbc=True)


def calculator():
    return Mandacaru(method="adapt-vqe",
                     pool="ceo-ovp",
                     mapping="jordan_wigner",
                     basis={"name": "GTO", "n_gaussians": 3},
                     h=0.15,
                     charge=0,
                     spin=False,
                     frozen_core=False,
                     initial_state="hartree-fock",
                     optimizer=Optimizer(method="COBYLA",
                                         maxiter=2000,
                                         tol=1e-12),
                     gradient="finite_difference",
                     max_iterations=14,
                     gradient_tolerance=1e-3,
                     quenching=True,
                     sparse="auto",
                     device="AER_simulator",
                     shots=0,
                     backend_provider="qiskit",
                     execute_circuits=False,
                     profile=True)


# 1. Optimize locally.
solvers, energies = [], []
print(f"{'d (A)':>8}{'E local (eV)':>16}{'ops':>6}{'cnot':>7}{'depth':>7}")
for distance in DISTANCES:
    atoms = lih(distance)
    atoms.calc = calculator()
    atoms.get_total_energy()
    result = atoms.calc.result
    solvers.append(atoms.calc.solver)
    energies.append(result.optimal_energy)
    print(f"{distance:>8.2f}{result.optimal_energy:>16.6f}"
          f"{result.num_operators:>6}{result.metrics.cnot_count:>7}"
          f"{result.metrics.depth:>7}")

plt.plot(DISTANCES, energies, "o-", label="local state vector")

# 2. Measure the optimized states on hardware, one Estimator job.
measured, stds, backend_name, job_id = [], [], "", ""
if HARDWARE:
    provider = QiskitProvider(device=HARDWARE, shots=SHOTS)
    backend_name = provider.backend().name
    print(f"\nmeasuring on {backend_name} ({SHOTS} shots)")
    measured = measure_energies(solvers, provider)          # eV
    job_id = provider.last_job.job_id() if provider.last_job else ""
    # The Estimator's standard errors are in the Hamiltonian's Hartree.
    stds = [float(r.data.stds) * HARTREE_TO_EV for r in provider.last_result]
    print(f"job {job_id}")
    print(f"{'d (A)':>8}{'E measured (eV)':>18}{'std (eV)':>12}")
    for distance, energy, std in zip(DISTANCES, measured, stds):
        print(f"{distance:>8.2f}{energy:>18.6f}{std:>12.4f}")
    plt.errorbar(DISTANCES, measured, yerr=stds, fmt="s--", label=backend_name)

# 3. Raw data, one row per distance.
with open(os.path.join(DATA, "lih_dissociation_ibm.csv"), "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["distance_A", "energy_local_eV", "num_operators",
                     "cnot_count", "circuit_depth",
                     "energy_measured_eV", "std_measured_eV", "shots",
                     "backend", "job_id", "optimal_parameters"])
    for i, distance in enumerate(DISTANCES):
        result = solvers[i].result
        writer.writerow([distance, energies[i], result.num_operators,
                         result.metrics.cnot_count, result.metrics.depth,
                         measured[i] if measured else "",
                         stds[i] if stds else "", SHOTS if HARDWARE else 0,
                         backend_name, job_id,
                         " ".join(f"{t:.10f}" for t in result.optimal_parameters)])

plt.xlabel("Li-H distance (Angstrom)")
plt.ylabel("energy (eV)")
plt.title("LiH, ADAPT-VQE (ceo-ovp pool, Jordan-Wigner)")
plt.legend()
plt.savefig(os.path.join(DATA, "lih_dissociation_ibm.png"), dpi=150)
