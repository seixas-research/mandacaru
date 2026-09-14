# -*- coding: utf-8 -*-
# file: examples/25_ADAPTVQE_H2_IBM.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""H2 dissociation curve with ADAPT-VQE on two qubits (fermionic pool, parity
mapping with the two-qubit reduction), optimized locally and measured on IBM
Quantum hardware.

The variational optimization runs on the local state vector.  The optimized
states are then measured with the Qiskit Runtime ``Estimator`` on a real
processor -- one job for the whole curve -- when ``HARDWARE`` names one:

    HARDWARE = "ibm_kingston,ibm_fez,ibm_marrakesh"   # least busy of these
    HARDWARE = "fake_kingston"                        # local rehearsal
    HARDWARE = None                                   # local only

Your IBM account must be saved once with ``QiskitRuntimeService.save_account``.
"""

import csv

import matplotlib.pyplot as plt
from ase import Atoms

from carcara.algorithms import Carcara
from carcara.algorithms.base import measure_energies
from carcara.backends.providers import QiskitProvider
from carcara.units import HARTREE_TO_EV

HARDWARE = None
SHOTS = 4096
#: Qiskit Runtime error mitigation: level 2 adds gate twirling and
#: zero-noise extrapolation to the readout mitigation of level 1.
ESTIMATOR_OPTIONS = {"resilience_level": 2}
#: Physical qubits to run on (Carcara qubit k on PHYSICAL_QUBITS[k]); None
#: lets the transpiler choose.  Pin a pair with good *current* readout: a
#: drifted qubit ruins the energy while its calibration record looks fine.
PHYSICAL_QUBITS = None

DISTANCES = [0.5, 0.6, 0.74, 0.9, 1.1, 1.4, 1.8]   # Angstrom
CELL = 10.0                                         # cubic cell edge (Angstrom)


def h2(distance):
    """H2 along z, centered in a cubic cell."""
    c = CELL / 2.0
    return Atoms("H2",
                 positions=[[c, c, c - distance / 2.0], [c, c, c + distance / 2.0]],
                 cell=[CELL, CELL, CELL], pbc=True)


def calculator():
    return Carcara(
        method="adapt-vqe",
        pool="fermionic",
        mapping="parity",
        two_qubit_reduction=True,       # H2 on 2 qubits instead of 4
        basis="FAO",
        h=0.10,
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
        device="AER_simulator",
        shots=0,
        backend_provider="qiskit",
        execute_circuits=False,
        profile=True,
        verbose=False,
    )


# 1. Optimize locally.
solvers, energies = [], []
print(f"{'d (A)':>8}{'E local (eV)':>16}{'ops':>6}")
for distance in DISTANCES:
    atoms = h2(distance)
    atoms.calc = calculator()
    atoms.get_total_energy()
    result = atoms.calc.result
    solvers.append(atoms.calc.solver)
    energies.append(result.optimal_energy)
    print(f"{distance:>8.2f}{result.optimal_energy:>16.6f}{result.num_operators:>6}")

plt.plot(DISTANCES, energies, "o-", label="local state vector")

# 2. Measure the optimized states on hardware, one Estimator job.
measured, stds, backend_name, job_id = [], [], "", ""
if HARDWARE:
    provider = QiskitProvider(device=HARDWARE, shots=SHOTS,
                              estimator_options=ESTIMATOR_OPTIONS,
                              physical_qubits=PHYSICAL_QUBITS)
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
with open("h2_dissociation_ibm.csv", "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["distance_A", "energy_local_eV", "num_operators",
                     "energy_measured_eV", "std_measured_eV", "shots",
                     "backend", "job_id", "optimal_parameters"])
    for i, distance in enumerate(DISTANCES):
        result = solvers[i].result
        writer.writerow([distance, energies[i], result.num_operators,
                         measured[i] if measured else "",
                         stds[i] if stds else "", SHOTS if HARDWARE else 0,
                         backend_name, job_id,
                         " ".join(f"{t:.10f}" for t in result.optimal_parameters)])

plt.xlabel("H-H distance (Angstrom)")
plt.ylabel("energy (eV)")
plt.title("H2 (FAO), ADAPT-VQE, parity mapping, 2 qubits")
plt.legend()
plt.savefig("h2_dissociation_ibm.png", dpi=150)
