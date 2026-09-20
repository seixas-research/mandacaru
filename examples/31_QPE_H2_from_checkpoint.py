# -*- coding: utf-8 -*-
# file: examples/31_QPE_H2_from_checkpoint.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""H2: a checkpointed ADAPT-VQE wavefunction as the input state of QPE.

Three steps, each reading the previous one's file:

1. ADAPT-VQE on H2 (FAO, 4 qubits) writes a **wavefunction checkpoint** after
   every accepted operator -- the reference determinant, the generators, the
   angles, the qubit Hamiltonian and the solver's progress.  Interrupt the
   run and ``resume=`` the same file to continue it.
2. The checkpoint is loaded back with no driver at all: its state vector, its
   energy and its Qiskit circuit come straight from the file.
3. **Quantum phase estimation** takes that state and reads the eigenvalue off
   the phase.  The variational state overlaps the exact ground state almost
   perfectly, so the ground-state reading dominates; a tight energy window
   then buys resolution without more qubits.

QPE is simulated as the full 2^(system + evaluation) state vector, so the
memory is estimated and checked *before* anything is allocated.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.algorithms import QuantumPhaseEstimation, qpe_memory_estimate
from mandacaru.core import load_checkpoint

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
from mandacaru.optimizers import Optimizer
os.makedirs(DATA, exist_ok=True)
CHECKPOINT = os.path.join(DATA, "h2_wavefunction.json")

# 1. ADAPT-VQE, checkpointing as it grows ---------------------------------- #
atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
              cell=[6.0, 6.0, 6.0])
atoms.center()
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis="FAO",
                       h=0.25,
                       pool="fermionic",
                       optimizer=Optimizer(method="L-BFGS-B",
                                           maxiter=2000,
                                           tol=1e-12),
                       gradient_tolerance=1e-6,
                       checkpoint=CHECKPOINT)       # written after every operator
energy = atoms.get_potential_energy()
print(f"ADAPT-VQE: E = {energy:.6f} eV with "
      f"{atoms.calc.result.num_operators} operators -> {CHECKPOINT}")

# 2. The file on its own ----------------------------------------------------- #
checkpoint = load_checkpoint(CHECKPOINT)
print(checkpoint.summary())
print(f"state vector from the file: <H> = {checkpoint.expectation():.8f} Ha, "
      f"circuit: {checkpoint.circuit().num_qubits} qubits")

# 3. QPE from the checkpoint -------------------------------------------------- #
qpe = QuantumPhaseEstimation(n_evaluation_qubits=10)
wide = qpe.run(checkpoint)                    # prints the memory estimate
exact = float(np.min(wide.exact_energies))    # dense path: the spectrum is free
print(f"wide window:  reading {wide.energy:+.6f} eV, exact {exact:+.6f} eV, "
      f"resolution {wide.resolution:.2e} eV, P = {wide.probability:.3f}")

window = (checkpoint.energy - 0.02, checkpoint.energy + 0.02)   # Hartree
tight = QuantumPhaseEstimation(n_evaluation_qubits=10, energy_window=window,
                               verbose=False).run(checkpoint)
print(f"tight window: reading {tight.energy:+.6f} eV, exact {exact:+.6f} eV, "
      f"resolution {tight.resolution:.2e} eV, P = {tight.probability:.3f}")

# The estimate for a run that must not be attempted on this machine.
print(qpe_memory_estimate(n_system=24, n_evaluation=16).summary())

# Plot the two distributions around the ground state. ----------------------- #
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, result, title in zip(axes, (wide, tight), ("default window",
                                                    "tight window")):
    ax.bar(result.energies, result.probabilities, width=result.resolution,
           color="tab:blue")
    ax.axvline(exact, color="tab:red", lw=1, ls="--", label="exact E0")
    ax.set_xlim(exact - 25 * result.resolution, exact + 25 * result.resolution)
    ax.set_xlabel("energy (eV)")
    ax.set_title(f"{title}: {result.resolution:.1e} eV / bin")
    ax.legend()
axes[0].set_ylabel("probability")
fig.tight_layout()
fig.savefig(os.path.join(DATA, "qpe_h2.png"), dpi=150)
print("wrote", os.path.join(DATA, "qpe_h2.png"))
