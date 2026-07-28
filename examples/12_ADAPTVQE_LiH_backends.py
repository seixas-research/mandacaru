# -*- coding: utf-8 -*-
# file: examples/12_ADAPTVQE_LiH_backends.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""LiH with ADAPT-VQE on all three backend providers: Qiskit, Braket and Cirq.

This example exercises two features together.

**1. The Hamiltonian disk cache.**  Building LiH's qubit Hamiltonian means running
the real-space one- and two-body integrals and then the Jordan-Wigner mapping --
by far the most expensive stage, and completely independent of which SDK runs the
circuits.  So it is built **once** with ``save_hamiltonian=...`` and every backend
run afterwards uses ``load_hamiltonian=...``, which skips the integrals and the
fermion-to-qubit transformation entirely.  Both modes go through
:class:`~carcara.algorithms.QuantumCalculator`: the build is the ASE-calculator
path (``atoms.calc = QuantumCalculator(method="adapt-vqe", ...)``), the cached
runs use its **direct mode**
(``QuantumCalculator(method="adapt-vqe", load_hamiltonian=...).run()``).

**2. Multi-backend circuit execution.**  With ``execute_circuits=True`` the ansatz
is no longer evaluated by the internal NumPy state-vector backend: each
:math:`e^{\\theta_k A_k}` is compiled to an explicit Pauli-rotation circuit
(basis change, CNOT ladder, :math:`R_z`, uncompute) and *executed* on that SDK's
local state-vector simulator.  The decomposition is exact for these generators,
so the three providers must agree with each other -- and with the internal
backend -- to machine precision.  That equivalence is what this script asserts.

.. note::

   LiH on a uniform real-space grid is **qualitative** (the contracted Li 1s core
   is only partially resolved at a practical spacing).  The invariant checked here
   is *self-consistent*: every backend must recover the exact (FCI) ground state
   **of the Hamiltonian it is given**, and they must all agree.

   Circuit execution is orders of magnitude slower than the matrix backend -- one
   simulator invocation per energy evaluation.  Braket in particular takes tens of
   seconds here; that is expected.
"""

from __future__ import annotations

import os
import time

import numpy as np
from ase import Atoms

from carcara.algorithms import QuantumCalculator
from carcara.backends.providers import BACKEND_PROVIDERS, provider_available
from carcara.units import from_hartree

# All generated files (logs, CSV, plots) go to examples/data/.
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

HAMILTONIAN_FILE = os.path.join(DATA, "lih_hamiltonian.parquet")
POOL = "qeb"
MAX_ITERATIONS = 12

# --------------------------------------------------------------------------- #
# 1. Build the Hamiltonian once and cache it to disk.
# --------------------------------------------------------------------------- #

# LiH at the center of the cell: the auto-generated grid is centered on the cell,
# so the atoms must sit inside it (see examples/02_ADAPTVQE_LiH.py).
atoms = Atoms("LiH",
              positions=[[7.5, 7.5, 7.5 - 0.7975], [7.5, 7.5, 7.5 + 0.7975]],
              cell=[[15.0, 0.0, 0.0], [0.0, 15.0, 0.0], [0.0, 0.0, 15.0]],
              pbc=True)

t0 = time.perf_counter()
atoms.calc = QuantumCalculator(method="adapt-vqe", pool=POOL,
                               basis={"name": "FAO"}, h=0.25,
                               mapping="jordan_wigner",
                               max_iterations=MAX_ITERATIONS,
                               verbose=False, save_hamiltonian=HAMILTONIAN_FILE)
atoms.get_total_energy()
build_seconds = time.perf_counter() - t0

reference = atoms.calc.result
n_qubits = atoms.calc.n_qubits

# Exact reference: lowest eigenvalue of the qubit Hamiltonian (FCI).
h_matrix = atoms.calc.hamiltonian.to_matrix()
exact_ha = float(np.linalg.eigvalsh(0.5 * (h_matrix + h_matrix.conj().T)).min())

print(f"LiH: {n_qubits // 2} spatial orbitals ({n_qubits} qubits), "
      f"num_particles={atoms.calc.num_particles}")
print(f"Hamiltonian built in {build_seconds:.2f} s and cached to "
      f"{HAMILTONIAN_FILE!r}")
print(f"FCI (exact diagonalization) = {exact_ha:.8f} Ha "
      f"({from_hartree(exact_ha, 'eV'):.6f} eV)")
print()

# --------------------------------------------------------------------------- #
# 2. Run the same ADAPT-VQE on each provider, from the cached Hamiltonian.
# --------------------------------------------------------------------------- #

header = (f"{'provider':<10}{'E (Ha)':>16}{'err vs FCI':>13}{'ops':>6}"
          f"{'cnots':>7}{'depth':>7}{'time':>9}")
print(header)
print("-" * len(header))

# The internal NumPy state-vector backend, for comparison.
t0 = time.perf_counter()
matrix_run = QuantumCalculator(method="adapt-vqe", pool=POOL,
                               load_hamiltonian=HAMILTONIAN_FILE,
                               max_iterations=MAX_ITERATIONS,
                               verbose=False).run()
print(f"{'(matrix)':<10}{matrix_run.optimal_energy:>16.8f}"
      f"{matrix_run.optimal_energy - exact_ha:>13.2e}"
      f"{matrix_run.num_operators:>6}{matrix_run.metrics.cnot_count:>7}"
      f"{matrix_run.metrics.depth:>7}{time.perf_counter() - t0:>8.1f}s")

results = {}
for provider in BACKEND_PROVIDERS:
    if not provider_available(provider):
        print(f"{provider:<10}  (SDK not installed -- skipped)")
        continue

    t0 = time.perf_counter()
    driver = QuantumCalculator(
                      method="adapt-vqe",
                      pool=POOL,
                      load_hamiltonian=HAMILTONIAN_FILE,   # no integrals, no map
                      backend_provider=provider,
                      execute_circuits=True,               # really run circuits
                      max_iterations=MAX_ITERATIONS,
                      verbose=False)
    result = driver.run()
    elapsed = time.perf_counter() - t0
    results[provider] = result

    print(f"{provider:<10}{result.optimal_energy:>16.8f}"
          f"{result.optimal_energy - exact_ha:>13.2e}"
          f"{result.num_operators:>6}{result.metrics.cnot_count:>7}"
          f"{result.metrics.depth:>7}{elapsed:>8.1f}s")

# --------------------------------------------------------------------------- #
# 3. Verify behavioral equivalence.
# --------------------------------------------------------------------------- #

print()
for provider, result in results.items():
    # Every backend recovers the FCI ground state of this Hamiltonian ...
    assert abs(result.optimal_energy - exact_ha) < 1e-4, \
        f"{provider} missed FCI by {result.optimal_energy - exact_ha:.2e} Ha"
    # ... grows the same ansatz as the matrix backend.  The *order* of
    # symmetry-degenerate operators can differ: their screening gradients are
    # equal to within the optimizer's own convergence noise, so an arbitrarily
    # small numerical difference decides the tie.  The selected *set* and the
    # energy are what must match.
    assert set(result.operators) == set(matrix_run.operators), \
        f"{provider} selected a different set of operators"
    assert result.num_operators == matrix_run.num_operators, \
        f"{provider} grew a different number of operators"
    # ... and agrees with it far below chemical accuracy.
    assert abs(result.optimal_energy - matrix_run.optimal_energy) < 1e-6, \
        f"{provider} disagrees with the matrix backend"

if results:
    spread = (max(r.optimal_energy for r in results.values())
              - min(r.optimal_energy for r in results.values()))
    print(f"All {len(results)} providers agree: energy spread {spread:.2e} Ha "
          f"(chemical accuracy is 1.6e-3 Ha)")
    print("Same ansatz on every backend: "
          + " -> ".join(matrix_run.operators[:4])
          + (" -> ..." if matrix_run.num_operators > 4 else ""))
