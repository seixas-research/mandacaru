# -*- coding: utf-8 -*-
# file: examples/run_adapt_vqe.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""ADAPT-VQE on H2 with the four operator pools, compared on hardware cost.

Runs ADAPT-VQE (Grimsley et al., 2019) on H2 in the Hartree-Fock molecular-orbital
basis with each of the four operator pools

    fermionic  -- spin-adapted fermionic excitations (Jordan-Wigner)
    qubit      -- individual JW Pauli strings (qubit-ADAPT)
    qeb        -- qubit-excitation generators (Z-strings dropped)
    ceo        -- coupled-exchange operators (shared entangling structure)

and prints a comparative summary of convergence rate, CNOT count and final circuit
depth.  Each ansatz is transpiled to a native {CNOT, U} gate set with Qiskit to
measure its structural cost.

Run with::

    python examples/run_adapt_vqe.py
"""

from __future__ import annotations

import numpy as np

from carcara.algorithms import AdaptVQE
from carcara.core import HydrogenicIntegrals, minimal_hydrogenic_basis
from carcara.integrals import Grid
from carcara.optimizers import Optimizer

POOLS = ["fermionic", "qubit", "qeb", "ceo"]


def build_h2(bond_length: float = 0.74):
    """H2 molecular Hamiltonian in the RHF molecular-orbital basis (4 qubits)."""
    nuclei = [(1.0, np.array([0.0, 0.0, -bond_length / 2])),
              (1.0, np.array([0.0, 0.0, +bond_length / 2]))]
    grid = Grid(center=[0.0, 0.0, 0.0], box_size=6.0, h=0.20)
    integrals = HydrogenicIntegrals(nuclei, minimal_hydrogenic_basis(nuclei), grid)
    H = integrals.molecular_hamiltonian(mo_basis=True, n_electrons=2)
    return H, integrals


def main() -> None:
    H, _ = build_h2()

    # Exact reference: lowest eigenvalue of the qubit Hamiltonian (FCI).
    h_matrix = H.map_to_qubits("jordan_wigner").to_matrix()
    exact = float(np.linalg.eigvalsh(0.5 * (h_matrix + h_matrix.conj().T)).min())
    print(f"H2 (minimal hydrogenic basis, RHF-MO), FCI energy = {exact:+.8f} Ha\n")

    header = (f"{'pool':10s} {'E (Ha)':>13s} {'E-FCI':>11s} {'#ops':>5s} "
              f"{'#params':>7s} {'CNOTs':>6s} {'depth':>6s} {'nfev':>6s}  "
              f"{'converged':>9s}")
    print(header)
    print("-" * len(header))

    summary = {}
    for name in POOLS:
        adapt = AdaptVQE(H, name, num_particles=(1, 1), n_spatial_orbitals=2,
                         optimizer=Optimizer("L-BFGS-B", maxiter=2000))
        res = adapt.run(max_iterations=15, gradient_tol=1e-6)
        summary[name] = res
        print(f"{name:10s} {res.optimal_energy:+13.8f} "
              f"{res.optimal_energy - exact:+11.2e} "
              f"{res.num_operators:5d} {len(res.optimal_parameters):7d} "
              f"{res.metrics.cnot_count:6d} {res.metrics.depth:6d} "
              f"{res.num_evaluations:6d} {str(res.converged):>9s}")

    print("\nOperator selection order (label per ADAPT iteration):")
    for name in POOLS:
        labels = " -> ".join(it.operator_label for it in summary[name].iterations)
        print(f"  {name:10s}: {labels}")

    # Headline benchmark: same ground state, fewer CNOTs for hardware-minded pools.
    ferm = summary["fermionic"].metrics.cnot_count
    print("\nCNOTs to reach FCI, relative to the fermionic pool:")
    for name in POOLS:
        c = summary[name].metrics.cnot_count
        print(f"  {name:10s}: {c:3d} CNOTs  ({c / ferm:.2f}x fermionic)")

    for name in POOLS:
        assert abs(summary[name].optimal_energy - exact) < 1e-6, name
    print("\nAll four pools reached the FCI ground state to < 1e-6 Ha.")


if __name__ == "__main__":
    main()
