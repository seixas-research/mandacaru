# -*- coding: utf-8 -*-
# file: examples/h2_adapt_ceo_ase.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""H2 ground state with ADAPT-VQE (CEO pool) as an ASE calculator.

End-to-end demonstration wiring three pieces together:

* **ASE integration** -- the molecule is defined once as an ASE :class:`ase.Atoms`
  object and :class:`~carcara.algorithms.ADAPTVQE` is attached to it as an ASE
  *calculator* (``atoms.calc = ADAPTVQE(...)``).  Calling
  ``atoms.get_total_energy()`` builds the Hamiltonian from the current geometry
  (via the supplied ``hamiltonian_builder``) and drives ADAPT-VQE, returning the
  ground-state energy in **eV** (the ASE convention).
* **Hydrogenic basis** -- a minimal one-orbital-per-atom hydrogenic basis
  (:func:`minimal_hydrogenic_basis`) sampled on a real-space grid.
* **ADAPT-VQE with the CEO pool** -- the coupled-exchange-operator pool; on H2 in
  the Hartree-Fock MO basis it reaches the FCI ground state with a single
  operator.  The classical optimizer is the default **COBYLA**.

A structured runtime trace is written to ``examples/data/output.txt`` following
the ADAPT ``output.txt`` protocol, with energies in **eV** and lengths in
**Angstrom** by default.

Run with::

    PYTHONPATH=src python examples/h2_adapt_ceo_ase.py
"""

from __future__ import annotations

import os

import numpy as np
from ase import Atoms

from carcara.algorithms import ADAPTVQE
from carcara.core import HydrogenicIntegrals, minimal_hydrogenic_basis
from carcara.integrals import Grid
from carcara.units import from_hartree
from carcara.utils import parse_output

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")


def nuclei_from_ase(atoms: Atoms):
    """``[(Z, position), ...]`` with the true nuclear charges, read from ASE."""
    return [(float(Z), np.asarray(R, dtype=float))
            for Z, R in zip(atoms.get_atomic_numbers(), atoms.get_positions())]


def h2_hamiltonian_builder(atoms: Atoms):
    """ASE-calculator hook: current geometry -> (Hamiltonian, num_particles, n_orb).

    The RHF molecular-orbital Hamiltonian is rebuilt from the atoms each time the
    geometry changes, so the same calculator works at any bond length.
    """
    nuclei = nuclei_from_ase(atoms)
    grid = Grid(center=[0.0, 0.0, 0.0], box_size=6.0, h=0.20)
    integrals = HydrogenicIntegrals(nuclei, minimal_hydrogenic_basis(nuclei), grid)
    hamiltonian = integrals.molecular_hamiltonian(mo_basis=True, n_electrons=2)
    return hamiltonian, (1, 1), 2


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    output_file = os.path.join(DATA_DIR, "output.txt")

    # 1. Define the molecule via ASE and attach ADAPTVQE as its calculator.
    atoms = Atoms("H2", positions=[[0.0, 0.0, -0.37], [0.0, 0.0, +0.37]])
    atoms.calc = ADAPTVQE(
        pool="ceo",
        hamiltonian_builder=h2_hamiltonian_builder,   # <-- ASE integration
        run_options={"max_iterations": 15, "gradient_tol": 1e-4,
                     "output_file": output_file})

    # 2. Asking ASE for the energy runs the whole ADAPT-VQE simulation.
    energy_ev = atoms.get_total_energy()              # eV (ASE convention)
    result = atoms.calc.adapt_result
    energy_ha = result.optimal_energy

    # Exact reference: lowest eigenvalue of the qubit Hamiltonian (FCI).
    h_matrix = atoms.calc.hamiltonian.to_matrix()
    exact_ha = float(np.linalg.eigvalsh(0.5 * (h_matrix + h_matrix.conj().T)).min())
    exact_ev = float(from_hartree(exact_ha, "eV"))

    print("H2 from ASE Atoms:", atoms.get_chemical_symbols(),
          "@ d = 0.74 Angstrom")
    print("Default classical optimizer:", atoms.calc.optimizer.method)
    print(f"\natoms.get_total_energy() = {energy_ev:+.6f} eV  "
          f"({energy_ha:+.8f} Ha)")
    print(f"FCI reference            = {exact_ev:+.6f} eV  ({exact_ha:+.8f} Ha)")
    print(f"error vs FCI             = {energy_ev - exact_ev:+.2e} eV")

    print("\nADAPT-VQE (CEO pool) final ansatz")
    print("-" * 48)
    print(f"  operators     = {result.num_operators} "
          f"({' -> '.join(result.operators)})")
    print(f"  CNOTs / depth = {result.metrics.cnot_count} / {result.metrics.depth}")
    print(f"  total gates   = {result.metrics.total_gates}")
    print(f"  converged     = {result.converged}")

    assert abs(energy_ha - exact_ha) < 1e-4, "CEO ADAPT missed FCI"
    print("\nReached the FCI ground state to < 1e-4 Ha.")

    # 3. Read back the structured (eV / Angstrom) trace the loop wrote live.
    parsed = parse_output(output_file)
    print(f"\nWrote structured trace -> {output_file}")
    print(f"  units          : {parsed['setup'].get('energy_unit')} / "
          f"{parsed['metadata'].get('units')}")
    print(f"  optimizer      : {parsed['setup'].get('classical_optimizer')}")
    for it in parsed["iterations"]:
        print(f"  iteration {it['index']}: selected {it['selected_operator']}, "
              f"E = {it.get('expressivity_E')}, "
              f"energy = {it['energy']:+.4f} {it['energy_unit']}")


if __name__ == "__main__":
    main()
