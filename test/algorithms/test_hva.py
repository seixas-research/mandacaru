# -*- coding: utf-8 -*-
# file: test/algorithms/test_hva.py

# This code is part of Mandacaru.
# MIT License

"""Fixed Hamiltonian layers through the Mandacaru VQE entry point."""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.circuits.hva import hamiltonian_groups
from mandacaru.core.mapping import Fermion
from mandacaru.core.sector import ParticleSector


def h2():
    """Two-electron molecule in the common small regression cell."""
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[6] * 3)


def test_two_hamiltonian_layers_lower_the_hf_energy():
    """A second layer can generate real correlation from the HF reference."""
    atoms = h2()
    atoms.calc = Mandacaru(method="hva", layers=2, h=0.35, trace=False)
    energy = atoms.get_potential_energy()
    result = atoms.calc.result
    assert result.success
    assert result.num_parameters == 4
    assert energy < result.reference_energy - 0.1
    assert "Wecker2015" in atoms.calc.citation_keys()
    assert np.linalg.norm(atoms.calc.ansatz.state(result.optimal_parameters)) \
        == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("mapping", ["jordan_wigner", "parity",
                                     "parity_reduced", "bravyi_kitaev"])
def test_hva_preserves_particle_number_under_every_mapping(mapping):
    """Full Hamiltonian groups keep the state in its declared sector."""
    atoms = h2()
    atoms.calc = Mandacaru(method="hva", layers=2, mapping=mapping,
                           h=0.35, trace=False)
    atoms.get_potential_energy()
    solver = atoms.calc.solver
    psi = solver.ansatz.state(solver.result.optimal_parameters)
    sector = ParticleSector(solver.n_qubits, solver.num_particles, mapping)
    assert np.linalg.norm(psi - sector.embed(sector.project(psi))) < 1e-9
    assert solver.result.optimal_energy < solver.result.reference_energy - 0.1


def test_direct_hva_reuses_classical_integrals():
    """The RHF output is a complete direct-mode HVA problem."""
    atoms = h2()
    atoms.calc = Mandacaru(method="rhf", h=0.35, trace=False)
    atoms.get_potential_energy()
    direct = Mandacaru(method="hva", trace=False,
                       **atoms.calc.result.as_quantum_problem())
    result = direct.run()
    assert result.optimal_energy < result.reference_energy - 0.1


def test_custom_groups_must_reconstruct_the_hamiltonian():
    """A missing interaction group cannot masquerade as an HVA model."""
    hamiltonian = (Fermion.creation(0) * Fermion.annihilation(0)
                   + Fermion.creation(1) * Fermion.annihilation(1)
                   + Fermion({((0, True), (1, True),
                               (1, False), (0, False)): 0.5}, n_modes=2))
    groups = hamiltonian_groups(hamiltonian)
    assert len(groups) == 2
    with pytest.raises(ValueError, match="sum to"):
        hamiltonian_groups(hamiltonian, groups[:1])


def test_hva_refuses_gate_execution_without_a_group_trotter_rule():
    """Circuit execution cannot silently change the fixed group exponentials."""
    with pytest.raises(ValueError, match="local exact"):
        Mandacaru(method="hva", execute_circuits=True)


def test_hva_dry_run_does_not_construct_groups():
    """A size estimate stops before mapping or ansatz construction."""
    atoms = h2()
    estimate = Mandacaru(method="hva", h=0.35, trace=False).dry_run(atoms)
    assert estimate.n_qubits == 4
