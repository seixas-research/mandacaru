# -*- coding: utf-8 -*-
# file: test/algorithms/test_mean_field.py

# This code is part of Mandacaru.
# MIT License

"""Classical RHF/UHF calculator paths and direct quantum handoff."""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.core.mapping import Fermion
from mandacaru.units import from_hartree


def h2():
    """Small closed-shell molecule on a fixed real-space box."""
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[6] * 3)


def test_rhf_is_classical_and_exports_the_same_quantum_reference():
    """The exported Hamiltonian starts ADAPT at the reported RHF energy."""
    atoms = h2()
    atoms.calc = Mandacaru(method="rhf", h=0.35, trace=False)
    energy_ev = atoms.get_potential_energy()
    result = atoms.calc.result
    assert result.success
    assert energy_ev == pytest.approx(result.optimal_energy)
    assert result.reference_energy == pytest.approx(energy_ev)
    assert result.num_particles == (1, 1)
    assert isinstance(result.fermion_hamiltonian, Fermion)
    assert atoms.calc.ansatz is None
    assert not hasattr(atoms.calc.solver, "_h_matrix")
    assert result.optimal_parameters.size == 0
    assert "Roothaan1951" in atoms.calc.citation_keys()
    assert "Kraft1988" not in atoms.calc.citation_keys()

    for mapping in ("jordan_wigner", "parity", "parity_reduced",
                    "bravyi_kitaev"):
        operator = result.qubit_hamiltonian(mapping)
        state = result.reference_state(mapping)
        assert operator.num_qubits == int(np.log2(state.size))
        expectation = np.vdot(state, operator.to_sparse_matrix() @ state).real
        assert from_hartree(expectation, "eV") == pytest.approx(
            energy_ev, abs=1e-6)

    post_hf = Mandacaru(method="adapt-vqe", trace=False,
                        **result.as_quantum_problem())
    assert post_hf.run().reference_energy == pytest.approx(energy_ev, abs=1e-7)


def test_uhf_closed_shell_exports_its_own_natural_orbital_model():
    """An even-shell UHF calculation passes a basis-consistent model on."""
    atoms = h2()
    atoms.calc = Mandacaru(method="uhf", h=0.35, trace=False)
    energy_ev = atoms.get_potential_energy()
    result = atoms.calc.result
    assert result.success
    assert atoms.calc.ansatz is None
    assert result.scf.n_alpha == result.scf.n_beta == 1
    assert "PopleNesbet1954" in atoms.calc.citation_keys()
    post_hf = Mandacaru(method="adapt-vqe", trace=False,
                        **result.as_quantum_problem())
    assert post_hf.run().reference_energy == pytest.approx(
        result.reference_energy, abs=1e-7)
    assert energy_ev <= result.reference_energy + 1e-7
    operator = result.qubit_hamiltonian("parity_reduced")
    state = result.reference_state("parity_reduced")
    expectation = np.vdot(state, operator.to_sparse_matrix() @ state).real
    assert from_hartree(expectation, "eV") == pytest.approx(
        result.reference_energy, abs=1e-6)


def test_uhf_handles_an_open_shell_that_rhf_rejects():
    """The hydrogen doublet is self-interaction-free and cannot use RHF."""
    atom = Atoms("H", positions=[[0, 0, 0]], cell=[6] * 3)
    atom.calc = Mandacaru(method="uhf", h=0.35, trace=False)
    energy = atom.get_potential_energy()
    assert np.isfinite(energy)
    assert atom.calc.result.num_particles == (1, 0)
    assert np.linalg.norm(atom.calc.result.scf_state()) == pytest.approx(1.0)
    atom.calc = Mandacaru(method="rhf", h=0.35, trace=False)
    with pytest.raises(ValueError, match="RHF requires"):
        atom.get_potential_energy()


def test_even_high_spin_uhf_export_uses_uhf_orbitals():
    """A triplet's exported Hamiltonian matches its natural-orbital state."""
    atoms = h2()
    atoms.set_initial_magnetic_moments([1, 1])
    atoms.calc = Mandacaru(method="uhf", h=0.35, trace=False)
    atoms.get_potential_energy()
    result = atoms.calc.result
    assert result.num_particles == (2, 0)
    operator = result.qubit_hamiltonian()
    state = result.reference_state()
    expectation = np.vdot(state, operator.to_sparse_matrix() @ state).real
    assert from_hartree(expectation, "eV") == pytest.approx(
        result.reference_energy, abs=1e-6)


def test_stretched_uhf_state_is_the_lower_broken_symmetry_solution():
    """The true spin-broken determinant is exported in each qubit encoding."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 2.5]], cell=[8] * 3)
    atoms.calc = Mandacaru(method="rhf", h=0.35, trace=False)
    rhf_energy = atoms.get_potential_energy()
    atoms.calc = Mandacaru(method="uhf", h=0.35, trace=False)
    uhf_energy = atoms.get_potential_energy()
    result = atoms.calc.result
    assert uhf_energy < rhf_energy - 1.0
    assert result.reference_energy > uhf_energy + 1.0
    for mapping in ("jordan_wigner", "parity", "parity_reduced",
                    "bravyi_kitaev"):
        operator = result.qubit_hamiltonian(mapping)
        state = result.scf_state(mapping)
        assert np.linalg.norm(state) == pytest.approx(1.0, abs=1e-9)
        expectation = np.vdot(state, operator.to_sparse_matrix() @ state).real
        assert from_hartree(expectation, "eV") == pytest.approx(
            uhf_energy, abs=1e-6)


def test_classical_force_request_is_refused_before_scf():
    """A missing orbital-response derivative cannot return a quantum force."""
    atoms = h2()
    atoms.calc = Mandacaru(method="rhf", h=0.35, trace=False)
    with pytest.raises(NotImplementedError, match="orbital response"):
        atoms.get_forces()
    assert atoms.calc.result is None


@pytest.mark.parametrize("option", [
    {"shots": 100}, {"taper": True}, {"active_space": {"orbitals": 2}},
    {"load_hamiltonian": "unused.json"}])
def test_classical_options_cannot_silently_request_quantum_work(option):
    """Quantum-only controls are rejected at calculator construction."""
    with pytest.raises(ValueError, match="classical|full SCF"):
        Mandacaru(method="rhf", trace=False, **option)


def test_ghf_is_rhf_for_a_closed_shell_and_exports_its_spinor_problem():
    """Without spin-orbit coupling GHF finds the RHF determinant; its export
    is written in the spinors, whose reference is that determinant."""
    energies = {}
    for method in ("rhf", "ghf"):
        atoms = h2()
        atoms.calc = Mandacaru(method=method, h=0.35, trace=False)
        energies[method] = atoms.get_potential_energy()
    result = atoms.calc.result
    assert energies["ghf"] == pytest.approx(energies["rhf"], abs=1e-7)
    assert result.scf.kramers_pairing < 1e-8
    assert "JimenezHoyos2011" in atoms.calc.citation_keys()
    operator = result.qubit_hamiltonian("jordan_wigner")
    state = result.reference_state("jordan_wigner")
    expectation = np.vdot(state, operator.to_sparse_matrix() @ state).real
    assert from_hartree(expectation, "eV") == pytest.approx(energies["ghf"],
                                                            abs=1e-6)
    with pytest.raises(ValueError, match="alpha-parity"):
        result.qubit_hamiltonian("parity_reduced")


@pytest.mark.slow
def test_ghf_finds_the_spin_orbit_ground_configuration_of_lead():
    """Pb 6p^2 with a Dirac dataset: GHF lies 85 mHa below the RHF
    determinant in the same spin-orbit Hamiltonian (6p1/2^2), and its
    occupied spinors come out in Kramers pairs without being forced to."""
    from mandacaru.algorithms import GHF

    atom = Atoms("Pb", positions=[[0, 0, 0]], cell=[9.0] * 3)
    atom.center()
    atom.calc = Mandacaru(method="ghf", h=0.25, directory="lda-dirac",
                          basis={"name": "PAW-LCAO", "size": "SZ"},
                          trace=False)
    energy = atom.get_potential_energy()
    result = atom.calc.result
    integrals = atom.calc.solver._gradient_context["integrals"]
    M = integrals.n_orbitals
    h = np.zeros((2 * M, 2 * M), dtype=complex)
    h[:M, :M] = h[M:, M:] = integrals.one_body()
    h += integrals.spin_orbit_matrix()
    rhf = integrals.hartree_fock(14)
    determinant = GHF(h, integrals.two_body(), 14).energy_of(
        GHF.collinear_spinors(rhf.mo_coefficients, rhf.mo_coefficients, 7, 7))
    constant = integrals.constant_energy + integrals.nuclear_repulsion
    below = from_hartree(determinant + constant, "eV") - energy
    assert result.success and below > from_hartree(0.08, "eV")
    assert result.scf.kramers_pairing < 1e-8
