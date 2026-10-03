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


@pytest.mark.parametrize("method", ["rhf", "uhf", "ghf"])
def test_scf_writes_the_shared_run_log(method, tmp_path, capsys):
    """``txt=`` holds the shared blocks, with a classical setup and summary."""
    from mandacaru.utils.logging import parse_output, reset_log

    path = str(tmp_path / f"{method}.txt")
    atoms = h2()
    atoms.calc = Mandacaru(method=method, h=0.35, txt=path)
    energy_ev = atoms.get_potential_energy()
    reset_log(path)
    # The file is the whole report: standard output stays quiet.
    assert "[SYSTEM]" not in capsys.readouterr().out
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    for block in ("[SYSTEM]", "[BASIS]", "[ELECTRONS]", "[SCF SETUP]",
                  "[SCF SUMMARY]", "[PERFORMANCE]"):
        assert block in text
    assert method.upper() in text
    # Nothing quantum was built, so nothing quantum is reported.
    for block in ("[OPTIMIZATION SETUP]", "[ITERATIONS]",
                  "[VARIATIONAL QUANTUM SUMMARY]"):
        assert block not in text

    parsed = parse_output(path)
    for key in ("mapping", "qubits", "Hamiltonian", "Z2 tapering"):
        assert key not in parsed["electrons"]
    assert parsed["electrons"]["electrons (alpha, beta)"] == "(1, 1)"
    assert "Hartree-Fock" in parsed["setup"]["scf_method"]
    summary = parsed["summary"]
    assert summary["converged"] == "True"
    assert float(summary["optimal_energy_eV"]) == pytest.approx(
        energy_ev, abs=1e-9)
    assert int(summary["scf_iterations"]) == \
        atoms.calc.result.num_evaluations


def test_rhf_summary_reports_the_frontier_orbitals(tmp_path):
    from mandacaru.utils.logging import parse_output, reset_log

    path = str(tmp_path / "rhf.txt")
    atoms = h2()
    atoms.calc = Mandacaru(method="rhf", h=0.35, txt=path)
    atoms.get_potential_energy()
    reset_log(path)
    summary = parse_output(path)["summary"]
    scf = atoms.calc.result.scf
    assert float(summary["homo_lumo_gap_eV"]) == pytest.approx(
        from_hartree(scf.homo_lumo_gap, "eV"), abs=1e-6)


class TestTheDeterminantDensity:
    """Populations, densities and dipoles of a mean-field run come from its
    orbitals: no state vector is built (DZP water would need 2^46 amplitudes)."""

    @staticmethod
    def _water(method, **options):
        from ase.build import molecule
        atoms = molecule("H2O")
        atoms.center(vacuum=2.5)
        atoms.calc = Mandacaru(method=method, h=0.3, trace=False,
                               basis={"name": "PAW-LCAO", "size": "SZ"},
                               population="hirshfeld", **options)
        atoms.get_potential_energy()
        return atoms

    @pytest.mark.parametrize("method", ["rhf", "dft"])
    def test_a_closed_shell_run_reports_charges_and_a_dipole(self, method):
        atoms = self._water(method)
        calc = atoms.calc
        charges = calc.results["charges"]
        assert charges.sum() == pytest.approx(0.0, abs=1e-8)
        assert charges[0] < 0.0 < charges[1]                 # O pulls charge
        assert charges[1] == pytest.approx(charges[2], abs=1e-6)
        assert calc.get_total_magnetic_moment() == pytest.approx(0.0, abs=1e-10)
        occupations = calc.natural_orbitals().occupations
        assert np.allclose(occupations, np.round(occupations), atol=1e-8)
        assert occupations.sum() == pytest.approx(8.0)
        dipole = calc.get_dipole_moment()
        assert abs(dipole[2]) > 0.1 and np.allclose(dipole[:2], 0.0, atol=1e-6)

    def test_the_rdm_is_the_determinants(self):
        """Idempotent, with the electron count in each spin block."""
        atoms = self._water("dft")
        gamma = atoms.calc.solver.mean_field_rdm()
        M = gamma.shape[0] // 2
        for block in (gamma[:M, :M], gamma[M:, M:]):
            assert np.allclose(block @ block, block, atol=1e-10)
            assert np.trace(block).real == pytest.approx(4.0)

    def test_an_open_shell_uhf_run_has_its_moment_on_the_atoms(self):
        from ase.build import molecule
        atoms = molecule("O2")
        atoms.center(vacuum=2.5)
        atoms.set_initial_magnetic_moments([1.0, 1.0])
        atoms.calc = Mandacaru(method="uhf", h=0.3, trace=False,
                               basis={"name": "PAW-LCAO", "size": "SZ"},
                               population="hirshfeld")
        atoms.get_potential_energy()
        assert atoms.calc.get_total_magnetic_moment() == pytest.approx(2.0)
        assert np.allclose(atoms.calc.results["magmoms"], 1.0, atol=1e-6)
        # UHF orbitals and the natural-orbital model basis are both in the
        # Loewdin basis: the RDM is idempotent with 7 and 5 electrons.
        gamma = atoms.calc.solver.mean_field_rdm()
        M = gamma.shape[0] // 2
        for block, count in ((gamma[:M, :M], 7.0), (gamma[M:, M:], 5.0)):
            assert np.allclose(block @ block, block, atol=1e-10)
            assert np.trace(block).real == pytest.approx(count)
