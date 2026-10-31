# -*- coding: utf-8 -*-
# file: test/algorithms/test_mean_field.py

# This code is part of Mandacaru.
# MIT License

"""Classical RHF/UHF calculator paths and direct quantum handoff."""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.units import from_hartree
from mandacaru.utils.citations import citation_keys


def h2():
    """Small closed-shell molecule on a fixed real-space box."""
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[6] * 3)


def expectation(operator, state):
    """<state| operator |state> in Hartree for a native Pauli operator."""
    return np.vdot(state, operator.matrix() @ state).real


def test_rhf_is_classical_and_exports_the_same_quantum_reference():
    """The exported Hamiltonian starts ADAPT at the reported RHF energy."""
    atoms = h2()
    atoms.calc = Mandacaru(method="rhf", grid_spacing=0.35, trace=False)
    energy_ev = atoms.get_potential_energy()
    result = atoms.calc.result
    assert result.success
    assert energy_ev == pytest.approx(result.optimal_energy)
    assert result.reference_energy == pytest.approx(energy_ev)
    assert result.num_particles == (1, 1)
    assert result.hamiltonian.fermion_terms()
    assert atoms.calc.solver is None
    assert result.optimal_parameters.size == 0
    keys = citation_keys(method=atoms.calc.method)
    assert "Roothaan1951" in keys
    assert "Kraft1988" not in keys

    for mapping in ("jordan_wigner", "parity", "parity_reduced",
                    "bravyi_kitaev"):
        operator = result.qubit_hamiltonian(mapping)
        state = result.reference_state(mapping)
        assert operator.num_qubits == int(np.log2(state.size))
        assert from_hartree(expectation(operator, state), "eV") == pytest.approx(
            energy_ev, abs=1e-6)

    post_hf = Mandacaru(method="adapt-vqe", trace=False,
                        **result.as_quantum_problem())
    assert post_hf.run().reference_energy == pytest.approx(energy_ev, abs=1e-7)


def test_uhf_closed_shell_exports_its_own_natural_orbital_model():
    """An even-shell UHF calculation passes a basis-consistent model on."""
    atoms = h2()
    atoms.calc = Mandacaru(method="uhf", grid_spacing=0.35, trace=False)
    energy_ev = atoms.get_potential_energy()
    result = atoms.calc.result
    assert result.success
    assert atoms.calc.solver is None
    assert tuple(result.scf.num_particles) == (1, 1)
    assert "PopleNesbet1954" in citation_keys(method=atoms.calc.method)
    post_hf = Mandacaru(method="adapt-vqe", trace=False,
                        **result.as_quantum_problem())
    assert post_hf.run().reference_energy == pytest.approx(
        result.reference_energy, abs=1e-7)
    assert energy_ev <= result.reference_energy + 1e-7
    operator = result.qubit_hamiltonian("parity_reduced")
    state = result.reference_state("parity_reduced")
    assert from_hartree(expectation(operator, state), "eV") == pytest.approx(
        result.reference_energy, abs=1e-6)


def test_uhf_handles_an_open_shell_that_rhf_rejects():
    """The hydrogen doublet is self-interaction-free and cannot use RHF."""
    atom = Atoms("H", positions=[[0, 0, 0]], cell=[6] * 3)
    atom.calc = Mandacaru(method="uhf", grid_spacing=0.35, trace=False)
    energy = atom.get_potential_energy()
    assert np.isfinite(energy)
    assert atom.calc.result.num_particles == (1, 0)
    assert np.linalg.norm(atom.calc.result.scf_state()) == pytest.approx(1.0)
    atom.calc = Mandacaru(method="rhf", grid_spacing=0.35, trace=False)
    with pytest.raises(ValueError, match="invalid SCF"):
        atom.get_potential_energy()


def test_even_high_spin_uhf_export_uses_uhf_orbitals():
    """A triplet's exported Hamiltonian matches its natural-orbital state."""
    atoms = h2()
    atoms.set_initial_magnetic_moments([1, 1])
    atoms.calc = Mandacaru(method="uhf", grid_spacing=0.35, trace=False)
    atoms.get_potential_energy()
    result = atoms.calc.result
    assert result.num_particles == (2, 0)
    operator = result.qubit_hamiltonian()
    state = result.reference_state()
    assert from_hartree(expectation(operator, state), "eV") == pytest.approx(
        result.reference_energy, abs=1e-6)


def test_stretched_uhf_state_is_the_lower_broken_symmetry_solution():
    """The true spin-broken determinant is exported in each qubit encoding."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 2.5]], cell=[8] * 3)
    atoms.calc = Mandacaru(method="rhf", grid_spacing=0.35, trace=False)
    rhf_energy = atoms.get_potential_energy()
    atoms.calc = Mandacaru(method="uhf", grid_spacing=0.35, trace=False)
    uhf_energy = atoms.get_potential_energy()
    result = atoms.calc.result
    assert uhf_energy < rhf_energy - 1.0
    for mapping in ("jordan_wigner", "parity", "parity_reduced",
                    "bravyi_kitaev"):
        operator = result.qubit_hamiltonian(mapping)
        state = result.scf_state(mapping)
        assert np.linalg.norm(state) == pytest.approx(1.0, abs=1e-9)
        assert from_hartree(expectation(operator, state), "eV") == pytest.approx(
            uhf_energy, abs=1e-6)
        # The closed-shell determinant of the exported model lies far above.
        closed = result.hamiltonian.reference_state(mapping)
        assert from_hartree(expectation(operator, closed), "eV") > uhf_energy + 1.0


def test_classical_force_request_is_refused_before_scf():
    """GHF has no native gradient: the force request fails before any SCF."""
    atoms = h2()
    atoms.calc = Mandacaru(method="ghf", grid_spacing=0.35, trace=False)
    with pytest.raises(NotImplementedError, match="not implemented for ghf"):
        atoms.get_forces()
    assert atoms.calc.result is None


@pytest.mark.parametrize("method", ["rhf", "dft"])
def test_all_electron_analytic_forces_are_allowed_under_float32(method):
    """The refusal is for forces that difference the energy; these do not."""
    forces = {}
    for precision in ("float64", "float32"):
        atoms = h2()
        atoms.calc = Mandacaru(method=method, grid_spacing=0.25, precision=precision,
                               resources={"threads": 1})
        forces[precision] = atoms.get_forces()
    scale = max(np.abs(forces["float64"]).max(), 10.0)
    assert np.abs(forces["float32"] - forces["float64"]).max() / scale <= 1e-6


@pytest.mark.parametrize("option", [
    {"shots": 100}, {"load_hamiltonian": "unused.json"}])
def test_classical_options_cannot_silently_request_quantum_work(option):
    """Quantum-only controls are rejected at calculator construction."""
    with pytest.raises(ValueError, match="classical|mean-field calculations"):
        Mandacaru(method="rhf", trace=False, **option)


@pytest.mark.parametrize("method", ["rhf", "uhf", "ghf", "dft", "vqe", "adapt-vqe"])
@pytest.mark.parametrize("option", [
    {"seed": 3}, {"max_length": 5}, {"min_length": 1}, {"temperature": 0.05},
    {"length_penalty": 0.01}, {"warm_start": False}, {"replace_start": "inherit"},
    {"proposal_temperature": 0.5}, {"proposal": "uniform"},
    {"move_weights": {"insert": 1.0}}])
def test_search_controls_are_refused_outside_the_search_methods(method, option):
    """These steer the architecture search of mcas-vqe and valqa only."""
    with pytest.raises(ValueError, match="does not use"):
        Mandacaru(method=method, trace=False, **option)


@pytest.mark.parametrize("method, option", [
    ("rhf", {"xc": "pbe"}), ("uhf", {"xc": "pbe"}), ("ghf", {"xc": "lda"}),
    ("vqe", {"xc": "pbe"}), ("adapt-vqe", {"xc": "pbe"}),
    ("rhf", {"spin": True}), ("ghf", {"unrestricted": True}),
    ("adapt-vqe", {"spin": True}),
    ("rhf", {"kpts": (2, 1, 1)}), ("adapt-vqe", {"smearing": 0.1}),
    ("vqe", {"periodic_scf_options": {"mixing": 0.2}}),
    ("rhf", {"hubbard_u": {"Ni": {"l": "d", "U": 3.0}}}),
    ("adapt-vqe", {"hubbard_u": {"Ni": {"l": "d", "U": 3.0}}}),
    ("rhf", {"pool": "qeb"}), ("dft", {"pool": "ceo"})])
def test_options_of_another_method_are_refused(method, option):
    with pytest.raises(ValueError, match="does not use"):
        Mandacaru(method=method, trace=False, **option)


def test_options_a_method_uses_are_accepted():
    """The refusal is by method: each of these is meaningful where it is given."""
    Mandacaru(method="dft", xc="pbe", spin=True, trace=False)
    Mandacaru(method="uhf", spin=True, trace=False)
    Mandacaru(method="dft", kpts=(2, 2, 1), smearing=0.1, trace=False)
    Mandacaru(method="dft", hubbard_u={"Ni": {"l": "d", "U": 3.0}}, trace=False)
    Mandacaru(method="mcas-vqe", seed=3, max_length=8, temperature=0.05, trace=False)
    Mandacaru(method="vqe", pool="qeb", trace=False)
    # a default value is never "given": naming it changes nothing
    Mandacaru(method="rhf", xc=None, spin=False, pool="fermionic", seed=None, trace=False)


def test_the_hamiltonian_export_controls_stay_available_to_classical_methods():
    """mapping and active_space shape the quantum problem a classical run exports."""
    Mandacaru(method="rhf", mapping="parity", active_space={"orbitals": 2}, trace=False)


def test_ghf_is_rhf_for_a_closed_shell_and_exports_its_spinor_problem():
    """Without spin-orbit coupling GHF finds the RHF determinant; its export
    is written in the spinors, whose reference is that determinant."""
    energies = {}
    for method in ("rhf", "ghf"):
        atoms = h2()
        atoms.calc = Mandacaru(method=method, grid_spacing=0.35, trace=False)
        energies[method] = atoms.get_potential_energy()
    result = atoms.calc.result
    assert energies["ghf"] == pytest.approx(energies["rhf"], abs=1e-7)
    assert result.scf.snapshot()["kramers_pairing"] < 1e-8
    assert "JimenezHoyos2011" in citation_keys(method=atoms.calc.method)
    operator = result.qubit_hamiltonian("jordan_wigner")
    state = result.reference_state("jordan_wigner")
    assert from_hartree(expectation(operator, state), "eV") == pytest.approx(
        energies["ghf"], abs=1e-6)
    with pytest.raises(ValueError, match="alpha parity"):
        result.qubit_hamiltonian("parity_reduced")


@pytest.mark.parametrize("method", ["rhf", "uhf", "ghf"])
def test_scf_writes_the_shared_run_log(method, tmp_path, capsys):
    """``txt=`` holds the shared blocks and a framed ``[SCF SUMMARY]``."""
    from mandacaru.utils.logging import parse_output, reset_log

    path = str(tmp_path / f"{method}.txt")
    atoms = h2()
    atoms.calc = Mandacaru(method=method, grid_spacing=0.35, txt=path)
    energy_ev = atoms.get_potential_energy()
    reset_log(path)
    # The file is the whole report: standard output stays quiet.
    assert "[SYSTEM]" not in capsys.readouterr().out
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    for block in ("[SYSTEM]", "[BASIS]", "[ELECTRONS]", "[SCF SUMMARY]"):
        assert block in text
    # Nothing quantum was built, so nothing quantum is reported.
    for block in ("[OPTIMIZATION SETUP]", "[ITERATIONS]",
                  "[VARIATIONAL QUANTUM SUMMARY]"):
        assert block not in text

    parsed = parse_output(path)
    # The [ELECTRONS] block's thirteen-field layout names the counts this way.
    assert parsed["electrons"]["electrons (alpha, beta)"] == "(1, 1)"
    summary = parsed["summary"]
    assert summary["converged"] == "True"
    assert float(summary["energy_eV"]) == pytest.approx(energy_ev, abs=1e-9)


class TestTheDeterminantDensity:
    """Densities of a mean-field run come from its orbitals: no state vector
    is built (DZP water would need 2^46 amplitudes)."""

    @staticmethod
    def _water(method):
        from ase.build import molecule
        atoms = molecule("H2O")
        atoms.center(vacuum=2.5)
        atoms.calc = Mandacaru(method=method, grid_spacing=0.3, trace=False,
                               basis={"name": "PAW-LCAO", "size": "SZ"})
        atoms.get_potential_energy()
        return atoms

    @pytest.mark.parametrize("method", ["rhf", "dft"])
    def test_a_closed_shell_run_has_integer_natural_occupations(self, method):
        calc = self._water(method).calc
        occupations = calc.natural_orbitals().occupations
        assert np.allclose(occupations, np.round(occupations), atol=1e-8)
        assert occupations.sum() == pytest.approx(8.0)

    def test_a_closed_shell_dft_run_has_no_net_moment(self):
        calc = self._water("dft").calc
        assert calc.get_magnetic_moment() == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize("method", ["rhf", "dft"])
    def test_the_rdm_is_the_determinants(self, method):
        """Idempotent, with the electron count in each spin block."""
        gamma = self._water(method).calc.native_reference.rdm()
        M = gamma.shape[0] // 2
        for block in (gamma[:M, :M], gamma[M:, M:]):
            assert np.allclose(block @ block, block, atol=1e-10)
            assert np.trace(block).real == pytest.approx(4.0)

    def test_an_open_shell_uhf_run_has_its_moment_in_the_spin_blocks(self):
        from ase.build import molecule
        atoms = molecule("O2")
        atoms.center(vacuum=2.5)
        atoms.set_initial_magnetic_moments([1.0, 1.0])
        atoms.calc = Mandacaru(method="uhf", grid_spacing=0.3, trace=False,
                               basis={"name": "PAW-LCAO", "size": "SZ"})
        atoms.get_potential_energy()
        alpha, beta = atoms.calc.result.num_particles
        assert alpha - beta == 2
        # UHF orbitals and the natural-orbital model basis are both in the
        # Loewdin basis: the RDM is idempotent with 7 and 5 electrons.
        gamma = atoms.calc.native_reference.rdm()
        M = gamma.shape[0] // 2
        for block, count in ((gamma[:M, :M], 7.0), (gamma[M:, M:], 5.0)):
            assert np.allclose(block @ block, block, atol=1e-10)
            assert np.trace(block).real == pytest.approx(count)
