# -*- coding: utf-8 -*-
# file: test/algorithms/test_hva.py

# This code is part of Mandacaru.
# MIT License

"""Fixed Hamiltonian layers through the Mandacaru VQE entry point."""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.backends.providers import build_provider, provider_available
from mandacaru.circuits.hva import hamiltonian_groups
from mandacaru.core.mapping import Fermion
from mandacaru.core.sector import ParticleSector
from mandacaru.units import from_hartree


def h2():
    """Two-electron molecule in the common small regression cell."""
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[6] * 3)


def test_two_hamiltonian_layers_lower_the_hf_energy():
    """Two layers approach the exact ground energy in the same particle sector."""
    atoms = h2()
    atoms.calc = Mandacaru(method="hva", layers=2, h=0.35, trace=False)
    energy = atoms.get_potential_energy()
    result = atoms.calc.result
    assert result.success
    assert result.num_parameters == 4
    assert energy < result.reference_energy - 0.1
    sector = ParticleSector(atoms.calc.solver.n_qubits,
                            atoms.calc.solver.num_particles,
                            atoms.calc.solver.mapping)
    exact = np.linalg.eigvalsh(
        sector.restrict(atoms.calc.solver.hamiltonian).toarray()).min()
    assert energy == pytest.approx(from_hartree(exact, "eV"), abs=1e-4)
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
    levels = direct.energy_levels(1)
    assert levels.ground_state_energy == pytest.approx(result.optimal_energy,
                                                      abs=1e-6)


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


def test_spin_resolved_grouping_has_stable_parameter_order():
    """The spin preset names physical groups and keeps their Hamiltonian sum."""
    atoms = h2()
    atoms.calc = Mandacaru(method="rhf", h=0.35, trace=False)
    atoms.get_potential_energy()
    calc = Mandacaru(method="hva", grouping="spin_resolved", layers=1,
                     trace=False, **atoms.calc.result.as_quantum_problem())
    ansatz = calc.solver.ansatz
    assert ansatz.parameter_names == (
        "layer_1:one_body_a", "layer_1:one_body_b",
        "layer_1:two_body_aa", "layer_1:two_body_ab",
        "layer_1:two_body_bb")
    assert ansatz.num_parameters == 5


def test_zero_custom_group_is_rejected():
    """A redundant zero generator cannot consume a variational angle."""
    hamiltonian = Fermion.creation(0) * Fermion.annihilation(0)
    with pytest.raises(ValueError, match="nonzero"):
        hamiltonian_groups(hamiltonian, (
            Fermion({}, n_modes=1), hamiltonian))


def test_hva_refuses_gate_execution_without_a_group_trotter_rule():
    """Circuit execution cannot silently change the fixed group exponentials."""
    with pytest.raises(ValueError, match="evolution='trotter'"):
        Mandacaru(method="hva", execute_circuits=True)
    from mandacaru.backends.providers import QiskitProvider
    with pytest.raises(ValueError, match="evolution='trotter'"):
        Mandacaru(method="hva", measurement_provider=QiskitProvider())


def test_hva_dry_run_does_not_construct_groups():
    """A size estimate stops before mapping or ansatz construction."""
    atoms = h2()
    estimate = Mandacaru(method="hva", h=0.35, trace=False).dry_run(atoms)
    assert estimate.n_qubits == 4


def test_product_formula_matches_provider_circuit():
    """The local HVA objective prepares the circuit's finite-step state."""
    atoms = h2()
    atoms.calc = Mandacaru(method="hva", evolution="trotter", steps=2,
                           h=0.35, trace=False)
    atoms.get_potential_energy()
    solver = atoms.calc.solver
    theta = solver.result.optimal_parameters
    n, occupied, generators, angles, _ = solver.ansatz_problem()
    from mandacaru.backends.providers import QiskitProvider
    circuit_state = QiskitProvider().statevector(n, occupied, generators, angles)
    local_state = solver.ansatz.state(theta)
    assert abs(np.vdot(circuit_state, local_state)) == pytest.approx(1.0,
                                                                     abs=1e-9)


@pytest.mark.slow
@pytest.mark.parametrize("provider_name", ["qiskit", "cirq", "braket"])
def test_product_formula_matches_each_available_provider(provider_name):
    """All provider circuits consume the same tied-angle rotation stream."""
    if not provider_available(provider_name):
        pytest.skip(f"{provider_name} is not installed")
    atoms = h2()
    atoms.calc = Mandacaru(method="rhf", h=0.35, trace=False)
    atoms.get_potential_energy()
    calc = Mandacaru(method="hva", evolution="trotter", order=1,
                     backend_provider=provider_name, execute_circuits=False,
                     trace=False, **atoms.calc.result.as_quantum_problem())
    theta = np.full(calc.solver.ansatz.num_parameters, 0.23)
    local = calc.solver.ansatz.state(theta)
    provider = build_provider(provider_name)
    circuit = provider.statevector(*calc.solver.ansatz_problem(theta)[:4])
    assert abs(np.vdot(local, circuit)) == pytest.approx(1.0, abs=1e-9)


def test_product_formula_converges_to_exact_group_evolution():
    """Increasing Strang steps approaches full-group exponentials."""
    atoms = h2()
    atoms.calc = Mandacaru(method="rhf", h=0.35, trace=False)
    atoms.get_potential_energy()
    problem = atoms.calc.result.as_quantum_problem()
    exact = Mandacaru(method="hva", trace=False, **problem)
    short = Mandacaru(method="hva", evolution="trotter", steps=1,
                      trace=False, **problem)
    long = Mandacaru(method="hva", evolution="trotter", steps=8,
                     trace=False, **problem)
    theta = np.array([0.23, -0.17, 0.31, 0.12])
    target = exact.solver.ansatz.state(theta)
    error_short = 1 - abs(np.vdot(target, short.solver.ansatz.state(theta)))
    error_long = 1 - abs(np.vdot(target, long.solver.ansatz.state(theta)))
    assert error_long < error_short
    assert error_long < 1e-6


def test_product_formula_energy_agrees_across_encodings():
    """A finite-step physical HVA gives the same energy in every mapping."""
    atoms = h2()
    atoms.calc = Mandacaru(method="rhf", h=0.35, trace=False)
    atoms.get_potential_energy()
    problem = atoms.calc.result.as_quantum_problem()
    theta = np.array([0.23, -0.17, 0.31, 0.12])
    energies = []
    for mapping in ("jordan_wigner", "parity", "parity_reduced",
                    "bravyi_kitaev"):
        calc = Mandacaru(method="hva", evolution="trotter", steps=2,
                         mapping=mapping, trace=False, **problem)
        solver = calc.solver
        energies.append(solver.energy(solver.ansatz.state(theta)))
    assert max(energies) - min(energies) < 1e-9


def test_hva_exact_and_product_formula_checkpoints(tmp_path):
    """Checkpoint states match either exact or compiled HVA evolution."""
    for evolution in ("exact", "trotter"):
        path = tmp_path / f"{evolution}.json"
        atoms = h2()
        atoms.calc = Mandacaru(method="hva", evolution=evolution,
                               checkpoint=str(path), h=0.35, trace=False)
        atoms.get_potential_energy()
        solver = atoms.calc.solver
        assert path.is_file()
        state = solver.ansatz.state(solver.result.optimal_parameters)
        saved = solver.checkpoint.state_vector()
        assert abs(np.vdot(saved, state)) == pytest.approx(1.0, abs=1e-9)
        if evolution == "exact":
            with pytest.raises(ValueError, match="exact HVA"):
                solver.checkpoint.problem()
        else:
            assert solver.checkpoint.problem()[0] == solver.n_qubits
        second = h2()
        second.calc = Mandacaru(method="hva", evolution=evolution,
                                resume=str(path), h=0.35, trace=False)
        second.get_potential_energy()
        assert second.calc.result.optimal_energy == pytest.approx(
            solver.result.optimal_energy, abs=1e-7)
        changed = h2()
        changed.calc = Mandacaru(
            method="hva", evolution=evolution, resume=str(path),
            layers=3 if evolution == "exact" else 2,
            steps=2 if evolution == "trotter" else 1,
            h=0.35, trace=False)
        with pytest.raises(ValueError, match="HVA checkpoint differs"):
            changed.get_potential_energy()


def test_hva_checkpoint_record_keeps_caller_labels_and_kinds():
    """Explicit ``labels=`` and ``kinds=`` are stored, not overwritten."""
    atoms = h2()
    atoms.calc = Mandacaru(method="hva", h=0.35, trace=False)
    atoms.get_potential_energy()
    solver = atoms.calc.solver
    parameters = solver.result.optimal_parameters
    default = solver._checkpoint_record(solver.ansatz, parameters, None, {})
    assert set(default.kinds) == {"hva"}
    size = len(default.labels)
    labels = [f"custom_{k}" for k in range(size)]
    kinds = ["custom"] * size
    record = solver._checkpoint_record(solver.ansatz, parameters, None, {},
                                       labels=labels, kinds=kinds)
    assert record.labels == labels
    assert record.kinds == kinds


def test_hva_taper_keeps_all_fixed_groups():
    """Z2 reduction cannot remove a physical HVA group."""
    atoms = h2()
    atoms.calc = Mandacaru(method="hva", taper=True, h=0.35, trace=False)
    tapered_energy = atoms.get_potential_energy()
    solver = atoms.calc.solver
    assert solver._taper_info is not None
    assert solver.ansatz.n_qubits == solver.hamiltonian.num_qubits
    assert not solver._taper_info.dropped
    full = h2()
    full.calc = Mandacaru(method="hva", h=0.35, trace=False)
    assert tapered_energy == pytest.approx(full.get_potential_energy(),
                                           abs=1e-6)


def test_hva_executes_compiled_qiskit_circuit():
    """The explicit circuit mode optimizes its own product-formula state."""
    atoms = h2()
    atoms.calc = Mandacaru(method="hva", evolution="trotter",
                           execute_circuits=True, h=0.35, trace=False)
    energy = atoms.get_potential_energy()
    assert energy < atoms.calc.result.reference_energy - 0.1
    solver = atoms.calc.solver
    state = solver.ansatz.state(solver.result.optimal_parameters)
    assert np.linalg.norm(state) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("reduction", [{}, {"taper": True},
                                        {"mapping": "parity_reduced"}])
def test_hva_force_path_uses_its_optimized_state(reduction):
    """The inherited RDM gradient accepts each supported HVA register."""
    atoms = h2()
    atoms.calc = Mandacaru(method="hva", h=0.35, trace=False, **reduction)
    forces = atoms.get_forces()
    assert forces.shape == (2, 3)
    assert np.all(np.isfinite(forces))


def test_hva_respects_active_orbital_reduction():
    """Group mapping uses the reduced fermionic model and particle counts."""
    atoms = h2()
    atoms.calc = Mandacaru(method="hva", active_orbitals=1,
                           h=0.35, trace=False)
    energy = atoms.get_potential_energy()
    assert np.isfinite(energy)
    assert atoms.calc.n_qubits == 2
    assert atoms.calc.solver.ansatz.n_modes == 2


def test_hva_final_measurement_uses_compiled_product_formula():
    """The calculator measures the optimized circuit through its normal path."""
    from mandacaru.backends.providers import QiskitProvider
    atoms = h2()
    atoms.calc = Mandacaru(
        method="hva", evolution="trotter", h=0.35, trace=False,
        measurement_provider=QiskitProvider(device="statevector"))
    measured = atoms.get_potential_energy()
    assert atoms.calc.measurement_plan is not None
    assert measured == pytest.approx(atoms.calc.result.optimal_energy,
                                     abs=1e-6)


def test_hva_rejects_a_symmetry_leaking_custom_group():
    """Fixed layers cannot discard a group that breaks the tapered sector."""
    n0 = Fermion.creation(0) * Fermion.annihilation(0)
    n1 = Fermion.creation(1) * Fermion.annihilation(1)
    flip = (Fermion.creation(0) * Fermion.annihilation(1)
            + Fermion.creation(1) * Fermion.annihilation(0))
    hamiltonian = n0 + n1
    with pytest.raises(ValueError, match="symmetry-changing group"):
        Mandacaru(method="hva", hamiltonian=hamiltonian,
                  num_particles=(1, 0), hva_groups=(hamiltonian + flip,
                                                       -1.0 * flip),
                  taper=True, trace=False).run()


@pytest.mark.parametrize("mapping", ["jordan_wigner", "parity",
                                     "parity_reduced", "bravyi_kitaev"])
def test_actual_uhf_reference_matches_unrestricted_state(mapping):
    """Spin-resolved Givens preparation reproduces broken-symmetry UHF."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 2.5]], cell=[8] * 3)
    atoms.calc = Mandacaru(method="uhf", h=0.35, trace=False)
    uhf_energy = atoms.get_potential_energy()
    reference = atoms.calc.result
    calc = Mandacaru(method="hva", reference=reference, mapping=mapping,
                     trace=False)
    ansatz = calc.solver.ansatz
    expected = reference.scf_state(mapping)
    assert abs(np.vdot(expected, ansatz.reference_state())) == pytest.approx(
        1.0, abs=1e-9)
    assert from_hartree(calc.solver.reference_energy(), "eV") == pytest.approx(
        uhf_energy, abs=1e-6)


def test_actual_uhf_circuit_preparation_matches_local_state():
    """The same Givens network can be exported to a circuit provider."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 2.5]], cell=[8] * 3)
    atoms.calc = Mandacaru(method="uhf", h=0.35, trace=False)
    atoms.get_potential_energy()
    calc = Mandacaru(method="hva", reference=atoms.calc.result,
                     evolution="trotter", execute_circuits=True,
                     trace=False)
    ansatz = calc.solver.ansatz
    theta = np.zeros(ansatz.num_parameters)
    problem = calc.solver.ansatz_problem(theta)
    from mandacaru.backends.providers import QiskitProvider
    circuit_state = QiskitProvider().statevector(*problem[:4])
    assert abs(np.vdot(circuit_state, ansatz.reference_state())) == \
        pytest.approx(1.0, abs=1e-9)


def test_uhf_reference_tapers_when_its_symmetry_sector_is_valid():
    """A spin-broken determinant can still conserve the available Z2 sector."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 2.5]], cell=[8] * 3)
    atoms.calc = Mandacaru(method="uhf", h=0.35, trace=False)
    energy = atoms.get_potential_energy()
    calc = Mandacaru(method="hva", reference=atoms.calc.result,
                     taper=True, trace=False)
    assert calc.solver._taper_info is not None
    assert from_hartree(calc.solver.reference_energy(), "eV") == \
        pytest.approx(energy, abs=1e-6)
    circuit = Mandacaru(method="hva", reference=atoms.calc.result,
                        taper=True, evolution="trotter",
                        execute_circuits=True, trace=False)
    zero = np.zeros(circuit.solver.ansatz.num_parameters)
    n, occupied, generators, angles, _ = circuit.solver.ansatz_problem(zero)
    from mandacaru.backends.providers import QiskitProvider
    prepared = QiskitProvider().statevector(n, occupied, generators, angles)
    assert abs(np.vdot(prepared,
                       circuit.solver.ansatz.reference_state())) == \
        pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("evolution", ["exact", "trotter"])
def test_actual_uhf_checkpoint_round_trip(evolution, tmp_path):
    """A checkpoint includes the fixed UHF preparation before HVA layers."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 2.5]], cell=[8] * 3)
    atoms.calc = Mandacaru(method="uhf", h=0.35, trace=False)
    atoms.get_potential_energy()
    reference = atoms.calc.result
    path = tmp_path / "hva.json"
    calc = Mandacaru(method="hva", reference=reference,
                     evolution=evolution, checkpoint=str(path), trace=False)
    result = calc.run()
    saved = calc.solver.checkpoint.state_vector()
    actual = calc.solver.ansatz.state(result.optimal_parameters)
    assert abs(np.vdot(saved, actual)) == pytest.approx(1.0, abs=1e-9)
    resumed = Mandacaru(method="hva", reference=reference,
                        evolution=evolution, resume=str(path), trace=False)
    assert resumed.run().optimal_energy == pytest.approx(
        result.optimal_energy, abs=1e-6)
