"""Echo unitarity, linear response, spectroscopy, and real ADAPT-VQE handoff."""

from pathlib import Path
import runpy

import numpy as np
import pytest
from scipy.linalg import expm

from mandacaru import Mandacaru
from mandacaru.algorithms import QuantumEchoes, quantum_echoes, time_evolve
from mandacaru.core import Fermion, PauliSum, electric_dipole_potential


@pytest.mark.parametrize("order", [1, 2, 4])
def test_unperturbed_echo_returns_generic_state(order):
    psi = np.array([1, 2j, -1j, 3], complex) / np.sqrt(15)
    echo = QuantumEchoes(PauliSum({"YX": 0.8, "ZI": -0.5}),
                         PauliSum({"XX": 0.7}), order=order)
    result = echo.run(psi, 2.3, tau_p=0, steps=4)
    np.testing.assert_allclose(result.state, psi, atol=1e-13)
    assert result.amplitude == pytest.approx(1)
    assert result.fidelity == pytest.approx(1)
    assert result.response == pytest.approx(0, abs=1e-13)


def test_echo_matches_independent_dense_protocol():
    h = PauliSum({"YX": 0.8, "ZI": -0.5, "II": 0.2})
    v = PauliSum({"XX": 0.7, "YI": 0.3, "II": 0.1})
    psi = np.array([1, 2j, -1j, 3], complex) / np.sqrt(15)
    initial = psi.copy()
    time, tau = 0.8, -0.012
    u = expm(-1j*time*h.to_matrix())
    vm = v.to_matrix()
    exact = u.conj().T @ expm(-1j*tau*vm) @ u @ psi
    result = quantum_echoes(psi, h, v, time, tau_p=tau, steps=16, order=4, kick_steps=3)
    np.testing.assert_allclose(result.state, exact, atol=3e-8)
    assert result.correlation == pytest.approx(np.vdot(psi, vm @ u.conj().T @ vm @ u @ psi), abs=3e-8)
    assert result.response == pytest.approx(np.vdot(exact, vm @ exact).real - np.vdot(psi, vm @ psi).real, abs=3e-8)
    np.testing.assert_array_equal(psi, initial)


def test_ground_state_spectroscopy_has_correct_gap_phase_and_response():
    # H = diag(0, gap), V = X. The exact correlation is exp(+i gap*t).
    gap, tau = 1.3, 0.002
    h = PauliSum({"I": gap/2, "Z": -gap/2})
    echo = QuantumEchoes(h, PauliSum({"X": 1.0}))
    for time in [0, 0.3, 1.1, 2.4]:
        result = echo.run([1, 0], time, tau_p=tau)
        assert result.correlation == pytest.approx(np.exp(1j*gap*time))
        assert result.fidelity == pytest.approx(np.cos(tau)**2)
        assert result.response == pytest.approx(np.sin(2*tau)*np.sin(gap*time))
        assert result.response/tau == pytest.approx(2*np.sin(gap*time), abs=6e-6)


def test_tau_bound_depends_on_operator_norm_and_cannot_be_evaded_by_steps():
    echo = QuantumEchoes(PauliSum({"Z": 1}), PauliSum({"X": 1000}))
    with pytest.raises(ValueError, match="pulse is not small"):
        echo.run([1, 0], 0.1, tau_p=1e-3, kick_steps=1000)
    result = echo.run([1, 0], 0.1, tau_p=5e-5)
    assert result.perturbation_bound == pytest.approx(0.05)


@pytest.mark.parametrize("limit", [0, -0.1, 0.11, np.inf, np.nan])
def test_limit_must_enforce_smallness(limit):
    with pytest.raises(ValueError, match="max_perturbation"):
        QuantumEchoes(PauliSum({"Z": 1}), PauliSum({"X": 1}), max_perturbation=limit)


def test_pulse_validation_zero_potential_and_zero_time():
    h = PauliSum({"Z": 1})
    zero = quantum_echoes([1, 0], h, PauliSum(num_qubits=1), 1, tau_p=10)
    np.testing.assert_allclose(zero.state, [1, 0], atol=1e-14)
    assert zero.correlation == zero.response == zero.perturbation_bound == 0
    echo = QuantumEchoes(h, PauliSum({"X": 1}))
    result = echo.run([1, 0], 0, tau_p=0.01)
    np.testing.assert_allclose(result.state, [np.cos(0.01), -1j*np.sin(0.01)])
    for tau in [np.nan, np.inf, 1j]:
        with pytest.raises(ValueError, match="tau_p"):
            echo.run([1, 0], 1, tau_p=tau)
    with pytest.raises(ValueError, match="qubit counts"):
        QuantumEchoes(h, PauliSum({"XX": 1}))
    with pytest.raises(ValueError, match="kick_steps"):
        echo.run([1, 0], 1, kick_steps=0)


@pytest.mark.parametrize("mapping", ["jordan_wigner", "parity", "bravyi_kitaev"])
def test_adapt_ground_state_handoff(mapping, tmp_path):
    # Finite two-site, two-electron model: ADAPT must prepare a rotated state.
    one_body = np.kron(np.eye(2), [[-0.7, -0.2], [-0.2, 0.4]])
    h = Fermion.from_integrals(one_body)
    calc = Mandacaru(method="adapt-vqe", hamiltonian=h, pool="fermionic",
                     num_particles=(1, 1), n_spatial_orbitals=2,
                     mapping=mapping, profile=False, max_iterations=8,
                     gradient_tolerance=1e-7,
                     optimizer={"method": "BFGS", "maxiter": 100, "tol": 1e-10})
    result = calc.run()
    assert result.num_operators > 0
    checkpoint = calc.solver.checkpoint
    psi = checkpoint.state_vector()
    hq = checkpoint.hamiltonian
    assert np.vdot(psi, hq.to_matrix() @ psi).real == pytest.approx(
        np.linalg.eigvalsh(hq.to_matrix())[0], abs=1e-7)
    positions = np.zeros((3, 2, 2))
    positions[2] = np.diag([-0.5, 0.5])
    potential = electric_dipole_potential(positions, [0, 0, 0.1], mapping=mapping)
    echo = QuantumEchoes(hq, potential, order=4)
    expected = echo.run(psi, 0.5, tau_p=0.01, steps=4)
    path = tmp_path / "adapt.json"
    checkpoint.save(path)
    for initial in [checkpoint, path]:
        actual = echo.run(initial, 0.5, tau_p=0.01, steps=4)
        np.testing.assert_allclose(actual.state, expected.state, atol=1e-12)
    propagated = time_evolve(checkpoint, hq, 0.5, steps=4, order=4)
    np.testing.assert_allclose(propagated, expm(-0.5j*hq.to_matrix()) @ psi, atol=2e-9)


def test_three_dimensional_h2_example(tmp_path, monkeypatch):
    """Validate physical geometry, the AO/MO dipole transform, and the echo."""
    example = Path(__file__).resolve().parents[2] / "examples" / "quantum_echoes.py"
    module = runpy.run_path(str(example))
    build = module["build_h2_problem"]
    output = tmp_path / "output.txt"
    atoms, integrals, h, r_mo = build(txt=str(output))
    bond = atoms.positions[1] - atoms.positions[0]
    direction = bond / np.linalg.norm(bond)
    assert atoms.get_chemical_symbols() == ["H", "H"]
    assert not atoms.pbc.any()
    assert np.linalg.norm(bond) == pytest.approx(0.74)
    assert np.all(np.abs(bond) > 0)
    assert all(n > 1 for n in integrals.grid.shape)
    assert np.any(np.abs(integrals.two_body()) > 0.1)  # electron-electron interaction
    assert h is atoms.calc.hamiltonian
    assert r_mo.shape == (3, 2, 2)
    np.testing.assert_allclose(r_mo, r_mo.conj().swapaxes(1, 2), atol=1e-12)
    assert np.all(np.abs(r_mo[:, 0, 1]) > 0.1)  # x, y, z transition moments

    # Independent real-space MO contraction checks units and overlap handling.
    grid = integrals.grid
    ao = np.stack([orb.evaluate(grid.X, grid.Y, grid.Z).ravel()
                   for orb in integrals.basis])
    c = integrals._lowdin_x() @ integrals.mo_coefficients
    mo = c.T @ ao
    np.testing.assert_allclose(mo.conj() @ mo.T * grid.dV, np.eye(2), atol=1e-12)
    expected = np.stack([(mo.conj()*axis.ravel()) @ mo.T * grid.dV
                         for axis in (grid.X, grid.Y, grid.Z)])
    np.testing.assert_allclose(r_mo, expected, atol=1e-12)
    # Inversion symmetry removes permanent dipoles; the cubic diagonal makes
    # the three transition components equal without imposing it in the builder.
    np.testing.assert_allclose(np.diagonal(r_mo, axis1=1, axis2=2), 0, atol=1e-10)
    np.testing.assert_allclose(r_mo[0], r_mo[1], atol=1e-10)
    np.testing.assert_allclose(r_mo[1], r_mo[2], atol=1e-10)

    calc = atoms.calc
    ground = calc.result
    assert ground.converged
    assert ground.metrics.cnot_count is not None
    # The same report as the other ASE examples, with actual built-basis and
    # integration metadata, not a manually injected basis header.
    from mandacaru.utils import parse_output

    report = parse_output(str(output))
    assert report["basis"]["name"] == "HAO"
    assert report["basis"]["family"] == "all-electron"
    assert report["basis"]["basis_functions"] == "2"
    assert report["basis"]["functions"] == [
        {"symbol": "H", "atoms": 2, "functions_per_atom": 1}]
    assert (tmp_path / "references.bib").is_file()
    for section in ("SYSTEM", "BASIS", "ELECTRONS", "OPTIMIZATION SETUP",
                    "ITERATIONS", "VARIATIONAL QUANTUM SUMMARY", "PERFORMANCE"):
        assert f"[{section}]" in output.read_text()
    prepared = calc.solver.checkpoint
    psi = prepared.state_vector()
    hm = prepared.hamiltonian.to_matrix()
    sector = [5, 6, 9, 10]  # one alpha and one beta electron in four qubits
    assert ground.in_units("Ha") == pytest.approx(
        np.linalg.eigvalsh(hm[np.ix_(sector, sector)])[0], abs=1e-8)
    assert ground.num_operators > 0
    v = electric_dipole_potential(r_mo, field=0.01*direction)
    result = QuantumEchoes(prepared.hamiltonian, v).run(
        prepared, time=0.5, tau_p=0.01, steps=40)
    u = expm(-0.5j*hm)
    exact = u.conj().T @ expm(-0.01j*v.to_matrix()) @ u @ psi
    np.testing.assert_allclose(result.state, exact, rtol=0, atol=1e-8)
    assert 0 < result.perturbation_bound < 0.1
    assert abs(result.response) > 1e-9
    # Run the reporting/FFT portion on that same converged calculation.
    main = module["main"]
    monkeypatch.setitem(main.__globals__, "build_h2_problem",
                        lambda txt: (atoms, integrals, h, r_mo))
    main(txt=str(output))
    logged = parse_output(str(output))["quantum_echoes"]
    assert len(logged["samples"]) == 9
    assert len(logged["spectrum"]) == 1024
    assert logged["spectrum_steps_per_sample"] == 10
    energies, vectors = np.linalg.eigh(hm[np.ix_(sector, sector)])
    transition = vectors.conj().T @ v.to_matrix()[np.ix_(sector, sector)] @ vectors[:, 0]
    transition[0] = 0  # remove elastic line
    allowed_gap = energies[np.argmax(np.abs(transition)**2)] - energies[0]
    largest = max(logged["peaks"], key=lambda row: row["magnitude"])
    assert abs(largest["energy_ha"] - allowed_gap) <= logged["spectrum_resolution_ha"]
