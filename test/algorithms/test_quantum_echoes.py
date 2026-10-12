"""Echo unitarity, linear response, spectroscopy, and real ADAPT-VQE handoff."""

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
                     convergence={"gradient": 1e-7},
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

