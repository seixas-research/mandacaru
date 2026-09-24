"""Product formulas against independent dense exponentials and Qiskit."""

import numpy as np
import pytest
from scipy.linalg import expm

from mandacaru.algorithms import SuzukiTrotter, time_evolve
from mandacaru.core import PauliSum, WavefunctionCheckpoint


@pytest.fixture
def state():
    rng = np.random.default_rng(17)
    psi = rng.normal(size=4) + 1j*rng.normal(size=4)
    return psi / np.linalg.norm(psi)


@pytest.mark.parametrize("label", ["II", "XI", "IX", "YI", "IY", "ZY", "YY", "ZZ"])
@pytest.mark.parametrize("time", [0, 0.83, -0.61])
def test_single_pauli_is_exact(label, time, state):
    h = PauliSum({label: -0.71})
    result = time_evolve(state, h, time, steps=3)
    np.testing.assert_allclose(result, expm(-1j*time*h.to_matrix()) @ state, atol=2e-14)


@pytest.mark.parametrize("order,ratio", [(1, 1.8), (2, 3.8), (4, 15), (6, 55)])
def test_expected_convergence_order(order, ratio, state):
    h = PauliSum({"XI": 0.7, "ZY": -0.9, "IZ": 0.4, "II": 1.2})
    formula = SuzukiTrotter(h, order=order)
    exact = expm(-1j*0.8*h.to_matrix()) @ state
    errors = [np.linalg.norm(formula.evolve(state, 0.8, steps=n) - exact) for n in (2, 4)]
    assert errors[0] / errors[1] > ratio


@pytest.mark.parametrize("order", [1, 2, 4, 6])
def test_adjoint_and_norm(order, state):
    h = PauliSum({"XI": 0.7, "ZY": 0.9, "II": -0.5})
    formula = SuzukiTrotter(h, order=order)
    original = state.copy()
    forward = formula.evolve(state, 2.0, steps=3)
    assert np.linalg.norm(forward) == pytest.approx(1.0, abs=2e-13)
    np.testing.assert_allclose(formula.evolve(forward, 2.0, steps=3, inverse=True),
                               state, atol=3e-13)
    np.testing.assert_array_equal(state, original)


def test_commuting_terms_and_zero_hamiltonian(state):
    h = PauliSum({"XX": 0.3, "YY": 0.8, "ZZ": -0.2})
    np.testing.assert_allclose(time_evolve(state, h, 2),
                               expm(-2j*h.to_matrix()) @ state, atol=2e-14)
    np.testing.assert_array_equal(time_evolve(state, PauliSum(num_qubits=2), 3), state)


def test_checkpoint_input(tmp_path):
    ck = WavefunctionCheckpoint(2, [0], [PauliSum({"XY": 0.4j})], np.array([0.3]))
    path = tmp_path / "state.json"
    ck.save(path)
    h = PauliSum({"YX": 0.7, "ZZ": 0.3})
    expected = expm(-0.4j*h.to_matrix()) @ ck.state_vector()
    for initial in (ck, path, str(path)):
        np.testing.assert_allclose(time_evolve(initial, h, 0.4), expected, atol=1e-14)


@pytest.mark.parametrize("order", [1, 2, 4])
@pytest.mark.parametrize("inverse", [False, True])
def test_exported_circuit_matches_native_state(order, inverse, state):
    from qiskit.quantum_info import Statevector

    formula = SuzukiTrotter(PauliSum({"YX": 0.7, "ZI": -0.4, "II": 0.8}), order=order)
    circuit = formula.circuit(0.6, steps=2, inverse=inverse).decompose(reps=2)
    actual = Statevector(state).evolve(circuit).data
    np.testing.assert_allclose(actual, formula.evolve(state, 0.6, steps=2, inverse=inverse),
                               atol=3e-14)


def test_no_matrix_materialization(monkeypatch, state):
    def forbidden(*args, **kwargs):
        raise AssertionError("matrix materialized")

    monkeypatch.setattr(PauliSum, "to_matrix", forbidden)
    monkeypatch.setattr(PauliSum, "to_sparse_matrix", forbidden)
    time_evolve(state, PauliSum({"YZ": 0.7, "XI": 0.5}), 1.0, steps=3)


@pytest.mark.parametrize("order", [0, -2, 3, 2.5, True])
def test_invalid_orders(order):
    with pytest.raises(ValueError, match="order"):
        SuzukiTrotter(PauliSum({"X": 1}), order=order)


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_invalid_steps(steps):
    with pytest.raises(ValueError, match="steps"):
        time_evolve([1, 0], PauliSum({"X": 1}), 1.0, steps=steps)


@pytest.mark.parametrize("time", [np.nan, np.inf, 1j, [1.0]])
def test_invalid_time(time):
    with pytest.raises(ValueError, match="time"):
        time_evolve([1, 0], PauliSum({"X": 1}), time)


@pytest.mark.parametrize("state", [[0, 0], [2, 0], [np.nan, 0], [np.inf, 0],
                                  [1, 0, 0], [[1, 0], [0, 0]]])
def test_invalid_state(state):
    with pytest.raises(ValueError, match="initial_state"):
        time_evolve(state, PauliSum({"Z": 1}), 1.0)


def test_invalid_operator_and_defensive_copy():
    with pytest.raises(ValueError, match="Hermitian"):
        SuzukiTrotter(PauliSum({"X": 1j}))
    with pytest.raises(TypeError, match="PauliSum"):
        SuzukiTrotter(np.eye(2))
    h = PauliSum({"X": 0.5})
    formula = SuzukiTrotter(h)
    h.terms["X"] = 7
    np.testing.assert_allclose(formula.evolve([1, 0], 1), [np.cos(0.5), -1j*np.sin(0.5)])
