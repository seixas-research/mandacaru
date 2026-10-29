"""SDK execution must supply the objective used by the native optimizer."""
import numpy as np
import pytest

from mandacaru import Mandacaru, _native


@pytest.mark.parametrize("provider", ["qiskit", "braket", "cirq"])
def test_sdk_objective_and_parameter_shift_match_native_analytic_gradient(provider, monkeypatch):
    context = _native.Context(threads=1)
    operator = context.pauli_operator(4, [("ZIII", .3, 0.),
                                         ("XXII", .2, 0.), ("YYII", .2, 0.)])
    with Mandacaru(method="vqe", hamiltonian=operator, n_spatial_orbitals=2,
                   num_particles=(1, 0), backend_provider=provider,
                   execute_circuits=True, atomic_units=True,
                   optimizer={"maxiter": 0}, resources={"threads": 1}) as calc:
        calls = []
        original = calc.provider.statevector

        def statevector(*args):
            calls.append(tuple(args[-1]))
            return original(*args)

        monkeypatch.setattr(calc.provider, "statevector", statevector)
        result = calc.run(initial_parameters=[.21])
        value = calc.problem.evaluate([.21], trotter=True)
        assert result.energy_hartree == pytest.approx(value["energy_ha"], abs=1e-12)
        assert result.final_max_gradient == pytest.approx(abs(value["gradient"][0]), abs=2e-12)
        assert len(set(calls)) >= 3  # the objective and both parameter-shift circuits
        assert result._raw["execution"] == provider


def test_shot_gradients_are_measured_commutators_and_budget_precedes_submission(monkeypatch):
    context = _native.Context(threads=1)
    operator = context.pauli_operator(4, [("ZIII", .3, 0.),
                                         ("XXII", .2, 0.), ("YYII", .2, 0.)])
    with Mandacaru(method="adapt-vqe", hamiltonian=operator, n_spatial_orbitals=2,
                   num_particles=(1, 0), backend_provider="qiskit", shots=4096,
                   optimizer={"method": "COBYLA", "maxiter": 4},
                   max_iterations=1, measurement_budget={"observables": 0}) as calc:
        monkeypatch.setattr(calc.provider, "expectation_values",
                            lambda *args: pytest.fail("budget must precede SDK execution"))
        with pytest.raises(ValueError, match="measurement observables budget"):
            calc.run()


def test_native_solver_uses_sdk_measurements_instead_of_replacing_them():
    context = _native.Context(threads=1)
    problem = context.quantum_problem(
        context.pauli_operator(4, [("ZIII", .3, 0.), ("XXII", .2, 0.), ("YYII", .2, 0.)]),
        2, (1, 0), sector=True)
    calls = []

    def measured(labels, angles, observables):
        calls.append((labels, angles, observables))
        # A deliberately different device state distinguishes SDK and native objectives.
        return {"expectations": [1. if set(label) <= {"I", "Z"} else 0.
                                 for label in observables]}

    raw = problem.solve(trotter=True, optimizer="COBYLA", max_iterations=0,
                        initial_parameters=[.21], executor=measured)
    assert raw["energy_ha"] == pytest.approx(.3)
    assert raw["energy_ha"] != pytest.approx(problem.evaluate([.21])["energy_ha"])
    assert raw["final_max_gradient"] == pytest.approx(0.)
    assert calls


def test_sdk_error_retains_its_type_and_releases_context():
    context = _native.Context(threads=1)
    problem = context.quantum_problem(
        context.pauli_operator(4, [("ZIII", .3, 0.)]), 2, (1, 0), sector=True)

    def failed(*args):
        raise LookupError("SDK transport failed")

    with pytest.raises(LookupError, match="SDK transport failed"):
        problem.solve(trotter=True, executor=failed)
    assert np.isfinite(problem.solve()["energy_ha"])


def test_measured_adapt_pool_screening_matches_independent_commutator_expectations():
    from scipy.linalg import expm

    context = _native.Context(threads=1)
    problem = context.quantum_problem(
        context.pauli_operator(4, [("ZIII", .3, 0.), ("XXII", .2, 0.), ("YYII", .2, 0.)]),
        2, (1, 0), sector=True)
    single = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
              "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1., -1.])}
    matrices = {}

    def matrix(label):
        if label not in matrices:
            value = np.ones((1, 1))
            for letter in label:
                value = np.kron(value, single[letter])
            matrices[label] = value
        return matrices[label]

    requests = []

    def oracle(labels, angles, observables):
        requests.append(tuple(observables))
        state = np.zeros(16, complex)
        state[8] = 1.
        for label, angle in zip(labels, angles):
            state = expm(1j * angle * matrix(label)) @ state
        return {"expectations": [float(np.vdot(state, matrix(label) @ state).real)
                                 for label in observables]}

    raw = problem.solve(adapt=True, trotter=True, optimizer="COBYLA", max_iterations=12,
                        max_growth=1, executor=oracle)
    exact = problem.evaluate(raw["parameters"].tolist(), indices=raw["indices"], trotter=True)
    assert raw["energy_ha"] == pytest.approx(exact["energy_ha"], abs=1e-12)
    commutator_labels = {label for op in problem.pool_observables for label, _, _ in op
                         if label.strip("I")}
    assert any(set(request) == commutator_labels for request in requests)
    state = problem.expand_state(raw["state"])
    gradients = []
    for observable in problem.pool_observables:
        operator = sum((real + 1j * imag) * matrix(label) for label, real, imag in observable)
        gradients.append(abs(np.vdot(state, operator @ state)))
    assert raw["final_max_gradient"] == pytest.approx(max(gradients), abs=1e-12)
