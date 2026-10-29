"""Independent state/gradient and checkpoint checks for restored algorithms."""

import json
from pathlib import Path

from ase import Atoms
import numpy as np
import pytest
from scipy.linalg import expm

from mandacaru import Mandacaru, _native


@pytest.mark.parametrize("pool", ["fermionic", "qubit", "qeb", "ceo", "ceo-ovp", "spin-orbit"])
def test_pool_rotations_match_independent_dense_exponentials_and_derivatives(pool):
    context = _native.Context(threads=1)
    h = context.pauli_operator(4, [("ZIII", .3, 0.), ("IXII", .17, 0.), ("IIYZ", .11, 0.)])
    problem = context.quantum_problem(h, 2, (1, 1), pool=pool, sector=False)
    indices = list(range(min(3, len(problem.pool_terms))))
    parameters = np.array([-2.3, 1e-9, 4.1])[:len(indices)]
    operators = [context.pauli_operator(4, problem.pool_terms[i]).matrix() for i in indices]
    initial = np.zeros(16, complex)
    initial[sum(1 << (3-q) for q in problem.reference_qubits)] = 1.

    def state(theta):
        result = initial
        for angle, operator in zip(theta, operators):
            result = expm(angle * operator) @ result
        return result

    actual = problem.state(parameters.tolist(), indices=indices, trotter=True)
    assert np.allclose(actual, state(parameters), atol=3e-12)
    value = problem.evaluate(parameters.tolist(), indices=indices, trotter=True)
    for k, gradient in enumerate(value["gradient"]):
        plus, minus = parameters.copy(), parameters.copy()
        plus[k] += 1e-6
        minus[k] -= 1e-6
        a, b = state(plus), state(minus)
        numerical = (np.vdot(a, h.matrix() @ a).real - np.vdot(b, h.matrix() @ b).real) / 2e-6
        assert gradient == pytest.approx(numerical, abs=2e-8)


@pytest.fixture
def reference(tmp_path):
    atoms = Atoms("H2", positions=[[2.5, 2.5, 2.13], [2.5, 2.5, 2.87]], cell=[5] * 3)
    with Mandacaru(
        method="rhf", atoms=atoms, grid_spacing=0.5, resources={"threads": 1}
    ) as calc:
        atoms.get_potential_energy()
        result = calc.result
    path = tmp_path / "hamiltonian.json"
    result.save_hamiltonian(path)
    return result, path


@pytest.mark.parametrize(
    "mode,order,steps", [("exact", 2, 1), ("trotter", 1, 1), ("trotter", 2, 2)]
)
def test_hva_matches_independent_group_exponentials_and_logical_gradients(
    reference, mode, order, steps, tmp_path
):
    result, path = reference
    spec = {
        "name": "hva",
        "layers": 2,
        "evolution": mode,
        "order": order,
        "steps": steps,
    }
    checkpoint = tmp_path / "checkpoint.json"
    with Mandacaru(
        method="vqe", ansatz=spec, load_hamiltonian=path, checkpoint=checkpoint
    ) as calc:
        optimized = calc.run()
        assert optimized.success
        assert optimized.num_parameters == 4
        problem = calc.problem
        parameters = np.array([0.21, -0.13, 0.11, 0.3])
        value = problem.evaluate(parameters, indices=list(range(4)), trotter=True)
        payload = json.loads(path.read_text())
        groups = payload["metadata"]["groups"]
        context = _native.Context(threads=1)
        source = np.zeros(16, complex)
        source[10] = 1.0
        expected = source
        for theta, group in zip(parameters, groups * 2):
            if mode == "exact":
                op = context.pauli_operator(4, [tuple(t) for t in group["terms"]])
                expected = expm(-1j * theta * op.matrix()) @ expected
            else:
                for _ in range(steps):
                    terms = (
                        group["terms"]
                        if order == 1
                        else group["terms"] + group["terms"][::-1]
                    )
                    for term in terms:
                        op = context.pauli_operator(4, [tuple(term)])
                        expected = (
                            expm(-1j * theta / (steps * order) * op.matrix()) @ expected
                        )
        actual = problem.state(parameters, indices=list(range(4)), trotter=True)
        assert np.allclose(actual, expected, atol=2e-12)
        for i in range(4):
            plus = parameters.copy()
            plus[i] += 1e-6
            minus = parameters.copy()
            minus[i] -= 1e-6
            numerical = (
                problem.evaluate(plus, indices=list(range(4)), trotter=True)[
                    "energy_ha"
                ]
                - problem.evaluate(minus, indices=list(range(4)), trotter=True)[
                    "energy_ha"
                ]
            ) / 2e-6
            assert value["gradient"][i] == pytest.approx(numerical, abs=2e-8)
        assert np.allclose(
            calc.solver.checkpoint.state_vector(), optimized.state(), atol=1e-11
        )
        assert np.allclose(
            calc.solver.ansatz.state(optimized.optimal_parameters),
            optimized.state(),
            atol=1e-11,
        )
        assert calc.solver.ansatz.circuit_serializable == (mode == "trotter")
        assert optimized.metrics.num_operators == 4
        assert (optimized.metrics.cnot_count is None) == (mode == "exact")
        energy = optimized.energy_hartree
    with Mandacaru(
        method="vqe", ansatz=spec, load_hamiltonian=path, resume=checkpoint
    ) as calc:
        assert calc.run().energy_hartree == pytest.approx(energy, abs=1e-9)


def test_deflation_reports_bare_levels_and_orthogonal_states(reference):
    result, path = reference
    with Mandacaru(
        method="vqe", load_hamiltonian=path, resources={"threads": 1}
    ) as calc:
        calc.run()
        levels = calc.energy_levels(num_states=2, restarts=4)
        operator = calc.result.qubit_hamiltonian().matrix()
        basis = calc.problem.basis_indices
        exact = np.linalg.eigvalsh(operator[np.ix_(basis, basis)])[:2]
        assert np.allclose(levels.in_units("Ha"), exact, atol=1e-8)
        assert np.allclose(levels.states.conj() @ levels.states.T, np.eye(2), atol=2e-5)
        assert np.allclose(levels.in_units("eV"), levels.energies)
        assert levels.num_evaluations > 0


@pytest.mark.parametrize(
    "method", ["SLSQP", "BFGS", "L-BFGS", "CG", "COBYLA", "Nelder-Mead", "SPSA"]
)
def test_named_optimizers_execute_the_same_quantum_problem(method):
    path = Path(__file__).parents[1] / "migration/reference/h2_hamiltonian.json"
    with Mandacaru(
        method="vqe",
        load_hamiltonian=path,
        optimizer={"method": method, "maxiter": 1000, "tol": 1e-10},
        resources={"threads": 1},
    ) as calc:
        result = calc.run()
        assert result.energy_hartree == pytest.approx(
            -0.9507746213870719, abs=1e-6 if method == "SPSA" else 1e-8
        )
        assert result.num_evaluations > 0
