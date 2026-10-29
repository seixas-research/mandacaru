"""Numerical termination and checkpoint regressions exposed by the examples."""

import json
from pathlib import Path

import numpy as np
import pytest

from mandacaru import Mandacaru, _native
from mandacaru.core.checkpoint import WavefunctionCheckpoint
from mandacaru.core.mapping import PauliSum


def test_checkpoint_preserves_small_hamiltonian_and_generator_terms(tmp_path):
    h = PauliSum({"II": -0.7, "ZI": 5.123456789012345e-13, "IZ": 3e-14})
    generator = PauliSum({"XY": 0.5j, "YX": -0.5j, "ZZ": 7.2e-13j})
    original = WavefunctionCheckpoint(
        2, [0], [generator], np.array([0.23]), hamiltonian=h
    )
    path = tmp_path / "checkpoint.json"
    original.save(path)
    loaded = WavefunctionCheckpoint.load(path)
    assert loaded.hamiltonian.terms == original.hamiltonian.terms
    assert loaded.generators[0].terms == original.generators[0].terms
    assert np.array_equal(loaded.parameters, original.parameters)
    assert np.allclose(loaded.state_vector(), original.state_vector(), atol=1e-14)


def test_saved_solver_view_can_resume_a_hamiltonian_with_small_terms(tmp_path):
    reference = Path(__file__).parents[1] / "migration/reference/h2_hamiltonian.json"
    payload = json.loads(reference.read_text())
    payload["terms"].extend([["IZZZ", 5e-13, 0.0], ["ZZZI", 3e-14, 0.0]])
    hpath = tmp_path / "hamiltonian.json"
    hpath.write_text(json.dumps(payload))
    checkpoint = tmp_path / "checkpoint.json"
    with Mandacaru(
        method="adapt-vqe", load_hamiltonian=hpath, resources={"threads": 1}
    ) as calc:
        result = calc.run()
        assert result.success
        energy = result.energy_hartree
        calc.solver.checkpoint.save(checkpoint)
    with Mandacaru(
        method="adapt-vqe",
        load_hamiltonian=hpath,
        resume=checkpoint,
        resources={"threads": 1},
    ) as calc:
        resumed = calc.run()
        assert resumed.success
        assert resumed.energy_hartree == pytest.approx(energy, abs=1e-12)


@pytest.mark.parametrize("method", ["adapt-vqe", "subspace-adapt-vqe"])
def test_public_result_retains_inner_failures_and_honors_growth_budget(method):
    reference = Path(__file__).parents[1] / "migration/reference/h2_hamiltonian.json"
    with Mandacaru(
        method=method,
        load_hamiltonian=reference,
        max_iterations=3,
        optimizer={"method": "BFGS", "maxiter": 0},
        resources={"threads": 1},
    ) as calc:
        with pytest.warns(RuntimeWarning, match="inner optimizer did not converge"):
            result = calc.run()
        assert not result.success
        assert result.num_operators == 3
        assert result.final_max_gradient > 1e-7
        assert len(result.optimizer_failures) == 3
        assert all("budget" in message for _, message in result.optimizer_failures)


@pytest.mark.parametrize("mapping", ["jordan_wigner", "parity", "bravyi_kitaev"])
def test_spin_orbit_pool_handles_complex_spin_mixing_and_checkpoint_resume(
    mapping, tmp_path
):
    context = _native.Context(threads=1)
    terms = [
        ([(p, True), (p, False)], energy, 0.0)
        for p, energy in enumerate([-0.7, 0.8, 0.3, 0.9])
    ]
    terms.extend(
        [([(2, True), (0, False)], 0.0, 0.2), ([(0, True), (2, False)], 0.0, -0.2)]
    )
    hamiltonian = context.fermion_operator(4, [1, 0], terms, mapping=mapping)
    options = dict(
        method="adapt-vqe",
        pool="spin-orbit",
        hamiltonian=hamiltonian,
        n_spatial_orbitals=2,
        num_particles=(1, 0),
        mapping=mapping,
        convergence={"gradient": 1e-9},
        resources={"threads": 1},
    )
    checkpoint = tmp_path / "spin-orbit.json"
    with Mandacaru(**options) as calc:
        result = calc.run()
        assert result.success
        assert calc.problem.dimension == 16
        assert result.energy_hartree == pytest.approx(-0.2 - np.sqrt(0.29), abs=1e-11)
        gamma, _ = calc.problem.rdms(result.state(), two_body=False)
        assert np.trace(gamma).real == pytest.approx(1.0, abs=1e-12)
        assert np.trace(gamma[2:, 2:]).real > 0.01
        assert np.allclose(
            calc.solver.checkpoint.state_vector(), result.state(), atol=1e-12
        )
        assert np.allclose(
            calc.solver.ansatz.state(result.optimal_parameters),
            result.state(),
            atol=1e-12,
        )
        calc.solver.checkpoint.save(checkpoint)
    with Mandacaru(**options, resume=checkpoint) as calc:
        assert calc.run().success
    with Mandacaru(**options, sector=True) as calc:
        with pytest.raises(ValueError, match="sector"):
            calc.run()
