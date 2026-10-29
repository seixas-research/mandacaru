from pathlib import Path
import numpy as np
import pytest
from mandacaru import Mandacaru
from mandacaru.analysis.expressibility import (
    active_space_dimension,
    calculate_kl_divergence,
    sample_pqc_fidelities,
)

REFERENCE = Path(__file__).parents[1] / "migration/reference/h2_hamiltonian.json"


def test_live_growth_observer_can_reenter_its_context_and_sample_native_states():
    seen = []
    with Mandacaru(
        method="adapt-vqe", load_hamiltonian=REFERENCE, resources={"threads": 1}
    ) as calc:

        def callback(info):
            fidelities = sample_pqc_fidelities(
                info["ansatz"], 100, np.random.default_rng(5)
            )
            assert np.all((fidelities >= 0) & (fidelities <= 1))
            assert (
                calculate_kl_divergence(
                    fidelities,
                    calc.n_qubits,
                    10,
                    active_space_dimension(calc.n_qubits, calc.num_particles),
                )
                >= 0
            )
            seen.append(info)

        result = calc.run(callback=callback)
        assert result.success
        assert len(seen) == len(result.snapshot()["iterations"])
        assert seen[-1]["energy"] == pytest.approx(result.optimal_energy, abs=1e-11)
        assert np.allclose(
            seen[-1]["ansatz"].state(seen[-1]["parameters"]), result.state(), atol=1e-12
        )


def test_observer_exceptions_preserve_the_original_exception_and_allow_recovery():
    with Mandacaru(
        method="adapt-vqe", load_hamiltonian=REFERENCE, resources={"threads": 1}
    ) as calc:

        def fail(info):
            raise LookupError("intentional observer failure")

        with pytest.raises(LookupError, match="intentional observer failure"):
            calc.run(callback=fail)
        assert calc.result is None
        assert calc.run().success


def test_haar_bin_probabilities_and_sector_dimensions_have_exact_limits():
    assert active_space_dimension(4, (1, 1)) == 4
    assert active_space_dimension(40, (1, 1)) == 400
    assert active_space_dimension(4) == 16
    uniform = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
    assert calculate_kl_divergence(uniform, 1, 2, 2) == pytest.approx(0.0, abs=1e-14)
    assert calculate_kl_divergence(np.zeros(10), 1, 2, 2) == pytest.approx(
        np.log(2.0), abs=1e-14
    )
    assert calculate_kl_divergence(np.zeros(10), 2, 2, 4) == pytest.approx(
        -np.log(0.875), abs=1e-14
    )
    assert calculate_kl_divergence(np.ones(10), 2, 2, 1) == 0
