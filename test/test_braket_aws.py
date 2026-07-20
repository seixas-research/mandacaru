# -*- coding: utf-8 -*-
# file: test/test_braket_aws.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Amazon Braket compatibility: the shot-based, QPU-runnable execution path.

A real QPU never returns a state vector -- it returns *samples*.  These tests pin
the consequences:

* Braket **rejects** the ``StateVector`` result type whenever ``shots > 0``, so
  the exact path is simulator-only and the driver must say so clearly;
* the shot path assembles ``<H>`` from qubit-wise-commuting measurement groups
  and converges to the exact energy as ``1/sqrt(shots)``;
* naming a Braket device configures the provider (and an AWS device carries its
  ARN), while a QPU without ``shots`` is refused up front rather than at
  submission time.

Everything runs on Braket's **local** simulator, which is the same code path a
QPU takes (shots in, counts out) -- only the device differs.  Nothing here
contacts AWS or costs money.
"""

from __future__ import annotations

import numpy as np
import pytest

from carcara.algorithms import ADAPTVQE, VQE
from carcara.backends.hardware import (available_devices, device_arn,
                                       device_provider, get_device,
                                       is_aws_device, is_simulator,
                                       normalize_device, require_runnable,
                                       requires_shots)
from carcara.backends.measurement import (energy_from_group_counts,
                                          is_qubit_wise_commuting, merge_basis,
                                          pauli_expectation_from_counts,
                                          qubit_wise_commuting_groups,
                                          shot_noise_estimate)
from carcara.backends.providers import build_provider, provider_available
from carcara.circuits.adapt_ansatz import AdaptAnsatz
from carcara.circuits.pools import build_pool
from carcara.core import PauliSum

braket_only = pytest.mark.skipif(not provider_available("braket"),
                                 reason="amazon-braket-sdk not installed")


@pytest.fixture(scope="module")
def h2_problem():
    """A small H2 problem: ``(hamiltonian, n_qubits, occupied, generators)``."""
    from ase import Atoms
    atoms = Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
                  cell=[[6, 0, 0], [0, 6, 0], [0, 0, 6]], pbc=True)
    calc = ADAPTVQE(pool="qeb", basis="FAO", h=0.4, verbose=False,
                    max_iterations=1)
    atoms.calc = calc
    atoms.get_total_energy()

    n = calc.n_qubits
    pool = build_pool("qeb", n // 2, calc.num_particles)
    ansatz = AdaptAnsatz(n, pool.occupied_orbitals, "jordan_wigner")
    for op in pool.operators()[:2]:
        ansatz.append(op)
    return (calc.hamiltonian, n, ansatz.reference_qubits(),
            ansatz.pauli_generators, ansatz)


# --------------------------------------------------------------------------- #
# Qubit-wise-commuting grouping (provider independent).
# --------------------------------------------------------------------------- #

class TestQubitWiseCommuting:
    @pytest.mark.parametrize("a,b,expected", [
        ("ZZ", "ZI", True),        # same letter where both act
        ("XI", "IZ", True),        # disjoint support
        ("XX", "ZZ", False),       # different letters on the same qubits
        ("XZ", "XX", False),
        ("II", "XY", True),
    ])
    def test_pairwise_rule(self, a, b, expected):
        assert is_qubit_wise_commuting(a, b) is expected

    def test_merge_basis(self):
        assert merge_basis(["ZII", "IXI", "IIY"]) == "ZXY"
        assert merge_basis(["ZZI", "ZII"]) == "ZZI"
        with pytest.raises(ValueError, match="not qubit-wise commuting"):
            merge_basis(["XI", "ZI"])

    def test_grouping_covers_every_term_exactly_once(self):
        pauli = PauliSum({"II": -1.0, "ZI": 0.3, "IZ": -0.2, "ZZ": 0.1,
                          "XX": 0.05, "YY": -0.05, "XI": 0.02})
        groups, identity = qubit_wise_commuting_groups(pauli)
        assert identity == pytest.approx(-1.0)
        seen = [label for _basis, payload in groups for label, _c in payload]
        assert sorted(seen) == sorted(l for l in pauli.terms if set(l) != {"I"})
        # Every group really is simultaneously measurable.
        for basis, payload in groups:
            assert merge_basis([label for label, _ in payload]) == basis

    def test_grouping_reduces_the_circuit_count(self, h2_problem):
        hamiltonian = h2_problem[0]
        n_terms = len([l for l in hamiltonian.simplify().terms
                       if set(l) != {"I"}])
        groups, _ = qubit_wise_commuting_groups(hamiltonian)
        # The point of grouping: far fewer circuits than Pauli terms.
        assert len(groups) < n_terms

    def test_expectation_from_counts(self):
        # |00> in the Z basis: <ZI> = <IZ> = <ZZ> = +1.
        counts = {"00": 1000}
        assert pauli_expectation_from_counts("ZI", counts) == pytest.approx(1.0)
        assert pauli_expectation_from_counts("ZZ", counts) == pytest.approx(1.0)
        # |01>: <IZ> = -1, <ZZ> = -1, identity is always +1.
        counts = {"01": 1000}
        assert pauli_expectation_from_counts("IZ", counts) == pytest.approx(-1.0)
        assert pauli_expectation_from_counts("ZZ", counts) == pytest.approx(-1.0)
        assert pauli_expectation_from_counts("II", counts) == pytest.approx(1.0)

    def test_energy_from_counts_reproduces_a_known_state(self):
        """|01> has a exactly computable energy for a diagonal Hamiltonian."""
        pauli = PauliSum({"II": 0.5, "ZI": 0.25, "IZ": -0.125})
        groups, identity = qubit_wise_commuting_groups(pauli)
        counts = [{"01": 4096} for _ in groups]
        # <H> = 0.5 + 0.25*(+1) + (-0.125)*(-1) = 0.875
        assert energy_from_group_counts(groups, identity, counts) == \
            pytest.approx(0.875)

    def test_shot_noise_estimate_scales_as_inverse_sqrt(self):
        pauli = PauliSum({"II": -1.0, "ZI": 0.3, "XX": 0.4})
        assert shot_noise_estimate(pauli, 100) == pytest.approx(
            2.0 * shot_noise_estimate(pauli, 400), rel=1e-9)


# --------------------------------------------------------------------------- #
# Device registry: AWS Braket entries.
# --------------------------------------------------------------------------- #

class TestDeviceRegistry:
    def test_braket_devices_are_registered(self):
        for name in ("braket-local", "braket-sv1", "braket-ionq-aria",
                     "braket-iqm-garnet", "braket-rigetti-ankaa"):
            assert name in available_devices()
            assert device_provider(name) == "braket"

    def test_managed_simulators_and_qpus_are_classified(self):
        assert is_simulator("braket-sv1") is True
        assert is_simulator("braket-ionq-aria") is False
        assert is_aws_device("braket-sv1") is True
        assert is_aws_device("braket-local") is False
        # Every AWS device carries the ARN the SDK needs.
        assert device_arn("braket-sv1").startswith("arn:aws:braket")
        assert device_arn("braket-local") is None

    def test_qpus_require_shots(self):
        assert requires_shots("braket-ionq-aria") is True
        assert requires_shots("braket-sv1") is False
        assert requires_shots("AER_simulator") is False

    def test_raw_arn_is_accepted(self):
        arn = "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1"
        assert normalize_device(arn) == "braket-ionq-aria"
        # An ARN this build does not know by name still resolves, as a QPU.
        future = "arn:aws:braket:eu-west-2::device/qpu/vendor/Future-9"
        assert normalize_device(future) == future
        device = get_device(future)
        assert device.is_aws and not device.simulator
        assert requires_shots(future) is True

    def test_braket_devices_are_runnable(self):
        for name in ("braket-local", "braket-sv1", "braket-ionq-aria"):
            assert require_runnable(name) == normalize_device(name)
        # ibm-quantum is still only a reserved label.
        with pytest.raises(NotImplementedError):
            require_runnable("ibm-quantum")

    def test_unknown_device_rejected(self):
        with pytest.raises(ValueError, match="unknown device"):
            normalize_device("some-quantum-thing")


# --------------------------------------------------------------------------- #
# Provider: AWS targeting and the shot-based energy.
# --------------------------------------------------------------------------- #

@braket_only
class TestBraketProvider:
    def test_local_vs_aws_targeting(self):
        local = build_provider("braket", device="braket_sv")
        assert local.is_aws_device is False
        aws = build_provider("braket",
                             device=device_arn("braket-ionq-aria"), shots=100)
        assert aws.is_aws_device is True
        # Constructing the provider must not contact AWS -- only device() does.
        assert aws._device is None

    def test_options_are_not_cached_across_instances(self):
        a = build_provider("braket", shots=100)
        b = build_provider("braket", shots=200)
        assert a is not b and a.shots == 100 and b.shots == 200
        # Default-configured providers are still shared.
        assert build_provider("braket") is build_provider("braket")

    def test_statevector_is_refused_when_shots_are_set(self, h2_problem):
        """This is the incompatibility that rules the exact path out on a QPU."""
        _h, n, occupied, generators, _ansatz = h2_problem
        provider = build_provider("braket", shots=512)
        with pytest.raises(ValueError, match="shots > 0"):
            provider.statevector(n, occupied, generators, np.zeros(len(generators)))

    def test_braket_itself_rejects_statevector_with_shots(self):
        """The upstream constraint the message above reports, pinned directly."""
        from braket.circuits import Circuit
        from braket.devices import LocalSimulator
        circuit = Circuit().h(0).cnot(0, 1)
        circuit.state_vector()
        with pytest.raises(ValueError, match="StateVector"):
            LocalSimulator().run(circuit, shots=100)

    def test_measurement_circuit_is_a_native_gate_set(self, h2_problem):
        """Every gate emitted must be one a Braket QPU can accept."""
        _h, n, occupied, generators, _ansatz = h2_problem
        provider = build_provider("braket", shots=100)
        circuit = provider.build(n, occupied, generators,
                                 np.full(len(generators), 0.3),
                                 measure_basis="X" * n)
        allowed = {"I", "X", "H", "S", "Si", "Rz", "CNot"}
        used = {ins.operator.name for ins in circuit.instructions}
        assert used <= allowed, f"unsupported gates: {used - allowed}"

    def test_shot_energy_converges_to_the_exact_value(self, h2_problem):
        hamiltonian, n, occupied, generators, ansatz = h2_problem
        theta = np.array([0.31, -0.2][:len(generators)])
        exact = float(np.real(np.vdot(ansatz.state(theta),
                                      hamiltonian.to_matrix()
                                      @ ansatz.state(theta))))

        # Sampling is random, so compare *averaged* errors over repeats rather
        # than a single draw, and span a wide enough shot range (200 -> 50000,
        # i.e. 250x, so ~16x less noise) that the 1/sqrt(shots) trend dominates
        # the fluctuation between repeats.
        def mean_error(shots, repeats=3):
            errors = []
            for _ in range(repeats):
                provider = build_provider("braket", shots=shots)
                measured = provider.energy(n, occupied, generators, theta,
                                           hamiltonian)
                errors.append(abs(measured - exact))
            return float(np.mean(errors))

        coarse, fine = mean_error(200), mean_error(50000)
        assert fine < coarse, f"error did not shrink: {coarse:.2e} -> {fine:.2e}"
        # And the fine estimate must sit inside the analytic shot-noise bound.
        assert fine < 5 * shot_noise_estimate(hamiltonian, 50000)

    def test_shots_zero_is_exact(self, h2_problem):
        hamiltonian, n, occupied, generators, ansatz = h2_problem
        theta = np.array([0.31, -0.2][:len(generators)])
        exact = float(np.real(np.vdot(ansatz.state(theta),
                                      hamiltonian.to_matrix()
                                      @ ansatz.state(theta))))
        provider = build_provider("braket", shots=0)
        assert provider.energy(n, occupied, generators, theta,
                               hamiltonian) == pytest.approx(exact, abs=1e-9)

    def test_measurement_groups_are_exposed_for_cost_planning(self, h2_problem):
        hamiltonian = h2_problem[0]
        provider = build_provider("braket", shots=100)
        groups = provider.measurement_groups(hamiltonian)
        # One quantum task per group per energy evaluation -- what a QPU bills.
        assert 0 < len(groups) < len(hamiltonian.simplify().terms)


# --------------------------------------------------------------------------- #
# Driver wiring.
# --------------------------------------------------------------------------- #

@braket_only
class TestDriverOnBraket:
    def test_naming_a_braket_device_selects_the_provider(self):
        driver = ADAPTVQE(device="braket-local")
        assert driver.backend_provider == "braket"
        assert driver.execute_circuits is True

    def test_aws_device_is_passed_to_the_provider_by_arn(self):
        driver = ADAPTVQE(device="braket-sv1")
        options = driver._provider_options()
        assert options["device"] == device_arn("braket-sv1")

    def test_qpu_without_shots_is_refused_up_front(self):
        with pytest.raises(ValueError, match="shots > 0"):
            ADAPTVQE(device="braket-ionq-aria")
        # ... and accepted with shots.
        driver = ADAPTVQE(device="braket-ionq-aria", shots=1000)
        assert driver.shots == 1000
        assert driver._provider_options()["shots"] == 1000

    def test_shots_require_the_braket_provider(self):
        with pytest.raises(NotImplementedError, match="braket"):
            ADAPTVQE(backend_provider="cirq", shots=100)

    def test_shots_imply_circuit_execution(self):
        driver = ADAPTVQE(backend_provider="braket", shots=256)
        assert driver.execute_circuits is True

    def test_vqe_energy_at_uses_the_shot_path(self, h2_problem):
        """The whole driver -- not just the provider -- runs on measurements."""
        from carcara.circuits import UCCSD
        hamiltonian, n, _occupied, _generators, _ansatz = h2_problem
        num_particles = (n // 4 or 1, n // 4 or 1)

        exact = VQE(hamiltonian, UCCSD(n // 2, num_particles), verbose=False)
        measured = VQE(hamiltonian,
                       UCCSD(n // 2, num_particles, trotter=True,
                             provider=build_provider("braket", shots=8000)),
                       verbose=False, backend_provider="braket", shots=8000)
        theta = np.zeros(exact.ansatz.num_parameters)

        # The measured energy is a statistical estimate: the off-diagonal Pauli
        # terms average to zero on a determinant but still fluctuate shot to
        # shot, so agreement is asserted within the shot-noise bound.
        assert measured.energy_at(theta) == pytest.approx(
            exact.energy_at(theta),
            abs=5 * shot_noise_estimate(hamiltonian, 8000))
