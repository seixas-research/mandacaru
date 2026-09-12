# -*- coding: utf-8 -*-
# file: test/test_ibm_quantum.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""IBM Quantum hardware through the Qiskit provider (Qiskit Runtime).

* the device registry runs ``ibm-quantum`` (least busy), any named ``ibm_*``
  processor and the local ``fake_*`` rehearsal backends, all through the
  ``qiskit`` provider, the real ones requiring ``shots > 0``;
* :class:`QiskitProvider` with ``shots > 0`` runs the measured-energy
  protocol -- one measurement circuit per qubit-wise-commuting group,
  bit-strings in Carcará order -- on the local sampler and on a Qiskit
  Runtime fake backend (the real IBM code path: transpiled to the processor,
  executed by ``SamplerV2`` locally), converging to the exact energy;
* the drivers keep the *ansatz* on the internal state-vector backend when
  shots are measured (ADAPT's screening needs amplitudes) while every energy
  goes through the provider, so a whole ADAPT-VQE / VQE run works end to end
  with ``shots > 0``;
* naming real hardware never touches the network here: the service is mocked.
"""

import numpy as np
import pytest

from carcara.algorithms import ADAPTVQE, VQE
from carcara.backends.hardware import (available_devices, device_provider,
                                       get_device, is_fake_device,
                                       is_ibm_device, is_simulator,
                                       normalize_device, require_runnable,
                                       requires_shots)
from carcara.backends.measurement import shot_noise_estimate
from carcara.backends.providers import QiskitProvider, build_provider
from carcara.circuits.adapt_ansatz import AdaptAnsatz
from carcara.circuits.pools import build_pool


def _h2():
    from ase import Atoms
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.center(vacuum=3.0)
    return atoms


@pytest.fixture(scope="module")
def h2_problem():
    """``(hamiltonian, n_qubits, occupied, generators, thetas, exact_energy)``."""
    atoms = _h2()
    calc = ADAPTVQE(pool="ceo", basis="FAO", h=0.4, verbose=False,
                    profile=False, max_iterations=3)
    atoms.calc = calc
    atoms.get_total_energy()
    n = calc.n_qubits
    pool = build_pool("ceo", n // 2, calc.num_particles)
    ansatz = AdaptAnsatz(n, pool.occupied_orbitals, "jordan_wigner")
    for label in calc.result.operators:
        ansatz.append(next(op for op in pool.operators() if op.label == label))
    theta = np.asarray(calc.result.optimal_parameters, dtype=float)
    return (calc.hamiltonian, n, ansatz.reference_qubits(),
            ansatz.pauli_generators, theta, calc.result.optimal_energy)


class TestRegistry:
    def test_ibm_quantum_is_runnable_hardware(self):
        assert "ibm-quantum" in available_devices()
        assert require_runnable("ibm-quantum") == "ibm-quantum"
        assert requires_shots("ibm-quantum") and not is_simulator("ibm-quantum")
        assert device_provider("ibm-quantum") == "qiskit"
        assert is_ibm_device("ibm-quantum") and not is_fake_device("ibm-quantum")

    def test_named_processor_accepted_verbatim(self):
        assert normalize_device("ibm_torino") == "ibm_torino"
        assert normalize_device("IBM_Fez") == "ibm_fez"
        device = get_device("ibm_torino")
        assert device.is_ibm and device.runnable and not device.simulator
        assert device.provider == "qiskit" and device.qubits is None
        assert requires_shots("ibm_torino")

    def test_fake_backend_is_a_local_simulator(self):
        device = get_device("fake_manila")
        assert device.is_fake and device.simulator and device.runnable
        assert device.provider == "qiskit"
        assert not requires_shots("fake_manila")
        assert is_fake_device("fake_manila") and not is_ibm_device("fake_manila")

    def test_unknown_names_still_rejected(self):
        with pytest.raises(ValueError, match="unknown device"):
            normalize_device("quantum-thing")


class TestDriverConfiguration:
    def test_hardware_requires_shots(self):
        with pytest.raises(ValueError, match="shots > 0"):
            ADAPTVQE(pool="ceo", basis="FAO", device="ibm-quantum")
        with pytest.raises(ValueError, match="shots > 0"):
            VQE(basis="FAO", device="ibm_torino", verbose=False)

    def test_ibm_device_configures_the_qiskit_provider(self):
        driver = ADAPTVQE(pool="ceo", basis="FAO", device="ibm_torino",
                          shots=1024, verbose=False,
                          backend_options={"instance": "crn:x",
                                           "optimization_level": 2})
        assert driver.backend_provider == "qiskit"
        assert driver.execute_circuits is True
        options = driver._provider_options()
        assert options == {"instance": "crn:x", "optimization_level": 2,
                           "shots": 1024, "device": "ibm_torino"}
        provider = driver.circuit_provider()
        assert isinstance(provider, QiskitProvider)
        assert provider.is_ibm_device and provider.shots == 1024
        assert provider.optimization_level == 2 and provider.instance == "crn:x"
        # The ansatz never sees a shot-based provider.
        assert driver.ansatz_provider() is None

    def test_exact_qiskit_path_is_unchanged(self):
        driver = ADAPTVQE(pool="ceo", basis="FAO", verbose=False)
        assert driver.shots == 0 and driver.execute_circuits is False
        assert driver._provider_options() == {}
        assert driver.circuit_provider() is None
        exact = ADAPTVQE(pool="ceo", basis="FAO", verbose=False,
                         execute_circuits=True)
        assert exact.circuit_provider() is build_provider("qiskit")   # cached
        assert exact.ansatz_provider() is exact.circuit_provider()

    def test_shots_refused_for_cirq(self):
        with pytest.raises(NotImplementedError, match="qiskit"):
            ADAPTVQE(pool="ceo", basis="FAO", backend_provider="cirq",
                     shots=100, verbose=False)


class TestQiskitProviderShots:
    def test_measurement_circuit_rotates_and_measures(self, h2_problem):
        _h, n, occupied, generators, theta, _e = h2_problem
        provider = QiskitProvider(shots=10)
        qc = provider.build(n, occupied, generators, theta, measure_basis="XYZZ")
        ops = qc.count_ops()
        assert ops.get("measure", 0) == n
        assert ops.get("h", 0) >= 2 and ops.get("sdg", 0) >= 1
        plain = provider.build(n, occupied, generators, theta)
        assert plain.count_ops().get("measure", 0) == 0

    def test_counts_are_in_carcara_order(self):
        # HF reference with Carcará qubit 0 occupied -> bit-string '1000'.
        provider = QiskitProvider(shots=50)
        qc = provider.build(4, [0], [], [], measure_basis="ZZZZ")
        result = provider.sampler().run([qc], shots=50).result()
        assert provider._counts(result[0]) == {"1000": 50}

    def test_statevector_refused_with_shots(self, h2_problem):
        _h, n, occupied, generators, theta, _e = h2_problem
        with pytest.raises(ValueError, match="shot-based"):
            QiskitProvider(shots=100).statevector(n, occupied, generators, theta)

    @pytest.mark.parametrize("device", ["statevector", "fake_manila"])
    def test_measured_energy_converges_to_exact(self, h2_problem, device):
        h, n, occupied, generators, theta, exact = h2_problem
        shots = 20000
        provider = QiskitProvider(device=device, shots=shots)
        groups = provider.measurement_groups(h)
        assert 1 <= len(groups) < len(h.simplify().terms)
        energy = provider.energy(n, occupied, generators, theta, h)
        sigma = shot_noise_estimate(h, shots)
        assert abs(energy - exact) < max(6 * sigma, 0.02)
        if device == "fake_manila":
            assert provider.backend().name == "fake_manila"
            compiled = provider.transpile(provider.build(
                n, occupied, generators, theta, measure_basis="Z" * n))
            assert compiled.count_ops().get("measure", 0) == n

    def test_unknown_fake_backend(self):
        with pytest.raises(ValueError, match="fake backend"):
            QiskitProvider(device="fake_nowhere", shots=1).backend()

    def test_ibm_resolution_uses_the_runtime_service(self, monkeypatch):
        calls = []

        class Service:
            def least_busy(self, **kwargs):
                calls.append(("least_busy", kwargs))
                return "least-busy-backend"

            def backend(self, name):
                calls.append(("backend", name))
                return f"backend:{name}"

        monkeypatch.setattr(QiskitProvider, "_service", lambda self: Service())
        assert QiskitProvider(device="ibm-quantum", shots=1).backend() == \
            "least-busy-backend"
        assert QiskitProvider(device="ibm_torino", shots=1).backend() == \
            "backend:ibm_torino"
        assert calls == [("least_busy", {"operational": True, "simulator": False}),
                         ("backend", "ibm_torino")]


class TestDriversEndToEnd:
    def test_adapt_vqe_with_measured_energies(self, h2_problem):
        exact = h2_problem[-1]
        atoms = _h2()
        atoms.calc = ADAPTVQE(pool="ceo", basis="FAO", h=0.4, verbose=False,
                              profile=False, max_iterations=3,
                              device="AER_simulator", shots=6000,
                              optimizer="COBYLA")
        atoms.get_total_energy()
        result = atoms.calc.result
        assert isinstance(atoms.calc.circuit_provider(), QiskitProvider)
        assert result.num_operators >= 1
        assert abs(result.optimal_energy - exact) < 0.05

    def test_vqe_on_a_fake_ibm_backend(self, h2_problem):
        exact = h2_problem[-1]
        atoms = _h2()
        atoms.calc = VQE(basis="FAO", h=0.4, verbose=False,
                         device="fake_manila", shots=6000, optimizer="COBYLA")
        atoms.get_total_energy()
        provider = atoms.calc.circuit_provider()
        assert provider.is_fake_device and provider.backend().name == "fake_manila"
        assert abs(atoms.calc.result.optimal_energy - exact) < 0.05
