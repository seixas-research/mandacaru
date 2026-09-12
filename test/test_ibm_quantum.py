# -*- coding: utf-8 -*-
# file: test/test_ibm_quantum.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""IBM Quantum hardware through the Qiskit provider and the Estimator.

Everything here runs locally: the exact local estimator, a Qiskit Runtime
fake backend (the real transpile-to-processor path), and a mocked Runtime
service for the device selection.  No QPU time is ever used by the tests.
"""

import numpy as np
import pytest

from carcara.algorithms import ADAPTVQE, VQE
from carcara.algorithms.base import measure_energies
from carcara.backends.hardware import (available_devices, device_provider,
                                       get_device, is_fake_device,
                                       is_ibm_device, is_simulator,
                                       normalize_device, require_runnable,
                                       requires_shots)
from carcara.backends.providers import QiskitProvider, build_provider


def _h2():
    from ase import Atoms
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.center(vacuum=3.0)
    return atoms


@pytest.fixture(scope="module")
def h2_run():
    """An exact local ADAPT-VQE run on H2 (the driver keeps its ansatz)."""
    atoms = _h2()
    calc = ADAPTVQE(pool="ceo", basis="FAO", h=0.4, verbose=False,
                    profile=False, max_iterations=3)
    atoms.calc = calc
    atoms.get_total_energy()
    return calc


class TestRegistry:
    def test_ibm_quantum_is_runnable_hardware(self):
        assert "ibm-quantum" in available_devices()
        assert require_runnable("ibm-quantum") == "ibm-quantum"
        assert requires_shots("ibm-quantum") and not is_simulator("ibm-quantum")
        assert device_provider("ibm-quantum") == "qiskit"
        assert is_ibm_device("ibm-quantum") and not is_fake_device("ibm-quantum")

    def test_named_processor_accepted_verbatim(self):
        assert normalize_device("ibm_kingston") == "ibm_kingston"
        assert normalize_device("IBM_Fez") == "ibm_fez"
        device = get_device("ibm_kingston")
        assert device.is_ibm and device.runnable and not device.simulator
        assert device.provider == "qiskit" and requires_shots("ibm_kingston")

    def test_fake_backend_is_a_local_simulator(self):
        device = get_device("fake_manila")
        assert device.is_fake and device.simulator and device.runnable
        assert device.provider == "qiskit" and not requires_shots("fake_manila")

    def test_unknown_names_still_rejected(self):
        with pytest.raises(ValueError, match="unknown device"):
            normalize_device("quantum-thing")


class TestDriverConfiguration:
    def test_hardware_requires_shots(self):
        with pytest.raises(ValueError, match="shots > 0"):
            ADAPTVQE(pool="ceo", basis="FAO", device="ibm-quantum")
        with pytest.raises(ValueError, match="shots > 0"):
            VQE(basis="FAO", device="ibm_kingston", verbose=False)

    def test_ibm_device_configures_the_qiskit_provider(self):
        driver = ADAPTVQE(pool="ceo", basis="FAO", device="ibm_kingston",
                          shots=1024, verbose=False,
                          backend_options={"instance": "crn:x"})
        assert driver.backend_provider == "qiskit"
        assert driver.execute_circuits is True
        provider = driver.circuit_provider()
        assert isinstance(provider, QiskitProvider)
        assert provider.is_ibm_device and provider.shots == 1024
        assert provider.instance == "crn:x"
        assert provider.precision == pytest.approx(1 / 32)
        assert driver.ansatz_provider() is None     # ansatz stays internal

    def test_exact_qiskit_path_is_unchanged(self):
        driver = ADAPTVQE(pool="ceo", basis="FAO", verbose=False)
        assert driver.shots == 0 and driver.circuit_provider() is None
        exact = ADAPTVQE(pool="ceo", basis="FAO", verbose=False,
                         execute_circuits=True)
        assert exact.circuit_provider() is build_provider("qiskit")

    def test_shots_refused_for_cirq(self):
        with pytest.raises(NotImplementedError, match="qiskit"):
            ADAPTVQE(pool="ceo", basis="FAO", backend_provider="cirq",
                     shots=100, verbose=False)


class TestEstimatorEnergies:
    def test_exact_local_estimator_matches_the_state_vector(self, h2_run):
        provider = QiskitProvider()                  # shots = 0, exact
        exact = h2_run.result.optimal_energy
        assert provider.precision == 0.0
        assert h2_run.measured_energy(provider) == pytest.approx(exact, abs=1e-9)
        # The Estimator route (energies) gives the same number.
        assert provider.energies([h2_run.ansatz_problem()])[0] == \
            pytest.approx(exact, abs=1e-9)

    @pytest.mark.parametrize("device", ["statevector", "fake_manila"])
    def test_sampled_estimator_converges(self, h2_run, device):
        provider = QiskitProvider(device=device, shots=4096)
        energy = h2_run.measured_energy(provider)
        assert abs(energy - h2_run.result.optimal_energy) < 0.05
        if device == "fake_manila":
            assert provider.backend().name == "fake_manila"
            isa, observable = provider.pub(*h2_run.ansatz_problem())
            assert isa.layout is not None
            assert observable.num_qubits == provider.backend().num_qubits

    def test_one_job_for_several_problems(self, h2_run):
        provider = QiskitProvider(shots=4096)
        energies = measure_energies([h2_run, h2_run], provider)
        assert len(energies) == 2
        assert all(abs(e - h2_run.result.optimal_energy) < 0.05 for e in energies)
        assert provider.last_job is not None

    def test_statevector_refused_when_it_cannot_exist(self, h2_run):
        problem = h2_run.ansatz_problem()[:4]
        with pytest.raises(ValueError, match="state vector"):
            QiskitProvider(shots=100).statevector(*problem)
        with pytest.raises(ValueError, match="state vector"):
            QiskitProvider(device="fake_manila").statevector(*problem)

    def test_processor_needs_shots(self, h2_run):
        with pytest.raises(ValueError, match="shots > 0"):
            QiskitProvider(device="fake_manila", shots=0).energies(
                [h2_run.ansatz_problem()])

    def test_unknown_fake_backend(self):
        with pytest.raises(ValueError, match="fake backend"):
            QiskitProvider(device="fake_nowhere", shots=1).backend()


class TestDeviceSelection:
    class _Status:
        def __init__(self, operational, pending):
            self.operational, self.pending_jobs = operational, pending

    class _Backend:
        def __init__(self, name, operational=True, pending=0):
            self.name, self._status = name, TestDeviceSelection._Status(
                operational, pending)

        def status(self):
            return self._status

    class _Service:
        def __init__(self, backends):
            self._backends = {b.name: b for b in backends}
            self.calls = []

        def least_busy(self, **kwargs):
            self.calls.append(("least_busy", kwargs))
            return self._backends["ibm_fez"]

        def backend(self, name):
            self.calls.append(("backend", name))
            return self._backends[name]

    @pytest.fixture
    def service(self, monkeypatch):
        svc = self._Service([self._Backend("ibm_kingston", pending=5),
                             self._Backend("ibm_fez", pending=1),
                             self._Backend("ibm_marrakesh", operational=False)])
        monkeypatch.setattr(QiskitProvider, "_service", lambda self: svc)
        return svc

    def test_least_busy_of_the_account(self, service):
        backend = QiskitProvider(device="ibm-quantum", shots=1).backend()
        assert backend.name == "ibm_fez"
        assert service.calls == [("least_busy", {"operational": True,
                                                 "simulator": False})]

    def test_named_processor(self, service):
        assert QiskitProvider(device="ibm_kingston", shots=1).backend().name \
            == "ibm_kingston"

    def test_least_busy_of_a_list_skips_non_operational(self, service):
        provider = QiskitProvider(
            device="ibm_kingston, ibm_fez, ibm_marrakesh", shots=1)
        assert provider.backend().name == "ibm_fez"
        assert provider.backend() is provider.backend()          # cached

    def test_no_operational_candidate(self, service):
        with pytest.raises(RuntimeError, match="operational"):
            QiskitProvider(device="ibm_marrakesh,ibm_marrakesh", shots=1).backend()


class TestDriversEndToEnd:
    def test_adapt_vqe_with_estimated_energies(self, h2_run):
        atoms = _h2()
        atoms.calc = ADAPTVQE(pool="ceo", basis="FAO", h=0.4, verbose=False,
                              profile=False, max_iterations=3,
                              device="AER_simulator", shots=4096,
                              optimizer="COBYLA")
        atoms.get_total_energy()
        assert isinstance(atoms.calc.circuit_provider(), QiskitProvider)
        assert abs(atoms.calc.result.optimal_energy
                   - h2_run.result.optimal_energy) < 0.05

    def test_vqe_on_a_fake_ibm_backend(self, h2_run):
        atoms = _h2()
        atoms.calc = VQE(basis="FAO", h=0.4, verbose=False,
                         device="fake_manila", shots=4096, optimizer="COBYLA")
        atoms.get_total_energy()
        assert atoms.calc.circuit_provider().backend().name == "fake_manila"
        assert abs(atoms.calc.result.optimal_energy
                   - h2_run.result.optimal_energy) < 0.05
