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

import subprocess
import sys
import textwrap

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


def _isolated(code: str) -> str:
    """Run ``code`` in a fresh interpreter and return its stdout.

    Transpiling to a fake processor allocates Qiskit circuit data on Rust
    worker threads; when the same process has also driven the Braket / Cirq
    SDKs (the providers tests), a later garbage collection can spin forever
    in Qiskit's allocator freeing them.  A subprocess keeps the fake-backend
    checks out of that trap without weakening them.
    """
    run = subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                         capture_output=True, text=True, timeout=600,
                         check=False)
    assert run.returncode == 0, run.stderr[-2000:]
    return run.stdout


_H2_RUN = """
    import warnings; warnings.simplefilter("ignore")
    from ase import Atoms
    from carcara.algorithms import ADAPTVQE, VQE
    from carcara.backends.providers import QiskitProvider
    def _h2():
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        atoms.center(vacuum=3.0)
        return atoms
    atoms = _h2()
    calc = ADAPTVQE(pool="ceo", basis="FAO", h=0.4, verbose=False,
                    profile=False, max_iterations=3)
    atoms.calc = calc
    atoms.get_total_energy()
    exact = calc.result.optimal_energy
"""


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

    def test_sampled_estimator_converges(self, h2_run):
        provider = QiskitProvider(shots=4096)
        energy = h2_run.measured_energy(provider)
        assert abs(energy - h2_run.result.optimal_energy) < 0.05

    def test_fake_processor_transpiles_and_estimates(self):
        # A fake processor carries its noise model (qiskit-aer), so the
        # tolerance is loose; the checks are the layout and the plumbing.
        out = _isolated(_H2_RUN + """
    provider = QiskitProvider(device="fake_manila", shots=4096)
    energy = calc.measured_energy(provider)
    isa, observable = provider.pub(*calc.ansatz_problem())
    print(provider.backend().name, isa.layout is not None,
          observable.num_qubits == provider.backend().num_qubits,
          abs(energy - exact) < 0.4)
""")
        assert out.split() == ["fake_manila", "True", "True", "True"]

    def test_one_job_for_several_problems(self, h2_run):
        provider = QiskitProvider(shots=4096)
        energies = measure_energies([h2_run, h2_run], provider)
        assert len(energies) == 2
        assert all(abs(e - h2_run.result.optimal_energy) < 0.05 for e in energies)
        assert provider.last_job is None and len(provider.last_result) == 2
        assert float(provider.last_result[0].data.stds) >= 0.0   # exact sampler reports 0

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

    def test_physical_qubits_pin_the_layout(self):
        out = _isolated(_H2_RUN + """
    provider = QiskitProvider(device="fake_manila", shots=1024,
                              physical_qubits=[3, 4, 1, 2])
    isa, observable = provider.pub(*calc.ansatz_problem())
    # Carcará qubit k -> physical_qubits[k]; wire n-1-k carries qubit k.
    # (The *initial* layout is what the pin fixes; routing may permute the
    # final one, which apply_layout accounts for.)
    layout = isa.layout.initial_index_layout(filter_ancillas=True)
    print([layout[4 - 1 - k] for k in range(4)])
    energy = calc.measured_energy(provider)
    print(abs(energy - exact) < 0.4)
""")
        assert out.split("\n")[0] == "[3, 4, 1, 2]"
        assert out.split()[-1] == "True"
        with pytest.raises(ValueError, match="entries"):
            QiskitProvider(device="fake_manila", shots=1,
                           physical_qubits=[0, 1]).pub(
                4, [0, 2], [], [], __import__("carcara.core", fromlist=["PauliSum"]).PauliSum({"ZIII": 1.0}))

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

    def test_vqe_on_a_fake_ibm_backend(self):
        out = _isolated(_H2_RUN + """
    atoms = _h2()
    atoms.calc = VQE(basis="FAO", h=0.4, verbose=False,
                     device="fake_manila", shots=4096, optimizer="COBYLA")
    atoms.get_total_energy()
    print(atoms.calc.circuit_provider().backend().name,
          abs(atoms.calc.result.optimal_energy - exact) < 0.4)
""")
        assert out.split() == ["fake_manila", "True"]
