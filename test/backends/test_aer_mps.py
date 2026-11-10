"""The ``"AER_mps"`` device: Aer's matrix-product-state simulator for circuits too wide for a state
vector.

It is opt-in, sampled (so ``shots > 0``), and refuses what it cannot do instead of selecting
another calculation. The 50-qubit run of its real circuits lives in
``test/architecture/test_shot_path_memory.py``; here a small register checks that the device
reproduces the exact energy within shot noise, and the options and refusals are pinned.
"""
import numpy as np
import pytest

from mandacaru import Mandacaru
from mandacaru.backends import available_devices, is_simulator, normalize_device
from mandacaru.backends.hardware import get_device, requires_shots
from mandacaru.backends.providers import QiskitProvider
from mandacaru.core.mapping import Fermion

ORBITALS = 4


def hamiltonian():
    """Orbital energies, hops along a spin-block chain and an on-site repulsion."""
    terms = {}
    for p in range(ORBITALS):
        for spin in (0, 1):
            i = p + spin * ORBITALS
            terms[((i, True), (i, False))] = 0.2 * (p + 1)
    for p in range(ORBITALS - 1):
        for spin in (0, 1):
            i, j = p + spin * ORBITALS, p + 1 + spin * ORBITALS
            terms[((i, True), (j, False))] = -0.3
            terms[((j, True), (i, False))] = -0.3
    terms[((0, True), (ORBITALS, True), (ORBITALS, False), (0, False))] = 0.5
    return Fermion(terms)


def calculation(**options):
    options.setdefault("optimizer", {"method": "COBYLA", "maxiter": 0})
    return Mandacaru(method="vqe", hamiltonian=hamiltonian(), n_spatial_orbitals=ORBITALS,
                     num_particles=(1, 1), atomic_units=True, trace=False,
                     resources={"threads": 1}, **options)


class TestTheDevice:
    def test_it_is_a_registered_simulator_that_needs_shots(self):
        assert "AER_mps" in available_devices()
        for alias in ("aer_mps", "aer-mps", "mps", "AER_mps"):
            assert normalize_device(alias) == "AER_mps"
        device = get_device("mps")
        assert device.simulator and device.shots_only and device.provider == "qiskit"
        assert is_simulator("AER_mps") and requires_shots("AER_mps")
        assert not device.is_ibm and not device.is_fake

    def test_it_is_neither_hardware_nor_local_state_vector(self):
        provider = QiskitProvider(device="AER_mps", shots=100)
        assert provider.is_mps_device and not provider.is_ibm_device
        assert not provider.is_hardware and not provider.is_local
        with pytest.raises(ValueError, match="cannot return a state vector"):
            provider.statevector(2, [0], [], [])

    def test_the_backend_is_the_matrix_product_state_method_with_the_cap_and_seed(self):
        provider = QiskitProvider(device="AER_mps", shots=100, max_bond_dimension=12, seed=4)
        options = provider.backend().options
        assert options.method == "matrix_product_state"
        assert options.matrix_product_state_max_bond_dimension == 12
        assert options.seed_simulator == 4
        assert provider.with_options(shots=50).max_bond_dimension == 12


class TestWhatItRefuses:
    def test_no_shots_is_refused_when_the_calculator_is_built(self):
        with pytest.raises(ValueError, match="requires shots > 0"):
            calculation(device="AER_mps")
        with pytest.raises(ValueError, match="requires shots > 0"):
            QiskitProvider(device="AER_mps", shots=0)

    @pytest.mark.parametrize("value", [0, -3, 2.5, True, "8"])
    def test_a_bond_dimension_must_be_a_positive_integer(self, value):
        with pytest.raises(ValueError, match="positive integer"):
            QiskitProvider(device="AER_mps", shots=100, max_bond_dimension=value)

    @pytest.mark.parametrize("device", ["statevector", "fake_manila", "ibm_kingston"])
    def test_the_bond_dimension_belongs_to_the_mps_device(self, device):
        with pytest.raises(ValueError, match="option of the 'AER_mps' device"):
            QiskitProvider(device=device, shots=100, max_bond_dimension=8)

    @pytest.mark.parametrize("option, value", [
        ("physical_qubits", [0, 1]),
        ("enable_fractional_gates", True),
        ("estimator_options", {"resilience_level": 1}),
    ])
    def test_processor_options_are_refused(self, option, value):
        with pytest.raises(ValueError, match=f"{option} describes a processor"):
            QiskitProvider(device="AER_mps", shots=100, **{option: value})

    def test_a_bond_dimension_through_the_calculator_validates_too(self):
        with pytest.raises(ValueError, match="positive integer"):
            calculation(device="AER_mps", shots=100, backend_options={"max_bond_dimension": 0})
        with pytest.raises(ValueError, match="option of the 'AER_mps' device"):
            calculation(device="fake_manila", shots=100,
                        backend_options={"max_bond_dimension": 4})


def test_a_small_register_reproduces_the_exact_energy_within_shot_noise():
    """The MPS circuits are the exact ansatz: at a fixed point the sampled energy is the exact
    native one to the standard error the 1-norm bound gives (the bond cap is not reached)."""
    exact = calculation()
    exact.run()
    parameters = 0.1 * np.sin(np.arange(1, len(exact.problem.pool_labels) + 1))
    reference = exact.run(initial_parameters=parameters).energy_hartree
    shots = 100_000
    sampled = calculation(device="AER_mps", shots=shots,
                          backend_options={"max_bond_dimension": 64, "seed": 7})
    energy = sampled.run(initial_parameters=parameters).energy_hartree
    one_norm = sum(np.hypot(re, im) for label, re, im in sampled.problem.hamiltonian_terms
                   if label.strip("I"))
    assert abs(energy - reference) < 4 * one_norm / np.sqrt(shots)
    assert sampled.provider.backend().options.method == "matrix_product_state"
    assert sampled.provider.run_usage()["jobs"] >= 1


def test_an_mps_job_too_large_for_the_machine_is_refused_by_name(monkeypatch):
    """One MPS job holds all its bases at once (about 0.67 MB each): above the limit the run is
    refused before anything is simulated, naming the option that splits it."""
    from mandacaru.integrations import circuits

    monkeypatch.setattr(circuits, "MPS_BASES_PER_JOB", 1)
    with pytest.raises(ValueError, match="max_bases_per_job"):
        calculation(device="AER_mps", shots=100).run()
