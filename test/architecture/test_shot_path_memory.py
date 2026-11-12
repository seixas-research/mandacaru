"""The hardware rule: a shot-based run keeps its classical memory polynomial in the qubit count.

CLAUDE.md promises that no ``2^n`` array, sector-sized vector or matrix, and no local state-vector
screening, lies on the path of a ``shots > 0`` run, up to 50 qubits. This is the test that
forbids them: a 25-orbital (50-qubit) problem is run through ``Mandacaru`` for ``vqe`` and
``adapt-vqe``, tapered or not, and the run must

* finish (the old path built a state-vector ``Basis`` at construction and never did),
* ask the device for its numbers (the executor is called) and never for a state vector,
* leave the native context's peak allocation under 400 MB -- one ``2^25`` complex128 array alone
  is 512 MB, and the particle sector of 5 + 5 electrons in these orbitals has 2.8 billion
  determinants -- and raise the process's peak resident size by under 1.5 GB (a delta, so what
  an earlier test of the session used does not count),
* construct no NumPy array of ``2^25`` elements or more,
* refuse every state-vector method of the problem by name.

The SDK end of the path is stubbed in the tests above: ``QiskitProvider.expectation_batch`` is replaced by a
function that returns a smooth landscape. The local Qiskit estimators are exact state-vector
simulators (and the fake backends run on Aer's), so no SDK simulates 50 qubits here; what those tests
pin is everything Mandacaru itself builds around the circuit. Hardware has no such limit. The
last tests run the real circuits on the opt-in ``"AER_mps"`` device (Aer's matrix-product-state
simulator): a few generators at a small bond dimension, with no stub.
"""
import resource
import sys
import time

import numpy as np
import pytest

from mandacaru import Mandacaru, _native
from mandacaru.core.mapping import Fermion

ORBITALS = 25
EXPONENTIAL = 2 ** 25
#: Native allocation ceiling, bytes. A single 2^25 complex128 array is 512 MB.
NATIVE_PEAK = 400 * 10 ** 6
PROCESS_GROWTH = 1.5 * 10 ** 9
MEASUREMENT_BUDGET = {"observables": 500_000, "bases": 100_000}


def rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def hamiltonian():
    """Orbital energies, spin-block hops along the chain, and an on-site repulsion."""
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
    for p in range(3):
        terms[((p, True), (p + ORBITALS, True), (p + ORBITALS, False), (p, False))] = 0.5
    return Fermion(terms)


class Landscape:
    """Stands for the SDK: Pauli expectations that vary smoothly with the angles."""

    def __init__(self):
        self.calls = 0
        self.observables = 0

    def __call__(self, n_qubits, occupied, requests, max_bases_per_job=None):
        self.calls += 1
        answers = []
        for _generators, thetas, labels in requests:
            self.observables += len(labels)
            total = float(np.sum(thetas))
            values = {label: 0.1 * np.cos(len(label.strip("I")) + total) for label in labels}
            answers.append((values, dict.fromkeys(labels, 0.0)))
        return answers


@pytest.fixture
def no_exponential_numpy(monkeypatch):
    """Every NumPy constructor refuses an array of 2^25 elements or more."""
    for name in ("zeros", "empty", "ones", "full", "array", "asarray", "ascontiguousarray"):
        original = getattr(np, name)

        def guarded(*args, _original=original, _name=name, **kwargs):
            out = _original(*args, **kwargs)
            if getattr(out, "size", 0) >= EXPONENTIAL:
                raise AssertionError(f"np.{_name} built an array of {out.size} elements")
            return out

        monkeypatch.setattr(np, name, guarded)


@pytest.mark.parametrize("particles", [(1, 0), (1, 1)])
@pytest.mark.parametrize("taper", [False, True])
@pytest.mark.parametrize("method", ["vqe", "adapt-vqe"])
def test_a_fifty_qubit_shot_run_builds_no_exponential_object(
        method, taper, particles, no_exponential_numpy):
    resident_before = rss_bytes()
    # The fermionic pool keeps this a test of the register-free memory architecture: the
    # default ``ceo`` pool of ``adapt-vqe`` has 588,300 generators at 50 qubits and is refused
    # by the budget before it is planned (``test_the_default_ceo_pool_of_fifty_qubits_...``).
    calc = Mandacaru(
        method=method, pool="fermionic", hamiltonian=hamiltonian(), n_spatial_orbitals=ORBITALS,
        num_particles=particles, taper=taper, shots=64, backend_provider="qiskit",
        atomic_units=True, trace=False, max_iterations=1,
        optimizer={"method": "COBYLA", "maxiter": 2, "maxfev": 4},
        measurement_budget=MEASUREMENT_BUDGET, parallelization={"threads": 1})
    landscape = Landscape()
    calc.provider.expectation_batch = landscape

    def statevector(*args, **kwargs):
        raise AssertionError("a shot-based run asked the SDK for a state vector")

    calc.provider.statevector = statevector
    result = calc.run()

    problem = calc.problem
    assert problem.register_free
    assert problem.num_qubits < 50 if taper else problem.num_qubits == 50
    assert landscape.calls > 0 and landscape.observables > 0
    assert "state" not in result._raw and result._raw["state_kind"] == "none"
    assert np.isfinite(result.in_units("Ha"))
    assert calc.context.memory_peak_bytes < NATIVE_PEAK, calc.context.memory_peak_bytes
    assert rss_bytes() - resident_before < PROCESS_GROWTH, rss_bytes() - resident_before
    assert (problem.taper is not None) == taper


def test_the_rdms_of_a_fifty_qubit_shot_run_are_refused_by_the_budget_without_a_state(
        no_exponential_numpy):
    """``measured_rdms()`` at 50 qubits needs about a million Pauli strings (the plan alone is
    polynomial: ``crates/mandacaru-core/tests/rdm_measurement.rs`` builds it at 59 MB). The
    default observables ceiling refuses it while the plan is being built, before any grouping or
    job, and nothing on the way forms a state or a ``2^n`` object."""
    resident_before = rss_bytes()
    calc = Mandacaru(
        method="vqe", pool="fermionic", hamiltonian=hamiltonian(), n_spatial_orbitals=ORBITALS,
        num_particles=(1, 1), shots=64, backend_provider="qiskit", atomic_units=True,
        trace=False, optimizer={"method": "COBYLA", "maxiter": 2, "maxfev": 4},
        measurement_budget=MEASUREMENT_BUDGET, parallelization={"threads": 1})
    landscape = Landscape()
    calc.provider.expectation_batch = landscape

    def statevector(*args, **kwargs):
        raise AssertionError("measured RDMs asked the SDK for a state vector")

    calc.provider.statevector = statevector
    calc.run()
    calls = landscape.calls
    with pytest.raises(ValueError, match="observables budget"):
        calc.measured_rdms()
    assert landscape.calls == calls
    assert calc.context.memory_peak_bytes < NATIVE_PEAK, calc.context.memory_peak_bytes
    assert rss_bytes() - resident_before < PROCESS_GROWTH, rss_bytes() - resident_before


@pytest.mark.parametrize("particles", [(1, 0), (1, 1)])
def test_the_default_ceo_pool_of_fifty_qubits_is_refused_before_it_is_planned(particles):
    """The refusal costs nothing: the default ``ceo`` pool has 588,300 generators at 50 qubits
    and its gradient step needs 7.8 million Pauli observables against the default ceiling of
    50,000. The native walk stops as soon as the distinct strings pass the ceiling, so the run
    is refused in seconds and a fraction of a GB, not after minutes and 1.9 GB of planning, and
    the device is never asked for anything. The message names the pool, its size, the budget and
    the three ways out."""
    resident_before = rss_bytes()
    started = time.perf_counter()
    calc = Mandacaru(
        method="adapt-vqe", hamiltonian=hamiltonian(), n_spatial_orbitals=ORBITALS,
        num_particles=particles, shots=64, backend_provider="qiskit", atomic_units=True,
        trace=False, max_iterations=1, optimizer={"method": "COBYLA", "maxiter": 2, "maxfev": 4},
        parallelization={"threads": 1})
    landscape = Landscape()
    calc.provider.expectation_batch = landscape
    with pytest.raises(ValueError) as refused:
        calc.run()
    elapsed = time.perf_counter() - started
    message = str(refused.value)
    assert "'ceo' pool has 588,300 generators" in message
    assert "more than 50,000 Pauli observables" in message
    assert 'pool="fermionic"' in message and '"qeb"' in message
    assert 'measurement_budget={"observables": N}' in message
    assert landscape.calls == 0
    assert elapsed < 30, elapsed       # measured 2.1 s; the unrefused plan took 360 s
    assert rss_bytes() - resident_before < 10 ** 9, rss_bytes() - resident_before


def test_every_state_vector_method_of_a_shot_problem_refuses_by_name():
    calc = Mandacaru(
        method="vqe", hamiltonian=hamiltonian(), n_spatial_orbitals=ORBITALS,
        num_particles=(1, 0), shots=64, backend_provider="qiskit", atomic_units=True,
        trace=False, optimizer={"method": "COBYLA", "maxiter": 2, "maxfev": 3},
        parallelization={"threads": 1})
    calc.provider.expectation_batch = Landscape()
    result = calc.run()
    problem = calc.problem
    state = np.zeros(4, dtype=complex)
    for what, call in (
            ("state", lambda: problem.state([], indices=[])),
            ("evaluate", lambda: problem.evaluate([])),
            ("rdms", lambda: problem.rdms(state)),
            ("expand_state", lambda: problem.expand_state(state)),
            ("dimension", lambda: problem.dimension),
            ("basis_indices", lambda: problem.basis_indices),
            ("energy_levels", lambda: problem.energy_levels()),
            ("solve_subspace", lambda: problem.solve_subspace([2.0, 1.0])),
            ("solve", lambda: problem.solve(adapt=True))):
        with pytest.raises(Exception, match="register-free"):
            call()
    with pytest.raises(ValueError, match="no state vector"):
        result.natural_orbitals()


def test_a_sparse_matrix_is_refused_on_a_shot_run():
    with pytest.raises(Exception, match="register-free"):
        Mandacaru(
            method="vqe", hamiltonian=hamiltonian(), n_spatial_orbitals=ORBITALS,
            num_particles=(1, 0), shots=64, backend_provider="qiskit", sparse=True,
            atomic_units=True, trace=False, parallelization={"threads": 1}).run()


@pytest.mark.parametrize("sector", [True, False])
def test_the_native_problem_of_fifty_qubits_builds_register_free(sector):
    """The construction that never finished: a 50-qubit problem, with and without the sector."""
    n = 50
    context = _native.Context(threads=1)
    operator = context.pauli_operator(
        n, [("Z" + "I" * (n - 1), 1.0, 0.0), ("I" * n, -1.0, 0.0)])
    problem = context.quantum_problem(
        operator, 25, [5, 5], sector=sector, pool="qeb", register_free=True)
    assert problem.register_free and problem.num_qubits == n
    assert len(problem.pool_labels) > 10_000
    assert context.memory_peak_bytes < NATIVE_PEAK
    with pytest.raises(Exception, match="register-free"):
        problem.dimension


def mps_calculation(particles, *, backend_options, **options):
    return Mandacaru(
        method="adapt-vqe", pool="fermionic", hamiltonian=hamiltonian(), n_spatial_orbitals=ORBITALS,
        num_particles=particles, shots=64, device="AER_mps", atomic_units=True, trace=False,
        max_iterations=1, optimizer={"method": "COBYLA", "maxiter": 4, "maxfev": 4},
        backend_options={"max_bond_dimension": 16, "seed": 5, **backend_options},
        measurement_budget=MEASUREMENT_BUDGET, parallelization={"threads": 1}, **options)


@pytest.mark.parametrize("particles, max_bases_per_job", [((1, 0), None), ((1, 1), 100)])
def test_a_fifty_qubit_shot_run_executes_its_real_circuits_on_the_mps_simulator(
        particles, max_bases_per_job, no_exponential_numpy):
    """No stub: the ansatz circuit of 50 qubits runs on Aer's matrix-product-state simulator,
    the energies and the pool gradients are sampled from it, within 2 GB. A (1,1) ADAPT step
    measures about 13,000 gradient bases; one job of all of them held 8.7 GB in the Estimator
    (124 s), so the run bounds a job with ``max_bases_per_job`` (17 s, 0.35 GB)."""
    resident_before = rss_bytes()
    options = {} if max_bases_per_job is None else {"max_bases_per_job": max_bases_per_job}
    calc = mps_calculation(particles, backend_options=options)
    submitted = []
    original = calc.provider._run_pubs
    calc.provider._run_pubs = lambda pubs, bases=None: (
        submitted.append(len(pubs)), original(pubs, bases=bases))[1]
    result = calc.run()
    problem = calc.problem
    assert problem.register_free and problem.num_qubits == 50
    assert calc.provider.is_mps_device and not calc.provider.is_hardware
    assert calc.provider.backend().options.method == "matrix_product_state"
    assert np.isfinite(result.in_units("Ha"))
    assert "state" not in result._raw and result._raw["state_kind"] == "none"
    usage = calc.provider.run_usage()
    assert usage["jobs"] > 0 and usage["shots"] > 0
    assert usage["jobs"] == len(submitted)
    assert calc.context.memory_peak_bytes < NATIVE_PEAK
    assert rss_bytes() - resident_before < 2 * 10 ** 9, rss_bytes() - resident_before
