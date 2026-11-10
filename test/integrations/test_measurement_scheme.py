"""``measurement_scheme="double-factorized"``: the ``O(M)`` basis-rotation energy measurement.

The numerics (factorization, Givens networks, diagonal observables, assembly) are Rust and are
pinned against ``test/migration/reference/factorization.json`` in
``crates/mandacaru-core/tests/factorized_measurement.rs``. These tests pin what Python owns: the
option and its refusals, the circuit path through ``Mandacaru`` on H2 and LiH (the factorized
energy is the commuting-group energy and the native state-vector energy up to the leaf
truncation), the number of bases against the qubit-wise commuting groups, a seeded shot run,
and the job and shot accounting (one job per energy, for either scheme).
"""
import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru, _native
from mandacaru.backends.measurement import RunBudgetExceeded
from mandacaru.core.mapping import Fermion
from mandacaru.integrations import measurement_scheme

#: The freeze's geometries and basis (``test/migration/freeze_factorization.py``).
CASES = {"H2": (0.74, 8.0), "LiH": (1.60, 9.0)}
OPTIONS = {"basis": "HAO", "grid_spacing": 0.35}
#: Leaf truncation of the energy: 3e-14 for H2 and 4e-13 for LiH at the trial point. (The frozen
#: LiH factorized energy sits 1.65e-9 off the exact one because Qiskit's ``ObservablesArray``
#: drops coefficients below 1e-8, not because of the factorization; see
#: ``crates/mandacaru-core/tests/factorized_measurement.rs``.)
TRUNCATION = {"H2": 1e-11, "LiH": 1e-10}
FACTORIZED = {"measurement_scheme": "double-factorized"}
#: Shot noise of the seeded H2 run, Ha (the three seeds tried deviate by 0.8 to 2.3 mHa).
TOLERANCE_SHOTS = 0.01


@pytest.fixture(scope="module")
def models():
    """The molecular model of each case, from a classical run (the integrals once)."""
    out = {}
    for name, (distance, cell) in CASES.items():
        atoms = Atoms(name, positions=[(0, 0, 0), (0, 0, distance)], cell=[cell] * 3)
        atoms.center()
        atoms.calc = Mandacaru(method="rhf", trace=False, **OPTIONS)
        atoms.get_potential_energy()
        out[name] = atoms.calc.native_reference.model
    return out


def circuit_run(model, *, method="vqe", parameters=None, **options):
    """A circuit calculation on a model: the state-vector device unless ``shots`` is given."""
    options.setdefault("optimizer", {"method": "COBYLA", "maxiter": 0})
    calc = Mandacaru(method=method, native_model=model, execute_circuits=True,
                     atomic_units=True, trace=False, **options)
    result = calc.run(initial_parameters=parameters)
    return calc, result


def pool_size(calc):
    return len(calc.problem.pool_terms)


def trial_parameters(model):
    """A reproducible nonzero point of the ansatz (its length is the pool's)."""
    calc, _ = circuit_run(model)
    return 0.08 * np.sin(np.arange(1, pool_size(calc) + 1))


class TestOption:
    def test_the_default_is_the_commuting_groups(self, models):
        calc, _ = circuit_run(models["H2"])
        assert calc.problem.factorized_measurement is None

    def test_a_name_other_than_the_two_schemes_is_refused(self, models):
        with pytest.raises(ValueError, match="unknown measurement_scheme 'factorized'"):
            Mandacaru(method="vqe", native_model=models["H2"], execute_circuits=True,
                      measurement_scheme="factorized")

    def test_it_needs_a_circuit_device(self, models):
        with pytest.raises(ValueError, match="state-vector solver measures nothing"):
            Mandacaru(method="vqe", native_model=models["H2"], **FACTORIZED)

    @pytest.mark.parametrize("method", ["rhf", "dft", "subspace-vqe", "vasqa"])
    def test_other_methods_refuse_it(self, models, method):
        with pytest.raises(ValueError, match="does not use: measurement_scheme"):
            Mandacaru(method=method, execute_circuits=True, **FACTORIZED)

    def test_a_supplied_qubit_hamiltonian_has_no_integrals(self):
        hamiltonian = Fermion({((0, True), (0, False)): 0.5, ((1, True), (1, False)): 0.5})
        with pytest.raises(ValueError, match="carries none"):
            Mandacaru(method="vqe", hamiltonian=hamiltonian, n_spatial_orbitals=1,
                      num_particles=(1, 1), execute_circuits=True, **FACTORIZED)

    @pytest.mark.parametrize("extra, match", [
        ({"mapping": "parity_reduced"}, "Jordan-Wigner picture"),
        ({"taper": True}, "taper=True"),
        ({"ansatz": "hva"}, "ansatz='hva'"),
    ])
    def test_what_it_cannot_measure_is_refused(self, models, extra, match):
        with pytest.raises(ValueError, match=match):
            Mandacaru(method="vqe", native_model=models["H2"], execute_circuits=True,
                      **FACTORIZED, **extra)

    def test_a_non_molecular_model_is_refused(self):
        with pytest.raises(ValueError, match="not a molecular model"):
            measurement_scheme.check_model(object(), _native)

    def test_the_native_problem_validates_the_scheme_and_refuses_a_tapered_register(self, models):
        with pytest.raises(ValueError, match="unknown measurement scheme"):
            models["H2"].problem(measurement="x")
        with pytest.raises(ValueError, match="cannot be tapered"):
            models["H2"].problem(measurement="double-factorized", sector=False, taper=True)
        with pytest.raises(ValueError, match="Jordan-Wigner picture"):
            models["H2"].problem(measurement="double-factorized", mapping="parity")


@pytest.mark.parametrize("name", ["H2", "LiH"])
def test_the_factorized_energy_is_the_commuting_group_and_the_exact_energy(models, name):
    model = models[name]
    parameters = trial_parameters(model)
    qwc, qwc_result = circuit_run(model, parameters=parameters)
    factorized, result = circuit_run(model, parameters=parameters, **FACTORIZED)
    exact = factorized.problem.evaluate(parameters, trotter=True)["energy_ha"]
    assert abs(qwc_result.energy_hartree - exact) < 1e-10
    assert abs(result.energy_hartree - qwc_result.energy_hartree) < TRUNCATION[name]
    assert abs(result.energy_hartree - exact) < TRUNCATION[name]
    assert result._raw["execution"] == "qiskit"


def native_groups(calc):
    """The qubit-wise commuting groups of the problem's Hamiltonian."""
    labels = sorted({label for label, _re, _im in calc.problem.hamiltonian_terms
                     if label.strip("I")})
    return calc.context.qwc_groups(labels)


@pytest.mark.parametrize("name", ["H2", "LiH"])
def test_it_needs_fewer_measurement_bases_than_the_commuting_groups(models, name):
    calc, _ = circuit_run(models[name], **FACTORIZED)
    plan = calc.problem.factorized_measurement
    groups = len(native_groups(calc))
    # L + 1 bases: the one-body term and one per leaf, 3 and 6 leaves for H2 and LiH.
    assert plan["bases"] == {"H2": 4, "LiH": 7}[name]
    assert plan["truncation_error"] < 1e-10
    if name == "LiH":
        assert plan["bases"] < groups, (plan["bases"], groups)
    assert groups > 0


def test_adapt_vqe_factorizes_its_energies_and_keeps_its_pool_gradients(models):
    options = {"optimizer": {"method": "BFGS", "maxiter": 40}, "max_iterations": 3,
               "convergence": {"gradient": 1e-4, "energy": 1e-6}}
    _, reference = circuit_run(models["H2"], method="adapt-vqe", **options)
    calc, result = circuit_run(models["H2"], method="adapt-vqe", **options, **FACTORIZED)
    assert calc.problem.factorized_measurement["bases"] == 4
    assert abs(result.energy_hartree - reference.energy_hartree) < 1e-9


def test_a_seeded_shot_run_agrees_with_the_exact_energy_within_shot_noise(models):
    """Local shot simulator, fixed seed. The estimator draws ``N(0, 1/sqrt(shots))`` per Pauli
    string, so a basis carries ``sqrt(sum c^2) / sqrt(shots)``: at 1e5 shots H2's energy has a
    standard error of about 2 mHa and the tolerance is several of them."""
    model = models["H2"]
    parameters = trial_parameters(model)
    exact = circuit_run(model, parameters=parameters)[0].problem.evaluate(
        parameters, trotter=True)["energy_ha"]
    calc, result = circuit_run(
        model, parameters=parameters, shots=100_000, backend_provider="qiskit",
        backend_options={"seed": 11}, **FACTORIZED)
    assert calc.problem.register_free
    assert result._raw["state_kind"] == "none"
    assert abs(result.energy_hartree - exact) < TOLERANCE_SHOTS


def test_the_bases_of_one_energy_are_one_job_and_the_ledger_charges_what_the_cost_arithmetic_says(
        models):
    """LiH's seven bases are seven different circuits in one batch, so one job per energy and no
    circuit is reused: the ledger charges exactly the cost arithmetic. (H2's last leaf has the
    identity rotation, the same circuit as the one-body basis; the batch still carries it as a
    circuit of its own, so the count is the same.)"""
    model = models["LiH"]
    shots = 1000
    calc, _ = circuit_run(model, shots=shots, backend_provider="qiskit",
                          backend_options={"seed": 3}, **FACTORIZED)
    bases = calc.problem.factorized_measurement["bases"]
    jobs, total_shots = measurement_scheme.energy_cost(calc)
    assert (jobs, total_shots) == (1, bases * shots)
    assert (jobs, total_shots) == calc.context.measurement_cost(
        bases, shots=shots, noise_factors=1, randomizations=1, shots_per_randomization=0)
    usage = calc.provider.run_usage()
    assert usage["jobs"] >= 1
    assert usage["shots"] == usage["jobs"] * total_shots


def test_a_limit_on_the_bases_of_a_job_splits_the_batch_as_the_cost_arithmetic_counts(models):
    model = models["LiH"]
    calc, _ = circuit_run(model, shots=1000, backend_provider="qiskit",
                          backend_options={"seed": 3, "max_bases_per_job": 3}, **FACTORIZED)
    bases = calc.problem.factorized_measurement["bases"]
    jobs, _ = measurement_scheme.energy_cost(calc)
    assert (bases, jobs) == (7, 3)
    usage = calc.provider.run_usage()
    assert usage["jobs"] % jobs == 0 and usage["jobs"] >= jobs


def test_a_split_commuting_group_energy_is_priced_as_it_runs(models):
    """The preflight's cost of a commuting-group energy is what the executor submits: the groups
    as the executor forms them (unweighted), packed whole into jobs of ``max_bases_per_job``. The
    weighted grouping of the provider's own ``energies`` API is not the solver's route."""
    from mandacaru.integrations.circuits import energy_bases, energy_measurement_cost

    shots = 1000
    calc, _ = circuit_run(models["LiH"], shots=shots, backend_provider="qiskit",
                          backend_options={"seed": 3, "max_bases_per_job": 4})
    bases = energy_bases(calc)
    jobs, total_shots = energy_measurement_cost(calc)
    assert bases > 4 and jobs == -(-bases // 4) and total_shots == bases * shots
    usage = calc.provider.run_usage()
    evaluations = usage["jobs"] // jobs
    assert evaluations >= 1 and usage["jobs"] == evaluations * jobs
    assert usage["shots"] == evaluations * total_shots
    assert calc._energy_measurement == {
        "scheme": "merged QWC measurement bases", "bases_per_energy": bases,
        "jobs_per_energy": jobs, "shots_per_energy": total_shots}


def test_braket_submits_a_task_per_basis_and_the_cost_says_so():
    from types import SimpleNamespace

    driver = SimpleNamespace(
        context=_native.Context(threads=1), _shots=1000,
        problem=SimpleNamespace(factorized_measurement={"bases": 4}),
        provider=SimpleNamespace(name="braket", shots=1000))
    assert measurement_scheme.energy_cost(driver) == (4, 4000)


@pytest.mark.parametrize("device", ["AER_simulator", "fake_manila"])
@pytest.mark.parametrize("scheme", ["qwc", "double-factorized"])
def test_an_energy_is_one_job_on_the_local_and_the_fake_devices(models, device, scheme):
    """The gate of the batched measurement: whatever the scheme, each energy evaluation of a
    shot-based run is one submission carrying every circuit it needs."""
    batches = []
    jobs = []
    calc = Mandacaru(
        method="vqe", native_model=models["H2"], execute_circuits=True, atomic_units=True,
        trace=False, shots=2000, device=device, backend_options={"seed": 2},
        optimizer={"method": "COBYLA", "maxiter": 4, "maxfev": 4},
        measurement_scheme=scheme)
    batch = calc.provider.expectation_batch
    submit = calc.provider._run_pubs

    def counting_batch(n_qubits, occupied, requests, *args, **kwargs):
        batches.append(len(requests))
        return batch(n_qubits, occupied, requests, *args, **kwargs)

    def counting_submit(pubs, bases=None):
        jobs.append(len(pubs))
        return submit(pubs, bases=bases)

    calc.provider.expectation_batch = counting_batch
    calc.provider._run_pubs = counting_submit
    calc.run()
    assert len(jobs) == len(batches) >= 2
    assert calc.provider.run_usage()["jobs"] == len(jobs)
    plan = calc.problem.factorized_measurement
    if scheme == "qwc":
        assert plan is None and set(batches) == {1}
    else:
        assert set(batches) == {plan["bases"]} and plan["bases"] > 1
        assert set(jobs) == {plan["bases"]}


def test_the_run_budget_preflight_prices_a_factorized_energy(models):
    """H2 costs one job per energy (its 4 bases ride in one batch), so three evaluations need 3
    jobs: a ceiling of 2 refuses the run before anything is queued."""
    with pytest.raises(RunBudgetExceeded, match="over the run budget") as refused:
        circuit_run(models["H2"], shots=200, backend_provider="qiskit",
                    backend_options={"seed": 1, "run_budget": {"jobs": 2}},
                    optimizer={"method": "COBYLA", "maxiter": 3, "maxfev": 3}, **FACTORIZED)
    assert 'backend_options={"run_budget": {\'jobs\': 3}}' in str(refused.value)
    circuit_run(models["H2"], shots=200, backend_provider="qiskit",
                backend_options={"seed": 1, "run_budget": {"jobs": 3}},
                optimizer={"method": "COBYLA", "maxiter": 3, "maxfev": 3}, **FACTORIZED)


def test_the_measurement_budget_bounds_the_bases_of_an_energy(models):
    with pytest.raises(ValueError, match=r'bases budget exceeded: 4 > 2;.*'
                       r'measurement_budget=\{"bases": N\}.*N >= 4'):
        circuit_run(models["H2"], shots=200, backend_provider="qiskit",
                    measurement_budget={"bases": 2}, **FACTORIZED)


@pytest.mark.parametrize("scheme", ["qwc", "double-factorized"])
def test_a_shot_run_logs_how_its_energy_is_measured(scheme, models, tmp_path):
    """The run log owns the energy measurement's scheme, bases, jobs and shots in one block."""
    from mandacaru.utils.logging import parse_output

    log = tmp_path / "run.txt"
    circuit_run(models["H2"], shots=1000, backend_provider="qiskit", measurement_scheme=scheme,
                txt=str(log))
    block = parse_output(log)["energy_measurement"]
    assert ("double-factorized" in block["scheme"]) == (scheme == "double-factorized")
    assert int(block["bases_per_energy"]) >= 1 and int(block["jobs_per_energy"]) == 1
    assert int(block["shots_per_energy"]) == 1000 * int(block["bases_per_energy"])
