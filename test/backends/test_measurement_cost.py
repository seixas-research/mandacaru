# -*- coding: utf-8 -*-
# file: test/test_measurement_scaling.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""What a hardware measurement costs, and the six things that make it smaller.

Written after a 24-qubit LiH/PAW-LCAO-TZP job died on IBM Runtime with "error code
1336; Program runtime ran out of memory" -- 37 minutes of queue, then nothing,
and a converged optimization discarded with it.  The job had submitted 97,980
Pauli observables in one PUB because an energy-only ``get_potential_energy()``
measured the full 1- and 2-RDM operator set it never used.

Each class here pins one of the fixes, and the arithmetic behind it:

* an energy goes as **one weighted observable**, not N single-Pauli ones;
* a plan sizes the job *before* it is queued, and a budget refuses it;
* observables are chunked into jobs of bounded size, and completed chunks are
  kept when a later one fails;
* the run-wide budget stops a paid run before it crosses its ceiling;
* double factorization needs ``O(M)`` measurement bases where qubit-wise
  commuting grouping needs ``O(M^3)``.
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.backends.factorization import (double_factorization,
                                              spatial_double_factorization)
from mandacaru.backends.measurement import (DEFAULT_MEASUREMENT_BUDGET,
                                            MeasurementBudgetError,
                                            MeasurementPlan,
                                            chunk_labels_by_basis,
                                            measurement_plan, planned_jobs,
                                            qubit_wise_commuting_groups,
                                            resolve_measurement_budget)
from mandacaru.backends.providers import (QiskitProvider, build_provider,
                                          qpu_usage)
from mandacaru.core.mapping import Fermion, PauliSum

qiskit = pytest.importorskip("qiskit")


def h2(distance: float = 0.74) -> Atoms:
    atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, distance)],
                  cell=[8.0, 8.0, 8.0])
    atoms.center()
    return atoms


def lih(distance: float = 1.60) -> Atoms:
    atoms = Atoms("LiH", positions=[(0, 0, 0), (0, 0, distance)],
                  cell=[9.0, 9.0, 9.0])
    atoms.center()
    return atoms


# --------------------------------------------------------------------------- #
# 2. The pre-flight plan, and the budget that refuses a job.
# --------------------------------------------------------------------------- #

class TestThePlanIsMadeBeforeTheJobIsQueued:

    def test_resilience_multiplies_the_circuit_count(self):
        plain = QiskitProvider(device="statevector", shots=100)
        mitigated = QiskitProvider(device="statevector", shots=100,
                                   estimator_options={"resilience_level": 2})
        labels = ["XXII", "ZZII", "IIXY"]
        a = measurement_plan(plain, 4, labels)
        b = measurement_plan(mitigated, 4, labels)
        # ZNE's three noise factors and the twirling randomizations are
        # circuits the processor runs.  At 100 shots Runtime's "auto" rule
        # takes 64 shots per randomization, so two randomizations: 6x.
        assert b.circuit_instances == a.circuit_instances * 3 * 2
        # The randomizations share the shots; the factors each pay in full.
        assert b.total_shots == a.total_shots * 3 * 128 // 100

    def test_the_nested_runtime_schema_is_read(self):
        """``resilience.zne`` and an explicit ``zne_mitigation`` override the
        level default; 32 randomizations split 4096 shots, not multiply them.
        The flat ``zne`` spelling Runtime does not accept used to be read and
        this one ignored: one noise factor and 131,072 shots were reported."""
        options = {"resilience_level": 0,
                   "resilience": {"zne_mitigation": True,
                                  "zne": {"noise_factors": [1, 3, 5, 7, 9]}},
                   "twirling": {"enable_gates": True,
                                "num_randomizations": 32}}
        plan = measurement_plan(QiskitProvider(shots=4096,
                                               estimator_options=options),
                                1, ["Z"])
        assert (plan.noise_factors, plan.twirls, plan.shots_per_twirl) == \
            (5, 32, 128)
        assert plan.total_shots == 5 * 4096

    def test_the_options_dataclass_reads_like_the_dict(self):
        from qiskit_ibm_runtime.options import EstimatorOptions

        options = {"resilience": {"zne_mitigation": True,
                                  "zne": {"noise_factors": [1, 2]}},
                   "twirling": {"enable_gates": False,
                                "enable_measure": False}}
        as_dict = measurement_plan(QiskitProvider(shots=1000,
                                                  estimator_options=options),
                                   1, ["Z"])
        as_object = measurement_plan(
            QiskitProvider(shots=1000,
                           estimator_options=EstimatorOptions(**options)),
            1, ["Z"])
        assert (as_dict.noise_factors, as_dict.twirls) == (2, 1)
        assert (as_object.noise_factors, as_object.twirls) == (2, 1)

    def test_an_explicit_off_beats_the_level(self):
        options = {"resilience_level": 2,
                   "resilience": {"zne_mitigation": False},
                   "twirling": {"enable_gates": False,
                                "enable_measure": False}}
        plan = measurement_plan(QiskitProvider(shots=1000,
                                               estimator_options=options),
                                1, ["Z"])
        assert (plan.noise_factors, plan.twirls, plan.total_shots) == \
            (1, 1, 1000)

    def test_an_unset_level_is_runtimes_default_on_a_processor(self):
        """Level 1 on IBM hardware (measurement twirling); no mitigation on
        the local estimator, which ignores Runtime options."""
        ibm = measurement_plan(QiskitProvider(device="ibm_test", shots=4096),
                               1, ["Z"])
        local = measurement_plan(QiskitProvider(shots=4096), 1, ["Z"])
        assert ibm.twirls == 32 and ibm.noise_factors == 1
        assert local.twirls == 1 and local.total_shots == 4096

    def test_the_budget_refuses_an_oversized_job_before_any_job(self):
        """The refusal precedes the first submission: nothing is queued."""
        for method, options in (("vqe", {}),
                                ("adapt-vqe", {"pool": "qeb",
                                               "max_iterations": 2})):
            calc = Mandacaru(method=method, basis="HAO", grid_spacing=0.30,
                             trace=False, shots=1024,
                             measurement_budget={"observables": 3}, **options)
            with pytest.raises(ValueError, match="observables budget exceeded"):
                calc.get_potential_energy(h2())
            assert calc.provider.run_usage()["jobs"] == 0

    def test_a_bases_ceiling_refuses_too(self):
        calc = Mandacaru(method="vqe", basis="HAO", grid_spacing=0.30,
                         trace=False, shots=1024,
                         measurement_budget={"bases": 1})
        with pytest.raises(ValueError, match="bases budget exceeded"):
            calc.get_potential_energy(h2())
        assert calc.provider.run_usage()["jobs"] == 0

    def test_budget_false_checks_nothing(self):
        plan = MeasurementPlan(n_qubits=4, observables=10 ** 9, bases=10 ** 9,
                               shots_per_basis=1, noise_factors=1, twirls=1)
        plan.check(False)                       # must not raise
        with pytest.raises(MeasurementBudgetError):
            plan.check(None)                    # the defaults do

    def test_an_unknown_budget_key_is_refused(self):
        with pytest.raises(ValueError, match="unknown measurement_budget"):
            resolve_measurement_budget({"observable": 10})
        with pytest.raises(TypeError):
            resolve_measurement_budget(17)

    def test_budget_none_is_the_documented_default(self):
        assert resolve_measurement_budget(None) == DEFAULT_MEASUREMENT_BUDGET

    def test_a_deep_circuit_is_reported_as_noise(self):
        plan = MeasurementPlan(n_qubits=24, observables=1, bases=1,
                               shots_per_basis=1, noise_factors=1, twirls=1,
                               two_qubit_gates=2223)
        assert plan.fidelity < 1e-2
        assert any("noise" in note for note in plan.warnings())

    def test_a_shallow_circuit_is_not(self):
        plan = MeasurementPlan(n_qubits=4, observables=1, bases=1,
                               shots_per_basis=1, noise_factors=1, twirls=1,
                               two_qubit_gates=18)
        assert plan.fidelity > 0.9
        assert plan.warnings() == []


class TestTheIdentityIsCountedOnce:
    """``drop_identity=False`` used to put the constant in a measured group
    *and* return it as the constant: ``2 I + Z`` on ``|0>`` gave 5, not 3."""

    @pytest.mark.parametrize("drop", [True, False])
    def test_either_way(self, drop):
        from mandacaru.backends.measurement import energy_from_group_counts

        h = PauliSum({"I": 2.0, "Z": 1.0}, num_qubits=1)
        groups, constant = qubit_wise_commuting_groups(h, drop_identity=drop)
        counts = [{"0": 10} for _ in groups]
        assert energy_from_group_counts(groups, constant, counts) == \
            pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# 4a. Chunking: several bounded jobs instead of one unbounded PUB.
# --------------------------------------------------------------------------- #

class TestChunkingByMeasurementBasis:

    LABELS = ["XXII", "XIII", "ZZII", "IZZI", "IIXY", "YYII", "IIIZ", "ZIII"]

    def test_a_chunk_is_a_whole_number_of_bases(self):
        from mandacaru.backends.measurement import _greedy_bases

        chunks = chunk_labels_by_basis(self.LABELS, 1)
        # One basis per chunk at limit 1.  The count is this module's own
        # greedy grouping, not `qubit_wise_commuting_groups`: that one orders
        # by coefficient (the energy's terms matter most) and a plan orders by
        # label, so the two agree on the partition but not always on its size.
        assert len(chunks) == len(_greedy_bases(self.LABELS))
        # Every label lands in exactly one chunk, none is lost or duplicated.
        assert sorted(l for chunk in chunks for l in chunk) == sorted(self.LABELS)

    def test_no_limit_is_one_submission(self):
        assert chunk_labels_by_basis(self.LABELS, None) == [self.LABELS]
        assert planned_jobs(self.LABELS, None) == 1

    def test_a_generous_limit_does_not_split(self):
        assert len(chunk_labels_by_basis(self.LABELS, 1000)) == 1

    def test_qpu_usage_sums_over_every_job(self):
        class Job:
            def __init__(self, name):
                self.name = name

            def job_id(self):
                return self.name

            def metrics(self):
                return {"usage": {"quantum_seconds": 2.0, "seconds": 3.0}}

        class Provider:
            device_spec = "ibm_test"
            shots = 100
            jobs = [Job("a"), Job("b"), Job("c")]
            last_job = jobs[-1]

        usage = qpu_usage(Provider(), wall_time_s=1.0)
        assert usage["qpu_jobs"] == 3
        assert usage["qpu_job_ids"] == "a, b, c"
        # Reporting only the last job would have said 2.0 -- a third of the bill.
        assert usage["qpu_seconds"] == pytest.approx(6.0)
        assert usage["qpu_billed_seconds"] == pytest.approx(9.0)


# --------------------------------------------------------------------------- #
# 5. Double factorization: O(M) measurement bases.
# --------------------------------------------------------------------------- #

class TestDoubleFactorization:
    """``H`` as a one-body term plus squares of one-body terms.

    Motta *et al.*, npj Quantum Inf. **7**, 83 (2021); Huggins *et al.*, npj
    Quantum Inf. **7**, 23 (2021).  Each leaf is diagonal in an orbital basis
    of its own, so one measurement circuit per leaf yields every occupation
    correlator it needs -- ``L + 1`` bases instead of ``O(M^3)``.
    """








    def test_a_mismatched_tensor_is_refused(self):
        with pytest.raises(ValueError, match="does not match"):
            double_factorization(np.zeros((4, 4)), np.zeros((3, 3, 3, 3)))
        with pytest.raises(ValueError, match="does not match"):
            spatial_double_factorization(np.zeros((4, 4)),
                                         np.zeros((3, 3, 3, 3)))
        with pytest.raises(ValueError, match="do not match"):
            spatial_double_factorization(np.zeros((4, 4)),
                                         factors=np.zeros((2, 3, 3)))


class TestTheSpatialFactorization:
    """The factorization of spatial integrals (no ``(2M)^4`` tensor) is the
    spin-orbital one: the same operator, weights, one-norm and count."""



    def test_pair_factors_give_the_tensor_factorization(self):
        """From ``k < M^2`` factors and a coupling (Cholesky factors plus an
        augmentation's channels) the leaves come from a ``k x k`` problem,
        with the tensor's result."""
        rng = np.random.default_rng(4)
        m, k = 4, 6
        F = rng.normal(size=(k, m, m))
        F = F + F.transpose(0, 2, 1)
        K = np.diag([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        K[4, 5] = K[5, 4] = 1.0
        K[4, 4] = 0.3
        chemist = np.einsum("ij,ipr,jqs->prqs", K, F, F)
        eri = chemist.transpose(0, 2, 1, 3)               # <pq|rs> = (pr|qs)
        h = rng.normal(size=(m, m))
        h = h + h.T
        dense = spatial_double_factorization(h, eri)
        factored = spatial_double_factorization(h, factors=F, coupling=K)
        assert factored.measurement_bases == dense.measurement_bases
        assert np.allclose(sorted(l.weight for l in factored.leaves),
                           sorted(l.weight for l in dense.leaves), atol=1e-10)
        assert np.allclose(factored.one_body, dense.one_body, atol=1e-12)
        difference = (dense.to_fermion().map_to_qubits()
                      + (-1.0) * factored.to_fermion().map_to_qubits()
                      ).simplify()
        assert max((abs(complex(c)) for c in difference.terms.values()),
                   default=0.0) < 1e-10


# --------------------------------------------------------------------------- #
# 5b. The Givens networks that make the factorized measurement runnable.
# --------------------------------------------------------------------------- #

class TestGivensRotations:
    """Any real orthogonal ``U`` is a product of nearest-neighbor rotations.

    Clements *et al.*, Optica **3**, 1460 (2016); Kivlichan *et al.*, Phys.
    Rev. Lett. **120**, 110501 (2018).  Under Jordan-Wigner each one is a
    two-qubit gate on adjacent wires, so the basis change a factorized
    measurement needs is a circuit an ordinary processor can run.
    """

    @staticmethod
    def orthogonal(m, seed=0):
        q, _r = np.linalg.qr(np.random.default_rng(seed).normal(size=(m, m)))
        return q

    @pytest.mark.parametrize("m", [2, 3, 4, 6])
    def test_the_decomposition_reproduces_the_matrix(self, m):
        from scipy.linalg import expm

        from mandacaru.backends.factorization import givens_decomposition

        unitary = self.orthogonal(m, seed=m)
        rotations, signs = givens_decomposition(unitary)
        assert len(rotations) <= m * (m - 1) // 2
        assert all(abs(abs(s) - 1.0) < 1e-12 for s in signs)

        rebuilt = np.eye(m)
        for p, theta in rotations:
            block = np.zeros((m, m))
            block[p, p + 1], block[p + 1, p] = -theta, theta
            rebuilt = rebuilt @ expm(block)
        assert np.abs(rebuilt @ np.diag(signs) - unitary).max() < 1e-12

    @pytest.mark.parametrize("m", [2, 3, 4])
    def test_every_rotation_is_on_adjacent_modes(self, m):
        from mandacaru.backends.factorization import givens_decomposition

        rotations, _signs = givens_decomposition(self.orthogonal(m, seed=m))
        assert all(0 <= p < m - 1 for p, _theta in rotations)

    def test_a_non_orthogonal_matrix_is_refused(self):
        from mandacaru.backends.factorization import givens_decomposition

        with pytest.raises(ValueError, match="real orthogonal"):
            givens_decomposition(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_the_generator_is_two_commuting_strings(self):
        from mandacaru.backends.factorization import givens_generator

        generator = givens_generator(4, 1)
        labels = [l for l in generator.terms if set(l) != {"I"}]
        assert len(labels) == 2
        # Commuting, which is what lets a provider emit the gate exactly.
        from mandacaru.backends.providers import pauli_rotations

        pauli_rotations(generator, 0.3)          # raises if they do not

    def test_the_circuit_induces_the_right_single_particle_matrix(self):
        r"""The convention, extracted rather than assumed.

        Two reversals cancel in :func:`basis_rotation` -- the elimination order
        against the order a circuit applies its gates -- and getting it wrong
        changes the energy by 0.1 Ha while every individual piece still looks
        right.  So this measures what the circuit *does*: for the product
        :math:`W` the circuit realizes, :math:`W^\dagger a_k W` must be
        :math:`(UD)^T` acting on the ladder operators.
        """
        from scipy.linalg import expm

        from mandacaru.backends.factorization import (basis_rotation,
                                                      givens_decomposition)

        m = 4
        unitary = self.orthogonal(m, seed=7)
        _rotations, signs = givens_decomposition(unitary)
        generators, thetas = basis_rotation(m, unitary)

        annihilators = [
            Fermion({((p, False),): 1.0 + 0j}, n_modes=m)
            .map_to_qubits().to_matrix() for p in range(m)]
        basis = np.stack([a.ravel() for a in annihilators]).T
        # A circuit applies generators[0] first, so the operator product is
        # reversed with respect to the list.
        circuit = np.eye(2 ** m, dtype=complex)
        for generator, theta in zip(generators, thetas):
            circuit = expm(theta * generator.to_matrix()) @ circuit

        induced = np.zeros((m, m), dtype=complex)
        for k in range(m):
            rotated = (circuit.conj().T @ annihilators[k] @ circuit).ravel()
            induced[k], *_ = np.linalg.lstsq(basis, rotated, rcond=None)
        assert np.abs(induced - (unitary @ np.diag(signs)).T).max() < 1e-10



# --------------------------------------------------------------------------- #
# Paid-job controls: reconfiguring a retry, chunked energies, resumed chunks,
# the run-wide budget and one grouping for plan and execution.
# --------------------------------------------------------------------------- #

class TestARetryRunsOnTheOptionsItAsksFor:
    """A retry used to ``setattr`` onto the live provider: ``device=`` set an
    attribute nothing read, and a cached Estimator kept the old resilience
    level.  ``with_options`` rebuilds the provider instead."""

    def test_the_provider_is_rebuilt(self):
        old = QiskitProvider(device="ibm_old", shots=100,
                             estimator_options={"resilience_level": 2})
        old._estimator = object()                        # a cached primitive
        new = old.with_options(device="ibm_new",
                               estimator_options={"resilience_level": 0})
        assert new.device_spec == "ibm_new" and new._estimator is None
        assert new.estimator_options == {"resilience_level": 0}
        assert old.device_spec == "ibm_old"

    def test_an_unknown_option_is_refused(self):
        with pytest.raises(ValueError, match="unknown option"):
            QiskitProvider().with_options(devise="ibm_new")

class TestEnergiesAreChunkedToo:
    """Only ``expectation_values`` honored ``max_bases_per_job``: an energy
    went as one PUB whatever the limit, while the plan reported a split."""

    H = PauliSum({"X": 1.0, "Z": 0.5, "I": 2.0}, num_qubits=1)

    def test_the_limit_splits_the_submission(self, monkeypatch):
        provider = QiskitProvider(max_bases_per_job=1)
        calls = []
        original = provider._run_pubs

        def spy(pubs, bases=None):
            calls.append(len(pubs))
            return original(pubs, bases=bases)

        monkeypatch.setattr(provider, "_run_pubs", spy)
        energy = provider.energies([(1, [], [], [], self.H)])[0]
        assert calls == [1, 1]                   # two bases, two jobs
        assert energy == pytest.approx(2.5)      # 2 + 0.5 <0|Z|0>, <0|X|0> = 0
        assert provider.jobs_per_energy(self.H) == 2

    def test_chunked_equals_unchunked(self):
        state = (2, [0], [PauliSum({"XY": 0.5j, "YX": -0.5j}, num_qubits=2)],
                 [0.4])
        h = PauliSum({"ZI": 1.0, "IZ": -0.4, "XX": 0.3, "YY": 0.3,
                      "II": 0.1}, num_qubits=2)
        whole = build_provider("qiskit").energies([(*state, h)])[0]
        split = build_provider("qiskit", max_bases_per_job=1).energies(
            [(*state, h)])[0]
        assert split == pytest.approx(whole, abs=1e-12)

    def test_the_plan_predicts_the_executed_jobs(self):
        atoms = lih()
        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO",
                               grid_spacing=0.30, pool="qeb", max_iterations=3,
                               trace=False)
        atoms.get_potential_energy()
        problem = atoms.calc.problem
        hamiltonian = PauliSum({label: complex(re, im) for label, re, im
                                in problem.hamiltonian_terms},
                               num_qubits=problem.num_qubits)
        provider = build_provider("qiskit", device="statevector",
                                  max_bases_per_job=2)
        plan = measurement_plan(provider, problem.num_qubits,
                                list(hamiltonian.terms),
                                hamiltonian=hamiltonian,
                                jobs=provider.jobs_per_energy(hamiltonian))
        assert plan.jobs > 1
        assert plan.jobs == provider.jobs_per_energy(hamiltonian)


class TestCompletedChunksAreKept:
    """A failure in chunk two used to discard chunk one's values."""

    def test_a_retry_submits_only_the_missing_chunks(self, monkeypatch):
        provider = QiskitProvider(max_bases_per_job=1)
        labels = ["XI", "ZI", "IX"]
        original = provider._run_pubs
        calls = {"n": 0}

        def second_fails(pubs, bases=None):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("chunk 2 failed")
            return original(pubs, bases=bases)

        monkeypatch.setattr(provider, "_run_pubs", second_fails)
        with pytest.raises(RuntimeError, match="chunk 2"):
            provider.expectation_values(2, [0], [], [], labels)
        assert provider.completed_chunks() == 1
        calls["n"] = 10
        values, _stds = provider.expectation_values(2, [0], [], [], labels)
        assert calls["n"] == 10 + (len(chunk_labels_by_basis(labels, 1)) - 1)
        assert set(values) == set(labels)
        assert provider.completed_chunks() == 0


class TestTheRunBudget:
    """On by default on a processor: an optimization there submits a job per
    objective evaluation, and only a run-wide count keeps it in a quota."""

    def test_it_refuses_before_queuing(self):
        from mandacaru.backends.measurement import RunBudgetExceeded

        provider = QiskitProvider(shots=100, run_budget={"jobs": 2})
        h = PauliSum({"Z": 1.0}, num_qubits=1)
        provider.energy(1, [], [], [], h)
        provider.energy(1, [], [], [], h)
        with pytest.raises(RunBudgetExceeded, match="jobs 3 > 2"):
            provider.energy(1, [], [], [], h)
        assert provider.run_usage()["jobs"] == 2

    def test_it_is_on_by_default_for_hardware_only(self):
        from mandacaru.backends.measurement import (
            DEFAULT_HARDWARE_RUN_BUDGET, resolve_run_budget)

        assert resolve_run_budget(None, hardware=True) == \
            DEFAULT_HARDWARE_RUN_BUDGET
        # Jobs only: a job's size is the measurement budget's to bound.
        assert DEFAULT_HARDWARE_RUN_BUDGET["total_shots"] is None
        assert resolve_run_budget(None, hardware=False) is None
        assert resolve_run_budget(False, hardware=True) is None
        assert QiskitProvider(device="ibm_test").is_hardware
        assert not QiskitProvider(device="fake_fez").is_hardware

    def test_the_ledger_survives_a_rebuild(self):
        provider = QiskitProvider(shots=100)
        provider.energy(1, [], [], [], PauliSum({"Z": 1.0}, num_qubits=1))
        assert provider.with_options(shots=200).run_usage()["jobs"] == 1

    def test_the_shot_ceiling_is_checked_up_front_too(self):
        """Jobs fit, shots do not: refused before the first job, not at the
        job where the ceiling is crossed."""
        from mandacaru.backends.measurement import RunBudgetExceeded

        calc = Mandacaru(method="vqe", basis="HAO", grid_spacing=0.4, shots=1000,
                         trace=False,
                         backend_options={"run_budget": {
                             "jobs": None, "total_shots": 50_000}})
        with pytest.raises(RunBudgetExceeded, match="total_shots"):
            calc.get_potential_energy(h2())
        assert calc.provider.run_usage()["jobs"] == 0

    @pytest.mark.parametrize("method, options", [
        ("vqe", {}), ("adapt-vqe", {"pool": "qeb", "max_iterations": 3})])
    def test_a_run_the_budget_stops_keeps_its_best_point(self, method,
                                                         options):
        """Nelder-Mead has no evaluation bound, so the pre-flight lets it
        start; the ledger stops it at job 6 and the best point is the
        result, flagged unconverged -- not an exception that loses it."""
        calc = Mandacaru(method=method, basis="HAO", grid_spacing=0.4, shots=1000,
                         optimizer={"method": "Nelder-Mead", "maxiter": 200},
                         trace=False,
                         backend_options={"seed": 3,
                                          "run_budget": {"jobs": 6}},
                         **options)
        with pytest.warns(RuntimeWarning, match="run budget"):
            energy = calc.get_potential_energy(h2())
        assert np.isfinite(energy)
        assert calc.provider.run_usage()["jobs"] == 6

    def test_the_ceiling_is_never_crossed(self):
        """Whatever else a stopped run keeps, the ledger refuses the submission
        that would cross the ceiling, so the quota is never overspent."""
        calc = Mandacaru(method="vqe", basis="HAO", grid_spacing=0.4, shots=1000,
                         optimizer={"method": "Nelder-Mead", "maxiter": 200},
                         trace=False,
                         backend_options={"run_budget": {"jobs": 10}})
        with pytest.warns(RuntimeWarning, match="jobs 11 > 10"):
            calc.get_potential_energy(h2())
        assert calc.provider.run_usage()["jobs"] == 10

    def test_an_optimization_that_cannot_fit_is_refused_up_front(self):
        from mandacaru.backends.measurement import RunBudgetExceeded

        calc = Mandacaru(method="vqe", basis="HAO", grid_spacing=0.4, shots=1000,
                         trace=False,
                         backend_options={"run_budget": {"jobs": 10}})
        with pytest.raises(RunBudgetExceeded, match="jobs 201 > 10"):
            calc.get_potential_energy(h2())


class TestOneGroupingForPlanAndExecution:
    """The plan grouped labels lexically and Braket by coefficient: on a
    15-term Hamiltonian that was 8 planned bases for 9 executed ones."""

    def test_the_counts_agree(self):
        rng = np.random.default_rng(7)
        letters = "IXYZ"
        terms = {}
        while len(terms) < 15:
            label = "".join(rng.choice(list(letters), size=3))
            if label != "III":
                terms[label] = float(rng.normal())
        h = PauliSum(terms, num_qubits=3)
        executed, _identity = qubit_wise_commuting_groups(h)
        plan = measurement_plan(QiskitProvider(shots=100), 3, list(terms),
                                hamiltonian=h)
        assert plan.bases == len(executed)
