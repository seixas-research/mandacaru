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
* a budget refuses an oversized job *before* it is queued;
* observables are chunked into jobs of bounded size, and completed chunks are
  kept when a later one fails;
* the run-wide budget stops a paid run before it crosses its ceiling.

The double factorization that needs ``O(M)`` bases where qubit-wise commuting
grouping needs ``O(M^3)`` is tested natively in
``test/migration/test_native_factorization.py``.
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.backends.measurement import (DEFAULT_MEASUREMENT_BUDGET,
                                            chunk_labels_by_basis,
                                            qubit_wise_commuting_groups,
                                            qwc_partition,
                                            resilience_multipliers,
                                            resolve_measurement_budget)
from mandacaru.backends.providers import QiskitProvider, qpu_usage
from mandacaru.core.mapping import PauliSum

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
        # ZNE's three noise factors and the twirling randomizations are
        # circuits the processor runs.  At 100 shots Runtime's "auto" rule
        # takes 64 shots per randomization, so two randomizations.
        assert resilience_multipliers(plain) == (1, 1, 100)
        assert resilience_multipliers(mitigated) == (3, 2, 64)

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
        provider = QiskitProvider(shots=4096, estimator_options=options)
        assert resilience_multipliers(provider) == (5, 32, 128)

    def test_the_options_dataclass_reads_like_the_dict(self):
        from qiskit_ibm_runtime.options import EstimatorOptions

        options = {"resilience": {"zne_mitigation": True,
                                  "zne": {"noise_factors": [1, 2]}},
                   "twirling": {"enable_gates": False,
                                "enable_measure": False}}
        as_dict = resilience_multipliers(
            QiskitProvider(shots=1000, estimator_options=options))
        as_object = resilience_multipliers(
            QiskitProvider(shots=1000,
                           estimator_options=EstimatorOptions(**options)))
        assert as_dict[:2] == (2, 1)
        assert as_object[:2] == (2, 1)

    def test_an_explicit_off_beats_the_level(self):
        options = {"resilience_level": 2,
                   "resilience": {"zne_mitigation": False},
                   "twirling": {"enable_gates": False,
                                "enable_measure": False}}
        provider = QiskitProvider(shots=1000, estimator_options=options)
        assert resilience_multipliers(provider) == (1, 1, 1000)

    def test_an_unset_level_is_runtimes_default_on_a_processor(self):
        """Level 1 on IBM hardware (measurement twirling); no mitigation on
        the local estimator, which ignores Runtime options."""
        ibm = resilience_multipliers(QiskitProvider(device="ibm_test",
                                                    shots=4096))
        local = resilience_multipliers(QiskitProvider(shots=4096))
        assert ibm[:2] == (1, 32)
        assert local == (1, 1, 4096)

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

    def test_a_refusal_names_the_option_and_the_size_to_pass(self):
        """The default ceilings stay; a large run is told what to pass instead."""
        import re

        calc = Mandacaru(method="vqe", basis="HAO", grid_spacing=0.30,
                         trace=False, shots=1024,
                         measurement_budget={"bases": 1})
        with pytest.raises(ValueError) as refused:
            calc.get_potential_energy(h2())
        message = str(refused.value)
        found = re.search(r"bases budget exceeded: (\d+) > 1;", message)
        assert found and int(found.group(1)) > 1
        assert (f'pass measurement_budget={{"bases": N}} sized for this run '
                f"(here N >= {found.group(1)})") in message
        calc = Mandacaru(method="vqe", basis="HAO", grid_spacing=0.30,
                         trace=False, shots=1024,
                         measurement_budget={"observables": 3})
        with pytest.raises(ValueError, match=r'measurement_budget=\{"observables": N\}'):
            calc.get_potential_energy(h2())

    def test_a_run_budget_refusal_names_the_option_and_the_size_to_pass(self):
        from mandacaru.backends.measurement import RunBudgetExceeded

        provider = QiskitProvider(shots=100, run_budget={"jobs": 1})
        provider.expectation_values(1, [], [], [], ["Z"])
        with pytest.raises(RunBudgetExceeded) as refused:
            provider.expectation_values(1, [], [], [], ["Z"])
        assert 'backend_options={"run_budget": {\'jobs\': 2}}' in str(refused.value)

    def test_budget_false_checks_nothing(self):
        assert resolve_measurement_budget(False) == dict.fromkeys(
            DEFAULT_MEASUREMENT_BUDGET)

    def test_an_unknown_budget_key_is_refused(self):
        with pytest.raises(ValueError, match="unknown measurement_budget"):
            resolve_measurement_budget({"observable": 10})
        with pytest.raises(TypeError):
            resolve_measurement_budget(17)

    def test_budget_none_is_the_documented_default(self):
        assert resolve_measurement_budget(None) == DEFAULT_MEASUREMENT_BUDGET


class TestTheIdentityIsCountedOnce:
    """``drop_identity=False`` used to put the constant in a measured group
    *and* return it as the constant: ``2 I + Z`` on ``|0>`` gave 5, not 3."""

    @pytest.mark.parametrize("drop", [True, False])
    def test_either_way(self, drop):
        h = PauliSum({"I": 2.0, "Z": 1.0}, num_qubits=1)
        groups, constant = qubit_wise_commuting_groups(h, drop_identity=drop)
        measured = sum(complex(c).real for _basis, payload in groups
                       for label, c in payload if set(label) == {"I"})
        assert constant + measured == pytest.approx(2.0)
        assert (measured == 0.0) is drop


# --------------------------------------------------------------------------- #
# 4a. Chunking: several bounded jobs instead of one unbounded PUB.
# --------------------------------------------------------------------------- #

class TestChunkingByMeasurementBasis:

    LABELS = ["XXII", "XIII", "ZZII", "IZZI", "IIXY", "YYII", "IIIZ", "ZIII"]

    def test_a_chunk_is_a_whole_number_of_bases(self):
        chunks = chunk_labels_by_basis(self.LABELS, 1)
        # One basis per chunk at limit 1.  The count is the lexical
        # partition's, not `qubit_wise_commuting_groups`: that one orders by
        # coefficient (the energy's terms matter most), so the two agree on
        # the partition but not always on its size.
        assert len(chunks) == len(qwc_partition(self.LABELS))
        # Every label lands in exactly one chunk, none is lost or duplicated.
        assert sorted(l for chunk in chunks for l in chunk) == sorted(self.LABELS)

    def test_no_limit_is_one_submission(self):
        assert chunk_labels_by_basis(self.LABELS, None) == [self.LABELS]

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


class TestSeveralCircuitsRideInOneJob:
    """``expectation_batch``: a measurement that needs several circuits (the double-factorized
    energy has one per basis) is one submission, not one per circuit."""

    REQUESTS = [
        ([PauliSum({"XY": 0.5j, "YX": -0.5j}, num_qubits=2)], [0.4], ["ZI", "IZ", "XX"]),
        ([PauliSum({"XY": 0.5j, "YX": -0.5j}, num_qubits=2)], [-0.9], ["ZZ", "YY"]),
        ([], [], ["ZI"]),
    ]

    def test_the_circuits_are_one_job_with_the_values_one_by_one_would_give(self, monkeypatch):
        provider = QiskitProvider()
        jobs = []
        original = provider._run_pubs

        def spy(pubs, bases=None):
            jobs.append((len(pubs), bases))
            return original(pubs, bases=bases)

        monkeypatch.setattr(provider, "_run_pubs", spy)
        batch = provider.expectation_batch(2, [0], self.REQUESTS)
        assert len(jobs) == 1 and jobs[0][0] == len(self.REQUESTS)
        assert provider.run_usage()["jobs"] == 1
        for (generators, thetas, labels), (values, _stds) in zip(self.REQUESTS, batch):
            alone, _ = QiskitProvider().expectation_values(2, [0], generators, thetas, labels)
            assert set(values) == set(labels)
            for label in labels:
                assert values[label] == pytest.approx(alone[label], abs=1e-12)

    def test_a_basis_limit_packs_the_circuits_into_jobs_of_at_most_that_many_bases(
            self, monkeypatch):
        provider = QiskitProvider(max_bases_per_job=2)
        jobs = []
        original = provider._run_pubs

        def spy(pubs, bases=None):
            jobs.append(bases)
            return original(pubs, bases=bases)

        monkeypatch.setattr(provider, "_run_pubs", spy)
        requests = [([], [], ["ZI"]), ([], [], ["IZ"]), ([], [], ["ZZ"]), ([], [], ["ZI", "IZ"])]
        batch = provider.expectation_batch(2, [0], requests)
        assert jobs == [2, 2]                    # four one-basis circuits, two bases to a job
        assert all(len(values) == len(r[2]) for (values, _), r in zip(batch, requests))
        assert provider.run_usage()["jobs"] == 2

    def test_a_failed_job_keeps_what_the_earlier_ones_measured(self, monkeypatch):
        provider = QiskitProvider(max_bases_per_job=1)
        requests = [([], [], ["ZI"]), ([], [], ["XI"])]
        original = provider._run_pubs
        calls = {"n": 0}

        def second_fails(pubs, bases=None):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("job 2 failed")
            return original(pubs, bases=bases)

        monkeypatch.setattr(provider, "_run_pubs", second_fails)
        with pytest.raises(RuntimeError, match="job 2"):
            provider.expectation_batch(2, [0], requests)
        assert provider.completed_chunks() == 1
        calls["n"] = 10
        batch = provider.expectation_batch(2, [0], requests)
        assert calls["n"] == 11                  # only the missing job is submitted
        assert [set(values) for values, _ in batch] == [{"ZI"}, {"XI"}]
        assert provider.completed_chunks() == 0


class TestTheRunBudget:
    """On by default on a processor: an optimization there submits a job per
    objective evaluation, and only a run-wide count keeps it in a quota."""

    def test_it_refuses_before_queuing(self):
        from mandacaru.backends.measurement import RunBudgetExceeded

        provider = QiskitProvider(shots=100, run_budget={"jobs": 2})
        provider.expectation_values(1, [], [], [], ["Z"])
        provider.expectation_values(1, [], [], [], ["Z"])
        with pytest.raises(RunBudgetExceeded, match="jobs 3 > 2"):
            provider.expectation_values(1, [], [], [], ["Z"])
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
        provider.expectation_values(1, [], [], [], ["Z"])
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
