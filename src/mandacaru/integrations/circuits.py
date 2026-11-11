"""Quantum SDK transport for the Rust variational solver."""
from types import SimpleNamespace
import numpy as np

from .. import _native
from ..backends.hardware import get_device
from ..backends.providers import build_provider, normalize_provider
from ..backends.measurement import (
    RunBudgetExceeded,
    resilience_multipliers,
    resolve_measurement_budget,
    resolve_run_budget,
)
from . import measurement_scheme



#: Largest number of measurement bases one MPS-simulator job may hold (about 0.7 GB).
MPS_BASES_PER_JOB = 1000

def budget_refusal(key, count, limit):
    """The ``ValueError`` of a ``measurement_budget`` ceiling: the numbers, and the option that
    lifts it, sized for this run."""
    return ValueError(
        f"measurement {key} budget exceeded: {count} > {limit}; the default ceilings are "
        f"sized for small molecules, so a large run passes its own: "
        f"pass measurement_budget={{\"{key}\": N}} sized for this run (here N >= {count})")


def pool_refusal(pool, size, over, limit):
    """The ``ValueError`` of an ADAPT pool whose gradient step cannot fit the observable budget,
    raised after the native walk found the budget crossed (a lower bound: the pool was not
    walked to its end), with the three ways out."""
    return ValueError(
        f"the {pool!r} pool has {size:,} generators and measuring its gradients needs more than "
        f"{limit:,} Pauli observables (the first {over['generators']:,} commutators already hold "
        f"{over['strings']:,} distinct strings): measurement observables budget exceeded; the "
        f"default ceilings are sized for small molecules, so choose a smaller pool, "
        f"pool=\"fermionic\" or \"qeb\", or pass measurement_budget={{\"observables\": N}} "
        f"sized for this run (N above {over['strings']:,}; the full count is known only once "
        f"the whole pool is planned)")


def configure(driver, *, provider, device, shots, options, measurement_budget):
    if type(shots) is not int or shots < 0:
        raise ValueError("shots must be a nonnegative integer")
    selected = get_device(device)
    name = normalize_provider(provider or selected.provider or "qiskit")
    if selected.provider is not None and name != selected.provider:
        raise ValueError("the circuit provider disagrees with the device")
    if (selected.is_fake or not selected.simulator or selected.shots_only) and shots == 0:
        raise ValueError("this device requires shots > 0")
    options = dict(options or {})
    if name != "cirq":
        options.update(shots=shots)
        if name == "braket":
            options["device"] = selected.arn or "braket_sv"
        elif selected.provider is not None:
            options["device"] = selected.name
    elif shots:
        raise ValueError("Cirq currently supports exact circuit execution")
    driver.provider = build_provider(name, context=driver.context, **options)
    driver._measurement_budget = resolve_measurement_budget(measurement_budget)
    driver._shots = shots
    driver._gradient_measurement = None
    driver._energy_measurement = None


def qwc_measurement_cost(driver, bases):
    """``(jobs, shots)`` of one batch of ``bases`` commuting groups on the driver's provider, as
    :meth:`CircuitExecutor` submits it: whole groups packed ``max_bases_per_job`` to a job (one
    job unbounded), one task per basis on Braket."""
    provider = driver.provider
    factors, twirls, per_twirl = resilience_multipliers(provider)
    return driver.context.measurement_cost(
        bases, max_bases_per_job=getattr(provider, "max_bases_per_job", None) or None,
        job_per_basis=provider.name == "braket", shots=int(driver._shots),
        noise_factors=factors, randomizations=twirls, shots_per_randomization=per_twirl)


def energy_bases(driver):
    """The measurement bases of one energy: the factorized plan's ``L + 1``, or the commuting
    groups of the Hamiltonian's strings, grouped exactly as the executor groups them."""
    plan = driver.problem.factorized_measurement
    if plan is not None:
        return plan["bases"]
    labels = [label for label, _re, _im in driver.problem.hamiltonian_terms]
    return len(driver.context.qwc_groups(labels))


def energy_measurement_cost(driver):
    """``(jobs, shots)`` of one energy evaluation of a shot-based run: what the preflight checks
    against the run budget and what the run log reports. Both schemes are batches of whole bases
    (`measurement_scheme.energy_cost` prices the factorized one the same way)."""
    cost = measurement_scheme.energy_cost(driver)
    if cost is None:
        cost = qwc_measurement_cost(driver, energy_bases(driver))
    return cost


def preflight_run_budget(driver, opts):
    """Refuse an optimization that cannot fit the provider's run budget.

    Called before the first job is queued, so a refusal costs nothing. A fixed
    ansatz has a known cap on its energy evaluations (``None`` when the
    optimizer is not capped by its iteration count, e.g. Nelder-Mead); that
    many evaluations are checked against the jobs and shots left. An unbounded
    method is left to the provider's per-job check, which stops the run where
    the budget ends.
    """
    if not driver._shots or driver.method != "vqe":
        return
    method = opts["optimizer"]
    if method == "Nelder-Mead":
        return      # not capped by its iteration count: the ledger stops it
    # COBYLA's intended budget is its iteration count unless `maxfev` says more.
    cap = opts["max_evaluations"]
    if method == "COBYLA" and "maxfev" not in driver._optimizer:
        cap = opts["max_iterations"]
    evaluations = _native.optimizer_evaluation_bound(
        method, opts["max_iterations"], cap, opts["track_best"])
    if evaluations is None:
        return
    provider = driver.provider
    budget = resolve_run_budget(provider.run_budget, provider.is_hardware)
    if budget is None:
        return
    cost = energy_measurement_cost(driver)
    usage = provider.run_usage()
    _after, exceeded = _native.budget_check(
        (usage["jobs"], usage["shots"]), tuple(cost), evaluations,
        job_limit=budget.get("jobs"), shot_limit=budget.get("total_shots"))
    over = [f"{key} {total:,} > {limit:,}" for key, total, limit in exceeded]
    if over:
        sized = {key: total for key, total, _limit in exceeded}
        raise RunBudgetExceeded(
            f"the optimizer may evaluate the energy {evaluations:,} times on "
            f"{provider!r}, over the run budget ({'; '.join(over)}).  Lower the "
            "optimizer's maxiter (maxfev for COBYLA) or the shots, or pass "
            f"backend_options={{\"run_budget\": {sized!r}}} sized for this run if you mean it.")


def measure_rdms(driver):
    """The spin-summed spatial 1- and 2-RDMs of the optimized circuit, measured on the device.

    Rust lists the Pauli strings of the RDM elements (``QuantumProblem.rdm_measurement``,
    refused past ``measurement_budget["observables"]``), groups them qubit-wise by their X/Y
    pattern (polynomial) and builds the ansatz circuit; each basis is one request, rotated so its
    strings are read as Z strings, and all of them go in one ``expectation_batch`` (one job on a
    Qiskit device, under the provider's run budget). Rust assembles the matrices. No state vector is
    formed. Conventions: ``one[p, q] = sum_s <a+_ps a_qs>``, ``two[p, q, r, s] =
    sum_st <a+_ps a+_qt a_st a_rs>`` over the active spatial orbitals (the index order of the
    model's two-electron tensor), real parts; the errors propagate the device's standard errors
    with the covariances of strings read in one basis on the same shots (each pair's product
    string is measured in that basis), and are ``None`` when the device reports none.
    """
    if driver.result is None or driver.problem is None:
        raise ValueError("measured RDMs need a finished variational run: call get_potential_energy() first")
    if not driver._execute_circuits:
        raise ValueError(
            "measured RDMs need a circuit device (shots > 0 or execute_circuits=True); a "
            "state-vector run reads its density matrices from the state (natural_orbitals())")
    if driver.method not in ("vqe", "adapt-vqe"):
        raise NotImplementedError(
            f"measured RDMs are implemented for vqe and adapt-vqe, not {driver.method!r}")
    problem = driver.problem
    raw = driver.result._raw
    labels, angles = problem.circuit([int(i) for i in raw.get("indices", ())],
                                     [float(t) for t in raw.get("parameters", ())])
    return measure_rdms_of(driver, [(labels, angles)])[0]


def rdm_plan(driver):
    """The RDM measurement plan of the driver's problem, its string and basis counts checked
    against ``measurement_budget`` before any job."""
    plan = driver.problem.rdm_measurement(driver._measurement_budget.get("observables"))
    for key, count in (("observables", len(plan.observables)), ("bases", plan.bases)):
        limit = driver._measurement_budget.get(key)
        if limit is not None and count > limit:
            raise budget_refusal(key, count, limit)
    return plan


def measure_rdms_of(driver, circuits):
    """The measured RDMs (see :func:`measure_rdms`) of every circuit ``(labels, angles)`` of
    ``circuits``, all in one ``expectation_batch`` (one job): the plan's bases times the circuits."""
    problem = driver.problem
    plan = rdm_plan(driver)
    # Rust groups the strings (polynomial, by their X/Y pattern) and rotates each basis to Z:
    # one request per basis, so the SDK reads Z strings and never regroups a million labels.
    strings, bases = len(plan.observables), plan.bases
    requests, readers = [], []
    for labels, angles in circuits:
        for tail, observables in plan.requests():
            circuit = [SimpleNamespace(terms={label: 1j}) for label in labels]
            circuit += [SimpleNamespace(terms={label: 1j}) for label, _angle in tail]
            requests.append((circuit, list(angles) + [angle for _label, angle in tail],
                             [label for label, _index in observables]))
            readers.append(observables)
    measured = driver.provider.expectation_batch(
        problem.num_qubits, problem.reference_qubits, requests)
    results = []
    per_circuit = len(readers) // max(len(circuits), 1)
    for c in range(len(circuits)):
        values, errors = [0.0] * plan.measured, [0.0] * plan.measured
        reported = True
        chunk = slice(c * per_circuit, (c + 1) * per_circuit)
        for observables, (read, spread) in zip(readers[chunk], measured[chunk]):
            reported = reported and spread is not None
            for label, index in observables:
                values[index] = read[label]
                if spread is not None:
                    errors[index] = spread[label]
        one, two, one_error, two_error = plan.assemble(
            values, errors if reported else None, int(driver._shots) or None)
        results.append({"one": one, "two": two, "one_error": one_error, "two_error": two_error,
                        "observables": strings, "bases": bases})
    return results


class CircuitExecutor:
    """Translate native Pauli rotations to SDK requests; return raw results.

    The solver hands over a **batch** of circuits per measurement (one for the commuting-group
    energy, one per basis for the double-factorized energy, one for the pool gradients), and the
    batch is one ``provider.expectation_batch`` call: one job on a Qiskit device.

    A submission the run budget refuses is not an error to the solver: the
    executor answers ``{"stop": message}`` and the solver ends at the best
    point it has reached.
    """

    def __init__(self, driver):
        self.driver = driver
        self.problem = driver.problem
        self.provider = driver.provider
        #: Shots per basis (0 on an exact device): Rust measures a sampled energy's standard
        #: error from them.
        self.shots = int(driver._shots)
        self.stopped = None
        # The measurements of the previous call, by circuit: a call at a point already measured
        # asks the device only for the strings it lacks. Replaced at every call.
        self._values = {}
        self._states = {}
        plan = driver.problem.factorized_measurement
        if driver._shots and plan is not None:
            # The energy is measured in `bases` bases, each a commuting group of its own,
            # in one batch: `__call__` checks the observables of the batch as it is asked for.
            limit = driver._measurement_budget.get("bases")
            if limit is not None and plan["bases"] > limit:
                raise budget_refusal("bases", plan["bases"], limit)
        if driver._shots:
            jobs, shots = energy_measurement_cost(driver)
            driver._energy_measurement = {
                "scheme": ("double-factorized (one basis per leaf, plus the one-body basis)"
                           if plan is not None else "merged QWC measurement bases"),
                "bases_per_energy": energy_bases(driver),
                "jobs_per_energy": jobs,
                "shots_per_energy": shots,
            }
            self._check_mps_job(driver._energy_measurement["bases_per_energy"])
        if driver._shots and driver.method.endswith("adapt-vqe"):
            # The walk stops once the distinct strings pass the budget: an oversized pool is
            # refused after one batch of commutators, before the gradient plan is built.
            limit = driver._measurement_budget.get("observables")
            paulis = self.problem.pool_paulis(limit)
            if paulis["over"] is not None:
                raise pool_refusal(driver._pool, paulis["pool_size"], paulis["over"], limit)
            labels = paulis["labels"]
            groups = driver.context.qwc_groups(labels)
            self._check_budget(len(labels), len(groups))
            self._check_mps_job(len(groups))
            jobs, shots = qwc_measurement_cost(driver, len(groups))
            driver._gradient_measurement = {
                "scheme": "native commutators [H, A_k], merged QWC measurement bases",
                "first_step": "classical (exact at the reference determinant)",
                "measured_operators": paulis["measured_operators"],
                "strings_per_step": len(labels),
                "bases_per_step": len(groups),
                "jobs_per_step": jobs,
                "shots_per_step": shots,
            }

    def _check_mps_job(self, bases):
        """Refuse an MPS simulation whose one job would hold more bases than this machine can.

        The MPS simulator holds every basis of a job at once: measured 2026-11-09, about 0.67 MB
        per basis (a 13,000-basis gradient job of a 50-qubit register took 8.7 GB), against
        0.35 GB at 100 bases per job.
        """
        if not getattr(self.provider, "is_mps_device", False):
            return
        per_job = self.provider.max_bases_per_job or bases
        if per_job > MPS_BASES_PER_JOB:
            raise ValueError(
                f"the MPS simulator would hold {per_job:,} measurement bases in one job (about "
                f"{_native.mps_job_gigabytes(per_job):.1f} GB); pass backend_options={{\"max_bases_per_job\": "
                f"{MPS_BASES_PER_JOB}}} or fewer")

    def _check_budget(self, observables, bases):
        for key, count in (("observables", observables), ("bases", bases)):
            limit = self.driver._measurement_budget.get(key)
            if limit is not None and count > limit:
                raise budget_refusal(key, count, limit)

    def __call__(self, requests):
        if self.stopped is not None:
            return {"stop": self.stopped}
        try:
            return self._execute(requests)
        except RunBudgetExceeded as exc:
            self.stopped = str(exc)
            return {"stop": self.stopped}

    def _execute(self, requests):
        n_qubits, reference = self.problem.num_qubits, self.problem.reference_qubits

        def circuit(labels, angles):
            return [SimpleNamespace(terms={label: 1j}) for label in labels], angles

        if not self.driver._shots:
            states = {}
            for labels, angles, _observables, _priced in requests:
                point = (tuple(labels), tuple(angles))
                if point in states:
                    continue
                state = self._states.get(point)
                if state is None:
                    state = np.asarray(self.provider.statevector(
                        n_qubits, reference, *circuit(labels, angles)), dtype=complex)
                states[point] = state
            self._states = states
            return {"states": [states[(tuple(labels), tuple(angles))]
                               for labels, angles, _observables, _priced in requests]}
        known, pending, priced = [], [], []
        for position, (labels, angles, observables, count) in enumerate(requests):
            point = (tuple(labels), tuple(angles))
            values = self._values.get(point, {})
            known.append((point, values))
            missing = [label for label in observables if label not in values]
            if missing:
                pending.append((position, labels, angles, missing))
                # The budget prices what the device is asked for: the leading `count` strings.
                # The covariance products after them ride on the same shots of the same basis.
                priced.append([label for label in observables[:count] if label not in values])
        if pending:
            self._check_budget(*self.driver.context.batch_size(priced))
            measured = self.provider.expectation_batch(
                n_qubits, reference,
                [(*circuit(labels, angles), missing) for _, labels, angles, missing in pending])
            for (position, *_), (values, _errors) in zip(pending, measured):
                known[position][1].update(values)
        self._values = dict(known)
        return {"expectations": [[values[label] for label in observables]
                                 for (_point, values), (_l, _a, observables, _p)
                                 in zip(known, requests)]}
