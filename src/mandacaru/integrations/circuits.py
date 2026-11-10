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
    needed = {"jobs": usage["jobs"] + evaluations * cost[0],
              "total_shots": usage["shots"] + evaluations * cost[1]}
    over = [f"{key} {needed[key]:,} > {budget[key]:,}" for key in needed
            if budget.get(key) is not None and needed[key] > budget[key]]
    if over:
        sized = {key: needed[key] for key in needed
                 if budget.get(key) is not None and needed[key] > budget[key]}
        raise RunBudgetExceeded(
            f"the optimizer may evaluate the energy {evaluations:,} times on "
            f"{provider!r}, over the run budget ({'; '.join(over)}).  Lower the "
            "optimizer's maxiter (maxfev for COBYLA) or the shots, or pass "
            f"backend_options={{\"run_budget\": {sized!r}}} sized for this run if you mean it.")


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
            paulis = self.problem.pool_paulis
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
                f"{0.67e-3 * per_job:.1f} GB); pass backend_options={{\"max_bases_per_job\": "
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
            for labels, angles, _observables in requests:
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
                               for labels, angles, _observables in requests]}
        known, pending = [], []
        for position, (labels, angles, observables) in enumerate(requests):
            point = (tuple(labels), tuple(angles))
            values = self._values.get(point, {})
            known.append((point, values))
            missing = [label for label in observables if label not in values]
            if missing:
                pending.append((position, labels, angles, missing))
        if pending:
            self._check_budget(
                sum(len(missing) for *_, missing in pending),
                sum(len(self.driver.context.qwc_groups(missing)) for *_, missing in pending))
            measured = self.provider.expectation_batch(
                n_qubits, reference,
                [(*circuit(labels, angles), missing) for _, labels, angles, missing in pending])
            for (position, *_), (values, _errors) in zip(pending, measured):
                known[position][1].update(values)
        self._values = dict(known)
        return {"expectations": [[values[label] for label in observables]
                                 for (_point, values), (_l, _a, observables)
                                 in zip(known, requests)]}
