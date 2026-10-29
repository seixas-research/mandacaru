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
    resolve_shots_per_energy,
)
from ..core import PauliSum


def configure(driver, *, provider, device, shots, options, measurement_budget):
    if type(shots) is not int or shots < 0:
        raise ValueError("shots must be a nonnegative integer")
    selected = get_device(device)
    name = normalize_provider(provider or selected.provider or "qiskit")
    if selected.provider is not None and name != selected.provider:
        raise ValueError("the circuit provider disagrees with the device")
    if (selected.is_fake or not selected.simulator) and shots == 0:
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
    hamiltonian = PauliSum(
        {label: complex(re, im) for label, re, im in driver.problem.hamiltonian_terms},
        num_qubits=driver.problem.num_qubits)
    usage = provider.run_usage()
    needed = {"jobs": usage["jobs"] + evaluations * max(int(provider.jobs_per_energy(hamiltonian)), 1),
              "total_shots": usage["shots"] + evaluations * resolve_shots_per_energy(provider, hamiltonian)}
    over = [f"{key} {needed[key]:,} > {budget[key]:,}" for key in needed
            if budget.get(key) is not None and needed[key] > budget[key]]
    if over:
        raise RunBudgetExceeded(
            f"the optimizer may evaluate the energy {evaluations:,} times on "
            f"{provider!r}, over the run budget ({'; '.join(over)}).  Lower the "
            "optimizer's maxiter or the shots, or raise run_budget= in "
            "backend_options if you mean it.")


class CircuitExecutor:
    """Translate native Pauli rotations to SDK requests; return raw results.

    A submission the run budget refuses is not an error to the solver: the
    executor answers ``{"stop": message}`` and the solver ends at the best
    point it has reached.
    """

    def __init__(self, driver):
        self.driver = driver
        self.problem = driver.problem
        self.provider = driver.provider
        self.stopped = None
        self._point = None
        self._values = {}
        self._state = None
        if driver._shots and driver.method.endswith("adapt-vqe"):
            observables = self.problem.pool_observables
            labels = sorted({label for terms in observables for label, _, _ in terms
                             if label.strip("I")})
            groups = driver.context.qwc_groups(labels)
            self._check_budget(labels, groups)
            limit = getattr(self.provider, "max_bases_per_job", None)
            jobs = len(groups) if self.provider.name == "braket" else (
                (len(groups) + limit - 1) // limit if limit else bool(groups))
            factors, twirls, per_twirl = resilience_multipliers(self.provider)
            driver._gradient_measurement = {
                "scheme": "native commutators [H, A_k], merged QWC measurement bases",
                "first_step": "classical (exact at the reference determinant)",
                "measured_operators": sum(bool(terms) for terms in observables),
                "strings_per_step": len(labels),
                "bases_per_step": len(groups),
                "jobs_per_step": int(jobs),
                "shots_per_step": len(groups) * factors * (
                    twirls * per_twirl if twirls > 1 else driver._shots),
            }

    def _check_budget(self, labels, groups):
        for key, count in (("observables", len(labels)), ("bases", len(groups))):
            limit = self.driver._measurement_budget.get(key)
            if limit is not None and count > limit:
                raise ValueError(f"measurement {key} budget exceeded: {count} > {limit}")

    def __call__(self, labels, angles, observables):
        if self.stopped is not None:
            return {"stop": self.stopped}
        try:
            return self._execute(labels, angles, observables)
        except RunBudgetExceeded as exc:
            self.stopped = str(exc)
            return {"stop": self.stopped}

    def _execute(self, labels, angles, observables):
        point = (tuple(labels), tuple(angles))
        if point != self._point:
            self._point, self._values, self._state = point, {}, None
        generators = [SimpleNamespace(terms={label: 1j}) for label in labels]
        args = (self.problem.num_qubits, self.problem.reference_qubits,
                generators, angles)
        if not self.driver._shots:
            if self._state is None:
                self._state = np.asarray(self.provider.statevector(*args), dtype=complex)
            return {"state": self._state}
        missing = [label for label in observables if label not in self._values]
        if missing:
            self._check_budget(missing, self.driver.context.qwc_groups(missing))
            values, _errors = self.provider.expectation_values(*args, missing)
            self._values.update(values)
        return {"expectations": [self._values[label] for label in observables]}
