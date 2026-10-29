"""Quantum SDK transport for the Rust variational solver."""
from types import SimpleNamespace
import numpy as np

from ..backends.hardware import get_device
from ..backends.providers import build_provider, normalize_provider
from ..backends.measurement import resolve_measurement_budget, resilience_multipliers


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


class CircuitExecutor:
    """Translate native Pauli rotations to SDK requests; return raw results."""

    def __init__(self, driver):
        self.driver = driver
        self.problem = driver.problem
        self.provider = driver.provider
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
