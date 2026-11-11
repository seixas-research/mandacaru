"""Forces of a variational state on a circuit device (owner, 2026-11-10: "Measure the Hessian",
"Exact, limit 256").

The force is the analytic one of ``response.rs`` evaluated on the density matrices measured on
the device. The parameter gradient and Hessian are measured with the shift rule of the
circuit's Pauli rotations (``QuantumProblem.device_hessian``: ``1 + 2 R^2`` energy evaluations
for ``R`` rotations, one batch), ``H z = g`` is solved natively, and the Z-vector correction is
applied in full: ``z . g(R)`` is the shift-rule sum of ``2 R`` shifted circuits' energies, so their
measured RDMs (one more batch, with the state's own) turn it into a density. Every circuit is
planned and checked against ``measurement_budget["circuit_instances"]`` and the run budget
before the first job. The statistical error is propagated exactly from the measured RDM errors
(``MeanField.device_gradient``, up to 256 unique elements) and, on a sampled device, from the
standard errors of the Hessian job's energies through ``z`` (``Cov(z)`` from
``QuantumProblem.device_hessian``; a Hessian eigenvalue within five of its standard errors of
zero refuses by name). When a measurement basis needs more covariance products than the
classical limit (``energy_error``), the Hessian energies carry no standard error: the force is
still measured and returned, its error is ``None`` with the reason (owner, 2026-11-10: "Run,
error None"), and the five-sigma curvature check, which needs those errors, is skipped and said so
once. No state vector is formed.
"""
import numpy as np

from .. import _native
from ..backends.measurement import RunBudgetExceeded, resolve_run_budget
from .circuits import CircuitExecutor, measure_rdms_of, rdm_plan

#: What the statistical error leaves out, written once into ``force_details``.
EXCLUDED = ("the covariance between different RDM elements measured on the same circuit (the RDM "
            "measurement reports each element's error with its in-basis covariances only), and "
            "the second-order terms of the propagation (the product of the Hessian and RDM noises, "
            "and the (sigma/lambda)^2 bias of 1/lambda, under 4 percent at the five-sigma "
            "curvature threshold)")


def _flat(rdm, errors=True):
    def flat(value):
        return None if value is None else np.ascontiguousarray(value, dtype=float).ravel()
    if not errors:
        return flat(rdm["one"]), flat(rdm["two"]), None, None
    return flat(rdm["one"]), flat(rdm["two"]), flat(rdm["one_error"]), flat(rdm["two_error"])


def _refuse_over_budget(driver, rotations, evaluations, circuits, rdm_circuits, bases):
    instances = circuits + rdm_circuits * bases
    what = (f"{instances:,} circuits: {evaluations:,} energy evaluations ({circuits:,} circuits) "
            f"for the parameter Hessian of {rotations} Pauli rotations, and {rdm_circuits} RDM "
            f"circuits of {bases} bases each")
    limit = driver._measurement_budget.get("circuit_instances")
    if limit is not None and instances > limit:
        raise ValueError(
            f"the device force needs {what}, over measurement_budget['circuit_instances'] = "
            f"{limit:,}; pass measurement_budget={{\"circuit_instances\": N}} sized for this run "
            f"(here N >= {instances}) if you mean it")
    provider = driver.provider
    budget = resolve_run_budget(provider.run_budget, provider.is_hardware)
    if budget is not None and driver._shots:
        usage = provider.run_usage()
        _after, exceeded = _native.budget_check(
            (usage["jobs"], usage["shots"]), (2, instances * int(driver._shots)), 1,
            job_limit=budget.get("jobs"), shot_limit=budget.get("total_shots"))
        if exceeded:
            over = "; ".join(f"{key} {total:,} > {limit:,}" for key, total, limit in exceeded)
            raise RunBudgetExceeded(
                f"the device force needs {what} in two jobs, over the run budget ({over}); "
                "pass backend_options={\"run_budget\": ...} sized for this run if you mean it")
    return instances


def device_force(result, driver, internal):
    """``(gradient, details)`` of the device state: the gradient in Hartree per Bohr."""
    if driver.method not in ("vqe", "adapt-vqe"):
        raise NotImplementedError(
            f"device forces are implemented for vqe and adapt-vqe, not {driver.method!r}")
    problem, reference, model = result._problem, result._reference, result._model
    raw = result._raw
    indices = [int(i) for i in raw.get("indices", ())]
    parameters = [float(t) for t in raw.get("parameters", ())]
    rotations, evaluations, circuits = problem.device_hessian_plan(
        indices, parameters, sampled=bool(driver._shots))
    plan = rdm_plan(driver)
    rdm_circuits = 1 + 2 * rotations
    instances = _refuse_over_budget(driver, rotations, evaluations, circuits, rdm_circuits, plan.bases)
    executor = CircuitExecutor(driver)
    measured = problem.device_hessian(indices, parameters, executor)
    if executor.stopped is not None:
        raise RunBudgetExceeded(executor.stopped)
    z = measured["z"]
    # Past the product limit the Hessian has no error, so the force has none: nothing to propagate.
    unavailable = measured.get("hessian_error_unavailable")
    errors = unavailable is None
    shifted = [] if z is None else problem.shifted_circuits(
        indices, parameters, [float(v) for v in z])
    base_circuit = problem.circuit(indices, parameters)
    rdms = measure_rdms_of(driver, [base_circuit] + [(labels, angles) for labels, angles, *_k in shifted])
    out = reference.device_gradient(
        model, *_flat(rdms[0], errors),
        z=None if z is None else np.ascontiguousarray(z, dtype=float),
        z_covariance=measured.get("z_covariance"),
        shifted=[(k, coefficient, *_flat(rdm, errors))
                 for (_l, _a, k, coefficient), rdm in zip(shifted, rdms[1:])],
        internal=internal)
    details = {
        "scheme": "analytic",
        "orbital_response": "CPHF-and-natural-orbital-Lagrangian",
        "density_matrices": "measured on the device",
        "stationarity": "stationary" if z is None else "lagrangian",
        "parameter_gradient_norm": measured["gradient_norm"],
        "hessian_rule": "Pauli-rotation shift rule, +-pi/4 (exact)",
        "hessian_rotations": rotations,
        "hessian_energy_evaluations": evaluations,
        "hessian_circuits": circuits,
        "rdm_circuits": len(rdms),
        "device_circuits": instances,
        "directions": out["directions"],
        "reference_stationarity": out["reference_stationarity"],
        "cphf_residual": out["cphf_residual"],
    }
    if errors:
        details.update({
            "gradient_error_ha_per_bohr": out["gradient_error"],
            "z_vector_error_ha_per_bohr": out["z_vector_error"],
            "error_elements": out["error_elements"],
            "error_excludes": EXCLUDED,
        })
    else:
        details.update({"gradient_error_ha_per_bohr": None, "gradient_error_reason": unavailable})
        if z is not None:
            # Under the gradient floor no Hessian is inverted, so there is no check to skip.
            details["hessian_curvature_check"] = (
                "skipped: the five-sigma curvature check needs the Hessian energies' standard "
                "errors, and they are not reported")
    if z is not None:
        for key in ("z_vector_norm", "hessian_cutoff", "hessian_lowest_curvature",
                    "hessian_highest_curvature", "z_vector_dropped", "z_vector_dropped_gradient_norm"):
            details[key] = measured[key]
        if errors:
            for key in ("hessian_lowest_curvature_error", "hessian_curvature_significance"):
                details[key] = measured[key]
    return out["gradient"], details
