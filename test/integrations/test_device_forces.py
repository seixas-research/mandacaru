# -*- coding: utf-8 -*-
# file: test/integrations/test_device_forces.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Forces on a circuit device (``integrations.device_forces``): measured RDMs, a parameter
Hessian measured with the shift rule, the full Z-vector correction and the exact propagation of
the RDM errors and of the Hessian energies' errors through ``z``. The references: the
state-vector analytic force at the same parameters (the exact circuit device must equal it), the
observed scatter of repeated sampled forces, and the refusals (before any job for the budget,
before the RDM job for a curvature that shot noise swamps)."""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.units import BOHR_TO_ANGSTROM, HARTREE_TO_EV

OPTIONS = dict(method="vqe", basis="PAW-LCAO", grid_spacing=0.4, active_space={"orbitals": 2},
               trace=False, optimizer={"method": "COBYLA", "maxiter": 6, "tol": 1e-6},
               parallelization={"threads": 1})


def h2(**options):
    atoms = Atoms("H2", positions=[[2.5, 2.5, 2.1], [2.5, 2.5, 2.9]], cell=[5.0] * 3)
    atoms.calc = Mandacaru(**{**OPTIONS, **options})
    atoms.get_potential_energy()
    return atoms


def state_vector_gradient(calc):
    """The analytic force of the state the device's circuit prepares, from its state vector."""
    result = calc.result
    raw = result._raw
    indices = [int(i) for i in raw["indices"]]
    parameters = np.ascontiguousarray(raw["parameters"], dtype=float)
    state = raw["state"]
    out = calc.native_reference.quantum_gradient(
        result._problem, result._model, np.ascontiguousarray(state), indices=indices,
        parameters=parameters, product_form=True)
    return out["gradient"], out


def test_on_an_exact_circuit_device_the_force_is_the_state_vector_force():
    atoms = h2(execute_circuits=True)
    calc = atoms.calc
    force = atoms.get_forces()
    details = calc.result.force_details
    assert details["density_matrices"] == "measured on the device"
    assert details["stationarity"] == "lagrangian"   # 6 COBYLA steps: off the optimum
    assert details["hessian_energy_evaluations"] == 1 + 2 * details["hessian_rotations"] ** 2
    assert np.all(details["gradient_error_ha_per_bohr"] == 0) and details["error_elements"] == 0
    assert np.all(details["z_vector_error_ha_per_bohr"] == 0)
    assert details["hessian_lowest_curvature_error"] is None
    gradient, exact = state_vector_gradient(calc)
    assert exact["stationarity"] == "lagrangian"
    expected = -gradient * HARTREE_TO_EV / BOHR_TO_ANGSTROM
    raw = calc.result.forces(project_translation=False)
    assert np.max(np.abs(raw - expected)) < 1e-8 * HARTREE_TO_EV / BOHR_TO_ANGSTROM
    assert force.shape == (2, 3)


@pytest.mark.slow  # 115 s (measured 2026-11-10, load 5-7): 8 sampled device forces
def test_a_sampled_device_force_scatters_as_its_reported_error():
    """Sampled on noiseless Aer: the bond force of 8 seeds (the Hessian and the RDMs re-measured
    at the same optimized parameters) scatters by the reported error, which now carries both the
    RDM noise and the Hessian energies' noise through ``z``. 8 samples estimate a standard
    deviation to about 25 percent, hence the band. A shot run holds no state vector, so the exact
    reference is the exact-device test."""
    atoms = h2(shots=4000)
    calc = atoms.calc
    bond, reported, through_z = [], [], []
    for seed in range(8):
        calc.provider.seed, calc.provider._estimator = 200 + seed, None
        calc.result.__dict__.pop("_device_forces", None)
        calc.result.forces(project_translation=False)
        details = calc.result.force_details
        bond.append(-(calc.result._device_forces[False][0][1, 2]))
        reported.append(details["gradient_error_ha_per_bohr"][1, 2])
        through_z.append(details["z_vector_error_ha_per_bohr"][1, 2])
    details = calc.result.force_details
    assert details["stationarity"] == "lagrangian" and details["error_elements"] > 0
    assert details["hessian_curvature_significance"] > 5
    assert details["error_excludes"].startswith("the covariance between different RDM elements")
    error, z_part = float(np.mean(reported)), float(np.mean(through_z))
    # The Hessian's noise is wired in: a nonzero part, and the total above the RDM part alone.
    assert 0 < z_part < error, (z_part, error)
    observed = float(np.std(bond, ddof=1))
    print(f"bond force scatter {observed:.3e}, reported {error:.3e} (through z {z_part:.3e}) Ha/Bohr")
    assert 0.5 < observed / error < 1.8, (observed, error)


def _double_inflection(atoms):
    """The double excitation's angle where its own curvature vanishes (singles at zero). Its
    generator ``G`` has ``G^3 = -G`` on this register, so along it ``E = A + B cos 2t + C sin 2t``
    exactly, and the exact shift-rule gradient and Hessian at ``t = 0`` (``g = 2C``,
    ``H = -4B``) give the zero of ``H(t) = -4 (B cos 2t + C sin 2t)``."""
    from mandacaru.integrations.circuits import CircuitExecutor
    calc = atoms.calc
    problem, indices = calc.result._problem, [int(i) for i in calc.result._raw["indices"]]
    measured = problem.device_hessian(indices, [0.0] * len(indices), CircuitExecutor(calc))
    double = int(np.argmax(np.abs(measured["gradient"])))
    g, h = float(measured["gradient"][double]), float(measured["hessian"][double, double])
    b, c = -h / 4.0, g / 2.0
    theta = 0.5 * np.arctan(-b / c)
    assert abs(b * np.cos(2 * theta) + c * np.sin(2 * theta)) < 1e-12
    return double, theta


def test_a_curvature_swamped_by_shot_noise_is_refused_before_the_rdm_job():
    """At the double excitation's inflection the true curvature is zero, so the measured one is
    shot noise: the force refuses by name with the curvature and its standard error, after the
    Hessian job and before any RDM circuit."""
    double, theta = _double_inflection(h2(execute_circuits=True))
    atoms = h2(shots=4000)
    calc = atoms.calc
    parameters = np.zeros(len(calc.result._raw["parameters"]))
    parameters[double] = theta
    calc.result._raw["parameters"] = parameters
    before = calc.provider.run_usage()
    with pytest.raises(ValueError, match=r"curvature .* standard error .* within 5 standard errors"):
        calc.result.forces()
    after = calc.provider.run_usage()
    assert after["jobs"] == before["jobs"] + 1, (before, after)   # the Hessian job only


def test_a_device_force_over_its_circuit_budget_is_refused_before_any_job():
    atoms = h2(execute_circuits=True)
    calc = atoms.calc
    calc._measurement_budget = {**calc._measurement_budget, "circuit_instances": 10}
    usage = calc.provider.run_usage()
    with pytest.raises(ValueError, match=r"the device force needs [\d,]+ circuits.*circuit_instances"):
        calc.result.forces()
    assert calc.provider.run_usage() == usage


class _PastTheProductLimit:
    """The native problem, except that ``device_hessian`` answers as it does when a measurement
    basis needs more covariance products than the classical limit (the Rust test
    ``a_basis_past_the_product_limit_measures_the_hessian_without_an_error`` builds that case):
    the Hessian is measured, its energies carry no error, and the reason comes back."""

    REASON = ("not reported: a measurement basis of 820 strings needs more than 20000 covariance "
              "product strings, past the classical limit of the estimate")

    def __init__(self, problem):
        self._problem = problem

    def __getattr__(self, name):
        return getattr(self._problem, name)

    def device_hessian(self, indices, parameters, executor, **_options):
        z = np.zeros(len(parameters))
        z[-1] = 0.01
        return {"z": z, "gradient_norm": 0.26, "hessian_error_unavailable": self.REASON,
                "z_covariance": None, "z_vector_norm": 0.01, "hessian_cutoff": 1e-10,
                "hessian_lowest_curvature": 0.5, "hessian_highest_curvature": 2.2,
                "z_vector_dropped": 0, "z_vector_dropped_gradient_norm": 0.0}


def test_past_the_product_limit_the_device_force_runs_and_reports_no_error(tmp_path):
    """Owner, 2026-11-10: "Run, error None". The force is measured and returned, its error is
    ``None`` with the reason, ``error_excludes`` is not written, and the run log states once that
    the five-sigma curvature check was skipped (a Z vector is solved, so there was a check to skip;
    under the gradient floor the line is not written)."""
    out = str(tmp_path / "output.txt")
    atoms = h2(shots=200, txt=out)
    calc = atoms.calc
    calc.result._problem = _PastTheProductLimit(calc.result._problem)
    force = atoms.get_forces()
    assert force.shape == (2, 3) and np.all(np.isfinite(force)) and abs(force[1, 2]) > 0.1
    details = calc.result.force_details
    assert details["gradient_error_ha_per_bohr"] is None
    assert details["gradient_error_reason"] == _PastTheProductLimit.REASON
    assert "error_excludes" not in details and "z_vector_error_ha_per_bohr" not in details
    assert details["stationarity"] == "lagrangian" and "hessian_curvature_significance" not in details
    text = open(out).read()
    assert text.count("hessian_curvature_check: skipped") == 1
    assert text.count("gradient_error_reason: not reported") == 1
    assert "error_excludes" not in text
