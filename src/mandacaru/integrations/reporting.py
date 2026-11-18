"""Render native scientific results using Mandacaru's structured text protocol."""

from types import SimpleNamespace
from .. import _native
from ..utils.logging import Logger, STDOUT
from . import run_log_fields
from .basis_report import basis_report


def write_forces(driver, result, geometry, *, internal, translated):
    import numpy as np
    from .. import _native
    from ase.data import chemical_symbols
    from ..utils.logging import append_forces
    from .history import DiskOptimizationHistory

    forces = driver.results["forces"]
    history = getattr(driver, "_optimization_history", None)
    if history is None or not isinstance(history, DiskOptimizationHistory):
        scratch = getattr(getattr(driver, "resources", None), "scratch_dir", None)
        history = DiskOptimizationHistory(scratch_dir=scratch)
        driver._optimization_history = history
    summary = _native.force_summary(np.ascontiguousarray(forces, dtype=float))
    entry = dict(energy=float(driver.results["energy"]),
                 max_force=summary["max"], net_force=summary["net"])
    if history and geometry.same_as(driver._last_force_geometry):
        history.replace_last(entry)
    else:
        history.append(entry)
    driver._last_force_geometry = geometry
    driver._summary_written = False
    convention = (
        "forces = negative energy derivative projected onto internal molecular directions"
        if internal else
        "forces = -dE/dR with its mean translation removed (ASE sign)"
        if translated else "forces = -dE/dR (ASE sign)"
    )
    details = getattr(result, "force_details", {})
    for target in _targets(driver):
        append_forces(
            target, [chemical_symbols[z] for z in geometry.numbers],
            driver.results["forces"], convention=convention,
            extra={key: value for key, value in details.items()
                   if key != "gradient"},
        )


def write_excitations(driver, view, *, tolerance):
    """Append the ``[EXCITED STATES]`` block of one ``get_excitations`` request to the run log
    (``driver.txt``, and standard output when tracing)."""
    from ..utils.logging import append_excitations

    for target in _targets(driver):
        append_excitations(target, view, tolerance=tolerance)


def write_real_time(driver, view, *, seconds):
    """Append the ``[REAL TIME]`` block of one ``get_real_time`` request to the run log
    (``driver.txt``, and standard output when tracing)."""
    from ..utils.logging import append_real_time

    for target in _targets(driver):
        append_real_time(target, view, seconds=seconds)


def write_record(driver, title, view):
    """Append the block ``title`` of a view with a native ``record`` (``[DIELECTRIC FUNCTION]``,
    a crystal's ``[EXCITED STATES]``) to the run log (``driver.txt``, and standard output when
    tracing)."""
    from ..utils.logging import append_record

    for target in _targets(driver):
        append_record(target, title, view)


def write_hubbard_response(driver, resp):
    """Append ``[HUBBARD RESPONSE]`` for a linear-response ``U`` measurement to the run log."""
    from ..utils.logging import append_block

    targets = _targets(driver)
    if not targets:
        return
    unit = resp.energy_unit
    u_charge = resp.u_charge
    diag = [float(u_charge[i, i]) for i in range(resp.shells)]
    fields = {
        "shells": resp.shells,
        "channels": resp.channels,
        "self_consistent": resp.self_consistent,
        f"alphas_{unit}": ", ".join(f"{a:.6f}" for a in resp.alphas),
        f"u_charge_diag_{unit}": ", ".join(f"{u:.6f}" for u in diag),
        "nonlinearity": resp.nonlinearity,
        "asymmetry": f"{resp.asymmetry:.3e}",
        "condition": f"{resp.condition:.3e}",
        "floor": f"{resp.floor:.3e}",
    }
    if resp.j_response is not None:
        fields[f"j_response_{unit}"] = ", ".join(f"{j:.6f}" for j in resp.j_response)
    for target in targets:
        append_block(target, "[HUBBARD RESPONSE]", fields)


def write_optimization_summary(driver, atoms=None, fmax=None, optimizer=None, force=False, status=None):
    from ase.data import chemical_symbols
    from ..structure import geometry_value
    from ..utils.logging import append_optimization_summary

    targets = _targets(driver)
    history = getattr(driver, "_optimization_history", [])
    if not targets or getattr(driver, "_summary_written", False) or not history or (len(history) < 2 and not force):
        return False
    converged = None
    if optimizer is not None:
        if fmax is None:
            fmax = optimizer.fmax
        converged = bool(optimizer.converged())
    geometry = driver._last_force_geometry if atoms is None else geometry_value(atoms)
    for target in targets:
        append_optimization_summary(
            target, history, symbols=[chemical_symbols[z] for z in geometry.numbers],
            positions=geometry.positions, units=geometry.units, energy_unit="eV",
            fmax=fmax, converged=converged, status=status,
        )
    driver._summary_written = True
    return True


#: State-vector pairs sampled per row of the ``expr`` column (twice as many preparations).
EXPRESSIVITY_PAIRS = 400


def _targets(driver):
    return ([str(driver.txt)] if driver.txt is not None else []) + ([STDOUT] if driver.trace else [])


def write_performance(driver, result, wall_time_s, *, solver_ran):
    """Append the ``[PERFORMANCE]`` block of one ``calculate`` call.

    The block closes the step, after the forces: the nuclear gradient is
    usually the largest stage of a relaxation step. The stages are the clocks
    around the native calls (``Mandacaru._stage``); the solver's own time is
    the one the native result reports, and is left out when the solver did not
    run (forces requested on a result that already existed).
    """
    from ..backends.providers import qpu_usage
    from ..utils.logging import append_performance
    from ..resources import layout_label
    from ..utils.profiling import Timings

    targets = _targets(driver)
    if not targets:
        return
    # The block owns the processor layout: the one place the run log writes it.
    resources = Timings(parallelization=layout_label(driver.context)).resources()
    extra = {}
    if solver_ran and result._problem is not None:
        extra["solver_wall_time_s"] = result.timings["solve_seconds"]
    if driver.provider is not None and driver._shots:
        extra.update(qpu_usage(driver.provider, cumulative=True))
    for target in targets:
        append_performance(target, stages=dict(driver._stages), wall_time_s=wall_time_s,
                           resources=resources, extra=extra or None)


def _write_scf(log, driver, result):
    """``[SCF SETUP]`` and ``[SCF ITERATIONS]`` from the native SCF's saved history."""
    snapshot = driver.native_reference.snapshot() if result._problem is not None else result._raw
    log.write_block("SCF SETUP", run_log_fields.scf_setup(driver, snapshot))
    log.write_scf_iterations(run_log_fields.scf_rows(driver, snapshot), result.energy_unit)


def _write_active_space(log, driver, result):
    model = result.hamiltonian
    # The block owns the warning that the selection splits a degenerate orbital group where no
    # deterministic frame fixes it (a crystal's supercell): the energy then depends on the
    # eigensolver's choice inside the group, and a force is refused (`MeanField::split_groups`).
    split = result._reference.split_groups(model)
    log.write_active_space(
        {
            "method": driver._active_spec.get("method", "energy"),
            "frozen_orbitals": len(model.frozen_indices),
            "warning": (
                f"the active space splits {len(split)} exactly degenerate orbital group"
                f"{'' if len(split) == 1 else 's'} ("
                + "; ".join(group["description"] for group in split)
                + "); orbitals inside a group are the eigensolver's choice here, so this "
                "energy depends on it and forces are refused; keep each group whole"
            ) if split else None,
        },
        {
            "orbitals": [["index", "role"]]
            + [
                [
                    i,
                    "frozen"
                    if i in model.frozen_indices
                    else "active"
                    if i in model.orbital_indices
                    else "deleted",
                ]
                for i in range(driver.integrals.n_orbitals)
            ]
        },
    )


def _expressivity(driver, result, count):
    """The expressivity of the ansatz after ``count`` parameters, from native samples."""
    from ..analysis.expressibility import active_space_dimension, calculate_kl_divergence

    # The sector lives on the spin-orbitals, however many qubits tapering took off.
    modes = _native.spin_orbital_count(result.n_spatial_orbitals)
    fidelities = result._problem.sample_fidelities(
        result._raw["indices"][:count], pairs=EXPRESSIVITY_PAIRS, seed=0, product_form=True
    )
    return calculate_kl_divergence(
        fidelities, driver.n_qubits, num_bins=75,
        dimension=active_space_dimension(modes, driver.num_particles),
    )


def _write_variational(log, driver, result, log_expressivity):
    if driver._energy_measurement is not None:
        log.write_block("ENERGY MEASUREMENT", driver._energy_measurement, framed=True)
    if driver._gradient_measurement is not None:
        log.write_block("GRADIENT MEASUREMENT", driver._gradient_measurement, framed=True)
    lineage, start_energy = run_log_fields.lineage(driver, result)
    # The thresholds belong to the growth loop of the ADAPT methods; a search has no growth
    # budget, its `max_steps` is one of the search lines of `extra`.
    searching = driver._search is not None
    grows = driver.method.endswith("adapt-vqe") and not searching
    log.write_optimizer_setup(
        driver._optimizer["method"],
        result.reference_energy,
        result.energy_unit,
        convergence=driver._convergence if grows else None,
        max_iterations=None if searching else driver._max_growth,
        start_energy=start_energy,
        extra=run_log_fields.optimizer_setup(driver, result),
        gradient_method="analytic",
        gradient_units="Ha/rad",
        lineage=lineage,
    )
    for i, row in enumerate(result.iterations, 1):
        log.write_iteration(
            i,
            [SimpleNamespace(label=row.operator_label)],
            [row.max_gradient],
            0,
            _expressivity(driver, result, row.num_parameters) if log_expressivity else None,
            row.energy,
            row.num_parameters,
            energy_unit=result.energy_unit,
            metrics=row.metrics,
            optimizer_steps=row.optimizer_steps,
            energy_error=row.energy_error,
        )
    for row in result.steps:
        log.write_chain_step(
            step=row.step, move=row.move, length=len(row.proposed),
            energy=row.energy, delta_energy=row.delta_energy,
            current_energy=row.current_energy, temperature=row.temperature,
            log_acceptance=row.log_acceptance, accepted=row.accepted,
            energy_unit=result.energy_unit, action=f"{row.move}@{row.position}")
    log.write_summary(
        result.success,
        result.optimal_energy,
        num_operators=result.num_operators,
        energy_unit=result.energy_unit,
        energy_error=result.energy_error,
        energy_error_reason=result.energy_error_reason,
        correlation_energy=result.correlation_energy,
        num_parameters=result.num_parameters,
        final_max_gradient=result.final_max_gradient,
        num_evaluations=result.num_evaluations,
        optimizer_steps=result.optimizer_steps,
        optimizer=driver._optimizer["method"],
        extra={**(run_log_fields.search_summary(result) if searching else {}),
               "termination": result.timings["termination"],
               **({"pruned_operators": len(result.pruned_operators)} if driver._prune else {})},
    )


def write_calculation(driver, result, geometry, *, log_expressivity=False):
    targets = _targets(driver)
    if not targets:
        return
    quantum = result._problem is not None
    # A crystal has no register.
    with Logger(targets, n_qubits=driver.n_qubits if quantum else None) as log:
        if geometry is not None:
            from ase.data import chemical_symbols

            log.write_system(
                symbols=[chemical_symbols[z] for z in geometry.numbers],
                positions=geometry.positions,
                units=geometry.units,
                cell=geometry.cell, pbc=geometry.pbc,
                magmoms=(geometry.magnetic_moments if geometry.moment_vectors is None else geometry.moment_vectors),
            )
        log.write_basis(*basis_report(driver, geometry))
        log.write_electrons(run_log_fields.electrons(driver, result, geometry))
        if geometry is not None:
            log.write_block("SYMMETRY", run_log_fields.symmetry(driver, result, geometry))
        if geometry is not None and driver.native_reference is not None:
            _write_scf(log, driver, result)
        elif getattr(result, "_crystal", None) is not None:
            log.write_block("SCF SETUP", run_log_fields.crystal_scf_setup(driver, result))
            log.write_scf_iterations(run_log_fields.crystal_scf_rows(driver, result),
                                     result.energy_unit, energy_name="free_energy")
        if driver._active_spec is not None and result._reference is not None:
            _write_active_space(log, driver, result)
        if quantum:
            _write_variational(log, driver, result, log_expressivity)
        else:
            log.write_block(
                "SCF SUMMARY",
                {
                    "converged": result.success,
                    "occupation": getattr(driver.native_reference, "occupation", None),
                    f"energy_{result.energy_unit}": result.optimal_energy,
                    **result._raw.get("excited_state", {}),
                    **run_log_fields.hubbard(result),
                },
                framed=True,
            )
