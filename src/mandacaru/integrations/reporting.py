"""Render native scientific results using Mandacaru's structured text protocol."""

from types import SimpleNamespace
from ..utils.logging import Logger, STDOUT
from . import run_log_fields
from .basis_report import basis_report


def write_forces(driver, result, geometry, *, internal, translated):
    import numpy as np
    from ase.data import chemical_symbols
    from ..utils.logging import append_forces

    forces = driver.results["forces"]
    history = getattr(driver, "_optimization_history", [])
    entry = dict(energy=float(driver.results["energy"]),
                 max_force=float(np.linalg.norm(forces, axis=1).max()),
                 net_force=float(np.abs(forces.sum(axis=0)).max()))
    if history and geometry.same_as(driver._last_force_geometry):
        history[-1] = entry
    else:
        history.append(entry)
    driver._optimization_history = history
    driver._last_force_geometry = geometry
    driver._summary_written = False
    targets = ([str(driver.txt)] if driver.txt is not None else []) + ([STDOUT] if driver.trace else [])
    convention = (
        "forces = negative energy derivative projected onto internal molecular directions"
        if internal else
        "forces = -dE/dR with its mean translation removed (ASE sign)"
        if translated else "forces = -dE/dR (ASE sign)"
    )
    details = getattr(result, "force_details", {})
    for target in targets:
        append_forces(
            target, [chemical_symbols[z] for z in geometry.numbers],
            driver.results["forces"], convention=convention,
            extra={key: value for key, value in details.items() if key != "gradient"},
        )


def write_optimization_summary(driver, atoms=None, fmax=None, optimizer=None, force=False, status=None):
    from ase.data import chemical_symbols
    from ..structure import geometry_value
    from ..utils.logging import append_optimization_summary

    targets = ([str(driver.txt)] if driver.txt is not None else []) + ([STDOUT] if driver.trace else [])
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
    return ([str(driver.txt)] if driver.txt is not None else []) + (
        [STDOUT] if driver.trace else [])


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
    from ..utils.profiling import Timings

    targets = _targets(driver)
    if not targets:
        return
    threads = driver.context.threads
    resources = Timings(n_cores=threads, backend=f"Rust ({threads} threads)").resources()
    # The peak resident size is the one memory figure of the block.
    resources.pop("resident_memory_MiB")
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
    log.write_active_space(
        {
            "method": driver._active_spec.get("method", "energy"),
            "frozen_orbitals": len(model.frozen_indices),
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

    # The sector lives on the spin-orbitals: a tapered register has two more.
    modes = driver.n_qubits + (2 if driver._mapping == "parity_reduced" else 0)
    fidelities = result._problem.sample_fidelities(
        result._raw["indices"][:count], pairs=EXPRESSIVITY_PAIRS, seed=0, product_form=True
    )
    return calculate_kl_divergence(
        fidelities, driver.n_qubits, num_bins=75,
        dimension=active_space_dimension(modes, driver.num_particles),
    )


def _write_variational(log, driver, result, log_expressivity):
    if driver._gradient_measurement is not None:
        log.write_block("GRADIENT MEASUREMENT", driver._gradient_measurement, framed=True)
    lineage, start_energy = run_log_fields.lineage(driver)
    # The thresholds belong to the growth loop of the ADAPT methods.
    grows = driver.method.endswith("adapt-vqe") and driver._search is None
    log.write_optimizer_setup(
        driver._optimizer["method"],
        result.reference_energy,
        result.energy_unit,
        convergence=driver._convergence if grows else None,
        energy_scale=run_log_fields.energy_factor(driver),
        max_iterations=driver._max_growth if driver._search is None else driver._search["max_steps"],
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
        correlation_energy=result.correlation_energy,
        num_parameters=result.num_parameters,
        final_max_gradient=result.final_max_gradient,
        num_evaluations=result.num_evaluations,
        optimizer_steps=result.optimizer_steps,
        optimizer=driver._optimizer["method"],
        extra={"termination": result.timings["termination"]},
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
                cell=geometry.cell,
                pbc=geometry.pbc,
                magmoms=geometry.magnetic_moments,
            )
        log.write_basis(*basis_report(driver, geometry))
        log.write_electrons(run_log_fields.electrons(driver, result, geometry))
        if geometry is not None and driver.native_reference is not None:
            _write_scf(log, driver, result)
        if driver._active_spec is not None and result._reference is not None:
            _write_active_space(log, driver, result)
        if quantum:
            _write_variational(log, driver, result, log_expressivity)
        else:
            log.write_block(
                "SCF SUMMARY",
                {
                    "converged": result.success,
                    f"energy_{result.energy_unit}": result.optimal_energy,
                    **run_log_fields.hubbard(result),
                },
                framed=True,
            )
