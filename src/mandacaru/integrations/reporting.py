"""Render native scientific results using Mandacaru's structured text protocol."""

from types import SimpleNamespace
from ..utils.logging import Logger, STDOUT


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


def write_calculation(driver, result, geometry):
    targets = []
    if driver.txt is not None:
        targets.append(str(driver.txt))
    if driver.trace:
        targets.append(STDOUT)
    if not targets:
        return
    with Logger(targets, n_qubits=driver.n_qubits) as log:
        if geometry is not None:
            from ase.data import chemical_symbols

            log.write_system(
                symbols=[chemical_symbols[z] for z in geometry.numbers],
                positions=geometry.positions,
                units=geometry.units,
                cell=geometry.cell,
                pbc=geometry.pbc,
            )
        log.write_basis(
            {"name": str(driver.basis), "spatial_orbitals": result.n_spatial_orbitals}
        )
        log.write_electrons(
            {
                "mapping": driver._mapping,
                "n_qubits": driver.n_qubits,
                "num_particles": result.num_particles,
            }
        )
        if driver._active_spec is not None and result._reference is not None:
            model = result.hamiltonian
            log.write_active_space(
                {
                    "method": driver._active_spec.get("method", "energy"),
                    "spatial_orbitals": model.n_spatial_orbitals,
                    "frozen_orbitals": len(model.frozen_indices),
                    "source_orbitals": driver.integrals.n_orbitals,
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
        if result._problem is not None:
            if driver._gradient_measurement is not None:
                log.write_block("GRADIENT MEASUREMENT", driver._gradient_measurement, framed=True)
            log.write_optimizer_setup(
                driver._optimizer["method"],
                result.reference_energy,
                result.energy_unit,
                max_iterations=driver._max_growth if driver._search is None else driver._search["max_steps"],
                extra={
                    "pool": driver._pool,
                    "pool_size": len(driver.problem.pool_labels),
                },
                gradient_method="analytic",
                gradient_units="Ha/rad",
            )
            for i, row in enumerate(result.iterations, 1):
                log.write_iteration(
                    i,
                    [SimpleNamespace(label=row.operator_label)],
                    [row.max_gradient],
                    0,
                    None,
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
                reference_energy=result.reference_energy,
                correlation_energy=result.correlation_energy,
                num_parameters=result.num_parameters,
                final_max_gradient=result.final_max_gradient,
                num_evaluations=result.num_evaluations,
                optimizer_steps=result.optimizer_steps,
                optimizer=driver._optimizer["method"],
            )
        else:
            log.write_block(
                "SCF SUMMARY",
                {
                    "converged": result.success,
                    f"energy_{result.energy_unit}": result.optimal_energy,
                },
                framed=True,
            )
