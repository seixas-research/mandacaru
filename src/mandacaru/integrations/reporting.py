"""Render native scientific results using Mandacaru's structured text protocol."""

from types import SimpleNamespace
from ..utils.logging import Logger, STDOUT


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
