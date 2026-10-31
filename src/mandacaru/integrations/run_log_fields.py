"""The field dictionaries of the run log's blocks.

``reporting.write_calculation`` writes the log; this module decides what each
block says. Every value is read from the native result or from an option the
calculator already validated, and is formatted here: nothing is computed.
"""

import inspect

from .. import _native
from ..units import BOHR_TO_ANGSTROM, HARTREE_TO_EV

#: How a fermion-to-qubit mapping is spelled in the ``[ELECTRONS]`` block: the
#: transformations' own names, not the identifiers the code uses.
MAPPING_LABELS = {"jordan_wigner": "Jordan-Wigner", "parity": "parity",
                  "parity_reduced": "reduced parity", "bravyi_kitaev": "Bravyi-Kitaev"}

#: What each mean-field method is called in ``[SCF SETUP]``.
SCF_LABELS = {"rhf": "restricted Hartree-Fock (closed shell)",
              "uhf": "unrestricted Hartree-Fock",
              "ghf": "generalized (spinor) Hartree-Fock"}


def energy_factor(driver):
    """Hartree into the unit the log's energies are written in."""
    return 1.0 if driver._atomic_units else HARTREE_TO_EV


def grid_text(grid):
    """The realized grid: points and, for an orthogonal grid, the spacing per axis."""
    step = grid.step_bohr
    points = " x ".join(str(n) for n in grid.shape)
    if any(step[i][j] != 0 for i in range(3) for j in range(3) if i != j):
        return f"{points} (non-orthogonal step vectors)"
    spacing = " x ".join(f"{step[i][i] * BOHR_TO_ANGSTROM:.4f}" for i in range(3))
    return f"{points} (spacing {spacing} Angstrom)"


def two_body_text(driver):
    """The Cholesky line of a run that asked for ``two_body=``, else ``None``."""
    if not driver._two_body_requested:
        return None
    report = _native.integrals_report(driver.integrals)
    head = f"Cholesky factors, tolerance {driver._cholesky_tolerance:.3e} Ha"
    if report["cholesky_rank"] is None:
        return f"{head}, not built"
    residual = report["cholesky_residual"]
    return (f"{head}, rank {report['cholesky_rank']} "
            f"(basis functions {report['n_orbitals']}), largest residual diagonal "
            + ("not computed" if residual is None else f"{residual:.3e} Ha"))


def kpoints_text(driver):
    n1, n2, n3 = driver._kpts_spec["size"]
    mesh = f"{n1}x{n2}x{n3} Monkhorst-Pack"
    return f"Gamma ({mesh})" if (n1, n2, n3) == (1, 1, 1) else mesh


def electrons(driver, result, geometry):
    """The ``[ELECTRONS]`` block: how the electronic problem was posed.

    A calculation from a geometry states its discretization first; one from a
    Hamiltonian has none. A classical run writes no mapping, reference state,
    Z2 line, Hamiltonian size or register: a line reading ``Jordan-Wigner``
    would describe a calculation that did not happen.
    """
    quantum = result._problem is not None
    fields = {}
    if geometry is not None and driver.integrals is not None:
        fields["grid spacing"] = f"{driver._grid_spacing:g} Angstrom (requested)"
        fields["grid points"] = grid_text(driver.integrals.grid)
        fields["kinetic operator"] = driver._kinetic
        fields["two-body integrals"] = two_body_text(driver)
        fields["k-points"] = kpoints_text(driver)
        fields["charge"] = str(driver._charge)
    alpha, beta = result.num_particles
    fields["spin-polarized"] = f"{alpha != beta} (multiplicity {abs(alpha - beta) + 1})"
    if quantum:
        fields["reference state"] = "hartree-fock"
        fields["Z2 tapering"] = "none"
        fields["mapping"] = MAPPING_LABELS.get(driver._mapping, driver._mapping)
        fields["Hamiltonian"] = f"{_native.problem_term_count(result._problem)} Pauli terms"
    fields["spatial orbitals"] = str(result.n_spatial_orbitals)
    fields["electrons (alpha, beta)"] = str(result.num_particles)
    if quantum:
        fields["qubits"] = str(driver.n_qubits)
    return fields


def scf_setup(driver, snapshot):
    """The ``[SCF SETUP]`` block: which SCF ran and what it stops on.

    The defaults are the ones the native method declares, read from its
    signature, so Python holds no copy of them.
    """
    if snapshot["functional"] is not None:
        entry, label = driver.integrals.kohn_sham, f"Kohn-Sham DFT ({snapshot['functional']})"
    else:
        entry, label = driver.integrals.mean_field, SCF_LABELS[snapshot["method"]]
    options = {name: parameter.default
               for name, parameter in inspect.signature(entry).parameters.items()
               if parameter.default is not inspect.Parameter.empty}
    options.update(driver._scf_options)
    return {
        "scf_method": label,
        "scf_max_iterations": options["max_iterations"],
        "scf_convergence_Hartree": (f"{options['tolerance']:g} (energy change and "
                                    "residual)"),
        "scf_acceleration": "DIIS" if options["diis"] else "none",
        "scf_history": "iterations of the lowest-energy start",
    }


def hubbard(result):
    """The DFT+U lines of ``[SCF SUMMARY]``: the correction's energy and its definition.

    The one owner of these facts: ``hubbard_energy`` and, per element, the shell with its
    ``U``, ``J`` and ``U_eff``, all read from the converged result in its energy unit. Empty
    for a run without the correction.
    """
    shells = result.hubbard
    if shells is None:
        return {}
    unit = result.energy_unit
    fields = {f"hubbard_energy_{unit}": result.hubbard_energy}
    for shell in shells:
        key = f"hubbard_{shell['symbol']}_{shell['l']}"
        if key not in fields:
            fields[key] = (f"U {shell['U']:.6f} {unit}, J {shell['J']:.6f} {unit}, "
                           f"U_eff {shell['U_eff']:.6f} {unit}")
    return fields


def scf_rows(driver, snapshot):
    """The saved SCF history as table rows, in the log's energy unit."""
    factor = energy_factor(driver)
    return [{"iter": iteration, "energy": energy * factor,
             "dE": None if delta is None else delta * factor, "residual": residual}
            for iteration, energy, delta, residual in snapshot["history"]]


def lineage(driver):
    """``(fields, start_energy)`` of a resumed run; ``({}, None)`` for a fresh one."""
    resumed = driver._resume_lineage
    if resumed is None:
        return {}, None
    factor = energy_factor(driver)
    unit = "Ha" if driver._atomic_units else "eV"
    energy = resumed["energy_ha"] * factor
    return ({"resumed_from": str(driver._resume_path),
             "restored_operators": resumed["num_operators"],
             "restored_parameters": resumed["num_parameters"],
             f"restored_energy_{unit}": f"{energy:.10f}",
             # The native resume raises on any mismatch, so this is a checked fact.
             "resume_same_hamiltonian": True}, energy)


def backend_text(driver, problem):
    """How the state vector is held, for ``state_vector_backend``."""
    if driver._shots:
        return ("none (shots > 0: every expectation value is measured on the "
                "device, no state vector)")
    register = f"2^{problem.num_qubits}"
    kind = "sparse " if driver._sparse else ""
    if driver._sector:
        return f"{kind}particle-number sector, {problem.dimension} states of {register}"
    return f"{kind}full register, {problem.dimension} amplitudes of {register}"


def optimizer_setup(driver, result):
    """The pool, the state-vector backend and the device lines of ``[OPTIMIZATION SETUP]``."""
    problem = driver.problem
    extra = {"pool": driver._pool,
             "pool_class": _native.pool_kind(driver._pool),
             "pool_size": len(problem.pool_labels)}
    if (driver._precision == "float32" and driver._optimizer.get("tol") is not None
            and not driver._execute_circuits):
        extra["optimizer_tolerance"] = f"{driver._optimizer['tol']:g} Ha"
    extra["reoptimize_all_parameters"] = True
    extra["state_vector_backend"] = backend_text(driver, problem)
    extra["device"] = driver._device
    extra["backend_provider"] = "none" if driver.provider is None else driver.provider.name
    extra["circuit_execution"] = driver._execute_circuits
    extra["shots"] = (f"{driver._shots} (sampled expectation values)" if driver._shots
                      else "0 (exact expectation values)")
    return extra
