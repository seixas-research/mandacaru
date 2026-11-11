"""The field dictionaries of the run log's blocks.

``reporting.write_calculation`` writes the log; this module decides what each
block says. Every value is read from the native result or from an option the
calculator already validated, and is formatted here: nothing is computed.
"""

import inspect

import numpy as np

from .. import _native
from . import tapering

#: How a fermion-to-qubit mapping is spelled in the ``[ELECTRONS]`` block: the
#: transformations' own names, not the identifiers the code uses.
MAPPING_LABELS = {"jordan_wigner": "Jordan-Wigner", "parity": "parity",
                  "parity_reduced": "reduced parity", "bravyi_kitaev": "Bravyi-Kitaev"}

#: What each mean-field method is called in ``[SCF SETUP]``.
SCF_LABELS = {"rhf": "restricted Hartree-Fock (closed shell)",
              "uhf": "unrestricted Hartree-Fock",
              "ghf": "generalized (spinor) Hartree-Fock"}


def grid_text(grid):
    """The realized grid: points and, for an orthogonal grid, the spacing per axis."""
    step = grid.step_bohr
    points = " x ".join(str(n) for n in grid.shape)
    if any(step[i][j] != 0 for i in range(3) for j in range(3) if i != j):
        return f"{points} (non-orthogonal step vectors)"
    diagonal = _native.to_user_units(
        np.ascontiguousarray([float(step[i][i]) for i in range(3)]), "length")
    spacing = " x ".join(f"{value:.4f}" for value in diagonal)
    return f"{points} (spacing {spacing} Angstrom)"


def two_body_text(driver, result=None):
    """The Cholesky line of a run that asked for ``two_body=``, else ``None``; with the
    double-factorized representation, also its truncation and what the model kept."""
    if not driver._two_body_requested:
        return None
    report = _native.integrals_report(driver.integrals)
    unit = "Ha" if driver._atomic_units else "eV"
    tolerance = _native.convert_energy(float(driver._two_body_tolerance_ev), "eV", unit)
    head = f"Cholesky factors, tolerance {tolerance:.3e} {unit}"
    if report["cholesky_rank"] is None:
        text = f"{head}, not built"
    else:
        residual = report["cholesky_residual"]
        text = (f"{head}, rank {report['cholesky_rank']} "
                f"(basis functions {report['n_orbitals']}), largest residual diagonal "
                + ("not computed" if residual is None
                   else f"{_native.energy_from_hartree(float(residual), unit):.3e} {unit}"))
    if driver._two_body_representation == "dense":
        return text
    truncation = _native.convert_energy(float(driver._two_body_truncation_ev), "eV", unit)
    text += f"; double-factorized active-space tensor, truncation {truncation:.3e} {unit}"
    model = None if result is None else getattr(result, "_model", None)
    if model is None:
        return f"{text}, not built"
    kept = model.two_body_report
    bound = _native.energy_from_hartree(float(kept["error_bound_ha"]), unit)
    return (f"{text}, {kept['kept']} of {kept['eigenpairs']} eigenpairs in {kept['leaves']} "
            f"of {kept['candidates']} leaves ({kept['stored_numbers']} numbers), "
            f"energy change bound {bound:.3e} {unit}")


def loaded_two_body_text(driver):
    """The two-body line of a run from a Hamiltonian file that records a double-factorized
    representation: the truncation it was built with and the bound it reached."""
    record = driver._loaded_two_body
    unit = "Ha" if driver._atomic_units else "eV"
    truncation = _native.convert_energy(float(record["truncation_ev"]), "eV", unit)
    bound = _native.convert_energy(float(record["error_bound_ev"]), "eV", unit)
    return (f"{record['representation']} active-space tensor (recorded in the loaded "
            f"Hamiltonian), truncation {truncation:.3e} {unit}, energy change bound "
            f"{bound:.3e} {unit}")


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
    crystal = getattr(result, "_crystal", None)
    if crystal is not None:
        return crystal_electrons(driver, result, crystal)
    fields = {}
    if geometry is not None and driver.integrals is not None:
        fields["grid spacing"] = f"{driver._grid_spacing:g} Angstrom (requested)"
        fields["grid points"] = grid_text(driver.integrals.grid)
        fields["kinetic operator"] = driver._kinetic
        fields["two-body integrals"] = two_body_text(driver, result)
        fields["k-points"] = kpoints_text(driver)
        fields["charge"] = str(driver._charge)
    elif driver._loaded_two_body is not None:
        fields["two-body integrals"] = loaded_two_body_text(driver)
    alpha, beta = result.num_particles
    fields["spin-polarized"] = f"{alpha != beta} (multiplicity {abs(alpha - beta) + 1})"
    if quantum:
        fields["reference state"] = "hartree-fock"
        fields["Z2 tapering"] = tapering.text(driver, result._problem)
        fields["mapping"] = MAPPING_LABELS.get(driver._mapping, driver._mapping)
        fields["Hamiltonian"] = f"{_native.problem_term_count(result._problem)} Pauli terms"
    fields["spatial orbitals"] = str(result.n_spatial_orbitals)
    fields["electrons (alpha, beta)"] = str(result.num_particles)
    if quantum:
        fields["qubits"] = str(driver.n_qubits)
    return fields


def crystal_electrons(driver, result, crystal):
    """The ``[ELECTRONS]`` block of a crystal: its grid, k mesh and electrons per cell.

    Particle counts per spin are not written: a crystal's occupations are smeared, its
    moment need not be an integer, and the moment is the SCF's result, not the posing.
    """
    points = len(result._raw["weights"])
    return {
        "grid spacing": f"{driver._grid_spacing:g} Angstrom (requested)",
        "grid points": grid_text(crystal.grid),
        "k-points": f"{kpoints_text(driver)}, {points} in the calculation",
        "charge": str(driver._charge),
        "spin-polarized": str(result._raw["n_spins"] == 2),
        "spin-orbit (spinor)": str(result._raw["spinor"]),
        "basis functions per cell": str(crystal.n_orbitals),
        "electrons per cell": f"{crystal.electrons:g}",
    }


def crystal_scf_setup(driver, result):
    """The ``[SCF SETUP]`` block of a crystal: the options its SCF ran with.

    The defaults are those the native solve declares, read from its signature (as for a
    molecule); the smearing is the one the result reports.
    """
    options = {name: parameter.default
               for name, parameter in inspect.signature(result._crystal.solve).parameters.items()
               if parameter.default is not inspect.Parameter.empty}
    options.update(driver._periodic_options)
    method, width = result._raw["smearing"]
    return {
        "scf_method": f"periodic Kohn-Sham DFT ({result._raw['functional']}), Bloch states",
        "scf_max_iterations": options["max_iterations"],
        "scf_convergence_Hartree": (f"{options['tolerance']:g} (free-energy change), density "
                                    f"residual {options['density_tolerance']:g}"),
        "scf_mixing": (f"{options['algorithm']}, mixing {options['mixing']:g}, "
                       f"history {options['history']}"),
        "scf_smearing": f"{method}, width {width:.6g} Ha",
        "scf_eigensolver": options["eigensolver"],
    }


def crystal_scf_rows(driver, result):
    """A crystal's SCF history as table rows: the free energy per cell, its change, the
    density residual and, spin-polarized, the magnetization per cell."""
    unit = "Ha" if driver._atomic_units else "eV"
    spin = result._raw["n_spins"] == 2
    rows = []
    for iteration, free, delta, residual, _moment_residual, magnetization in result._raw["history"]:
        row = {"iter": iteration, "energy": _native.energy_from_hartree(float(free), unit),
               "dE": None if delta is None else _native.energy_from_hartree(float(delta), unit),
               "residual": residual}
        if spin:
            row["moment"] = magnetization
        rows.append(row)
    return rows


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
    unit = "Ha" if driver._atomic_units else "eV"
    return [{"iter": iteration, "energy": _native.energy_from_hartree(float(energy), unit),
             "dE": None if delta is None else _native.energy_from_hartree(float(delta), unit),
             "residual": residual}
            for iteration, energy, delta, residual in snapshot["history"]]


#: Operator labels a ``start_operators`` line names before it counts the rest.
START_OPERATORS_SHOWN = 8


def lineage(driver, result=None):
    """``(fields, start_energy)`` of a run that did not start from the reference.

    A resumed run says what it was restored from, and its first ``dE`` is measured
    from the restored energy (``start_energy``). A search with ``transfer=True`` says
    whether its chain started from the previous geometry's ansatz or from the empty
    sequence, and why; its table measures nothing from that energy, so the second item
    is ``None``. A fresh run, or a search without ``transfer``, has no lineage.
    """
    resumed = driver._resume_lineage
    if resumed is not None:
        unit = "Ha" if driver._atomic_units else "eV"
        energy = _native.energy_from_hartree(float(resumed["energy_ha"]), unit)
        return ({"resumed_from": str(driver._resume_path),
                 "restored_operators": resumed["num_operators"],
                 "restored_parameters": resumed["num_parameters"],
                 f"restored_energy_{unit}": f"{energy:.10f}",
                 # The native resume raises on any mismatch, so this is a checked fact.
                 "resume_same_hamiltonian": True}, energy)
    if driver._search is None or result is None:
        return {}, None
    return _transfer_start(driver, result), None


def _transfer_start(driver, result):
    """The ``start`` lines of a search that asked for ``transfer=True``.

    ``result.transfer`` already says whether the carried ansatz survived its
    relaxation (``accepted``) and why it did not (``reason``). The overlap confidence
    is a measurement, so it is written whenever the native comparison ran; the
    placeholder of a first geometry has no ``indices`` and writes none. The start
    energy is the relaxed energy of the carried ansatz: an empty start's energy is the
    reference energy, which the block already owns.
    """
    carried = result.transfer
    if not carried:
        return {}
    fields = {}
    if carried["accepted"]:
        count = len(carried["indices"])
        fields["start"] = f"transferred ansatz ({count} operator{'' if count == 1 else 's'})"
    else:
        fields["start"] = f"empty ({' '.join(str(carried['reason']).split())})"
    if "indices" in carried:
        fields["transfer_confidence"] = f"{carried['confidence']:.10f}"
    relaxation = result._raw.get("seed_relaxation")
    if carried["accepted"]:
        if relaxation is not None:
            unit = result.energy_unit
            fields[f"start_energy_{unit}"] = (
                f"{_native.energy_from_hartree(float(relaxation['energy_ha']), unit):.10f}")
        labels = result._problem.pool_labels
        names = [labels[i] for i in carried["indices"]]
        shown = " ".join(names[:START_OPERATORS_SHOWN])
        more = len(names) - START_OPERATORS_SHOWN
        fields["start_operators"] = shown + (f" (+{more} more)" if more > 0 else "")
    return fields


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
    """The pool, the search or growth lines, the backend and the device lines of ``[OPTIMIZATION SETUP]``."""
    problem = driver.problem
    extra = {"pool": driver._pool,
             "pool_class": _native.pool_kind(driver._pool),
             "pool_size": len(problem.pool_labels)}
    if (driver._precision == "float32" and driver._optimizer.get("tol") is not None
            and not driver._execute_circuits):
        extra["optimizer_tolerance"] = f"{driver._optimizer['tol']:g} Ha"
    if driver._search is not None:
        extra.update(search_setup(driver, result))
    else:
        # `quenching=False` is accepted by vqe and adapt-vqe only; every other method re-optimizes all.
        extra["reoptimize_all_parameters"] = driver._quenching
        extra["growth"] = ("tetris (disjoint-support operators per step)" if driver._tetris
                           else "single (largest gradient)")
        extra["pruning"] = "on" if driver._prune else "off"
    extra["state_vector_backend"] = backend_text(driver, problem)
    extra["device"] = driver._device
    extra["backend_provider"] = "none" if driver.provider is None else driver.provider.name
    extra["circuit_execution"] = driver._execute_circuits
    extra["shots"] = (f"{driver._shots} (sampled expectation values)" if driver._shots
                      else "0 (exact expectation values)")
    return extra


def search_setup(driver, result):
    """The controls of a ``vasqa`` or ``valsqa`` chain, as the native search received them.

    Each is read from ``driver._search`` (the validated options, with the seed the run
    drew when none was given) and written in the log's energy unit; the search runs in
    Hartree. ``transfer_threshold`` and ``transfer_steps`` are the options of a run that
    asked for ``transfer=True``: what the transfer did is the lineage's.
    """
    search = driver._search
    unit = result.energy_unit

    def convert(value):
        return _native.energy_from_hartree(float(value), unit)

    weights = ", ".join(f"{name} {weight:g}" for name, weight
                        in zip(("insert", "delete", "replace"), search["move_weights"]))
    initial, final = search["temperatures"]
    extra = {"max_steps": search["max_steps"],
             "max_length": search["max_length"],
             "move_weights": weights,
             f"temperature_initial_{unit}": f"{convert(initial):.10g}",
             f"temperature_final_{unit}": f"{convert(final):.10g}",
             f"length_penalty_{unit}": f"{convert(search['length_penalty']):.10g}",
             "proposal": search["proposal"],
             "proposal_temperature": f"{search['proposal_temperature']:g}"}
    if driver.method == "valsqa":
        extra["proposal_model"] = (
            f"{'ready' if result.proposal_ready else 'not used'} ({result.proposal_reason})")
        update = result._raw.get("model_update")
        if update is not None:
            extra["proposal_model_update"] = update
    extra["warm_start"] = search["warm_start"]
    extra["replace_start"] = search["replace_start"]
    extra["screen_insertions"] = search["screen_insertions"]
    if driver._search_record is not None:
        extra["record_directory"] = str(driver._search_record)
    extra["seed"] = search["seed"]
    if driver._transfer:
        extra["transfer_threshold"] = f"{driver._transfer_threshold:g}"
        if driver._transfer_steps is not None:
            extra["transfer_steps"] = driver._transfer_steps
    return extra


def search_summary(result):
    """The lines a search adds to the summary: what the chain did, not how it was set.

    ``final_energy`` is the energy of the state the chain ended on, which is not the
    ``optimal_energy`` when the best ansatz came earlier (or when ``length_penalty``
    made a shorter one the optimum). ``acceptance_by_move`` is accepted over proposed.
    """
    unit = result.energy_unit
    moves = ", ".join(f"{name} {taken}/{tried}"
                      for name, (taken, tried) in result.acceptance_by_move.items())
    return {f"final_energy_{unit}": f"{result.final_energy:.10f}",
            "acceptance_by_move": moves,
            "num_architectures": result.num_architectures}
