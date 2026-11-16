"""The calculator side of the quantum methods on a crystal: option checks and native views.

``vqe``, ``adapt-vqe``, ``vasqa`` and ``valsqa`` on a structure with a periodic axis solve the
Born-von Karman supercell of a Gamma-centered ``kpts`` mesh (Gamma alone by default), on the
hydrogen-like or Gaussian basis (``molecular::periodic``) or on PAW-LCAO, built from the crystal
setup (``molecular::periodic::paw``); the Hamiltonian, the unfolded mean-field bands, the spectral
function and the tetrahedron density of states are computed in Rust (``quantum::spectral``,
``periodic::tetrahedron``). This module validates options, hands arrays
across and wraps the results; it computes nothing.
"""

from __future__ import annotations

import json

import numpy as np

from .. import _native


def _check_supercell_options(calculation):
    if calculation._explicit_grid is not None:
        raise ValueError("the periodic supercell builds its own grid, commensurate with the "
                         "primitive cell: set grid_spacing instead of grid")
    if calculation._kinetic != "spectral":
        raise ValueError("the periodic supercell Hamiltonian uses the spectral kinetic operator, "
                         "which is periodic on the cell: use kinetic='spectral'")
    if calculation._precision != "float64":
        raise ValueError("the periodic supercell Hamiltonian is built in float64 only: use "
                         "precision='float64'")


def check_geometry(calculation, geometry):
    """Run-time checks of a quantum method on a periodic geometry; returns the mesh size."""
    if not calculation.capabilities.periodic:
        raise ValueError(
            f"method={calculation.method!r} has no periodic form: use vqe, adapt-vqe, vasqa "
            "or valsqa for a crystal"
        )
    if calculation._crystal_orbitals == "k-space":
        raise NotImplementedError(
            "crystal_orbitals='k-space' (Bloch orbitals with momentum-conserving excitations) "
            "is not implemented yet: pass crystal_orbitals='real-space' to solve the supercell "
            "in its real orbitals"
        )
    _check_supercell_options(calculation)
    spec = calculation._kpts_spec
    if not (spec.get("gamma") or all(n % 2 == 1 for n in spec.get("size", (1, 1, 1)))):
        raise ValueError(
            "a quantum method on a crystal solves the Born-von Karman supercell of a "
            "Gamma-centered mesh: pass kpts={'size': (n1, n2, n3), 'gamma': True}"
        )
    size = tuple(int(n) for n in spec.get("size", (1, 1, 1)))
    for axis, (periodic, n) in enumerate(zip(geometry.pbc, size)):
        if not periodic and n != 1:
            raise ValueError(
                f"kpts has {n} points along lattice vector {'abc'[axis]}, which is not "
                "periodic: use 1 there"
            )
    return size


def paw_supercell_integrals(calculation, geometry, size):
    """The PAW-LCAO supercell Hamiltonian of the Gamma-centered mesh ``size``.

    The crystal setup request is the one ``method='dft'`` builds (``setup_block.setup_request``);
    the binding sets the crystal up at Gamma and the supercell forms its mesh itself
    (``molecular::periodic::paw``). The datasets follow the molecular rule, ``lda-sr`` unless
    ``paw_dataset`` names another set; a spin-orbit set is built as a spinor setup, which the
    supercell refuses by name. UPAW-LCAO is refused until it is gated.
    """
    from ..models.setup import pseudopotential_family, resolve_basis
    from ..pseudopotentials.environment import with_functional_set
    from .ase import atoms_from_geometry
    from .setup_block import setup_request, spin_orbit_symbols

    atoms, (name, options) = atoms_from_geometry(geometry), resolve_basis(calculation._basis)
    family = pseudopotential_family(name)
    if family.name != "paw-lcao":
        raise NotImplementedError(f"a {family.name.upper()} basis is not available for the "
                                  "periodic quantum methods yet: use basis='PAW-LCAO' or 'HAO'")
    options = with_functional_set(family.name, options, calculation.xc)
    spinor = bool(spin_orbit_symbols(atoms, options, family.name))
    request, paths = setup_request(calculation, atoms, family, options, [0.0] * len(atoms), spinor)
    # The engine's default is the calculator's (1e-8 eV); only a requested tolerance is passed.
    tolerance = calculation._cholesky_tolerance if calculation._two_body_requested else None
    return _native.paw_supercell_integrals(calculation.context, json.dumps(request), paths, list(
        size), cholesky_tolerance=tolerance)


#: Central-difference step of the crystal forces, Bohr: the engine's.
FORCE_STEP = _native.PERIODIC_FORCE_STEP


def check_derivatives(calculation, properties):
    """Which derivatives of a quantum method on a crystal exist, refused before any work.

    On a PAW-LCAO supercell the forces are analytic (Bloch-VQE stage 6a, the Rust gradient of the
    energy per cell). On the HAO basis they are central finite differences of the supercell energy
    per cell (owner Q2 keeps that route). The HAO energy is exactly
    invariant under a rigid translation of the crystal (band-limited orbitals and a G-space ion
    potential, ``molecular::periodic``: 1e-12 eV per cell), so the forces sum to zero, and a
    displacement only changes phases, so the energy is smooth in the positions. Stress is refused:
    a strain moves the grid's wavevectors against the band limit, and the set of plane waves kept
    changes discontinuously (jumps of 6 to 30 meV per cell in the second difference of E under a
    1 percent uniaxial strain, H2 chain at grid_spacing 0.2 and 0.25, 2026-11-10).
    """
    if "stress" in properties:
        raise NotImplementedError(
            "the stress of a quantum method on a crystal is not implemented: a strain changes the "
            "set of plane waves inside the band limit of the supercell orbitals, so a finite "
            "difference of the energy jumps; use method='dft' with a PAW-LCAO basis for stress"
        )
    if "forces" in properties:
        if calculation._execute_circuits:
            raise NotImplementedError(
                "crystal forces need the state vector (HAO differences its energies, PAW-LCAO "
                "differentiates it); a circuit run (shots, a device, execute_circuits) has none")
        if calculation._basis_spec is not None and calculation.method in (
                "vasqa", "valsqa") and calculation._constructor_options.get("seed") is None:
            raise ValueError(
                "crystal forces of an architecture search difference energies of separate "
                "searches: pass seed= so every displaced search runs the same chain"
            )


def _displaced_copy(calculation):
    """The same calculation for a displaced geometry, without logs, checkpoints or caches."""
    from ..calculation import Mandacaru

    options = dict(calculation._constructor_options)
    options.update(atoms=None, trace=False, txt=None, checkpoint=None, resume=None,
                   save_hamiltonian=None, save_crystal=None, record=None)
    if not isinstance(options["resources"], (dict, type(None))):
        options["resources"] = None
    return Mandacaru(**options)


def finite_difference_forces(calculation, geometry, result):
    """Central-difference forces (eV/A) on the atoms of the primitive cell.

    Displacing a primitive atom moves every image of it, so the derivative of the energy per cell
    is the force on that atom. Each displaced energy is a full calculation with the same options.
    The error is the grid convergence of the band-limited basis (H2 chain, bond derivative:
    -5.155, -5.033, -5.321, -5.153, -5.036, -5.056 eV/A at grid_spacing 0.25, 0.2, 0.15, 0.125,
    0.1, 0.075), not the step (-5.03257, -5.03272, -5.03333 eV/A at 1, 2, 4 mA).

    An active space that splits an exactly degenerate orbital group is refused before any
    displaced run: no deterministic frame covers a periodic grid, so every displaced
    calculation keeps whichever members its eigensolver returned, and the differences are not a
    derivative (LiH chain, 3 k-points, 4 orbitals cutting a +-k pair, 2026-11-12: net force
    0.016 eV/A, components changing with the step, nonzero forces across a mirror plane).
    """
    import itertools

    from ..structure import geometry_value
    from .ase import atoms_from_geometry

    result._reference.require_unsplit(result.hamiltonian, "the crystal force of a quantum method")

    atoms = atoms_from_geometry(geometry)
    step = _native.to_user_units(float(FORCE_STEP), "length")
    energies = {}
    for index, axis, sign in itertools.product(range(len(atoms)), range(3), (1.0, -1.0)):
        moved = atoms.copy()
        moved.positions = _native.displaced(
            np.ascontiguousarray(atoms.positions, dtype=float), index, axis, sign, step)
        copy = _displaced_copy(calculation)
        copy.run(geometry_value(moved))
        energies[index, axis, sign] = _native.energy_to_hartree(float(copy.results["energy"]), "eV")
    plus = [[energies[i, a, 1.0] for a in range(3)] for i in range(len(atoms))]
    minus = [[energies[i, a, -1.0] for a in range(3)] for i in range(len(atoms))]
    raw = _native.periodic_quantum_forces_fd(plus, minus, FORCE_STEP)
    result.force_details = {"scheme": raw["scheme"], "step_bohr": FORCE_STEP}
    return _native.to_user_units(np.ascontiguousarray(raw["forces"], dtype=float), "force")


def supercell(calculation):
    """The native description of the last supercell (mesh, k-points, orbital map)."""
    integrals = getattr(calculation, "integrals", None)
    if integrals is None or calculation.result is None:
        raise RuntimeError("run a quantum calculation on a crystal first")
    try:
        return _native.bloch_supercell(integrals)
    except ValueError:
        raise RuntimeError(
            "the last calculation was not a periodic supercell: k-resolved views need a "
            "quantum method on a crystal"
        ) from None


def is_supercell(calculation):
    try:
        return calculation.native_reference is not None and bool(supercell(calculation))
    except (AttributeError, RuntimeError):
        return False


def bands_at(calculation, fractional):
    """Unfolded mean-field eigenvalues (Hartree) and Lowdin weights at fractional k-points."""
    supercell(calculation)
    raw = _native.bloch_bands(calculation.native_reference,
                              [list(map(float, k)) for k in fractional])
    return np.asarray(raw["eigenvalues"]), np.asarray(raw["weights"])


def spectral_function(calculation, *, energies=None, eta=0.1, kpoints=None, krylov=200,
                      path=None, irreducible=False, symprec=1e-5):
    """The Lehmann spectral function of the correlated supercell state at mesh k-points,
    or along a band path (``path``): the mesh points on it, in path order."""
    import warnings

    from ..analysis.bloch import SpectralFunction
    from ..analysis.kspace import KPath
    from . import kspace

    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be positive and finite (eV)")
    if type(krylov) is not int or krylov < 1:
        raise ValueError("krylov must be a positive integer")
    if path is not None and kpoints is not None:
        raise ValueError("give either kpoints or a band path, not both")
    if irreducible and path is None:
        raise ValueError(
            "irreducible=True reduces the points of a band path: pass path=, e.g. path='GXMG'"
        )
    cell = supercell(calculation)
    if calculation._shots:
        raise NotImplementedError(
            "the spectral function is computed from the state vector; a shot-based run has none"
        )
    problem = calculation.problem
    raw_state = calculation.result._raw.get("state")
    if problem is None or raw_state is None:
        raise RuntimeError("the spectral function needs the state of a single-state variational run")
    reference = getattr(calculation, "_bloch_reference", None)
    if reference is None:
        raise RuntimeError("run a quantum calculation on a crystal first")
    state = np.ascontiguousarray(raw_state, dtype=np.complex128)
    eta_ha = _native.energy_to_hartree(float(eta), "eV")
    axis_ha = None if energies is None else _native.energy_to_hartree(
        np.ascontiguousarray(energies, dtype=float), "eV")
    if path is None:
        points = np.asarray(cell["kpoints"] if kpoints is None else kpoints, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("kpoints must be a sequence of fractional (k1, k2, k3)")
        raw = _native.bloch_spectral_function(
            calculation.native_reference, reference.hamiltonian, problem, state,
            [list(map(float, k)) for k in points],
            eta_ha=eta_ha, axis_ha=axis_ha, krylov=krylov,
        )
    else:
        lattice, pbc, positions, numbers = kspace.structure(calculation)
        text, explicit = kspace.path_arguments(path)
        raw = _native.bloch_path_spectral_function(
            calculation.native_reference, reference.hamiltonian, problem, state,
            lattice, list(pbc),
            eta_ha=eta_ha, path=text, path_points=explicit, irreducible=bool(irreducible),
            positions=positions, numbers=numbers, symprec=float(symprec),
            axis_ha=axis_ha, krylov=krylov,
        )
    kpath = None if path is None else KPath.from_native(raw["kpath"])
    audit = raw.get("audit")
    if kpath is not None and not kpath.complete:
        warnings.warn(
            f"the band path does not pass through {', '.join(kpath.missing)} on this k-mesh: "
            "the spectral function is resolved at the mesh points on the path only "
            "(use a mesh that contains them)", RuntimeWarning, stacklevel=3)
    if audit is not None and not audit["symmetric"]:
        warnings.warn(
            f"the state is not symmetric under the point group of the crystal: its spectral "
            f"function at k = {audit['image']} differs from that of "
            f"the irreducible point {audit['representative']} by "
            f"{audit['residual']:.3g} (relative L1 over the axis), yet irreducible=True gave the "
            "irreducible point's spectrum to all of its star; resolve the full path with "
            "irreducible=False", RuntimeWarning, stacklevel=3)
    poles = {
        "energies": _native.to_user_units(np.ascontiguousarray(raw["energies"], dtype=float), "energy"),
        "weights": np.asarray(raw["weights"]),
        "k": np.asarray(raw["k"]),
        "orbital": np.asarray(raw["orbital"]),
        "spin": np.asarray(raw["spin"]),
        "addition": np.asarray(raw["addition"]),
    }
    return SpectralFunction(
        kpoints=np.asarray(raw["kpoints"]),
        energies=_native.to_user_units(np.ascontiguousarray(raw["axis_ha"], dtype=float), "energy"),
        # A density per Hartree to per eV: the same division by the factor.
        values=_native.energy_to_hartree(np.ascontiguousarray(raw["values"], dtype=float), "eV"),
        eta=float(eta),
        poles=poles,
        sum_rule=np.asarray(raw["sum_rule"]),
        dominant_energies=_native.to_user_units(np.ascontiguousarray(raw["dominant_energy"], dtype=float), "energy"),
        dominant_weights=np.asarray(raw["dominant_weight"]),
        removal_edge=_native.to_user_units(float(raw["removal_edge_ha"]), "energy"),
        addition_edge=_native.to_user_units(float(raw["addition_edge_ha"]), "energy"),
        chemical_potential=_native.to_user_units(float(raw["chemical_potential_ha"]), "energy"),
        gap=_native.to_user_units(float(raw["gap_ha"]), "energy"),
        ground_energy=_native.to_user_units(float(raw["ground_energy_ha"]), "energy"),
        mesh=tuple(cell["mesh"]),
        kpath=kpath,
        audit=audit,
    )


def primitive_shells(calculation):
    """``(primitive atom, l)`` of every primitive orbital, the PDOS grouping key."""
    cell = supercell(calculation)
    return list(zip(cell["primitive_atom"], cell["primitive_l"]))
