"""The calculator side of the quantum methods on a crystal: option checks and native views.

``vqe``, ``adapt-vqe``, ``vasqa`` and ``valsqa`` on a structure with a periodic axis solve the
Born-von Karman supercell of a Gamma-centered ``kpts`` mesh (Gamma alone by default); the Hamiltonian, the unfolded mean-field bands, the spectral function and the
tetrahedron density of states are computed in Rust (``molecular::periodic``,
``quantum::spectral``, ``periodic::tetrahedron``). This module validates options, hands arrays
across and wraps the results; it computes nothing.
"""

from __future__ import annotations

import numpy as np

from .. import _native
from ..units import HARTREE_TO_EV


def _gamma_centered(spec):
    size = tuple(spec.get("size", (1, 1, 1)))
    return bool(spec.get("gamma", False)) or all(n % 2 == 1 for n in size)


def _check_supercell_options(calculation):
    if calculation._basis_spec is None:
        raise NotImplementedError(
            "a PAW-LCAO or UPAW-LCAO basis is not available for the periodic quantum methods "
            "yet: the supercell Hamiltonian is all-electron; use basis='HAO' or a Gaussian recipe"
        )
    if calculation._explicit_grid is not None:
        raise ValueError(
            "the periodic supercell builds its own grid, commensurate with the primitive cell: "
            "set grid_spacing instead of grid"
        )
    if calculation._kinetic != "spectral":
        raise ValueError(
            "the periodic supercell Hamiltonian uses the spectral kinetic operator, which is "
            "periodic on the cell: use kinetic='spectral'"
        )
    if calculation._precision != "float64":
        raise ValueError(
            "the periodic supercell Hamiltonian is built in float64 only: use precision='float64'"
        )


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
    if not _gamma_centered(spec):
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


def refuse_derivatives():
    """Forces and stress of a quantum method on a crystal are not available.

    Only finite differences of the energy could give them, and that energy depends on where the
    atoms sit relative to the fixed periodic grid: on an H2 chain (a = 3.0 A, HAO, 2 k-points)
    a rigid translation of the crystal by 0.001 A changes the energy per cell by 6 meV at
    grid_spacing 0.4 and by 0.65 eV after 0.1 A at grid_spacing 0.2 (the Hartree-Fock reference
    alike), so a central difference returns grid artifacts of several eV/A (measured
    2026-11-10). The analytic route is Stage 6 of the Bloch plan.
    """
    raise NotImplementedError(
        "forces and stress of a quantum method on a crystal are not implemented: a finite "
        "difference of the supercell energy measures the grid alignment of the all-electron "
        "basis (several eV/A), not the force; use method='dft' with a PAW-LCAO basis for crystal "
        "forces and stress"
    )


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
    integrals = getattr(calculation, "integrals", None)
    if integrals is None or calculation.result is None or calculation.native_reference is None:
        return False
    try:
        _native.bloch_supercell(integrals)
    except ValueError:
        return False
    return True


def bands_at(calculation, fractional):
    """Unfolded mean-field eigenvalues (Hartree) and Lowdin weights at fractional k-points."""
    supercell(calculation)
    raw = _native.bloch_bands(
        calculation.native_reference, [list(map(float, k)) for k in fractional]
    )
    return np.asarray(raw["eigenvalues"]), np.asarray(raw["weights"])


def spectral_function(calculation, *, energies=None, eta=0.1, kpoints=None, krylov=200):
    """The Lehmann spectral function of the correlated supercell state at mesh k-points."""
    from ..analysis.bloch import SpectralFunction

    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be positive and finite (eV)")
    if type(krylov) is not int or krylov < 1:
        raise ValueError("krylov must be a positive integer")
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
    points = np.asarray(cell["kpoints"] if kpoints is None else kpoints, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("kpoints must be a sequence of fractional (k1, k2, k3)")
    raw = _native.bloch_spectral_function(
        calculation.native_reference,
        reference.hamiltonian,
        problem,
        np.ascontiguousarray(raw_state, dtype=np.complex128),
        [list(map(float, k)) for k in points],
        eta_ha=eta / HARTREE_TO_EV,
        axis_ha=None if energies is None else np.ascontiguousarray(
            np.asarray(energies, dtype=float) / HARTREE_TO_EV),
        krylov=krylov,
    )
    poles = {
        "energies": np.asarray(raw["energies"]) * HARTREE_TO_EV,
        "weights": np.asarray(raw["weights"]),
        "k": np.asarray(raw["k"]),
        "orbital": np.asarray(raw["orbital"]),
        "spin": np.asarray(raw["spin"]),
        "addition": np.asarray(raw["addition"]),
    }
    removal, addition = raw["removal_edge_ha"], raw["addition_edge_ha"]
    return SpectralFunction(
        kpoints=np.asarray(raw["kpoints"]),
        energies=np.asarray(raw["axis_ha"]) * HARTREE_TO_EV,
        values=np.asarray(raw["values"]) / HARTREE_TO_EV,
        eta=float(eta),
        poles=poles,
        sum_rule=np.asarray(raw["sum_rule"]),
        dominant_energies=np.asarray(raw["dominant_energy"]) * HARTREE_TO_EV,
        dominant_weights=np.asarray(raw["dominant_weight"]),
        removal_edge=removal * HARTREE_TO_EV,
        addition_edge=addition * HARTREE_TO_EV,
        ground_energy=raw["ground_energy_ha"] * HARTREE_TO_EV,
        mesh=tuple(cell["mesh"]),
    )


def primitive_shells(calculation):
    """``(primitive atom, l)`` of every primitive orbital, the PDOS grouping key."""
    cell = supercell(calculation)
    return list(zip(cell["primitive_atom"], cell["primitive_l"]))
