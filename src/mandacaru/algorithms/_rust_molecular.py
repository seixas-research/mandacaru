"""ASE snapshots, option normalization and result views for native molecules."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np

from ._rust_quantum import _RustDriver, _integer
from .mean_field import MeanFieldResult
from ..units import ANGSTROM_TO_BOHR, HARTREE_TO_EV


def basis_specification(spec):
    from ._hamiltonian_from_atoms import resolve_basis, PER_ELEMENT
    from ..basis.gaussian_families import GaussianRecipe, parse_basis_name
    name, options = resolve_basis(spec)
    if name == PER_ELEMENT:
        from ase.data import atomic_numbers
        elements = {}
        for element, family in options.items():
            key = "*" if element == "*" else str(atomic_numbers[element] if isinstance(element, str) else int(element))
            elements[key] = basis_specification(family)
        return {"name": "per_element", "elements": elements}
    key = name.upper().replace(" ", "")
    if key in ("HAO", "FULLATOMICORBITALS"):
        if set(options) - {"virtual_orbitals"}:
            raise TypeError("native HAO options: virtual_orbitals")
        return {"name": "HAO", "virtual_orbitals": _integer(options.get("virtual_orbitals", 0), "virtual_orbitals")}
    if key == "GTO":
        if set(options) - {"n_gaussians"}:
            raise TypeError("native GTO options: n_gaussians")
        n = _integer(options.get("n_gaussians", 3), "n_gaussians", 1)
        if n > 10:
            raise ValueError("native Gaussian fits contain at most 10 primitives")
        recipe = GaussianRecipe(name=f"STO-{n}G", family="sto", core=(n,), valence=(n,))
    else:
        if options:
            raise TypeError("native named Gaussian recipes take no extra basis options")
        aliases = {"631G(D)": "6-31G(d)", "631G*": "6-31G*", "6-31GD": "6-31G(d)"}
        try:
            recipe = parse_basis_name(aliases.get(key, name))
        except ValueError as exc:
            raise NotImplementedError("the native molecular engine supports HAO and Mandacaru's Gaussian recipes") from exc
    return {"name": "gaussian", "recipe": asdict(recipe)}


def configure_geometry(driver, atoms):
    from .. import _native
    from .._native_adapter import geometry_from_atoms
    from ._hamiltonian_from_atoms import coherent_positions

    start = perf_counter()
    # Unwrapping is an ASE integration operation; numerical work begins with
    # an owned snapshot in Bohr and no retained ASE object in the engine.
    work = atoms.copy()
    work.positions = coherent_positions(atoms)
    geometry = geometry_from_atoms(work)
    driver.context = _native.Context(**driver.options)
    grid = driver._explicit_grid
    if grid is not None and not isinstance(grid, _native.SpatialGrid):
        grid = _native.SpatialGrid(
            grid.shape, [float(grid.X.flat[0]), float(grid.Y.flat[0]), float(grid.Z.flat[0])],
            np.ascontiguousarray(grid.step, dtype=float), periodic=bool(grid.periodic))
    driver.integrals = driver.context.molecular_integrals(
        geometry, basis_spec=json.dumps(driver._basis_spec),
        spacing_bohr=driver._grid_spacing * ANGSTROM_TO_BOHR, grid=grid,
        kinetic=driver._kinetic, cholesky_tolerance=driver._cholesky_tolerance)
    unresolved = [i for i, ratio in enumerate(driver.integrals.resolution_ratios) if abs(ratio - 1.) > .25]
    if unresolved:
        warnings.warn(f"the real-space grid does not resolve {len(unresolved)} native basis function(s) to 25% kinetic accuracy (indices {unresolved}); refine grid_spacing before interpreting energies", RuntimeWarning, stacklevel=3)
    particles = driver.integrals.particles(
        charge=driver._charge, n_electrons=driver._n_electrons,
        magnetic_moment=float(np.sum(atoms.get_initial_magnetic_moments())))
    kind = getattr(driver, "kind", "rhf" if particles[0] == particles[1] else "uhf")
    driver.native_reference = driver.integrals.mean_field(kind, particles, **driver._scf_options)
    if hasattr(driver, "kind"):
        driver.reference_snapshot = driver.native_reference.snapshot()
    driver.native_model = (driver.native_reference.model if driver._active_spec is None
                           else driver.native_reference.active_model(json.dumps(driver._active_spec)))
    driver.num_particles = tuple(driver.native_model.num_particles)
    driver.n_spatial_orbitals = driver.native_model.n_spatial_orbitals
    driver.mapping = driver._requested_mapping or "jordan_wigner"
    driver.n_qubits = 2 * driver.n_spatial_orbitals - (2 if driver.mapping == "parity_reduced" else 0)
    if driver._save_hamiltonian:
        path = ("hamiltonian.json" if driver._save_hamiltonian is True
                else str(driver._save_hamiltonian))
        if not Path(path).suffix:
            path += ".json"
        driver.native_model.save(path, mapping=driver.mapping)
    driver._geometry_seconds = perf_counter() - start
    if not hasattr(driver, "kind"):
        if not driver.native_reference.converged:
            raise RuntimeError("the native mean-field reference did not converge; increase scf_options['max_iterations'] before a quantum solve")
        driver._configure()


class NativeMeanFieldResult(MeanFieldResult):
    """Existing result fields with native mapping and state preparation."""

    def __repr__(self):
        return f"MeanFieldResult(method={self.method!r}, energy={self.optimal_energy:.9f} {self.energy_unit}, success={self.success})"

    def qubit_hamiltonian(self, mapping="jordan_wigner"):
        from ..core.mapping import PauliSum
        operator = self._native_model.operator(self._check_mapping(mapping))
        return PauliSum({p: complex(re, im) for p, re, im in operator.terms}, num_qubits=operator.num_qubits)

    def reference_state(self, mapping="jordan_wigner"):
        return self._native_model.reference_state(self._check_mapping(mapping))

    def as_quantum_problem(self):
        return {"engine": "rust", "native_model": self._native_model}

    def scf_state(self, mapping="jordan_wigner"):
        if self.method == "uhf":
            if self._native_model.n_spatial_orbitals != self._native_reference.model.n_spatial_orbitals:
                raise NotImplementedError("the native SCF determinant with an active space is not yet wired")
            return self._native_reference.state(mapping)
        return self.reference_state(mapping)


class _RustMeanField(_RustDriver):
    classical_mean_field = True
    analytic_gradient = False

    def __init__(self, **kwargs):
        if kwargs.get("load_hamiltonian") is not None or kwargs.get("native_model") is not None:
            raise ValueError("native mean-field calculations require a geometry and spatial integrals")
        if kwargs.get("optimizer") is not None or "sector" in kwargs or "sparse" in kwargs:
            raise TypeError("native mean-field calculations use scf_options and two_body; optimizer, sector and sparse select quantum solves")
        self.include_singles, self.trotter = True, False
        super().__init__(**kwargs)
        if self.kind == "ghf" and self._requested_mapping == "parity_reduced":
            raise ValueError("the GHF spinor Hamiltonian has no fixed alpha parity; use jordan_wigner, parity or bravyi_kitaev")

    @property
    def hamiltonian(self):
        return None if self.result is None else self.result.fermion_hamiltonian

    def run(self, geometry=None, cell=None):
        from .hartree_fock import RHFResult, UHFResult, GHFResult
        if self.dry_run:
            raise NotImplementedError("the native molecular engine does not yet provide dry-run estimates")
        if geometry is not None:
            if cell is not None:
                geometry = geometry.copy()
                geometry.set_cell(cell)
            configure_geometry(self, geometry)
        if not hasattr(self, "reference_snapshot"):
            raise ValueError("mean-field run() requires geometry=Atoms or an attached ASE calculator")
        raw = self.reference_snapshot
        common = dict(electronic_energy=raw["electronic_energy"], converged=raw["converged"],
                      n_iterations=raw["n_iterations"], h_mo=raw["h_mo"], eri_mo=raw["eri_mo"],
                      history=[dict(iter=i, energy=e, dE=de, residual=r) for i, e, de, r in raw["history"]])
        na, nb = raw["num_particles"]
        if self.kind == "rhf":
            scf = RHFResult(**common, mo_energies=np.asarray(raw["energies"][0]),
                            mo_coefficients=raw["coefficients"][0], n_occupied=na)
        elif self.kind == "uhf":
            scf = UHFResult(**common, n_alpha=na, n_beta=nb,
                            mo_energies_alpha=np.asarray(raw["energies"][0]), mo_energies_beta=np.asarray(raw["energies"][1]),
                            mo_coefficients_alpha=raw["coefficients"][0], mo_coefficients_beta=raw["coefficients"][1],
                            natural_occupations=np.asarray(raw["natural_occupations"]), natural_orbitals=raw["model_orbitals"],
                            reference_energy=raw["reference_energy"])
        else:
            scf = GHFResult(**common, n_electrons=na + nb, mo_energies=np.asarray(raw["energies"][0]),
                            mo_coefficients=raw["coefficients"][0], mode_coefficients=raw["model_orbitals"],
                            kramers_pairing=raw["kramers_pairing"])
        self.fermion_hamiltonian = _HamiltonianView(self.native_model)
        self._scf = scf
        self.result = NativeMeanFieldResult(
            method=self.kind, optimal_energy=(raw["electronic_energy"] + raw["nuclear_repulsion"]) * self._scale,
            reference_energy=(raw["reference_energy"] + raw["nuclear_repulsion"]) * self._scale,
            scf=scf, model_orbitals=raw["model_orbitals"][:, self.native_model.orbital_indices], fermion_hamiltonian=self.fermion_hamiltonian,
            num_particles=self.num_particles, n_spatial_orbitals=self.n_spatial_orbitals, energy_unit=self.energy_unit,
            timings={"engine": "rust", "geometry_seconds": self._geometry_seconds,
                     "memory_peak_bytes": self.context.memory_peak_bytes, "threads": self.context.threads})
        self.result._native_model = self.native_model
        self.result._native_reference = self.native_reference
        report = f"Rust {self.kind.upper()}: {self.result.optimal_energy:.12f} {self.energy_unit}; converged={self.result.success}\n"
        if self.verbose:
            print(report, end="")
        if self.txt is not None:
            Path(self.txt).write_text(report)
        return self.result

    def mean_field_rdm(self):
        if self._active_spec is not None:
            raise NotImplementedError("native mean-field RDM export with an active space is not wired")
        return self.native_reference.rdm()


class RustRHF(_RustMeanField):
    kind = "rhf"


class RustUHF(_RustMeanField):
    kind = "uhf"


class RustGHF(_RustMeanField):
    kind = "ghf"


class _HamiltonianView:
    """Materialize a Python operator only when its public result field is read."""

    def __init__(self, model):
        self.model = model

    def build(self):
        from ..core.mapping import Fermion
        return Fermion({tuple(l): complex(re, im) for l, re, im in self.model.fermion_terms()},
                       n_modes=2 * self.model.n_spatial_orbitals)
