"""Native calculation orchestration, independent of ASE and quantum SDKs."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from . import _native
from .resources import CalculationResources
from .results import CalculationResult
from .specifications import (
    METHODS,
    FUNCTIONALS,
    MAPPINGS,
    basis_specification,
)
from .structure import geometry_value
from .units import ANGSTROM_TO_BOHR

DEFAULT_METHOD = "adapt-vqe"
STABLE_METHODS = tuple(METHODS)


def _method_key(name):
    return str(name).lower().replace("_", "-")


def resolve_method(name):
    if name not in METHODS:
        raise ValueError(
            f"unsupported native method {name!r}; available: {', '.join(METHODS)}"
        )
    return name, METHODS[name]


class Mandacaru:
    """One Rust calculation and its owned state, resources and result views.

    ``run(Geometry(...))`` needs no ASE installation. Geometry-protocol objects
    can also be supplied. An explicit ASE Calculator adapter lives in
    ``mandacaru.integrations.ase``. Scientific arrays remain owned by Rust.
    """

    def __init__(
        self,
        method=DEFAULT_METHOD,
        *,
        basis="HAO",
        grid_spacing=0.2,
        grid=None,
        resources=None,
        xc=None,
        spin=False,
        unrestricted=False,
        charge=0,
        n_electrons=None,
        scf_options=None,
        optimizer=None,
        mapping=None,
        active_space=None,
        load_hamiltonian=None,
        save_hamiltonian=None,
        native_model=None,
        checkpoint=None,
        resume=None,
        pool="fermionic",
        ansatz="uccsd",
        trotter=False,
        include_singles=True,
        sector=True,
        sparse=False,
        convergence=None,
        max_iterations=50,
        weights=None,
        num_states=2,
        atomic_units=False,
        trace=False,
        txt=None,
        kinetic="spectral",
        cholesky_tolerance=1e-10,
        project_translation=False,
        kpts=None,
        smearing=None,
        periodic_scf_options=None,
        symmetry=True,
        load_crystal=None,
        save_crystal=None,
    ):
        self._method, self._capabilities = resolve_method(method)
        if xc is not None and xc not in FUNCTIONALS:
            raise ValueError(f"expected a canonical XC name in {FUNCTIONALS}")
        if mapping is not None and mapping not in MAPPINGS:
            raise ValueError(f"expected a canonical mapping in {MAPPINGS}")
        if pool != "fermionic" or ansatz != "uccsd":
            raise NotImplementedError(
                "native quantum calculations provide the fermionic pool and UCCSD ansatz"
            )
        if project_translation is not False:
            raise NotImplementedError(
                "forces are the stationary native gradient without translation projection"
            )
        if not np.isfinite(grid_spacing) or grid_spacing <= 0:
            raise ValueError("grid_spacing must be positive and finite")
        if type(charge) is not int:
            raise TypeError("charge must be an integer")
        for value in (
            spin,
            unrestricted,
            trotter,
            include_singles,
            sector,
            sparse,
            atomic_units,
            trace,
            symmetry,
        ):
            if type(value) is not bool:
                raise TypeError("boolean calculation controls require booleans")
        if native_model is not None and resources:
            raise ValueError(
                "a supplied model retains its context; configure resources on its originating calculation"
            )
        self.resources = CalculationResources(
            **dict(resources or {}),
            context=None if native_model is None else native_model.context,
        )
        self.context = self.resources.context
        from copy import deepcopy

        basis = deepcopy(basis)
        self._basis = basis
        self._basis_spec = (
            None
            if (basis if isinstance(basis, str) else basis.get("name"))
            in ("PAW-LCAO", "UPAW-LCAO")
            else basis_specification(basis)
        )
        self._requested_xc = xc
        self._xc = "lda" if xc is None else xc
        self._unrestricted = spin or unrestricted
        self._charge = charge
        self._n_electrons = n_electrons
        self._scf_options = dict(scf_options or {})
        self._mapping = mapping or "jordan_wigner"
        self._active_spec = active_space
        self._grid_spacing = float(grid_spacing)
        self._explicit_grid = grid
        self._kinetic = kinetic
        self._cholesky_tolerance = cholesky_tolerance
        self._load_hamiltonian = load_hamiltonian
        self._save_hamiltonian = save_hamiltonian
        if (
            sum(x is not None for x in (native_model, load_hamiltonian, load_crystal))
            > 1
        ):
            raise ValueError("supply only one native model or cached problem")
        self._supplied_model = native_model
        self._checkpoint_path = checkpoint
        self._resume_path = resume
        self._trotter = trotter
        self._include_singles = include_singles
        self._sector = sector
        self._sparse = sparse
        self._max_growth = max_iterations
        if type(num_states) is not int or num_states < 1:
            raise ValueError("num_states must be a positive integer")
        self._weights = (
            np.arange(num_states, 0, -1, dtype=float)
            if weights is None
            else np.asarray(weights, dtype=float).copy()
        )
        if (
            self._weights.shape != (num_states,)
            or not np.isfinite(self._weights).all()
            or np.any(self._weights <= 0)
        ):
            raise ValueError(
                "weights must contain one positive finite value per requested state"
            )
        self._convergence = dict(convergence or {})
        self._atomic_units = atomic_units
        self.trace = trace
        self.txt = txt
        self._optimizer = {
            "method": "BFGS",
            **(
                dict(method=optimizer)
                if isinstance(optimizer, str)
                else dict(optimizer or {})
            ),
        }
        if self._optimizer["method"] not in ("BFGS", "L-BFGS"):
            raise ValueError("native optimizers are BFGS and L-BFGS")
        if set(self._optimizer) - {"method", "maxiter", "maxfev", "tol"}:
            raise ValueError("optimizer options are method, maxiter, maxfev and tol")
        if set(self._convergence) - {"gradient", "energy"}:
            raise ValueError("convergence options are gradient and energy")
        if not self.capabilities.quantum and (
            load_hamiltonian is not None or native_model is not None
        ):
            raise ValueError(
                "mean-field calculations require a geometry or cached crystal"
            )
        self._geometry = None
        self._attached_atoms = None
        self._force_grid = None
        self.integrals = None
        self.native_reference = None
        self.problem = None
        self.result = None
        self.results = {}
        self.checkpoint = None
        self._load_crystal = load_crystal
        self._save_crystal = save_crystal
        self._kpts_spec = dict(kpts or {"size": (1, 1, 1), "gamma": True})
        self._periodic_options = dict(periodic_scf_options or {})
        self._periodic_symmetry = symmetry
        self._periodic_basis = basis
        self._periodic_two_body_requested = False
        self._requested_mapping = mapping
        self._save_hamiltonian = save_hamiltonian
        self._periodic_atoms = None
        self.options = {
            "threads": self.context.threads,
            "memory_limit_bytes": self.context.memory_limit_bytes,
        }
        if smearing is not None:
            from numbers import Real
            from .units import HARTREE_TO_EV

            if isinstance(smearing, dict):
                if set(smearing) - {"method", "width"}:
                    raise ValueError("smearing accepts method and width")
                method, width = (
                    smearing.get("method", "fermi-dirac"),
                    smearing.get("width", 0.1),
                )
            else:
                method, width = (
                    ("fermi-dirac", smearing)
                    if isinstance(smearing, Real)
                    else smearing
                )
            self._periodic_options.update(
                smearing=method, width_ha=float(width) / HARTREE_TO_EV
            )

    @property
    def method(self):
        return self._method

    @property
    def capabilities(self):
        return self._capabilities

    @property
    def xc(self):
        return self._xc

    @property
    def basis(self):
        import copy

        return copy.deepcopy(self._basis)

    @property
    def implemented_properties(self):
        values = ["energy", "free_energy"]
        if (
            self.capabilities.forces
            and self._kinetic == "spectral"
            and self._load_crystal is None
            and not (self._geometry is not None and any(self._geometry.pbc))
        ):
            values.append("forces")
        return values

    def _grid(self, geometry):
        grid = self._force_grid if self._force_grid is not None else self._explicit_grid
        if grid is None:
            return None
        if isinstance(grid, _native.SpatialGrid):
            return grid
        return _native.SpatialGrid(
            grid.shape,
            [float(grid.X.flat[0]), float(grid.Y.flat[0]), float(grid.Z.flat[0])],
            np.ascontiguousarray(grid.step),
            periodic=bool(grid.periodic),
        )

    def _mean_field(self, geometry):
        if self._basis_spec is None:
            raise NotImplementedError(
                "molecular PAW setup has not been migrated to the native geometry interface"
            )
        self.integrals = self.context.molecular_integrals(
            geometry.native(),
            basis_spec=json.dumps(self._basis_spec),
            spacing_bohr=self._grid_spacing * ANGSTROM_TO_BOHR,
            grid=self._grid(geometry),
            kinetic=self._kinetic,
            cholesky_tolerance=self._cholesky_tolerance,
        )
        particles = self.integrals.particles(
            charge=self._charge,
            n_electrons=self._n_electrons,
            magnetic_moment=float(np.sum(geometry.magnetic_moments)),
        )
        if self.method == "dft":
            self.native_reference = self.integrals.kohn_sham(
                self.xc,
                particles,
                unrestricted=self._unrestricted or particles[0] != particles[1],
                **self._scf_options,
            )
        else:
            kind = (
                self.method
                if self.method in ("rhf", "uhf", "ghf")
                else ("rhf" if particles[0] == particles[1] else "uhf")
            )
            self.native_reference = self.integrals.mean_field(
                kind, particles, **self._scf_options
            )
        return CalculationResult(
            method=self.method,
            context=self.context,
            reference=self.native_reference,
            integrals=self.integrals,
            mapping=self._mapping,
            active_space=self._active_spec,
            atomic_units=self._atomic_units,
        )

    def _quantum(self, reference, initial_parameters):
        if self._load_hamiltonian is not None:
            with Path(self._load_hamiltonian).open("rb") as source:
                columnar = source.read(4) == b"PAR1"
            if columnar:
                from .integrations.columnar import hamiltonian_snapshot

                snapshot = hamiltonian_snapshot(self.context, self._load_hamiltonian)
            else:
                snapshot = self.context.read_snapshot(
                    str(self._load_hamiltonian), "mandacaru-qubit-hamiltonian"
                )
            self.problem = snapshot.problem(
                sector=self._sector,
                sparse=self._sparse,
                include_singles=self._include_singles,
            )
            if (
                self._requested_mapping is not None
                and self.problem.mapping != self._mapping
            ):
                raise ValueError("mapping disagrees with the cached Hamiltonian")
            self._mapping = self.problem.mapping
        else:
            model = (
                self._supplied_model
                if self._supplied_model is not None
                else reference.hamiltonian
            )
            self.problem = model.problem(
                mapping=self._mapping,
                sector=self._sector,
                sparse=self._sparse,
                include_singles=self._include_singles,
            )
        if reference is None and self._save_hamiltonian:
            if self._supplied_model is not None:
                self._supplied_model.save(
                    str(self._save_hamiltonian), mapping=self._mapping
                )
            else:
                snapshot.save(str(self._save_hamiltonian))
        opts = dict(
            adapt=self.method.endswith("adapt-vqe"),
            trotter=self._trotter,
            optimizer=self._optimizer["method"],
            max_iterations=self._optimizer.get("maxiter", 300),
            max_evaluations=self._optimizer.get("maxfev", 5000),
            tolerance=self._optimizer.get("tol", 1e-9),
            max_growth=self._max_growth,
            gradient_tolerance=self._convergence.get("gradient", 1e-7),
            energy_tolerance=self._convergence.get("energy"),
            initial_parameters=initial_parameters,
        )
        if self.method.startswith("subspace-"):
            if self._resume_path is not None or self._checkpoint_path is not None:
                raise NotImplementedError(
                    "subspace checkpoint restart is not implemented"
                )
            raw = self.problem.solve_subspace(
                [2.0, 1.0] if self._weights is None else self._weights, **opts
            )
        else:
            resumed = (
                None
                if self._resume_path is None
                else self.context.read_snapshot(
                    str(self._resume_path), "mandacaru-wavefunction-checkpoint"
                )
            )
            raw = self.problem.solve(resume=resumed, **opts)
            self.checkpoint = self.problem.checkpoint(
                raw["indices"],
                raw["parameters"].tolist(),
                product_form=opts["adapt"] or self._trotter,
                energy_ha=raw["energy_ha"],
                method=self.method,
                status_json=json.dumps(
                    {
                        "complete": True,
                        "converged": raw["success"],
                        "termination": raw["message"],
                    }
                ),
            )
            if self._checkpoint_path is not None:
                self.checkpoint.save(str(self._checkpoint_path))
        return CalculationResult(
            method=self.method,
            context=self.context,
            problem=self.problem,
            raw=raw,
            mapping=self._mapping,
            atomic_units=self._atomic_units,
        )

    def _periodic(self, geometry):
        if self.method != "dft":
            raise NotImplementedError(
                "periodic geometry calculations currently provide native DFT"
            )
        if self.xc not in ("lda", "pbe"):
            raise NotImplementedError("periodic SCF currently provides lda and pbe")
        if self._load_crystal is not None:
            self.periodic_problem = self.context.load_crystal(str(self._load_crystal))
        else:
            from .integrations.ase import atoms_from_geometry
            from .integrations.paw import configure

            configure(self, atoms_from_geometry(geometry))
        if (
            self._requested_xc is not None
            and self.xc != self.periodic_problem.functional
        ):
            raise ValueError("xc disagrees with the crystal functional")
        self._xc = self.periodic_problem.functional
        if self._save_crystal:
            self.periodic_problem.save(str(self._save_crystal))
        periodic = self.periodic_problem.solve(**self._periodic_options)
        return CalculationResult(
            method="dft",
            context=self.context,
            raw=periodic.snapshot(),
            periodic=periodic,
            crystal=self.periodic_problem,
            atomic_units=self._atomic_units,
        )

    def run(self, geometry=None, *, initial_parameters=None):
        with self.resources.lock:
            if self.resources.closed:
                raise RuntimeError("calculator is closed")
            value = geometry_value(geometry) if geometry is not None else self._geometry
            if value is not None:
                self._geometry = value
            if (value is not None and any(value.pbc)) or self._load_crystal is not None:
                result = self._periodic(value)
            elif value is not None:
                reference = self._mean_field(value)
                if not reference.success:
                    raise RuntimeError("native SCF did not converge")
                if self._save_hamiltonian:
                    reference.save_hamiltonian(self._save_hamiltonian)
                result = (
                    self._quantum(reference, initial_parameters)
                    if self.capabilities.quantum
                    else reference
                )
            elif self.capabilities.quantum and (
                self._load_hamiltonian is not None or self._supplied_model is not None
            ):
                result = self._quantum(None, initial_parameters)
            else:
                raise ValueError(
                    "provide a geometry, cached Hamiltonian, or cached crystal"
                )
            self.result = result
            self.results = {
                "energy": result.in_units("eV"),
                "free_energy": result.free_energy_ev,
            }
            report = f"{self.method}: {result.optimal_energy:.12f} {result.energy_unit}; converged={result.success}; {result.timings['termination']}\n"
            if self.trace:
                print(report, end="")
            if self.txt is not None:
                Path(self.txt).write_text(report)
            return result

    def calculate(self, atoms=None, properties=("energy",), system_changes=None):
        with self.resources.lock:
            if self.resources.closed:
                raise RuntimeError("calculator is closed")
            source = (
                atoms
                if atoms is not None
                else (
                    self._attached_atoms
                    if self._attached_atoms is not None
                    else self._geometry
                )
            )
            geometry = geometry_value(source)
            if "forces" in properties and (
                not self.capabilities.forces
                or any(geometry.pbc)
                or self._kinetic != "spectral"
            ):
                raise NotImplementedError(
                    "forces require a molecular RHF, UHF or DFT calculation with spectral kinetic energy"
                )
            reusable = self.result is not None and geometry.same_as(self._geometry)
            if "forces" in properties and self._force_grid is None:
                self._force_grid = (
                    self.integrals.grid
                    if reusable and self.integrals is not None
                    else self._grid(geometry)
                ) or _native.SpatialGrid.from_geometry(
                    geometry.native(), [self._grid_spacing * ANGSTROM_TO_BOHR] * 3
                )
            result = self.result if reusable else self.run(geometry)
            if "forces" in properties:
                self.results["forces"] = result.forces()
            return result

    def set_atoms(self, atoms):
        self._attached_atoms = atoms

    def get_property(self, name, atoms=None, allow_calculation=True):
        with self.resources.lock:
            if self.resources.closed:
                raise RuntimeError("calculator is closed")
            if name not in self.implemented_properties:
                raise NotImplementedError(
                    f"{name} is not implemented for {self.method}"
                )
            geometry = None if atoms is None else geometry_value(atoms)
            changed = geometry is not None and not geometry.same_as(self._geometry)
            if self.result is None or changed or name not in self.results:
                if not allow_calculation:
                    return None
                self.calculate(
                    atoms if atoms is not None else self._attached_atoms,
                    properties=(name,),
                )
            value = self.results[name]
            return (
                np.array(value, copy=True) if isinstance(value, np.ndarray) else value
            )

    def get_potential_energy(self, atoms=None, force_consistent=False):
        return self.get_property("energy", atoms)

    def get_forces(self, atoms=None):
        return self.get_property("forces", atoms)

    def reset(self):
        with self.resources.lock:
            self.result = None
            self.results = {}
            self._geometry = None
            self.integrals = None
            self.native_reference = None
            self.problem = None
            self._force_grid = None
            for name in ("periodic_problem", "_crystal_adapter", "_crystal_metadata"):
                if hasattr(self, name):
                    delattr(self, name)

    def close(self):
        with self.resources.lock:
            self.reset()
            self.resources.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
