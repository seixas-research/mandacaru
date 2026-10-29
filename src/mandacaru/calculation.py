"""ASE calculator and user interface for Rust-owned calculations."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, PropertyNotImplementedError
from .integrations.ase_properties import ASEProperties

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


class Mandacaru(ASEProperties, Calculator):
    """ASE calculator with scientific state and numerical algorithms owned by Rust.

    Attach directly with ``atoms.calc = Mandacaru(...)``. ``run(Geometry(...))``
    also accepts standalone geometry values. Each calculator owns its resources.
    """

    def __init__(
        self,
        method=DEFAULT_METHOD,
        *,
        atoms=None,
        hamiltonian=None,
        n_spatial_orbitals=None,
        num_particles=None,
        hamiltonian_format=None,
        basis="HAO",
        grid_spacing=0.2,
        precision="double",
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
        transfer=False,
        transfer_threshold=0.9,
        transfer_steps=None,
        update_between_geometries=False,
        pool="fermionic",
        ansatz="uccsd",
        trotter=False,
        include_singles=True,
        sector=None,
        sparse=False,
        convergence=None,
        max_iterations=50,
        weights=None,
        num_states=2,
        atomic_units=False,
        trace=False,
        txt=None,
        gradient="analytic",
        profile=None,
        execute_circuits=False,
        backend_provider=None,
        device="AER_simulator",
        shots=0,
        backend_options=None,
        measurement_budget=None,
        max_steps=None,
        min_length=0,
        max_length=20,
        move_weights=None,
        temperature=None,
        length_penalty=0.0,
        warm_start=True,
        replace_start="zero",
        screen_insertions=0,
        proposal_temperature=0.2,
        proposal="gradient",
        proposal_model=None,
        record=None,
        seed=None,
        kinetic="spectral",
        cholesky_tolerance=1e-10,
        two_body=None,
        paw_dataset=None,
        project_translation="auto",
        kpts=None,
        smearing=None,
        periodic_scf_options=None,
        symmetry=True,
        load_crystal=None,
        save_crystal=None,
    ):
        super().__init__()
        self._method, self._capabilities = resolve_method(method)
        if precision not in ("double", "single"):
            raise ValueError("precision must be 'double' or 'single'")
        self._precision = precision
        if gradient != "analytic":
            raise ValueError("the native state-vector solver uses analytic gradients")
        if profile is not None and type(profile) is not bool:
            raise TypeError("profile must be True, False or None")
        if type(execute_circuits) is not bool:
            raise TypeError("execute_circuits must be a boolean")
        self._execute_circuits = execute_circuits or shots > 0 or device != "AER_simulator"
        from .integrations.transfer import configure as configure_transfer

        configure_transfer(self, transfer=transfer, transfer_threshold=transfer_threshold,
                           transfer_steps=transfer_steps, update_between_geometries=update_between_geometries,
                           resume=resume)
        self._profile = profile
        if xc is not None and xc not in FUNCTIONALS:
            raise ValueError(f"expected a canonical XC name in {FUNCTIONALS}")
        if mapping is not None and mapping not in MAPPINGS:
            raise ValueError(f"expected a canonical mapping in {MAPPINGS}")
        if pool not in ("fermionic", "qubit", "qeb", "ceo", "ceo-ovp", "spin-orbit"):
            raise ValueError("unknown native operator pool")
        ansatz_spec = {"name": ansatz} if isinstance(ansatz, str) else dict(ansatz)
        if ansatz_spec.get("name") not in ("uccsd", "hva"):
            raise ValueError("native variational ansätze are uccsd and hva")
        self._hva = None
        if ansatz_spec["name"] == "hva":
            if method != "vqe" or pool != "fermionic":
                raise ValueError("HVA uses fixed VQE Hamiltonian groups")
            if set(ansatz_spec) - {
                "name",
                "layers",
                "evolution",
                "order",
                "steps",
                "seed_angle",
            }:
                raise ValueError("unknown HVA option")
            self._hva = {k: v for k, v in ansatz_spec.items() if k != "name"}
        elif len(ansatz_spec) != 1:
            raise ValueError("uccsd accepts only its name")
        self._pool = pool
        if sector is not None and type(sector) is not bool:
            raise TypeError(
                "sector must be True, False or None for automatic selection"
            )
        if type(project_translation) is not bool and project_translation != "auto":
            raise ValueError("project_translation must be True, False or 'auto'")
        self._project_translation = project_translation
        if not np.isfinite(grid_spacing) or grid_spacing <= 0:
            raise ValueError("grid_spacing must be positive and finite")
        if type(charge) is not int:
            raise TypeError("charge must be an integer")
        for value in (
            spin,
            unrestricted,
            trotter,
            include_singles,
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
        self.provider = None
        self._gradient_measurement = None
        if self._execute_circuits or backend_provider is not None:
            from .integrations.circuits import configure

            configure(self, provider=backend_provider, device=device, shots=shots,
                      options=backend_options, measurement_budget=measurement_budget)
        elif backend_options is not None or measurement_budget is not None:
            raise ValueError("backend options require a circuit provider or device")
        from copy import deepcopy

        basis = deepcopy(basis)
        self._basis = basis
        self._basis_spec = (
            None
            if (basis if isinstance(basis, str) else basis.get("name"))
            in ("PAW-LCAO", "UPAW-LCAO")
            else basis_specification(basis)
        )
        if paw_dataset is not None:
            if self._basis_spec is not None:
                raise ValueError("paw_dataset applies to a PAW-LCAO or UPAW-LCAO basis")
            selected = {"name": basis} if isinstance(basis, str) else dict(basis)
            if "paw_dataset" in selected and selected["paw_dataset"] != paw_dataset:
                raise ValueError("basis and calculator specify different PAW datasets")
            selected["paw_dataset"] = paw_dataset
            basis = selected
            self._basis = basis
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
        if two_body is not None:
            representation = (
                {"method": two_body} if isinstance(two_body, str) else dict(two_body)
            )
            if (
                set(representation) - {"method", "tolerance"}
                or representation.get("method", "cholesky") != "cholesky"
            ):
                raise ValueError(
                    "two_body currently accepts Cholesky factors and their tolerance"
                )
            cholesky_tolerance = representation.get("tolerance", cholesky_tolerance)
        self._cholesky_tolerance = cholesky_tolerance
        self._load_hamiltonian = load_hamiltonian
        self._save_hamiltonian = save_hamiltonian
        if (
            sum(x is not None for x in (native_model, load_hamiltonian, load_crystal))
            > 1
        ):
            raise ValueError("supply only one native model or cached problem")
        self._supplied_model = native_model
        self._supplied_hamiltonian = hamiltonian
        self._hamiltonian_orbitals = n_spatial_orbitals
        self._hamiltonian_particles = num_particles
        self._hamiltonian_format = hamiltonian_format
        if hamiltonian_format not in (None, "json", "parquet"):
            raise ValueError("hamiltonian_format must be json or parquet")
        if hamiltonian is not None:
            if (
                load_hamiltonian is not None
                or native_model is not None
                or load_crystal is not None
            ):
                raise ValueError(
                    "supply one Hamiltonian, native model, or cached problem"
                )
            if n_spatial_orbitals is None or num_particles is None:
                raise ValueError(
                    "a supplied Hamiltonian needs n_spatial_orbitals and num_particles"
                )
        self._checkpoint_path = checkpoint
        self._resume_path = resume
        self._trotter = trotter or self._hva is not None or self._execute_circuits or method in ("mcas-vqe", "valqa")
        self._include_singles = include_singles
        self._sector = (
            (pool not in ("qubit", "spin-orbit") and self._hva is None)
            if sector is None
            else sector
        )
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
        self._convergence = dict(
            {"gradient": 1e-3, "energy": 1e-3} if convergence is None else convergence
        )
        self._atomic_units = atomic_units
        self._search = None
        self._search_geometry = None
        if method in ("mcas-vqe", "valqa"):
            from .integrations.search import configure

            configure(self, max_steps=200 if max_steps is None else max_steps,
                      min_length=min_length, max_length=max_length,
                      move_weights=move_weights, temperature=temperature,
                      length_penalty=length_penalty, warm_start=warm_start,
                      replace_start=replace_start, screen_insertions=screen_insertions,
                      proposal_temperature=proposal_temperature, proposal=proposal,
                      record=record, seed=seed, proposal_model=proposal_model)
        elif max_steps is not None or record is not None or screen_insertions or proposal_model is not None:
            raise ValueError("architecture-search controls require mcas-vqe or valqa")
        self.trace = trace
        self.txt = txt
        self._optimizer = {
            "method": "SLSQP",
            "maxiter": 2000,
            "tol": 1e-12,
            **(
                dict(method=optimizer)
                if isinstance(optimizer, str)
                else dict(optimizer or {})
            ),
        }
        if self._optimizer["method"] not in _native.OPTIMIZERS:
            raise ValueError(
                f"expected a native optimizer in {tuple(_native.OPTIMIZERS)}"
            )
        if set(self._optimizer) - {"method", "maxiter", "maxfev", "tol"}:
            raise ValueError("optimizer options are method, maxiter, maxfev and tol")
        if set(self._convergence) - {"gradient", "energy"}:
            raise ValueError("convergence options are gradient and energy")
        if not self.capabilities.quantum and (
            load_hamiltonian is not None
            or native_model is not None
            or hamiltonian is not None
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
        self.solver = None
        self.result = None
        self.results = {}
        self.checkpoint = None
        self._load_crystal = load_crystal
        self._save_crystal = save_crystal
        self._kpts_spec = (
            {"size": (1, 1, 1), "gamma": True}
            if kpts is None
            else dict(kpts)
            if isinstance(kpts, dict)
            else {"size": tuple(kpts), "gamma": False}
        )
        size = self._kpts_spec.get("size", (1, 1, 1))
        if len(size) != 3 or any(type(n) is not int or n < 1 for n in size):
            raise ValueError("kpts needs three positive integer mesh sizes")
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

        if atoms is not None:
            atoms.calc = self

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
        if self.method == "dft":
            values.append("magmom")
        if (
            self.capabilities.forces
            and (self._basis_spec is not None or self.capabilities.quantum)
            and not self._execute_circuits
            and self._kinetic == "spectral"
            and self._load_crystal is None
            and not (self._geometry is not None and any(self._geometry.pbc))
        ):
            values.append("forces")
        return values

    @property
    def n_qubits(self):
        if self.problem is not None:
            return self.problem.num_qubits
        if self.result is None:
            m = self._hamiltonian_orbitals
            if m is None and self._supplied_model is not None:
                m = self._supplied_model.n_spatial_orbitals
            if m is not None:
                return 2 * m - (2 if self._mapping == "parity_reduced" else 0)
            raise RuntimeError("run a calculation before requesting the register size")
        if self._active_spec is not None:
            return self.result.qubit_hamiltonian().num_qubits
        return 2 * self.integrals.n_orbitals - (
            2 if self._mapping == "parity_reduced" else 0
        )

    @property
    def num_particles(self):
        if self.problem is not None:
            return tuple(self.problem.num_particles)
        if self.result is None:
            if self._hamiltonian_particles is not None:
                return tuple(self._hamiltonian_particles)
            if self._supplied_model is not None:
                return tuple(self._supplied_model.num_particles)
            raise RuntimeError("run a calculation before requesting electron counts")
        return self.result.num_particles

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

    def natural_orbitals(self, *, state=0):
        """Return native natural occupations and atomic/MO orbital coefficients."""
        if self.result is None:
            raise RuntimeError("run a calculation before requesting natural orbitals")
        return self.result.natural_orbitals(state=state)

    def volumetric_field(
        self,
        quantity="density",
        index=0,
        *,
        state=0,
        component="auto",
        grid=None,
        grid_spacing=None,
    ):
        """Return a native sampled molecular field, optionally on a finer grid."""
        if self.result is None:
            raise RuntimeError("run a calculation before requesting a molecular field")
        if grid_spacing is not None:
            if grid is not None:
                raise ValueError("specify grid or grid_spacing, not both")
            if not np.isfinite(grid_spacing) or grid_spacing <= 0:
                raise ValueError("field grid spacing must be positive and finite")
            if self._geometry is None:
                raise ValueError("field resampling requires the molecular geometry")
            grid = _native.SpatialGrid.from_geometry(
                self._geometry.native(), [grid_spacing * ANGSTROM_TO_BOHR] * 3
            )
        return self.result.volumetric_field(
            quantity, index, state=state, component=component, grid=grid
        )

    def write_cube(self, path, quantity="density", index=0, **options):
        """Sample a molecular field and write its cube file using Rust."""
        return self.volumetric_field(quantity, index, **options).write(
            path, format="cube"
        )

    def _mean_field(self, geometry):
        if (self._basis_spec is None and self.integrals is not None
                and self._force_grid is not None and self._geometry is not None
                and geometry.numbers == self._geometry.numbers
                and np.array_equal(geometry.cell, self._geometry.cell)):
            self.integrals = self.integrals.at_geometry(geometry.native())
        elif self._basis_spec is None:
            from .integrations.paw_molecular import build

            self.integrals = build(self, geometry)
        else:
            self.integrals = self.context.molecular_integrals(
                geometry.native(),
                basis_spec=json.dumps(self._basis_spec),
                spacing_bohr=self._grid_spacing * ANGSTROM_TO_BOHR,
                grid=self._grid(geometry),
                kinetic=self._kinetic,
                cholesky_tolerance=self._cholesky_tolerance,
                precision=self._precision,
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

    def _quantum(self, reference, initial_parameters, callback=None):
        if self._supplied_hamiltonian is not None:
            source = self._supplied_hamiltonian
            if isinstance(source, _native.PauliOperator):
                operator = source
            else:
                from .core.mapping import Fermion, PauliSum

                if isinstance(source, Fermion):
                    operator = self.context.fermion_operator(
                        2 * self._hamiltonian_orbitals,
                        self._hamiltonian_particles,
                        [
                            (list(term), float(value.real), float(value.imag))
                            for term, value in source.terms.items()
                        ],
                        mapping=self._mapping,
                    )
                elif isinstance(source, PauliSum):
                    operator = self.context.pauli_operator(
                        source.num_qubits,
                        [
                            (label, float(value.real), float(value.imag))
                            for label, value in source.terms.items()
                        ],
                    )
                else:
                    raise TypeError(
                        "hamiltonian must be a Fermion, PauliSum, or native PauliOperator"
                    )
            self.problem = self.context.quantum_problem(
                operator,
                self._hamiltonian_orbitals,
                self._hamiltonian_particles,
                mapping=self._mapping,
                sector=self._sector,
                include_singles=self._include_singles,
                sparse=self._sparse,
                pool=self._pool,
            )
        elif self._load_hamiltonian is not None:
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
                pool=self._pool,
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
                pool=self._pool,
            )
        if self._hva is not None:
            self.problem = self.problem.hva(**self._hva)
        if reference is None and self._save_hamiltonian:
            from .core.mapping import PauliSum
            from .core.serialization import save_hamiltonian

            save_hamiltonian(
                self._save_hamiltonian,
                PauliSum(
                    {
                        label: complex(re, im)
                        for label, re, im in self.problem.hamiltonian_terms
                    },
                    num_qubits=self.problem.num_qubits,
                ),
                mapping=self.problem.mapping,
                num_particles=self.problem.num_particles,
                n_spatial_orbitals=self.problem.n_spatial_orbitals,
                format=self._hamiltonian_format,
            )
        opts = dict(
            adapt=self.method.endswith("adapt-vqe"),
            trotter=self._trotter,
            optimizer=self._optimizer["method"],
            max_iterations=self._optimizer.get("maxiter", 300),
            max_evaluations=self._optimizer.get("maxfev", 5000),
            tolerance=self._optimizer.get("tol", 1e-9),
            max_growth=self._max_growth,
            gradient_tolerance=self._convergence.get("gradient"),
            energy_tolerance=(None if self._convergence.get("energy") is None else
                              self._convergence["energy"] / _native.HARTREE_TO_EV),
            initial_parameters=initial_parameters,
        )
        from .integrations.transfer import prepare as prepare_transfer

        carried = prepare_transfer(self, reference)
        if carried is not None and self._search is None:
            if initial_parameters is not None:
                raise ValueError("choose transferred or explicit initial parameters")
            opts.update(initial_parameters=carried["parameters"], initial_indices=carried["indices"])
        if self.method.startswith("subspace-"):
            if self._execute_circuits:
                raise NotImplementedError("subspace SDK execution requires a multi-reference executor")
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
            observer = None
            if callback is not None:
                from .analysis.states import AnsatzView

                def observer(info):
                    info = dict(info)
                    info["energy"] = info["energy_ha"] * (
                        1.0 if self._atomic_units else _native.HARTREE_TO_EV
                    )
                    info["ansatz"] = AnsatzView(self.problem, info["indices"], True)
                    callback(info)

            executor = None
            if self._execute_circuits:
                from .integrations.circuits import CircuitExecutor

                executor = CircuitExecutor(self)
            if self._search is not None:
                if callback is not None:
                    raise ValueError("growth callbacks apply to ADAPT, not architecture edits")
                from .integrations.search import run

                raw = run(self, initial_parameters)
            else:
                raw = self.problem.solve(resume=resumed, callback=observer,
                                         executor=executor, **opts)
            if executor is not None:
                raw["execution"] = self.provider.name
                raw["state_kind"] = "ideal-circuit" if self._shots else "statevector"
                raw["execution_usage"] = self.provider.run_usage()
            from .integrations.transfer import record as record_transfer

            record_transfer(self, raw)
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
        from .analysis.states import SolverView

        self.solver = SolverView(
            self.problem, raw, self.checkpoint, opts["adapt"] or self._trotter
        )
        result = CalculationResult(
            method=self.method,
            context=self.context,
            solver=self.solver,
            profile=self._profile,
            problem=self.problem,
            reference=None if reference is None else reference.scf,
            integrals=None if reference is None else self.integrals,
            active_space=self._active_spec,
            model=None if reference is None else reference.hamiltonian,
            raw=raw,
            mapping=self._mapping,
            atomic_units=self._atomic_units,
        )
        if result.optimizer_failures:
            warnings.warn(
                f"the inner optimizer did not converge in "
                f"{len(result.optimizer_failures)} inner minimizations; last message: "
                f"{result.optimizer_failures[-1][1]!r}; "
                f"outer convergence: {result.success} ({raw['message']})",
                RuntimeWarning,
                stacklevel=2,
            )
        return result

    def _periodic(self, geometry):
        if self.method != "dft":
            raise NotImplementedError(
                "periodic geometry calculations currently provide native DFT"
            )
        if self.xc not in ("lda", "pbe", "r2scan", "pbe+rvv10", "r2scan+rvv10"):
            raise NotImplementedError("periodic screened exact exchange requires its reciprocal exchange operator")
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
            raw=periodic.snapshot(include_matrices=False),
            periodic=periodic,
            crystal=self.periodic_problem,
            atomic_units=self._atomic_units,
        )

    def run(self, geometry=None, *, initial_parameters=None, callback=None):
        with self.resources.lock:
            if self.resources.closed:
                raise RuntimeError("calculator is closed")
            source = geometry if geometry is not None else self._attached_atoms
            value = geometry_value(source) if source is not None else self._geometry
            self._search_geometry = value
            from .integrations.transfer import capture as capture_transfer

            capture_transfer(self)
            self.result = None
            self.solver = None
            self.checkpoint = None
            self.native_reference = None
            self.results = {}
            self.problem = None
            if (value is not None and any(value.pbc)) or self._load_crystal is not None:
                result = self._periodic(value)
            elif value is not None:
                reference = self._mean_field(value)
                if not reference.success:
                    raise RuntimeError("native SCF did not converge")
                if self._save_hamiltonian:
                    reference.save_hamiltonian(
                        self._save_hamiltonian, format=self._hamiltonian_format
                    )
                result = (
                    self._quantum(reference, initial_parameters, callback)
                    if self.capabilities.quantum
                    else reference
                )
            elif self.capabilities.quantum and (
                self._load_hamiltonian is not None
                or self._supplied_model is not None
                or self._supplied_hamiltonian is not None
            ):
                result = self._quantum(None, initial_parameters, callback)
            else:
                raise ValueError(
                    "provide a geometry, cached Hamiltonian, or cached crystal"
                )
            self._geometry = value
            if isinstance(source, Atoms):
                self.atoms = source.copy()
            self.result = result
            self.results = {
                "energy": result.in_units("eV"),
                "free_energy": result.free_energy_ev,
            }
            if self.method == "dft":
                if result.periodic is not None:
                    self.results["magmom"] = result._raw["magnetic_moment"]
                else:
                    alpha, beta = self.native_reference.num_particles
                    self.results["magmom"] = float(alpha - beta)
            if self.trace or self.txt is not None:
                from .integrations.reporting import write_calculation

                write_calculation(self, result, value)
            return result

    def energy_levels(self, num_states=2, *, restarts=4, beta=None, seed=0):
        """Sequential variational deflation with native overlap penalties."""
        if self.method != "vqe":
            raise ValueError("sequential deflation requires a fixed VQE ansatz")
        if self.result is None:
            self.run()
        from .analysis.excitations import EnergyLevels
        from .units import HARTREE_TO_EV

        raw = self.problem.energy_levels(
            num_states=num_states,
            restarts=restarts,
            beta=beta,
            seed=seed,
            trotter=self._trotter,
            optimizer=self._optimizer["method"],
            max_iterations=self._optimizer.get("maxiter", 300),
            max_evaluations=self._optimizer.get("maxfev", 5000),
            tolerance=self._optimizer.get("tol", 1e-9),
        )
        factor = 1.0 if self._atomic_units else HARTREE_TO_EV
        return EnergyLevels(
            energies=raw["energies_ha"] * factor,
            states=self.problem.expand_state(raw["states"]),
            reference_energy=raw["reference_ha"] * factor,
            num_evaluations=raw["num_evaluations"],
            success=tuple(raw["success"]),
            messages=tuple(raw["messages"]),
            energy_unit="Ha" if self._atomic_units else "eV",
        )

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
            if isinstance(source, Atoms):
                Calculator.calculate(self, source, properties, system_changes)
            if "forces" in properties and (
                not self.capabilities.forces
                or any(geometry.pbc)
                or self._kinetic != "spectral"
            ):
                raise NotImplementedError(
                    "forces require a supported molecular native calculation with spectral kinetic energy"
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
                project = self._project_translation is not False and len(geometry.numbers) > 1
                internal = project and self.capabilities.quantum and self._basis_spec is None
                self.results["forces"] = result.forces(
                    internal=internal,
                    project_translation=project)
                from .integrations.reporting import write_forces

                write_forces(self, result, geometry, internal=internal, translated=project)
            return result

    def write_optimization_summary(self, atoms=None, fmax=None, optimizer=None, force=False, status=None):
        """Record the geometry history and ASE optimizer's completion verdict."""
        from .integrations.reporting import write_optimization_summary

        with self.resources.lock:
            return write_optimization_summary(self, atoms, fmax, optimizer, force, status)

    def set_atoms(self, atoms):
        self._attached_atoms = atoms

    def get_property(self, name, atoms=None, allow_calculation=True):
        with self.resources.lock:
            if self.resources.closed:
                raise RuntimeError("calculator is closed")
            if name not in self.implemented_properties:
                raise PropertyNotImplementedError(
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
        return self.get_property("free_energy" if force_consistent else "energy", atoms)

    def get_forces(self, atoms=None):
        return self.get_property("forces", atoms)

    def reset(self):
        with self.resources.lock:
            self.atoms = None
            self.result = None
            self.results = {}
            self._geometry = None
            self.integrals = None
            self.native_reference = None
            self.problem = None
            self.solver = None
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
