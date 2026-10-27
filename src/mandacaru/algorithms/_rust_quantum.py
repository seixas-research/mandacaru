"""Interface adapter for Rust-owned cached quantum solves (no Python callbacks)."""
from __future__ import annotations

from numbers import Integral
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..units import HARTREE_TO_EV


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class _State:
    """A Python view of a Rust-owned ansatz; evaluations remain in Rust."""

    def __init__(self, driver, indices=None):
        self.problem, self.trotter, self.indices = driver.problem, driver.trotter, indices

    def state(self, parameters):
        return self.problem.evaluate(
            parameters, trotter=self.trotter, indices=self.indices)["state"]


class _RustDriver:
    engine = "rust"
    writes_output_log = True
    supports_checkpoints = False
    adapt = False

    def __init__(self, *, load_hamiltonian=None, optimizer=None, mapping=None,
                 sector="auto", sparse="matrix-free", atomic_units=False,
                 engine_options=None, verbose=True, txt=None, references=True,
                 dry_run=False, precision="double", dump_buffers=False,
                 cache_dir=None, basis="HAO", grid=None, grid_spacing=0.20,
                 native_model=None, charge=0, spin=False, n_electrons=None,
                 kinetic="spectral", two_body=None, active_space=None,
                 scf_options=None, save_hamiltonian=False, hamiltonian_format="json",
                 checkpoint=None, resume=None):
        if (checkpoint is not None or resume is not None) and not self.supports_checkpoints:
            raise NotImplementedError("native checkpoints are supported for VQE and ADAPT-VQE")
        self.checkpoint_path = None if checkpoint is None else str(checkpoint)
        self.resume_path = None if resume is None else str(resume)
        self.checkpoint = None
        if precision != "double":
            raise NotImplementedError("the Rust quantum engine uses complex128 precision")
        if dump_buffers or cache_dir is not None:
            raise NotImplementedError("the Rust quantum engine does not write integral buffers")
        if sector not in (True, False, "auto") or type(sector) not in (bool, str):
            raise ValueError("sector must be True, False or 'auto'")
        if sparse not in (True, False, "auto", "matrix-free") or type(sparse) not in (bool, str):
            raise ValueError("sparse must be True, False, 'auto' or 'matrix-free'")
        if load_hamiltonian is not None and native_model is not None:
            raise ValueError("choose a cache or native_model, not both")
        if load_hamiltonian is not None or native_model is not None:
            if active_space is not None or charge != 0 or n_electrons is not None or grid is not None or scf_options or two_body is not None or save_hamiltonian or kinetic not in (None, "spectral"):
                raise ValueError("geometry options do not apply to a cached Hamiltonian or a precomputed native_model")
        self.load_hamiltonian = None if load_hamiltonian is None else str(load_hamiltonian)
        self.native_model = native_model
        self._sector_option = sector
        self._dense = sparse is False
        self._sparse = sparse is True
        self.atomic_units = bool(atomic_units)
        self.energy_unit = "Ha" if atomic_units else "eV"
        self._scale = 1.0 if atomic_units else HARTREE_TO_EV
        self.verbose, self.txt, self.dry_run = verbose, txt, dry_run
        from ..core.mapping import resolve_mapping
        self._requested_mapping = None if mapping is None else resolve_mapping(mapping)
        self.options = dict(engine_options or {})
        unknown = set(self.options) - {"threads", "memory_limit_bytes"}
        if unknown:
            raise ValueError(f"unknown engine_options: {sorted(unknown)}")
        if self.options.get("threads") is not None:
            self.options["threads"] = _integer(self.options["threads"], "threads", 1)
            if self.options["threads"] > 1024:
                raise ValueError("threads must be in 1..1024")
        if "memory_limit_bytes" in self.options:
            self.options["memory_limit_bytes"] = _integer(self.options["memory_limit_bytes"], "memory_limit_bytes", 1)
        if optimizer is None:
            optimizer = {"method": "BFGS"}
        elif isinstance(optimizer, str):
            optimizer = {"method": optimizer}
        if not isinstance(optimizer, dict):
            raise TypeError("the Rust engine takes an optimizer name or dictionary")
        optimizer = dict(optimizer)
        method = optimizer.pop("method", "BFGS")
        method = str(method).upper()
        if method not in ("BFGS", "L-BFGS"):
            raise NotImplementedError("engine='rust' currently provides BFGS and L-BFGS optimizers")
        unknown = set(optimizer) - {"maxiter", "maxfev", "tol"}
        if unknown:
            raise ValueError(f"unknown Rust optimizer options: {sorted(unknown)}")
        self._opt = {
            "optimizer": method,
            "max_iterations": _integer(optimizer.get("maxiter", 300), "maxiter"),
            "max_evaluations": _integer(optimizer.get("maxfev", 5000), "maxfev", 1),
            "tolerance": float(optimizer.get("tol", 1e-9)),
        }
        if not np.isfinite(self._opt["tolerance"]) or self._opt["tolerance"] <= 0:
            raise ValueError("optimizer tol must be positive and finite")
        self.result = None
        self.problem = None
        self.shots = 0
        self.log_targets = None
        self._explicit_grid, self._grid_spacing = grid, float(grid_spacing)
        if not np.isfinite(self._grid_spacing) or self._grid_spacing <= 0:
            raise ValueError("grid_spacing must be positive and finite")
        self._charge = _integer(abs(charge), "absolute charge") * (-1 if charge < 0 else 1)
        self._n_electrons = None if n_electrons is None else _integer(n_electrons, "n_electrons")
        kinetic = "spectral" if kinetic is None else kinetic
        self._kinetic = kinetic
        if kinetic not in ("spectral", "fd"):
            raise ValueError("native kinetic must be 'spectral' or 'fd'")
        from ._rust_molecular import basis_specification
        self._basis_spec = basis_specification(basis) if load_hamiltonian is None and native_model is None else None
        self._scf_options = dict(scf_options or {})
        if set(self._scf_options) - {"max_iterations", "tolerance", "diis", "guesses"}:
            raise ValueError("unknown scf_options; use max_iterations, tolerance, diis, guesses")
        for key in ("max_iterations", "guesses"):
            if key in self._scf_options:
                self._scf_options[key] = _integer(self._scf_options[key], key, 1)
        if "tolerance" in self._scf_options:
            tol = self._scf_options["tolerance"]
            if isinstance(tol, bool) or not np.isfinite(tol) or tol <= 0:
                raise ValueError("SCF tolerance must be finite and positive")
        if "diis" in self._scf_options and type(self._scf_options["diis"]) is not bool:
            raise ValueError("SCF diis must be a boolean")
        from .active_space import resolve_active_space_spec
        spec = resolve_active_space_spec(active_space)
        self._active_spec = None if spec is None else spec.as_dict()
        if spec is not None and (spec.method != "energy" or spec.threshold is not None or spec.symmetry or spec.correlating_pairs or isinstance(spec.orbitals, dict)):
            raise NotImplementedError("native active spaces use energy-ordered counts or explicit indices and a frozen core")
        if two_body == "direct":
            raise NotImplementedError("native integral-direct SCF is not yet implemented; use packed or Cholesky integrals")
        if two_body == "packed":
            two_body = None
        from ..integrals.cholesky import resolve_two_body, DEFAULT_TOLERANCE
        representation = resolve_two_body(two_body)
        self._cholesky_tolerance = ((DEFAULT_TOLERANCE if representation.tolerance is None else representation.tolerance)
                                    if representation.cholesky else None)
        self._save_hamiltonian = save_hamiltonian
        if save_hamiltonian and hamiltonian_format != "json":
            raise NotImplementedError("native cache export currently writes schema-v1 JSON")
        if save_hamiltonian and save_hamiltonian is not True and Path(save_hamiltonian).suffix not in ("", ".json"):
            raise ValueError("native Hamiltonian cache export uses .json paths")

    def _configure(self):
        from .. import _native

        if self.native_model is not None:
            if not isinstance(self.native_model, _native.MolecularModel):
                raise TypeError("native_model must be a native MolecularModel")
            self.context = self.native_model.context
            for key in ("threads", "memory_limit_bytes"):
                requested = self.options.get(key)
                if requested is not None and requested != getattr(self.context, key):
                    raise ValueError("native_model reuses its owning context; configure engine_options on the mean-field reference")
            self.problem = self.native_model.problem(
                mapping=self._requested_mapping or "jordan_wigner", sector=self._sector_option is not False,
                dense=self._dense, sparse=self._sparse, include_singles=self.include_singles)
        else:
            self.context = _native.Context(**self.options)
            from ..core.serialization import detect_format, load_hamiltonian
            if detect_format(self.load_hamiltonian) == "parquet":
                import json
                record = load_hamiltonian(self.load_hamiltonian, engine="rust",
                                          engine_options=self.options)
                payload = dict(format="mandacaru-qubit-hamiltonian", version=1,
                               num_qubits=record.hamiltonian.num_qubits,
                               mapping=record.mapping, num_particles=record.num_particles,
                               n_spatial_orbitals=record.n_spatial_orbitals,
                               metadata=record.metadata,
                               terms=[(p, complex(c).real, complex(c).imag)
                                      for p, c in record.hamiltonian.terms.items()])
                self.problem = self.context.parse_snapshot(json.dumps(payload), payload["format"]).problem(
                    sector=self._sector_option is not False, dense=self._dense,
                    sparse=self._sparse, include_singles=self.include_singles)
                self._cache_adapter = "python-parquet-adapter"
            else:
                self.problem = self.context.load_hamiltonian(
                    self.load_hamiltonian, sector=self._sector_option is not False,
                    dense=self._dense, sparse=self._sparse, include_singles=self.include_singles)
                self._cache_adapter = "native-json"
        self.mapping = self.problem.mapping
        if self._requested_mapping is not None:
            from ..core.mapping import resolve_mapping
            if resolve_mapping(self._requested_mapping) != self.mapping:
                raise ValueError("mapping disagrees with the cached Hamiltonian; the Rust engine does not remap a cache implicitly")
        self.n_qubits = self.problem.num_qubits
        self.n_spatial_orbitals = self.problem.n_spatial_orbitals
        self.num_particles = tuple(self.problem.num_particles)
        self._sector = (SimpleNamespace(indices=np.asarray(self.problem.basis_indices, dtype=np.int64))
                        if self._sector_option is not False else None)
        self.ansatz = _State(self)

    @property
    def hamiltonian(self):
        if self.problem is None:
            return None
        from ..core.mapping import PauliSum
        return PauliSum({p: complex(re, im) for p, re, im in self.problem.hamiltonian_terms},
                        num_qubits=self.n_qubits)

    def _check_shots_for_hardware(self):
        pass

    def calculate(self, atoms, *args, **kwargs):
        from ._rust_molecular import configure_geometry
        if self.load_hamiltonian is not None:
            raise ValueError("a cached Hamiltonian uses run(); it cannot be recomputed for an ASE geometry")
        configure_geometry(self, atoms)
        result = self.run()
        self.results = {"energy": result.in_units("eV")}

    def run(self, initial_parameters=None, geometry=None):
        if geometry is not None:
            from ._rust_molecular import configure_geometry
            configure_geometry(self, geometry)
        if self.dry_run:
            raise NotImplementedError("Rust cached solves do not yet provide dry-run estimates")
        if self.problem is None:
            raise ValueError("native run() needs a cache, native_model or geometry=Atoms")
        resumed = (None if self.resume_path is None else
                   self.context.read_snapshot(self.resume_path, "mandacaru-wavefunction-checkpoint"))
        raw = self.problem.solve(
            adapt=self.adapt, trotter=self.trotter, initial_parameters=initial_parameters,
            resume=resumed, **self._opt, **self._growth)
        timing = {"engine": "rust", "solve_seconds": raw["elapsed_seconds"],
                  "threads": self.context.threads, "memory_peak_bytes": self.context.memory_peak_bytes,
                  "dimension": self.problem.dimension, "termination": raw["message"]}
        if hasattr(self, "_cache_adapter"):
            timing["cache_engine"] = self._cache_adapter
        if hasattr(self, "_geometry_seconds"):
            timing["geometry_seconds"] = self._geometry_seconds
        common = dict(optimal_energy=raw["energy_ha"] * self._scale,
                      optimal_parameters=raw["parameters"], reference_energy=raw["reference_ha"] * self._scale,
                      num_evaluations=len(raw["history_ha"]), optimizer_steps=raw["optimizer_steps"],
                      timings=timing, energy_unit=self.energy_unit)
        if self.adapt:
            from .adapt_vqe import ADAPTVQEResult, AdaptIteration
            labels = self.problem.pool_labels
            iterations = [AdaptIteration(
                operator_label=labels[i], operator_kind="fermionic", max_gradient=g,
                energy=e * self._scale, cnot_count=None, depth=None, num_parameters=k + 1,
                optimizer_steps=steps, num_evaluations=evaluations)
                for k, (i, e, g, steps, evaluations, _, _) in enumerate(raw["iterations"])]
            failures = [(k + 1, it[6]) for k, it in enumerate(raw["iterations"]) if not it[5]]
            self.result = ADAPTVQEResult(**common, converged=raw["success"],
                final_max_gradient=raw["final_max_gradient"], operators=raw["operators"],
                iterations=iterations, optimizer_failures=failures,
                start="empty ansatz" if self.resume_path is None else f"resumed from {self.resume_path}")
            self.ansatz = _State(self, raw["indices"])
        else:
            from .vqe import VQEResult
            self.result = VQEResult(**common, history=[e * self._scale for e in raw["history_ha"]],
                                    success=raw["success"])
        self.state = raw["state"]
        import json
        from ..core.checkpoint import WavefunctionCheckpoint
        snapshot = self.problem.checkpoint(
            raw["indices"], raw["parameters"].tolist(),
            product_form=self.adapt or self.trotter, energy_ha=raw["energy_ha"],
            method="adapt-vqe" if self.adapt else "vqe",
            status_json=json.dumps({"complete": True, "converged": raw["success"],
                                    "iteration": len(raw["indices"]) if self.adapt else raw["optimizer_steps"],
                                    "num_evaluations": len(raw["history_ha"]),
                                    "termination": raw["message"]}))
        self.checkpoint = WavefunctionCheckpoint.from_payload(json.loads(snapshot.json))
        if self.checkpoint_path is not None:
            snapshot.save(self.checkpoint_path)
        report = (f"Rust {'ADAPT-VQE' if self.adapt else 'VQE'}: {self.result.optimal_energy:.12f} "
                  f"{self.energy_unit}; {raw['message']}; {len(raw['history_ha'])} evaluations\n")
        if self.verbose:
            print(report, end="")
        if self.txt is not None:
            Path(self.txt).write_text(report)
        return self.result


class RustVQE(_RustDriver):
    supports_checkpoints = True
    def __init__(self, *, ansatz="uccsd", **kwargs):
        from .ansatz_spec import resolve_ansatz
        spec = resolve_ansatz(ansatz)
        if spec is None or spec.name != "uccsd":
            raise NotImplementedError("engine='rust' currently supports the UCCSD ansatz")
        self.trotter = spec.options.get("trotter", False)
        self.include_singles = spec.options.get("include_singles", True)
        if type(self.trotter) is not bool or type(self.include_singles) is not bool:
            raise ValueError("trotter and include_singles must be booleans")
        self._growth = {}
        super().__init__(**kwargs)
        if not self.dry_run and (self.load_hamiltonian is not None or self.native_model is not None):
            self._configure()


class RustADAPT(_RustDriver):
    supports_checkpoints = True
    adapt = True

    def __init__(self, *, pool="fermionic", profile=False, gradient="analytic",
                 max_iterations=50, convergence=None, **kwargs):
        from .convergence import Convergence
        if pool != "fermionic" or profile or gradient != "analytic":
            raise NotImplementedError("engine='rust' supports pool='fermionic', profile=False and gradient='analytic'")
        resolved = Convergence.resolve(convergence)
        self._growth = {"max_growth": _integer(max_iterations, "max_iterations"),
                        "gradient_tolerance": resolved.gradient,
                        "energy_tolerance": resolved.energy_hartree}
        self.trotter = self.include_singles = True
        super().__init__(**kwargs)
        if not self.dry_run and (self.load_hamiltonian is not None or self.native_model is not None):
            self._configure()


class _RustSubspace:
    supports_checkpoints = False
    def __init__(self, *, num_states=2, weights=None, multiplicity=None, **kwargs):
        self.num_states = _integer(num_states, "num_states", 1)
        if multiplicity is not None:
            raise NotImplementedError("native UCCSD subspace searches use determinants; spin-adapted HVA references are not yet implemented")
        self.weights = (list(range(self.num_states, 0, -1))
                        if weights is None else list(weights))
        if len(self.weights) != self.num_states:
            raise ValueError("weights must match num_states")
        super().__init__(**kwargs)

    def run(self, initial_parameters=None, geometry=None):
        if geometry is not None:
            from ._rust_molecular import configure_geometry
            configure_geometry(self, geometry)
        if self.dry_run:
            raise NotImplementedError("native subspace dry-run estimates are not yet implemented")
        if self.problem is None:
            raise ValueError("native run() needs a cache, native_model or geometry=Atoms")
        raw = self.problem.solve_subspace(
            self.weights, adapt=self.adapt, trotter=self.trotter,
            initial_parameters=initial_parameters, **self._opt, **self._growth)
        from .subspace import SubspaceVQEResult, SubspaceADAPTVQEResult
        timing = {"engine": "rust", "solve_seconds": raw["elapsed_seconds"],
                  "threads": self.context.threads,
                  "memory_peak_bytes": self.context.memory_peak_bytes,
                  "dimension": self.problem.dimension, "termination": raw["message"]}
        common = dict(energies=np.asarray(raw["energies_ha"]) * self._scale,
                      optimal_parameters=raw["parameters"],
                      weights=np.asarray(self.weights, dtype=float),
                      states=list(raw["states"]),
                      reference_energy=raw["reference_ha"] * self._scale,
                      num_evaluations=len(raw["history_ha"]),
                      timings=timing, energy_unit=self.energy_unit)
        if self.adapt:
            self.result = SubspaceADAPTVQEResult(
                **common, converged=raw["success"],
                final_max_gradient=raw["final_max_gradient"],
                operators=raw["operators"])
        else:
            self.result = SubspaceVQEResult(**common, success=raw["success"])
        self.state = self.result.states[0]
        self.state_reference_indices = raw["reference_indices"]
        report = (f"Rust subspace search: {self.result.optimal_energy:.12f} "
                  f"{self.energy_unit}; {raw['message']}; "
                  f"{len(raw['history_ha'])} evaluations\n")
        if self.verbose:
            print(report, end="")
        if self.txt is not None:
            Path(self.txt).write_text(report)
        return self.result

    def calculate(self, atoms, *args, **kwargs):
        result = self.run(geometry=atoms)
        self.results = {"energy": float(result.in_units("eV")[0])}


class RustSubspaceVQE(_RustSubspace, RustVQE):
    pass


class RustSubspaceADAPT(_RustSubspace, RustADAPT):
    pass
