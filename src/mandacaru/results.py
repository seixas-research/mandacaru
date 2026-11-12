"""Result views over native-owned scientific states."""

from __future__ import annotations
import os
import numpy as np
from functools import cached_property
from . import _native


class CalculationResult:
    def __init__(
        self,
        *,
        method,
        context,
        reference=None,
        integrals=None,
        problem=None,
        raw=None,
        periodic=None,
        crystal=None,
        mapping="jordan_wigner",
        active_space=None,
        two_body=("dense", None),
        atomic_units=False,
        solver=None,
        model=None,
        profile=None,
        dispersion=None,
    ):
        self.method = method
        self.solver = solver
        self.context = context
        self._reference = reference
        self._integrals = integrals
        self._problem = problem
        self.periodic = periodic
        self._crystal = crystal
        self._raw = reference.snapshot() if raw is None else raw
        self._model = model
        self._profile = profile
        self._mapping = mapping
        self._active_space = active_space
        # (representation, truncation in Hartree) of the active-space Hamiltonian's two-body part.
        self._two_body = two_body
        self.energy_unit = "Ha" if atomic_units else "eV"
        self._dispersion = dispersion

    @property
    def dispersion(self):
        """Native dispersion correction result dict, or None if no dispersion was applied."""
        return self._dispersion

    @property
    def reference(self):
        return self._reference

    @property
    def success(self):
        return bool(self._raw.get("converged", self._raw.get("success", False)))

    @property
    def converged(self):
        return self.success

    def _in_unit(self, value_ha):
        """``value_ha`` (Hartree, a float or an array) in the result's energy unit, in Rust."""
        return _native.to_user_units(value_ha, "energy", atomic_units=self.energy_unit == "Ha")

    @staticmethod
    def _total(*addends_ha):
        """The sum of Hartree energies, added in order in Rust."""
        return _native.to_user_units(
            np.array(addends_ha, dtype=float), "energy", atomic_units=True, total=True)

    @property
    def energy_hartree(self):
        if self._reference is not None and self._problem is None:
            parts = [self._raw["electronic_energy"], self._raw["nuclear_repulsion"]]
        elif "energies_ha" in self._raw:
            parts = [float(self._raw["energies_ha"][0])]
        else:
            parts = [self._raw.get(
                "energy_ha", self._raw.get("extrapolated_energy", self._raw.get("energy"))
            )]
        if self._dispersion is not None:
            parts.append(self._dispersion.get("energy_ha", 0.0))
        return self._total(*parts) if len(parts) > 1 else parts[0]

    def in_units(self, units):
        if units == "Ha":
            return self.energy_hartree
        if units == "eV":
            return _native.to_user_units(self.energy_hartree, "energy")
        raise ValueError("energy units must be Ha or eV")

    @property
    def hubbard_energy(self):
        """The on-site DFT+U energy in the result's energy unit; ``None`` without DFT+U."""
        value = self._raw.get("hubbard_energy")
        return value if value is None else self._in_unit(value)

    #: The energies of a DFT+U shell: the key of the result view and of the native view.
    _HUBBARD_ENERGIES = (("U", "u"), ("J", "j"), ("U_eff", "u_eff"), ("energy", "energy"))

    @property
    def hubbard(self):
        """The DFT+U shells of a periodic run, one dictionary per atom, or ``None``.

        ``atom``, ``symbol``, ``l`` (``"d"`` or ``"f"``), ``U``, ``J``, ``U_eff`` and the shell's
        ``energy`` in the result's energy unit, and the occupation matrices ``n``
        (``spins x (2l+1) x (2l+1)``, Hermitian, in the basis of complex harmonics ordered
        ``m = -l..l``) with their ``trace`` per spin. ``n`` is the PAW partial-wave
        projection: a filled shell has an occupation below 1.
        """
        shells = self._raw.get("hubbard")
        if shells is None:
            return None
        return [
            dict(
                atom=s["atom"],
                symbol=s["symbol"],
                l=s["shell"],
                **{name: self._in_unit(s[f"{key}_ha"]) for name, key in self._HUBBARD_ENERGIES},
                trace=np.array(s["trace"]),
                n=s["n"].copy(),
            )
            for s in shells
        ]

    @property
    def free_energy_hartree(self):
        parts = [self._raw.get("free_energy", self.energy_hartree)]
        if self._dispersion is not None and "free_energy" in self._raw:
            parts.append(self._dispersion.get("energy_ha", 0.0))
        return self._total(*parts) if len(parts) > 1 else parts[0]

    @property
    def free_energy_ev(self):
        return _native.to_user_units(float(self.free_energy_hartree), "energy")

    @property
    def optimal_energy(self):
        return self.in_units(self.energy_unit)

    @property
    def energy_error(self):
        """The shot-noise standard error of the energy, in the energy unit (Rust, from the counts
        with the in-basis covariances); ``None`` without shots: a state-vector energy has none."""
        error = self._raw.get("energy_error_ha")
        return None if error is None else self._in_unit(error)

    @property
    def energy_error_reason(self):
        """Why a shot run reports no ``energy_error`` (a basis past the classical limit of the
        covariance products); ``None`` when it reports one or measured no shots."""
        return self._raw.get("energy_error_reason")

    @property
    def energies(self):
        values = np.asarray(
            self._raw.get("energies_ha", [self.energy_hartree]), dtype=float
        )
        return self._in_unit(values)

    @property
    def states(self):
        return self.state()

    @property
    def _reference_hartree(self):
        return (
            self.energy_hartree
            if self._reference is not None and self._problem is None
            else self._raw.get("reference_ha", self.energy_hartree)
        )

    @property
    def reference_energy(self):
        return self._in_unit(self._reference_hartree)

    @property
    def correlation_energy(self):
        return _native.difference_to_user(
            self.energy_hartree, self._reference_hartree, "energy",
            atomic_units=self.energy_unit == "Ha")

    @property
    def scf(self):
        return self._reference if self._reference is not None else self.periodic

    @property
    def optimal_parameters(self):
        return np.array(self._raw.get("parameters", []), copy=True)

    @property
    def num_parameters(self):
        return len(self._raw.get("parameters", []))

    @property
    def operators(self):
        return list(self._raw.get("operators", []))

    @property
    def optimizer_steps(self):
        return self._raw.get("optimizer_steps", 0)

    @property
    def num_evaluations(self):
        return len(self._raw.get("history_ha", []))

    @property
    def completed(self):
        """Whether execution completed; separate from convergence certification."""
        return bool(self._raw.get("completed", True))

    @property
    def architecture(self):
        return tuple(self._raw.get("indices", ()))

    @property
    def optimal_cost(self):
        return self._in_unit(self._raw.get("optimal_cost_ha", self.energy_hartree))

    @property
    def best_energy(self):
        return self._in_unit(self._raw.get("best_energy_ha", self.energy_hartree))

    @property
    def final_energy(self):
        return self._in_unit(self._raw.get("final_energy_ha", self.energy_hartree))

    @property
    def best_energy_operators(self):
        indices = self._raw.get("best_energy_indices", self._raw.get("indices", ()))
        labels = self._problem.pool_labels
        return [labels[i] for i in indices]

    @property
    def best_energy_parameters(self):
        return np.array(self._raw.get("best_energy_parameters", self._raw.get("parameters", ())), copy=True)

    @property
    def final_operators(self):
        indices = self._raw.get("final_indices", self._raw.get("indices", ()))
        labels = self._problem.pool_labels
        return [labels[i] for i in indices]

    @property
    def final_parameters(self):
        return np.array(self._raw.get("final_parameters", self._raw.get("parameters", ())), copy=True)

    @property
    def acceptance_by_move(self):
        return {name: tuple(counts) for name, counts in self._raw.get("acceptance_by_move", {}).items()}

    @property
    def num_architectures(self):
        return self._raw.get("num_architectures", 0)

    @property
    def num_screenings(self):
        return self._raw.get("num_screenings", 0)

    @property
    def proposal_ready(self):
        return self._raw.get("proposal_ready", False)

    @property
    def proposal_reason(self):
        return self._raw.get("proposal_reason")

    @property
    def exxdiv_treatment(self):
        return self._raw.get("exxdiv_treatment", "none")

    @property
    def transfer(self):
        return dict(self._raw.get("transfer", {}))

    @cached_property
    def steps(self):
        from types import SimpleNamespace

        rows = []
        for source in self._raw.get("chain_steps", ()):
            row = dict(source)
            for field in ("source_energy", "energy", "source_cost", "cost",
                          "delta_energy", "delta_cost", "temperature"):
                row[field] = self._in_unit(row[field])
            row["current"] = row["proposed"] if row["accepted"] else row["source"]
            row["current_energy"] = row["energy"] if row["accepted"] else row["source_energy"]
            rows.append(SimpleNamespace(**row))
        return tuple(rows)

    @property
    def final_max_gradient(self):
        return self._raw.get("final_max_gradient", 0.0)

    @property
    def optimizer_failures(self):
        """Inner minimizations that did not certify convergence."""
        if "optimizer_failures" in self._raw:
            return list(self._raw["optimizer_failures"])
        failures = [
            (step, row[6])
            for step, row in enumerate(self._raw.get("iterations", ()), 1)
            if not row[5]
        ]
        seed = self._raw.get("seed_relaxation")
        if seed is not None and not seed["success"]:
            failures.insert(0, ("transferred sequence", seed["message"]))
        return failures

    @cached_property
    def iterations(self):
        from types import SimpleNamespace
        from .analysis.states import SolverView

        labels = self._problem.pool_labels if self._problem is not None else []
        output = []
        errors = self._raw.get("iteration_energy_errors_ha") or ()
        for position, row in enumerate(self._raw.get("iterations", ())):
            index, energy, gradient, steps, evaluations, success, message, count = row
            error = errors[position] if position < len(errors) else None
            metrics = None
            if self._profile is True or (
                self._profile is None and self._problem.num_qubits <= 12
            ):
                metrics = SolverView(
                    self._problem, {"indices": self._raw["indices"][:count]}, None, True
                ).metrics
            label = labels[index]
            output.append(
                SimpleNamespace(
                    operator_label=label,
                    operator_kind=label.split("(")[0],
                    max_gradient=gradient,
                    energy=self._in_unit(energy),
                    energy_error=None if error is None else self._in_unit(error),
                    num_parameters=count,
                    cnot_count=None if metrics is None else metrics.cnot_count,
                    depth=None if metrics is None else metrics.depth,
                    metrics=metrics,
                    optimizer_steps=steps,
                    num_evaluations=evaluations,
                    success=success,
                    message=message,
                )
            )
        return tuple(output)

    @property
    def num_operators(self):
        return len(self._raw.get("indices", []))

    @property
    def pruned_operators(self):
        """``(growth step, label)`` of every operator ``prune=True`` removed; empty otherwise."""
        return [(int(step), str(label)) for step, label in self._raw.get("pruned", [])]

    @property
    def metrics(self):
        if self.solver is None:
            raise ValueError("circuit metrics require a quantum result")
        return self.solver.metrics

    @property
    def excitation_energies(self):
        return _native.relative_to_first([float(e) for e in self.energies])

    @property
    def num_particles(self):
        if self._crystal is not None:
            return _native.electron_counts(
                np.ascontiguousarray(self._raw["occupations"], dtype=float),
                [float(w) for w in self._raw["weights"]])
        return (
            tuple(self._raw["num_particles"])
            if self._problem is None
            else tuple(self._problem.num_particles)
        )

    @property
    def n_spatial_orbitals(self):
        if self._crystal is not None:
            return self._crystal.n_orbitals
        return (
            self._integrals.n_orbitals
            if self._problem is None
            else self._problem.n_spatial_orbitals
        )

    @property
    def timings(self):
        return {
            "memory_peak_bytes": self.context.memory_peak_bytes,
            "solve_seconds": self._raw.get("elapsed_seconds", 0.0),
            "termination": self._raw.get(
                "message", "converged" if self.success else "not converged"
            ),
        }

    @property
    def hamiltonian(self):
        """Explicit request for the transformed molecular Hamiltonian."""
        if self._reference is None:
            raise ValueError("a molecular reference is required for integral export")
        if self._model is None:
            representation, truncation = self._two_body
            if self._active_space is None and representation == "dense":
                model = self._reference.model
            else:
                import json

                # The double-factorized representation of the whole space is the active
                # space of every orbital, frozen none.
                model = self._reference.active_model(
                    json.dumps(self._active_space or {}),
                    representation=representation, truncation=truncation)
            self._model = model
        return self._model

    def qubit_hamiltonian(self, mapping=None):
        mapping = self._mapping if mapping is None else mapping
        if self._problem is not None:
            if mapping != self._problem.mapping:
                raise ValueError("mapping disagrees with the native quantum problem")
            return self.context.pauli_operator(
                self._problem.num_qubits, self._problem.hamiltonian_terms
            )
        return self.hamiltonian.operator(mapping)

    def save_hamiltonian(self, path, *, mapping=None, format=None):
        """Write the molecular Hamiltonian as a native qubit-Hamiltonian file.

        ``format`` is ``"json"`` or ``"parquet"``; ``None`` follows the path
        (``.parquet`` and ``.pq`` write Parquet) and falls back to JSON. Both
        formats carry the same document, including the native term groups
        that Trotterized runs read back.
        """
        from .core.serialization import (EXTENSION_FORMATS, resolve_format,
                                         warn_extension_mismatch)

        mapping = self._mapping if mapping is None else mapping
        path = os.fspath(path)
        if format is None:
            extension = os.path.splitext(path)[1].lower()
            format = EXTENSION_FORMATS.get(extension, "json")
        format = resolve_format(format)
        warn_extension_mismatch(path, format, stacklevel=2)
        # The native model owns the document (its term groups included) and
        # writes it in either format directly, with no staged JSON file.
        if format == "json":
            self.hamiltonian.save(path, mapping=mapping)
        else:
            self.hamiltonian.save_parquet(path, mapping=mapping)
        return path

    def as_quantum_problem(self):
        """Export native molecular orbitals and the physical many-electron Hamiltonian."""
        model = self.hamiltonian
        return {
            "hamiltonian": model.operator(self._mapping),
            "n_spatial_orbitals": model.n_spatial_orbitals,
            "num_particles": tuple(model.num_particles),
            "mapping": self._mapping,
        }

    def _molecular_observables(self, *, state=0, **options):
        if self._problem is not None and self._raw.get("state_kind") == "none":
            raise ValueError(
                "a shot-based run builds no state vector (its Hamiltonian, pool and observables "
                "are Pauli operators measured on the device), so it has no density matrices or "
                "orbitals to read; run with shots=0 for the state"
            )
        if self._reference is None or self._integrals is None:
            raise ValueError(
                "molecular fields require the calculation's retained basis and orbitals"
            )
        if self._problem is None:
            if state != 0:
                raise IndexError("a mean-field calculation has one state")
            return self._reference.observables(**options)
        states = self._raw.get("state", self._raw.get("states"))
        if np.ndim(states) == 1:
            if state != 0:
                raise IndexError("this calculation has one state")
        else:
            states = states[state]
        gamma, _ = self._problem.rdms(np.ascontiguousarray(states), two_body=False)
        model = self.hamiltonian
        return self._reference.observables(
            gamma=gamma,
            model=model,
            **options,
        )

    def natural_orbitals(self, *, state=0):
        """Spin-summed natural orbitals diagonalized in Rust, in occupation order."""
        from .analysis.fields import NaturalOrbitals

        values = self._molecular_observables(state=state)
        return NaturalOrbitals(**values)

    def volumetric_field(
        self, quantity="density", index=0, *, state=0, component="auto", grid=None
    ):
        """Sample a molecular density, density difference or natural orbital in Rust."""
        from . import _native
        from .analysis.fields import SamplingGrid, VolumetricField

        if isinstance(grid, SamplingGrid):
            grid = grid.native
        elif grid is not None and not isinstance(grid, _native.SpatialGrid):
            grid = _native.SpatialGrid(
                grid.shape,
                [float(grid.X.flat[0]), float(grid.Y.flat[0]), float(grid.Z.flat[0])],
                np.ascontiguousarray(grid.step),
                periodic=grid.periodic,
            )
        values = self._molecular_observables(
            state=state,
            grid=grid,
            quantity=quantity,
            index=index,
            component=component,
        )
        orbital = quantity == "natural_orbital"
        return VolumetricField(
            quantity=quantity,
            data=values["data"],
            grid=SamplingGrid(values["grid"]),
            numbers=np.asarray(values["numbers"]),
            positions=np.asarray(values["positions"]),
            charges=np.asarray(values["charges"], dtype=float),
            units="Bohr^-3/2" if orbital else "e/Bohr^3",
            integral=values["integral"],
            norm=values["norm"],
            n_electrons=values["n_electrons"],
            context=self.context,
            index=index if orbital else None,
            occupation=values["occupations"][index] if orbital else None,
            component=values["component"],
            augmentation_charge=values["augmentation_charge"],
        )

    def mp2_natural_orbitals(self):
        from .analysis.correlation import MP2Result

        return MP2Result(**self.hamiltonian.mp2())

    def scf_state(self, mapping=None):
        if self._problem is not None:
            if mapping is not None and mapping != self._problem.mapping:
                raise ValueError("mapping disagrees with the quantum problem")
            return self._problem.state([], indices=[], trotter=True)
        if self._reference is None:
            raise ValueError("no mean-field reference was retained")
        return self._reference.state(self._mapping if mapping is None else mapping)

    def reference_state(self, mapping=None):
        return self.scf_state(mapping)

    def forces(self, *, internal=False, project_translation=False):
        """Forces in eV/Angstrom, analytic.

        A variational state (PAW-LCAO only) is differentiated with the response of its
        orbitals (canonical Hartree-Fock, MP2 natural orbitals, one CPHF solve) and the
        Lagrangian (Z-vector) correction of its residual parameter gradient;
        ``force_details`` reports ``stationarity``, ``parameter_gradient_norm``,
        ``z_vector_norm`` and the Hessian solve. A stationary mean-field state's gradient, all-electron
        or PAW-LCAO, is analytic, and ``force_details`` names the scheme and the stationarity
        of the state it differentiated.
        """
        if hasattr(self, "_periodic_forces"):
            return self._periodic_forces
        if self._problem is not None:
            if "states" in self._raw:
                raise ValueError(
                    "variational forces are refused for the states of a subspace method: an ensemble has no "
                    "single parameter gradient, and its state-averaged Lagrangian correction is not implemented")
            if "execution" in self._raw:
                driver = getattr(self, "_device", lambda: None)()
                if self._reference is None or self._model is None or driver is None:
                    raise ValueError("device forces need the molecular reference and the calculator that ran the circuit")
                from .integrations.device_forces import device_force

                if not hasattr(self, "_device_forces"):
                    self._device_forces = {}
                if bool(internal) not in self._device_forces:
                    self._device_forces[bool(internal)] = device_force(self, driver, internal)
                gradient, details = self._device_forces[bool(internal)]
                self.force_details = dict(details)
                if project_translation and not internal:
                    gradient = self.context.remove_force_translation(np.ascontiguousarray(gradient))
                return _native.to_user_units(np.ascontiguousarray(gradient, dtype=float), "gradient_to_force",
                                             plus=self._dispersion_forces())
            if self._reference is None or self._model is None or "state" not in self._raw:
                raise ValueError("variational forces need a single molecular state and its retained reference")
            if not hasattr(self, "_gradient_cache"):
                self._gradient_cache = {}
            cache_key = bool(internal)
            if cache_key not in self._gradient_cache:
                self._gradient_cache[cache_key] = self._reference.quantum_gradient(
                    self._problem, self._model,
                    np.ascontiguousarray(self._raw["state"]),
                    internal=internal,
                    indices=[int(i) for i in self._raw["indices"]],
                    parameters=np.ascontiguousarray(self._raw["parameters"], dtype=float),
                    product_form=self.solver.ansatz.product_form)
            gradient = self._gradient_cache[cache_key]["gradient"]
            self.force_details = {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in self._gradient_cache[cache_key].items()
            }
        elif self.periodic is not None:
            # The analytic crystal force of the Rust engine (Hartree/Bohr), converted once here.
            if not hasattr(self, "_crystal_forces"):
                self._crystal_forces = self.periodic.forces()
            return _native.to_user_units(
                np.ascontiguousarray(self._crystal_forces, dtype=float), "force",
                plus=self._dispersion_forces())
        elif self._reference is None:
            raise ValueError("stationary forces require a native mean-field state")
        else:
            stationary = self._reference.gradient()
            gradient = stationary["gradient"]
            self.force_details = {
                key: value for key, value in stationary.items()
                if not isinstance(value, np.ndarray)
            }
        if project_translation and not internal:
            gradient = self.context.remove_force_translation(np.ascontiguousarray(gradient))
        return _native.to_user_units(
            np.ascontiguousarray(gradient, dtype=float), "gradient_to_force",
            plus=self._dispersion_forces())

    def _dispersion_forces(self):
        """The dispersion forces (eV/Angstrom) a result adds to its own, or ``None``."""
        if self._dispersion is not None and "forces_ev_angstrom" in self._dispersion:
            return np.ascontiguousarray(self._dispersion["forces_ev_angstrom"], dtype=float)
        return None

    def stress(self, voigt=True):
        """Stress tensor in eV/Angstrom^3 (ASE Voigt order ``[xx, yy, zz, yz, xz, xy]`` by default, or 3x3).

        The analytic stress of the Rust engine at the pins of the crystal (node counts, ion width,
        basis): ``sigma = (g + g^T) / (2 Omega)`` with ``g_ab = dF/d eps_ab``, the sign ASE uses.
        """
        if hasattr(self, "_crystal_stress"):
            s = self._crystal_stress
            if s.shape == (6,):
                from ase.stress import voigt_6_to_full_3x3_stress
                full = _native.to_user_units(
                    np.ascontiguousarray(voigt_6_to_full_3x3_stress(s), dtype=float), "stress")
                return _native.to_user_units(np.ascontiguousarray(s, dtype=float), "stress") \
                    if voigt else full
            else:
                full = _native.to_user_units(np.ascontiguousarray(s, dtype=float), "stress")
                if not voigt:
                    return np.array(full, copy=True)
                return np.array([full[0, 0], full[1, 1], full[2, 2], full[1, 2], full[0, 2], full[0, 1]])
        if self.periodic is None:
            raise ValueError("stress tensor requires a periodic crystal calculation")
        if self._dispersion is not None:
            raise NotImplementedError(
                "the crystal stress does not include the Tkatchenko-Scheffler dispersion "
                "(its stress is not available), so it would not be the derivative of the energy"
            )
        if not hasattr(self, "_crystal_stress"):
            self._crystal_stress = self.periodic.stress()["sigma"]
        full = _native.to_user_units(
            np.ascontiguousarray(self._crystal_stress, dtype=float), "stress")
        if not voigt:
            return np.array(full, copy=True)
        return np.array([full[0, 0], full[1, 1], full[2, 2], full[1, 2], full[0, 2], full[0, 1]])

    def snapshot(self, *, include_hamiltonian=False):
        if self._reference is not None and self._problem is None:
            raw = self._reference.snapshot(include_hamiltonian=include_hamiltonian)
        else:
            raw = {
                k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v
                for k, v in self._raw.items()
            }
        if self._dispersion is not None:
            raw["dispersion"] = self._dispersion
        return raw

    def state(self):
        if "state" in self._raw:
            return self._problem.expand_state(np.ascontiguousarray(self._raw["state"]))
        if "states" in self._raw:
            return self._problem.expand_state(np.ascontiguousarray(self._raw["states"]))
        return self.scf_state()
