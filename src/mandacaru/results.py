"""Result views over native-owned scientific states."""

from __future__ import annotations
import numpy as np
from functools import cached_property
from .units import HARTREE_TO_EV, BOHR_TO_ANGSTROM


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
        atomic_units=False,
        solver=None,
        model=None,
        profile=None,
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
        self.energy_unit = "Ha" if atomic_units else "eV"

    @property
    def success(self):
        return bool(self._raw.get("converged", self._raw.get("success", False)))

    @property
    def converged(self):
        return self.success

    @property
    def energy_hartree(self):
        if self._reference is not None and self._problem is None:
            return self._raw["electronic_energy"] + self._raw["nuclear_repulsion"]
        if "energies_ha" in self._raw:
            return float(self._raw["energies_ha"][0])
        return self._raw.get(
            "energy_ha", self._raw.get("extrapolated_energy", self._raw.get("energy"))
        )

    def in_units(self, units):
        if units == "Ha":
            return self.energy_hartree
        if units == "eV":
            return self.energy_hartree * HARTREE_TO_EV
        raise ValueError("energy units must be Ha or eV")

    @property
    def free_energy_ev(self):
        return self._raw.get("free_energy", self.energy_hartree) * HARTREE_TO_EV

    @property
    def optimal_energy(self):
        return self.in_units(self.energy_unit)

    @property
    def energies(self):
        values = np.asarray(
            self._raw.get("energies_ha", [self.energy_hartree]), dtype=float
        )
        return values.copy() if self.energy_unit == "Ha" else values * HARTREE_TO_EV

    @property
    def states(self):
        return self.state()

    @property
    def reference_energy(self):
        value = (
            self.energy_hartree
            if self._reference is not None and self._problem is None
            else self._raw.get("reference_ha", self.energy_hartree)
        )
        return value if self.energy_unit == "Ha" else value * HARTREE_TO_EV

    @property
    def correlation_energy(self):
        return self.optimal_energy - self.reference_energy

    @property
    def scf(self):
        return self._reference

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
        return self._raw.get("optimal_cost_ha", self.energy_hartree) * (
            1. if self.energy_unit == "Ha" else HARTREE_TO_EV)

    @property
    def best_energy(self):
        return self._raw.get("best_energy_ha", self.energy_hartree) * (
            1. if self.energy_unit == "Ha" else HARTREE_TO_EV)

    @property
    def final_energy(self):
        return self._raw.get("final_energy_ha", self.energy_hartree) * (
            1. if self.energy_unit == "Ha" else HARTREE_TO_EV)

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
    def transfer(self):
        return dict(self._raw.get("transfer", {}))

    @cached_property
    def steps(self):
        from types import SimpleNamespace

        factor = 1. if self.energy_unit == "Ha" else HARTREE_TO_EV
        rows = []
        for source in self._raw.get("chain_steps", ()):
            row = dict(source)
            for field in ("source_energy", "energy", "source_cost", "cost",
                          "delta_energy", "delta_cost", "temperature"):
                row[field] *= factor
            row["current"] = row["proposed"] if row["accepted"] else row["source"]
            row["current_energy"] = row["energy"] if row["accepted"] else row["source_energy"]
            rows.append(SimpleNamespace(**row))
        return tuple(rows)

    @property
    def final_max_gradient(self):
        return self._raw.get("final_max_gradient", 0.0)

    @property
    def optimizer_failures(self):
        """Growth steps whose inner minimizer did not certify convergence."""
        if "optimizer_failures" in self._raw:
            return list(self._raw["optimizer_failures"])
        return [
            (step, row[6])
            for step, row in enumerate(self._raw.get("iterations", ()), 1)
            if not row[5]
        ]

    @cached_property
    def iterations(self):
        from types import SimpleNamespace
        from .analysis.states import SolverView

        labels = self._problem.pool_labels if self._problem is not None else []
        output = []
        for row in self._raw.get("iterations", ()):
            index, energy, gradient, steps, evaluations, success, message, count = row
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
                    energy=energy
                    * (1.0 if self.energy_unit == "Ha" else HARTREE_TO_EV),
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
    def metrics(self):
        if self.solver is None:
            raise ValueError("circuit metrics require a quantum result")
        return self.solver.metrics

    @property
    def excitation_energies(self):
        energies = self.energies
        return energies - energies[0]

    @property
    def num_particles(self):
        if self._crystal is not None:
            occupations = self._raw["occupations"]
            counts = np.sum(
                occupations * np.asarray(self._raw["weights"])[None, :, None],
                axis=(1, 2),
            )
            return tuple(counts) if len(counts) == 2 else (counts[0] / 2, counts[0] / 2)
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
            "threads": self.context.threads,
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
            if self._active_space is None:
                model = self._reference.model
            else:
                import json

                model = self._reference.active_model(json.dumps(self._active_space))
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
        mapping = self._mapping if mapping is None else mapping
        if format == "parquet" or (format is None and str(path).endswith(".parquet")):
            from .core.mapping import PauliSum
            from .core.serialization import save_hamiltonian

            operator = self.hamiltonian.operator(mapping)
            terms = {label: complex(re, im) for label, re, im in operator.terms}
            return save_hamiltonian(
                path,
                PauliSum(terms, num_qubits=operator.num_qubits),
                mapping=mapping,
                num_particles=self.hamiltonian.num_particles,
                n_spatial_orbitals=self.hamiltonian.n_spatial_orbitals,
                format="parquet",
            )
        self.hamiltonian.save(str(path), mapping=mapping)
        return str(path)

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
        if self._problem is not None:
            if self._reference is None or self._model is None or "state" not in self._raw:
                raise ValueError("variational forces need a single molecular state and its retained reference")
            if "execution" in self._raw:
                raise ValueError("variational forces currently require the local native state")
            import json

            if not hasattr(self, "_gradient_cache"):
                self._gradient_cache = {}
            if internal not in self._gradient_cache:
                self._gradient_cache[internal] = self._reference.quantum_gradient(
                    self._problem, self._model,
                    np.ascontiguousarray(self._raw["state"]),
                    active_spec=None if self._active_space is None else json.dumps(self._active_space),
                    internal=internal)
            gradient = self._gradient_cache[internal]["gradient"]
            self.force_details = dict(self._gradient_cache[internal])
        elif self._reference is None:
            raise ValueError("stationary forces require a native mean-field state")
        else:
            gradient = self._reference.gradient()["gradient"]
        if project_translation and not internal:
            gradient = self.context.remove_force_translation(np.ascontiguousarray(gradient))
        return -gradient * (HARTREE_TO_EV / BOHR_TO_ANGSTROM)

    def snapshot(self, *, include_hamiltonian=False):
        if self._reference is not None and self._problem is None:
            return self._reference.snapshot(include_hamiltonian=include_hamiltonian)
        return {
            k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v
            for k, v in self._raw.items()
        }

    def state(self):
        if "state" in self._raw:
            return self._problem.expand_state(np.ascontiguousarray(self._raw["state"]))
        if "states" in self._raw:
            return self._problem.expand_state(np.ascontiguousarray(self._raw["states"]))
        return self.scf_state()
