"""Result views over native-owned scientific states."""

from __future__ import annotations
import numpy as np
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
    ):
        self.method = method
        self.context = context
        self._reference = reference
        self._integrals = integrals
        self._problem = problem
        self.periodic = periodic
        self._crystal = crystal
        self._raw = reference.snapshot() if raw is None else raw
        self._model = None
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
        if self._reference is not None:
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
            if self._reference is not None
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
    def final_max_gradient(self):
        return self._raw.get("final_max_gradient", 0.0)

    @property
    def num_operators(self):
        return len(self._raw.get("indices", []))

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
            model = self._reference.model
            if self._active_space is not None:
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

    def save_hamiltonian(self, path, *, mapping=None):
        self.hamiltonian.save(
            str(path), mapping=self._mapping if mapping is None else mapping
        )

    def mp2_natural_orbitals(self):
        from .analysis.correlation import MP2Result

        return MP2Result(**self.hamiltonian.mp2())

    def scf_state(self, mapping=None):
        if self._reference is None:
            raise ValueError("no mean-field reference was retained")
        return self._reference.state(self._mapping if mapping is None else mapping)

    def reference_state(self, mapping=None):
        return self.scf_state(mapping)

    def forces(self):
        if self._reference is None:
            raise NotImplementedError(
                "stationary forces require a native mean-field state"
            )
        return -self._reference.gradient()["gradient"] * (
            HARTREE_TO_EV / BOHR_TO_ANGSTROM
        )

    def snapshot(self, *, include_hamiltonian=False):
        if self._reference is not None:
            return self._reference.snapshot(include_hamiltonian=include_hamiltonian)
        return {
            k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v
            for k, v in self._raw.items()
        }

    def state(self):
        if "state" in self._raw:
            return np.array(self._raw["state"], copy=True)
        if "states" in self._raw:
            return np.array(self._raw["states"], copy=True)
        return self.scf_state()
