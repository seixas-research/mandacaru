"""Unit-aware views over native sequential deflation results."""

from dataclasses import dataclass
import numpy as np
from .. import _native


@dataclass
class EnergyLevels:
    energies: np.ndarray
    states: np.ndarray
    reference_energy: float
    num_evaluations: int
    success: tuple[bool, ...]
    messages: tuple[str, ...]
    energy_unit: str = "eV"

    @property
    def num_states(self):
        return len(self.energies)

    @property
    def ground_state_energy(self):
        return float(self.energies[0])

    @property
    def excitation_energies(self):
        return _native.relative_to_first([float(e) for e in self.energies])

    @property
    def gaps(self):
        return _native.gaps([float(e) for e in self.energies])

    def in_units(self, units="eV"):
        if units not in ("Ha", "eV"):
            raise ValueError("energy units must be Ha or eV")
        if units == self.energy_unit:
            return self.energies.copy()
        return _native.convert_energy(
            np.ascontiguousarray(self.energies, dtype=float), self.energy_unit, units)

    def excitation_energies_in_units(self, units="eV"):
        return _native.relative_to_first([float(e) for e in self.in_units(units)])
