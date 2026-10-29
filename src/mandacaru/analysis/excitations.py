"""Unit-aware views over native sequential deflation results."""

from dataclasses import dataclass
import numpy as np
from ..units import HARTREE_TO_EV


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
        return self.energies - self.energies[0]

    @property
    def gaps(self):
        return np.diff(self.energies)

    def in_units(self, units="eV"):
        if units not in ("Ha", "eV"):
            raise ValueError("energy units must be Ha or eV")
        factor = (1.0 if self.energy_unit == "Ha" else 1.0 / HARTREE_TO_EV) * (
            1.0 if units == "Ha" else HARTREE_TO_EV
        )
        return self.energies.copy() * factor

    def excitation_energies_in_units(self, units="eV"):
        values = self.in_units(units)
        return values - values[0]
