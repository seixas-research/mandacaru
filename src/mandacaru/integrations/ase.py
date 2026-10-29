"""Optional ASE adapter for the application-independent calculation API."""

from ase import Atoms
from ase.calculators.calculator import (
    Calculator,
    all_changes,
    PropertyNotImplementedError,
)
from ..calculation import Mandacaru
from .ase_properties import ASEProperties
from ..structure import geometry_value
import numpy as np
from ..units import HARTREE_TO_EV


def geometry_from_atoms(atoms):
    """Copy an ASE geometry into native ownership in Bohr."""
    return geometry_value(atoms).native()


def atoms_from_geometry(geometry):
    native = geometry.native()
    from ..units import BOHR_TO_ANGSTROM

    atoms = Atoms(
        numbers=native.numbers,
        positions=native.positions_bohr * BOHR_TO_ANGSTROM,
        cell=native.cell_bohr * BOHR_TO_ANGSTROM,
        pbc=native.pbc,
    )
    atoms.set_initial_magnetic_moments(geometry.magnetic_moments)
    return atoms


class ASECalculator(ASEProperties, Calculator):
    def __init__(self, calculation=None, **options):
        super().__init__()
        if calculation is not None and options:
            raise ValueError("supply either a calculation or its constructor options")
        self.calculation = Mandacaru(**options) if calculation is None else calculation

    @property
    def implemented_properties(self):
        return self.calculation.implemented_properties

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        try:
            self.calculation.calculate(
                geometry_value(self.atoms), properties=properties
            )
        except NotImplementedError as exc:
            raise PropertyNotImplementedError(str(exc)) from exc
        self.results = dict(self.calculation.results)

    def close(self):
        self.calculation.close()

    def _calculation_owner(self):
        return self.calculation
