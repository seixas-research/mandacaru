"""Optional ASE adapter for the application-independent calculation API."""

from ase import Atoms
from ase.calculators.calculator import (
    Calculator,
    all_changes,
    PropertyNotImplementedError,
)
from ..calculation import Mandacaru
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


class ASECalculator(Calculator):
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

    def _periodic_snapshot(self):
        result = self.calculation.result
        if result is None or result.periodic is None:
            raise RuntimeError("run a periodic calculation first")
        return result.snapshot()

    def get_fermi_level(self):
        return self._periodic_snapshot()["fermi_level"] * HARTREE_TO_EV

    def get_number_of_spins(self):
        return self._periodic_snapshot()["n_spins"]

    def get_number_of_bands(self):
        return self._periodic_snapshot()["eigenvalues"].shape[-1]

    def get_eigenvalues(self, kpt=0, spin=0):
        return self._periodic_snapshot()["eigenvalues"][spin, kpt] * HARTREE_TO_EV

    def get_ibz_k_points(self):
        return self._periodic_snapshot()["fractional_kpoints"]

    def get_k_point_weights(self):
        return np.asarray(self._periodic_snapshot()["weights"])

    def get_bz_k_points(self):
        calculation = self.calculation
        if calculation._load_crystal is not None:
            raise NotImplementedError("a cached crystal stores its irreducible mesh")
        return calculation.context.kmesh(
            calculation._kpts_spec["size"],
            gamma=calculation._kpts_spec["gamma"],
            time_reversal=False,
        )["full"]

    def band_structure(self, path=None, npoints=100):
        from ase.spectrum.band_structure import BandStructure
        from .paw import band_input

        calculation = self.calculation
        self._periodic_snapshot()
        if not hasattr(calculation, "_crystal_adapter"):
            raise NotImplementedError("off-mesh bands require the PAW setup adapter")
        bandpath = self.atoms.cell.bandpath(
            path=path, npoints=npoints, pbc=self.atoms.pbc
        )
        metadata, arrays = band_input(calculation._crystal_adapter, bandpath.kpts)
        native = calculation.result.periodic.bands(metadata, arrays)
        return BandStructure(
            bandpath,
            energies=native["eigenvalues"] * HARTREE_TO_EV,
            reference=self.get_fermi_level(),
        )

    def dos(self, width=0.1, npoints=2001, energies=None):
        levels = self._periodic_snapshot()["eigenvalues"] * HARTREE_TO_EV
        axis = (
            np.linspace(levels.min() - 8 * width, levels.max() + 8 * width, npoints)
            if energies is None
            else np.asarray(energies, dtype=float)
        )
        values = self.calculation.result.periodic.dos(
            np.ascontiguousarray(axis / HARTREE_TO_EV), width_ha=width / HARTREE_TO_EV
        )
        return axis, values[:, 0] / HARTREE_TO_EV

    def pdos(self, width=0.1, npoints=2001, energies=None):
        axis, _ = self.dos(width, npoints, energies)
        calculation = self.calculation
        if not hasattr(calculation, "_orbital_groups"):
            raise NotImplementedError("a cached crystal has no atom-to-orbital labels")
        values = (
            calculation.result.periodic.dos(
                np.ascontiguousarray(axis / HARTREE_TO_EV),
                width_ha=width / HARTREE_TO_EV,
                projected=True,
            )
            / HARTREE_TO_EV
        )
        groups = {}
        for column, shell in enumerate(calculation._orbital_groups):
            groups[shell] = groups.get(shell, 0.0) + values[:, column]
        return axis, groups
