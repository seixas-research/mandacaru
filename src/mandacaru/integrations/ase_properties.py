"""ASE electronic-structure views over a calculator's native state."""

import numpy as np
from ..units import HARTREE_TO_EV


class ASEProperties:
    def wannier(self, n_functions=None, *, guess="bonds", windows=None,
                bands=None, kpts=None, max_iterations=2000, tolerance=1e-10,
                step=0.5):
        """Localize periodic Bloch states with the native PAW Wannier solver.

        Energy windows use eV on the ASE eigenvalue scale. Centers and
        spreads are returned in Angstrom and Angstrom squared.
        """
        from .wannier import calculate

        return calculate(self._calculation_owner(), n_functions, guess=guess,
                         windows=windows, bands=bands, kpts=kpts,
                         max_iterations=max_iterations, tolerance=tolerance,
                         step=step)

    def _calculation_owner(self):
        return self

    def _periodic_snapshot(self):
        result = self._calculation_owner().result
        if result is None or result.periodic is None:
            raise RuntimeError("run a periodic calculation first")
        return result.snapshot()

    def _spectrum(self):
        calculation = self._calculation_owner()
        if calculation.result is None:
            raise RuntimeError("run a calculation before requesting electronic levels")
        if calculation.result.periodic is not None:
            state = self._periodic_snapshot()
            return np.asarray(state["eigenvalues"]), np.asarray(state["weights"])
        reference = calculation.native_reference
        if reference is None:
            raise RuntimeError(
                "a cached quantum Hamiltonian has no molecular orbital spectrum"
            )
        return np.asarray(reference.snapshot()["energies"])[:, None, :], np.ones(1)

    def get_fermi_level(self):
        calculation = self._calculation_owner()
        if calculation.result is not None and calculation.result.periodic is not None:
            return self._periodic_snapshot()["fermi_level"] * HARTREE_TO_EV
        levels, _ = self._spectrum()
        particles, _ = self._molecular_occupations(levels)
        occupied = [
            levels[spin, 0, particles[spin] - 1]
            for spin in range(levels.shape[0])
            if particles[spin]
        ]
        if not occupied:
            raise ValueError("an empty molecule has no occupied Fermi level")
        return max(occupied) * HARTREE_TO_EV

    def _molecular_occupations(self, levels):
        calculation = self._calculation_owner()
        particles = calculation.native_reference.num_particles
        if calculation.method == "ghf":
            return (sum(particles),), 1.0
        if levels.shape[0] == 1:
            return (particles[0],), 2.0
        return particles, 1.0

    def get_number_of_spins(self):
        return self._spectrum()[0].shape[0]

    def get_spin_polarized(self):
        return self.get_number_of_spins() == 2

    def get_number_of_bands(self):
        return self._spectrum()[0].shape[-1]

    def get_eigenvalues(self, kpt=0, spin=0):
        return self._spectrum()[0][spin, kpt].copy() * HARTREE_TO_EV

    def get_occupation_numbers(self, kpt=0, spin=0):
        calculation = self._calculation_owner()
        if calculation.result is not None and calculation.result.periodic is not None:
            return self._periodic_snapshot()["occupations"][spin, kpt].copy()
        levels, _ = self._spectrum()
        particles, degeneracy = self._molecular_occupations(levels)
        occupations = np.zeros_like(levels[spin, kpt])
        occupations[: particles[spin]] = degeneracy
        return occupations

    def get_ibz_k_points(self):
        calculation = self._calculation_owner()
        if calculation.result is not None and calculation.result.periodic is not None:
            return self._periodic_snapshot()["fractional_kpoints"]
        return np.zeros((1, 3))

    def get_k_point_weights(self):
        return self._spectrum()[1].copy()

    def get_bz_k_points(self):
        calculation = self._calculation_owner()
        if calculation._load_crystal is not None:
            raise NotImplementedError("a cached crystal stores its irreducible mesh")
        return calculation.context.kmesh(
            calculation._kpts_spec["size"],
            gamma=calculation._kpts_spec["gamma"],
            time_reversal=False,
        )["full"]

    def _bands_at(self, fractional):
        from .paw import band_input

        calculation = self._calculation_owner()
        if not hasattr(calculation, "_crystal_adapter"):
            raise NotImplementedError(
                "off-mesh bands require geometry and its PAW setup"
            )
        adapter = calculation._crystal_adapter
        m, points = adapter.M, adapter.grid.X.size
        available = (
            calculation.context.memory_limit_bytes
            - calculation.context.memory_used_bytes
        )
        per_point = 3 * 16 * m * points + 16 * m * m * 20
        block = max(1, min(8, available // max(1, 2 * per_point)))
        chunks = []
        for start in range(0, len(fractional), block):
            metadata, arrays = band_input(adapter, fractional[start : start + block])
            native = calculation.result.periodic.bands(metadata, arrays)
            chunks.append(native["eigenvalues"])
        return np.concatenate(chunks, axis=1)

    def band_structure(self, path=None, npoints=100):
        from ase.spectrum.band_structure import BandStructure

        self._periodic_snapshot()
        calculation = self._calculation_owner()
        atoms = calculation.atoms if calculation.atoms is not None else self.atoms
        bandpath = atoms.cell.bandpath(path=path, npoints=npoints, pbc=atoms.pbc)
        return BandStructure(
            bandpath,
            energies=self._bands_at(bandpath.kpts) * HARTREE_TO_EV,
            reference=self.get_fermi_level(),
        )

    def dos(self, width=0.1, npoints=2001, energies=None, kpts=None):
        calculation = self._calculation_owner()
        if not np.isfinite(width) or width <= 0:
            raise ValueError("DOS width must be positive and finite")
        levels, weights = self._spectrum()
        if kpts is not None:
            self._periodic_snapshot()
            spec = kpts if isinstance(kpts, dict) else {"size": kpts, "gamma": True}
            mesh = calculation.context.kmesh(
                spec["size"], gamma=spec.get("gamma", True), time_reversal=False
            )
            fractional = mesh["full"]
            levels = self._bands_at(fractional)
            weights = np.full(len(fractional), 1.0 / len(fractional))
        low, high = levels.min() * HARTREE_TO_EV, levels.max() * HARTREE_TO_EV
        axis = (
            np.linspace(low - 8 * width, high + 8 * width, npoints)
            if energies is None
            else np.asarray(energies, dtype=float)
        )
        degeneracy = 2.0 if levels.shape[0] == 1 else 1.0
        if calculation.result.periodic is not None and self._periodic_snapshot()["spinor"]:
            degeneracy = 1.0
        if calculation.native_reference is not None and calculation.method == "ghf":
            degeneracy = 1.0
        values = calculation.context.density_of_states(
            np.ascontiguousarray(levels, dtype=np.float64),
            np.ascontiguousarray(weights, dtype=np.float64),
            np.ascontiguousarray(axis / HARTREE_TO_EV),
            width_ha=width / HARTREE_TO_EV,
            degeneracy=degeneracy,
        )
        return axis, values / HARTREE_TO_EV

    def pdos(self, width=0.1, npoints=2001, energies=None):
        axis, _ = self.dos(width, npoints, energies)
        calculation = self._calculation_owner()
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
