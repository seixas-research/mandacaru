"""ASE electronic-structure views over a calculator's native state."""

import numpy as np
from .. import _native
from ..units import HARTREE_TO_EV
from . import bloch as _bloch


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
        if _bloch.is_supercell(calculation):
            # The supercell mean field unfolded onto the mesh points, per primitive cell.
            points = _bloch.supercell(calculation)["kpoints"]
            levels, _ = _bloch.bands_at(calculation, points)
            return levels, np.full(len(points), 1.0 / len(points))
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
        if _bloch.is_supercell(calculation):
            # The highest occupied level of the supercell mean field.
            levels = np.asarray(calculation.native_reference.snapshot()["energies"])[:, None, :]
        else:
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
        if _bloch.is_supercell(calculation):
            raise NotImplementedError(
                "the unfolded mean-field bands of a supercell carry no per-k occupations; "
                "the correlated occupations are in get_spectral_function()"
            )
        if calculation.result is not None and calculation.result.periodic is not None:
            return self._periodic_snapshot()["occupations"][spin, kpt].copy()
        levels, _ = self._spectrum()
        particles, degeneracy = self._molecular_occupations(levels)
        occupations = np.zeros_like(levels[spin, kpt])
        occupations[: particles[spin]] = degeneracy
        return occupations

    def get_ibz_k_points(self):
        calculation = self._calculation_owner()
        if _bloch.is_supercell(calculation):
            return np.asarray(_bloch.supercell(calculation)["kpoints"])
        if calculation.result is not None and calculation.result.periodic is not None:
            return self._periodic_snapshot()["fractional_kpoints"]
        return np.zeros((1, 3))

    def get_k_point_weights(self):
        return self._spectrum()[1].copy()

    @property
    def kpoints(self):
        """The mesh points a supercell calculation resolves (fractional, ``(nk, 3)``)."""
        return np.asarray(_bloch.supercell(self._calculation_owner())["kpoints"])

    def get_spectral_function(self, energies=None, eta=0.1, kpoints=None, krylov=200):
        """The Lehmann spectral function ``A(k, omega)`` of the correlated state.

        Available after a quantum method on a crystal (``vqe``, ``adapt-vqe``, ``vasqa``,
        ``valsqa`` on a periodic structure), without shots, tapering or an active space. ``energies`` is the axis in eV (by default
        2001 points spanning the poles), ``eta`` the Lorentzian half-width in eV, ``kpoints``
        a subset of the mesh points (fractional; the default is all of them) and ``krylov`` the
        Lanczos vectors per part. See :class:`~mandacaru.analysis.bloch.SpectralFunction`.
        """
        return _bloch.spectral_function(
            self._calculation_owner(), energies=energies, eta=eta, kpoints=kpoints,
            krylov=krylov,
        )

    def bands(self, spin=0, **options):
        """``(nk, n_orbitals)``: the strongest pole of each orbital channel of ``A(k, omega)`` (eV)."""
        return self.get_spectral_function(**options).bands(spin)

    def band_weights(self, spin=0, **options):
        """``(nk, n_orbitals)``: the weight of the pole :meth:`bands` reports."""
        return self.get_spectral_function(**options).band_weights(spin)

    def get_bz_k_points(self):
        calculation = self._calculation_owner()
        if _bloch.is_supercell(calculation):
            return np.asarray(_bloch.supercell(calculation)["kpoints"])
        if calculation._load_crystal is not None:
            raise NotImplementedError("a cached crystal stores its irreducible mesh")
        return calculation.context.kmesh(
            calculation._kpts_spec["size"],
            gamma=calculation._kpts_spec["gamma"],
            time_reversal=False,
        )["full"]

    def _bands_at(self, fractional):
        """Eigenvalues (Hartree) at fractional k-points, from the Rust crystal (``bands_at``)."""
        calculation = self._calculation_owner()
        if _bloch.is_supercell(calculation):
            return _bloch.bands_at(calculation, fractional)[0]
        periodic = getattr(getattr(self._calculation_owner(), "result", None), "periodic", None)
        if periodic is None:
            raise RuntimeError("run periodic DFT before asking for bands off its mesh")
        native = periodic.bands_at([list(map(float, k)) for k in fractional])
        return np.asarray(native["eigenvalues"])

    def band_structure(self, path=None, npoints=100):
        """The mean-field band structure along an ASE band path (eV).

        A DFT crystal is diagonalized at every path point. A quantum method on a crystal
        reports the bands of its supercell mean-field reference unfolded onto the primitive
        cell: exact at the mesh points, a Fourier interpolation of the supercell Fock matrix
        between them. The correlated quasiparticle energies are those of
        :meth:`get_spectral_function`, at the mesh points only.
        """
        from ase.spectrum.band_structure import BandStructure

        calculation = self._calculation_owner()
        if not _bloch.is_supercell(calculation):
            self._periodic_snapshot()
        atoms = calculation.atoms if calculation.atoms is not None else self.atoms
        bandpath = atoms.cell.bandpath(path=path, npoints=npoints, pbc=atoms.pbc)
        return BandStructure(
            bandpath,
            energies=self._bands_at(bandpath.kpts) * HARTREE_TO_EV,
            reference=self.get_fermi_level(),
        )

    def dos(self, width=0.1, npoints=2001, energies=None, kpts=None, integration="gaussian"):
        """Density of states ``(energies, values)`` in eV and states per eV per cell.

        ``integration="gaussian"`` broadens every level by a Gaussian of standard deviation
        ``width``; ``"tetrahedron"`` uses the linear tetrahedron method on the full
        Gamma-centered mesh ``kpts`` (the calculation's mesh by default) and ignores ``width``.
        """
        calculation = self._calculation_owner()
        if integration not in ("gaussian", "tetrahedron"):
            raise ValueError("integration must be 'gaussian' or 'tetrahedron'")
        if integration == "tetrahedron":
            return self._tetrahedron(npoints, energies, kpts)[:2]
        if not np.isfinite(width) or width <= 0:
            raise ValueError("DOS width must be positive and finite")
        levels, weights = self._spectrum()
        if kpts is not None:
            if not _bloch.is_supercell(calculation):
                self._periodic_snapshot()
            spec = kpts if isinstance(kpts, dict) else {"size": kpts, "gamma": True}
            mesh = calculation.context.kmesh(
                spec["size"], gamma=spec.get("gamma", True), time_reversal=False
            )
            fractional = mesh["full"]
            levels = self._bands_at(fractional)
            weights = np.full(len(fractional), 1.0 / len(fractional))
        axis = self._energy_axis(levels, npoints, energies, 8 * width)
        degeneracy = self._degeneracy(levels)
        values = calculation.context.density_of_states(
            np.ascontiguousarray(levels, dtype=np.float64),
            np.ascontiguousarray(weights, dtype=np.float64),
            np.ascontiguousarray(axis / HARTREE_TO_EV),
            width_ha=width / HARTREE_TO_EV,
            degeneracy=degeneracy,
        )
        return axis, values / HARTREE_TO_EV

    @staticmethod
    def _energy_axis(levels, npoints, energies, margin):
        if energies is not None:
            return np.asarray(energies, dtype=float)
        low, high = levels.min() * HARTREE_TO_EV, levels.max() * HARTREE_TO_EV
        return np.linspace(low - margin, high + margin, npoints)

    def _degeneracy(self, levels):
        calculation = self._calculation_owner()
        degeneracy = 2.0 if levels.shape[0] == 1 else 1.0
        if calculation.result.periodic is not None and self._periodic_snapshot()["spinor"]:
            degeneracy = 1.0
        if calculation.native_reference is not None and calculation.method == "ghf":
            degeneracy = 1.0
        return degeneracy

    def _tetrahedron(self, npoints, energies, kpts, projected=False):
        """Native linear-tetrahedron DOS on a full Gamma-centered mesh (eV, per eV per cell)."""
        calculation = self._calculation_owner()
        bloch = _bloch.is_supercell(calculation)
        if not bloch and calculation.result.periodic is None:
            raise ValueError("the tetrahedron method needs a crystal: a molecule has no k mesh")
        if kpts is None:
            size = tuple(_bloch.supercell(calculation)["mesh"]) if bloch else tuple(
                calculation._kpts_spec["size"]
            )
        else:
            spec = kpts if isinstance(kpts, dict) else {"size": kpts, "gamma": True}
            if not spec.get("gamma", True) and any(n % 2 == 0 for n in spec["size"]):
                raise ValueError("the tetrahedron method needs a Gamma-centered mesh")
            size = tuple(spec["size"])
        fractional = calculation.context.kmesh(list(size), gamma=True, time_reversal=False)["full"]
        if bloch:
            levels, weights = _bloch.bands_at(calculation, fractional)
        else:
            raw = calculation.result.periodic.bands_at([list(map(float, k)) for k in fractional])
            levels = np.asarray(raw["eigenvalues"])
            # (spins, nk, orbital, band) -> (spins, nk, band, orbital)
            weights = np.swapaxes(np.asarray(raw["projections"]), 2, 3)
        atoms = calculation.atoms if calculation.atoms is not None else self.atoms
        reciprocal = np.ascontiguousarray(atoms.cell.reciprocal(), dtype=float)
        axis = self._energy_axis(levels, npoints, energies, 0.5)
        raw = _native.tetrahedron_dos(
            calculation.context,
            list(size),
            reciprocal,
            [list(map(float, k)) for k in fractional],
            np.ascontiguousarray(levels, dtype=np.float64),
            np.ascontiguousarray(axis / HARTREE_TO_EV),
            projections=np.ascontiguousarray(weights, dtype=np.float64) if projected else None,
            degeneracy=self._degeneracy(levels),
        )
        values = np.asarray(raw["dos"]) / HARTREE_TO_EV
        projection = np.asarray(raw["projected"]) / HARTREE_TO_EV if projected else None
        return axis, values, projection

    def pdos(self, width=0.1, npoints=2001, energies=None, integration="gaussian", kpts=None):
        """Projected density of states ``(energies, {(atom, l): values})`` in eV.

        The projections are the Lowdin weights of each band on the atomic orbitals, so the
        channels add up to :meth:`dos`. ``integration`` is as in :meth:`dos`; a quantum method
        on a crystal (unfolded supercell bands) takes ``"tetrahedron"`` only.
        """
        calculation = self._calculation_owner()
        if integration not in ("gaussian", "tetrahedron"):
            raise ValueError("integration must be 'gaussian' or 'tetrahedron'")
        bloch = _bloch.is_supercell(calculation)
        if bloch:
            labels = _bloch.primitive_shells(calculation)
        elif not hasattr(calculation, "_orbital_groups"):
            raise NotImplementedError("a cached crystal has no atom-to-orbital labels")
        else:
            labels = calculation._orbital_groups
        if integration == "tetrahedron":
            axis, _, values = self._tetrahedron(npoints, energies, kpts, projected=True)
        else:
            if bloch:
                raise NotImplementedError(
                    "the projected density of a supercell's unfolded bands is computed with "
                    "integration='tetrahedron'"
                )
            if kpts is not None:
                raise ValueError("a Gaussian PDOS uses the calculation's own mesh: omit kpts")
            axis, _ = self.dos(width, npoints, energies)
            values = (
                calculation.result.periodic.dos(
                    np.ascontiguousarray(axis / HARTREE_TO_EV),
                    width_ha=width / HARTREE_TO_EV,
                    projected=True,
                )
                / HARTREE_TO_EV
            )
        if values.shape[1] != len(labels):
            raise NotImplementedError(
                "the projection channels do not match the atom-to-orbital labels of this basis"
            )
        groups = {}
        for column, shell in enumerate(labels):
            groups[shell] = groups.get(shell, 0.0) + values[:, column]
        return axis, groups
