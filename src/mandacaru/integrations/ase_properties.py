"""ASE electronic-structure views over a calculator's native state."""

import numpy as np
from .. import _native
from . import bloch as _bloch
from . import kspace as _kspace


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

    def get_berry_phase(self, direction, kpts=None, bands=None):
        """Berry phases of closed k-point strings along ``G_direction``
        (:class:`~mandacaru.analysis.berry.BerryPhase`).

        ``direction`` is the index 0, 1 or 2 of a reciprocal lattice vector. ``kpts`` is the
        Gamma-anchored mesh ``(n1, n2, n3)`` or ``{"size": ...}`` (default: the calculation's
        ``kpts`` size): its entry along ``direction`` is the string length (at least 2), the
        other two the transverse mesh. ``bands`` is ``None`` (the occupied bands of an insulator,
        per spin channel), a band index, or a half-open ``(lo, hi)`` group isolated from its
        neighbors. The view holds the phases, the Wilson-loop Wannier charge centers and the
        Berry connection in the parallel-transport gauge. A DFT crystal only: a molecule raises
        ``ValueError``, a quantum method on a crystal ``NotImplementedError``.
        """
        from .berry import berry_phase

        return berry_phase(self._calculation_owner(), direction, kpts=kpts, bands=bands)

    def get_berry_curvature(self, plane, kpts=None, offset=0.0, bands=None):
        """The Berry curvature and Chern number of a k plane
        (:class:`~mandacaru.analysis.berry.BerryCurvature`).

        ``plane`` is the index 0, 1 or 2 of the reciprocal vector ``G_c`` normal to the plane,
        which the other two span in cyclic order. ``kpts`` is as in :meth:`get_berry_phase`; its
        two in-plane entries (each at least 2) are the plane's mesh. ``offset`` is the plane's
        fractional coordinate along ``G_c``, in [0, 1). ``bands`` is as in
        :meth:`get_berry_phase`: one isolated band gives the band-resolved curvature. The
        curvature and the Chern number are oriented along ``G_a x G_b`` and flip sign when the
        cell is relabeled into the other handedness.
        """
        from .berry import berry_curvature

        return berry_curvature(self._calculation_owner(), plane, kpts=kpts, offset=offset,
                               bands=bands)

    def get_chern_number(self, plane, kpts=None, offset=0.0, bands=None):
        """The Chern number of the k plane normal to ``G_plane``, summed over spin channels
        (:meth:`get_berry_curvature` holds the per-channel numbers). It is oriented along
        ``G_a x G_b`` (``a, b`` the in-plane directions in cyclic order), so it changes sign when
        the cell is relabeled into the other handedness; the Hall conductivity does not."""
        return self.get_berry_curvature(plane, kpts=kpts, offset=offset,
                                        bands=bands).chern_number

    def get_polarization(self, kpts=None):
        """The Berry-phase electric polarization of an insulating crystal
        (:class:`~mandacaru.analysis.berry.Polarization`).

        Strings along the three reciprocal vectors are cut from the mesh ``kpts`` (default: the
        calculation's ``kpts`` size; at least 2 points per direction) and diagonalized at the
        converged Hamiltonian. The view holds the vector in C/m^2 on the branch nearest zero,
        its quanta, the raw coordinates for differences along a path, and the electronic and
        ionic parts. A metal or a fractionally occupied state raises ``ValueError``. String phases
        that spread by more than pi / 2 over the transverse mesh return the result with a
        warning; below that, compare two transverse meshes to judge the sampling.
        """
        from .berry import polarization

        return polarization(self._calculation_owner(), kpts=kpts)

    def get_anomalous_hall(self, kpts=None):
        """The intrinsic anomalous Hall conductivity with its provenance
        (:class:`~mandacaru.analysis.berry.AnomalousHall`).

        An insulator gives the quantized conductivity of its Chern numbers; a metal the Fermi-sea
        term of the converged smearing from the occupation-weighted lattice field, with the
        stride-2 sub-mesh value ``sigma_coarse`` as its convergence estimate, at the Fermi level
        re-solved on that mesh. Warnings name a missing or poor estimate and a mesh where most
        plaquettes merge band groups: a metal's value converges slowly with ``kpts``. ``kpts`` is
        the mesh (at least 2 per direction; even and at least 8 for the estimate).
        """
        from .berry import anomalous_hall

        return anomalous_hall(self._calculation_owner(), kpts=kpts, stacklevel=3)

    def get_anomalous_hall_conductivity(self, kpts=None):
        """``(3, 3)``: the anomalous Hall conductivity ``sigma_ab`` in S/cm,
        ``(e^2/h)(1/2 pi) eps_abd K_d`` with ``K = sign(det G) sum_c nu_c G_c``: for an insulator
        ``nu_c`` is the Chern number of the plane family times the electrons per band, for a metal
        the intrinsic Fermi-sea term (:meth:`get_anomalous_hall` holds both, the convergence
        estimate and the per-layer numbers).
        """
        from .berry import anomalous_hall

        return anomalous_hall(self._calculation_owner(), kpts=kpts, stacklevel=3).sigma

    def get_excitations(self, num_states=5, *, approximation="tda", spin=None,
                        kernel="adiabatic", tolerance=1e-5, max_iterations=200):
        """The lowest ``num_states`` linear-response excitations of a converged molecule
        (:class:`~mandacaru.analysis.excitations.Excitations`).

        Time-dependent DFT on a ``method="dft"`` state (``xc="lda"``, ``"pbe"`` or ``"hse06"``:
        the adiabatic kernel of the functional, with the short-range exact exchange of the
        screened hybrid in both A and B) and TDHF/CIS on an ``rhf`` or ``uhf`` state.
        ``approximation`` is ``"tda"`` (Tamm-Dancoff) or ``"casida"`` (the full equations).
        A closed shell is spin-adapted: ``spin`` is ``"singlet"`` (the default) or
        ``"triplet"``. An unrestricted reference (an open shell, or ``spin=True`` /
        ``unrestricted=True``) gives the spin-conserving excitations of both spins, with each
        root's ``<S^2>``; ``spin`` must be left unset there. ``kernel`` is ``"adiabatic"`` (the
        reference's whole kernel), ``"hartree"`` (the Coulomb kernel only) or
        ``"independent"`` (bare orbital-energy differences). ``tolerance`` bounds each root's
        residual (atomic units). A request that would cut a degenerate multiplet returns the
        whole multiplet. On a PAW-LCAO basis the kernel is the exact second derivative of the
        engine's energy, whose one-center terms are linearized; the view reports, per root,
        the first-order shift the missing one-center Hartree kernel would make. The view holds
        the length-form strengths and the velocity form (with the PAW one-center part of the
        gradient), which agree only as the basis is completed. Meta-GGAs, rVV10, DFT+U,
        spin-orbit datasets, fractional occupations, crystals and ``precision="float32"``
        raise. Each call appends an ``[EXCITED STATES]`` block to the
        run log.
        """
        from .excitations import excitations

        return excitations(self._calculation_owner(), num_states, approximation=approximation,
                           spin=spin, kernel=kernel, tolerance=tolerance,
                           max_iterations=max_iterations)

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
            return levels, _native.uniform_weights(len(points))
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
            return _native.to_user_units(float(self._periodic_snapshot()["fermi_level"]), "energy")
        if _bloch.is_supercell(calculation):
            # The highest occupied level of the supercell mean field.
            levels = np.asarray(calculation.native_reference.snapshot()["energies"])[:, None, :]
        else:
            levels, _ = self._spectrum()
        particles, _ = self._molecular_occupations(levels)
        highest = _native.highest_occupied(
            np.ascontiguousarray(np.asarray(levels)[:, 0, :], dtype=float),
            [int(p) for p in particles])
        return _native.to_user_units(highest, "energy")

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
        return _native.to_user_units(
            np.ascontiguousarray(self._spectrum()[0][spin, kpt], dtype=float), "energy")

    def get_occupation_numbers(self, kpt=0, spin=0):
        calculation = self._calculation_owner()
        if _bloch.is_supercell(calculation):
            raise NotImplementedError(
                "the unfolded mean-field bands of a supercell carry no per-k occupations; "
                "the correlated occupations are in get_spectral_function()"
            )
        if calculation.result is not None and calculation.result.periodic is not None:
            return self._periodic_snapshot()["occupations"][spin, kpt].copy()
        self._spectrum()  # a cached quantum Hamiltonian has no orbital spectrum: raises
        # Rust's occupations: aufbau, or the ensemble of a degenerate frontier shell.
        return np.array(calculation.native_reference.snapshot()["occupation_numbers"][spin])

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

    def get_spectral_function(self, energies=None, eta=0.1, kpoints=None, krylov=200, *,
                              path=None, irreducible=False):
        """The Lehmann spectral function ``A(k, omega)`` of the correlated state.

        Available after a quantum method on a crystal (``vqe``, ``adapt-vqe``, ``vasqa``,
        ``valsqa`` on a periodic structure), without shots, tapering or an active space. ``energies`` is the axis in eV (by default
        2001 points spanning the poles), ``eta`` the Lorentzian half-width in eV, ``kpoints``
        a subset of the mesh points (fractional; the default is all of them) and ``krylov`` the
        Lanczos vectors per part. See :class:`~mandacaru.analysis.bloch.SpectralFunction`.

        ``path`` resolves it along a band path instead: a string of the lattice's high-symmetry
        labels (``"GXMG"``; a comma jumps; see :meth:`band_path`), a sequence of fractional
        points, or ``None`` with ``kpoints`` for the mesh. The result holds the mesh points on
        the path in path order and ``.kpath``. ``irreducible=True`` resolves only the
        irreducible representative of every star of the path's points and audits the symmetry
        of the state against one more point (``.audit``), warning when it is broken.
        """
        return _bloch.spectral_function(
            self._calculation_owner(), energies=energies, eta=eta, kpoints=kpoints,
            krylov=krylov, path=path, irreducible=irreducible,
        )

    def symmetry(self, symprec=1e-5):
        """The space group of the structure (:class:`~mandacaru.analysis.kspace.CrystalSymmetry`).

        A cell with a non-periodic axis is read as the crystal that repeats along it.
        """
        return _kspace.symmetry(self._calculation_owner(), symprec)

    def irreducible_zone(self, symprec=1e-5):
        """The ``kpts`` mesh reduced by the crystal's point group and time reversal.

        See :class:`~mandacaru.analysis.kspace.IrreducibleZone`: the irreducible points, their
        weights (adding up to 1) and the star of every mesh point. The group is the subgroup of
        the space group's rotations that map the mesh onto itself.
        """
        return _kspace.irreducible_zone(self._calculation_owner(), symprec)

    def band_path(self, path=None):
        """The mesh points of ``kpts`` on a band path (:class:`~mandacaru.analysis.kspace.KPath`).

        ``path`` is a string of high-symmetry labels, a sequence of fractional points (each
        optionally a ``(label, point)`` pair) or ``None`` for the default path of the lattice.
        The points of the named lattices are listed in ``periodic::kpath::convention``; a lattice
        without named points raises and asks for explicit points.
        """
        return _kspace.band_path(self._calculation_owner(), path)

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
            energies=_native.to_user_units(
                np.ascontiguousarray(self._bands_at(bandpath.kpts), dtype=float), "energy"),
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
            weights = _native.uniform_weights(len(fractional))
        axis = self._energy_axis(levels, npoints, energies, width, 8.0)
        degeneracy = self._degeneracy(levels)
        values = calculation.context.density_of_states(
            np.ascontiguousarray(levels, dtype=np.float64),
            np.ascontiguousarray(weights, dtype=np.float64),
            _native.energy_to_hartree(np.ascontiguousarray(axis, dtype=float), "eV"),
            width_ha=_native.energy_to_hartree(float(width), "eV"),
            degeneracy=degeneracy,
        )
        # States per Hartree to states per eV: the same division by the factor.
        return axis, _native.energy_to_hartree(np.ascontiguousarray(values, dtype=float), "eV")

    @staticmethod
    def _energy_axis(levels, npoints, energies, margin, scale=1.0):
        if energies is not None:
            return np.asarray(energies, dtype=float)
        return _native.energy_axis(np.ascontiguousarray(levels, dtype=float), int(npoints),
                                   float(margin), scale=float(scale))

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
            _native.energy_to_hartree(np.ascontiguousarray(axis, dtype=float), "eV"),
            projections=np.ascontiguousarray(weights, dtype=np.float64) if projected else None,
            degeneracy=self._degeneracy(levels),
        )
        per_ev = lambda a: _native.energy_to_hartree(np.ascontiguousarray(a, dtype=float), "eV")  # noqa: E731
        values = per_ev(raw["dos"])
        projection = per_ev(raw["projected"]) if projected else None
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
                _native.energy_to_hartree(np.ascontiguousarray(calculation.result.periodic.dos(
                    _native.energy_to_hartree(np.ascontiguousarray(axis, dtype=float), "eV"),
                    width_ha=_native.energy_to_hartree(float(width), "eV"),
                    projected=True,
                ), dtype=float), "eV")
            )
        if values.shape[1] != len(labels):
            raise NotImplementedError(
                "the projection channels do not match the atom-to-orbital labels of this basis"
            )
        names = {repr(shell): shell for shell in labels}
        keys, sums = _native.column_groups(
            np.ascontiguousarray(values, dtype=float), [repr(shell) for shell in labels])
        return axis, {names[key]: total for key, total in zip(keys, sums)}
