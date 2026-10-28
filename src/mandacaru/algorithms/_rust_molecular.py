"""ASE snapshots, option normalization and result views for native molecules."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np

from ._rust_quantum import _RustDriver, _integer
from .mean_field import MeanFieldResult
from ..units import ANGSTROM_TO_BOHR, HARTREE_TO_EV


def basis_specification(spec):
    from ._hamiltonian_from_atoms import resolve_basis, PER_ELEMENT
    from ..basis.gaussian_families import GaussianRecipe, parse_basis_name
    name, options = resolve_basis(spec)
    if name == PER_ELEMENT:
        from ase.data import atomic_numbers
        elements = {}
        for element, family in options.items():
            key = "*" if element == "*" else str(atomic_numbers[element] if isinstance(element, str) else int(element))
            elements[key] = basis_specification(family)
        return {"name": "per_element", "elements": elements}
    key = name.upper().replace(" ", "")
    if key in ("HAO", "FULLATOMICORBITALS"):
        if set(options) - {"virtual_orbitals"}:
            raise TypeError("native HAO options: virtual_orbitals")
        return {"name": "HAO", "virtual_orbitals": _integer(options.get("virtual_orbitals", 0), "virtual_orbitals")}
    if key in ("NAO-AE", "NAOAE"):
        from ..basis import nao_ae
        defaults = dict(tier=nao_ae.DEFAULT_TIER, onset=nao_ae.DEFAULT_ONSET,
                        width=nao_ae.DEFAULT_WIDTH, scale=nao_ae.DEFAULT_SCALE,
                        extra=None, linear_dependence_tol=nao_ae.DEFAULT_LINEAR_DEPENDENCE_TOL,
                        tail_norm=nao_ae.DEFAULT_TAIL_NORM, points=nao_ae.DEFAULT_POINTS,
                        r_max=nao_ae.DEFAULT_R_MAX)
        if set(options) - set(defaults):
            raise TypeError(f"native NAO-AE options: {', '.join(defaults)}")
        spec = dict(defaults, **options)
        spec["tier"] = _integer(spec["tier"], "tier")
        spec["points"] = _integer(spec["points"], "points", 4)
        spec["wall_radius"] = (float(spec["onset"]) + float(spec["width"])) * ANGSTROM_TO_BOHR
        spec["onset"] = float(spec["onset"]) * ANGSTROM_TO_BOHR
        spec["width"] = float(spec["width"]) * ANGSTROM_TO_BOHR
        if spec["extra"] is not None:
            spec["extra"] = [(_integer(n, "extra n", 1), _integer(l, "extra l"), float(z))
                             for n, l, z in spec["extra"]]
        return dict(name="NAO-AE", **spec)
    if key == "NAO":
        from ..basis.multizeta import resolve_zeta, resolve_split_scheme, DEFAULT_NAO_SIZE, DEFAULT_SPLIT_NORM
        from ..basis.nao import energy_shift_to_rc, DEFAULT_ENERGY_SHIFT
        if set(options) - {"energy_shift", "r_c", "n_grid", "size", "zeta_split"}:
            raise TypeError("native NAO options: energy_shift, r_c, n_grid, size, zeta_split")
        n_zeta, n_polarization = resolve_zeta(options.get("size", DEFAULT_NAO_SIZE))
        split, tails = resolve_split_scheme(options.get("zeta_split"))
        radius = options.get("r_c")
        radius = energy_shift_to_rc(options.get("energy_shift", DEFAULT_ENERGY_SHIFT)) if radius is None else float(radius)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("NAO r_c must be finite and positive")
        return {"name": "NAO", "r_c": radius, "n_grid": _integer(options.get("n_grid", 2000), "n_grid", 4),
                "n_zeta": n_zeta, "n_polarization": n_polarization,
                "split_norm": DEFAULT_SPLIT_NORM if split is None else split,
                "tail_norms": tails}
    if key == "GTO":
        if set(options) - {"n_gaussians"}:
            raise TypeError("native GTO options: n_gaussians")
        n = _integer(options.get("n_gaussians", 3), "n_gaussians", 1)
        if n > 10:
            raise ValueError("native Gaussian fits contain at most 10 primitives")
        recipe = GaussianRecipe(name=f"STO-{n}G", family="sto", core=(n,), valence=(n,))
    else:
        if options:
            raise TypeError("native named Gaussian recipes take no extra basis options")
        aliases = {"631G(D)": "6-31G(d)", "631G*": "6-31G*", "6-31GD": "6-31G(d)"}
        try:
            recipe = parse_basis_name(aliases.get(key, name))
        except ValueError as exc:
            raise NotImplementedError("the native molecular engine supports HAO and Mandacaru's Gaussian recipes") from exc
    return {"name": "gaussian", "recipe": asdict(recipe)}


def configure_geometry(driver, atoms):
    from .. import _native
    from .._native_adapter import geometry_from_atoms
    from ._hamiltonian_from_atoms import coherent_positions

    if getattr(driver, "_dft", False):
        if driver._is_periodic(atoms):
            from ._rust_periodic import configure
            return configure(driver, atoms)
        driver._periodic = False
        if driver._load_crystal is not None or driver._pseudo_setup:
            raise NotImplementedError("native molecular PAW is pending; a cached crystal requires run()")
        if driver._kpts_explicit or driver._smearing_explicit or driver._save_crystal or driver._periodic_options_explicit:
            raise ValueError("kpts, smearing and save_crystal require a periodic geometry")
    start = perf_counter()
    # Unwrapping is an ASE integration operation; numerical work begins with
    # an owned snapshot in Bohr and no retained ASE object in the engine.
    work = atoms.copy()
    work.positions = coherent_positions(atoms)
    geometry = geometry_from_atoms(work)
    driver.context = _native.Context(**driver.options)
    grid = driver._explicit_grid
    if grid is not None and not isinstance(grid, _native.SpatialGrid):
        grid = _native.SpatialGrid(
            grid.shape, [float(grid.X.flat[0]), float(grid.Y.flat[0]), float(grid.Z.flat[0])],
            np.ascontiguousarray(grid.step, dtype=float), periodic=bool(grid.periodic))
    driver.integrals = driver.context.molecular_integrals(
        geometry, basis_spec=json.dumps(driver._basis_spec),
        spacing_bohr=driver._grid_spacing * ANGSTROM_TO_BOHR, grid=grid,
        kinetic=driver._kinetic, cholesky_tolerance=driver._cholesky_tolerance)
    unresolved = [i for i, ratio in enumerate(driver.integrals.resolution_ratios) if abs(ratio - 1.) > .25]
    if unresolved:
        warnings.warn(f"the real-space grid does not resolve {len(unresolved)} native basis function(s) to 25% kinetic accuracy (indices {unresolved}); refine grid_spacing before interpreting energies", RuntimeWarning, stacklevel=3)
    particles = driver.integrals.particles(
        charge=driver._charge, n_electrons=driver._n_electrons,
        magnetic_moment=float(np.sum(atoms.get_initial_magnetic_moments())))
    kind = getattr(driver, "kind", "rhf" if particles[0] == particles[1] else "uhf")
    if getattr(driver, "_dft", False):
        driver.kind = kind = "uhf" if driver._unrestricted or particles[0] != particles[1] else "rhf"
        driver.native_reference = driver.integrals.kohn_sham(
            driver.xc, particles, unrestricted=kind == "uhf", **driver._scf_options)
    else:
        driver.native_reference = driver.integrals.mean_field(kind, particles, **driver._scf_options)
    if hasattr(driver, "kind"):
        driver.reference_snapshot = driver.native_reference.snapshot()
    driver.native_model = (driver.native_reference.model if driver._active_spec is None
                           else driver.native_reference.active_model(json.dumps(driver._active_spec)))
    driver.num_particles = tuple(driver.native_model.num_particles)
    driver.n_spatial_orbitals = driver.native_model.n_spatial_orbitals
    driver.mapping = driver._requested_mapping or "jordan_wigner"
    driver.n_qubits = 2 * driver.n_spatial_orbitals - (2 if driver.mapping == "parity_reduced" else 0)
    if driver._save_hamiltonian:
        path = ("hamiltonian.json" if driver._save_hamiltonian is True
                else str(driver._save_hamiltonian))
        if not Path(path).suffix:
            path += ".json"
        driver.native_model.save(path, mapping=driver.mapping)
    driver._geometry_seconds = perf_counter() - start
    if not hasattr(driver, "kind"):
        if not driver.native_reference.converged:
            raise RuntimeError("the native mean-field reference did not converge; increase scf_options['max_iterations'] before a quantum solve")
        driver._configure()


class NativeMeanFieldResult(MeanFieldResult):
    """Existing result fields with native mapping and state preparation."""

    def __repr__(self):
        return f"MeanFieldResult(method={self.method!r}, energy={self.optimal_energy:.9f} {self.energy_unit}, success={self.success})"

    def qubit_hamiltonian(self, mapping="jordan_wigner"):
        from ..core.mapping import PauliSum
        operator = self._native_model.operator(self._check_mapping(mapping))
        return PauliSum({p: complex(re, im) for p, re, im in operator.terms}, num_qubits=operator.num_qubits)

    def reference_state(self, mapping="jordan_wigner"):
        return self._native_model.reference_state(self._check_mapping(mapping))

    def as_quantum_problem(self):
        return {"engine": "rust", "native_model": self._native_model}

    def mp2_natural_orbitals(self):
        """Closed-shell MP2/FNO analysis in Hartree, computed from native integrals."""
        if self.method != "rhf" or not self.success:
            raise ValueError("MP2 requires a converged closed-shell Hartree–Fock reference")
        from .mp2 import MP2Result
        return MP2Result(**self._native_model.mp2())

    def scf_state(self, mapping="jordan_wigner"):
        if self._unrestricted:
            if self._native_model.n_spatial_orbitals != self._native_reference.model.n_spatial_orbitals:
                raise NotImplementedError("the native SCF determinant with an active space is not yet wired")
            return self._native_reference.state(mapping)
        return self.reference_state(mapping)


class _RustMeanField(_RustDriver):
    classical_mean_field = True
    analytic_gradient = True

    def __init__(self, **kwargs):
        if kwargs.get("load_hamiltonian") is not None or kwargs.get("native_model") is not None:
            raise ValueError("native mean-field calculations require a geometry and spatial integrals")
        if kwargs.get("optimizer") is not None or "sector" in kwargs or "sparse" in kwargs:
            raise TypeError("native mean-field calculations use scf_options and two_body; optimizer, sector and sparse select quantum solves")
        self.include_singles, self.trotter = True, False
        self.ansatz = None
        super().__init__(**kwargs)
        if getattr(self, "kind", None) == "ghf" and self._requested_mapping == "parity_reduced":
            raise ValueError("the GHF spinor Hamiltonian has no fixed alpha parity; use jordan_wigner, parity or bravyi_kitaev")

    @property
    def hamiltonian(self):
        if getattr(self, "_periodic", False):
            return None
        return None if self.result is None else self.result.fermion_hamiltonian

    def run(self, geometry=None, cell=None):
        if getattr(self, "_dft", False) and (self._periodic or self._load_crystal is not None or self._is_periodic(geometry)):
            from ._rust_periodic import run
            return run(self, geometry)
        from .hartree_fock import RHFResult, UHFResult, GHFResult
        if getattr(self, "_dft", False):
            from .dft import KohnShamResult as RHFResult, KohnShamUResult as UHFResult
        if self.dry_run:
            raise NotImplementedError("the native molecular engine does not yet provide dry-run estimates")
        if geometry is not None:
            if cell is not None:
                geometry = geometry.copy()
                geometry.set_cell(cell)
            configure_geometry(self, geometry)
        if not hasattr(self, "reference_snapshot"):
            raise ValueError("mean-field run() requires geometry=Atoms or an attached ASE calculator")
        raw = self.reference_snapshot
        common = dict(electronic_energy=raw["electronic_energy"], converged=raw["converged"],
                      n_iterations=raw["n_iterations"], h_mo=raw["h_mo"], eri_mo=raw["eri_mo"],
                      history=[dict(iter=i, energy=e, dE=de, residual=r) for i, e, de, r in raw["history"]])
        if raw["functional"] is not None:
            common.update(functional=raw["functional"], hartree_energy=raw["hartree_energy"],
                          xc_energy=raw["xc_energy"])
            if self.kind == "rhf":
                common["reference_energy"] = raw["reference_energy"]
        na, nb = raw["num_particles"]
        if self.kind == "rhf":
            scf = RHFResult(**common, mo_energies=np.asarray(raw["energies"][0]),
                            mo_coefficients=raw["coefficients"][0], n_occupied=na)
        elif self.kind == "uhf":
            scf = UHFResult(**common, n_alpha=na, n_beta=nb,
                            mo_energies_alpha=np.asarray(raw["energies"][0]), mo_energies_beta=np.asarray(raw["energies"][1]),
                            mo_coefficients_alpha=raw["coefficients"][0], mo_coefficients_beta=raw["coefficients"][1],
                            natural_occupations=np.asarray(raw["natural_occupations"]), natural_orbitals=raw["model_orbitals"],
                            reference_energy=raw["reference_energy"])
        else:
            scf = GHFResult(**common, n_electrons=na + nb, mo_energies=np.asarray(raw["energies"][0]),
                            mo_coefficients=raw["coefficients"][0], mode_coefficients=raw["model_orbitals"],
                            kramers_pairing=raw["kramers_pairing"])
        self.fermion_hamiltonian = _HamiltonianView(self.native_model)
        self._scf = scf
        self.result = NativeMeanFieldResult(
            method="dft" if getattr(self, "_dft", False) else self.kind, optimal_energy=(raw["electronic_energy"] + raw["nuclear_repulsion"]) * self._scale,
            reference_energy=(raw["reference_energy"] + raw["nuclear_repulsion"]) * self._scale,
            scf=scf, model_orbitals=raw["model_orbitals"][:, self.native_model.orbital_indices], fermion_hamiltonian=self.fermion_hamiltonian,
            num_particles=self.num_particles, n_spatial_orbitals=self.n_spatial_orbitals, energy_unit=self.energy_unit,
            timings={"engine": "rust", "geometry_seconds": self._geometry_seconds,
                     "memory_peak_bytes": self.context.memory_peak_bytes, "threads": self.context.threads})
        self.result._unrestricted = self.kind == "uhf"
        self.result._native_model = self.native_model
        self.result._native_reference = self.native_reference
        report = f"Rust {self.kind.upper()}: {self.result.optimal_energy:.12f} {self.energy_unit}; converged={self.result.success}\n"
        if self.verbose:
            print(report, end="")
        if self.txt is not None:
            Path(self.txt).write_text(report)
        return self.result

    def _from_energy_units(self, value, units):
        from ..units import convert_energy
        return float(convert_energy(value, self.energy_unit, units))

    def check_force_support(self):
        if getattr(self, "_periodic", False):
            raise NotImplementedError("native periodic SCF forces/stress need PAW setup derivatives; only the Ewald contribution is native in this milestone")
        if self.kind == "ghf" or self._kinetic != "spectral" or self._active_spec is not None:
            raise NotImplementedError("native stationary forces require full-space RHF/UHF or collinear molecular DFT and spectral kinetic energy")
        if self._cholesky_tolerance is not None and not getattr(self, "_dft", False):
            raise NotImplementedError("native HF forces require packed integrals; Cholesky derivative forces are not implemented")

    def nuclear_gradient(self):
        self.check_force_support()
        from .forces import ForceResult, HA_BOHR_TO_EV_ANGSTROM
        raw = self.native_reference.gradient()
        scale = HA_BOHR_TO_EV_ANGSTROM
        gradient = raw["gradient"] * scale
        return ForceResult(forces=-gradient, gradient=gradient,
                           hellmann_feynman=raw["hellmann_feynman"] * scale,
                           pulay=raw["pulay"] * scale, n_electrons=sum(self.num_particles),
                           details={"engine": "rust", "stationary_scf": True,
                                    "functional": getattr(self, "xc", None)})

    def mean_field_rdm(self):
        if self._active_spec is not None:
            raise NotImplementedError("native mean-field RDM export with an active space is not wired")
        return self.native_reference.rdm()


class RustRHF(_RustMeanField):
    kind = "rhf"


class RustUHF(_RustMeanField):
    kind = "uhf"


class RustGHF(_RustMeanField):
    kind = "ghf"
    analytic_gradient = False


class RustDFT(_RustMeanField):
    _dft = True

    @staticmethod
    def _is_periodic(atoms):
        return atoms is not None and bool(np.any(atoms.pbc))

    def __init__(self, *, xc=None, spin=False, unrestricted=False, kpts=None,
                 smearing=None, periodic_scf_options=None, symmetry=True,
                 load_crystal=None, save_crystal=None, **kwargs):
        from ..integrals.exchange_correlation import resolve_functional
        from .periodic_dft import resolve_smearing
        from ._hamiltonian_from_atoms import resolve_basis, pseudopotential_family
        self._explicit_xc = xc is not None
        self.xc = resolve_functional("lda" if xc is None else xc)
        if self.xc not in ("lda", "pbe", "r2scan", "hse06", "pbe+rvv10", "r2scan+rvv10"):
            raise NotImplementedError("unknown canonical molecular XC functional")
        if type(spin) is not bool or type(unrestricted) is not bool or type(symmetry) is not bool:
            raise ValueError("native spin, unrestricted and symmetry must be booleans")
        self._unrestricted = spin or unrestricted
        self._periodic = False
        self._periodic_symmetry = symmetry
        self._kpts_explicit, self._smearing_explicit = kpts is not None, smearing is not None
        if kpts is None:
            kpts = {"size": (1, 1, 1), "gamma": True}
        from ._hamiltonian_from_atoms import monkhorst_pack_kpts
        size, gamma, _mesh = monkhorst_pack_kpts(kpts)
        self._kpts_spec = {"size": size, "gamma": gamma}
        method, width = resolve_smearing(smearing)
        self._periodic_options_explicit = bool(periodic_scf_options)
        self._periodic_two_body_requested = kwargs.get("two_body") is not None
        self._periodic_options = dict(periodic_scf_options or {})
        if set(self._periodic_options) - {"max_iterations", "tolerance", "density_tolerance", "mixing", "history"}:
            raise ValueError("periodic_scf_options: max_iterations, tolerance, density_tolerance, mixing, history")
        for name in ("max_iterations", "history"):
            if name in self._periodic_options:
                self._periodic_options[name] = _integer(self._periodic_options[name], name, 1)
        self._periodic_options.update(smearing=method, width_ha=width)
        if isinstance(save_crystal, bool) or isinstance(load_crystal, bool):
            raise ValueError("load_crystal and save_crystal require explicit file paths")
        self._load_crystal, self._save_crystal = load_crystal, save_crystal
        self._periodic_basis = kwargs.get("basis", "HAO")
        name, _ = resolve_basis(self._periodic_basis)
        family = pseudopotential_family(name)
        self._pseudo_setup = family is not None
        if self._pseudo_setup:
            kwargs["basis"] = "HAO"  # only the geometry-aware adapter consumes the PAW specification
        if load_crystal is not None:
            if save_crystal or self._kpts_explicit or kwargs.get("active_space") is not None or kwargs.get("grid") is not None or kwargs.get("charge", 0) or kwargs.get("n_electrons") is not None or kwargs.get("scf_options") or spin or unrestricted or not symmetry or kwargs.get("two_body") is not None or kwargs.get("save_hamiltonian") or kwargs.get("kinetic") not in (None, "spectral"):
                raise ValueError("geometry, mesh and molecular SCF options do not apply to a cached crystal")
        super().__init__(spin=spin, **kwargs)

    def crystal_stress(self, *args, **kwargs):
        raise NotImplementedError("native periodic stress requires PAW strain derivatives; Context.ewald provides the ionic Ewald contribution")

    def _periodic_result(self):
        if not getattr(self, "_periodic", False) or self.result is None:
            raise RuntimeError("a converged native periodic DFT calculation is required")
        return self._periodic_snapshot

    def get_fermi_level(self):
        return float(self._periodic_result()["fermi_level"] * HARTREE_TO_EV)

    def get_number_of_spins(self):
        return int(self._periodic_result()["n_spins"])

    def get_eigenvalues(self, kpt=0, spin=0):
        return self._periodic_result()["eigenvalues"][spin, kpt].copy() * HARTREE_TO_EV

    def get_ibz_k_points(self):
        return self._periodic_result()["fractional_kpoints"].copy()

    def get_k_point_weights(self):
        return np.array(self._periodic_result()["weights"], dtype=float)

    def get_number_of_bands(self):
        return self._periodic_result()["eigenvalues"].shape[-1]

    def get_bz_k_points(self):
        if self._load_crystal is not None:
            raise NotImplementedError("cached crystal getters expose its stored irreducible mesh")
        return self.context.kmesh(self._kpts_spec["size"], gamma=self._kpts_spec["gamma"], time_reversal=False)["full"]

    def band_structure(self, path=None, npoints=100):
        from ase.spectrum.band_structure import BandStructure
        from ._rust_periodic import kpoint_payload
        self._periodic_result()
        if not hasattr(self, "_crystal_adapter"):
            raise NotImplementedError("off-mesh bands need the PAW setup adapter; a cached crystal contains its stored mesh")
        bandpath = self._periodic_atoms.cell.bandpath(path=path, npoints=npoints, pbc=self._periodic_atoms.pbc)
        c = self._crystal_adapter
        data = c.kpoint_matrices(c.cartesian_kpoints(bandpath.kpts))
        native = self.native_periodic_result.bands(json.dumps([kpoint_payload(c, d, frac) for d, frac in zip(data, bandpath.kpts)]))
        return BandStructure(bandpath, energies=native["eigenvalues"]*HARTREE_TO_EV, reference=self.get_fermi_level())

    def dos(self, width=.1, npoints=2001, kpts=None, energies=None, integration="gaussian", curvature_correction=None):
        self._periodic_result()
        if kpts is not None or integration != "gaussian" or curvature_correction is not None:
            raise NotImplementedError("native DOS currently broadens the stored SCF mesh with Gaussians")
        levels = self._periodic_snapshot["eigenvalues"] * HARTREE_TO_EV
        axis = np.linspace(levels.min()-8*width, levels.max()+8*width, _integer(npoints, "npoints", 2)) if energies is None else np.ascontiguousarray(energies, dtype=float)
        values = self.native_periodic_result.dos(np.ascontiguousarray(axis/HARTREE_TO_EV), width_ha=width/HARTREE_TO_EV)
        return axis, values[:, 0]/HARTREE_TO_EV

    def pdos(self, width=.1, npoints=2001, kpts=None, energies=None, integration="gaussian", curvature_correction=None):
        axis, _ = self.dos(width, npoints, kpts, energies, integration, curvature_correction)
        if not hasattr(self, "_orbital_groups"):
            raise NotImplementedError("the cached snapshot has no public orbital-to-atom labels")
        values = self.native_periodic_result.dos(np.ascontiguousarray(axis/HARTREE_TO_EV), width_ha=width/HARTREE_TO_EV, projected=True)/HARTREE_TO_EV
        groups = {}
        for column, shell in enumerate(self._orbital_groups):
            groups[shell] = groups.get(shell, 0.) + values[:, column]
        return axis, groups



class _HamiltonianView:
    """Materialize a Python operator only when its public result field is read."""

    def __init__(self, model):
        self.model = model

    def build(self):
        from ..core.mapping import Fermion
        return Fermion({tuple(l): complex(re, im) for l, re, im in self.model.fermion_terms()},
                       n_modes=2 * self.model.n_spatial_orbitals)
