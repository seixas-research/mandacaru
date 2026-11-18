"""ASE geometry and unit views for native Wannier calculations."""
from __future__ import annotations

import json
import warnings

import numpy as np
from .. import _native
from ..units import from_hartree, to_angstrom, to_bohr, to_hartree


def _atoms_for_wannier(calculation):
    if calculation.atoms is not None:
        return calculation.atoms
    if calculation._load_crystal is None:
        raise RuntimeError("periodic Wannier guesses by symbol/bonds require ASE Atoms")
    from pathlib import Path
    from ase import Atoms

    payload = json.loads(Path(calculation._load_crystal).read_text(encoding="utf-8"))
    grid = payload["grid"]
    sites = payload.get("sites")
    if not sites or "centers" not in sites or "datasets" not in sites:
        raise RuntimeError("loaded crystal snapshot lacks 'sites' geometry for Wannier guesses")
    shape = np.asarray(grid["shape"], dtype=float)
    step = np.asarray(grid["step"], dtype=float).reshape(3, 3)
    cell_ang = np.asarray(to_angstrom((step * shape[None, :]).T), dtype=float)
    positions_ang = np.asarray(to_angstrom(np.asarray(sites["centers"], dtype=float)), dtype=float)
    symbols = [entry.get("symbol") or Path(entry["file"]).stem for entry in sites["datasets"]]
    return Atoms(symbols=symbols, positions=positions_ang, cell=cell_ang, pbc=True)


def _trials(guess, calculation):
    if isinstance(guess, str) and guess in ("bonds", "sp3"):
        atoms = _atoms_for_wannier(calculation)
        centers = to_bohr(atoms.positions)
        # The neighbors within 6 Angstrom of every atom, periodic images included, are the engine's.
        neighbors = _native.neighbor_vectors(
            np.ascontiguousarray(atoms.cell.array, dtype=float), [bool(p) for p in atoms.pbc],
            np.ascontiguousarray(atoms.positions, dtype=float), 6.0)
        key = "neighbors" if guess == "sp3" else "bond_neighbors"
        return [
            dict(center=center.tolist(), label=f"{atom.symbol}{i}",
                 **{key: to_bohr(neighbors[i]).tolist()})
            for i, (atom, center) in enumerate(zip(atoms, centers))
        ]
    if isinstance(guess, dict):
        atoms = _atoms_for_wannier(calculation)
        centers = to_bohr(atoms.positions)
        records = []
        for selector, names in guess.items():
            names = [names] if isinstance(names, str) else list(names)
            sites = (
                [i for i, atom in enumerate(atoms) if atom.symbol == selector]
                if isinstance(selector, str) else [int(selector)]
            )
            if not sites:
                raise ValueError(f"the Wannier guess names no atoms for {selector!r}")
            for site in sites:
                for name in names:
                    records.append(dict(center=centers[site].tolist(), orbital=name,
                                        label=f"{atoms[site].symbol}{site}"))
        return records
    centers = np.asarray(guess, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("a Wannier guess names bonds, sp3, atomic orbitals or centers of shape (n,3)")
    return [dict(center=to_bohr(center).tolist(), orbital="s",
                 label=f"center{i}") for i, center in enumerate(centers)]


def _mesh_size(kpts, fallback):
    if kpts is None:
        if fallback is None:
            raise ValueError("give kpts when constructing Wannier functions from a loaded snapshot")
        mesh = fallback
    else:
        mesh = kpts
    size = mesh["size"] if isinstance(mesh, dict) else mesh
    return [int(n) for n in size]


def calculate(calculation, n_functions, *, guess, windows, bands, kpts,
              max_iterations, tolerance, step, localization="spread",
              populations="mulliken", exponent=2):
    if calculation.result is None or calculation.result.periodic is None:
        raise RuntimeError("run periodic DFT before constructing Wannier functions")
    spec = dict(size=_mesh_size(kpts, calculation._kpts_spec), bands=bands,
                n_functions=n_functions, trials=_trials(guess, calculation),
                localization=str(localization), populations=str(populations),
                exponent=int(exponent))
    if windows is not None:
        if not isinstance(windows, dict) or "outer" not in windows:
            raise ValueError("Wannier windows specify an outer energy pair and an optional frozen pair in eV")
        spec["windows"] = {
            name: None if pair is None else [to_hartree(float(e), "eV") for e in pair]
            for name, pair in windows.items()
        }
    native = calculation.result.periodic.wannier(
        json.dumps(spec, allow_nan=False), max_iterations=max_iterations, tolerance=tolerance,
        step=step)
    return (WannierResult(native, 0) if native.n_spins == 1 else SpinWannierResult(native))


def _modes(functions, n_functions):
    """``[(n, (R1, R2, R3)), ...]`` of a fragment given as indices or ``(n, R)``; ``None`` is every
    function of the home cell."""
    if functions is None:
        return None
    modes = []
    for item in functions:
        if isinstance(item, (int, np.integer)):
            n, R = int(item), (0, 0, 0)
        else:
            n, R = item
            n, R = int(n), tuple(int(x) for x in R)
            if len(R) != 3:
                raise ValueError(f"a lattice vector has three integers; got {R!r}")
        if not 0 <= n < n_functions:
            raise IndexError(f"function {n} of {n_functions}")
        modes.append((n, R))
    return modes


class DownfoldedProblem:
    r"""The many-body problem of a fragment of Wannier functions (Hartree).

    ``kohn_sham`` is the Kohn-Sham Hamiltonian between the fragment's functions, ``two_body`` the
    Coulomb integrals :math:`\langle pq|rs\rangle` (physicists' notation), ``density_matrix`` the
    Kohn-Sham one-particle density matrix :math:`\gamma_{pq} = \langle c^\dagger_p c_q\rangle` and
    ``double_counting`` the Hartree and exchange potential of the fragment's own electrons, which
    ``kohn_sham`` already holds. The model's one-body part ``one_body`` is
    :math:`h^{KS} - V^{dc}`, so its Hartree-Fock Fock matrix at :math:`\gamma` is :math:`h^{KS}`
    again. A restricted crystal's arrays are spatial (``kohn_sham`` ``(n, n)``, ``two_body``
    ``(n, n, n, n)``, ``density_matrix`` summed over spin); a spin-polarized one has a channel
    axis (``(2, n, n)``) and ``two_body[s, t]`` is :math:`\langle p^s q^t|r^s s^t\rangle`.
    ``functions`` names each mode ``(n, R)`` and ``num_particles`` is ``(up, down)``.
    """

    def __init__(self, native):
        self._native = native
        for note in native.warnings:
            warnings.warn(note, stacklevel=4)

    def _channels(self, array):
        return array if self.spin_polarized else array[0]

    @property
    def spin_polarized(self) -> bool:
        return self._native.spins == 2

    @property
    def spinor(self) -> bool:
        return bool(self._native.spinor)

    @property
    def functions(self) -> tuple:
        return tuple((n, tuple(R)) for n, R in self._native.functions)

    @property
    def n_spatial_orbitals(self) -> int:
        n = int(self._native.n_spatial_orbitals)
        if self.spinor:
            if n % 2 != 0:
                raise ValueError(
                    f"spinor fragment has {n} spin-orbitals, which is not a whole number of spatial pairs"
                )
            return n // 2
        return n

    @property
    def num_particles(self) -> tuple:
        return tuple(self._native.num_particles)

    @property
    def n_electrons(self) -> int:
        return int(sum(self._native.num_particles))

    @property
    def screening(self):
        return self._native.screening

    @property
    def coulomb(self) -> str:
        return self._native.coulomb

    @property
    def dielectric_head(self):
        """The inverse dielectric head the screening took for q -> 0, or ``None``."""
        return self._native.dielectric_head

    @property
    def kohn_sham(self) -> np.ndarray:
        return self._channels(self._native.kohn_sham)

    @property
    def density_matrix(self) -> np.ndarray:
        return self._channels(self._native.density_matrix)

    @property
    def double_counting(self) -> np.ndarray:
        return self._channels(self._native.double_counting)

    @property
    def one_body(self) -> np.ndarray:
        return self._channels(self._native.one_body("wannier"))

    @property
    def two_body(self) -> np.ndarray:
        two = self._native.two_body("wannier")
        return two if self.spin_polarized else two[0, 0]

    def natural_orbitals(self):
        """``(occupations, C)``: the eigenvectors of :attr:`density_matrix` (columns, in the
        fragment's functions), most occupied first; within a block of equal occupations the
        eigenvectors of :attr:`kohn_sham`, lowest first. A spin-polarized problem has one pair per
        spin channel."""
        occupations, rotation = self._native.natural_occupations, self._native.natural_rotation
        if self.spin_polarized:
            return [(occupations[s], rotation[s]) for s in range(2)]
        return occupations[0], rotation[0]

    def integrals(self, basis: str = "wannier") -> tuple:
        """``(h, <pq|rs>)`` of the model (Hartree), shaped as the fields: in the fragment's Wannier
        functions, or with ``basis="natural"`` in :meth:`natural_orbitals`: the same model,
        rotated."""
        h = self._native.one_body(basis)
        g = self._native.two_body(basis)
        if self.spin_polarized:
            return h, g
        return h[0], g[0, 0]

    def fermion_hamiltonian(self, basis: str = "wannier"):
        """The spin-orbital :class:`~mandacaru.core.mapping.Fermion` Hamiltonian (Hartree, no
        constant) in ``basis`` (:meth:`integrals`), written from the spatial (or spin-orbital for
        spinor crystals) integrals."""
        from ..core.mapping import Fermion

        if self.spinor:
            return Fermion.from_integrals(*self.integrals(basis))
        return Fermion.from_spatial_integrals(*self.integrals(basis))

    def as_quantum_problem(self) -> dict:
        """Options for ``Mandacaru(method='adapt-vqe', **options)``.

        The model is written in the natural orbitals of the Kohn-Sham density matrix (per spin
        channel for a spin-polarized crystal), most occupied first, so the reference determinant
        (the first orbitals of each spin occupied) is the one nearest the Kohn-Sham state; a
        determinant of localized functions is a poor start.
        """
        return {
            "hamiltonian": self.fermion_hamiltonian("natural"),
            "num_particles": tuple(int(x) for x in self.num_particles),
            "n_spatial_orbitals": self.n_spatial_orbitals,
        }


class WannierResult:
    """Unit-converted view of one native localized spin channel."""

    def __init__(self, native, spin):
        self._native = native
        self._spin = spin
        self._snapshot = native.snapshot(spin=spin)

    @property
    def n_functions(self):
        return len(self._snapshot["centers"])

    @property
    def centers(self):
        return to_angstrom(self._snapshot["centers"])

    @property
    def spreads(self):
        return _native.to_user_units(np.ascontiguousarray(self._snapshot["spreads"], dtype=float), "area")

    @property
    def total_spread(self):
        parts = [float(self._snapshot[name]) for name in
                 ("omega_invariant", "omega_offdiagonal", "omega_diagonal")]
        return _native.to_user_units(_native.total(parts), "area")

    @property
    def converged(self):
        return self._snapshot["converged"] and self._snapshot["disentanglement_converged"]

    @property
    def labels(self):
        return tuple(self._snapshot["labels"])

    @property
    def trial_weights(self):
        """``(n_functions, n_trials)``: ``|<w_n|g_m>|^2``, the weight of trial orbital ``m``
        (labeled by :attr:`labels`) in function ``n``. A localized function need not keep the
        character of the trial it started from, so select functions by these weights (for
        example the two of largest ``e_g`` weight on an atom), not by index."""
        return self._snapshot["trial_weights"].copy()

    @property
    def kpoints(self):
        return self._snapshot["kpoints"].copy()

    @property
    def lattice_vectors(self):
        return self._snapshot["lattice_vectors"].copy()

    @property
    def hamiltonian_R(self):
        return from_hartree(self._snapshot["hamiltonian_R"], "eV")

    @property
    def positions_R(self):
        """Wannier position operator ``A(R) = <0 | r | R>`` of shape ``(3, n_R, n_W, n_W)`` in Angstrom."""
        return _native.to_user_units(self._snapshot["positions_R"], "length")

    @property
    def position_asymmetry(self):
        """Frobenius norm of the pre-Hermitization position mismatch ``A(R) - A(-R)^H`` in Angstrom."""
        return float(to_angstrom(self._snapshot["position_asymmetry"]))

    @property
    def localization(self):
        return str(self._snapshot["localization"])

    @property
    def populations(self):
        value = self._snapshot["populations"]
        return None if value is None else str(value)

    @property
    def exponent(self):
        value = self._snapshot["exponent"]
        return None if value is None else int(value)

    def interpolate(self, kpoints, *, eigenvectors=False):
        raw = self._native.interpolate(
            np.ascontiguousarray(kpoints, dtype=float),
            spin=self._spin, eigenvectors=bool(eigenvectors))
        if eigenvectors:
            energies, vectors = raw
            return from_hartree(energies, "eV"), vectors
        return from_hartree(raw, "eV")

    def band_structure(self, path=None, npoints=100):
        """Wannier-interpolated band structure along ``path`` (eV), as an ASE ``BandStructure``."""
        return _band_structure(
            self._snapshot["cell_bohr"], self._snapshot["fermi_ha"], path, npoints,
            lambda kpts: self.interpolate(kpts)[None, :, :])

    def anomalous_hall(self, kpts=None, *, refine=True):
        """Anomalous Hall conductivity by Wannier interpolation (:class:`~mandacaru.analysis.berry.AnomalousHall`)."""
        return _anomalous_hall(self._native, _mesh_size(kpts, self._snapshot["size"]), refine)

    def downfold(self, functions=None, num_particles=None, **options):
        """The many-body problem of a fragment of these functions (:class:`DownfoldedProblem`).

        ``functions`` lists the fragment's modes: an index ``n`` (the function of the home cell)
        or ``(n, R)`` with ``R`` an integer lattice vector; default every function of the home
        cell. ``num_particles`` ``(up, down)`` defaults to the Kohn-Sham count, the trace of the
        density matrix rounded (it must be within 0.1 of an integer) and shared equally between
        the spins. The interaction is the Coulomb interaction on the Born-von Karman supercell of
        the k mesh, truncated beyond half its shortest lattice vector, and bare unless
        ``screening`` says otherwise; ``screening``, ``screening_cutoff``, ``screening_bands`` and
        ``coulomb`` are the options of the native downfolding.
        """
        if self._native.n_spins != 1:
            raise ValueError("downfold the spin-polarized result, not one of its spin channels")
        return _downfold(self._native, functions, num_particles, self.n_functions, options)


def _band_structure(cell_bohr, fermi_ha, path, npoints, eval_energies):
    from ase.dft.kpoints import BandPath
    from ase.spectrum.band_structure import BandStructure
    from . import kspace as _kspace

    cell = np.ascontiguousarray(to_angstrom(cell_bohr), dtype=float)
    text, points = _kspace.path_arguments(path)
    raw = _native.ase_band_path(cell, [True, True, True], int(npoints), text, points)
    bandpath = BandPath(cell, kpts=raw["kpoints"], special_points=raw["special_points"],
                        path=raw["path"])
    return BandStructure(bandpath, energies=eval_energies(bandpath.kpts),
                         reference=from_hartree(float(fermi_ha), "eV"))


def _anomalous_hall(native, size, refine):
    from ..analysis.berry import AnomalousHall

    return AnomalousHall.from_native(
        native.anomalous_hall(size, refine=bool(refine)), stacklevel=3)


def _downfold(native, functions, num_particles, n_functions, options):
    modes = _modes(functions, n_functions)
    if num_particles is not None:
        num_particles = tuple(int(x) for x in num_particles)
        if len(num_particles) != 2:
            raise ValueError(f"num_particles is (up, down); got {num_particles!r}")
    return DownfoldedProblem(native.downfold(modes, num_particles, **options))


class SpinWannierResult:
    """Two spin channels sharing one native calculation owner."""

    def __init__(self, native):
        self._native = native
        self.up = WannierResult(native, 0)
        self.down = WannierResult(native, 1)

    @property
    def centers(self):
        return np.stack([self.up.centers, self.down.centers])

    @property
    def total_spread(self):
        return np.array([self.up.total_spread, self.down.total_spread])

    @property
    def converged(self):
        return self.up.converged and self.down.converged

    @property
    def localization(self):
        return self.up.localization

    @property
    def populations(self):
        return self.up.populations

    @property
    def exponent(self):
        return self.up.exponent

    def interpolate(self, kpoints, *, eigenvectors=False):
        if eigenvectors:
            e_up, v_up = self.up.interpolate(kpoints, eigenvectors=True)
            e_dn, v_dn = self.down.interpolate(kpoints, eigenvectors=True)
            return np.stack([e_up, e_dn]), np.stack([v_up, v_dn])
        return np.stack([self.up.interpolate(kpoints), self.down.interpolate(kpoints)])

    def band_structure(self, path=None, npoints=100):
        """Wannier-interpolated spin-polarized band structure along ``path`` (eV), as an ASE ``BandStructure``."""
        snap = self.up._snapshot
        return _band_structure(snap["cell_bohr"], snap["fermi_ha"], path, npoints, self.interpolate)

    def anomalous_hall(self, kpts=None, *, refine=True):
        """Anomalous Hall conductivity by Wannier interpolation (:class:`~mandacaru.analysis.berry.AnomalousHall`)."""
        return _anomalous_hall(self._native, _mesh_size(kpts, self.up._snapshot["size"]), refine)

    def downfold(self, functions=None, num_particles=None, **options):
        """The many-body problem of a fragment of these functions, one set per spin channel
        (:meth:`WannierResult.downfold`)."""
        return _downfold(self._native, functions, num_particles, self.up.n_functions, options)
