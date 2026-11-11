"""ASE geometry and unit views for native Wannier calculations."""
from __future__ import annotations

import json
import warnings

import numpy as np
from .. import _native
from ..units import from_hartree, to_angstrom, to_bohr, to_hartree


def _trials(guess, atoms):
    centers = to_bohr(atoms.positions)
    if isinstance(guess, str) and guess in ("bonds", "sp3"):
        from ase.neighborlist import neighbor_list

        first, vectors = neighbor_list("iD", atoms, 6.0)
        key = "neighbors" if guess == "sp3" else "bond_neighbors"
        return [
            dict(center=center.tolist(), label=f"{atom.symbol}{i}",
                 **{key: to_bohr(vectors[first == i]).tolist()})
            for i, (atom, center) in enumerate(zip(atoms, centers))
        ]
    if isinstance(guess, dict):
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


def calculate(calculation, n_functions, *, guess, windows, bands, kpts,
              max_iterations, tolerance, step):
    if calculation.result is None or calculation.result.periodic is None:
        raise RuntimeError("run periodic DFT before constructing Wannier functions")
    problem = calculation.periodic_problem
    centers, channels = problem.centers, problem.channel_labels
    atoms = calculation.atoms
    mesh = calculation._kpts_spec if kpts is None else kpts
    size = mesh["size"] if isinstance(mesh, dict) else mesh
    spec = dict(size=[int(n) for n in size], bands=bands,
                n_functions=n_functions, trials=_trials(guess, atoms),
                centers=centers,
                channel_labels=channels)
    if windows is not None:
        if not isinstance(windows, dict) or "outer" not in windows:
            raise ValueError("Wannier windows specify an outer energy pair and an optional frozen pair in eV")
        spec["windows"] = {
            name: None if pair is None else [to_hartree(float(e), "eV") for e in pair]
            for name, pair in windows.items()
        }
    setup = calculation.result.periodic.wannier_setup(json.dumps(spec, allow_nan=False))
    setup.add_points_at(_native.mesh_fractions(spec["size"]))
    native = setup.localize(max_iterations=max_iterations, tolerance=tolerance, step=step)
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
    def functions(self) -> tuple:
        return tuple((n, tuple(R)) for n, R in self._native.functions)

    @property
    def n_spatial_orbitals(self) -> int:
        return self._native.n_spatial_orbitals

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
        constant) in ``basis`` (:meth:`integrals`), written from the spatial integrals."""
        from ..core.mapping import Fermion

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
    def kpoints(self):
        return self._snapshot["kpoints"].copy()

    @property
    def lattice_vectors(self):
        return self._snapshot["lattice_vectors"].copy()

    @property
    def hamiltonian_R(self):
        return from_hartree(self._snapshot["hamiltonian_R"], "eV")

    def interpolate(self, kpoints):
        return from_hartree(self._native.interpolate(np.ascontiguousarray(kpoints, dtype=float),
                                                     spin=self._spin), "eV")

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

    def interpolate(self, kpoints):
        return np.stack([self.up.interpolate(kpoints), self.down.interpolate(kpoints)])

    def downfold(self, functions=None, num_particles=None, **options):
        """The many-body problem of a fragment of these functions, one set per spin channel
        (:meth:`WannierResult.downfold`)."""
        return _downfold(self._native, functions, num_particles, self.up.n_functions, options)
