# -*- coding: utf-8 -*-
# file: algorithms/bloch.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Bloch / k-point variational eigensolvers for periodic systems (1-, 2-, 3-D).

The periodic drivers are reached through the single entry point, by name::

    atoms.calc = Mandacaru(method="bloch-adapt-vqe",
                           kpts={"size": (2, 1, 1), "gamma": True},
                           basis="HAO", h=0.35)
    energy_per_cell = atoms.get_potential_energy()   # eV, per primitive cell
    spectral = atoms.calc.get_spectral_function()    # A(E, k)
    epsilon = atoms.calc.bands(spectral=spectral)    # eV, (nk, n_bands)
    weight = atoms.calc.band_weights(spectral=spectral)
    mu = atoms.calc.get_fermi_level(spectral=spectral)

``atoms`` is the **primitive cell**: ``atoms.cell`` sets the lattice vectors and
``atoms.pbc`` selects the periodic directions (at least one must be periodic;
1-D chains and 2-D slabs are supported).  ``kpts`` is the Brillouin-zone
Monkhorst-Pack sampling in ASE's own spelling and **must be Gamma-centered** --
see :class:`_BlochMixin` for why.

The drivers provide the two things a periodic calculation needs:

* **Total energy using all k-points** -- a correlated solver cannot be run per
  k-point and summed (the two-electron interaction couples crystal momenta), so
  the energy uses the **Born-von Karman equivalence**: an ``(n1, n2, n3)``
  Gamma-centered mesh is a Gamma-point calculation on the ``(n1, n2, n3)``
  supercell, and the energy per cell is ``E(supercell) / n_cells``.  The
  supercell is built with :meth:`ase.Atoms.repeat` -- the primitive cell
  repeated, with **no extra padding** -- so the primitive cell is what you
  size: its extent along the non-periodic directions is the empty space the
  supercell gets.  Its Hamiltonian is built with lattice-summed electrostatics
  (:class:`~mandacaru.core.periodic.PeriodicIntegrals`), which is what makes it
  a crystal rather than a molecule in a box.

  This is a finite-supercell estimate, and how tightly it settles depends on the
  gap.  An **insulating** H2 chain (a = 3.0, d = 0.74 Angstrom, 10 Angstrom
  vacuum, HAO, h = 0.30, SLSQP) gives -29.315 / -28.142 / -28.156 / -28.108 eV
  per cell at 1 to 4 k-points: the first refinement moves it by 1.2 eV and the
  rest stay inside a 0.05 eV band -- though not monotonically, so read it as a
  band rather than as a sequence converging from one side.  A **metal** swings
  an order of magnitude more, because meshes differ in whether they are
  degenerate at the Fermi level: the half-filled H chain (a = 1.0 Angstrom,
  h = 0.35) gives -10.577 / -12.940 / -11.592 eV per cell at 4, 6 and 8
  k-points, since 4 and 8 are degenerate at ``E_F`` while 6 is closed-shell.
  Read convergence off a gapped system, or off meshes of a single shell type.

* **Band structure** -- with the two-body term on, a one-particle level is not
  an eigenvalue of anything: the weight of a given ``k`` spreads over several
  ``N+-1`` eigenstates.  What replaces the band is the **spectral function**
  ``A(E, k)`` (:meth:`_BlochMixin.get_spectral_function`), built by the Lehmann
  representation from the correlated ground state the run produced;
  :meth:`_BlochMixin.bands` reports its dominant pole per orbital and
  :meth:`_BlochMixin.band_weights` how much weight that pole carries.  The
  exact check is the sum rule: removal and addition weights add to 1 for every
  k, spin and orbital, since ``{c, c^dagger} = 1``.

  **It is resolved only at the commensurate k-points** the supercell realizes,
  so there is no continuous band path --
  :meth:`_BlochMixin.band_structure` raises.  A finer band needs a larger
  ``kpts``, hence a larger supercell and proportionally more qubits.

**Forces and stress are available** on the periodic path, through
:mod:`mandacaru.algorithms.periodic_forces`.  The Born-von Karman equivalents
of the two molecular terms are what that module builds: Hellmann-Feynman
differentiates the Ewald potential of the whole ion lattice together with the
analytic ion-ion Ewald force, and Pulay re-samples the displaced atom's basis
functions *including their periodic images*.  The gradient is computed on the
supercell and folded back to the primitive cell.  ``atoms.get_forces()``
requires ``force_method='rdm'`` -- the default -- since the differentiated-SCF
path rebuilds :math:`-Z/r` potentials a lattice-summed Hamiltonian was never
assembled from.  :meth:`~mandacaru.algorithms.calculator.Mandacaru.get_stress`
strains the cell, the atoms and the grid together, holding the node count fixed
so the strained grid stays commensurate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

import numpy as np

#: The stable periodic method names, in the spelling ``Mandacaru`` takes.
BLOCH_METHODS = ("bloch-vqe", "bloch-adapt-vqe")

#: Tolerance for deciding that a mesh point is the Gamma point.
GAMMA_TOLERANCE = 1e-12

#: Largest relative disagreement between a symmetry-equivalent k-point and its
#: representative that ``irreducible=True`` accepts without warning.  The
#: spectral function inherits the symmetry of the *state*, not of the lattice,
#: and a state need not have it: a partially filled degenerate manifold -- a
#: metallic mesh -- makes the mean-field reference pick one member and break
#: the point group, measured at 17 % of the peak on a half-filled 2x2x1 square
#: lattice.  So the reduction is audited rather than assumed.
SYMMETRY_RESIDUAL_TOLERANCE = 1e-6


@dataclass
class SpectralFunction:
    """The one-particle spectral function ``A(E, k)`` of the interacting state.

    Attributes
    ----------
    energies : ndarray, shape (nE,)
        Energy grid in **eV**.
    kpoints : ndarray, shape (nk, 3)
        Fractional coordinates of the commensurate k-points it is resolved at.
    weights : ndarray, shape (nk, nE)
        ``A(k, E)`` in eV^-1, summed over spin and over the orbitals of the
        primitive cell, Lorentzian-broadened by :attr:`eta`.
    poles : list of (ndarray, ndarray)
        Per k-point, the unbroadened Lehmann poles and their weights -- the
        removal (below the chemical potential) and addition branches together,
        summed over the orbitals of the primitive cell and over spin.
    orbital_poles : list of list of (ndarray, ndarray)
        The same poles resolved per primitive-cell orbital, ``[ik][nu]``, which
        is what :meth:`_BlochMixin.bands` reads the quasiparticle peak from.
    eta : float
        Lorentzian broadening in eV.
    chemical_potential : float
        Midpoint of the gap between the highest removal pole and the lowest
        addition pole, in eV: the interacting ``mu``.
    sum_rule : float
        Largest deviation from 1 of the per-operator weight sum.  An exact
        identity, so anything above ~1e-10 means the construction is wrong.
    kpath : KPointPath or None
        Set when the spectrum was asked for along a high-symmetry path.  Its
        :attr:`~mandacaru.algorithms.kpath.KPointPath.distances` is the x-axis
        a 2-D or 3-D band plot needs, since there the sampling mesh is not
        itself a line.  ``None`` for a plain mesh, where the fractional
        coordinate is the natural axis.
    irreducible : IrreducibleZone or None
        Set when the spectrum was evaluated on the irreducible wedge and spread
        over the rest by symmetry.  Every k-point of :attr:`kpoints` still
        carries its own spectrum; this records which ones were computed and
        which were filled in.
    """

    energies: np.ndarray
    kpoints: np.ndarray
    weights: np.ndarray
    poles: list
    orbital_poles: list
    eta: float
    chemical_potential: float
    sum_rule: float
    kpath: object = None
    irreducible: object = None

    @property
    def distances(self) -> np.ndarray:
        """x-axis for a band plot: path length if on a path, else the index.

        A 1-D chain can be plotted against its fractional coordinate, but a
        square or hexagonal lattice cannot -- the mesh is two-dimensional and
        only a path through it is a line.  This is the one axis that is correct
        in every case.
        """
        if self.kpath is not None:
            return np.asarray(self.kpath.distances, dtype=float)
        return np.arange(len(self.kpoints), dtype=float)

    def write(self, path) -> str:
        """Write the broadened spectrum as long-format CSV; returns the path.

        All three fractional components are written.  Writing only the first
        was enough for a chain and silently dropped ``k_y`` for any lattice
        sampled in more than one direction, which is every 2-D or 3-D case.
        """
        import csv

        path = os.fspath(path)
        axis = self.distances
        labels = {}
        if self.kpath is not None:
            for label, position in zip(self.kpath.labels,
                                       self.kpath.label_distances):
                # A path can meet the same distance twice only at a jump, where
                # either label names the same point.
                labels.setdefault(round(float(position), 10), label)

        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["k_index", "k1", "k2", "k3", "k_distance_inv_ang",
                             "label", "energy_eV", "spectral_weight_per_eV"])
            for ik, kpt in enumerate(self.kpoints):
                tag = labels.get(round(float(axis[ik]), 10), "")
                for energy, weight in zip(self.energies, self.weights[ik]):
                    writer.writerow([ik, f"{kpt[0]:.10f}", f"{kpt[1]:.10f}",
                                     f"{kpt[2]:.10f}", f"{axis[ik]:.10f}", tag,
                                     f"{energy:.10f}", f"{weight:.10e}"])
        return path


class _BlochMixin:
    """The periodic layer, mixed over a molecular variational driver.

    The driver it is mixed over solves the Born-von Karman supercell: this
    class replaces the geometry with ``atoms.repeat(kpts)`` on the way in and
    divides the energy by the number of cells on the way out, so everything the
    molecular driver owns -- the run log, the result object, the timings, the
    option validation -- keeps working untouched.

    Validation is split by what is known when.  The **constructor** sees only
    the options, so it checks that ``kpts`` was given, is well formed and is
    Gamma-centered, and that this is not direct mode.  The **first calculation**
    (and the dry run) sees the geometry, so it checks the periodicity there,
    before any basis, grid or Hamiltonian is built.

    ``kpts`` must be Gamma-centered because the Born-von Karman equivalence is
    an identity between a Gamma-centered mesh and the Gamma point of the
    corresponding supercell.  A shifted mesh corresponds to a supercell
    calculation at a *non*-Gamma k-point, which this engine cannot do, so
    accepting one and returning the Gamma-centered answer would be silently
    wrong.  Write ``kpts={"size": (2, 1, 1), "gamma": True}``.
    """

    #: The supercell is a crystal: its Hamiltonian is built with lattice-summed
    #: electrostatics (:class:`~mandacaru.core.periodic.PeriodicIntegrals`), and
    #: this flag is what routes ``Mandacaru._forces`` to
    #: :mod:`mandacaru.algorithms.periodic_forces` -- where Hellmann-Feynman
    #: differentiates the Ewald potential of the ion lattice and the analytic
    #: Ewald ion-ion energy, and Pulay the image-summed basis.  Validated
    #: against a central difference of the same fixed-state energy at
    #: 1.5e-4 eV/Angstrom.
    periodic_hamiltonian = True

    def __init__(self, *, kpts=None, **kwargs):
        if kpts is None:
            raise ValueError(
                "a periodic method needs a Brillouin-zone sampling: pass "
                "kpts={'size': (n1, n2, n3), 'gamma': True} (the mesh must be "
                "Gamma-centered, which is what the Born-von Karman supercell "
                "equivalence is an identity for).")
        for name in ("hamiltonian", "load_hamiltonian"):
            if kwargs.get(name) is not None:
                raise ValueError(
                    f"a periodic method cannot run in direct mode ({name}=): "
                    "the lattice comes from an ASE Atoms primitive cell, and a "
                    "bare Hamiltonian carries no cell or pbc.  Use "
                    "atoms.calc = Mandacaru(method=..., kpts=...) with a "
                    "geometry, or a molecular method for a direct problem.")
        super().__init__(kpts=kpts, **kwargs)
        if not self._mesh_is_gamma_centered():
            n1, n2, n3 = self.kpts
            raise ValueError(
                f"kpts={self.kpts} resolves to a Monkhorst-Pack mesh that does "
                "not contain the Gamma point, so it is not a Born-von Karman "
                f"supercell: write kpts={{'size': ({n1}, {n2}, {n3}), "
                "'gamma': True}} instead.  The equivalence this method uses "
                "holds only for a Gamma-centered mesh.")
        self._primitive = None         # last primitive cell seen

    # -- option-level checks ---------------------------------------------- #
    def _mesh_is_gamma_centered(self) -> bool:
        """Whether the resolved mesh actually contains ``k = 0``."""
        return bool(np.any(np.all(np.abs(self.kpoints) < GAMMA_TOLERANCE,
                                  axis=1)))

    @property
    def n_supercells(self) -> int:
        """Primitive cells in the Born-von Karman supercell."""
        return int(np.prod([int(k) for k in self.kpts]))

    def _grid_commensurate(self):
        r"""The mesh: the supercell grid must resolve a *primitive* translation.

        The supercell spans ``n_i`` primitive cells along lattice vector ``i``,
        so a primitive translation is ``N_i / n_i`` grid steps.  Unless that is
        a whole number the grid is **not** invariant under the translation
        Bloch's theorem is built on: the repeated atoms sample it at different
        offsets, and sites that are copies of one another stop being
        equivalent.  Measured on a 2x2x1 square lattice at ``h = 0.35`` (15
        nodes, 7.5 steps per cell): the four equivalent sites' on-site energies
        spread by 0.10 Ha, a band pair that C4 makes degenerate came out
        1.8e-3 Ha apart, and ``A(E, k)`` at two C4-equivalent k-points differed
        by 17 % of its peak.  At 16 nodes the first two are zero to round-off.

        Rounding the node count up to a multiple of ``kpts`` costs a slightly
        finer grid than was asked for and buys exact lattice symmetry.
        """
        return tuple(int(k) for k in self.kpts)

    # -- geometry-level checks -------------------------------------------- #
    def _check_periodic_atoms(self, atoms) -> None:
        """Refuse a geometry the periodic method cannot mean anything for.

        Runs before any basis, grid or Hamiltonian is built, so a wrong
        geometry costs nothing.
        """
        periodic = [i for i in range(3) if bool(atoms.pbc[i])]
        if not periodic:
            raise ValueError(
                "a periodic method needs periodic boundary conditions, and "
                "this geometry has none: set atoms.pbc (e.g. "
                "pbc=[True, False, False] for a 1-D chain, [True, True, False] "
                "for a slab), or use method='adapt-vqe' for a molecule.")

        axes = "abc"
        for i in range(3):
            if i not in periodic and int(self.kpts[i]) != 1:
                raise ValueError(
                    f"kpts has {self.kpts[i]} k-points along direction "
                    f"{axes[i]}, which is not periodic (atoms.pbc = "
                    f"{tuple(bool(p) for p in atoms.pbc)}): a Brillouin zone "
                    "exists only along a periodic direction.  Use 1 there, or "
                    f"make {axes[i]} periodic.")

        cell = np.asarray(atoms.cell, dtype=float)
        lengths = np.linalg.norm(cell, axis=1)
        if np.any(lengths <= 0.0):
            raise ValueError(
                "every lattice vector of the primitive cell must have a "
                "non-zero length: the cell is also the box along the "
                "non-periodic directions (e.g. cell=[a, 10, 10] for a 1-D "
                f"chain); got lengths {lengths.round(3).tolist()} Angstrom.")
        rows = cell[periodic]
        if np.linalg.matrix_rank(rows, tol=1e-10) < len(periodic):
            raise ValueError(
                "the lattice vectors of the periodic directions are "
                "degenerate (zero or collinear), so they do not span a "
                f"{len(periodic)}-D lattice: got "
                f"{rows.round(4).tolist()} for directions "
                f"{[axes[i] for i in periodic]}.")

    # -- the Born-von Karman supercell ------------------------------------ #
    def supercell(self, atoms=None):
        """The ``kpts`` Born-von Karman supercell as an ASE ``Atoms``.

        Built with :meth:`ase.Atoms.repeat`: the primitive cell repeated and
        nothing else -- no padding is added, so the primitive cell's extent
        along the non-periodic directions is the empty space it carries.
        """
        atoms = self._geometry(atoms)
        return atoms.repeat(tuple(int(k) for k in self.kpts))

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=None):
        """Solve the supercell, report the energy **per primitive cell**."""
        from ase.calculators.calculator import all_changes

        if system_changes is None:
            system_changes = all_changes
        atoms = atoms if atoms is not None else self.atoms
        self._check_periodic_atoms(atoms)
        self._primitive = atoms.copy()
        super().calculate(self.supercell(atoms), properties, system_changes)
        n_cells = self.n_supercells
        for key in ("energy", "free_energy"):
            if key in self.results and self.results[key] is not None:
                self.results[key] = self.results[key] / n_cells

    # -- the k-point mesh is realized, not sampled ------------------------ #
    def _check_kpts(self) -> None:
        """No-op: the mesh *is* the supercell, so there is nothing to refuse.

        The molecular guard rejects a denser mesh because the engine solves a
        Gamma-point problem.  Here the mesh has already been turned into the
        supercell, and the problem actually solved is that supercell's Gamma
        point -- which is exactly what the engine does.
        """

    def _kpts_label(self) -> str:
        """The one place the mesh and the supercell are stated (``[ELECTRONS]``)."""
        n1, n2, n3 = (int(k) for k in self.kpts)
        return (f"{len(self.kpoints)} k-points ({n1}x{n2}x{n3} Monkhorst-Pack, "
                f"Gamma-centered) = {n1}x{n2}x{n3} Born-von Karman supercell "
                f"({self.n_supercells} cells)")

    # -- dry run ----------------------------------------------------------- #
    def estimate_qubits(self, atoms=None):
        """Qubit requirements of the **supercell**, without running anything."""
        atoms = self._geometry(atoms)
        self._check_periodic_atoms(atoms)
        estimate = super().estimate_qubits(self.supercell(atoms))
        n1, n2, n3 = (int(k) for k in self.kpts)
        return replace(
            estimate,
            basis=(f"{estimate.basis} [{n1}x{n2}x{n3} Born-von Karman "
                   f"supercell, {self.n_supercells} cells]"))

    # -- geometry bookkeeping ---------------------------------------------- #
    def _geometry(self, atoms=None):
        """The primitive cell to work from."""
        if atoms is not None:
            return atoms
        # `self.atoms` is the *supercell* once a calculation has run, so the
        # primitive cell kept aside takes precedence.
        if self._primitive is not None:
            return self._primitive
        if getattr(self, "atoms", None) is not None:
            return self.atoms
        raise ValueError(
            "no geometry: the band structure needs the primitive cell.  "
            "Either pass it -- calc.bands(kpts, atoms=atoms) -- or attach the "
            "calculator and evaluate it first (atoms.calc = calc; "
            "atoms.get_potential_energy()).")

    def _periodic_directions(self, atoms):
        return [i for i in range(3) if bool(atoms.pbc[i])]

    @property
    def dimension(self) -> int:
        """Number of periodic directions (1, 2 or 3) of the last geometry."""
        return len(self._periodic_directions(self._geometry()))

    @property
    def n_bands(self) -> int:
        """Bands = spatial orbitals per primitive cell.

        A property of the basis, so it is available before anything is run --
        unlike the bands themselves, which need the correlated state.
        """
        from .dry_run import count_basis_functions

        per_atom, _label = count_basis_functions(self._geometry(), self.basis)
        return int(sum(count for _symbol, count in per_atom))

    # -- bands, as the peaks of the interacting spectral function ---------- #
    def bands(self, spectral=None, atoms=None, **kwargs) -> np.ndarray:
        """Quasiparticle bands: the dominant pole of ``A(E, k)`` per orbital.

        With the two-body term on, a one-particle level is not an eigenvalue of
        anything -- the weight of a given ``k`` spreads over several ``N+-1``
        eigenstates.  What is plotted as a band is the **quasiparticle peak**:
        the pole carrying the most weight.  Read it together with
        :meth:`band_weights`, because a peak holding 0.4 of the spectral weight
        is not the same object as one holding 0.99, and only the second is a
        band in the textbook sense.

        Returns an ``(nk, n_bands)`` array in **eV, measured from the
        N-electron ground state**, at the commensurate k-points of the mesh --
        ``self.kpoints``, in that order, and no others.  A finer band needs a
        larger mesh, hence a larger supercell and more qubits.

        Extra keyword arguments go to :meth:`get_spectral_function`, so
        ``calc.bands(path="GXMG")`` gives the rows along a high-symmetry path
        (in path order) and ``calc.bands(irreducible=True)`` computes only the
        symmetry-inequivalent ones.  In 2-D and 3-D the mesh is not a line, so
        a band plot needs ``path=``; pair it with the matching
        ``get_spectral_function(...).kpath.distances`` for the x-axis.
        """
        spectral = (spectral if spectral is not None
                    else self.get_spectral_function(atoms=atoms, **kwargs))
        return np.array([[poles[np.argmax(weights)]
                          for poles, weights in per_orbital]
                         for per_orbital in spectral.orbital_poles])

    def band_weights(self, spectral=None, atoms=None, **kwargs) -> np.ndarray:
        """Spectral weight carried by each quasiparticle peak of :meth:`bands`.

        ``(nk, n_bands)``, each between 0 and 1.  A value near 1 means the peak
        is a sharp one-particle excitation; a small one means the weight is
        spread over satellites and the "band" there is not well defined.
        """
        spectral = (spectral if spectral is not None
                    else self.get_spectral_function(atoms=atoms, **kwargs))
        return np.array([[weights.max() for _poles, weights in per_orbital]
                         for per_orbital in spectral.orbital_poles])

    def band_structure(self, *args, **kwargs):
        """Refused: a finite supercell has no continuous band path.

        The Bloch operators exist only at the commensurate k-points the
        Born-von Karman supercell realizes, so there is no way to evaluate an
        interacting band between them.  Use :meth:`bands` on the mesh, or
        enlarge ``kpts`` to resolve more of the zone.
        """
        raise NotImplementedError(
            "a continuous band path is not available for an interacting "
            f"periodic calculation: the {self.n_supercells}-cell Born-von "
            "Karman supercell defines Bloch operators at its own "
            f"{len(self.kpoints)} commensurate k-points and nowhere else.  Use "
            "calc.bands() on that mesh, or raise kpts to resolve more of the "
            "Brillouin zone (at a proportional cost in qubits).")

    def electrons_per_cell(self, atoms=None) -> int:
        """Electrons in one primitive cell.

        ``charge`` and ``n_electrons`` describe the *supercell* the driver
        solves, so they are brought back to a single cell here.
        """
        atoms = self._geometry(atoms)
        n_cells = self.n_supercells
        if getattr(self, "n_electrons", None) is not None:
            total = float(self.n_electrons)
        else:
            total = (float(sum(atoms.get_atomic_numbers())) * n_cells
                     - float(self.charge or 0))
        per_cell = total / n_cells
        if abs(per_cell - round(per_cell)) > 1e-9:
            raise ValueError(
                f"{total:g} electrons over {n_cells} cells is not a whole "
                "number per cell, so the bands cannot be filled cell by cell.")
        return int(round(per_cell))

    def get_fermi_level(self, spectral=None, atoms=None, **kwargs) -> float:
        """The interacting chemical potential (eV), from ``A(E, k)``.

        Midway between the highest removal pole and the lowest addition pole --
        between minus the ionization energy and minus the electron affinity of
        the supercell.  That is the chemical potential of the correlated state,
        not a filling convention applied to one-particle levels, and it is
        measured from the N-electron ground state like the bands.
        """
        spectral = (spectral if spectral is not None
                    else self.get_spectral_function(atoms=atoms, **kwargs))
        return spectral.chemical_potential

    # -- the interacting band structure: A(E, k) --------------------------- #
    def _spectral_context(self):
        """The finished run's state, Hamiltonian and orbital rotation."""
        if getattr(self, "result", None) is None:
            raise ValueError(
                "the spectral function needs the correlated ground state, so "
                "the energy has to have been computed first: "
                "atoms.calc = Mandacaru(...); atoms.get_potential_energy().")
        # Imported here: calculator.py reads BLOCH_METHODS from this
        # module at import time, so the dependency cannot be mutual.
        from .calculator import _method_key

        if _method_key(self.mapping) != _method_key("jordan_wigner"):
            raise NotImplementedError(
                f"the spectral function is implemented for the Jordan-Wigner "
                f"mapping only; this run used {self.mapping!r}.  A tapered "
                "register (parity_reduced) encodes the particle number itself, "
                "so the N+-1 sectors it would need do not exist there.")
        context = getattr(self, "_gradient_context", None) or {}
        integrals = context.get("integrals")
        if integrals is None or getattr(integrals, "mo_coefficients", None) is None:
            raise ValueError(
                "the spectral function needs the orbital rotation of the run "
                "that produced the state; this run did not keep one.")
        if context.get("frozen"):
            raise NotImplementedError(
                "the spectral function does not support a frozen core yet: the "
                "removal branch would have to reach the frozen orbitals.")
        if context.get("deleted"):
            raise NotImplementedError(
                "the spectral function does not support a truncated virtual "
                "space (active_orbitals=): the addition branch puts an extra "
                "electron into the virtual orbitals, and the ones the selector "
                "deleted are exactly where it would go.  The peaks it found "
                "would be those of the truncated model presented as the "
                "material's.  Run the spectral function with the full virtual "
                "space.")
        psi = self.ansatz.state(self.result.optimal_parameters)
        # The reference energy has to be taken in whatever representation the
        # ansatz works in, because `energy()` contracts with the Hamiltonian in
        # that same representation -- restricted to the sector when the run is
        # a sector run.  Take it first, then move the state.
        reference = float(np.real(self.energy(psi)))
        # Above `SECTOR_AUTO_QUBITS` the ansatz works inside the
        # particle-number sector, so the state comes back with the sector's
        # dimension rather than 2**n.  The creation operators below move
        # between sectors, which only means anything in the full space.
        full = 2 ** self.n_qubits
        if psi.size != full:
            from ..core.sector import ParticleSector

            sector = ParticleSector(self.n_qubits, self.num_particles,
                                    self.mapping)
            if psi.size != sector.dim:
                raise ValueError(
                    f"the optimized state has {psi.size} amplitudes, which is "
                    f"neither the full register ({full}) nor the "
                    f"{self.num_particles} sector ({sector.dim}).")
            psi = sector.embed(psi)
        return psi, np.asarray(integrals.mo_coefficients), reference

    def _bloch_creation(self, kpt, nu, spin, coefficients, n_orbitals):
        """``c^dagger`` of the Bloch combination, as a qubit operator.

        ``c^dagger_{k,nu,sigma} = N^-1/2 sum_R e^{2 pi i k.R} c^dagger_{R,nu,sigma}``,
        with each cell-local orbital expanded over the supercell's own orbitals
        by the rotation the run used.  Spin-orbitals are blocked: alpha first.
        """
        from ..core import Fermion

        n_cells = self.n_supercells
        per_cell = n_orbitals // n_cells
        operator = Fermion()
        for cell, translation in enumerate(self._cell_translations()):
            phase = np.exp(2j * np.pi * float(np.dot(kpt, translation)))
            phase /= np.sqrt(n_cells)
            local = cell * per_cell + nu
            for orbital in range(n_orbitals):
                amplitude = phase * coefficients[local, orbital]
                if abs(amplitude) < 1e-14:
                    continue
                index = orbital + (0 if spin == 0 else n_orbitals)
                operator = operator + Fermion.creation(index) * amplitude
        return operator.map_to_qubits(method=self.mapping,
                                      n_modes=2 * n_orbitals).to_sparse_matrix()

    def _cell_translations(self):
        """Lattice translations of the supercell, in ``atoms.repeat`` order."""
        n1, n2, n3 = (int(k) for k in self.kpts)
        return [(i, j, k) for i in range(n1) for j in range(n2)
                for k in range(n3)]

    def _sector_eigenbasis(self, particles, hamiltonian, cache):
        """Diagonalize one ``(n_alpha, n_beta)`` sector, once.

        The sector Hamiltonian depends only on the particle numbers -- not on
        the k-point, the orbital or the spin channel -- so without this the
        same two matrices are rebuilt and re-diagonalized for every pole set,
        which dominates the cost as soon as the register is large.
        """
        from ..core.sector import ParticleSector

        if particles not in cache:
            sector = ParticleSector(self.n_qubits, particles, self.mapping)
            matrix = sector.restrict(hamiltonian).toarray()
            values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
            cache[particles] = (sector, values, vectors)
        return cache[particles]

    def _lehmann(self, chi, particles, hamiltonian, reference, addition,
                 cache=None):
        """Poles and weights of ``chi`` in one particle-number sector."""
        if min(particles) < 0:
            return np.empty(0), np.empty(0)
        sector, values, vectors = self._sector_eigenbasis(
            particles, hamiltonian, {} if cache is None else cache)
        overlap = vectors.conj().T @ sector.project(chi)
        poles = (values - reference) if addition else (reference - values)
        return poles, np.abs(overlap) ** 2

    # -- which k-points to evaluate, and where each one is reported --------- #
    def band_path(self, path=None, atoms=None, warn: bool = True):
        """The mesh k-points lying on a high-symmetry path.

        ``path=None`` takes the lattice's own default path from ASE, restricted
        to the periodic directions -- ``"MGXM"`` for a square lattice,
        ``"GMKG"`` for a hexagonal one, ``"GXWKGLUWLK,UX"`` for FCC.  Pass a
        string to choose, e.g. ``calc.band_path("GXMG")``.

        Returns a :class:`~mandacaru.algorithms.kpath.KPointPath`.  Its
        ``indices`` gather rows out of a mesh-resolved quantity and its
        ``distances`` is the x-axis; nothing is interpolated, so a point of the
        path the mesh does not carry is listed in ``missing`` instead of being
        invented.  Available before anything is run: it is a property of the
        lattice and the mesh, not of the state.
        """
        from .kpath import band_path as _band_path

        atoms = self._geometry(atoms)
        return _band_path(np.asarray(atoms.cell[:], dtype=float), self.kpoints,
                          path=path, pbc=atoms.pbc, warn=warn)

    def symmetry(self, symprec: float = None, atoms=None):
        """The space group of the primitive cell, from spglib.

        Returns a :class:`~mandacaru.core.symmetry.SymmetryInfo`.  Available
        before anything is run.
        """
        from ..core.symmetry import DEFAULT_SYMPREC, crystal_symmetry

        return crystal_symmetry(
            self._geometry(atoms),
            DEFAULT_SYMPREC if symprec is None else float(symprec))

    def irreducible_zone(self, symprec: float = None,
                         time_reversal: bool = True, atoms=None):
        """Reduce the k-point mesh to the irreducible Brillouin zone.

        Returns an :class:`~mandacaru.core.symmetry.IrreducibleZone`: one
        representative per symmetry orbit, the orbit sizes as weights, and the
        mapping that spreads a per-wedge quantity back over the mesh.

        This is a **cost** reduction for
        :meth:`get_spectral_function`, not a size reduction for the
        calculation.  The Born-von Karman supercell is fixed by the full
        ``n1 x n2 x n3`` mesh -- every k-point of that mesh already lives inside
        the one supercell being solved -- so the qubit count is unchanged.
        Available before anything is run.
        """
        from ..core.symmetry import irreducible_kpoints

        return irreducible_kpoints(
            self.kpoints,
            self.symmetry(symprec=symprec, atoms=atoms),
            time_reversal=time_reversal)

    # -- ASE's informal band-structure calculator interface ---------------- #
    def get_bz_k_points(self) -> np.ndarray:
        """Every k-point of the mesh, fractional.  ASE's spelling."""
        return np.asarray(self.kpoints, dtype=float)

    def get_ibz_k_points(self) -> np.ndarray:
        """The irreducible k-points, fractional.  ASE's spelling."""
        return self.irreducible_zone().points

    def get_k_point_weights(self) -> np.ndarray:
        """Normalized weights of :meth:`get_ibz_k_points`; they sum to 1."""
        zone = self.irreducible_zone()
        return zone.weights / float(zone.weights.sum())

    def _audit_point(self, evaluation):
        """A k-point the wedge claims to cover, for checking that it does.

        Returns ``(index_into_output, representative_slot)`` for one member of
        the largest orbit that is not its own representative, or ``None`` when
        the reduction merged nothing (then there is nothing to audit).  One
        extra Lehmann evaluation is a small price for not having to take the
        symmetry of the state on faith.
        """
        counts = np.bincount(evaluation.mapping,
                             minlength=len(evaluation.points))
        order = np.argsort(counts)[::-1]
        for slot in order:
            if counts[slot] < 2:
                break
            members = np.flatnonzero(evaluation.mapping == slot)
            for index in members:
                if index != evaluation.indices[slot]:
                    return int(index), int(slot)
        return None

    def _spectral_kpoints(self, path=None, irreducible=False, symprec=None,
                          time_reversal=True):
        """Resolve ``(output, evaluation, kpath, zone)`` for a spectrum.

        ``output`` is what gets reported, ``evaluation.points`` what is actually
        computed, and ``evaluation.mapping`` sends the second onto the first.
        Exact repeats are always collapsed -- a path returning to ``Gamma``
        names it twice and the second visit is the same operator on the same
        state -- and with ``irreducible`` the collapse uses the full point group
        instead of only the identity.
        """
        from ..core.symmetry import SymmetryInfo, irreducible_kpoints

        kpath = None
        if path is not None and path is not False:
            kpath = (path if hasattr(path, "indices")
                     else self.band_path(None if path is True else path))
            output = np.asarray(self.kpoints, dtype=float)[kpath.indices]
        else:
            output = np.asarray(self.kpoints, dtype=float)

        if irreducible:
            symmetry = self.symmetry(symprec=symprec)
        else:
            # The identity alone: this still collapses exact duplicates, which
            # is a correctness-free saving, without asserting any symmetry.
            symmetry = SymmetryInfo(
                number=1, international="P1", point_group="1",
                rotations=np.eye(3, dtype=int)[None, :, :],
                translations=np.zeros((1, 3)), symprec=0.0)
        evaluation = irreducible_kpoints(
            output, symmetry,
            time_reversal=bool(time_reversal) and bool(irreducible))
        return output, evaluation, kpath, (evaluation if irreducible else None)

    def get_spectral_function(self, energies=None, eta: float = 0.2,
                              points: int = 400, atoms=None, path=None,
                              irreducible: bool = False,
                              symprec: float = None,
                              time_reversal: bool = True) -> SpectralFunction:
        r"""The interacting band structure: ``A(E, k)`` by Lehmann representation.

        This **carries the two-body term**: it is built from the correlated
        ground state the run produced, so the weight at a pole measures how
        much of a true one-particle excitation that pole is.  :meth:`bands`
        reads its poles, so the bands carry it too.
        For a single determinant it collapses to unit weights at the mean-field
        eigenvalues, which is the right limit.

        ``A_nu(k, E)`` is the removal branch plus the addition branch,

        .. math::

            A(k, E) = \sum_m |\langle m | c_{k} | \Psi_0 \rangle|^2
                        \delta(E - (E_0 - E_m))
                    + \sum_n |\langle n | c^\dagger_{k} | \Psi_0 \rangle|^2
                        \delta(E - (E_n - E_0)),

        with the ``N-1`` and ``N+1`` eigenstates obtained by diagonalizing the
        Hamiltonian in those particle-number sectors, summed over spin and over
        the orbitals of the primitive cell and broadened into Lorentzians of
        half-width ``eta``.

        Energies are measured **from the N-electron ground state**: a removal
        pole sits at ``E_0 - E_m`` (negative, the occupied side) and an addition
        pole at ``E_n - E_0`` (positive, the empty side), so the chemical
        potential falls between the two branches rather than at zero.

        **It is resolved only at the commensurate k-points** -- the
        ``n1 x n2 x n3`` mesh the supercell realizes, and no others: the Bloch
        combinations exist only there.  A finer band needs a bigger mesh, hence
        a bigger supercell and more qubits.

        Parameters
        ----------
        energies : array_like, optional
            Energy grid in eV.  By default it spans the poles with a margin of
            five broadening widths.
        eta : float
            Lorentzian broadening in eV.
        points : int
            Grid size when ``energies`` is not given.
        path : str or KPointPath, optional
            Restrict the output to the mesh k-points lying on a high-symmetry
            path -- ``"GXMG"``, or ``True`` for the lattice's own default path.
            In one dimension the mesh already *is* a line and this changes only
            the ordering and the axis; in two or three dimensions it is what
            makes a band plot possible at all.  Nothing is interpolated: see
            :mod:`mandacaru.algorithms.kpath`.
        irreducible : bool
            Evaluate only the symmetry-inequivalent k-points and spread the
            result over the rest.  The spectral function is invariant under the
            crystal's point group, so this is exact, not an approximation --
            and it is checked, since a full evaluation must reproduce it.  It
            saves Lehmann evaluations only; the qubit count is fixed by the
            mesh and does not change.
        symprec : float, optional
            Tolerance for the symmetry search (default
            :data:`~mandacaru.core.symmetry.DEFAULT_SYMPREC`).
        time_reversal : bool
            Include ``k -> -k`` among the symmetries.  True for the
            unpolarized states this path solves.

        Notes
        -----
        Exactly repeated k-points are evaluated once whatever ``irreducible``
        says: a path that returns to ``Gamma`` names it twice, and the second
        visit is the same operator on the same state.
        """
        from ..units import from_hartree

        # Cited here rather than from the configuration: whether a run reaches
        # the spectral function at all, and whether it uses the symmetry
        # reduction, is not knowable when the driver is built.
        self._cite("Lehmann1954")
        if irreducible:
            self._cite("Togo2018")
        if atoms is not None:
            self._primitive = atoms.copy()
        output, evaluation, kpath, zone = self._spectral_kpoints(
            path=path, irreducible=irreducible, symprec=symprec,
            time_reversal=time_reversal)
        psi, coefficients, reference = self._spectral_context()   # Ha
        hamiltonian = self._as_pauli_sum(self.hamiltonian, self.n_qubits)
        n_orbitals = coefficients.shape[0]
        per_cell = n_orbitals // self.n_supercells
        alpha, beta = self.num_particles

        all_poles, all_weights, deviations = [], [], []
        orbital_poles = []
        sectors: dict = {}          # one eigendecomposition per particle number
        audit = self._audit_point(evaluation) if irreducible else None
        eval_points = list(evaluation.points)
        if audit is not None:
            eval_points.append(output[audit[0]])

        for kpt in eval_points:
            per_orbital = []
            for nu in range(per_cell):
                poles_nu, weights_nu = [], []
                for spin in (0, 1):
                    step = (1, 0) if spin == 0 else (0, 1)
                    creation = self._bloch_creation(kpt, nu, spin, coefficients,
                                                    n_orbitals)
                    add_p, add_w = self._lehmann(
                        creation @ psi, (alpha + step[0], beta + step[1]),
                        hamiltonian, reference, addition=True, cache=sectors)
                    rem_p, rem_w = self._lehmann(
                        creation.conj().T @ psi,
                        (alpha - step[0], beta - step[1]),
                        hamiltonian, reference, addition=False, cache=sectors)
                    deviations.append(abs(add_w.sum() + rem_w.sum() - 1.0))
                    poles_nu.append(np.concatenate([rem_p, add_p]))
                    weights_nu.append(np.concatenate([rem_w, add_w]))
                per_orbital.append(
                    (from_hartree(np.concatenate(poles_nu), "eV"),
                     np.concatenate(weights_nu)))
            orbital_poles.append(per_orbital)
            all_poles.append(np.concatenate([p for p, _w in per_orbital]))
            all_weights.append(np.concatenate([w for _p, w in per_orbital]))

        # Audit the symmetry before using it: the wedge is only as good as
        # the state's invariance, which is a property of the state and not of
        # the lattice.
        symmetry_residual = None
        if audit is not None:
            index, slot = audit
            measured = all_weights.pop()
            all_poles.pop()
            orbital_poles.pop()
            scale = max(float(np.abs(all_weights[slot]).max()), 1e-30)
            symmetry_residual = float(
                np.abs(measured - all_weights[slot]).max() / scale)
            if zone is not None:
                zone.symmetry_residual = symmetry_residual
            if symmetry_residual > SYMMETRY_RESIDUAL_TOLERANCE:
                import warnings

                warnings.warn(
                    f"irreducible=True assumed A(E, k) is invariant under "
                    f"{evaluation.symmetry.international}, but k="
                    f"{np.round(output[index], 4).tolist()} disagrees with its "
                    f"representative "
                    f"{np.round(evaluation.points[slot], 4).tolist()} by "
                    f"{symmetry_residual:.2%} of the peak weight.  The lattice "
                    "has the symmetry; the state does not -- the usual cause "
                    "is a partially filled degenerate manifold (a metallic "
                    "mesh), where the reference picks one member of the "
                    "manifold and breaks the point group.  Re-run with "
                    "irreducible=False for this system.",
                    RuntimeWarning, stacklevel=2)

        # Spread the computed k-points over the reported ones.  With no
        # symmetry and no repeats this is the identity; with either, it is what
        # makes one evaluation serve several k-points.
        spread = evaluation.mapping
        orbital_poles = [orbital_poles[i] for i in spread]
        all_poles = [all_poles[i] for i in spread]
        all_weights = [all_weights[i] for i in spread]

        flat = np.concatenate(all_poles)
        keep = np.concatenate(all_weights) > 1e-12
        if energies is None:
            lo, hi = flat[keep].min(), flat[keep].max()
            energies = np.linspace(lo - 5 * eta, hi + 5 * eta, int(points))
        energies = np.asarray(energies, dtype=float)

        weights = np.empty((len(output), energies.size))
        for ik, (poles, amplitude) in enumerate(zip(all_poles, all_weights)):
            lorentz = (eta / np.pi) / ((energies[None, :] - poles[:, None]) ** 2
                                       + eta ** 2)
            weights[ik] = amplitude @ lorentz

        # The interacting chemical potential: between the last removal pole and
        # the first addition pole.  Poles are signed by branch, so the split is
        # at the reference energy itself.
        occupied = flat[keep & (flat <= 0.0)]
        empty = flat[keep & (flat > 0.0)]
        mu = 0.5 * ((occupied.max() if occupied.size else 0.0)
                    + (empty.min() if empty.size else 0.0))
        return SpectralFunction(
            energies=energies, kpoints=output,
            weights=weights,
            poles=[(p, w) for p, w in zip(all_poles, all_weights)],
            orbital_poles=orbital_poles,
            eta=float(eta), chemical_potential=float(mu),
            sum_rule=float(max(deviations)),
            kpath=kpath, irreducible=zone)


def _bloch_drivers():
    """The periodic driver classes, built over the molecular ones."""
    from .adapt_vqe import ADAPTVQE
    from .vqe import VQE

    class BlochVQE(_BlochMixin, VQE):
        """VQE on the Born-von Karman supercell (``method="bloch-vqe"``)."""

    class BlochADAPTVQE(_BlochMixin, ADAPTVQE):
        """ADAPT-VQE on the Born-von Karman supercell (``"bloch-adapt-vqe"``)."""

    return {"bloch-vqe": BlochVQE, "bloch-adapt-vqe": BlochADAPTVQE}


#: Message for the name this module used to export.
RETIRED_BLOCH_CLASS = (
    "BlochCalculator is no longer a public class: the periodic driver is "
    "reached through the single entry point, like every other method.  Use\n\n"
    "    atoms.calc = Mandacaru(method='bloch-adapt-vqe',\n"
    "                           kpts={'size': (2, 1, 1), 'gamma': True})\n"
    "    energy_per_cell = atoms.get_potential_energy()\n"
    "    bands = atoms.calc.bands(kline)\n"
    "    e_fermi = atoms.calc.get_fermi_level()\n\n"
    "(or method='bloch-vqe').  basis, h and mapping are Mandacaru options; "
    "the k-point mesh is kpts, which must be Gamma-centered.  n_cells and "
    "n_images are gone -- the lattice window is the supercell itself.")
