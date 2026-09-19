# -*- coding: utf-8 -*-
# file: algorithms/bloch.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Bloch / k-point variational eigensolver for periodic systems (1-, 2-, 3-D).

:class:`BlochCalculator` wraps the localized-basis integral engine and the
molecular :class:`~mandacaru.algorithms.calculator.Mandacaru` to treat a
**periodic system** given as an ASE ``Atoms`` primitive cell (its ``cell`` and
``pbc`` define the lattice and which directions are periodic).  Like the
molecular calculator, the variational eigensolver is selected by the ``method``
argument (``"adapt-vqe"`` by default, or ``"vqe"``).  It provides the two
things a periodic calculation needs:

* **Band structure** -- the single-particle Bloch Hamiltonian
  ``H(k) = sum_R e^{i k.R} h^{(R)}``, ``S(k) = sum_R e^{i k.R} S^{(R)}`` is built
  from the real-space cell-to-cell one-body / overlap blocks ``h^{(R)}``,
  ``S^{(R)}`` and the generalized eigenproblem ``H(k) c = eps(k) S(k) c`` is solved
  at each k-point (:meth:`~BlochCalculator.bands`,
  :meth:`~BlochCalculator.band_structure`).  This is fully general in the
  lattice dimension: the Bloch phase uses the fractional coordinates
  ``e^{2 pi i k.n}`` over the integer lattice translations ``n`` of the periodic
  directions.  Being single-particle, the band structure is *independent of the
  variational method*.

* **Total energy using all k-points** -- a correlated solver cannot be run per
  k-point and summed (the two-electron interaction couples crystal momenta), so
  :meth:`~BlochCalculator.total_energy` uses the **Born-von Karman
  equivalence**: an ``(n1, n2, n3)`` Monkhorst-Pack mesh is a Gamma-point
  calculation on the ``(n1, n2, n3)`` supercell, and the energy per cell is
  ``E(supercell) / n_cells``.  The supercell is built with
  :meth:`ase.Atoms.repeat` -- its cell is the primitive cell repeated
  ``kmesh`` times, with **no extra padding** -- and run through the molecular
  calculator with the selected ``method`` (the box is the supercell's own
  ``cell``).  So the primitive cell is what you size: its extent along the
  non-periodic directions is the empty space the supercell gets.  This is a
  finite-supercell estimate that converges to the periodic total energy as the
  mesh is refined -- exact in the infinite-mesh limit.

Only the one-body and overlap integrals are needed for the bands (the FFT
two-body path is orthogonal-grid only), so :meth:`~BlochCalculator.bands`
works for non-orthogonal 2-D/3-D cells as well; the total energy uses the full
(two-body-inclusive) molecular calculator on the supercell.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass
class BandStructure:
    """Band energies along a k-path (the return value of :meth:`BlochCalculator.band_structure`).

    Attributes
    ----------
    x : ndarray, shape (nk,)
        Cumulative k-distance along the path (for the plot x-axis).
    energies : ndarray, shape (nk, nbands)
        Band energies in **eV**, sorted ascending at each k-point.
    xticks : ndarray
        x positions of the high-symmetry points.
    labels : list of str
        High-symmetry-point labels (e.g. ``["G", "X"]``).
    kpts : ndarray, shape (nk, 3)
        Fractional coordinates of the path k-points.
    """

    x: np.ndarray
    energies: np.ndarray
    xticks: np.ndarray
    labels: list
    kpts: np.ndarray


class BlochCalculator:
    """K-point variational calculator for a periodic system.

    The variational eigensolver that solves the Born-von Karman supercell in
    :meth:`total_energy` is selected by ``method``, exactly as for
    :class:`~mandacaru.algorithms.calculator.Mandacaru`; the Bloch
    Hamiltonian, band structure and supercell construction are method-independent.

    Parameters
    ----------
    atoms : ase.Atoms
        The **primitive cell** of the periodic system.  ``atoms.cell`` sets the
        lattice vectors and ``atoms.pbc`` selects the periodic directions (at
        least one must be periodic).  The cell is also the only source of
        padding: along the non-periodic directions its length is the empty
        space of the band-structure grid and of the Born-von Karman
        supercell, so size it in the geometry (e.g. a 1-D chain in a
        ``[a, 10, 10]`` cell).
    method : str
        Which variational eigensolver evaluates the supercell energy --
        ``"adapt-vqe"`` (default) or ``"vqe"`` (any method accepted by
        :class:`~mandacaru.algorithms.calculator.Mandacaru`).
    basis : str or dict
        Localized basis passed to :class:`~mandacaru.basis.BasisSet` and to the
        molecular calculator (default ``"FAO"``).
    mapping : str
        Fermion-to-qubit mapping (default ``"jordan_wigner"``).
    n_cells : int
        Half-width (in cells) of the lattice-translation range kept in the Bloch
        sum -- blocks ``h^{(R)}``, ``S^{(R)}`` for ``|n_i| <= n_cells`` on each
        periodic axis (default ``4``).
    n_images : int
        Half-width (in cells) of the nuclei window used to build the crystal
        (periodic) potential ``V`` for the one-body integrals; must be at least
        ``n_cells`` (default ``7``).
    h : float
        Real-space grid spacing in **Angstrom** for the band-structure integrals
        (default ``0.20``).
    """

    def __init__(self, atoms, method: str = "adapt-vqe", basis="FAO",
                 mapping: str = "jordan_wigner",
                 n_cells: int = 4, n_images: int = 7, h: float = 0.20):
        from .calculator import resolve_method
        self.method, _ = resolve_method(method)     # raises on unknown method
        self.atoms = atoms.copy()
        self.basis = basis
        self.mapping = mapping
        self.n_cells = int(n_cells)
        self.n_images = max(int(n_images), int(n_cells))
        self.h = float(h)
        self.periodic = [i for i in range(3) if bool(self.atoms.pbc[i])]
        if not self.periodic:
            raise ValueError(
                f"{type(self).__name__} needs at least one periodic direction; "
                "set atoms.pbc (e.g. pbc=[True, False, False] for a 1-D chain).")
        lengths = np.asarray(self.atoms.cell.lengths(), dtype=float)
        if np.any(lengths <= 0.0):
            raise ValueError(
                "every lattice vector of the primitive cell must have a "
                "non-zero length: the cell is the box along the non-periodic "
                "directions as well (e.g. cell=[a, 10, 10] for a 1-D chain); "
                f"got lengths {lengths.round(3).tolist()} Angstrom")
        self._blocks = None          # {n_tuple: (S_block, h_block)}
        self._nbands = None

    # -- lattice bookkeeping ---------------------------------------------- #
    def _cells(self, radius):
        """Integer lattice-translation tuples with ``|n_i| <= radius`` (0 on open axes)."""
        axes = [range(-radius, radius + 1) if i in self.periodic else [0]
                for i in range(3)]
        return [tuple(n) for n in product(*axes)]

    @property
    def dimension(self) -> int:
        """Number of periodic directions (1, 2 or 3)."""
        return len(self.periodic)

    @property
    def n_bands(self) -> int:
        """Number of Bloch bands = spatial orbitals per primitive cell."""
        if self._nbands is None:
            self._compute_blocks()
        return self._nbands

    # -- real-space integrals -> cell-to-cell Bloch blocks ---------------- #
    def _compute_blocks(self):
        """Compute and cache the overlap / one-body blocks ``{n: (S^n, h^n)}`` (Hartree)."""
        from ..basis import BasisSet
        from ..core import MolecularIntegrals
        from ..integrals import Grid

        cell = np.array(self.atoms.cell)                  # rows a1, a2, a3 (Angstrom)
        symbols = self.atoms.get_chemical_symbols()
        numbers = self.atoms.get_atomic_numbers()
        prim = self.atoms.get_positions()
        from ._hamiltonian_from_atoms import (PER_ELEMENT, per_element_basis,
                                              resolve_basis)
        name, options = resolve_basis(self.basis)
        if name == PER_ELEMENT:
            per_element_basis(options, symbols)  # validate for these symbols
            bset = BasisSet.build(options)
        else:
            bset = BasisSet.build(name, **options)

        # Orbitals live on the block cells (|n| <= n_cells); the crystal potential
        # is built from a wider nuclei window (|n| <= n_images).
        block_cells = self._cells(self.n_cells)
        basis_fns, cell_slice = [], {}
        for n in block_cells:
            shift = np.array(n) @ cell
            start = len(basis_fns)
            for sym, p in zip(symbols, prim):
                basis_fns += bset.atom(sym, center=p + shift, units="angstrom")
            cell_slice[n] = slice(start, len(basis_fns))

        nuclei = []
        for n in self._cells(self.n_images):
            shift = np.array(n) @ cell
            for Z, p in zip(numbers, prim):
                nuclei.append((float(Z), p + shift))

        # The box is the nuclei window (|n| <= n_images along every periodic
        # axis -- the outermost orbital cells sit n_images - n_cells cells
        # from the edge) and the primitive cell itself along the open axes:
        # the primitive cell is the only source of empty space.
        pts = np.array([p + np.array(n) @ cell
                        for n in block_cells for p in prim])
        window = np.array([2 * self.n_images + 1 if i in self.periodic else 1
                           for i in range(3)], dtype=float)
        grid = Grid(center=pts.mean(axis=0), box_size=0.0, h=self.h,
                    units="angstrom", cell=window[:, None] * cell)
        softening = 0.5 * min(grid.dx, grid.dy, grid.dz)
        integrals = MolecularIntegrals(nuclei, basis_fns, grid, units="angstrom",
                                       orthogonalize=False, softening=softening)
        S = integrals.overlap()
        h = integrals.one_body()                          # Hartree
        central = cell_slice[(0, 0, 0)]
        self._blocks = {n: (S[central, sl], h[central, sl])
                        for n, sl in cell_slice.items()}
        self._nbands = central.stop - central.start

    # -- Bloch Hamiltonian and bands -------------------------------------- #
    def bloch_hamiltonian(self, kpt):
        """Bloch matrices ``H(k), S(k)`` (Hartree) at fractional k-point ``kpt``.

        ``kpt`` is a length-3 vector of **fractional** (reduced) coordinates; the
        Bloch phase for lattice translation ``n`` is ``e^{2 pi i k.n}``.
        """
        if self._blocks is None:
            self._compute_blocks()
        kpt = np.asarray(kpt, dtype=float)
        M = self._nbands
        H = np.zeros((M, M), dtype=complex)
        S = np.zeros((M, M), dtype=complex)
        for n, (S_n, h_n) in self._blocks.items():
            phase = np.exp(2j * np.pi * float(np.dot(kpt, n)))
            H += phase * h_n
            S += phase * S_n
        return 0.5 * (H + H.conj().T), 0.5 * (S + S.conj().T)

    def bands(self, kpts) -> np.ndarray:
        """Band energies (eV) at the given **fractional** k-points.

        ``kpts`` is a single length-3 k-point or an ``(nk, 3)`` array; returns an
        ``(nk, n_bands)`` array of ascending band energies in eV (solving the
        generalized eigenproblem ``H(k) c = eps S(k) c`` at each k-point).
        """
        from scipy.linalg import eigh

        from ..units import from_hartree

        kpts = np.atleast_2d(np.asarray(kpts, dtype=float))
        out = np.empty((len(kpts), self.n_bands))
        for i, k in enumerate(kpts):
            H, S = self.bloch_hamiltonian(k)
            out[i] = np.sort(eigh(H, S, eigvals_only=True).real)
        return from_hartree(out, "eV")

    def band_structure(self, path=None, npoints: int = 200) -> BandStructure:
        """Band structure along an ASE ``bandpath`` (default: the lattice's own path).

        ``path`` is an ASE band-path string (e.g. ``"GXG"``); ``None`` lets ASE pick
        the default high-symmetry path for the lattice.  Returns a
        :class:`BandStructure` (cumulative k-distance, band energies, ticks/labels).
        """
        bp = self.atoms.cell.bandpath(path, npoints=npoints, pbc=self.atoms.pbc)
        energies = self.bands(bp.kpts)
        x, xticks, labels = bp.get_linear_kpoint_axis()
        return BandStructure(x=x, energies=energies, xticks=xticks,
                             labels=labels, kpts=bp.kpts)

    def monkhorst_pack(self, size, gamma: bool = True) -> np.ndarray:
        """Fractional Monkhorst-Pack mesh via ASE (Gamma-centered when ``gamma``)."""
        from ase.dft.kpoints import monkhorst_pack

        size = tuple(int(s) for s in size)
        mesh = monkhorst_pack(size)
        if gamma:
            mesh = mesh + np.array([0.5 / s if s % 2 == 0 else 0.0 for s in size])
        return mesh

    # -- total energy using all k-points (Born-von Karman supercell) ------ #
    def supercell(self, kmesh):
        """The ``kmesh`` Born-von Karman supercell as an ASE ``Atoms``.

        Built with :meth:`ase.Atoms.repeat`: the cell is the primitive cell
        repeated ``kmesh`` times along each axis and nothing else -- no
        padding is added, so the primitive cell's extent along the
        non-periodic directions is the empty space the supercell carries.  The
        molecular calculator grids that cell, centered on the atoms.
        """
        return self.atoms.repeat(tuple(int(k) for k in kmesh))

    def total_energy(self, kmesh, **solver_kwargs):
        """Total energy **per cell** (eV) using all ``kmesh`` k-points.

        Runs a molecular :class:`~mandacaru.algorithms.calculator.Mandacaru`
        with the selected ``method`` on the ``kmesh`` Born-von Karman supercell
        (box = the supercell's ``cell``, i.e. the primitive cell repeated --
        see :meth:`supercell`) and divides by the number of cells.
        Extra keyword arguments (``h``, ``pool``, ``max_iterations``,
        ``temperature``, ...) are forwarded to the solver; which arguments are
        valid depends on the selected method.

        Returns ``(energy_per_cell_eV, result)`` where ``result`` is the
        method's result object (``ADAPTVQEResult`` / ``VQEResult``).
        """
        from .calculator import Mandacaru

        atoms = self.supercell(kmesh)
        options = dict(basis=self.basis, mapping=self.mapping, h=self.h)
        options.update(solver_kwargs)
        atoms.calc = Mandacaru(method=self.method,
                               **options)
        energy = atoms.get_total_energy()                 # eV
        n_cells = int(np.prod([int(k) for k in kmesh]))
        return energy / n_cells, atoms.calc.result

