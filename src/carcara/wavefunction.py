# -*- coding: utf-8 -*-
# file: wavefunction.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Atomic system front-end.

``Wavefunction`` is now a thin *facade* over three decoupled pieces:

* :class:`carcara.basis.FullAtomicOrbital` -- the single source of truth for the
  hydrogen-like orbital math (previously duplicated in two methods here);
* :class:`carcara.integrals.Grid` -- the shared real-space integration grid;
* :class:`carcara.integrals.IntegralEngine` -- the basis-agnostic engine that
  samples any basis and dispatches the heavy integrals to the parallel C
  backend (with a NumPy fallback).

This class keeps its historical, tested API (coordinate conversions, orbital
evaluation, Coulomb potential, one-body integrals) but no longer implements the
physics inline: it prepares geometry/grids and delegates.  Swapping FAO
orbitals for Wannier functions is a matter of feeding different
:class:`~carcara.basis.base.BasisFunction` objects to the engine -- nothing in
the integral core changes.
"""

import numpy as np
from ase import Atoms
from ase.io import read

from .basis import FullAtomicOrbital
from .integrals import Grid, IntegralEngine

ANGSTROM_TO_BOHR = 1.8897259886


def _points_to_h(box_size, points):
    """Spacing ``h`` (Bohr) reproducing ``points`` nodes over ``[-box, box]``.

    ``Grid`` now takes a spacing rather than a node count; this keeps the legacy
    ``points`` argument of the methods below working by handing ``Grid`` the
    equivalent ``h`` (the round-trip recovers exactly ``points`` nodes).  This
    facade works throughout in atomic units, so the grids below are built with
    ``units="bohr"``.
    """
    return 2.0 * box_size / (points - 1)


class Wavefunction:
    def __init__(self, atoms=None, atom_index=0, xyz_filename=None):
        """Initialize the system from an ASE ``Atoms`` object or an XYZ file.

        Parameters
        ----------
        atoms : ase.Atoms or str, optional
            Either an ASE :class:`~ase.Atoms` object (elements, positions and,
            for a crystal, the cell are read directly from it) or a path to an
            ``.xyz`` file.  Passing a path here is equivalent to the historical
            ``Wavefunction("file.xyz")`` call.
        atom_index : int
            Index of the atom acting as the reference nucleus/origin.
        xyz_filename : str, optional
            Explicit XYZ path (kept for the legacy keyword form); used only when
            ``atoms`` is not given.

        The chemical symbols come from ``atoms.get_chemical_symbols()`` and the
        positions from ``atoms.get_positions()``; the unit cell, when the object
        carries one, is read from ``atoms.get_cell()`` and exposed as
        :attr:`cell` (Angstrom) / :attr:`cell_bohr` (Bohr).
        """
        if atoms is None and xyz_filename is not None:
            atoms = xyz_filename
        if atoms is None:
            raise ValueError("provide an ase.Atoms object or an XYZ file path")
        if isinstance(atoms, str):
            atoms = read(atoms)
        elif not isinstance(atoms, Atoms):
            raise TypeError(
                "atoms must be an ase.Atoms object or a path to an XYZ file, "
                f"got {type(atoms).__name__}")

        self._setup_from_atoms(atoms, atom_index)

        print(f"Loaded {self.n_atoms} atom(s). Reference nucleus: "
              f"'{self.all_symbols[atom_index]}' (index {atom_index}, Z={self.Z})."
              + (f" Periodic cell present." if self.has_cell else ""))

    @classmethod
    def from_ase(cls, atoms, atom_index=0):
        """Build a :class:`Wavefunction` directly from an ASE ``Atoms`` object.

        Extracts the chemical elements (``atoms.get_chemical_symbols()``),
        atomic positions (``atoms.get_positions()``) and, for a crystal, the
        cell (``atoms.get_cell()``) -- no XYZ round-trip or manual geometry
        parsing required.

        Parameters
        ----------
        atoms : ase.Atoms
            The molecule or crystal.
        atom_index : int
            Reference nucleus/origin index.
        """
        if not isinstance(atoms, Atoms):
            raise TypeError(f"expected an ase.Atoms object, got {type(atoms).__name__}")
        return cls(atoms=atoms, atom_index=atom_index)

    def _setup_from_atoms(self, atoms, atom_index):
        """Populate state from an ASE ``Atoms`` object (elements/positions/cell)."""
        self.atoms = atoms
        self.atom_index = atom_index
        self.n_atoms = len(atoms)

        # Elements and positions straight from the ASE object.
        symbols = atoms.get_chemical_symbols()
        positions = np.asarray(atoms.get_positions(), dtype=float)  # (N, 3) Angstrom

        self.all_symbols = list(symbols)                                  # length N
        self.all_numbers = np.asarray(atoms.get_atomic_numbers())         # (N,)
        self.all_positions_bohr = positions * ANGSTROM_TO_BOHR            # (N, 3) Bohr

        # Unit cell (crystals).  ASE always returns a (3, 3) Cell; it is the zero
        # matrix for a non-periodic molecule, in which case there is no cell.
        cell = np.asarray(atoms.get_cell(), dtype=float)                  # (3, 3) Angstrom
        self.has_cell = bool(np.any(cell))
        self.cell = cell if self.has_cell else None
        self.cell_bohr = cell * ANGSTROM_TO_BOHR if self.has_cell else None
        self.pbc = np.asarray(atoms.get_pbc(), dtype=bool)

        # Reference atom (nucleus/origin).
        self.origin_cart = self.all_positions_bohr[atom_index]
        self.Z = int(self.all_numbers[atom_index])

    def __repr__(self):
        cell = "cell" if self.has_cell else "no cell"
        return (f"Wavefunction(nucleus='{self.all_symbols[self.atom_index]}', "
                f"Z={self.Z}, origin_cart={self.origin_cart}, "
                f"n_atoms={self.n_atoms}, {cell})")

    # --- Grid construction from the (possibly non-cubic) cell ---------------

    def grid_from_cell(self, h=0.20, padding=0.0):
        """Build an integration :class:`~carcara.integrals.Grid` from the cell.

        Uses the ASE cell tensor (arbitrary, possibly non-orthogonal lattice
        vectors) so the grid spans the full crystal cell rather than an implicit
        cube.  Requires the ``Atoms`` object to carry a cell.

        Parameters
        ----------
        h : float
            Target grid spacing in Bohr.
        padding : float
            Extra half-extent (Bohr) added around the cell bounding box.

        Returns
        -------
        Grid
            A uniform-spacing grid centered on the cell, non-cubic when the cell
            is.
        """
        if not self.has_cell:
            raise ValueError("this system has no unit cell; use a cubic Grid "
                             "with an explicit box_size instead")
        cell_bohr = self.cell_bohr
        center = 0.5 * cell_bohr.sum(axis=0)          # geometric center of the cell
        cell = cell_bohr.copy()
        if padding:
            # Grow every lattice vector length by 2*padding along its own axis.
            extent = np.abs(cell).sum(axis=0)
            scale = (extent + 2.0 * padding) / np.where(extent > 0, extent, 1.0)
            cell = cell * scale[np.newaxis, :]
        return Grid(center=center, box_size=0.0, h=h, units="bohr", cell=cell)

    # --- Coordinate Conversion ---

    @staticmethod
    def spherical_to_cartesian(sph_array):
        """Converts [r, theta, phi] into [x, y, z]."""
        r, theta, phi = sph_array[0], sph_array[1], sph_array[2]
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.array([x, y, z])

    @staticmethod
    def cartesian_to_spherical(cart_array):
        """Converts [x, y, z] into [r, theta, phi]."""
        x, y, z = cart_array[0], cart_array[1], cart_array[2]
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(np.clip(z / (r + 1e-15), -1.0, 1.0))
        phi = np.arctan2(y, x)
        return np.array([r, theta, phi])

    # --- Orbital construction (single source of truth via FullAtomicOrbital) ---

    def orbital(self, state, center=None, Z=None):
        """Build a :class:`FullAtomicOrbital` for this system.

        Centralizes orbital creation so no radial/angular formula is duplicated.
        """
        center = self.origin_cart if center is None else np.asarray(center, float)
        Z = self.Z if Z is None else Z
        n, l, m = state
        # This facade keeps positions in Bohr (atomic units) throughout.
        return FullAtomicOrbital(n, l, m, Z=Z, center=center, units="bohr")

    def _psi_on_cart_grid(self, state, origin_cart, X_abs, Y_abs, Z_abs, Z_nuclear=None):
        """Evaluate a hydrogen-like wavefunction on a Cartesian coordinate grid."""
        return self.orbital(state, center=origin_cart, Z=Z_nuclear).evaluate(
            X_abs, Y_abs, Z_abs)

    def calculate_wavefunction(self, quantum_state, pos, origin=None):
        """
        Calculates the hydrogen-like wavefunction at a position in absolute spherical coords.

        quantum_state : [n, l, m]
        pos           : np.array [r, theta, phi] in absolute space
        origin        : np.array [r, theta, phi] override; defaults to self.origin_cart
        """
        if origin is None:
            nuclei_origin_cart = self.origin_cart
        else:
            nuclei_origin_cart = self.spherical_to_cartesian(origin)

        pos_cart = self.spherical_to_cartesian(pos)
        orb = self.orbital(quantum_state, center=nuclei_origin_cart, Z=self.Z)
        return orb.evaluate(pos_cart[0], pos_cart[1], pos_cart[2])

    def coulomb_potential(self, pos, origin=None):
        """Calculates the single-nucleus Coulomb potential V(r) = -Z / r."""
        if origin is None:
            nuclei_origin_cart = self.origin_cart
        else:
            nuclei_origin_cart = self.spherical_to_cartesian(origin)

        pos_cart = self.spherical_to_cartesian(pos)
        if len(pos_cart.shape) > 1:
            relative_cart = pos_cart - nuclei_origin_cart[:, np.newaxis, np.newaxis]
        else:
            relative_cart = pos_cart - nuclei_origin_cart

        r, _, _ = self.cartesian_to_spherical(relative_cart)
        r = np.where(r == 0, 1e-15, r)

        return -self.Z / r

    # --- Potentials on a Cartesian grid (fed to the integral engine) ---

    def electron_nuclear_potential(self, X, Y, Z):
        """Electron-nuclear Coulomb potential summed over all nuclei, on a grid.

        V(r) = sum_I (-Z_I / |r - R_I|).  This is exactly the kind of external
        potential the basis-agnostic engine consumes.
        """
        V = np.zeros(np.broadcast(X, Y, Z).shape, dtype=float)
        for i in range(self.n_atoms):
            Rx, Ry, Rz = self.all_positions_bohr[i]
            r = np.sqrt((X - Rx)**2 + (Y - Ry)**2 + (Z - Rz)**2)
            r = np.where(r < 1e-15, 1e-15, r)
            V -= self.all_numbers[i] / r
        return V

    # --- Integrals (prepared here, executed by the C backend) ---

    def integrate_potential_energy(self, quantum_state, origin=None, box_size=10, points=60):
        """Numerically integrates <psi | V | psi> over a 3D Cartesian grid."""
        center = self.origin_cart if origin is None \
            else self.spherical_to_cartesian(origin)
        grid = Grid(center=center, box_size=box_size,
                    h=_points_to_h(box_size, points), units="bohr")

        orb = self.orbital(quantum_state, center=center, Z=self.Z)
        psi = orb.evaluate(grid.X, grid.Y, grid.Z)
        r = np.sqrt((grid.X - center[0])**2 + (grid.Y - center[1])**2
                    + (grid.Z - center[2])**2)
        r = np.where(r < 1e-15, 1e-15, r)
        V = -self.Z / r

        integrand = np.conj(psi) * V * psi
        return float(np.real(np.sum(integrand) * grid.dV))

    def one_body_integral(self, state_a, state_b, origin_a=None, origin_b=None,
                          Z_a=None, Z_b=None, box_size=10, points=50):
        """
        Computes one-body Hamiltonian matrix elements via the integral engine.

        T_ab = <psi_a | -1/2 nabla^2 | psi_b>                (kinetic energy)
        V_ab = <psi_a | sum_I (-Z_I / |r - R_I|) | psi_b>    (electron-nuclear Coulomb)

        The heavy lifting (finite-difference Laplacian + grid reductions) runs in
        the parallel C backend when available, otherwise in the NumPy fallback.
        The Coulomb potential sums over all nuclei loaded from the XYZ file.

        Parameters:
        state_a, state_b : [n, l, m] quantum numbers for bra and ket orbitals
        origin_a, origin_b : Cartesian origins [x, y, z] in Bohr; None -> self.origin_cart
        Z_a, Z_b : nuclear charges for each orbital; None -> self.Z
        box_size : half-edge length of the integration cube in Bohr
        points : number of grid points per dimension

        Returns:
        dict with keys 'kinetic', 'potential', 'total'
        """
        center_a = self.origin_cart.copy() if origin_a is None else np.asarray(origin_a, float)
        center_b = self.origin_cart.copy() if origin_b is None else np.asarray(origin_b, float)
        Z_a = self.Z if Z_a is None else Z_a
        Z_b = self.Z if Z_b is None else Z_b

        # Grid centered midway between the two orbital centers.
        center = (center_a + center_b) / 2.0
        grid = Grid(center=center, box_size=box_size,
                    h=_points_to_h(box_size, points), units="bohr")

        basis = [self.orbital(state_a, center=center_a, Z=Z_a),
                 self.orbital(state_b, center=center_b, Z=Z_b)]
        engine = IntegralEngine(basis, grid)

        # Legacy API returns atomic units (Hartree).
        T, V = engine.one_body(self.electron_nuclear_potential, energy_units="Ha")
        T_ab = float(np.real(T[0, 1]))
        V_ab = float(np.real(V[0, 1]))
        return {'kinetic': T_ab, 'potential': V_ab, 'total': T_ab + V_ab}
