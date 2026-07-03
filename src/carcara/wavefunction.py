# -*- coding: utf-8 -*-
# file: wavefunction.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Atomic system front-end.

``Wavefunction`` is now a thin *facade* over three decoupled pieces:

* :class:`carcara.basis.HydrogenicOrbital` -- the single source of truth for the
  hydrogen-like orbital math (previously duplicated in two methods here);
* :class:`carcara.integrals.Grid` -- the shared real-space integration grid;
* :class:`carcara.integrals.IntegralEngine` -- the basis-agnostic engine that
  samples any basis and dispatches the heavy integrals to the parallel C
  backend (with a NumPy fallback).

This class keeps its historical, tested API (coordinate conversions, orbital
evaluation, Coulomb potential, one-body integrals) but no longer implements the
physics inline: it prepares geometry/grids and delegates.  Swapping hydrogenic
orbitals for Wannier functions is a matter of feeding different
:class:`~carcara.basis.base.BasisFunction` objects to the engine -- nothing in
the integral core changes.
"""

import numpy as np
from ase.io import read

from .basis import HydrogenicOrbital
from .integrals import Grid, IntegralEngine

ANGSTROM_TO_BOHR = 1.8897259886


class Wavefunction:
    def __init__(self, xyz_filename, atom_index=0):
        """
        Initializes the system by reading a molecular/atomic configuration from an XYZ file.

        Parameters:
        xyz_filename (str): Path to the .xyz file.
        atom_index (int): The index of the atom acting as the reference nucleus/origin.
        """
        self.atoms = read(xyz_filename)
        self.atom_index = atom_index
        self.n_atoms = len(self.atoms)

        # Data for all atoms (positions in Bohr)
        self.all_positions_bohr = self.atoms.positions * ANGSTROM_TO_BOHR  # (N, 3)
        self.all_numbers = self.atoms.numbers                              # (N,)
        self.all_symbols = list(self.atoms.symbols)                        # length N

        # Reference atom (nucleus/origin)
        self.origin_cart = self.all_positions_bohr[atom_index]
        self.Z = int(self.all_numbers[atom_index])

        print(f"Loaded {self.n_atoms} atom(s). Reference nucleus: "
              f"'{self.all_symbols[atom_index]}' (index {atom_index}, Z={self.Z}).")

    def __repr__(self):
        return (f"Wavefunction(nucleus='{self.all_symbols[self.atom_index]}', "
                f"Z={self.Z}, origin_cart={self.origin_cart}, n_atoms={self.n_atoms})")

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

    # --- Orbital construction (single source of truth via HydrogenicOrbital) ---

    def orbital(self, state, center=None, Z=None):
        """Build a :class:`HydrogenicOrbital` for this system.

        Centralizes orbital creation so no radial/angular formula is duplicated.
        """
        center = self.origin_cart if center is None else np.asarray(center, float)
        Z = self.Z if Z is None else Z
        n, l, m = state
        return HydrogenicOrbital(n, l, m, Z=Z, center=center)

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
        grid = Grid(center=center, box_size=box_size, points=points)

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
        grid = Grid(center=center, box_size=box_size, points=points)

        basis = [self.orbital(state_a, center=center_a, Z=Z_a),
                 self.orbital(state_b, center=center_b, Z=Z_b)]
        engine = IntegralEngine(basis, grid)

        T, V = engine.one_body(self.electron_nuclear_potential)
        T_ab = float(np.real(T[0, 1]))
        V_ab = float(np.real(V[0, 1]))
        return {'kinetic': T_ab, 'potential': V_ab, 'total': T_ab + V_ab}
