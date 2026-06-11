# -*- coding: utf-8 -*-
# file: wavefunction.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

import numpy as np
from scipy import special, integrate
from ase.io import read


class Wavefunction:
    def __init__(self, xyz_filename, atom_index=0):
        """
        Initializes the system by reading a molecular/atomic configuration from an XYZ file.

        Parameters:
        xyz_filename (str): Path to the .xyz file.
        atom_index (int): The index of the atom acting as the reference nucleus/origin.
        """
        angstrom_to_bohr = 1.8897259886

        self.atoms = read(xyz_filename)
        self.atom_index = atom_index
        self.n_atoms = len(self.atoms)

        # Data for all atoms (positions in Bohr)
        self.all_positions_bohr = self.atoms.positions * angstrom_to_bohr  # shape (N, 3)
        self.all_numbers = self.atoms.numbers                               # shape (N,)
        self.all_symbols = list(self.atoms.symbols)                         # length N

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

    # --- Core Physics ---

    def _psi_on_cart_grid(self, state, origin_cart, X_abs, Y_abs, Z_abs, Z_nuclear=None):
        """Evaluates a hydrogen-like wavefunction on a Cartesian coordinate grid."""
        if Z_nuclear is None:
            Z_nuclear = self.Z

        X_rel = X_abs - origin_cart[0]
        Y_rel = Y_abs - origin_cart[1]
        Z_rel = Z_abs - origin_cart[2]

        r = np.sqrt(X_rel**2 + Y_rel**2 + Z_rel**2)
        r = np.where(r < 1e-15, 1e-15, r)
        theta = np.arccos(np.clip(Z_rel / r, -1.0, 1.0))
        phi = np.arctan2(Y_rel, X_rel)

        n, l, m = state
        a0 = 1.0
        rho = (2 * Z_nuclear * r) / (n * a0)

        num = special.factorial(n - l - 1)
        den = 2 * n * special.factorial(n + l)
        radial_norm = np.sqrt((2 * Z_nuclear / (n * a0))**3 * num / den)

        laguerre = special.genlaguerre(n - l - 1, 2 * l + 1)
        R_nl = radial_norm * np.exp(-rho / 2) * (rho ** l) * laguerre(rho)

        Y_lm = special.sph_harm_y(l, m, theta, phi)

        return R_nl * Y_lm

    def calculate_wavefunction(self, quantum_state, pos, origin=None):
        """
        Calculates the hydrogen-like wavefunction at a position in absolute spherical coords.

        quantum_state : [n, l, m]
        pos           : np.array [r, theta, phi] in absolute space
        origin        : np.array [r, theta, phi] override; defaults to self.origin_cart
        """
        n, l, m = quantum_state

        if origin is None:
            nuclei_origin_cart = self.origin_cart
        else:
            nuclei_origin_cart = self.spherical_to_cartesian(origin)

        pos_cart = self.spherical_to_cartesian(pos)
        if len(pos_cart.shape) > 1:
            relative_cart = pos_cart - nuclei_origin_cart[:, np.newaxis, np.newaxis]
        else:
            relative_cart = pos_cart - nuclei_origin_cart

        r, theta, phi = self.cartesian_to_spherical(relative_cart)
        r = np.where(r == 0, 1e-15, r)

        a0 = 1.0
        rho = (2 * self.Z * r) / (n * a0)

        num = special.factorial(n - l - 1)
        den = 2 * n * special.factorial(n + l)
        radial_norm = np.sqrt((2 * self.Z / (n * a0))**3 * num / den)

        laguerre = special.genlaguerre(n - l - 1, 2 * l + 1)
        R_nl = radial_norm * np.exp(-rho / 2) * (rho ** l) * laguerre(rho)

        Y_lm = special.sph_harm_y(l, m, theta, phi)

        return R_nl * Y_lm

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

    def integrate_potential_energy(self, quantum_state, origin=None, box_size=10, points=60):
        """Numerically integrates <psi | V | psi> over a 3D Cartesian grid."""
        if origin is None:
            center_cart = self.origin_cart
        else:
            center_cart = self.spherical_to_cartesian(origin)

        grid_1d = np.linspace(-box_size, box_size, points)
        dx = grid_1d[1] - grid_1d[0]
        dV = dx**3

        X_rel, Y_rel, Z_rel = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')

        X_abs = X_rel + center_cart[0]
        Y_abs = Y_rel + center_cart[1]
        Z_abs = Z_rel + center_cart[2]

        r_abs = np.sqrt(X_abs**2 + Y_abs**2 + Z_abs**2)
        theta_abs = np.arccos(np.clip(Z_abs / (r_abs + 1e-15), -1.0, 1.0))
        phi_abs = np.arctan2(Y_abs, X_abs)

        pos_grid = np.array([r_abs, theta_abs, phi_abs])

        psi = self.calculate_wavefunction(quantum_state, pos_grid, origin)
        V = self.coulomb_potential(pos_grid, origin)

        integrand = np.conj(psi) * V * psi
        return np.real(np.sum(integrand) * dV)

    def one_body_integral(self, state_a, state_b, origin_a=None, origin_b=None,
                          Z_a=None, Z_b=None, box_size=10, points=50):
        """
        Computes one-body Hamiltonian matrix elements via numerical integration.

        T_ab = <psi_a | -1/2 nabla^2 | psi_b>                (kinetic energy)
        V_ab = <psi_a | sum_I (-Z_I / |r - R_I|) | psi_b>    (electron-nuclear Coulomb)

        The Laplacian of psi_b is evaluated numerically using 6-point finite differences.
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
        center_a = self.origin_cart.copy() if origin_a is None else np.asarray(origin_a, dtype=float)
        center_b = self.origin_cart.copy() if origin_b is None else np.asarray(origin_b, dtype=float)

        if Z_a is None:
            Z_a = self.Z
        if Z_b is None:
            Z_b = self.Z

        # Place the grid midway between the two orbital centers
        center = (center_a + center_b) / 2.0
        grid_1d = np.linspace(-box_size, box_size, points)
        dx = grid_1d[1] - grid_1d[0]
        dV = dx**3

        X_rel, Y_rel, Z_rel = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
        X_abs = X_rel + center[0]
        Y_abs = Y_rel + center[1]
        Z_abs = Z_rel + center[2]

        psi_a = self._psi_on_cart_grid(state_a, center_a, X_abs, Y_abs, Z_abs, Z_a)
        psi_b = self._psi_on_cart_grid(state_b, center_b, X_abs, Y_abs, Z_abs, Z_b)

        # Laplacian of psi_b via central finite differences
        lap_psi_b = (
            self._psi_on_cart_grid(state_b, center_b, X_abs + dx, Y_abs,      Z_abs,      Z_b)
          + self._psi_on_cart_grid(state_b, center_b, X_abs - dx, Y_abs,      Z_abs,      Z_b)
          + self._psi_on_cart_grid(state_b, center_b, X_abs,      Y_abs + dx, Z_abs,      Z_b)
          + self._psi_on_cart_grid(state_b, center_b, X_abs,      Y_abs - dx, Z_abs,      Z_b)
          + self._psi_on_cart_grid(state_b, center_b, X_abs,      Y_abs,      Z_abs + dx, Z_b)
          + self._psi_on_cart_grid(state_b, center_b, X_abs,      Y_abs,      Z_abs - dx, Z_b)
          - 6.0 * psi_b
        ) / dx**2

        T_ab = np.real(np.sum(np.conj(psi_a) * (-0.5 * lap_psi_b)) * dV)

        # Electron-nuclear Coulomb potential summed over all nuclei
        V_en = np.zeros_like(X_abs, dtype=float)
        for i in range(self.n_atoms):
            Rx, Ry, Rz = self.all_positions_bohr[i]
            r_eI = np.sqrt((X_abs - Rx)**2 + (Y_abs - Ry)**2 + (Z_abs - Rz)**2)
            r_eI = np.where(r_eI < 1e-15, 1e-15, r_eI)
            V_en -= self.all_numbers[i] / r_eI

        V_ab = np.real(np.sum(np.conj(psi_a) * V_en * psi_b) * dV)

        return {'kinetic': T_ab, 'potential': V_ab, 'total': T_ab + V_ab}
