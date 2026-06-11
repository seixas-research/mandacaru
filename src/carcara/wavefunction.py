# -*- coding: utf-8 -*-
# file: integrals.py

# This code is part of Carcará. 
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br> 

import numpy as np
# import scipy.special as spe
from scipy import special, integrate
from ase.io import read


class Wavefunction:
    def __init__(self, xyz_filename, atom_index=0):
        """
        Initializes the system by reading a molecular/atomic configuration from an XYZ file.
        
        Parameters:
        xyz_filename (str): Path to the .xyz file.
        atom_index (int): The index of the atom in the file acting as the nucleus/origin.
        """
        # Read the structure using ASE
        self.atoms = read(xyz_filename)
        
        # Get the Cartesian coordinates of the target ion (nucleus)
        # ASE stores positions in Angstroms by default; we convert to Bohr (Atomic Units)
        angstrom_to_bohr = 1.8897259886
        self.origin_cart = self.atoms.positions[atom_index] * angstrom_to_bohr
        
        # Determine the atomic number Z dynamically from the chosen atom
        self.Z = self.atoms.numbers[atom_index]
        
        print(f"Loaded atom '{self.atoms.symbols[atom_index]}' at index {atom_index} as the nucleus.")
        print(f"Origin (Bohr): {self.origin_cart}")
        print(f"Atomic Number (Z): {self.Z}")

    # --- Coordinate Conversion Helper Methods ---
    @staticmethod
    def spherical_to_cartesian(sph_array):
        """Converts [r, theta, phi] array into [x, y, z] array."""
        r, theta, phi = sph_array[0], sph_array[1], sph_array[2]
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.array([x, y, z])

    @staticmethod
    def cartesian_to_spherical(cart_array):
        """Converts [x, y, z] array into [r, theta, phi] array."""
        x, y, z = cart_array[0], cart_array[1], cart_array[2]
        r = np.sqrt(x**2 + y**2 + z**2)
        # Avoid division by zero at origin
        theta = np.arccos(np.clip(z / (r + 1e-15), -1.0, 1.0))
        phi = np.arctan2(y, x)
        return np.array([r, theta, phi])

    # --- Core Physics Methods ---
    def calculate_wavefunction(self, quantum_state, pos, origin=None):
        """
        Calculates the hydrogen-like atom wavefunction value at a given position vector.
        
        quantum_state : list or tuple -> [n, l, m]
        pos           : np.array -> [r, theta, phi] relative to absolute space
        origin        : np.array -> Optional override for the origin in [r, theta, phi]. 
                                    Defaults to the class instance's loaded ASE nucleus origin.
        """
        n, l, m = quantum_state
        
        # 1. Determine local origin
        if origin is None:
            nuclei_origin_cart = self.origin_cart
        else:
            nuclei_origin_cart = self.spherical_to_cartesian(origin)
            
        # 2. Shift coordinates relative to the nucleus
        pos_cart = self.spherical_to_cartesian(pos)
        relative_cart = pos_cart - nuclei_origin_cart[:, np.newaxis, np.newaxis] if len(pos_cart.shape) > 1 else pos_cart - nuclei_origin_cart
        
        r, theta, phi = self.cartesian_to_spherical(relative_cart)
        
        # Guard against singularity at exact center
        r = np.where(r == 0, 1e-15, r)

        # 3. Radial Component Evaluation
        a0 = 1.0 
        rho = (2 * self.Z * r) / (n * a0)
        
        num = special.factorial(n - l - 1)
        den = 2 * n * special.factorial(n + l)
        radial_norm = np.sqrt((2 * self.Z / (n * a0))**3 * num / den)
        
        laguerre = special.genlaguerre(n - l - 1, 2 * l + 1)
        R_nl = radial_norm * np.exp(-rho / 2) * (rho ** l) * laguerre(rho)
        
        # 4. Angular Component Evaluation
        Y_lm = special.sph_harm(m, l, phi, theta)
        
        return R_nl * Y_lm


    def coulomb_potential(self, pos, origin=None):
        """Calculates the Coulomb potential energy V(r) = -Z / r."""
        if origin is None:
            nuclei_origin_cart = self.origin_cart
        else:
            nuclei_origin_cart = self.spherical_to_cartesian(origin)
            
        pos_cart = self.spherical_to_cartesian(pos)
        relative_cart = pos_cart - nuclei_origin_cart[:, np.newaxis, np.newaxis] if len(pos_cart.shape) > 1 else pos_cart - nuclei_origin_cart
        
        r, _, _ = self.cartesian_to_spherical(relative_cart)
        r = np.where(r == 0, 1e-15, r)
        
        return -self.Z / r

    def integrate_potential_energy(self, quantum_state, origin=None, box_size=10, points=60):
        """
        Integrates over a 3D grid in space, centered around the defined origin.
        """
        # Resolve center of grid in Cartesian space
        if origin is None:
            center_cart = self.origin_cart
        else:
            center_cart = self.spherical_to_cartesian(origin)

        # Build Cartesian Grid relative to the chosen center
        grid_1d = np.linspace(-box_size, box_size, points)
        dx = grid_1d[1] - grid_1d[0]
        dV = dx ** 3 
        
        X_rel, Y_rel, Z_rel = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
        
        # Shift back to absolute space coordinates to feed into class functions
        X_abs = X_rel + center_cart[0]
        Y_abs = Y_rel + center_cart[1]
        Z_abs = Z_rel + center_cart[2]
        
        # Convert absolute grid points to absolute spherical points
        r_abs = np.sqrt(X_abs**2 + Y_abs**2 + Z_abs**2)
        theta_abs = np.arccos(np.clip(Z_abs / (r_abs + 1e-15), -1.0, 1.0))
        phi_abs = np.arctan2(Y_abs, X_abs)
        
        pos_grid = np.array([r_abs, theta_abs, phi_abs])
        
        # Compute integrand matrix elements
        psi = self.calculate_wavefunction(quantum_state, pos_grid, origin)
        V = self.coulomb_potential(pos_grid, origin)
        
        integrand = np.conj(psi) * V * psi
        total_integral = np.sum(integrand) * dV
        
        return np.real(total_integral)