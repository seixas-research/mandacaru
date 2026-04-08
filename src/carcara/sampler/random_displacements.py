# -*- coding: utf-8 -*-
# file: random_displacements.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br> 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
from typing import Optional, List, Literal, Union, Dict, Type
from ase import Atoms
from ase.io import write
from ase.optimize import BFGS, LBFGS, FIRE
from ase.filters import UnitCellFilter
from ase.parallel import parprint as print

class RandomDisplacements:
    """
    Generates a dataset of atomic structures with controlled noise applied to positions and cell parameters.

    Parameters:
    ===========
    - atoms: Base Atoms object to generate samples from.
    - calculator: Optional ASE calculator for computing reference energies and forces.
    - noise_type: Type of noise to apply ('normal' or 'uniform').
    - seed: Random seed for reproducibility.

    Methods:
    ========
    - relax_structure: Optimizes the base structure using a specified algorithm.
    - generate_samples: Creates multiple noisy samples based on the relaxed structure.
    - save_to_xyz: Saves the generated samples to an XYZ file, optionally including reference energies and forces.
    """
    
    # Optimizer options for structure relaxation
    _OPTIMIZERS: Dict[str, Type] = {
        "BFGS": BFGS,
        "LBFGS": LBFGS,
        "FIRE": FIRE
    }

    def __init__(
        self,
        atoms: Atoms,
        calculator: Optional = None,
        noise_type: Literal['normal', 'uniform'] = 'normal',
        seed: int = 42
    ):
        if noise_type not in ['normal', 'uniform']:
            raise ValueError(f"Invalid noise type '{noise_type}'. Use 'normal' or 'uniform'.")
        
        self.atoms = atoms.copy()
        self.calculator = calculator
        self.noise_type = noise_type
        self.rng = np.random.default_rng(seed)
        self.samples: List[Atoms] = []
        
        if self.calculator:
            self.atoms.calc = self.calculator

    def relax_structure(
        self, 
        fmax: float = 0.01, 
        relax_cell: bool = False, 
        algorithm: str = 'BFGS',
        cell_mask: List[int] = [1, 1, 1, 1, 1, 1]
    ) -> Atoms:
        """
        Relaxes the structure using the specified optimization algorithm.
        Parameters:
        - fmax: Maximum force criterion for convergence.
        - relax_cell: Whether to allow cell relaxation.
        - algorithm: Optimization algorithm to use ('BFGS', 'LBFGS', 'FIRE').
        - cell_mask: Mask for cell relaxation (1 to relax, 0 to fix) in the order [a, b, c, alpha, beta, gamma].
        """
        if not self.calculator:
            raise ValueError("Calculator is required for structure relaxation.")

        target = UnitCellFilter(self.atoms, mask=cell_mask) if relax_cell else self.atoms
        
        opt_class = self._OPTIMIZERS.get(algorithm.upper())
        if not opt_class:
            raise ValueError(f"Unknown algorithm: {algorithm}. Use: {list(self._OPTIMIZERS.keys())}")

        dyn = opt_class(target, logfile=None, trajectory=None)
        dyn.run(fmax=fmax)
        self.atoms = target.copy()  # Update the base structure to the relaxed one
        return self.atoms

    def _apply_noise(self, array: np.ndarray, level: float) -> np.ndarray:
        """Applies normal or uniform noise to a numpy array."""
        if self.noise_type == 'normal':
            return array + self.rng.normal(0, level, size=array.shape)
        return array + self.rng.uniform(-level, level, size=array.shape)

    def generate_samples(
        self, 
        num_samples: int = 100,
        noise_type: Optional[Literal['normal', 'uniform']] = None,
        noise_level_pos: float = 0.20,
        noise_level_cell: float = 0.00,
        scale_cell: float = 1.0,
        cell_mode: Literal['xy', 'all', 'fixed'] = 'all',
        compute_energy_and_forces: bool = False,
        verbose: bool = False
    ) -> List[Atoms]:
        """Generates multiple samples with noise applied to positions and cell."""
        self.samples = []

        if noise_type is not None:
            self.noise_type = noise_type

        if self.noise_type not in ['normal', 'uniform']:
            raise ValueError("Invalid noise type. Use 'normal' or 'uniform'.")

        if cell_mode not in ['xy', 'all', 'fixed']:
            raise ValueError("Invalid cell mode. Use 'xy', 'all', or 'fixed'.")
        
        for i in range(num_samples):
            new_atoms = self.atoms.copy()

            # 1. Cell Scaling and Noise
            new_cell = new_atoms.get_cell()
            if cell_mode == 'xy':
                new_cell[:2, :2] *= scale_cell
                # Apply noise only to the upper 2x2 block (x, y)
                noise_block = self.rng.normal(0, noise_level_cell, (2, 2)) if self.noise_type == 'normal' \
                              else self.rng.uniform(-noise_level_cell, noise_level_cell, (2, 2))
                new_cell[:2, :2] += noise_block
            elif cell_mode == 'all':
                new_cell *= scale_cell
                new_cell = self._apply_noise(new_cell, noise_level_cell)
            # If cell_mode is 'fixed', do not modify the cell
            new_atoms.set_cell(new_cell, scale_atoms=True)

            # 2. Noise in Positions
            new_pos = self._apply_noise(new_atoms.get_positions(), noise_level_pos)
            new_atoms.set_positions(new_pos)

            # 3. Compute energy and forces if requested
            if compute_energy_and_forces and self.calculator:
                new_atoms.calc = self.calculator
                new_atoms.info['REF_energy'] = new_atoms.get_potential_energy()
                new_atoms.set_array('REF_forces', new_atoms.get_forces())
                new_atoms.calc = None  # Remove the calculator to avoid large files/writing errors
                if verbose:
                    print(f"Sample {i+1}/{num_samples}: Energy = {new_atoms.info['REF_energy']:.4f} eV")
            else:
                if verbose:
                    print(f"Sample {i+1}/{num_samples} generated.")

            # 4. Store the new sample
            self.samples.append(new_atoms)

        return self.samples
    
    
    def statistics(self, energy_and_forces: bool = True) -> Dict[str, Union[float, np.ndarray]]:
        """Computes statistics of the generated samples, including deviations and optionally energies/forces.
        
        Parameters:
        ===========
        - energy_and_forces: Whether to include energy and forces statistics (requires calculator).
        
        Returns:
        ========
        A dictionary containing mean and standard deviation of cell and position deviations, and optionally energies and forces.
        """
        if not self.samples:
            print("No samples generated. Call generate_samples() first.")
            return {}
        
        if energy_and_forces and not self.calculator:
            raise ValueError("Calculator is required to compute energy and forces statistics.")

        cell_deviations = np.array([np.linalg.norm(atoms.get_cell() - self.atoms.get_cell()) for atoms in self.samples])
        pos_deviations = np.array([np.linalg.norm(atoms.get_positions() - self.atoms.get_positions()) for atoms in self.samples])

        dict = {
            'num_samples': len(self.samples),
            'cell_deviation_mean': np.mean(cell_deviations),
            'cell_deviation_std': np.std(cell_deviations),
            'pos_deviation_mean': np.mean(pos_deviations),
            'pos_deviation_std': np.std(pos_deviations)
        }

        if energy_and_forces and self.calculator:
            energies = np.array([atoms.info.get('REF_energy', np.nan) for atoms in self.samples])
            forces = np.array([atoms.get_array('REF_forces') if 'REF_forces' in atoms.arrays else np.full(atoms.get_positions().shape, np.nan) for atoms in self.samples])
            dict['energy_mean'] = np.mean(energies)
            dict['energy_std'] = np.std(energies)
            dict['forces_mean'] = np.mean(forces)
            dict['forces_std'] = np.std(forces)

        return dict

    def save_to_xyz(self, filename: str = 'noisy_samples.xyz'):
        """Saves the generated samples."""
        if not self.samples:
            print("No samples generated. Call generate_samples() first.")
            return          # Avoid writing an empty file
        
        write(filename, self.samples, format='extxyz')
        print(f"Dataset saved successfully in {filename}")