import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

from collections import Counter
import warnings
warnings.filterwarnings("ignore")

from typing import Optional

from ase.io import read, write
from ase.optimize import BFGS, LBFGS, FIRE
from ase.filters import UnitCellFilter
from ase import Atoms


class RandomNoise:
    def __init__(self,
                 atoms: Atoms = None,
                 calculator: Optional = None,
                 noise_type: str = 'normal',
                 noise_level_position: float = 0.2,
                 noise_level_cell: float = 0.2,
                 scale_cell: float = 1.0,
                 cell_type: str = 'xy',
                 relax_first: bool = False,
                 relax_cell: bool = False):
        ''''
        Initialize the RandomNoise class.

        Parameters:
        ===========
        - atoms: ASE Atoms object representing the structure to which noise will be added.
        - noise_type: Type of noise to add ('normal' or 'uniform').
        - noise_level_position: Standard deviation (for normal) or range (for uniform) of noise added to atomic positions.
        - noise_level_cell: Standard deviation (for normal) or range (for uniform) of noise added to cell parameters.
        - scale_cell: Scaling factor for the noise added to cell parameters.
        - cell_type: Type of cell noise to add ('xy', 'all').
        - relax_first: Whether to relax the structure before adding noise.
        - relax_cell: Whether to relax the cell before adding noise.
        '''

        self.atoms = atoms
        self.calculator = calculator
        self.noise_type = noise_type
        self.noise_level_position = noise_level_position
        self.noise_level_cell = noise_level_cell
        self.scale_cell = scale_cell
        self.cell_type = cell_type
        self.relax_first = relax_first
        self.relax_cell = relax_cell
        self.energy_ref = None
        self.samples = None

        if self.relax_first and self.calculator is not None:
            self.atoms = relax_structure(self.atoms, calculator=self.calculator, relax_cell=self.relax_cell, algorithm='BFGS', fmax=0.01)
            self.energy_ref = self.atoms.get_potential_energy()

        def relax_structure(atoms, calculator, algorithm='BFGS', fmax=0.01, relax_cell=self.relax_cell, cell_mask=[1,1,1,1,1,1], inplace: bool = False) -> Atoms:
            '''
            Relax the structure using the specified algorithm.

                Parameters:
                ===========
                - atoms: ASE Atoms object representing the structure to relax.
                - calculator: ASE calculator to use for evaluating forces and energy during relaxation.
                - algorithm: Optimization algorithm to use ('BFGS', 'LBFGS', 'FIRE').
                - fmax: Maximum force criterion for convergence.
                - relax_cell: Whether to relax the cell parameters during relaxation.
                - cell_mask: Optional list to specify which cell parameters to relax (default is [1,1,1,1,1,1]).
                - inplace: Whether to modify the input Atoms object in place or return a new relaxed Atoms object.

                Returns:
                ========
                - Relaxed ASE Atoms object.
            '''
            if calculator is None:
                raise ValueError("Calculator must be provided for relaxation.")
            
            atoms.calc = calculator

            if relax_cell:
                if algorithm == 'BFGS':
                    dyn = BFGS(UnitCellFilter(atoms, mask=cell_mask), logfile=None, trajectory=None)
                elif algorithm == 'LBFGS':
                    dyn = LBFGS(UnitCellFilter(atoms, mask=cell_mask), logfile=None, trajectory=None)
                elif algorithm == 'FIRE':
                    dyn = FIRE(UnitCellFilter(atoms, mask=cell_mask), logfile=None, trajectory=None)
                else:
                    raise ValueError(f"Unknown algorithm: {algorithm}")
            else:
                if algorithm == 'BFGS':
                    dyn = BFGS(atoms, logfile=None, trajectory=None)
                elif algorithm == 'LBFGS':
                    dyn = LBFGS(atoms, logfile=None, trajectory=None)
                elif algorithm == 'FIRE':
                    dyn = FIRE(atoms, logfile=None, trajectory=None)
                else:
                    raise ValueError(f"Unknown algorithm: {algorithm}")
            
            dyn.run(fmax=fmax)

            if inplace:
                self.atoms = atoms

            return atoms
        
        def generate_random_noise(atoms, calculator: Optional = None, num_samples=100) -> list:
            '''
            Generate random noise samples by adding noise to the positions and cell parameters of the input structure.

                Parameters:
                ===========
                - atoms: ASE Atoms object representing the structure to which noise will be added.
                - calculator: ASE calculator to use for evaluating forces and energy of the generated samples.
                - num_samples: Number of random noise samples to generate.
                
                Returns:
                ========
                - List of ASE Atoms objects with added noise.

            '''
            
            if calculator is None:
                raise ValueError("Calculator must be provided to generate random noise.")
            
            atoms.calc = calculator
            noise_level_cell = self.noise_level_cell
            noise_level_position = self.noise_level_position

            samples = []
            for i in range(num_samples):
                new_atoms = atoms.copy()

                # Scale the cell parameters before adding noise
                new_cell = new_atoms.cell.copy()
                if cell_type == 'xy':
                    new_cell[:2, :2] *= scale_cell
                elif cell_type == 'all':
                    new_cell *= scale_cell
                else:
                    raise ValueError("Invalid cell_type. Choose 'xy' or 'all'.")
                new_atoms.set_cell(new_cell, scale_atoms=True)

                # Normal noise for positions and cell parameters
                if self.noise_type == 'normal':
                    noise_pos = np.random.normal(0, noise_level_position, size=new_atoms.positions.shape)
                    new_atoms.positions += noise_pos
                    if self.cell_type == 'xy':
                        noise_cell = np.zeros_like(new_atoms.cell)
                        noise_cell[:2, :2] = np.random.normal(0, noise_level_cell, size=(2, 2))
                        new_atoms.cell += noise_cell
                    else:
                        noise_cell = np.random.normal(0, noise_level_cell, size=(3,3))
                        new_atoms.cell += noise_cell
                # Uniform noise for positions and cell parameters
                elif self.noise_type == 'uniform':
                    noise_pos = np.random.uniform(-noise_level_position, noise_level_position, size=new_atoms.positions.shape)
                    new_atoms.positions += noise_pos
                    if self.cell_type == 'xy':
                        noise_cell = np.zeros_like(new_atoms.cell)
                        noise_cell[:2, :2] = np.random.uniform(-noise_level_cell, noise_level_cell, size=(2, 2))
                        new_atoms.cell += noise_cell
                    else:
                        noise_cell = np.random.uniform(-noise_level_cell, noise_level_cell, size=(3,3))
                        new_atoms.cell += noise_cell
                else:
                    raise ValueError(f"Unknown noise type: {self.noise_type}")
                
                samples.append(new_atoms.copy())

            self.samples = samples
            return samples
        
        def write_samples_to_xyz(samples, filename: str = 'noise_samples.xyz', clean_calculator: bool = True) -> None:
            '''
            Write the generated samples to an XYZ file, including forces and energy in the info dictionary.

                Parameters:
                ===========
                - samples: List of ASE Atoms objects representing the generated noise samples.
                - filename: Name of the output XYZ file.
                - clean_calculator: Whether to remove the calculator from the samples before writing to file.
                
            '''
            samples = self.samples

            for atom in samples:
                atom.info = {} # clear info dictionary
                atom.set_array('REF_forces', atom.get_forces())
                atom.info['REF_energy'] = atom.get_potential_energy()
                if clean_calculator:
                    atom.calc = None

            write(filename, samples, format='extxyz')

