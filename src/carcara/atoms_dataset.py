# -*- coding: utf-8 -*-
# file: atoms_dataset.py

# This code is part of Carcará. 
# MIT License
#
# Copyright (c) 2025 Leandro Seixas Rocha <leandro.fisica@gmail.com> 
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
import torch
from ase.io import read
from torch.utils.data import Dataset

class AtomsDataset(Dataset):
    def __init__(self, dataset=None):
        """
        Initializes the object with a dataset of atomic structures.

        Parameters
        ----------
        dataset : str or file-like
            Path to the dataset file or file-like object containing atomic structures.
            Must not be None.

        Raises
        ------
        ValueError
            If `dataset` is None.

        Attributes
        ----------
        dataset : list
            List of Atoms objects representing the structures read from the dataset.
        dataset_proc : list of dict
            List of dictionaries containing processed information (positions and atomic numbers)
            for each structure in the dataset.
        """
        if dataset is None:
            raise ValueError("Dataset cannot be None.")
        self.dataset = read(dataset, index=':') # => Atoms objects (list of structures)
        self.dataset_proc = [self._process_structure(structure) for structure in self.dataset] # list of dicts with positions and atomic numbers


    def _process_structure(self, structure):
        """
        Processes a structure object to extract atomic positions and atomic numbers as PyTorch tensors.
        Args:
            structure: An object with `get_positions()` and `get_atomic_numbers()` methods, which return
                the atomic positions and atomic numbers, respectively.
        Returns:
            dict: A dictionary containing:
                - 'positions' (torch.FloatTensor): Tensor of atomic positions.
                - 'atomic_numbers' (torch.LongTensor): Tensor of atomic numbers.
        Raises:
            ValueError: If the structure contains no atoms (empty positions or atomic numbers).
        """
        positions = structure.get_positions() # shape (N, 3) numpy array
        atomic_numbers = structure.get_atomic_numbers() # shape (N,) numpy array
        if len(positions) == 0 or len(atomic_numbers) == 0:
            raise ValueError("Structure must contain at least one atom.")
        
        positions = torch.tensor(positions, dtype=torch.float32) # convert to float32 tensor
        atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.int64) # convert to int64 tensor
        return {'positions': positions, 'atomic_numbers': atomic_numbers} # dict with positions and atomic numbers

    def __len__(self):
        ''' Returns the number of structures in the dataset. '''
        return len(self.dataset)
    
    def get_distance_matrices(self):
        """
        Computes the distance matrices for all structures in the dataset.
        Returns:
            list: A list of distance matrices, where each matrix corresponds to a structure in the dataset.
        """
        distance_matrices = []
        for structure in self.dataset_proc:
            positions = structure['positions']
            dist_matrix = torch.cdist(positions.unsqueeze(0), positions.unsqueeze(0)).squeeze(0)
            distance_matrices.append(dist_matrix)
        return distance_matrices
    
    def get_unique_atomic_numbers(self):
        """
        Retrieves the unique atomic numbers from all structures in the dataset.

        Example: for methanol (CH3OH), the unique atomic numbers are [1, 6, 8],
        corresponding to Hydrogen (1), Carbon (6), and Oxygen (8).
        
        Returns
        -------
            list: A list of unique atomic numbers.
        """
        unique_atomic_numbers = set()
        for structure in self.dataset_proc:
            unique_atomic_numbers.update(structure['atomic_numbers'].unique().tolist())
        return sorted(unique_atomic_numbers)
    

    def onehot_enconding(self):
        """
        Creates a one-hot encoding of the atomic numbers present in the dataset.
        This method generates a 2D tensor where each row corresponds to a unique atomic number,
        and each column corresponds to a position in the one-hot encoding.

        The one-hot encoding is constructed such that:
        - Each row corresponds to a unique atomic number.
        - The columns are indexed from 0 to the maximum atomic number minus one.
        - The value at position (i, j) is 1 if the atomic number i+1 corresponds to j+1, otherwise it is 0.

        Example:
        If the unique atomic numbers are [1, 6, 8], the one-hot encoding will be:
        [[1, 0, 0, 0, 0, 0, 0, 0],  # Hydrogen (1)
         [0, 0, 0, 0, 0, 1, 0, 0],  # Carbon (6)
         [0, 0, 0, 0, 0, 0, 0, 1]], # Oxygen (8)
        where each row corresponds to the one-hot encoding of an atomic number.

        Returns
        -------
        torch.Tensor
            A 2D tensor of shape (num_unique_atomic_numbers, max_atomic_number), where each row
            corresponds to the one-hot encoding of an atomic number.
        """
        unique_atomic_numbers = self.get_unique_atomic_numbers() # example: [1, 6, 8]
        max_atomic_number = max(unique_atomic_numbers)           # example: 8
        onehot = torch.zeros((len(unique_atomic_numbers), max_atomic_number), dtype=torch.float32) # example: shape (3, 8)
        
        for i, atomic_number in enumerate(unique_atomic_numbers): # example: i=1, atomic_number=6
            onehot[i, atomic_number - 1] = 1.0                    # example: onehot[1, 5] = 1.0 (for Carbon)

        return torch.tensor(onehot, dtype=torch.float32)
    
    def get_atomic_state(self):
        """
        Retrieves the atomic state for all structures in the dataset.

        Returns
        -------
            list: A list of dictionaries, each containing:
                - 'positions' (torch.FloatTensor): Tensor of atomic positions.
                - 'atomic_numbers' (torch.LongTensor): Tensor of atomic numbers.
                - 'onehot' (torch.FloatTensor): One-hot encoded tensor of atomic numbers.
        """
        atomic_states = []
        for structure in self.dataset_proc:
            onehot = self.onehot_enconding(structure['atomic_numbers'])
            atomic_states.append({
                'positions': structure['positions'],
                'atomic_numbers': structure['atomic_numbers'],
                'onehot': onehot
            })
        return atomic_states
    


