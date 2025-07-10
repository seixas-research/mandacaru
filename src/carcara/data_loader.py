# -*- coding: utf-8 -*-
# file: data_loader.py

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
from ase.io import read
import torch

class DataLoader:
    def __init__(self, dataset=None):
        """
        Initializes the data loader with a given dataset.
        Parameters:
            dataset (str or list): The dataset to load. If a string is provided, it will be read and processed.
                                   Must not be None.
        Raises:
            ValueError: If the dataset is None.
            ValueError: If the loaded dataset is not a list of structures.
        """
        if dataset is None:
            raise ValueError("Dataset cannot be None.")
        self.dataset = read(dataset, index=':')
        if not isinstance(self.dataset, list):
            raise ValueError("Dataset must be a list of structures.")
        self.dataset_proc = [self._process_structure(structure) for structure in self.dataset]

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
        positions = structure.get_positions()
        atomic_numbers = structure.get_atomic_numbers()
        if len(positions) == 0 or len(atomic_numbers) == 0:
            raise ValueError("Structure must contain at least one atom.")
        
        positions = torch.tensor(positions, dtype=torch.float32)
        atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.int64)
        return {'positions': positions, 'atomic_numbers': atomic_numbers}

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
        Returns:
            list: A list of unique atomic numbers.
        """
        unique_atomic_numbers = set()
        for structure in self.dataset_proc:
            unique_atomic_numbers.update(structure['atomic_numbers'].unique().tolist())
        return sorted(unique_atomic_numbers)
    

    def onehot_enconding(self, atomic_numbers):
        """
        Converts atomic numbers to one-hot encoded vectors.
        Args:
            atomic_numbers (torch.LongTensor): Tensor of atomic numbers.
        Returns:
            torch.FloatTensor: One-hot encoded tensor of atomic numbers.
        """
        unique_atomic_numbers = self.get_unique_atomic_numbers()
        num_elements = len(unique_atomic_numbers)
        onehot = torch.zeros((len(atomic_numbers), num_elements), dtype=torch.float32)
        
        for i, atomic_number in enumerate(atomic_numbers):
            if atomic_number in unique_atomic_numbers:
                index = unique_atomic_numbers.index(atomic_number)
                onehot[i, index] = 1.0
        
        return onehot
    
    def get_atomic_state(self):
        """
        Retrieves the atomic state for all structures in the dataset.
        Returns:
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
    


