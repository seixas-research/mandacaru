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
        self.dataset = [self._process_structure(structure) for structure in self.dataset]

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
    

    



