# -*- coding: utf-8 -*-
# file: data_utils.py

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

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.parallel import parprint as print

class DatasetManager:
    """
    A class to handle datasets of atomic configurations stored in XYZ format. It provides methods to split the dataset into training, validation, and testing sets based on specified ratios. The splits are saved as separate XYZ files.
    
    Attributes:
    ===========
    - filename: The path to the input XYZ file containing the dataset.
    - seed: The random seed for reproducibility when shuffling the dataset.
    - atoms: A list of Atoms objects read from the input file.
    - total_configs: The total number of configurations in the dataset.
    - rng: A NumPy random generator initialized with the specified seed.

    Methods:
    ========
    - split: A generic method to split the dataset based on provided ratios and save the splits
    - train_test_split: A convenience method to split the dataset into training and testing sets.
    - train_validation_test_split: A convenience method to split the dataset into training, validation, and testing sets.
    """
    def __init__(self, filename: str, seed: int = 42):
        self.filename = filename
        self.seed = seed
        self._path = Path(filename)
        self.atoms: List[Atoms] = read(self.filename, index=":")
        self.total_configs = len(self.atoms)
        self.rng = np.random.default_rng(self.seed)
        self.split_data = None


    def _get_shuffled_atoms(self) -> List[Atoms]:
        """
        Shuffle the dataset and return a list of Atoms objects in random order.
        """
        indices = np.arange(self.total_configs)
        self.rng.shuffle(indices)
        return [self.atoms[i] for i in indices]


    def _save_and_report(self, split_data: dict, verbose: bool = True):
        """Helper to save files and print logs."""
        if verbose:
            print(f"Total configurations: {self.total_configs}")
        
        for name, configs in split_data.items():
            fname = f"{name}_seed_{self.seed}.xyz"
            write(fname, configs, format="extxyz")
            if verbose:
                print(f"{name.capitalize()} set: {len(configs)} configurations")
        
        if verbose:
            files = ", ".join([f"{k}_seed_{self.seed}.xyz" for k in split_data.keys()])
            print(f"Files saved: {files}")


    def split(self, ratios: dict):
        """
        Generic method to split the dataset.

        Parameters:
        ===========
        - ratios: A dictionary where keys are split names (e.g., "train", "test") and values are the corresponding ratios (e.g., 0.8, 0.2). The ratios must sum to 1.0.

        Returns:
        ========
        - A dictionary where keys are split names (e.g., "train", "test") and values are lists of Atoms objects corresponding to each split.
        """
        for name, ratio in ratios.items():
            if ratio < 0 or ratio > 1:
                raise ValueError(f"Invalid ratio for '{name}': {ratio}. Ratios must be between 0 and 1.")
            
        if not np.isclose(sum(ratios.values()), 1.0, atol=1e-4):
            raise ValueError(f"The ratios must sum to 1.0. Current sum: {sum(ratios.values()):.4f}")

        shuffled_atoms = self._get_shuffled_atoms()
        split_results = {}
        current_idx = 0
        
        # Convert keys to a list to ensure order in the return
        keys = list(ratios.keys())
        
        for i, name in enumerate(keys):
            # If it's the last element, take the rest to avoid rounding errors
            
            if i == len(keys) - 1:
                split_results[name] = shuffled_atoms[current_idx:] # Example of split_results after first iteration: {"train": [Atoms1, Atoms2, ..., AtomsN]} where N is int(0.7 * total_configs)
            else:
                n_configs = int(ratios[name] * self.total_configs)
                split_results[name] = shuffled_atoms[current_idx : current_idx + n_configs]
                current_idx += n_configs

        self.split_data = split_results         # Example:
                                                # Input:  ratios={"train": 0.7, "test": 0.3}, total_configs=100
                                                # Output: self.split_data = {"train": [Atoms1, ..., Atoms70], "test": [Atoms71, ..., Atoms100]}


    def train_test_split(self, train_ratio: float = 0.8) -> dict:
        """
        Split the dataset into training and testing sets based on the specified ratio.
        """
        if train_ratio < 0 or train_ratio > 1:
            raise ValueError("train_ratio must be between 0 and 1.")
        
        test_fraction = 1.0 - train_ratio
        if test_fraction < 0 or test_fraction > 1:
            raise ValueError("train_ratio must be less than 1.0 to have a valid test set.")
        
        ratios = {"train": train_ratio, "test": test_fraction}
        return self.split(ratios=ratios)
    

    def train_valid_split(self, train_ratio: float = 0.8) -> dict:
        """
        Split the dataset into training and validation sets based on the specified ratios.
        """
        if train_ratio < 0 or train_ratio > 1:
            raise ValueError("train_ratio must be between 0 and 1.")
        
        valid_fraction = 1.0 - train_ratio
        if valid_fraction < 0 or valid_fraction > 1:
            raise ValueError("train_ratio must be between 0 and 1 to have a valid validation set.")

        ratios = {"train": train_ratio, "valid": valid_fraction}
        return self.split(ratios=ratios)


    def train_valid_test_split(self, train_ratio: float = 0.8, valid_ratio: float = 0.1) -> dict:
        """
        Split the dataset into training, validation, and testing sets based on the specified ratios.
        """
        if train_ratio < 0 or train_ratio > 1:
            raise ValueError("train_ratio must be between 0 and 1.")
        
        if valid_ratio < 0 or valid_ratio > 1:
            raise ValueError("valid_ratio must be between 0 and 1.")
        
        test_ratio = 1.0 - train_ratio - valid_ratio
        if test_ratio < 0 or test_ratio > 1:
            raise ValueError("test_ratio must be between 0 and 1. Check that train_ratio + valid_ratio is less than 1.")
            
        ratios = {"train": train_ratio, "valid": valid_ratio, "test": test_ratio}
        return self.split(ratios=ratios)
    

    def write_datasets(self, directory: str = ".", filenames: List[str] = None):
        if self.split_data is None:
            raise ValueError("No split data found. Please run the split method before writing datasets.")
        
        if filenames is not None and len(filenames) != len(self.split_data):
            raise ValueError(f"Number of filenames ({len(filenames)}) must match the number of splits ({len(self.split_data)}).")
        
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        if filenames is not None:
            for name, fname in zip(self.split_data.keys(), filenames):
                if not fname.endswith(".xyz"):
                    fname += ".xyz"
                write(path / fname, self.split_data[name], format="extxyz")
                print(f"{name.capitalize()} set: {len(self.split_data[name])} configurations")
        else:
            for name, configs in self.split_data.items():
                fname = f"{name}_seed_{self.seed}.xyz"
                write(path / fname, configs, format="extxyz")
                print(f"{name.capitalize()} set: {len(configs)} configurations")

        files = ", ".join([f"{k}_seed_{self.seed}.xyz" for k in self.split_data.keys()])
        print(f"Files saved: {files}")
    