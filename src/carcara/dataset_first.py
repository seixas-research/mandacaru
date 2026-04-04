# use this if data_utils.py is not working
# -*- coding: utf-8 -*-
# file: dataset_first.py

from ase.io import read, write
import numpy as np

class Dataset:
    def __init__(self, filename: str, seed: int = 42):
        self.filename = filename
        self.seed = seed
        self.atoms = read(self.filename, index=":")
        self.total_configs = len(self.atoms)
    

    def train_test_split(self, train_ratio: float = 0.8, verbose: bool = True):
        atoms = self.atoms
        total_configs = self.total_configs
        

        # Shuffle indices
        indices = np.arange(total_configs)
        np.random.seed(self.seed)                # For reproducibility
        np.random.shuffle(indices)

        # Define train/test split
        split_idx = int(train_ratio * total_configs)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        # Split configurations
        train_configs = [atoms[i] for i in train_indices]
        test_configs = [atoms[i] for i in test_indices]

        # Save train and test sets
        write("train.xyz", train_configs)
        write("test.xyz", test_configs)

        if verbose:
            print(f"Total configurations: {total_configs}")
            print(f"Training set: {len(train_configs)} configurations")
            print(f"Test set: {len(test_configs)} configurations")
            print("Files saved: train.xyz, test.xyz")

        return train_configs, test_configs


    def train_validation_test_split(self, train_ratio: float = 0.8, valid_ratio: float = 0.1, verbose: bool = True):
        atoms = self.atoms
        total_configs = self.total_configs

        if train_ratio < 0 or valid_ratio < 0 or train_ratio > 1 or valid_ratio > 1:
            raise ValueError("Train and validation ratios must be between 0 and 1.")
        

        if abs(train_ratio + valid_ratio - 1.0) > 1e-4:
            raise ValueError("Train and validation ratios must sum to less than 1.0.")
        
        
        # Shuffle indices
        indices = np.arange(total_configs)
        np.random.seed(self.seed)                # For reproducibility
        np.random.shuffle(indices)

        # Define train/validation/test split
        train_idx = int(train_ratio * total_configs)
        val_idx = int(valid_ratio * total_configs)
        
        train_indices = indices[:train_idx]
        val_indices = indices[train_idx:train_idx + val_idx]
        test_indices = indices[train_idx + val_idx:]

        # Split configurations
        train_configs = [atoms[i] for i in train_indices]
        val_configs = [atoms[i] for i in val_indices]
        test_configs = [atoms[i] for i in test_indices]

        # Save train, validation and test sets
        write("train.xyz", train_configs)
        write("validation.xyz", val_configs)
        write("test.xyz", test_configs)

        if verbose:
            print(f"Total configurations: {total_configs}")
            print(f"Training set: {len(train_configs)} configurations")
            print(f"Validation set: {len(val_configs)} configurations")
            print(f"Test set: {len(test_configs)} configurations")
            print("Files saved: train.xyz, validation.xyz, test.xyz")

        return train_configs, val_configs, test_configs