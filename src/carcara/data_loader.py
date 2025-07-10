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
    def __init__(self, dataset=[]):  # dataset is a list of file paths: ['file1.xyz', 'file2.xyz', ...]
        """
        Initializes the data loader with a list of dataset file paths.

        Args:
            dataset (list, optional): List of file paths to datasets in XYZ format. Defaults to an empty list.

        Attributes:
            dataset_files (list): Stores the provided list of dataset file paths.
            dataset (list): List of data objects read from the provided file paths using the 'read' function.
        """
        self.dataset_files = dataset
        self.dataset = []
        for i, data in enumerate(self.dataset_files):
            self.dataset.append(read(data, format='xyz'))

    def __len__(self):
        ''' Returns the number of items in the dataset. '''
        return len(self.dataset)
    



