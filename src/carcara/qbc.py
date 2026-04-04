# -*- coding: utf-8 -*-
# file: qbc.py

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
from ase import Atoms
import pandas as pd
from typing import Optional


class QueryByCommittee:
    def __init__(self,
                 atoms: Atoms,
                 number_of_committees: int = 5,
                 seeds: Optional[list] = None,
                 train_samples: Optional[list] = None,
                 model_config: Optional[dict] = None):
        self.atoms = atoms
        self.number_of_committees = number_of_committees
        self.seeds = seeds if seeds is not None else np.random.randint(0, 10000, size=number_of_committees).tolist()
        self.train_samples = train_samples
        self.model_config = model_config if model_config is not None else {}

    def train_model(self, seeds: Optional[int] = None, model_config: Optional[dict] = None):
        # train a MACE model with a seed and return the trained model
        pass

    def committee_predictions(self, models, samples):
        # Given a set of models and a set of samples, return the standard deviation of the forces for each sample.
        pass

    def select_samples(self, samples, stds, num_samples_to_select: int = 10):
        # Select the samples with the highest standard deviation in the predictions
        selected_indices = np.argsort(stds)[-num_samples_to_select:]
        selected_samples = [samples[i] for i in selected_indices]
        return selected_samples


    