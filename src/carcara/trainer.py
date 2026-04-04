# -*- coding: utf-8 -*-
# file: trainer.py

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

import sys
import logging
import warnings
import torch
import yaml
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional
from mace.cli.run_train import main as mace_run_train
from mace.cli.eval_configs import main as mace_eval_configs

# --- Global Environment Setup ---
# Fix for PyTorch 2.6+ where 'weights_only=True' is the default.
# Necessary for loading e3nn spherical harmonics constants.
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([slice])

warnings.filterwarnings("ignore")

class MACETrainer:
    """
    MACE Model Trainer specialized for Active Learning workflows.
    All configuration parameters are explicitly defined in the constructor.
    """
    def __init__(
        self,
        name: str = "AL_iteration_0",              # name of the model and output files (e.g., "AL_iteration_0_stagetwo.model")
        train_file: str = "train.xyz",             # path to training dataset (XYZ format)
        test_file: str = "test.xyz",               # optional test set for evaluation during training (can be same as train_file)
        valid_fraction: float = 0.10,              # fraction of training data to use for validation
        eval_interval: int = 5,                    # evaluate on validation set every N epochs (adjust based on dataset size and training time)
        max_num_epochs: int = 500,                 # maximum number of training epochs (adjust based on convergence behavior)
        batch_size: int = 5,                       # number of structures per batch (adjust based on GPU memory)
        device: str = "cpu",                       # "cpu" or "cuda" for training
        default_dtype: str = "float64",            # default data type for training (float32 or float64)
        r_max: float = 6.0,                        # cutoff radius for neighbor interactions (in Angstroms)
        num_channels: int = 64,                    # number of channels in MACE layers (MACE order)
        max_L: int = 1,                            # maximum angular momentum (MACE order) - 1
        correlation: int = 2,                      # maximum correlation order (MACE order)
        num_interaction: int = 2,                  # number of interaction blocks (MACE layers)
        energy_key: str = "REF_energy",            # keys in the XYZ file for energy
        forces_key: str = "REF_forces",            # keys in the XYZ file for forces
        energy_weight: float = 10.0,               # weight for energy in the loss function
        forces_weight: float = 1000.0,             # weight for forces in the loss function
        swa: bool = True,                          # whether to use Stochastic Weight Averaging (SWA)
        start_swa: int = 250,                      # epoch to start SWA
        ema: bool = True,                          # whether to use Exponential Moving Average (EMA)
        ema_decay: float = 0.99,                   # decay rate for EMA
        amsgrad: bool = True,                      # whether to use AMSGrad optimizer variant
        restart_latest: bool = True,               # whether to resume training from the last checkpoint
        save_cpu: bool = True,                     # whether to save model checkpoints on CPU
        seed: int = 999,                           # for reproducibility
        E0s: Optional[Dict[int, float]] = None     # dictionary of isolated atom energies {key: atomic number, value: energy in eV}
    ):
        # Parameters prone to change during Active Learning iterations
        self._name = name
        self._train_file = train_file
        self._max_num_epochs = max_num_epochs
        self._restart_latest = restart_latest
        
        # Static model and environment parameters
        self.test_file = test_file
        self.valid_fraction = valid_fraction
        self.eval_interval = eval_interval
        self.batch_size = batch_size
        self.device = device
        self.default_dtype = default_dtype
        self.r_max = r_max
        self.num_channels = num_channels
        self.max_L = max_L
        self.correlation = correlation
        self.num_interaction = num_interaction
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight
        self.swa = swa
        self.start_swa = start_swa
        self.ema = ema
        self.ema_decay = ema_decay
        self.amsgrad = amsgrad
        self.save_cpu = save_cpu
        self.seed = seed
        self.E0s = E0s or {}

    # --- Setters and Getters for dynamic AL parameters ---

    @property
    def name(self) -> str:
        """Name of the model and output files."""
        return self._name


    @name.setter
    def name(self, value: str):
        self._name = value


    @property
    def train_file(self) -> str:
        """Path to the training XYZ dataset."""
        return self._train_file


    @train_file.setter
    def train_file(self, value: str):
        if not Path(value).exists():
            print(f"Warning: Training file '{value}' does not exist.")
        self._train_file = value


    @property
    def max_num_epochs(self) -> int:
        """Maximum number of training epochs."""
        return self._max_num_epochs


    @max_num_epochs.setter
    def max_num_epochs(self, value: int):
        if value < 1:
            raise ValueError("max_num_epochs must be at least 1.")
        self._max_num_epochs = value


    @property
    def restart_latest(self) -> bool:
        """Whether to resume training from the last checkpoint."""
        return self._restart_latest


    @restart_latest.setter
    def restart_latest(self, value: bool):
        self._restart_latest = value


    def to_dict(self) -> Dict[str, Any]:
        """
        Creates a dictionary containing all parameters in a 
        MACE-compatible configuration format.
        """
        return {
            "model": "MACE",
            "name": self._name,
            "train_file": self._train_file,
            "test_file": self.test_file,
            "valid_fraction": self.valid_fraction,
            "eval_interval": self.eval_interval,
            "max_num_epochs": self._max_num_epochs,
            "batch_size": self.batch_size,
            "device": self.device,
            "default_dtype": self.default_dtype,
            "r_max": self.r_max,
            "num_channels": self.num_channels,
            "max_L": self.max_L,
            "correlation": self.correlation,
            "num_interaction": self.num_interaction,
            "energy_key": self.energy_key,
            "forces_key": self.forces_key,
            "energy_weight": self.energy_weight,
            "forces_weight": self.forces_weight,
            "swa": self.swa,
            "start_swa": self.start_swa,
            "ema": self.ema,
            "ema_decay": self.ema_decay,
            "amsgrad": self.amsgrad,
            "restart_latest": self._restart_latest,
            "save_cpu": self.save_cpu,
            "seed": self.seed,
            "E0s": self.E0s
        }


    def run_train(self):
        """
        Triggers MACE training by generating a temporary config file
        and calling the main CLI entry point.
        """
        # Clear logging to avoid redundant console outputs
        logging.getLogger().handlers.clear()
        
        config_data = self.to_dict()
        
        # MACE requires a file path for the configuration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml.dump(config_data, tmp)
            tmp_path = tmp.name

        print(f"--- MACE TRAINING STARTED: {self._name} ---")
        sys.argv = ["mace_run_train", "--config", tmp_path]
        
        try:
            mace_run_train()
        finally:
            # Clean up the temporary YAML file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


    def eval_configs(
        self, 
        configs_path: str,                    # path to the XYZ file containing configurations to evaluate
        model_path: str,                      # path to the trained MACE model checkpoint (e.g., "AL_iteration_0_stagetwo.model")
        output_path: str,                     # path to save the evaluation results (e.g., "AL_iteration_0_evaluation.xyz")
        default_dtype: Optional[str] = None   # optional data type for evaluation (overrides trainer's default_dtype if provided)
    ):
        """
        Evaluates a set of configurations using a trained MACE model.
        Equivalent to the mace_eval_configs CLI tool.
        """
        logging.getLogger().handlers.clear()
        
        # Use provided dtype or fallback to the trainer's default
        dtype = default_dtype or self.default_dtype
        
        print(f"--- EVALUATING CONFIGS: {configs_path} with model {model_path} ---")
        
        # Build command line arguments for the evaluation script
        sys.argv = [
            "mace_eval_configs",
            f"--configs={configs_path}",
            f"--model={model_path}",
            f"--output={output_path}",
            f"--default_dtype={dtype}",
            f"--device={self.device}"
        ]
        
        try:
            mace_eval_configs()
            print(f"Evaluation finished. Results saved to: {output_path}")
        except Exception as e:
            print(f"Error during evaluation: {e}")



if __name__ == "__main__":
    trainer = MACETrainer(
        name="AL_Cycle_0",
        train_file="dataset_0.xyz",
        E0s={42: -4.602, 16: -0.891}, # Mo and S isolated atom energies
        max_num_epochs=100,
        device="cpu"
    )

    # Execute training for the first cycle
    trainer.run_train()

    # In a real AL loop, you would now:
    # 1. Evaluate uncertainty of the trained model
    # 2. Collect/Calculate new data points
    # 3. Update the trainer for the next cycle:
    # trainer.name = "MoS2_AL_Cycle_1"
    # trainer.train_file = "dataset_combined.xyz"
    # trainer.restart_latest = True
    # trainer.run_train()