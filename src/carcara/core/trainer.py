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
from ase.parallel import parprint as print

# --- Global Environment Setup ---
# Fix for PyTorch 2.6+ where 'weights_only=True' is the default.
# Necessary for loading e3nn spherical harmonics constants.
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([slice])

warnings.filterwarnings("ignore")

class Trainer:
    """
    MACE Model Trainer for Active Learning workflows.
    All configuration parameters are explicitly defined in the constructor.
    """
    def __init__(
        self,
        # dynamics parameters
        name: str = "gen_0_model_0",              # name of the model and output files (e.g., "gen_0_model_0_stagetwo.model")
        train_file: str = "training.xyz",          # path to training dataset (XYZ format)
        valid_file: str = "validation.xyz",        # optional validation set for evaluation during training (can be same as train_file)
        max_num_epochs: int = 500,                 # maximum number of training epochs (adjust based on convergence behavior)
        restart_latest: bool = True,               # whether to resume training from the last checkpoint
        # static parameters
        num_channels: int = 32,                    # number of channels in MACE layers (MACE order)
        max_L: int = 1,                            # maximum angular momentum for spherical harmonics
        num_interaction: int = 2,                  # number of interaction blocks (MACE layers)
        correlation: int = 2,                      # maximum correlation order (MACE order)
        eval_interval: int = 5,                    # evaluate on validation set every N epochs (adjust based on dataset size and training time)
        batch_size: int = 10,                      # number of structures per batch (adjust based on GPU memory)
        valid_batch_size: int = 20,                # batch size for validation (defaults to train batch size if None)
        patience: int = 50,                        # epochs to wait for improvement before early stopping (adjust based on convergence behavior)
        device: str = "cpu",                       # "cpu" or "cuda" for training
        default_dtype: str = "float32",            # default data type for training (float32 or float64)
        r_max: float = 5.0,                        # cutoff radius for neighbor interactions (in Angstroms)
        energy_key: str = "REF_energy",            # keys in the XYZ file for energy
        forces_key: str = "REF_forces",            # keys in the XYZ file for forces
        E0s: Optional[Dict[int, float]] = None,    # dictionary of isolated atom energies {key: atomic number, value: energy in eV}
        energy_weight: float = 10.0,               # weight for energy in the loss function
        forces_weight: float = 1000.0,             # weight for forces in the loss function
        swa: bool = True,                          # whether to use Stochastic Weight Averaging (SWA)
        start_swa: int = 250,                      # epoch to start SWA
        ema: bool = True,                          # whether to use Exponential Moving Average (EMA)
        ema_decay: float = 0.99,                   # decay rate for EMA
        amsgrad: bool = True,                      # whether to use AMSGrad optimizer variant
        save_cpu: bool = True,                     # whether to save model checkpoints on CPU
        seed: int = 999                            # for reproducibility
    ):
        # Parameters prone to change during Active Learning iterations
        self._name = name
        self._train_file = train_file
        self._valid_file = valid_file
        self._max_num_epochs = max_num_epochs
        self._restart_latest = restart_latest

        # Static model and environment parameters
        self.eval_interval = eval_interval
        self.batch_size = batch_size
        self.patience = patience
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
        self.E0s = E0s or "Average"
        self.valid_batch_size = valid_batch_size
        self.swa = swa
        self.start_swa = start_swa
        self.ema = ema
        self.ema_decay = ema_decay
        self.amsgrad = amsgrad
        self.save_cpu = save_cpu
        self.seed = seed
        self.path = Path(".")


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
    def valid_file(self) -> str:
        """Path to the validation XYZ dataset."""
        return self._valid_file

    @valid_file.setter
    def valid_file(self, value: str):
        if not Path(value).exists():
            print(f"Warning: Validation file '{value}' does not exist.")
        self._valid_file = value

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
            "name": self._name,
            "train_file": self._train_file,
            "valid_file": self._valid_file,
            "max_num_epochs": self._max_num_epochs,
            "restart_latest": self._restart_latest,
            "num_channels": self.num_channels,
            "max_L": self.max_L,
            "num_interaction": self.num_interaction,
            "correlation": self.correlation,
            "eval_interval": self.eval_interval,
            "batch_size": self.batch_size,
            "valid_batch_size": self.valid_batch_size,
            "patience": self.patience,
            "device": self.device,
            "default_dtype": self.default_dtype,
            "r_max": self.r_max,
            "energy_key": self.energy_key,
            "forces_key": self.forces_key,
            "E0s": self.E0s,
            "energy_weight": self.energy_weight,
            "forces_weight": self.forces_weight,
            "swa": self.swa,
            "start_swa": self.start_swa,
            "ema": self.ema,
            "ema_decay": self.ema_decay,
            "amsgrad": self.amsgrad,
            "save_cpu": self.save_cpu,
            "seed": self.seed
        }


    def run_train(self, path: str = "."):
        """
        Triggers MACE training by generating a temporary config file
        and calling the main CLI entry point.
        """
        # Clear logging to avoid redundant console outputs
        logging.getLogger().handlers.clear()

        # logging.basicConfig(level=logging.INFO, filename=output_file, filemode="w")
                            # format="%(asctime)s - %(levelname)s - %(message)s")
        
        config_data = self.to_dict()

        self.path = Path(path)
        os.makedirs(self.path, exist_ok=True)
        os.chdir(self.path)

        # MACE requires a file path for the configuration, created at directory path
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml.dump(config_data, tmp)
            tmp_path = tmp.name

        sys.argv = ["mace_run_train", "--config", tmp_path]
        
        mace_run_train()

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        os.chdir("..")  # Return to original directory after training


    def clean_directories(self):
        """
        Utility method to clean logs, checkpoints, and results directories.
        """
        for subdir in ["logs", "checkpoints", "results"]:
            path = self.path / subdir
            if path.exists() and path.is_dir():
                for file in path.glob("*"):
                    file.unlink()
                path.rmdir()
                print(f"Deleted directory {path}")
            else:
                print(f"No directory found at {path}, skipping cleanup.")


    def save_config(self, output_path = None):
        """
        Saves the current configuration parameters to a YAML file.
        """
        if output_path is None:
            raise ValueError("Output path must be provided to save the configuration.")
        
        if not isinstance(output_path, str):
            raise ValueError("Output path must be a string.")
        
        config_data = self.to_dict()
        with open(output_path, 'w') as f:
            yaml.dump(config_data, f)
        print(f"Configuration saved to {output_path}")


    def eval_configs(
        self, 
        configs_path: str,                    # path to the XYZ file containing configurations to evaluate
        model_path: str,                      # path to the trained MACE model checkpoint (e.g., "AL_iteration_0_stagetwo.model")
        output_path: str,                     # path to save the evaluation results (e.g., "AL_iteration_0_evaluation.xyz")
        default_dtype: str = "float32"        # data type for evaluation (float32 or float64, defaults to trainer's default if not provided
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

