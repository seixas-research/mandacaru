# -*- coding: utf-8 -*-
# file: __init__.py

# This code is part of Carcará. 
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br> 

from .version import __version__

import warnings
warnings.filterwarnings("ignore")

import os
import platform
from socket import gethostname

# System
from sys import version as __python_version__
from sys import executable as __python_executable__

# ASE
from ase import __version__ as __ase_version__
from ase import __file__ as __ase_file__
from ase.parallel import parprint as print

# Numpy
from numpy import __version__ as __numpy_version__
from numpy import __file__ as __numpy_file__

# Scipy
from scipy import __version__ as __scipy_version__
from scipy import __file__ as __scipy_file__

# Matplotlib
from matplotlib import __version__ as __mpl_version__
from matplotlib import __file__ as __mpl_file__

# Pytest
from pytest import __version__ as __pytest_version__
from pytest import __file__ as __pytest_file__

# Carcará version
from .version import __version__


# Carcará BANNER
def banner():
    print("       _____                                  ")
    print("      / ____|                                 ")
    print("     | |     __ _ _ __ ___ __ _ _ __ __ _     ")
    print("     | |    / _` | '__/ __/ _` | '__/ _` |    ")
    print("     | |___| (_| | | | (_| (_| | | | (_| |    ")
    print("      \_____\__,_|_|  \___\__,_|_|  \__,_|    ")
    print("                                              ")
    print(f"    version: {__version__}                     ")
    print("    developed by:  Leandro Seixas Rocha      ")
    print("    homepage:      https://github.com/seixas-research/carcara")
    print("    documentation: https://carcara.readthedocs.io/")
    print("                                                  ")
    print("-----------------------------------------------------------------")
    print("                                                  ")
    print("System:")
    print(f"├── architecture: {platform.machine()}")
    print(f"├── platform: {platform.system()}")
    print(f"├── user: {os.environ['USER']}")
    print(f"├── hostname: {gethostname()}")
    print(f"├── cwd: {os.getcwd()}")
    print(f"└── PID: {os.getpid()}")
    print("                                               ")
    print("Python:")
    print(f"├── version: {__python_version__}      ")
    print(f"└── executable: {__python_executable__}      ")
    print("                                               ")
    print("Dependencies:")
    print(f"├── ase version:        {__ase_version__}    [{__ase_file__[:-11]}]")
    print(f"├── numpy version:      {__numpy_version__}    [{__numpy_file__[:-11]}]")
    print(f"├── scipy version:      {__scipy_version__}    [{__scipy_file__[:-11]}]")
    print(f"├── matplotlib version: {__mpl_version__}    [{__mpl_file__[:-11]}]")
    print(f"└── pytest version:     {__pytest_version__}    [{__pytest_file__[:-11]}]")
    print("                                                           ")