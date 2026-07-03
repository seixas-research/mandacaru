# -*- coding: utf-8 -*-
# file: integrals/__init__.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Real-space integral engine and its high-performance C backend."""

from ._backend import HAS_C_BACKEND
from .engine import IntegralEngine
from .grid import Grid
from .poisson import PoissonFFTSolver

__all__ = ["IntegralEngine", "Grid", "PoissonFFTSolver", "HAS_C_BACKEND"]
