# -*- coding: utf-8 -*-
# file: circuits/__init__.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Parameterized circuits: ansätze, excitation-gate generators and operator pools."""

from .adapt_ansatz import AdaptAnsatz
from .ansatz import UCCSD
from .base import Ansatz
from .gates import double_excitation, single_excitation
from .pools import (
    CEOPool,
    FermionicPool,
    PoolBase,
    PoolOperator,
    QEBPool,
    QubitPool,
    available_pools,
    build_pool,
)
from .profiling import CircuitMetrics, profile_ansatz

__all__ = [
    "Ansatz",
    "UCCSD",
    "AdaptAnsatz",
    "single_excitation",
    "double_excitation",
    "PoolBase",
    "PoolOperator",
    "FermionicPool",
    "QubitPool",
    "QEBPool",
    "CEOPool",
    "build_pool",
    "available_pools",
    "CircuitMetrics",
    "profile_ansatz",
]
