# -*- coding: utf-8 -*-
# file: circuits/__init__.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Parameterized circuits: ansätze, excitation-gate generators and operator pools."""

from .ansatz import UCCSD
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

__all__ = [
    "UCCSD",
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
]
