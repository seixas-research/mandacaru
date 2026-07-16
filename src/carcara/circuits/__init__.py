# -*- coding: utf-8 -*-
# file: circuits/__init__.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Parameterized circuits: ansätze and excitation-gate generators."""

from .ansatz import UCCSD
from .gates import double_excitation, single_excitation

__all__ = ["UCCSD", "single_excitation", "double_excitation"]
