# -*- coding: utf-8 -*-
# file: basis/__init__.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Localized single-particle basis functions.

Import the abstract :class:`BasisFunction` to inject a custom basis (e.g. future
Wannier functions) and :class:`HydrogenicOrbital` for the built-in analytic
hydrogen-like orbitals.
"""

from .base import BasisFunction
from .hydrogenic import HydrogenicOrbital

__all__ = ["BasisFunction", "HydrogenicOrbital"]
