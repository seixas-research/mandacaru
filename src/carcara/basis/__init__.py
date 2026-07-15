# -*- coding: utf-8 -*-
# file: basis/__init__.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Localized single-particle basis functions.

Every basis function implements the :class:`BasisFunction` contract (sample on a
grid), so any of them drops straight into the integral engine.  Built-ins:

* :class:`HydrogenicOrbital` -- analytic hydrogen-like orbitals (with Slater
  effective charges);
* :class:`NumericalAtomicOrbital` -- confined Sankey/SIESTA-type numerical
  orbitals on a radial grid;
* :class:`GaussianOrbital` -- contracted Gaussian-type orbitals.

Use the :class:`BasisSet` factory to build NAO or GTO bases by name.
"""

from .base import BasisFunction
from .factory import BasisSet, GTOBasisSet, NAOBasisSet
from .gaussian import GaussianOrbital
from .gto_data import available_bases, parse_nwchem, register
from .hydrogenic import HydrogenicOrbital
from .nao import (DEFAULT_ENERGY_SHIFT, NumericalAtomicOrbital,
                  energy_shift_to_rc)

__all__ = [
    "BasisFunction",
    "HydrogenicOrbital",
    "NumericalAtomicOrbital",
    "GaussianOrbital",
    "BasisSet",
    "NAOBasisSet",
    "GTOBasisSet",
    "energy_shift_to_rc",
    "DEFAULT_ENERGY_SHIFT",
    "register",
    "parse_nwchem",
    "available_bases",
]
