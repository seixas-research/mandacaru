# -*- coding: utf-8 -*-
# file: basis/__init__.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Localized single-particle basis functions.

Every basis function implements the :class:`BasisFunction` contract (sample on a
grid), so any of them drops straight into the integral engine.  Built-ins:

* :class:`FullAtomicOrbital` -- analytic hydrogen-like orbitals (with Slater
  effective charges);
* :class:`NumericalAtomicOrbital` -- confined Sankey/SIESTA-type numerical
  orbitals on a radial grid;
* :class:`GaussianOrbital` -- contracted Gaussian-type orbitals, used by the
  native STO-nG basis (:mod:`carcara.basis.sto_ng`).

Use the :class:`BasisSet` factory to build NAO or (STO-nG) GTO bases.  All
families are generated from scratch -- no tabulated basis-set data.

Two further modules support **pseudopotentials**, which remove the heavy-atom
core that the real-space grid cannot resolve:

* :mod:`carcara.basis.atomic_solver` -- the self-consistent spherical LDA atom
  that provides the all-electron reference;
* :mod:`carcara.basis.pseudopotential` -- Troullier-Martins norm-conserving
  pseudization in Kleinman-Bylander separable form.
"""

from .atomic_solver import AtomicResult, solve_atom, solve_radial
from .base import BasisFunction
from .factory import (BasisSet, FAOBasisSet, GTOBasisSet, NAOBasisSet,
                      Pople631GBasisSet)
from .gaussian import GaussianOrbital
from .fao import FullAtomicOrbital
from .nao import (DEFAULT_ENERGY_SHIFT, NumericalAtomicOrbital,
                  energy_shift_to_rc)
from .pople import pople_631g_shells
from .pseudopotential import (Channel, PseudoPotential, check_channel,
                              generate_pseudopotential, pseudize_channel,
                              report)
from .sto_ng import (occupied_subshells, slater_exponent, sto_ng_contraction,
                     sto_ng_shells)

__all__ = [
    "BasisFunction",
    "FullAtomicOrbital",
    "NumericalAtomicOrbital",
    "GaussianOrbital",
    "BasisSet",
    "FAOBasisSet",
    "NAOBasisSet",
    "GTOBasisSet",
    "Pople631GBasisSet",
    "energy_shift_to_rc",
    "DEFAULT_ENERGY_SHIFT",
    "sto_ng_contraction",
    "sto_ng_shells",
    "pople_631g_shells",
    "slater_exponent",
    "occupied_subshells",
    "solve_atom",
    "solve_radial",
    "AtomicResult",
    "generate_pseudopotential",
    "PseudoPotential",
    "Channel",
    "pseudize_channel",
    "check_channel",
    "report",
]
