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
* the **NAO-AE** family (:mod:`carcara.basis.nao_ae`) -- all-electron
  numerical atomic orbitals: the LDA atom's own shells under a smooth wall,
  plus hydrogen-like polarization / diffuse tiers sized from the atom;
* :class:`GaussianOrbital` -- contracted Gaussian-type orbitals, used by the
  native STO-nG basis (:mod:`carcara.basis.sto_ng`) and by every **named
  Gaussian family** -- Pople (``6-31+G*``, ``6-311+G(2df,2p)``, ...), Dunning
  (``cc-pVDZ`` ... ``aug-cc-pVDZ``, ``cc-pCVDZ``) and Karlsruhe (``def2-SVP``,
  ``def2-TZVP``, ...) -- whose structure is parsed from the name and whose
  numbers are generated natively (:mod:`carcara.basis.gaussian_families`).

Use the :class:`BasisSet` factory to build NAO or (STO-nG) GTO bases.  All
families are generated from scratch -- no tabulated basis-set data.

:mod:`carcara.basis.atomic_solver` is the self-consistent spherical LDA atom
behind the NAO-AE minimal basis (and behind the experimental pseudopotentials
in :mod:`carcara.experimental.pseudopotentials`).
"""

from .atomic_solver import AtomicResult, solve_atom, solve_radial
from .base import BasisFunction
from .factory import (BasisSet, FAOBasisSet, GaussianBasisSet, GTOBasisSet,
                      NAOAEBasisSet, NAOBasisSet, PerElementBasisSet,
                      Pople631GBasisSet)
from .gaussian_families import (NAMED_BASIS_SETS, GaussianRecipe,
                                available_basis_names, count_functions,
                                gaussian_shells, parse_basis_name,
                                shell_notation)
from .gaussian import GaussianOrbital
from .fao import FullAtomicOrbital
from .nao import (DEFAULT_ENERGY_SHIFT, NumericalAtomicOrbital,
                  energy_shift_to_rc)
from .nao_ae import (RadialFunction, build_species, confinement_potential,
                     effective_charge_for_radius, hydrogenic_function,
                     tier_specification)
from .pople import pople_631g_shells
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
    "NAOAEBasisSet",
    "GTOBasisSet",
    "RadialFunction",
    "build_species",
    "confinement_potential",
    "effective_charge_for_radius",
    "hydrogenic_function",
    "tier_specification",
    "Pople631GBasisSet",
    "GaussianBasisSet",
    "PerElementBasisSet",
    "GaussianRecipe",
    "NAMED_BASIS_SETS",
    "available_basis_names",
    "parse_basis_name",
    "gaussian_shells",
    "count_functions",
    "shell_notation",
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
]
