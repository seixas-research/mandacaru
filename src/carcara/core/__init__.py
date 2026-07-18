# -*- coding: utf-8 -*-
# file: core/__init__.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Electronic Hamiltonians and fermion-to-qubit mappings.

* :class:`MolecularIntegrals` -- one-/two-body integrals and the molecular
  Hamiltonian over a localized basis;
* :class:`Fermion` -- second-quantized fermionic operators and the three
  fermion-to-qubit mappings (Jordan-Wigner, parity, Bravyi-Kitaev);
* :class:`PauliSum` -- the qubit-operator output type.
"""

from .hamiltonian import (HydrogenicIntegrals, MolecularIntegrals,
                          minimal_fao_basis, minimal_hydrogenic_basis)
from .mapping import Fermion, PauliSum, bravyi_kitaev, jordan_wigner, parity

__all__ = [
    "MolecularIntegrals",
    "HydrogenicIntegrals",       # alias of MolecularIntegrals
    "minimal_fao_basis",
    "minimal_hydrogenic_basis",  # alias of minimal_fao_basis
    "Fermion",
    "PauliSum",
    "jordan_wigner",
    "parity",
    "bravyi_kitaev",
]
