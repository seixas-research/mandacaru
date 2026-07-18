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

from .hamiltonian import MolecularIntegrals, minimal_fao_basis
from .mapping import Fermion, PauliSum, bravyi_kitaev, jordan_wigner, parity
from .planewave import PlaneWaveIntegrals, plane_wave_vectors

__all__ = [
    "MolecularIntegrals",
    "minimal_fao_basis",
    "PlaneWaveIntegrals",
    "plane_wave_vectors",
    "Fermion",
    "PauliSum",
    "jordan_wigner",
    "parity",
    "bravyi_kitaev",
]
