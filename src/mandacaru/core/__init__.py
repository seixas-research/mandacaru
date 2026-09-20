# -*- coding: utf-8 -*-
# file: core/__init__.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Electronic Hamiltonians and fermion-to-qubit mappings.

* :class:`MolecularIntegrals` -- one-/two-body integrals and the molecular
  Hamiltonian over a localized basis;
* :class:`Fermion` -- second-quantized fermionic operators and the four
  fermion-to-qubit mappings (Jordan-Wigner, parity, reduced parity,
  Bravyi-Kitaev);
* :class:`PauliSum` -- the qubit-operator output type;
* :func:`save_hamiltonian` / :func:`load_hamiltonian` -- the on-disk Pauli-string
  cache (Apache Parquet or JSON) that lets a run skip the integrals and the
  mapping entirely; the format is auto-detected on load.
"""

from .hamiltonian import MolecularIntegrals, minimal_fao_basis
from .mapping import (Fermion, PauliSum, bravyi_kitaev, jordan_wigner, parity,
                      parity_reduced)
from .planewave import PlaneWaveIntegrals, plane_wave_vectors
from .serialization import (HAMILTONIAN_FORMATS, HamiltonianRecord,
                            HamiltonianHeader, detect_format,
                            load_hamiltonian, read_hamiltonian_header,
                            save_hamiltonian)
from .checkpoint import (WavefunctionCheckpoint, load_checkpoint,
                         prepare_state)

__all__ = [
    "MolecularIntegrals",
    "minimal_fao_basis",
    "PlaneWaveIntegrals",
    "plane_wave_vectors",
    "Fermion",
    "PauliSum",
    "jordan_wigner",
    "parity",
    "parity_reduced",
    "bravyi_kitaev",
    "HamiltonianRecord",
    "HAMILTONIAN_FORMATS",
    "save_hamiltonian",
    "load_hamiltonian",
    "read_hamiltonian_header",
    "HamiltonianHeader",
    "detect_format",
    "WavefunctionCheckpoint",
    "load_checkpoint",
    "prepare_state",
]
