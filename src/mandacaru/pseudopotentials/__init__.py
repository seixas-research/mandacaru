# -*- coding: utf-8 -*-
# file: pseudopotentials/__init__.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Pseudopotentials: valence-only Hamiltonians for the real-space engine.

Organized in **families** (:mod:`.families`, registry :data:`PSEUDO_FAMILIES`).
Two are shipped, both generated from Mandacaru's own self-consistent atom
(:mod:`mandacaru.basis.atomic_solver`) on the shared dataset layout
(:mod:`.dataset`) with the radial machinery of :mod:`.partial_waves`:
``"paw-lcao"``, Bloechl's projector augmented-wave datasets with an overlap
correction and frozen one-center terms (:mod:`.paw`); and ``"upaw-lcao"``
(alias ``"unitary-paw-lcao"``), the PAW construction with a vanishing norm
deficit, generated on demand.  The datasets are not part of the package: the
library is a repository checkout that ``MANDACARU_PAW_PATH`` names, set with
``mandacaru --set-paw DIR``, holding one folder per set (``lda-sr/`` by
default) (:mod:`.environment`).  The valence pseudo-atomic orbitals and
projectors are sampled on the real-space grid by :mod:`.orbitals`.  A family
is selected **as a basis**: ``basis="PAW-LCAO"``, ``basis={"name": "PAW-LCAO", "size": "DZP"}`` on any driver
(the family name is a basis name, with the multiple-zeta size hierarchy as
its options); see the *Pseudopotentials* guide of the manual.
"""

from .families import (COMMON_OPTIONS, DEFAULT_FAMILY, PSEUDO_FAMILIES,
                       FamilySpec, canonical_family_name, family_listing,
                       family_names, lookup_family, register_family,
                       resolve_family, unregister_family)
from .environment import (FAMILY_VARIABLES, LibraryPathError,
                          library_directory)
from .dataset import Channel, PseudoPotential
from .io import (FORMAT_VERSION, LIBRARY_ELEMENTS, LIBRARY_Z_MAX,
                 available_elements, library_file, load_pseudopotential,
                 save_pseudopotential)
from .orbitals import (KBProjector, PseudoAtomicOrbital, pseudo_basis,
                       valence_electrons)
from .partial_waves import log_derivative_ae
from .paw import (PAW_FAMILY, PAWChannel, PAWDataset, PAWIntegrals,
                  build_paw_library, check_paw_channel, compensation_coulomb,
                  compensation_potential, compensation_shape, generate_paw,
                  get_paw, log_derivative_paw, paw_coupling_blocks,
                  paw_eigenstate, paw_library_path, paw_overlap_blocks,
                  paw_projectors, paw_spectrum, reconstruct_ae, report_paw,
                  smooth_partial_waves)
from .paw import (UPAW_FAMILY, build_upaw_library, generate_upaw, get_upaw,
                  upaw_library_path)

__all__ = [
    "COMMON_OPTIONS", "DEFAULT_FAMILY", "PSEUDO_FAMILIES", "FamilySpec",
    "canonical_family_name", "family_listing", "family_names",
    "lookup_family", "register_family", "resolve_family",
    "unregister_family",
    "FAMILY_VARIABLES", "LibraryPathError", "library_directory",
    "Channel", "PseudoPotential",
    "FORMAT_VERSION", "LIBRARY_ELEMENTS", "LIBRARY_Z_MAX",
    "available_elements", "library_file", "load_pseudopotential",
    "save_pseudopotential",
    "KBProjector", "PseudoAtomicOrbital", "pseudo_basis", "valence_electrons",
    "log_derivative_ae",
    "PAW_FAMILY", "PAWChannel", "PAWDataset", "PAWIntegrals",
    "build_paw_library", "check_paw_channel", "compensation_coulomb",
    "compensation_potential", "compensation_shape", "generate_paw", "get_paw",
    "log_derivative_paw", "paw_coupling_blocks", "paw_eigenstate",
    "paw_library_path", "paw_overlap_blocks", "paw_projectors", "paw_spectrum",
    "reconstruct_ae", "report_paw", "smooth_partial_waves",
    "UPAW_FAMILY", "build_upaw_library", "generate_upaw", "get_upaw",
    "upaw_library_path",
]
