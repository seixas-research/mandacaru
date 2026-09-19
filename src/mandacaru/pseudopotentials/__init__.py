# -*- coding: utf-8 -*-
# file: pseudopotentials/__init__.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Pseudopotentials: valence-only Hamiltonians for the real-space engine.

Organized in **families** (:mod:`.families`, registry :data:`PSEUDO_FAMILIES`).
Three are shipped: ``"ncpp"`` (aliases ``"tm"``, ``"ncpp-tm"``), the
Troullier-Martins pseudization of the self-consistent LDA atom
(:mod:`mandacaru.basis.atomic_solver`) with Kleinman-Bylander separable
projectors (:mod:`.generation`); ``"oncvpsp"`` (alias ``"oncv"``),
Hamann's optimized norm-conserving Vanderbilt potentials with two projectors
per channel (:mod:`.oncv`); and ``"paw"``, Bloechl's projector augmented-wave
datasets with an overlap correction and frozen one-center terms
(:mod:`.paw`).  The on-disk library lives under ``library/``, one
subdirectory per family -- ``library/ncpp/`` (Troullier-Martins, :mod:`.io`),
``library/oncvpsp/`` and ``library/paw/`` -- and the valence
pseudo-atomic orbitals and projectors are sampled on the real-space grid by
:mod:`.orbitals`.  A family is selected **as a basis**: ``basis="NCPP"``,
``basis="ONCVPSP"``, ``basis={"name": "PAW", "size": "DZP"}`` on any driver
(the family name is a basis name, with the multiple-zeta size hierarchy as
its options); see the *Pseudopotentials* guide of the manual.
"""

from .families import (COMMON_OPTIONS, DEFAULT_FAMILY, PSEUDO_FAMILIES,
                       FamilySpec, canonical_family_name, family_listing,
                       family_names, lookup_family, register_family,
                       resolve_family, unregister_family)
from .generation import (Channel, PseudoPotential, check_channel,
                         generate_pseudopotential, pseudize_channel, report)
from .io import (FORMAT_VERSION, LEGACY_FAMILY, LIBRARY_ELEMENTS,
                 LIBRARY_Z_MAX, available_elements, build_library,
                 default_library_path, get_pseudopotential, library_root, library_file,
                 load_pseudopotential, save_pseudopotential)
from .orbitals import (KBProjector, PseudoAtomicOrbital, kb_coupling_blocks,
                       kb_projectors, pseudo_basis, valence_electrons)
from .oncv import (ONCV_FAMILY, ONCVChannel, ONCVPseudoPotential,
                   build_oncv_library, check_oncv_channel,
                   diagonalized_projectors, generate_oncv, get_oncv,
                   log_derivative_ae, log_derivative_ps, oncv_coupling_blocks,
                   oncv_library_path, oncv_projectors, radial_spectrum,
                   report_oncv)
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
    "Channel", "PseudoPotential", "check_channel", "generate_pseudopotential",
    "pseudize_channel", "report",
    "FORMAT_VERSION", "LEGACY_FAMILY",
    "LIBRARY_ELEMENTS", "LIBRARY_Z_MAX", "available_elements", "build_library",
    "default_library_path", "get_pseudopotential", "library_root", "library_file",
    "load_pseudopotential", "save_pseudopotential",
    "KBProjector", "PseudoAtomicOrbital", "kb_coupling_blocks",
    "kb_projectors", "pseudo_basis", "valence_electrons",
    "ONCV_FAMILY", "ONCVChannel", "ONCVPseudoPotential", "build_oncv_library",
    "check_oncv_channel", "diagonalized_projectors", "generate_oncv",
    "get_oncv", "log_derivative_ae", "log_derivative_ps",
    "oncv_coupling_blocks", "oncv_library_path", "oncv_projectors",
    "radial_spectrum", "report_oncv",
    "PAW_FAMILY", "PAWChannel", "PAWDataset", "PAWIntegrals",
    "build_paw_library", "check_paw_channel", "compensation_coulomb",
    "compensation_potential", "compensation_shape", "generate_paw", "get_paw",
    "log_derivative_paw", "paw_coupling_blocks", "paw_eigenstate",
    "paw_library_path", "paw_overlap_blocks", "paw_projectors", "paw_spectrum",
    "reconstruct_ae", "report_paw", "smooth_partial_waves",
    "UPAW_FAMILY", "build_upaw_library", "generate_upaw", "get_upaw",
    "upaw_library_path",
]
