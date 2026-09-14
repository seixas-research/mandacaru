# -*- coding: utf-8 -*-
# file: experimental/pseudopotentials/__init__.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Pseudopotentials -- **experimental**.

Organized in **families** (:mod:`.families`, registry :data:`PSEUDO_FAMILIES`).
Three are shipped: ``"tm"`` (aliases ``"ncpp"``, ``"ncpp-tm"``), the
Troullier-Martins pseudization of the self-consistent LDA atom
(:mod:`carcara.basis.atomic_solver`) with Kleinman-Bylander separable
projectors (:mod:`.generation`); ``"oncvpsp"`` (alias ``"oncv"``),
Hamann's optimized norm-conserving Vanderbilt potentials with two projectors
per channel (:mod:`.oncv`); and ``"paw"``, Bloechl's projector augmented-wave
datasets with an overlap correction and frozen one-center terms
(:mod:`.paw`).  The on-disk library lives under ``library/``
(:mod:`.io`; the ONCVPSP files in ``library/oncvpsp/``, the PAW files in
``library/paw/``), and the valence
pseudo-atomic orbitals and projectors are sampled on the real-space grid by
:mod:`.orbitals`.  Every driver reaches a family through
``pseudopotentials=True | "tm" | "oncv" | {"family": ..., ...}``; see
``docs/experimental/pseudopotentials.md``.
"""

from .families import (DEFAULT_FAMILY, PSEUDO_FAMILIES, FamilySpec,
                       family_names, normalize_pseudopotentials,
                       register_family, resolve_family,
                       unregister_family)
from .generation import (Channel, PseudoPotential, check_channel,
                         generate_pseudopotential, pseudize_channel, report)
from .io import (FORMAT_VERSION, LEGACY_FAMILY, LIBRARY_ELEMENTS,
                 LIBRARY_Z_MAX, available_elements, build_library,
                 default_library_path, get_pseudopotential, library_file,
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

__all__ = [
    "DEFAULT_FAMILY", "PSEUDO_FAMILIES", "FamilySpec", "family_names",
    "normalize_pseudopotentials", "register_family", "resolve_family",
    "unregister_family",
    "Channel", "PseudoPotential", "check_channel", "generate_pseudopotential",
    "pseudize_channel", "report",
    "FORMAT_VERSION", "LEGACY_FAMILY",
    "LIBRARY_ELEMENTS", "LIBRARY_Z_MAX", "available_elements", "build_library",
    "default_library_path", "get_pseudopotential", "library_file",
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
]
