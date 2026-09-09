# -*- coding: utf-8 -*-
# file: experimental/pseudopotentials/__init__.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Norm-conserving pseudopotentials -- **experimental**.

Troullier-Martins pseudization of the self-consistent LDA atom
(:mod:`carcara.basis.atomic_solver`) with Kleinman-Bylander separable
projectors (:mod:`.generation`), the on-disk library shipped under
``library/`` (:mod:`.io`), and the valence pseudo-atomic orbitals and
projectors sampled on the real-space grid (:mod:`.orbitals`).  Every driver
reaches this through ``pseudopotentials=True``; see
``docs/experimental/pseudopotentials.md``.
"""

from .generation import (Channel, PseudoPotential, check_channel,
                         generate_pseudopotential, pseudize_channel, report)
from .io import (LIBRARY_ELEMENTS, LIBRARY_Z_MAX, available_elements,
                 build_library, default_library_path, get_pseudopotential,
                 library_file, load_pseudopotential, save_pseudopotential)
from .orbitals import (KBProjector, PseudoAtomicOrbital, kb_projectors,
                       pseudo_basis, valence_electrons)

__all__ = [
    "Channel", "PseudoPotential", "check_channel", "generate_pseudopotential",
    "pseudize_channel", "report",
    "LIBRARY_ELEMENTS", "LIBRARY_Z_MAX", "available_elements", "build_library",
    "default_library_path", "get_pseudopotential", "library_file",
    "load_pseudopotential", "save_pseudopotential",
    "KBProjector", "PseudoAtomicOrbital", "kb_projectors", "pseudo_basis",
    "valence_electrons",
]
