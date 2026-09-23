# -*- coding: utf-8 -*-
# file: experimental/__init__.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Experimental features -- under development, not part of the stable API.

Everything in this package is usable but **not yet fully validated**: the
interfaces may change between releases, the numerics have been exercised on
small systems only, and the features are deliberately kept out of the main
documentation build (their notes live in ``docs/experimental/``) and off the
website.  Nothing here is a default anywhere in Mandacaru.

Contents (VASQE only -- the pseudopotentials graduated to
:mod:`mandacaru.pseudopotentials` on 2026-09-15):

* :mod:`mandacaru.experimental.vasqe` -- **VASQE**, the Variational Adaptive
  Stochastic Quantum Eigensolver (ADAPT-VQE with softmax operator selection
  and temperature annealing), plus its subspace-search variant
  :class:`~mandacaru.experimental.vasqe.SubspaceVASQE`.  Importing this package
  registers them with the unified calculator, so after
  ``import mandacaru.experimental`` the names ``Mandacaru(method="vasqe")`` and
  ``Mandacaru(method="subspace-vasqe")`` work; without that import they are
  unknown, by design.
"""

from ..algorithms.bloch import _BlochMixin
from ..algorithms.calculator import register_method
from .vasqe import (
    SubspaceVASQE,
    TEMPERATURE_SCHEDULES,
    VASQE,
    VASQEResult,
    annealed_temperature,
    softmax_selection_probabilities,
)



class BlochVASQE(_BlochMixin, VASQE):
    """VASQE over the Born-von Karman supercell (``method="bloch-vasqe"``).

    The periodic layer is composed over the solver exactly as the stable
    ``"bloch-vqe"`` / ``"bloch-adapt-vqe"`` are, so this method takes the same
    ``kpts`` and provides the same ``bands`` / ``get_spectral_function`` /
    ``get_fermi_level``.
    """


register_method("vasqe", VASQE)
register_method("subspace-vasqe", SubspaceVASQE)
register_method("bloch-vasqe", BlochVASQE)

__all__ = [
    "VASQE",
    "BlochVASQE",
    "VASQEResult",
    "SubspaceVASQE",
    "TEMPERATURE_SCHEDULES",
    "annealed_temperature",
    "softmax_selection_probabilities",
]
