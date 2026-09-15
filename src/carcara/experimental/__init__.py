# -*- coding: utf-8 -*-
# file: experimental/__init__.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Experimental features -- under development, not part of the stable API.

Everything in this package is usable but **not yet fully validated**: the
interfaces may change between releases, the numerics have been exercised on
small systems only, and the features are deliberately kept out of the main
documentation build (their notes live in ``docs/experimental/``) and off the
website.  Nothing here is a default anywhere in Carcará.

Contents (VASQE only -- the pseudopotentials graduated to
:mod:`carcara.pseudopotentials` on 2026-09-15):

* :mod:`carcara.experimental.vasqe` -- **VASQE**, the Variational Adaptive
  Stochastic Quantum Eigensolver (ADAPT-VQE with softmax operator selection
  and temperature annealing), plus its subspace-search variant
  :class:`~carcara.experimental.vasqe.SubspaceVASQE`.  Importing this package
  registers them with the unified calculator, so after
  ``import carcara.experimental`` the names ``Carcara(method="vasqe")`` and
  ``Carcara(method="subspace-vasqe")`` work; without that import they are
  unknown, by design.
"""

from ..algorithms.calculator import register_method
from .vasqe import (
    SubspaceVASQE,
    TEMPERATURE_SCHEDULES,
    VASQE,
    VASQEResult,
    annealed_temperature,
    softmax_selection_probabilities,
)

register_method("vasqe", VASQE)
register_method("subspace-vasqe", SubspaceVASQE)

__all__ = [
    "VASQE",
    "VASQEResult",
    "SubspaceVASQE",
    "TEMPERATURE_SCHEDULES",
    "annealed_temperature",
    "softmax_selection_probabilities",
]
