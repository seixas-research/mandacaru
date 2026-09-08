# -*- coding: utf-8 -*-
# file: experimental/__init__.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Experimental algorithms -- under development, not part of the stable API.

Everything in this package is usable but **not yet fully validated**: the
interfaces may change between releases, the numerics have been exercised on
small molecules only, and the methods are deliberately kept out of the main
documentation build (their notes live in ``docs/experimental/``).  Nothing here
is a default anywhere in Carcará -- the production adaptive solver is
:class:`~carcara.algorithms.adapt_vqe.ADAPTVQE`.

Current contents:

* :mod:`carcara.experimental.vasqe` -- **VASQE**, the Variational Adaptive
  Stochastic Quantum Eigensolver (ADAPT-VQE with softmax operator selection and
  temperature annealing), plus its subspace-search variant
  :class:`~carcara.experimental.vasqe.SubspaceVASQE`.

The experimental methods remain reachable from the unified calculator by name
(``QuantumCalculator(method="vasqe")`` / ``"subspace-vasqe"``), which imports
them lazily from here.
"""

from .vasqe import (
    SubspaceVASQE,
    TEMPERATURE_SCHEDULES,
    VASQE,
    VASQEResult,
    annealed_temperature,
    softmax_selection_probabilities,
)

__all__ = [
    "VASQE",
    "VASQEResult",
    "SubspaceVASQE",
    "TEMPERATURE_SCHEDULES",
    "annealed_temperature",
    "softmax_selection_probabilities",
]
