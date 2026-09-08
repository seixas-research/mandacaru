# -*- coding: utf-8 -*-
# file: algorithms/vasqe.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Deprecated location of VASQE -- it now lives in :mod:`carcara.experimental`.

VASQE is still under development and was moved out of the stable
:mod:`carcara.algorithms` package.  Importing from here keeps old scripts
running but emits a :class:`DeprecationWarning`; use
``from carcara.experimental import VASQE`` instead.
"""

import warnings as _warnings

from ..experimental.vasqe import (  # noqa: F401  (re-exported for compatibility)
    SubspaceVASQE,
    TEMPERATURE_SCHEDULES,
    VASQE,
    VASQEResult,
    annealed_temperature,
    softmax_selection_probabilities,
)

_warnings.warn(
    "carcara.algorithms.vasqe has moved to carcara.experimental.vasqe (VASQE is "
    "an experimental method); import it from carcara.experimental instead.",
    DeprecationWarning, stacklevel=2)

__all__ = ["VASQE", "VASQEResult", "SubspaceVASQE", "TEMPERATURE_SCHEDULES",
           "annealed_temperature", "softmax_selection_probabilities"]
