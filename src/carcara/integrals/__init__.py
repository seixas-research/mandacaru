# -*- coding: utf-8 -*-
# file: integrals/__init__.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Real-space integral engine and its high-performance C backend."""

from . import _backend
from ._backend import (BackendStatus, build_backend, check_backend,
                       ensure_backend)
from .engine import IntegralEngine
from .grid import Grid
from .poisson import PoissonFFTSolver
from .potentials import Potentials

__all__ = ["IntegralEngine", "Grid", "PoissonFFTSolver", "Potentials",
           "HAS_C_BACKEND", "BackendStatus", "build_backend", "check_backend",
           "ensure_backend"]


def __getattr__(name):
    """``HAS_C_BACKEND`` read live, so it cannot go stale.

    It used to be imported by value, which meant ``from carcara.integrals
    import HAS_C_BACKEND`` kept saying ``False`` after the library had been
    compiled in this same process -- the build is automatic and happens on the
    first integration, so that was the common case.  Resolving it on each
    access (PEP 562) removes the trap; :func:`check_backend` remains the fuller
    report.
    """
    if name == "HAS_C_BACKEND":
        return _backend.HAS_C_BACKEND
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
