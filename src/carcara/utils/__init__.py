# -*- coding: utf-8 -*-
# file: utils/__init__.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Utilities (structured logging, ...)."""

from .logging import AdaptOutputLogger, parse_output
from .profiling import Timings, backend_cores, peak_memory_mb

__all__ = ["AdaptOutputLogger", "parse_output",
           "Timings", "backend_cores", "peak_memory_mb"]
