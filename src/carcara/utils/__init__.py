# -*- coding: utf-8 -*-
# file: utils/__init__.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Utilities (structured logging, run dumps, profiling, ...)."""

from .dumps import dump_hamiltonian, dump_pool, resolve_dump_path
from .logging import AdaptOutputLogger, parse_output
from .profiling import Timings, backend_cores, peak_memory_mb

__all__ = ["AdaptOutputLogger", "parse_output",
           "dump_hamiltonian", "dump_pool", "resolve_dump_path",
           "Timings", "backend_cores", "peak_memory_mb"]
