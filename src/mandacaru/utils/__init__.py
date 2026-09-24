# -*- coding: utf-8 -*-
# file: utils/__init__.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Utilities (structured logging, run dumps, bibliography, profiling, ...)."""

from .bibliography import REFERENCES, bibtex
from .citations import citation_keys, write_references
from .cube import write_cube, write_volumetric, write_xsf
from .dumps import dump_hamiltonian, dump_pool, resolve_dump_path
from .logging import (AdaptOutputLogger, append_block, append_forces,
                      append_optimization_summary, append_performance,
                      append_quantum_echoes,
                      log_steps, parse_output, reset_log)
from .profiling import (Timings, backend_cores, cpu_count, current_memory_mb,
                         peak_memory_mb)

__all__ = ["AdaptOutputLogger", "parse_output", "append_block",
           "append_quantum_echoes",
           "append_forces",
           "append_optimization_summary", "append_performance",
           "log_steps", "reset_log",
           "dump_hamiltonian", "dump_pool", "resolve_dump_path",
           "Timings", "backend_cores", "cpu_count", "current_memory_mb",
           "peak_memory_mb",
           "write_cube", "write_xsf", "write_volumetric",
           "REFERENCES", "bibtex", "citation_keys",
           "write_references"]
