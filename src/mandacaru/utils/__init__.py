# -*- coding: utf-8 -*-
# file: utils/__init__.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Utilities (structured logging, run dumps, bibliography, profiling, 3D viewer, ...)."""

from importlib import import_module

from .bibliography import REFERENCES, bibtex
from .logging import (Logger, append_block, append_forces,
                      append_optimization_summary, append_performance,
                      append_quantum_echoes, append_nested_otoc,
                      log_steps, parse_output, reset_log)
from .profiling import (Timings, backend_cores, cpu_count, current_memory_mb,
                         peak_memory_mb)
from .viewer import Camera, Viewer3D

_LAZY_EXPORTS = {
    "citation_keys": "citations",
    "write_references": "citations",
    "write_cube": "cube",
    "write_volumetric": "cube",
    "write_xsf": "cube",
    "dump_hamiltonian": "dumps",
    "dump_pool": "dumps",
    "resolve_dump_path": "dumps",
}


def __getattr__(name):
    """Defer utilities that depend on core until their exports are requested.

    Integral kernels import profiling while core is still being initialized.
    Loading serializers or file writers here would re-enter core too early.
    """
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


__all__ = ["Logger", "parse_output", "append_block",
           "append_quantum_echoes", "append_nested_otoc",
           "append_forces",
           "append_optimization_summary", "append_performance",
           "log_steps", "reset_log",
           "dump_hamiltonian", "dump_pool", "resolve_dump_path",
           "Timings", "backend_cores", "cpu_count", "current_memory_mb",
           "peak_memory_mb",
           "write_cube", "write_xsf", "write_volumetric",
           "Viewer3D", "Camera",
           "REFERENCES", "bibtex", "citation_keys",
           "write_references"]
