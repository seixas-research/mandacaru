# -*- coding: utf-8 -*-
# file: pseudopotentials/native_context.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The native execution context the PAW radial routines run in."""

from functools import lru_cache

from .. import _native


@lru_cache(maxsize=8)
def context(threads=None):
    """The shared context of ``threads`` workers; ``None`` takes the one thread
    policy of the engine (``MANDACARU_NUM_THREADS``, ``OMP_NUM_THREADS``, then
    the hardware), which Rust owns."""
    return _native.Context(threads=threads)
