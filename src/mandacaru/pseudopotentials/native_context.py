# -*- coding: utf-8 -*-
# file: pseudopotentials/native_context.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The native execution context the PAW radial routines run in."""

from __future__ import annotations

from functools import lru_cache

from .. import _native


@lru_cache(maxsize=1)
def context() -> _native.Context:
    """One single-threaded context shared by the radial routines of a
    process (dataset builds parallelize across processes, not inside one)."""
    return _native.Context(threads=1)
