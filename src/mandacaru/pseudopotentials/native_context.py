# -*- coding: utf-8 -*-
# file: pseudopotentials/native_context.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The native execution context the PAW radial routines run in."""

from functools import lru_cache
import os

from .. import _native


@lru_cache(maxsize=8)
def context(threads=None):
    return _native.Context(threads=threads or int(os.environ.get("OMP_NUM_THREADS", "8")))
