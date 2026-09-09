# -*- coding: utf-8 -*-
# file: __init__.py

# This code is part of Carcará. 
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br> 

from .version import __version__


def __getattr__(name):
    # Lazy so `import carcara` stays light; `from carcara import Carcara` is
    # the recommended entry point.
    if name in ("Carcara", "BlochCalculator"):
        from . import algorithms
        return getattr(algorithms, name)
    raise AttributeError(f"module 'carcara' has no attribute {name!r}")


__all__ = ["__version__", "Carcara", "BlochCalculator"]