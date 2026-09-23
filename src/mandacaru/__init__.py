# -*- coding: utf-8 -*-
# file: __init__.py

# This code is part of Mandacaru. 
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br> 

from .version import __version__


def __getattr__(name):
    # Lazy so `import mandacaru` stays light; `from mandacaru import Mandacaru` is
    # the recommended entry point.
    if name == "Mandacaru":
        from . import algorithms
        return getattr(algorithms, name)
    if name == "BlochCalculator":
        from .algorithms.bloch import RETIRED_BLOCH_CLASS
        raise AttributeError(RETIRED_BLOCH_CLASS)
    raise AttributeError(f"module 'mandacaru' has no attribute {name!r}")


__all__ = ["__version__", "Mandacaru"]