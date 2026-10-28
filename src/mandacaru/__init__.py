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
        from .calculation import Mandacaru
        return Mandacaru
    if name == "Geometry":
        from .structure import Geometry
        return Geometry
    if name == "CalculationResources":
        from .resources import CalculationResources
        return CalculationResources
    if name in ("Viewer3D", "Camera"):
        from .utils import viewer
        return getattr(viewer, name)
    raise AttributeError(f"module 'mandacaru' has no attribute {name!r}")


__all__ = ["__version__", "Mandacaru", "Geometry", "CalculationResources", "Viewer3D", "Camera"]
