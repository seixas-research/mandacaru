# -*- coding: utf-8 -*-
# file: optimizers/__init__.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Classical optimizers for hybrid variational loops."""

from .optim import NAMED_OPTIMIZERS, Optimizer, OptimizeResult, resolve_optimizer

__all__ = ["Optimizer", "OptimizeResult", "resolve_optimizer",
           "NAMED_OPTIMIZERS"]
