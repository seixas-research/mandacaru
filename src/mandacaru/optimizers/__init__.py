# -*- coding: utf-8 -*-
# file: optimizers/__init__.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Classical optimizers for hybrid variational loops."""

from .optim import (DEFAULT_MAXITER, DEFAULT_OPTIMIZER, DEFAULT_TOL,
                    NAMED_OPTIMIZERS, Optimizer)

__all__ = ["Optimizer", "DEFAULT_OPTIMIZER", "DEFAULT_MAXITER", "DEFAULT_TOL",
           "NAMED_OPTIMIZERS"]
