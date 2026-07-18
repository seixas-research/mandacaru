# -*- coding: utf-8 -*-
# file: algorithms/vasqe.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

def _prob(x: np.ndarray, tau: float = 1.0):
    return np.exp(x/tau) / sum(np.exp(x/tau))