# -*- coding: utf-8 -*-
# file: basis/base.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Basis-function contract of the tabulated radial shells.

The functions that survive in Python are the tabulated radial shells of the PAW-LCAO basis (:class:`~mandacaru.pseudopotentials.orbitals.PseudoShell` and the pseudopotential
shells); every basis the calculation samples is built and evaluated by the Rust engine. A shell
meets this contract by evaluating its value and its gradient at Cartesian points.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BasisFunction(ABC):
    """A localized single-particle function :math:`\\phi(\\mathbf r)`."""

    #: Cartesian center of the function in Bohr, shape ``(3,)``.
    center: np.ndarray

    #: Whether the function is real: an ``R(r) S_lm`` with real spherical
    #: harmonics (every molecular basis) is ``float64``; anything else
    #: (complex harmonics of a crystal) is ``complex128``.
    real: bool = False

    @abstractmethod
    def evaluate(self, x, y, z) -> np.ndarray:
        """Values on Cartesian coordinates (Bohr), broadcast to the shape of the inputs."""
        raise NotImplementedError

    @abstractmethod
    def gradient(self, x, y, z, delta: float = 1e-3) -> np.ndarray:
        """``(3, n)`` gradient at the points ``x, y, z`` (flat arrays, Bohr)."""
        raise NotImplementedError
