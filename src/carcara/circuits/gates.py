# -*- coding: utf-8 -*-
# file: circuits/gates.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Fermionic excitation generators for coupled-cluster ansätze.

The unitary coupled-cluster ansatz is built from *anti-Hermitian* excitation
generators :math:`\kappa = T - T^\dagger`, whose exponential
:math:`e^{\theta \kappa}` is a particle-number-conserving unitary.  This module
provides the single and double excitation generators as
:class:`~carcara.core.mapping.Fermion` operators; a mapping (e.g. Jordan-Wigner)
turns each into a qubit :class:`~carcara.core.mapping.PauliSum`, and the ansatz
(:mod:`carcara.circuits.ansatz`) exponentiates them.
"""

from __future__ import annotations

from ..core.mapping import Fermion


def single_excitation(i: int, a: int) -> Fermion:
    r"""Anti-Hermitian single-excitation generator ``a+_a a_i - a+_i a_a``.

    Excites one particle from occupied spin-orbital ``i`` to virtual ``a``.
    """
    t = Fermion.creation(a) * Fermion.annihilation(i)
    return t - t.dagger()


def double_excitation(i: int, j: int, a: int, b: int) -> Fermion:
    r"""Anti-Hermitian double-excitation generator ``a+_a a+_b a_j a_i - h.c.``.

    Excites the pair of occupied spin-orbitals ``(i, j)`` into the virtual pair
    ``(a, b)``.
    """
    t = (Fermion.creation(a) * Fermion.creation(b)
         * Fermion.annihilation(j) * Fermion.annihilation(i))
    return t - t.dagger()
