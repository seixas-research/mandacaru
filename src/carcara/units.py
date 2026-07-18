# -*- coding: utf-8 -*-
# file: units.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Unit conventions and conversions.

Carcará computes in **atomic units** internally -- lengths in Bohr, energies in
Hartree -- because that is what the real-space integral kernels, the
finite-difference Laplacian and the FAO (Full Atomic Orbital) orbitals are written in.  Users,
however, usually think in **Angstrom** and **electronvolts**.

This module is the single source of truth for the conversion factors and small
helpers.  The user-facing classes (``Grid``, ``FullAtomicOrbital``,
``Potentials``, ``IntegralEngine``) accept lengths in Angstrom and return
energies in eV *by default*, converting to/from atomic units at their boundary,
while the numerical core (the integral engine, the C backend) works throughout
in atomic units (Bohr, Hartree).
"""

from __future__ import annotations

import numpy as np

# --- Physical conversion factors (CODATA-consistent) --------------------- #
BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM  # 1.8897259886...
HARTREE_TO_EV = 27.211386245988
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV

# Accepted spellings for the public ``units`` / ``energy_units`` arguments.
_LENGTH_TO_BOHR = {"angstrom": ANGSTROM_TO_BOHR, "a": ANGSTROM_TO_BOHR,
                   "bohr": 1.0, "au": 1.0, "a0": 1.0}
_ENERGY_FROM_HARTREE = {"ev": HARTREE_TO_EV,
                        "ha": 1.0, "hartree": 1.0, "au": 1.0}


def to_bohr(length, units: str = "angstrom"):
    """Convert a length (scalar or array) in ``units`` to Bohr."""
    try:
        factor = _LENGTH_TO_BOHR[units.lower()]
    except KeyError:
        raise ValueError(
            f"unknown length unit {units!r}; use 'angstrom' or 'bohr'") from None
    return np.asarray(length, dtype=float) * factor


def from_hartree(energy, units: str = "eV"):
    """Convert an energy (scalar or array, possibly complex) from Hartree to ``units``."""
    try:
        factor = _ENERGY_FROM_HARTREE[units.lower()]
    except KeyError:
        raise ValueError(
            f"unknown energy unit {units!r}; use 'eV' or 'Ha'") from None
    return energy * factor
