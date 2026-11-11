# -*- coding: utf-8 -*-
# file: units.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Unit conventions and conversions.

Mandacaru computes in **atomic units** internally -- lengths in Bohr, energies in
Hartree -- because that is what the real-space integral kernels, the
grid Laplacians and the HAO (Hydrogenic Atomic Orbital) orbitals are written in.  Users,
however, usually think in **Angstrom** and **electronvolts**.

This module is the single source of truth for the conversion factors and small
helpers.  The user-facing ``Grid`` accepts lengths in Angstrom and the results return
energies in eV *by default*, converting to/from atomic units at their boundary,
while the numerical core (the Rust engine) works throughout in atomic units
(Bohr, Hartree).

**Convention.**  Everything a user *sees* -- the result objects of
the variational drivers (``optimal_energy``, ``reference_energy``, energy
histories, ``EnergyLevels``, the subspace results, ``InteractionEnergy``,
``ForceResult``), the verbose printouts, the CLI, the band structures and the
hardware measurements -- is in **eV and Angstrom**.  Hartree and Bohr survive
only inside the internal layers: the integral engine, the second-quantized /
qubit Hamiltonians (``Fermion`` / ``PauliSum`` coefficients, the Hamiltonian
cache files), the basis and pseudopotential records and the SCF / gradient
mathematics.  ``atomic_units=True`` on a driver is the single opt-in that
switches its outputs to Hartree / Bohr.  Each conversion happens exactly once,
where a result object or a printout is built.
"""

from __future__ import annotations

import numpy as np

# --- Physical conversion factors (CODATA-consistent), defined in Rust ---- #
from . import _native
from ._native import BOHR_TO_ANGSTROM, HARTREE_TO_EV

#: The inverse factors, defined in Rust (assigned here so that converting by them reads as a
#: unit conversion to the numerics gate, ``scripts/numerics_budget.py``).
ANGSTROM_TO_BOHR = _native.ANGSTROM_TO_BOHR
EV_TO_HARTREE = _native.EV_TO_HARTREE

__all__ = [
    "ANGSTROM_TO_BOHR", "AU_DIPOLE_TO_DEBYE", "BOHR_TO_ANGSTROM", "DEFAULT_GRID_SPACING",
    "EV_TO_HARTREE", "HARTREE_TO_EV", "convert_energy", "energy_unit_label", "from_hartree",
    "to_angstrom", "to_bohr", "to_hartree",
]
#: One atomic unit of electric dipole moment (e Bohr) in debye: CODATA
#: 8.4783536255e-30 C m over 1 D = 1e-21 / c C m = 3.33564095198e-30 C m.
AU_DIPOLE_TO_DEBYE = 2.541746473

#: Default real-space ``grid_spacing`` (Angstrom) of every entry point -- the
#: drivers, the calculator, the Bloch driver, the command line and the grid.
DEFAULT_GRID_SPACING = 0.20

# The accepted spellings of the public ``units`` / ``energy_units`` arguments live with the
# conversions in Rust (``units::energy_factor``, ``units::length_factor_to_bohr``).


def _numeric(value):
    """``value`` as the native conversions take it: a float, a complex or a float/complex array."""
    if isinstance(value, (bool, int, float, complex, np.floating, np.complexfloating)):
        return value if isinstance(value, (complex, np.complexfloating)) else float(value)
    array = np.asarray(value)
    return np.ascontiguousarray(array, dtype=complex if array.dtype.kind == "c" else float)


def to_bohr(length, units: str = "angstrom"):
    """Convert a length (scalar or array) in ``units`` to Bohr."""
    from ._native import length_to_bohr
    return length_to_bohr(_numeric(length), str(units))


def to_angstrom(length_bohr):
    """Convert a length (float or float array) in Bohr to Angstrom."""
    from ._native import to_user_units
    if isinstance(length_bohr, (bool, int, float, np.floating)):
        return to_user_units(float(length_bohr), "length")
    return to_user_units(np.ascontiguousarray(length_bohr, dtype=float), "length")


def _energy_factor(units: str) -> float:
    """Hartree -> ``units`` factor for an accepted energy-unit spelling."""
    from ._native import energy_unit_factor
    if not isinstance(units, str):
        raise ValueError(f"unknown energy unit {units!r}; use 'eV' or 'Ha'")
    return energy_unit_factor(units)


def energy_unit_label(units: str) -> str:
    """Canonical label -- ``"eV"`` or ``"Ha"`` -- of an accepted energy unit."""
    return "Ha" if _energy_factor(units) == 1.0 else "eV"


def from_hartree(energy, units: str = "eV"):
    """Convert an energy (scalar or array, possibly complex) from Hartree to ``units``."""
    from ._native import energy_from_hartree
    _energy_factor(units)
    return energy_from_hartree(_numeric(energy), units)


def to_hartree(energy, units: str = "eV"):
    """Convert an energy (scalar or array) in ``units`` to Hartree."""
    from ._native import energy_to_hartree
    _energy_factor(units)
    return energy_to_hartree(_numeric(energy), units)


def convert_energy(energy, from_units: str, to_units: str):
    """Convert an energy (scalar or array) from ``from_units`` to ``to_units``.

    The factor is the ratio of the two Hartree factors, so this never
    introduces a conversion constant of its own; equal units return the value.
    """
    from ._native import convert_energy as _convert
    _energy_factor(from_units)
    _energy_factor(to_units)
    if _energy_factor(to_units) == _energy_factor(from_units):
        return energy
    return _convert(_numeric(energy), from_units, to_units)
