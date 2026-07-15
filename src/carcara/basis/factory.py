# -*- coding: utf-8 -*-
# file: basis/factory.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Unified factory for building localized basis sets.

``BasisSet.build`` selects a basis-set family and returns an object that turns
elements/geometries into lists of :class:`~carcara.basis.base.BasisFunction`
ready for :class:`~carcara.integrals.IntegralEngine`:

.. code-block:: python

    nao = BasisSet.build(method="NAO", energy_shift=0.03)
    gto = BasisSet.build(method="GTO", name="6-31G(d)")

    orbitals = gto.atom("O", center=[0.0, 0.0, 0.0])          # one atom
    basis = gto.molecule(["H", "H"], [[0, 0, 0], [0, 0, 0.74]])  # a geometry
"""

from __future__ import annotations

from collections.abc import Sequence

from ase.data import atomic_numbers, chemical_symbols

from .base import BasisFunction
from .gaussian import GaussianOrbital
from .gto_data import get_basis_data
from .hydrogenic import HydrogenicOrbital
from .nao import DEFAULT_ENERGY_SHIFT, NumericalAtomicOrbital, energy_shift_to_rc
from ._config import valence_subshells


def _to_atomic_number(element) -> int:
    """Accept an element symbol (``"O"``) or atomic number (``8``)."""
    if isinstance(element, str):
        try:
            return atomic_numbers[element.capitalize()]
        except KeyError:
            raise ValueError(f"unknown element symbol {element!r}") from None
    return int(element)


class BasisSet:
    """Base factory. Use :meth:`build` to obtain a concrete basis set."""

    method: str = ""

    @staticmethod
    def build(method: str, **kwargs) -> "BasisSet":
        """Construct a basis set of the requested ``method`` (``"NAO"`` or ``"GTO"``)."""
        key = method.upper()
        if key == "NAO":
            return NAOBasisSet(**kwargs)
        if key == "GTO":
            return GTOBasisSet(**kwargs)
        raise ValueError(f"unknown basis method {method!r}; use 'NAO' or 'GTO'")

    # -- interface --------------------------------------------------------- #

    def atom(self, element, center=(0.0, 0.0, 0.0),
             units: str = "angstrom") -> list[BasisFunction]:
        """Basis functions for one atom (``element`` symbol or Z) at ``center``."""
        raise NotImplementedError

    def molecule(self, symbols: Sequence, positions: Sequence,
                 units: str = "angstrom") -> list[BasisFunction]:
        """Concatenated basis functions for a whole geometry."""
        basis: list[BasisFunction] = []
        for sym, pos in zip(symbols, positions):
            basis.extend(self.atom(sym, center=pos, units=units))
        return basis


class NAOBasisSet(BasisSet):
    """Confined Numerical Atomic Orbitals (SIESTA/Sankey-type).

    Generates one radial function per valence subshell (all ``m``), each solved
    in a hard-wall sphere of radius ``r_c``.  The radial potential uses the
    Slater effective charge of the subshell, so valence orbitals of
    many-electron atoms have realistic extents.

    Parameters
    ----------
    energy_shift : float
        Confinement energy shift in eV (default ``0.03``); sets ``r_c``.
    r_c : float, optional
        Confinement radius in Bohr; overrides ``energy_shift`` when given.
    n_grid : int
        Radial finite-difference resolution.
    """

    method = "NAO"

    def __init__(self, energy_shift: float = DEFAULT_ENERGY_SHIFT,
                 r_c: float | None = None, n_grid: int = 2000):
        self.energy_shift = energy_shift
        self.r_c = float(r_c) if r_c is not None \
            else energy_shift_to_rc(energy_shift)
        self.n_grid = n_grid

    def atom(self, element, center=(0.0, 0.0, 0.0),
             units: str = "angstrom") -> list[BasisFunction]:
        Z = _to_atomic_number(element)
        orbitals: list[BasisFunction] = []
        for (n, l) in valence_subshells(Z):
            z_eff = HydrogenicOrbital.slater_effective_charge(Z, n, l)
            for m in range(-l, l + 1):
                orbitals.append(NumericalAtomicOrbital(
                    n, l, m, Z=z_eff, r_c=self.r_c, center=center,
                    units=units, n_grid=self.n_grid))
        return orbitals

    def __repr__(self) -> str:
        return f"NAOBasisSet(energy_shift={self.energy_shift}, r_c={self.r_c:.3f})"


class GTOBasisSet(BasisSet):
    """Contracted Gaussian-Type Orbitals from a named family.

    Parameters
    ----------
    name : str
        Basis-set family, e.g. ``"6-31G(d)"``, ``"cc-pVDZ"`` or ``"def2-SVP"``.
    """

    method = "GTO"

    def __init__(self, name: str):
        self.name = name
        self._data = get_basis_data(name)

    def atom(self, element, center=(0.0, 0.0, 0.0),
             units: str = "angstrom") -> list[BasisFunction]:
        Z = _to_atomic_number(element)
        if Z not in self._data:
            sym = chemical_symbols[Z] if Z < len(chemical_symbols) else Z
            raise ValueError(
                f"element {sym!r} is not in the embedded {self.name!r} data; "
                f"register it with carcara.basis.register")
        orbitals: list[BasisFunction] = []
        for (l, exps, coeffs) in self._data[Z]:
            for m in range(-l, l + 1):
                orbitals.append(GaussianOrbital(
                    l, m, exps, coeffs, center=center, units=units))
        return orbitals

    def __repr__(self) -> str:
        return f"GTOBasisSet(name={self.name!r})"
