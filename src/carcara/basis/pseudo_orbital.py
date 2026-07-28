# -*- coding: utf-8 -*-
# file: basis/pseudo_orbital.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Pseudo-atomic orbitals and Kleinman-Bylander projector functions.

A pseudopotential calculation needs two families of radial functions sampled on
the grid:

* the **pseudo-atomic orbitals** :math:`R^{\text{ps}}_{l}(r)\,Y_{lm}`, which form
  the valence basis.  They are the natural choice -- they are exactly the states
  the pseudopotential was built to reproduce, they are nodeless and smooth, and
  they contain no core, so the basis size drops with the electron count;
* the **Kleinman-Bylander projectors** :math:`\chi_{l}(r)\,Y_{lm}`, which carry
  the nonlocal part of the potential.

Both are radial tables from :mod:`carcara.basis.pseudopotential`, splined and
multiplied by a spherical harmonic -- the same construction
:class:`~carcara.basis.nao.NumericalAtomicOrbital` uses, so they drop into the
integral engine unchanged.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from ..units import to_bohr
from ._angular import spherical_coords, spherical_harmonic
from .base import BasisFunction


class _RadialTabulated(BasisFunction):
    """Shared machinery: spline a radial table and attach an angular factor."""

    def __init__(self, r, radial, l: int, m: int, center=None,
                 units: str = "angstrom"):
        if abs(int(m)) > int(l):
            raise ValueError(f"require |m| <= l, got l={l}, m={m}")
        self.l, self.m = int(l), int(m)
        self.center = (np.zeros(3) if center is None
                       else to_bohr(center, units))
        self._r = np.asarray(r, dtype=float)
        self._values = np.asarray(radial, dtype=float)
        self._r_max = float(self._r[-1])
        self._spline = CubicSpline(self._r, self._values, extrapolate=False)

    def radial(self, r) -> np.ndarray:
        """Interpolated radial function; zero beyond the tabulated range."""
        r = np.asarray(r, dtype=float)
        values = self._spline(np.clip(r, self._r[0], self._r_max))
        return np.where((r > self._r_max) | np.isnan(values), 0.0, values)

    def evaluate(self, x, y, z) -> np.ndarray:
        """Sample the function on Cartesian coordinates (Bohr)."""
        r, theta, phi = spherical_coords(x, y, z, self.center)
        return np.asarray(self.radial(r)
                          * spherical_harmonic(self.l, self.m, theta, phi),
                          dtype=np.complex128)


class PseudoAtomicOrbital(_RadialTabulated):
    r"""A valence pseudo-atomic orbital :math:`R^{\text{ps}}_{l}(r)Y_{lm}`.

    Parameters
    ----------
    pseudopotential : PseudoPotential
        The element's pseudopotential; supplies the radial table.
    l, m : int
        Angular momentum and magnetic quantum number.
    center : array_like
        Cartesian center in ``units``.
    """

    def __init__(self, pseudopotential, l: int, m: int, center=None,
                 units: str = "angstrom"):
        channel = pseudopotential.channels[int(l)]
        super().__init__(pseudopotential.r, channel.pseudo_radial, l, m,
                         center, units)
        self.symbol = pseudopotential.symbol
        self.n = int(channel.n)
        self.eigenvalue = float(channel.eigenvalue)

    @property
    def state(self) -> tuple[int, int, int]:
        return (self.n, self.l, self.m)

    def __repr__(self) -> str:
        return (f"PseudoAtomicOrbital({self.symbol}, n={self.n}, l={self.l}, "
                f"m={self.m}, eps={self.eigenvalue:.4f})")


class KBProjector(_RadialTabulated):
    r"""A Kleinman-Bylander projector :math:`\chi_{l}(r)Y_{lm}`.

    Carries the Kleinman-Bylander energy :attr:`kb_energy` and the index of the
    atom it belongs to, which is what the nonlocal matrix and its nuclear
    derivative need.
    """

    def __init__(self, pseudopotential, l: int, m: int, center=None,
                 units: str = "angstrom", atom_index: int = 0):
        super().__init__(pseudopotential.r, pseudopotential.projectors[int(l)],
                         l, m, center, units)
        self.symbol = pseudopotential.symbol
        self.kb_energy = float(pseudopotential.kb_energies[int(l)])
        self.atom_index = int(atom_index)
        self.r_cut = float(pseudopotential.channels[int(l)].r_cut)

    def __repr__(self) -> str:
        return (f"KBProjector({self.symbol}, l={self.l}, m={self.m}, "
                f"E_KB={self.kb_energy:+.4f}, atom={self.atom_index})")


# --------------------------------------------------------------------------- #
# Building the valence basis and the projector set for a molecule.
# --------------------------------------------------------------------------- #

def pseudo_basis(symbols, positions, potentials, units: str = "angstrom",
                 size="SZ", split_norm=None):
    """Valence pseudo-atomic basis for a molecule.

    Returns ``(functions, atom_of_orbital)``: the orbitals and which atom each
    belongs to.

    ``size`` selects the multiple-zeta / polarized hierarchy exactly as it does
    for the all-electron NAO family (``"SZ"``, ``"DZ"``, ``"DZP"``, ``"TZP"``,
    ...).  The **first zeta always comes from the pseudopotential** and cannot be
    replaced: the Troullier-Martins construction pseudizes each valence orbital
    inside its cutoff radius and the Kleinman-Bylander projectors are built from
    those specific pseudo-orbitals, so pairing the potential with an unrelated
    all-electron radial function would be inconsistent.  The extra zetas are
    split-valence refinements *of* that pseudo-orbital, and the polarization
    shell is split from the highest occupied channel, so everything stays
    matched to the potential it came from.
    """
    from .multizeta import (DEFAULT_SPLIT_NORM, RadialTable,
                            orbitals_from_tables, resolve_zeta, zeta_tables)

    n_zeta, n_polarization = resolve_zeta(size)
    split_norm = DEFAULT_SPLIT_NORM if split_norm is None else float(split_norm)

    functions, owners = [], []
    for index, (symbol, position) in enumerate(zip(symbols, positions)):
        pp = potentials[symbol]
        if n_zeta == 1 and n_polarization == 0:
            # Minimal valence set: the original path, unchanged.
            for l in sorted(pp.channels):
                for m in range(-l, l + 1):
                    functions.append(PseudoAtomicOrbital(
                        pp, l, m, center=position, units=units))
                    owners.append(index)
            continue

        tables = []
        l_max = max(pp.channels)
        for l in sorted(pp.channels):
            channel = pp.channels[l]
            tables.extend(zeta_tables(pp.r, channel.pseudo_radial,
                                      int(channel.n), l, n_zeta, split_norm))
        # Polarization: split the outermost channel up into l+1.  Solving a new
        # confined orbital is not an option here -- there is no pseudopotential
        # channel for an unoccupied l, so the shape is taken from the channel
        # being polarized.
        outermost = pp.channels[l_max]
        for offset in range(n_polarization):
            l = l_max + 1 + offset
            for table in zeta_tables(pp.r, outermost.pseudo_radial,
                                     l + 1, l, 1, split_norm):
                table.polarization = True
                tables.append(table)

        atom_functions = orbitals_from_tables(tables, center=position,
                                              units=units)
        functions.extend(atom_functions)
        owners.extend([index] * len(atom_functions))
    return functions, owners


def kb_projectors(symbols, positions, potentials, units: str = "angstrom"):
    """Kleinman-Bylander projectors for a molecule (one per ``(atom, l, m)``)."""
    projectors = []
    for index, (symbol, position) in enumerate(zip(symbols, positions)):
        pp = potentials[symbol]
        for l in sorted(pp.projectors):
            for m in range(-l, l + 1):
                projectors.append(KBProjector(pp, l, m, center=position,
                                              units=units, atom_index=index))
    return projectors


def valence_electrons(symbols, potentials) -> float:
    """Total valence charge -- the electron count a PP calculation carries."""
    return float(sum(potentials[symbol].valence_charge for symbol in symbols))
