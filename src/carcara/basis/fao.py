# -*- coding: utf-8 -*-
# file: basis/fao.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Full Atomic Orbitals (FAO): analytic single-particle atomic orbitals.

:class:`FullAtomicOrbital` is the *single source of truth* for the analytic
atomic wavefunction in Carcará (an eigenfunction of a screened Coulomb, i.e.
hydrogen-like, radial problem).  Both the real-space evaluation used by the
integral engine and the :class:`~carcara.wavefunction.Wavefunction` helpers
delegate here, so the radial/angular formula lives in exactly one place.
"""

from __future__ import annotations

import numpy as np
from scipy import special

from ..units import to_bohr
from ._config import (_AUFBAU_ORDER, _L_CAPACITY, _SLATER_GROUP_ORDER,
                      _slater_group)
from .base import BasisFunction

# Bohr radius in atomic units.  Kept explicit so the formula reads like a
# textbook and a different length unit could be plugged in later.
_A0 = 1.0
_R_EPS = 1e-15  # regularizes 1/r and the polar angle at the nucleus


class FullAtomicOrbital(BasisFunction):
    r"""A hydrogen-like orbital :math:`\psi_{nlm}(\mathbf r)` for nuclear charge Z.

    .. math::

        \psi_{nlm}(\mathbf r) = R_{nl}(r)\, Y_l^m(\theta, \phi)

    with the standard normalized radial part built from the associated Laguerre
    polynomials.  The orbital is centered at ``center`` (Cartesian, Bohr).

    Parameters
    ----------
    n, l, m : int
        Principal, azimuthal and magnetic quantum numbers.
    Z : float
        Effective nuclear charge used in the radial function.  See
        :meth:`slater_effective_charge` / :meth:`from_slater` to obtain it from
        Slater's rules.
    center : array_like, shape (3,)
        Cartesian center, in ``units``.  Defaults to the origin.
    units : {"angstrom", "bohr"}
        Unit of ``center`` (default ``"angstrom"``).  It is stored internally in
        Bohr, the atomic unit :meth:`evaluate` works in.
    """

    def __init__(self, n: int, l: int, m: int, Z: float = 1.0, center=None,
                 units: str = "angstrom"):
        if not (0 <= l < n):
            raise ValueError(f"require 0 <= l < n, got n={n}, l={l}")
        if abs(m) > l:
            raise ValueError(f"require |m| <= l, got l={l}, m={m}")
        self.n, self.l, self.m = int(n), int(l), int(m)
        self.Z = float(Z)
        self.center = (np.zeros(3) if center is None
                       else to_bohr(center, units))
        # Precompute the pieces that do not depend on the sampling point.
        num = special.factorial(self.n - self.l - 1)
        den = 2 * self.n * special.factorial(self.n + self.l)
        self._radial_norm = np.sqrt(
            (2 * self.Z / (self.n * _A0)) ** 3 * num / den)
        self._laguerre = special.genlaguerre(self.n - self.l - 1, 2 * self.l + 1)

    @property
    def state(self) -> tuple[int, int, int]:
        return (self.n, self.l, self.m)

    # --- Slater's rules --------------------------------------------------- #

    @staticmethod
    def slater_effective_charge(atomic_number: int, n: int, l: int) -> float:
        r"""Effective nuclear charge ``Z_eff = Z - S`` from Slater's rules.

        Builds the ground-state electron configuration of the neutral atom with
        atomic number ``Z = atomic_number`` and returns the effective charge
        seen by an electron in the ``(n, l)`` orbital, where the screening
        constant ``S`` follows Slater's rules (Slater, *Phys. Rev.* **36**, 57,
        1930):

        * electrons in outer groups contribute ``0``;
        * each other electron in the same group contributes ``0.35`` (``0.30``
          for the ``1s`` group);
        * for an ``ns``/``np`` electron, each electron in the ``n-1`` shell
          contributes ``0.85`` and each in shells ``<= n-2`` contributes ``1.00``;
        * for an ``nd``/``nf`` electron, every electron in an inner group
          contributes ``1.00``.

        Examples: ``slater_effective_charge(1, 1, 0) == 1.0`` (H 1s);
        ``slater_effective_charge(3, 1, 0) == 2.70`` (Li 1s);
        ``slater_effective_charge(3, 2, 0) == 1.30`` (Li 2s).
        """
        Z = int(atomic_number)
        if Z < 1:
            raise ValueError(f"atomic_number must be >= 1, got {atomic_number}")

        # Fill the neutral-atom configuration, accumulating Slater-group
        # occupancies keyed by (shell n, group in {'sp','d','f'}).
        group_occ: dict[tuple[int, str], int] = {}
        remaining = Z
        for (nn, ll) in _AUFBAU_ORDER:
            if remaining <= 0:
                break
            occ = min(_L_CAPACITY[ll], remaining)
            remaining -= occ
            key = (nn, _slater_group(ll))
            group_occ[key] = group_occ.get(key, 0) + occ
        if remaining > 0:
            raise ValueError(
                f"atomic number {Z} is beyond the supported filling table")

        target = (int(n), _slater_group(int(l)))
        if target not in group_occ:
            raise ValueError(
                f"orbital n={n}, l={l} is unoccupied in the ground state of "
                f"Z={Z}; Slater's rules do not define its screening")
        t_order = (target[0], _SLATER_GROUP_ORDER[target[1]])

        S = 0.0
        for (gn, gt), occ in group_occ.items():
            if (gn, gt) == target:
                per = 0.30 if (gn == 1 and gt == "sp") else 0.35
                S += per * (occ - 1)
            elif (gn, _SLATER_GROUP_ORDER[gt]) > t_order:
                continue  # outer groups do not screen
            elif target[1] == "sp":
                S += (0.85 if gn == target[0] - 1 else 1.00) * occ
            else:  # d or f electron: all inner groups screen fully
                S += 1.00 * occ
        return Z - S

    @classmethod
    def from_slater(cls, n: int, l: int, m: int, atomic_number: int,
                    center=None, units: str = "angstrom") -> "FullAtomicOrbital":
        """Build an orbital whose ``Z`` is the Slater effective charge.

        Convenience constructor equivalent to passing
        ``Z=slater_effective_charge(atomic_number, n, l)``.
        """
        Z = cls.slater_effective_charge(atomic_number, n, l)
        return cls(n, l, m, Z=Z, center=center, units=units)

    def evaluate(self, x, y, z) -> np.ndarray:
        """Sample :math:`\\psi_{nlm}` on Cartesian coordinates (Bohr)."""
        xr = np.asarray(x, dtype=float) - self.center[0]
        yr = np.asarray(y, dtype=float) - self.center[1]
        zr = np.asarray(z, dtype=float) - self.center[2]

        r = np.sqrt(xr * xr + yr * yr + zr * zr)
        r = np.where(r < _R_EPS, _R_EPS, r)
        theta = np.arccos(np.clip(zr / r, -1.0, 1.0))
        phi = np.arctan2(yr, xr)

        rho = (2 * self.Z * r) / (self.n * _A0)
        R_nl = self._radial_norm * np.exp(-rho / 2) * (rho ** self.l) \
            * self._laguerre(rho)
        Y_lm = special.sph_harm_y(self.l, self.m, theta, phi)
        return np.asarray(R_nl * Y_lm, dtype=np.complex128)

    def __repr__(self) -> str:
        return (f"FullAtomicOrbital(n={self.n}, l={self.l}, m={self.m}, "
                f"Z={self.Z}, center={self.center})")

