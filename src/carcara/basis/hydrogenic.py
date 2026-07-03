# -*- coding: utf-8 -*-
# file: basis/hydrogenic.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Hydrogen-like (hydrogenic) atomic orbitals.

This is the *single source of truth* for the analytic hydrogenic wavefunction
in Carcará.  Both the real-space evaluation used by the integral engine and the
legacy :class:`~carcara.wavefunction.Wavefunction` helpers delegate here, so the
radial/angular formula lives in exactly one place.
"""

from __future__ import annotations

import numpy as np
from scipy import special

from .base import BasisFunction

# Bohr radius in atomic units.  Kept explicit so the formula reads like a
# textbook and a different length unit could be plugged in later.
_A0 = 1.0
_R_EPS = 1e-15  # regularizes 1/r and the polar angle at the nucleus


class HydrogenicOrbital(BasisFunction):
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
        Effective nuclear charge used in the radial function.
    center : array_like, shape (3,)
        Cartesian center in Bohr.  Defaults to the origin.
    """

    def __init__(self, n: int, l: int, m: int, Z: float = 1.0, center=None):
        if not (0 <= l < n):
            raise ValueError(f"require 0 <= l < n, got n={n}, l={l}")
        if abs(m) > l:
            raise ValueError(f"require |m| <= l, got l={l}, m={m}")
        self.n, self.l, self.m = int(n), int(l), int(m)
        self.Z = float(Z)
        self.center = (np.zeros(3) if center is None
                       else np.asarray(center, dtype=float))
        # Precompute the pieces that do not depend on the sampling point.
        num = special.factorial(self.n - self.l - 1)
        den = 2 * self.n * special.factorial(self.n + self.l)
        self._radial_norm = np.sqrt(
            (2 * self.Z / (self.n * _A0)) ** 3 * num / den)
        self._laguerre = special.genlaguerre(self.n - self.l - 1, 2 * self.l + 1)

    @property
    def state(self) -> tuple[int, int, int]:
        return (self.n, self.l, self.m)

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
        return (f"HydrogenicOrbital(n={self.n}, l={self.l}, m={self.m}, "
                f"Z={self.Z}, center={self.center})")
