# -*- coding: utf-8 -*-
# file: basis/gaussian.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Contracted Gaussian-Type Orbitals (CGTOs).

A contracted Gaussian of angular momentum ``l`` is

.. math::

    \chi_{lm}(\mathbf r) = R(r)\, Y_l^m(\theta, \phi), \qquad
    R(r) = \sum_i d_i\, r^l e^{-\alpha_i r^2},

with ``d_i = c_i N(\alpha_i, l)`` the contraction coefficients ``c_i`` times the
per-primitive normalization

.. math::

    N(\alpha, l) = \sqrt{\frac{2\,(2\alpha)^{l+3/2}}{\Gamma(l+3/2)}},

which makes each primitive radially normalized (``int |N r^l e^{-\alpha r^2}|^2
r^2 dr = 1``).  The whole contraction is then renormalized so that
``int |R|^2 r^2 dr = 1``; combined with the orthonormal :math:`Y_l^m` this gives
a 3D-normalized orbital, matching the convention of the other Carcará bases.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma

from ..units import to_bohr
from ._angular import spherical_coords, spherical_harmonic
from .base import BasisFunction


def primitive_norm(alpha, l: int):
    r"""Radial normalization ``N(alpha, l)`` of a primitive ``r^l e^{-alpha r^2}``."""
    alpha = np.asarray(alpha, dtype=float)
    return np.sqrt(2.0 * (2.0 * alpha) ** (l + 1.5) / gamma(l + 1.5))


def _contracted_self_overlap(exponents, d, l: int) -> float:
    r"""``int |sum_i d_i r^l e^{-alpha_i r^2}|^2 r^2 dr`` in closed form."""
    a = exponents[:, None] + exponents[None, :]
    integ = gamma(l + 1.5) / (2.0 * a ** (l + 1.5))  # int r^{2l+2} e^{-a r^2} dr
    return float(d @ integ @ d)


class GaussianOrbital(BasisFunction):
    r"""A contracted Gaussian orbital ``R(r) Y_l^m`` for one ``(l, m)``.

    Parameters
    ----------
    l, m : int
        Azimuthal and magnetic quantum numbers (spherical, ``2l + 1`` per shell).
    exponents : array_like
        Gaussian exponents ``alpha_i`` (atomic units, Bohr^-2).
    coefficients : array_like
        Contraction coefficients ``c_i`` (for normalized primitives), as tabulated
        in standard basis-set files.
    center : array_like, shape (3,)
        Cartesian center, in ``units`` (default Angstrom); stored in Bohr.
    units : {"angstrom", "bohr"}
        Unit of ``center``.
    """

    def __init__(self, l: int, m: int, exponents, coefficients, center=None,
                 units: str = "angstrom"):
        if l < 0 or abs(m) > l:
            raise ValueError(f"require l >= 0 and |m| <= l, got l={l}, m={m}")
        self.l, self.m = int(l), int(m)
        self.exponents = np.asarray(exponents, dtype=float).ravel()
        self.coefficients = np.asarray(coefficients, dtype=float).ravel()
        if self.exponents.shape != self.coefficients.shape:
            raise ValueError("exponents and coefficients must have equal length")
        self.center = (np.zeros(3) if center is None
                       else to_bohr(center, units))

        # Absorb per-primitive normalization, then renormalize the contraction.
        d = self.coefficients * primitive_norm(self.exponents, self.l)
        d /= np.sqrt(_contracted_self_overlap(self.exponents, d, self.l))
        self._d = d

    @property
    def n_primitives(self) -> int:
        return self.exponents.size

    def radial(self, r) -> np.ndarray:
        """Contracted radial function ``R(r) = sum_i d_i r^l e^{-alpha_i r^2}``."""
        r = np.asarray(r, dtype=float)
        r2 = (r * r).ravel()
        # (n_prim, n_pts) primitive stack contracted over the primitive axis.
        prims = self._d[:, None] * np.exp(-self.exponents[:, None] * r2[None, :])
        R = prims.sum(axis=0).reshape(r.shape)
        return R * r ** self.l

    def evaluate(self, x, y, z) -> np.ndarray:
        """Sample ``chi_{lm}`` on Cartesian coordinates (Bohr)."""
        r, theta, phi = spherical_coords(x, y, z, self.center)
        R = self.radial(r)
        Y = spherical_harmonic(self.l, self.m, theta, phi)
        return np.asarray(R * Y, dtype=np.complex128)

    def __repr__(self) -> str:
        return (f"GaussianOrbital(l={self.l}, m={self.m}, "
                f"n_primitives={self.n_primitives})")
