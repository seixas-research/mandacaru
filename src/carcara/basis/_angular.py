# -*- coding: utf-8 -*-
# file: basis/_angular.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Shared angular machinery for ``R(r) Y_lm(theta, phi)`` basis functions.

Both the numerical atomic orbitals and the Gaussian orbitals factorize as a
radial function times a spherical harmonic, differing only in the radial part.
The Cartesian->spherical conversion and the (orthonormal, complex) spherical
harmonic live here so those classes share exactly one implementation, matching
the convention already used by :class:`~carcara.basis.fao.FullAtomicOrbital`.
"""

from __future__ import annotations

import numpy as np
from scipy import special

_R_EPS = 1e-15  # regularizes 1/r and the polar angle at the nucleus


def spherical_coords(x, y, z, center):
    """Return ``(r, theta, phi)`` of Cartesian points relative to ``center`` (Bohr).

    ``r`` is the true radius (used to mask a hard cutoff); the polar/azimuthal
    angles use a floored radius so they stay finite at the origin.
    """
    xr = np.asarray(x, dtype=float) - center[0]
    yr = np.asarray(y, dtype=float) - center[1]
    zr = np.asarray(z, dtype=float) - center[2]
    r = np.sqrt(xr * xr + yr * yr + zr * zr)
    r_safe = np.where(r < _R_EPS, _R_EPS, r)
    theta = np.arccos(np.clip(zr / r_safe, -1.0, 1.0))
    phi = np.arctan2(yr, xr)
    return r, theta, phi


def spherical_harmonic(l: int, m: int, theta, phi) -> np.ndarray:
    """Orthonormal complex spherical harmonic ``Y_l^m(theta, phi)``."""
    return special.sph_harm_y(l, m, theta, phi)
