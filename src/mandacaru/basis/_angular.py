# -*- coding: utf-8 -*-
# file: basis/_angular.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Shared angular machinery for ``R(r) Y_lm(theta, phi)`` basis functions.

Both the numerical atomic orbitals and the Gaussian orbitals factorize as a
radial function times a spherical harmonic, differing only in the radial part.
The Cartesian->spherical conversion and the (orthonormal, complex) spherical
harmonic live here so those classes share exactly one implementation, matching
the convention already used by :class:`~mandacaru.basis.hao.HydrogenicAtomicOrbital`.
"""

from __future__ import annotations

import numpy as np

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
    r"""Orthonormal complex spherical harmonic ``Y_l^m(theta, phi)``.

    The convention of :func:`scipy.special.sph_harm_y` (Condon-Shortley
    phase, :math:`Y_l^{-m} = (-1)^m \overline{Y_l^m}`), evaluated by the
    standard stable recurrence for the fully normalized associated Legendre
    functions:

    .. math::

        \bar P_m^m = (-1)^m \sqrt{\tfrac{1}{4\pi}
            \textstyle\prod_{k=1}^{m} \tfrac{2k+1}{2k}}\,\sin^m\theta,
        \qquad
        \bar P_{m+1}^m = \sqrt{2m+3}\,\cos\theta\,\bar P_m^m,

        \bar P_l^m = a_{lm}\,(\cos\theta\,\bar P_{l-1}^m
            - b_{lm}\,\bar P_{l-2}^m),\quad
        a_{lm} = \sqrt{\tfrac{4l^2 - 1}{l^2 - m^2}},\;
        b_{lm} = \sqrt{\tfrac{(l-1)^2 - m^2}{4(l-1)^2 - 1}} .

    A handful of vector operations per degree: basis sampling calls this for
    every function on every grid point, and the general-purpose SciPy routine
    cost 4 ms per call on a 68^3 grid -- a quarter of a PAW-LCAO Hamiltonian
    build.
    """
    l, m = int(l), int(m)
    order = abs(m)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if order > l:
        return np.zeros(np.broadcast(theta, phi).shape, dtype=complex)
    cos_t = np.cos(theta)
    # sin(theta) itself, not sqrt(1 - cos^2): the latter loses digits at the
    # poles.  theta lies in [0, pi], so it is never negative.
    sin_t = np.sin(theta) if order else None
    legendre = np.full(cos_t.shape, 1.0 / np.sqrt(4.0 * np.pi))
    for k in range(1, order + 1):
        legendre = legendre * (-np.sqrt((2.0 * k + 1.0) / (2.0 * k)) * sin_t)
    if l > order:
        previous, legendre = legendre, np.sqrt(2.0 * order + 3.0) * cos_t * legendre
        for degree in range(order + 2, l + 1):
            a = np.sqrt((4.0 * degree * degree - 1.0)
                        / (degree * degree - order * order))
            b = np.sqrt(((degree - 1.0) ** 2 - order * order)
                        / (4.0 * (degree - 1.0) ** 2 - 1.0))
            previous, legendre = legendre, a * (cos_t * legendre - b * previous)
    value = legendre * np.exp(1j * order * phi)
    if m < 0:
        value = (-1) ** order * np.conj(value)
    return value
