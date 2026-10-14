# -*- coding: utf-8 -*-
# file: basis/_angular.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Shared angular machinery for ``R(r) Y_lm(theta, phi)`` functions.

Every atom-centered function in Mandacaru -- basis functions of every family,
projectors, compensation multipoles -- factorizes as a radial function times a
spherical harmonic.  The Cartesian->spherical conversion and the harmonics
live here, in two conventions, and every caller states which one it uses:

* **real** (``real=True``), the harmonics of a molecule:

  .. math::

      S_{lm} = \sqrt2\,(-1)^m\,\mathrm{Re}\,Y_l^m \;(m > 0),\qquad
      S_{l0} = Y_l^0,\qquad
      S_{lm} = \sqrt2\,(-1)^m\,\mathrm{Im}\,Y_l^{|m|} \;(m < 0),

  ordered :math:`m = -l \ldots l`, so a p shell is :math:`(p_y, p_z, p_x)`
  and a d shell :math:`(d_{xy}, d_{yz}, d_{z^2}, d_{xz}, d_{x^2-y^2})`, each
  positive along its own lobe;
* **complex** (``real=False``), the Condon-Shortley :math:`Y_l^m` of
  :func:`scipy.special.sph_harm_y`, the harmonics of a crystal (whose Bloch
  sums are complex anyway, and whose C kernel and Wannier code index them by
  :math:`m`).

The two are related, shell by shell, by the unitary of
:func:`real_harmonic_transform`.
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


def sample_dtype(real: bool):
    """The dtype a function with real (``float64``) or complex
    (``complex128``) harmonics samples as."""
    return np.float64 if real else np.complex128


#: Offsets closer than this to a function's center are moved out to it along
#: z: the gradient of ``R(r) Y_lm`` is a smooth Cartesian function there
#: (``r Y_1m`` is linear), but its direction-dependent pieces are 0/0.
NODE_OFFSET = 1e-12


def angular_factor(l: int, m: int, x, y, z, *, real: bool):
    """``(r, offsets, Y, (dY/dx, dY/dy, dY/dz))`` at the Cartesian offsets
    ``x, y, z`` (Bohr) from a center: the part of ``R(r) Y_lm`` and of its
    gradient that does not depend on the radial function, so the zeta
    partners of one shell share it.  ``offsets`` are the (flat) offsets
    after a point on the center is moved out to :data:`NODE_OFFSET`."""
    x, y, z = (np.array(c, dtype=float).reshape(-1) for c in (x, y, z))
    r = np.sqrt(x * x + y * y + z * z)
    node = r < NODE_OFFSET
    if np.any(node):
        z[node] = NODE_OFFSET
        r[node] = NODE_OFFSET
    value, slopes = spherical_harmonic_gradient(l, m, x, y, z, real=real)
    return r, (x, y, z), value, slopes


def radial_angular_gradient(radial, radial_slope, l: int, m: int, x, y, z, *,
                            real: bool) -> np.ndarray:
    r"""``(3, n)`` gradient of :math:`R(r)\,Y_{lm}` at the Cartesian
    offsets ``x, y, z`` (Bohr) from its center, analytically:

    .. math::

        \nabla\,(R\,Y) = R'(r)\,\hat{\mathbf r}\,Y + R\,\nabla Y ,

    with ``radial`` and ``radial_slope`` the callables :math:`R(r)` and
    :math:`R'(r)` and :math:`\nabla Y` from :func:`spherical_harmonic_gradient`
    (``float64`` for ``real`` harmonics, ``complex128`` otherwise).
    """
    r, offsets, value, slopes = angular_factor(l, m, x, y, z, real=real)
    R, dR = radial(r), radial_slope(r)
    out = np.empty((3, r.size), dtype=np.float64 if real else np.complex128)
    for axis, (offset, dY) in enumerate(zip(offsets, slopes)):
        out[axis] = dR * offset / r * value + R * dY
    return out


def real_harmonic_transform(l: int) -> np.ndarray:
    r"""``(2l+1, 2l+1)`` unitary :math:`U` with :math:`S_{lm} = \sum_{m'}
    U_{mm'} Y_l^{m'}`, rows and columns indexed by ``m + l``.

    :math:`U_{m,m} = (-1)^m/\sqrt2` and :math:`U_{m,-m} = 1/\sqrt2` for
    :math:`m > 0`; :math:`U_{-\mu,-\mu} = i/\sqrt2` and
    :math:`U_{-\mu,\mu} = -i(-1)^\mu/\sqrt2` for :math:`\mu > 0`;
    :math:`U_{00} = 1`.  An operator in the complex harmonics, :math:`A`, is
    :math:`U A U^\dagger` in the real ones.
    """
    l = int(l)
    U = np.zeros((2 * l + 1, 2 * l + 1), dtype=complex)
    U[l, l] = 1.0
    root = 1.0 / np.sqrt(2.0)
    for m in range(1, l + 1):
        sign = (-1) ** m
        U[l + m, l + m] = sign * root
        U[l + m, l - m] = root
        U[l - m, l - m] = 1j * root
        U[l - m, l + m] = -1j * sign * root
    return U


def spherical_harmonic(l: int, m: int, theta, phi, *,
                       real: bool) -> np.ndarray:
    r"""Orthonormal spherical harmonic of degree ``l`` and order ``m``:
    real :math:`S_{lm}` (``real=True``, a float array) or complex
    :math:`Y_l^m` (``real=False``); see the module docstring.

    The complex one is the convention of :func:`scipy.special.sph_harm_y`
    (Condon-Shortley phase, :math:`Y_l^{-m} = (-1)^m \overline{Y_l^m}`),
    evaluated by the standard stable recurrence for the fully normalized
    associated Legendre functions:

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
        return np.zeros(np.broadcast(theta, phi).shape,
                        dtype=float if real else complex)
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
    if real:
        if m == 0:
            return np.broadcast_to(legendre, np.broadcast(theta, phi).shape
                                   ).copy()
        # sqrt(2) (-1)^m P_l^m cos(m phi) (m > 0) or sin(|m| phi) (m < 0):
        # the Condon-Shortley (-1)^m inside P cancels the outer one.
        trig = np.cos(order * phi) if m > 0 else np.sin(order * phi)
        return np.sqrt(2.0) * (-1) ** order * legendre * trig
    value = legendre * np.exp(1j * order * phi)
    if m < 0:
        value = (-1) ** order * np.conj(value)
    return value


def spherical_harmonic_gradient(l: int, m: int, x, y, z, *, real: bool):
    r"""``(Y, (dY/dx, dY/dy, dY/dz))`` of :func:`spherical_harmonic` at the
    Cartesian offsets ``x, y, z`` (``r > 0``), analytically, in the
    convention ``real`` states.

    Written as :math:`Y_l^m = Q_{lm}(z/r)\,((x + iy)/r)^m` for
    :math:`m \ge 0` -- :math:`Q_{lm}` the normalized associated Legendre
    function without its :math:`\sin^m\theta` -- a form smooth in Cartesian
    coordinates, so there is no pole on the z axis; :math:`m < 0` follows
    from :math:`Y_l^{-m} = (-1)^m\overline{Y_l^m}`.  The C Bloch kernel
    (``mandacaru_bloch.c``) evaluates the same expressions.
    """
    l, m = int(l), int(m)
    order = abs(m)
    x, y, z = (np.asarray(c, dtype=float) for c in (x, y, z))
    r = np.sqrt(x * x + y * y + z * z)
    u = z / r
    w = (x + 1j * y) / r
    q0 = np.full(u.shape, 1.0 / np.sqrt(4.0 * np.pi))
    for k in range(1, order + 1):
        q0 = q0 * -np.sqrt((2.0 * k + 1.0) / (2.0 * k))
    d0 = np.zeros(u.shape)
    if l == order:
        Q, dQ = q0, d0
    else:
        q1 = np.sqrt(2.0 * order + 3.0) * u * q0
        d1 = np.sqrt(2.0 * order + 3.0) * q0
        for degree in range(order + 2, l + 1):
            a = np.sqrt((4.0 * degree * degree - 1.0)
                        / (degree * degree - order * order))
            b = np.sqrt(((degree - 1.0) ** 2 - order * order)
                        / (4.0 * (degree - 1.0) ** 2 - 1.0))
            q0, d0, q1, d1 = (q1, d1, a * (u * q1 - b * q0),
                              a * (q1 + u * d1 - b * d0))
        Q, dQ = q1, d1
    wm = w ** order
    wm1 = w ** (order - 1) if order else np.zeros(u.shape)
    value = Q * wm
    gradient = []
    for axis, xa in enumerate((x, y, z)):
        du = ((1.0 if axis == 2 else 0.0) - u * xa / r) / r
        e = (1.0, 1j, 0.0)[axis]
        dw = (e - w * xa / r) / r
        gradient.append(dQ * du * wm + Q * order * wm1 * dw)
    if real:
        # S_lm = sqrt2 (-1)^m Re Y_l^|m| (m > 0) or Im (m < 0): the same parts
        # of the complex value and gradient.
        if m == 0:
            return np.real(value), tuple(np.real(g) for g in gradient)
        part = np.real if m > 0 else np.imag
        scale = np.sqrt(2.0) * (-1) ** order
        return (scale * part(value),
                tuple(scale * part(g) for g in gradient))
    if m < 0:
        sign = (-1) ** order
        value = sign * np.conj(value)
        gradient = [sign * np.conj(g) for g in gradient]
    return value, tuple(gradient)

