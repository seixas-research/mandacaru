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

from .. import _native

_R_EPS = 1e-15  # regularizes 1/r and the polar angle at the nucleus


def spherical_coords(x, y, z, center):
    """Return ``(r, theta, phi)`` of Cartesian points relative to ``center`` (Bohr).

    ``r`` is the true radius (used to mask a hard cutoff); the polar/azimuthal
    angles use a floored radius so they stay finite at the origin.
    """
    xa = np.ascontiguousarray(x, dtype=np.float64).ravel()
    ya = np.ascontiguousarray(y, dtype=np.float64).ravel()
    za = np.ascontiguousarray(z, dtype=np.float64).ravel()
    c = [float(center[0]), float(center[1]), float(center[2])]
    r, theta, phi = _native.spherical_coords(xa, ya, za, c)
    shape = np.shape(x)
    return r.reshape(shape), theta.reshape(shape), phi.reshape(shape)


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
    xa = np.ascontiguousarray(x, dtype=np.float64).ravel()
    ya = np.ascontiguousarray(y, dtype=np.float64).ravel()
    za = np.ascontiguousarray(z, dtype=np.float64).ravel()
    r, (xm, ym, zm), val, (gx, gy, gz) = _native.angular_factor(
        int(l), int(m), xa, ya, za, bool(real)
    )
    return r, (xm, ym, zm), val, (gx, gy, gz)


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
    return _native.real_harmonic_transform(int(l))


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
    t = np.asarray(theta, dtype=np.float64)
    p = np.asarray(phi, dtype=np.float64)
    b_shape = np.broadcast(t, p).shape
    if abs(m) > l:
        return np.zeros(b_shape, dtype=np.float64 if real else np.complex128)
    t_flat = np.ascontiguousarray(np.broadcast_to(t, b_shape), dtype=np.float64).ravel()
    p_flat = np.ascontiguousarray(np.broadcast_to(p, b_shape), dtype=np.float64).ravel()
    res = _native.spherical_harmonic_evaluate(l, m, t_flat, p_flat, bool(real))
    return np.asarray(res).reshape(b_shape)


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
    xa = np.ascontiguousarray(x, dtype=np.float64).ravel()
    ya = np.ascontiguousarray(y, dtype=np.float64).ravel()
    za = np.ascontiguousarray(z, dtype=np.float64).ravel()
    shape = np.shape(x)
    val, (gx, gy, gz) = _native.spherical_harmonic_gradient_evaluate(
        l, m, xa, ya, za, bool(real)
    )
    val = np.asarray(val).reshape(shape)
    gx = np.asarray(gx).reshape(shape)
    gy = np.asarray(gy).reshape(shape)
    gz = np.asarray(gz).reshape(shape)
    return val, (gx, gy, gz)
