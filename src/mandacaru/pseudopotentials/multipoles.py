# -*- coding: utf-8 -*-
# file: pseudopotentials/multipoles.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Multipole compensation charges for PAW-LCAO.

Blöchl's compensation charge restores the multipole moments the smooth density
lost inside the augmentation sphere.  Written out,

.. math::

    \hat n^A(\mathbf r) = \sum_{LM} Q^A_{LM}\, g_L(r)\, Y_{LM}(\hat r),
    \qquad
    Q^A_{LM} = \int \big(n^A - \tilde n^A\big)\, r^L\, Y^*_{LM}\, d^3r ,

so that :math:`\tilde n + \hat n` has the same moments as the all-electron
density and the electrostatics outside the sphere come out right.

Keeping only :math:`L = 0` -- the total charge -- is exact for an atom whose
valence is a closed **s** shell, because an s x s pair density has no angular
structure to lose.  It is not exact for anything with p or d valence: an
``s x p`` pair density carries an :math:`L = 1` **dipole** and ``p x p`` an
:math:`L = 2` **quadrupole**, and those are precisely the components a bond
creates, since an isolated atom has no s-p mixing to speak of.  On oxygen the
missing dipole is 28 % of the monopole's weight at the O-H distance, acting on
an augmentation term worth ~100 eV -- which is why O-H and C-H bonds did not
bind at all before this module existed.

What is here
------------
* :func:`shape_function` / :func:`shape_potential` -- the radial shape
  :math:`g_L` and its Coulomb potential, for any :math:`L`;
* :func:`gaunt` -- the angular coupling
  :math:`\int Y^*_{LM} Y^*_{l_1m_1} Y_{l_2m_2}\,d\Omega`, by quadrature, for
  real harmonics (a molecule) or complex ones (a crystal);
* :func:`radial_moments` -- :math:`\Delta^{(L)}_{ij} = \int (R_iR_j -
  \tilde R_i\tilde R_j)\,r^{L+2}\,dr` between any two partial waves of a
  dataset, reconstructed from what the dataset already stores.

Every number is computed in Rust (:mod:`mandacaru.pseudopotentials`' native
extension, ``paw_*``); this module adapts datasets and arrays to those
routines and carries no numerics of its own.

Conventions are fixed so that the :math:`L = 0` term reproduces the previous
monopole-only code exactly: :math:`g_L` is normalized to
:math:`\int g_L(r)\,r^{L+2}dr = 1`, which for :math:`L = 0` makes
:math:`g_0/4\pi` the old unit-charge shape, and
:math:`Q_{00} = q_{ij}/\sqrt{4\pi}`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .. import _native
from .native_context import context

#: Angular quadrature orders (the Rust rule's: exact for three harmonics up to
#: l = 3 with margin).
GAUNT_POLAR_POINTS = 24
GAUNT_AZIMUTHAL_POINTS = 48


def _flat(values) -> tuple:
    """``(contiguous 1-D float array, original shape)``."""
    values = np.asarray(values, dtype=float)
    return np.ascontiguousarray(values.ravel()), values.shape


def shape_function(radius, r_g: float, L: int = 0) -> np.ndarray:
    r"""``g_L(r) = N_L r^L (1 - r^2/r_g^2)^3`` inside ``r_g``, zero beyond,
    normalized to :math:`\int_0^{r_g} g_L(r)\,r^{L+2}\,dr = 1`."""
    flat, shape = _flat(radius)
    return _native.paw_shape_function(context(), flat, float(r_g),
                                      int(L)).reshape(shape)[()]


def shape_potential(radius, r_g: float, L: int = 0) -> np.ndarray:
    r"""Coulomb potential of ``g_L(r) Y_LM`` without the ``Y_LM`` factor,
    :math:`\frac{4\pi}{2L+1} r^{-(L+1)}` outside ``r_g``."""
    flat, shape = _flat(radius)
    return _native.paw_shape_potential(context(), flat, float(r_g),
                                       int(L)).reshape(shape)[()]


@lru_cache(maxsize=16)
def angular_grid(polar: int = GAUNT_POLAR_POINTS,
                 azimuthal: int = GAUNT_AZIMUTHAL_POINTS):
    """``(theta, phi, weights)`` of a Gauss-Legendre x uniform product rule
    normalized to ``int dOmega``."""
    grid = _native.paw_angular_grid(int(polar), int(azimuthal))
    for array in grid:
        array.flags.writeable = False
    return grid


@lru_cache(maxsize=None)
def _gaunt(real: bool, L: int, M: int, l1: int, m1: int, l2: int, m2: int):
    key = np.array([[L, M, l1, m1, l2, m2]], dtype=np.int64)
    value = _native.paw_gaunt(context(), key, bool(real))[0]
    return float(value.real) if real else complex(value)


def gaunt(L: int, M: int, l1: int, m1: int, l2: int, m2: int, *,
          real: bool):
    r"""``int Y*_{LM} Y*_{l1 m1} Y_{l2 m2} dOmega`` in the convention ``real``
    states: a float for the real harmonics, complex for the complex ones.
    Zero unless the triangle and parity rules are met and, for complex
    harmonics, ``M = m2 - m1``."""
    return _gaunt(bool(real), int(L), int(M), int(l1), int(m1), int(l2),
                  int(m2))


def multipole_range(l1: int, l2: int) -> list[int]:
    """The ``L`` a pair of waves with these momenta can carry."""
    return list(_native.paw_multipole_range(int(l1), int(l2)))


def _channel_arrays(dataset, l: int):
    channel = dataset.channels[int(l)]
    ae = np.ascontiguousarray([np.asarray(w, dtype=float)
                               for w in channel.ae_waves])
    return channel, ae


def partial_waves(dataset, l: int):
    """``(all-electron R_i, smooth R~_i)`` of channel ``l`` on ``dataset.r``.

    The smooth waves are rebuilt from the Bessel expansion the dataset stores
    inside the channel's ``r_cut``; beyond it they *are* the all-electron ones
    (matched there in value and three derivatives), which is what the
    expansion itself, divergent outside, must never be read as.
    """
    channel, ae = _channel_arrays(dataset, l)
    smooth = _native.paw_smooth_waves(
        context(), int(l), np.ascontiguousarray(dataset.r, dtype=float),
        float(channel.r_cut), ae,
        np.ascontiguousarray(channel.wave_coefficients, dtype=float),
        np.ascontiguousarray(channel.wavevectors, dtype=float))
    return list(ae), list(smooth)


def _pair(dataset, l1: int, l2: int):
    ae1, ps1 = partial_waves(dataset, l1)
    ae2, ps2 = partial_waves(dataset, l2)
    r_cut = max(float(dataset.channels[int(l1)].r_cut),
                float(dataset.channels[int(l2)].r_cut))
    return (np.ascontiguousarray(dataset.r, dtype=float), r_cut,
            np.array(ae1), np.array(ps1), np.array(ae2), np.array(ps2))


def radial_moments(dataset, l1: int, l2: int, L: int) -> np.ndarray:
    r"""``Delta^{(L)}_{ij} = int (R_i R_j - R~_i R~_j) r^{L+2} dr`` over the
    augmentation sphere, for waves ``i`` of channel ``l1`` and ``j`` of ``l2``.
    At ``l1 == l2`` and ``L == 0`` this is the stored ``overlap_correction``."""
    return _native.paw_radial_moments(context(), int(L), *_pair(dataset, l1, l2))


def radial_remainder(dataset, l1: int, l2: int, L: int, q,
                     shape_transform) -> np.ndarray:
    r"""``E^{(L)}_{ij}(q)``, ``(len(q), n1, n2)``: the Fourier-Bessel transform
    of the all-electron minus smooth pair density over the augmentation
    sphere, less the compensation shape that carries its multipole moment
    :math:`\Delta^{(L)}_{ij}` in the dual projector basis.
    ``shape_transform`` is :math:`\hat g_L(q)` on ``q``."""
    return _native.paw_radial_remainder(
        context(), int(L), *_pair(dataset, l1, l2),
        np.ascontiguousarray(q, dtype=float),
        np.ascontiguousarray(dataset.multipole_moments(l1, l2, L, "dual"),
                             dtype=float),
        np.ascontiguousarray(shape_transform, dtype=float))
