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
  :math:`\int Y^*_{LM} Y^*_{l_1m_1} Y_{l_2m_2}\,d\Omega`, by quadrature, so no
  Wigner-symbol dependency is needed;
* :func:`radial_moments` -- :math:`\Delta^{(L)}_{ij} = \int (R_iR_j -
  \tilde R_i\tilde R_j)\,r^{L+2}\,dr` between any two partial waves of a
  dataset, reconstructed from what the dataset already stores;
* :func:`multipole_coulomb_matrix` -- the interaction of every pair of
  compensation multipoles on two centers, in one vectorized quadrature.

Conventions are fixed so that the :math:`L = 0` term reproduces the previous
monopole-only code exactly: :math:`g_L` is normalized to
:math:`\int g_L(r)\,r^{L+2}dr = 1`, which for :math:`L = 0` makes
:math:`g_0/4\pi` the old unit-charge shape, and
:math:`Q_{00} = q_{ij}/\sqrt{4\pi}`.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson

from ..basis._angular import spherical_harmonic

#: Angular quadrature orders.  A product Gauss-Legendre (polar) x uniform
#: (azimuthal) rule is *exact* for a product of three spherical harmonics once
#: it integrates degree ``L + l1 + l2``; these cover l up to 3 (f) with margin.
GAUNT_POLAR_POINTS = 24
GAUNT_AZIMUTHAL_POINTS = 48
#: Radial points of the compensation-overlap quadrature.
MULTIPOLE_RADIAL_POINTS = 400
#: Below this the two centers are treated as coincident.
COINCIDENT_TOLERANCE = 1e-10


# --------------------------------------------------------------------------- #
# Radial shape and its potential.
# --------------------------------------------------------------------------- #

def shape_norm(r_g: float, L: int) -> float:
    r"""``1 / int_0^{r_g} r^{2L+2} (1 - r^2/r_g^2)^3 dr`` -- the factor that
    normalizes :func:`shape_function` to unit multipole moment.

    The integral is elementary:
    :math:`r_g^{2L+3}\big[\tfrac1{2L+3} - \tfrac3{2L+5} + \tfrac3{2L+7}
    - \tfrac1{2L+9}\big]`.
    """
    L = int(L)
    bracket = (1.0 / (2 * L + 3) - 3.0 / (2 * L + 5)
               + 3.0 / (2 * L + 7) - 1.0 / (2 * L + 9))
    return 1.0 / (float(r_g) ** (2 * L + 3) * bracket)


def shape_function(radius, r_g: float, L: int = 0) -> np.ndarray:
    r"""``g_L(r) = N_L r^L (1 - r^2/r_g^2)^3`` inside ``r_g``, zero beyond.

    Normalized to :math:`\int_0^{r_g} g_L(r)\,r^{L+2}\,dr = 1`, i.e. the
    charge distribution :math:`g_L Y_{LM}` has unit :math:`LM` multipole
    moment.  At ``L = 0`` this is :math:`4\pi` times the old unit-charge
    monopole shape.
    """
    radius = np.asarray(radius, dtype=float)
    x = radius / float(r_g)
    inside = x < 1.0
    out = np.zeros_like(radius)
    out[inside] = (shape_norm(r_g, L) * radius[inside] ** int(L)
                   * (1.0 - x[inside] ** 2) ** 3)
    return out


def shape_potential(radius, r_g: float, L: int = 0) -> np.ndarray:
    r"""Coulomb potential of ``g_L(r) Y_LM``, without the ``Y_LM`` factor.

    .. math::

        v_L(r) = \frac{4\pi}{2L+1}\Big[
            r^{-(L+1)}\!\int_0^r g_L s^{L+2} ds
          + r^{L}\!\int_r^\infty g_L s^{1-L} ds \Big],

    which is :math:`\frac{4\pi}{2L+1} r^{-(L+1)}` outside ``r_g`` (the shape
    carries unit moment) and finite at the origin.  Both radial integrals are
    polynomials, so they are evaluated in closed form.
    """
    radius = np.asarray(radius, dtype=float)
    L = int(L)
    r_g = float(r_g)
    prefactor = 4.0 * np.pi / (2 * L + 1)
    x = np.clip(radius / r_g, 0.0, None)
    norm = shape_norm(r_g, L)

    # int_0^r g_L s^{L+2} ds = N_L r_g^{2L+3} sum_k C_k x^{2L+3+2k}/(2L+3+2k)
    coefficients = ((0, 1.0), (1, -3.0), (2, 3.0), (3, -1.0))
    inner_charge = np.zeros_like(radius)
    tail = np.zeros_like(radius)
    xin = np.minimum(x, 1.0)
    for k, c in coefficients:
        power = 2 * L + 3 + 2 * k
        inner_charge += c * xin ** power / power
        # int_r^{r_g} g_L s^{1-L} ds = N_L r_g^{2-2L+...}; exponent 2 + 2k
        exponent = 2 * k + 2
        tail += c * (1.0 - xin ** exponent) / exponent
    # int_0^r g_L s^{L+2} ds and int_r^{r_g} g_L s^{1-L} ds = N_L r_g^2 (...)
    inner_charge *= norm * r_g ** (2 * L + 3)
    tail *= norm * r_g ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        near = prefactor * (np.where(radius > 0.0,
                                     inner_charge / np.maximum(radius, 1e-300)
                                     ** (L + 1), 0.0)
                            + radius ** L * tail)
        far = prefactor / np.maximum(radius, 1e-300) ** (L + 1)
    result = np.where(x < 1.0, near, far)
    if L == 0:
        # At the origin only the tail survives and is finite.
        result = np.where(radius <= 0.0, prefactor * tail, result)
    return result


# --------------------------------------------------------------------------- #
# Angular coupling.
# --------------------------------------------------------------------------- #

def _angular_grid(polar: int = GAUNT_POLAR_POINTS,
                  azimuthal: int = GAUNT_AZIMUTHAL_POINTS):
    """``(theta, phi, weights)`` of a product rule normalized to ``int dOmega``."""
    mu, wmu = np.polynomial.legendre.leggauss(int(polar))
    phi = 2.0 * np.pi * (np.arange(int(azimuthal)) + 0.5) / int(azimuthal)
    wphi = 2.0 * np.pi / int(azimuthal)
    theta = np.arccos(mu)
    T, P = np.meshgrid(theta, phi, indexing="ij")
    W = np.outer(wmu, np.full(int(azimuthal), wphi))
    return T.ravel(), P.ravel(), W.ravel()


_GAUNT_CACHE: dict = {}


def gaunt(L: int, M: int, l1: int, m1: int, l2: int, m2: int) -> complex:
    r"""``int Y*_{LM} Y*_{l1 m1} Y_{l2 m2} dOmega``.

    The angular factor of the multipole moment of the pair density
    :math:`\phi^*_1\phi_2`.  Evaluated by a product quadrature that is exact
    for these integrands, which keeps the module free of any Wigner-symbol
    dependency.  Zero unless ``M = m2 - m1`` and the triangle/parity rules of
    the three momenta are met; those cases short-circuit.
    """
    L, M, l1, m1, l2, m2 = (int(L), int(M), int(l1), int(m1), int(l2), int(m2))
    if M != m2 - m1:
        return 0.0 + 0.0j
    if L < abs(l1 - l2) or L > l1 + l2 or (L + l1 + l2) % 2:
        return 0.0 + 0.0j
    key = (L, M, l1, m1, l2, m2)
    cached = _GAUNT_CACHE.get(key)
    if cached is not None:
        return cached
    theta, phi, weights = _angular_grid()
    value = complex(np.sum(weights
                           * np.conj(spherical_harmonic(L, M, theta, phi))
                           * np.conj(spherical_harmonic(l1, m1, theta, phi))
                           * spherical_harmonic(l2, m2, theta, phi)))
    if abs(value) < 1e-12:
        value = 0.0 + 0.0j
    _GAUNT_CACHE[key] = value
    return value


def multipole_range(l1: int, l2: int) -> list[int]:
    """The ``L`` a pair of waves with these momenta can carry."""
    return [L for L in range(abs(int(l1) - int(l2)), int(l1) + int(l2) + 1)
            if (L + int(l1) + int(l2)) % 2 == 0]


# --------------------------------------------------------------------------- #
# Radial moments of a dataset's partial waves.
# --------------------------------------------------------------------------- #

def partial_waves(dataset, l: int):
    """``(all-electron R_i, smooth R~_i)`` of channel ``l`` on ``dataset.r``.

    The smooth waves are rebuilt from the Bessel expansion the dataset stores,
    so nothing has to be regenerated; :func:`radial_moments` at ``L = 0``
    reproduces the stored ``overlap_correction`` to ~1e-8, which is the check
    that the reconstruction is faithful.
    """
    from .oncv import _bessel_table

    channel = dataset.channels[int(l)]
    r = np.asarray(dataset.r, dtype=float)
    ae = [np.asarray(w, dtype=float) for w in channel.ae_waves]
    smooth = [np.asarray(c, dtype=float)
              @ _bessel_table(int(l), np.asarray(q, dtype=float), r)
              for c, q in zip(channel.wave_coefficients, channel.wavevectors)]
    return ae, smooth


def radial_moments(dataset, l1: int, l2: int, L: int) -> np.ndarray:
    r"""``Delta^{(L)}_{ij} = int (R_i R_j - R~_i R~_j) r^{L+2} dr`` over the
    augmentation sphere, for waves ``i`` of channel ``l1`` and ``j`` of ``l2``.

    At ``l1 == l2`` and ``L == 0`` this is the stored ``overlap_correction``
    :math:`q_{ij}`; every other ``L`` is a moment the monopole-only code threw
    away.
    """
    r = np.asarray(dataset.r, dtype=float)
    ae1, ps1 = partial_waves(dataset, l1)
    ae2, ps2 = partial_waves(dataset, l2)
    r_cut = max(float(dataset.channels[int(l1)].r_cut),
                float(dataset.channels[int(l2)].r_cut))
    inside = r <= r_cut
    weight = r[inside] ** (int(L) + 2)
    moments = np.zeros((len(ae1), len(ae2)))
    for i in range(len(ae1)):
        for j in range(len(ae2)):
            integrand = (ae1[i][inside] * ae2[j][inside]
                         - ps1[i][inside] * ps2[j][inside]) * weight
            moments[i, j] = float(simpson(integrand, x=r[inside]))
    return moments


# --------------------------------------------------------------------------- #
# Compensation-compensation Coulomb.
# --------------------------------------------------------------------------- #


def _harmonics(channels, theta, phi) -> np.ndarray:
    """``(n_channels, n_points)`` table of ``Y_LM`` on given directions."""
    return np.stack([spherical_harmonic(L, M, theta, phi) for L, M in channels])


def multipole_coulomb_matrix(r_a: float, channels_a, r_b: float, channels_b,
                             displacement, points: int = 50) -> np.ndarray:
    r"""All ``int int g_A,LM(1) g_B,L'M'(2)/r_12`` at once.

    Returns ``(len(channels_a), len(channels_b))``.  The per-pair form would
    repeat the same quadrature hundreds of times -- and the force needs the
    derivative with respect to each Cartesian component of ``displacement``,
    which multiplies that by six -- so the points, the distances and the
    harmonics are built once and every pair is one contraction against them.

    ``displacement`` is :math:`\mathbf R_B - \mathbf R_A`.  Unlike the
    monopole case the answer depends on the *direction* as well as the
    distance, which is exactly what the missing dipole terms describe.
    """
    displacement = np.asarray(displacement, dtype=float)
    channels_a, channels_b = list(channels_a), list(channels_b)
    theta, phi, wang = _angular_grid()
    x, wx = np.polynomial.legendre.leggauss(int(points))
    rp = 0.5 * r_b * (x + 1.0)
    wr = 0.5 * r_b * wx

    sin_t = np.sin(theta)
    direction = np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi),
                          np.cos(theta)], axis=1)              # (nang, 3)
    rel = displacement[None, None, :] + rp[:, None, None] * direction[None, :, :]
    dist = np.linalg.norm(rel, axis=2)                          # (nr, nang)
    safe = np.maximum(dist, 1e-300)
    polar_a = np.arccos(np.clip(rel[:, :, 2] / safe, -1.0, 1.0))
    azimuth_a = np.arctan2(rel[:, :, 1], rel[:, :, 0])

    # Potentials of every A multipole at the B quadrature points.
    v_a = np.stack([shape_potential(dist, r_a, L)
                    * spherical_harmonic(L, M, polar_a, azimuth_a)
                    for L, M in channels_a])                    # (na, nr, nang)
    # Densities of every B multipole on the same points.
    y_b = _harmonics(channels_b, theta, phi)                    # (nb, nang)
    g_b = np.stack([shape_function(rp, r_b, L) for L, _M in channels_b])
    density_b = g_b[:, :, None] * y_b[:, None, :]               # (nb, nr, nang)

    weight = (wr * rp * rp)[:, None] * wang[None, :]            # (nr, nang)
    return np.einsum("arg,brg,rg->ab", v_a, density_b, weight)
