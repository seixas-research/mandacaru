# -*- coding: utf-8 -*-
# file: test/basis/test_angular.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The spherical harmonics every atom-centered basis function is built on.

:func:`~mandacaru.basis._angular.spherical_harmonic` evaluates the
recurrence itself; SciPy's routine is the reference it must reproduce,
phases and all.
"""

import numpy as np
import pytest
from scipy import special

from mandacaru.basis._angular import spherical_harmonic


@pytest.fixture(scope="module")
def angles():
    rng = np.random.default_rng(0)
    theta = rng.uniform(0.0, np.pi, 5000)
    phi = rng.uniform(-np.pi, np.pi, 5000)
    # The poles and the equator, where naive recurrences lose digits.
    theta[:4] = [0.0, np.pi, 1e-12, np.pi / 2]
    return theta, phi


@pytest.mark.parametrize("l", range(9))
def test_it_matches_scipy_for_every_order(angles, l):
    theta, phi = angles
    for m in range(-l, l + 1):
        assert np.allclose(spherical_harmonic(l, m, theta, phi),
                           special.sph_harm_y(l, m, theta, phi),
                           rtol=0, atol=1e-13), (l, m)


def test_scalars_and_an_order_above_the_degree():
    assert spherical_harmonic(2, 1, 0.3, 0.7) == pytest.approx(
        special.sph_harm_y(2, 1, 0.3, 0.7), abs=1e-15)
    assert not np.any(spherical_harmonic(1, 2, np.ones(3), np.ones(3)))


def test_the_harmonics_are_orthonormal():
    # Gauss-Legendre in cos(theta) times a uniform phi grid is exact here.
    x, w = np.polynomial.legendre.leggauss(24)
    phi = np.linspace(0.0, 2 * np.pi, 48, endpoint=False)
    theta, phi = np.meshgrid(np.arccos(x), phi, indexing="ij")
    weight = np.repeat(w * (2 * np.pi / 48), 48)
    labels = [(l, m) for l in range(4) for m in range(-l, l + 1)]
    Y = np.array([spherical_harmonic(l, m, theta.ravel(), phi.ravel())
                  for l, m in labels])
    overlap = (Y.conj() * weight) @ Y.T
    assert np.allclose(overlap, np.eye(len(labels)), atol=1e-12)
