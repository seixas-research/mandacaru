# -*- coding: utf-8 -*-
# file: test/basis/test_angular.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The spherical harmonics every atom-centered basis function is built on.

:func:`~mandacaru.basis._angular.spherical_harmonic` evaluates the
recurrence itself; SciPy's routine is the reference the complex harmonics
must reproduce, phases and all, and the real ones are those transformed by
:func:`~mandacaru.basis._angular.real_harmonic_transform`.
"""

import numpy as np
import pytest
from scipy import special

from mandacaru.basis._angular import (real_harmonic_transform,
                                      spherical_harmonic,
                                      spherical_harmonic_gradient)


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
        assert np.allclose(spherical_harmonic(l, m, theta, phi, real=False),
                           special.sph_harm_y(l, m, theta, phi),
                           rtol=0, atol=1e-13), (l, m)


def test_scalars_and_an_order_above_the_degree():
    assert spherical_harmonic(2, 1, 0.3, 0.7, real=False) == pytest.approx(
        special.sph_harm_y(2, 1, 0.3, 0.7), abs=1e-15)
    for real in (True, False):
        assert not np.any(spherical_harmonic(1, 2, np.ones(3), np.ones(3),
                                             real=real))


@pytest.mark.parametrize("real", [True, False])
def test_the_harmonics_are_orthonormal(real):
    # Gauss-Legendre in cos(theta) times a uniform phi grid is exact here.
    x, w = np.polynomial.legendre.leggauss(24)
    phi = np.linspace(0.0, 2 * np.pi, 48, endpoint=False)
    theta, phi = np.meshgrid(np.arccos(x), phi, indexing="ij")
    weight = np.repeat(w * (2 * np.pi / 48), 48)
    labels = [(l, m) for l in range(4) for m in range(-l, l + 1)]
    Y = np.array([spherical_harmonic(l, m, theta.ravel(), phi.ravel(),
                                     real=real)
                  for l, m in labels])
    overlap = (Y.conj() * weight) @ Y.T
    assert np.allclose(overlap, np.eye(len(labels)), atol=1e-12)


class TestRealHarmonics:
    """The real harmonics of a molecule: ``S = U Y`` shell by shell."""

    @pytest.mark.parametrize("l", range(9))
    def test_they_are_the_complex_ones_transformed(self, angles, l):
        theta, phi = angles
        U = real_harmonic_transform(l)
        assert np.abs(U @ U.conj().T - np.eye(2 * l + 1)).max() < 1e-14
        Y = np.array([spherical_harmonic(l, m, theta, phi, real=False)
                      for m in range(-l, l + 1)])
        S = np.array([spherical_harmonic(l, m, theta, phi, real=True)
                      for m in range(-l, l + 1)])
        assert S.dtype == np.float64
        assert np.abs(U @ Y - S).max() < 1e-13

    def test_the_order_and_signs_are_the_cartesian_ones(self):
        """p is (y, z, x) and d is (xy, yz, z^2, xz, x^2 - y^2), each
        positive along its own lobe."""
        def at(l, m, v):
            v = np.asarray(v, float) / np.linalg.norm(v)
            theta = np.arccos(v[2])
            phi = np.arctan2(v[1], v[0])
            return float(spherical_harmonic(l, m, theta, phi, real=True))

        c1 = np.sqrt(3.0 / (4.0 * np.pi))
        assert at(1, -1, (0, 1, 0)) == pytest.approx(c1)
        assert at(1, 0, (0, 0, 1)) == pytest.approx(c1)
        assert at(1, 1, (1, 0, 0)) == pytest.approx(c1)
        assert at(2, -2, (1, 1, 0)) > 0 and at(2, -1, (0, 1, 1)) > 0
        assert at(2, 0, (0, 0, 1)) > 0 and at(2, 1, (1, 0, 1)) > 0
        assert at(2, 2, (1, 0, 0)) > 0 and at(2, 2, (0, 1, 0)) < 0
        assert at(2, -2, (1, -1, 0)) < 0

    @pytest.mark.parametrize("l", range(5))
    def test_the_gradient_is_the_transformed_one(self, l):
        rng = np.random.default_rng(l)
        x, y, z = rng.normal(size=(3, 200))
        U = real_harmonic_transform(l)
        values, gradients = zip(*(spherical_harmonic_gradient(
            l, m, x, y, z, real=False) for m in range(-l, l + 1)))
        for m in range(-l, l + 1):
            value, gradient = spherical_harmonic_gradient(l, m, x, y, z,
                                                          real=True)
            row = U[m + l]
            assert np.abs(row @ np.array(values) - value).max() < 1e-12
            for axis in range(3):
                expected = row @ np.array([g[axis] for g in gradients])
                assert np.abs(expected - gradient[axis]).max() < 1e-11
